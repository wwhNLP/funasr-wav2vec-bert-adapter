"""Algorithm regressions, plus optional direct execution of pinned fairseq2 source."""
import copy
import os
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
from test_adapter import smoke
from funasr_wav2vec_bert_adapter.models.wav2vec2.masker import Wav2Vec2Masker, compute_row_mask
from funasr_wav2vec_bert_adapter.models.wav2vec2.model import Wav2Vec2Model
from funasr_wav2vec_bert_adapter.models.w2vbert.model import W2VBertModel
from funasr_wav2vec_bert_adapter.models.wav2vec2.feature_extractor import Wav2Vec2FbankFeatureExtractor
from fairseq2_reference import load_reference, build_model, reference_key, BatchLayout


class AlgorithmTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(5)

    def test_equal_masks_and_utterance_local_negatives(self):
        valid = torch.arange(70)[None] < torch.tensor([70, 40, 60])[:, None]
        masker = Wav2Vec2Masker(8, temporal_mask_span_len=3, min_num_temporal_mask_spans=1)
        _, mask = masker(torch.randn(3, 70, 8), valid)
        self.assertTrue(torch.equal(mask.sum(-1), mask.sum(-1)[:1].expand(3)))
        self.assertFalse(mask[~valid].any())
        self.assertTrue(((valid & ~mask).sum(-1) > 0).all())
        # Encode each utterance and time index in the target; no probabilistic assertion.
        targets = torch.stack([torch.stack([torch.full((10,), i), torch.arange(10)], dim=-1)
                               for i in range(3)]).float()
        model = Wav2Vec2Model(**smoke.small_config())
        negatives = model._sample_distractors(targets)
        self.assertTrue((negatives[..., 0] == torch.arange(3)[:, None, None]).all())
        self.assertTrue((negatives[..., 1] != torch.arange(10)[None, :, None]).all())

    def test_stacking_and_spatial_mask_order(self):
        inputs = torch.arange(2 * 9 * 4).view(2, 9, 4).float()
        out, lengths = Wav2Vec2FbankFeatureExtractor(4, 2)(inputs, torch.tensor([9, 7]))
        torch.testing.assert_close(out, inputs[:, :8].reshape(2, 4, 8))
        self.assertEqual(lengths.tolist(), [4, 3])
        masker = Wav2Vec2Masker(8, temporal_mask_span_len=2, min_num_temporal_mask_spans=1,
            spatial_mask_span_len=2, max_spatial_mask_prob=0.5, min_num_spatial_mask_spans=1)
        output, mask = masker(torch.ones(2, 30, 8), None)
        # A spatially masked channel is zero even at temporal mask positions.
        self.assertTrue((output[mask] == 0).any())

    def test_mlm_targets_detached_and_loss_weights(self):
        model = W2VBertModel(w2v2_config=smoke.small_fbank_config(4), num_bert_encoder_layers=2)
        seqs = torch.randn(2, 40, 4)
        lengths = torch.tensor([40, 32])
        features = model.w2v2_model.run_frontend(seqs, lengths)
        _, states = model.w2v2_model.encode_features(features)
        features.seqs = states[2]
        output = model.w2v2_model.quantize_and_contrast(features)
        targets = model._get_target_indices(output.quantizer_output)
        self.assertFalse(targets.requires_grad)
        self.assertEqual(targets.dtype, torch.int64)
        self.assertTrue(output.quantizer_output.cb.requires_grad)
        losses = model.w2v2_model.compute_loss(output)
        torch.testing.assert_close(losses.aggregate, losses.contrastive + .1 * losses.diversity + 10 * losses.features_penalty)
        # Non-default weights must also be applied exactly.
        losses = model.w2v2_model.compute_loss(output, diversity_weight=.3, features_penalty_weight=2)
        torch.testing.assert_close(losses.aggregate, losses.contrastive + .3 * losses.diversity + 2 * losses.features_penalty)


@unittest.skipUnless(os.environ.get('FAIRSEQ2_REFERENCE'), 'Set FAIRSEQ2_REFERENCE to the pinned upstream checkout')
class UpstreamParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref = load_reference()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.set_num_threads(1)

    def test_mask_exact_rng_parity(self):
        for seed in (0, 11, 42):
            torch.manual_seed(seed)
            expected = self.ref.compute_row_mask((3, 70), 4, .65, torch.tensor([70, 40, 60]), 2)
            torch.manual_seed(seed)
            actual = compute_row_mask((3, 70), 4, .65, torch.tensor([70, 40, 60]), 2)
            self.assertTrue(torch.equal(actual, expected))

    def test_complete_model_outputs_losses_and_gradients(self):
        devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
        for device in devices:
            for train in (False, True):
                with self.subTest(device=device, train=train), patch('torch.compile', lambda f=None, **kw: f if f is not None else (lambda fn: fn)):
                    cfg = smoke.small_fbank_config(4)
                    cfg['quantizer_conf']['num_codebooks'] = 2
                    # Multiple groups exercise the BERT class/group axis ordering.
                    actual = W2VBertModel(w2v2_config=cfg, num_bert_encoder_layers=2, num_target_codebooks=2).to(device)
                    expected = build_model(self.ref, cfg).to(device)
                    state = {reference_key(k): v for k, v in actual.state_dict().items() if not k.endswith('curr_temp')}
                    expected.load_state_dict(state, strict=True)
                    actual.train(train); expected.train(train)
                    x = torch.randn(2, 43, 4, device=device)
                    lengths = torch.tensor([43, 34], device=device)
                    x1, x2 = x.clone().requires_grad_(), x.clone().requires_grad_()
                    torch.manual_seed(99)
                    ref_loss, ref_output = expected(x1, BatchLayout(x1, lengths.tolist()), bert_label_smoothing=.1)
                    torch.manual_seed(99)
                    loss, stats, _ = actual(x2, lengths, bert_label_smoothing=.1)
                    torch.testing.assert_close(loss, ref_loss.aggregate, rtol=2e-5, atol=2e-5)
                    for key, value in [('bert_loss', ref_loss.bert), ('w2v2_contrastive_loss', ref_loss.w2v2.contrastive),
                                       ('w2v2_diversity_loss', ref_loss.w2v2.diversity), ('w2v2_features_penalty', ref_loss.w2v2.features_penalty)]:
                        torch.testing.assert_close(stats[key], value, rtol=2e-5, atol=2e-5)
                    # Compare encoder states and contrastive logits, not only their summed losses.
                    torch.manual_seed(99)
                    features = actual.w2v2_model.run_frontend(x2, lengths)
                    encoded, states = actual.w2v2_model.encode_features(features)
                    features.seqs = states[2]
                    # Restore the quantizer's forward counter before replaying this branch.
                    actual.w2v2_model.quantizer.num_updates.copy_(expected.w2v2_model.quantizer.num_updates - int(train))
                    out = actual.w2v2_model.quantize_and_contrast(features)
                    torch.testing.assert_close(out.logits, ref_output.w2v2_output.logits, rtol=2e-5, atol=2e-5)
                    torch.testing.assert_close(out.encoder_output, ref_output.w2v2_output.encoder_output, rtol=2e-5, atol=2e-5)
                    ref_loss.aggregate.backward(); loss.backward()
                    torch.testing.assert_close(x1.grad, x2.grad, rtol=5e-4, atol=5e-5)
                    ref_params = dict(expected.named_parameters())
                    for name, param in actual.named_parameters():
                        grad = ref_params[reference_key(name)].grad
                        if grad is None:
                            self.assertIsNone(param.grad, name)
                        else:
                            torch.testing.assert_close(param.grad, grad, rtol=5e-4, atol=5e-5, msg=name)
                    self.assertEqual(actual.w2v2_model.quantizer.num_updates.item(), expected.w2v2_model.quantizer.num_updates.item())

    @unittest.skipUnless(torch.cuda.is_available(), "BF16 parity requires CUDA")
    def test_bf16_upstream_loss_and_quantizer_gradient(self):
        cfg = smoke.small_fbank_config(4)
        actual = W2VBertModel(w2v2_config=cfg, num_bert_encoder_layers=2).cuda()
        expected = build_model(self.ref, cfg).cuda()
        expected.load_state_dict({reference_key(k): v for k, v in actual.state_dict().items()
                                  if not k.endswith('curr_temp')}, strict=True)
        x = torch.randn(2, 48, 4, device='cuda')
        lengths = torch.tensor([48, 40], device='cuda')
        with patch('torch.compile', lambda f=None, **kw: f if f is not None else (lambda fn: fn)):
            torch.manual_seed(91)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                ref_loss, _ = expected(x, BatchLayout(x, lengths.tolist()))
            torch.manual_seed(91)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss, _, _ = actual(x, lengths)
        torch.testing.assert_close(loss, ref_loss.aggregate, rtol=1e-3, atol=1e-3)
        loss.backward(); ref_loss.aggregate.backward()
        torch.testing.assert_close(actual.w2v2_model.quantizer.entry_proj.weight.grad,
            expected.w2v2_model.quantizer.entry_proj.weight.grad, rtol=1e-2, atol=1e-2)



class TrainingReductionTests(unittest.TestCase):
    def test_unequal_microbatches_use_total_targets(self):
        from funasr_wav2vec_bert_adapter.training import PretrainingTrainer
        import tempfile
        with tempfile.TemporaryDirectory() as work:
            trainer = PretrainingTrainer(output_dir=work, device='cpu', accum_grad=2,
                                         grad_clip=0, avg_keep_nbest_models_type='loss')
            model = torch.nn.Linear(1, 1, bias=False)
            model.weight.data.fill_(1)
            optim = torch.optim.SGD(model.parameters(), lr=.1)
            scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=10)
            reference = copy.deepcopy(model)
            losses = []
            for i, n in enumerate((3, 7)):
                x = torch.full((n, 1), float(i + 1))
                loss = model(x).square().sum()
                losses.append(reference(x).square().sum())
                values = dict(loss_sum=loss, num_targets=torch.tensor(n), batch_idx=i)
                trainer.backward_step(model, None, values)
                trainer.update_step(model, optim, scheduler, None, values)
            (sum(losses) / 10).backward()
            expected = reference.weight.detach() - .1 * reference.weight.grad
            torch.testing.assert_close(model.weight, expected)
            self.assertEqual(scheduler.last_epoch, 1)
            trainer.close()

    def test_validation_is_target_weighted(self):
        from funasr_wav2vec_bert_adapter.training import PretrainingTrainer
        import tempfile
        class Model(torch.nn.Module):
            def forward(self, x):
                return x.square().sum(), {'num_targets': torch.tensor(x.numel())}, torch.tensor(1)
        class Loader(list):
            @property
            def batch_sampler(self):
                return self
            def set_epoch(self, epoch):
                pass
        with tempfile.TemporaryDirectory() as work:
            trainer = PretrainingTrainer(output_dir=work, device='cpu', avg_keep_nbest_models_type='loss')
            loader = Loader([{'x': torch.full((3,), 2.)}, {'x': torch.ones(7)}])
            model = Model().train()
            trainer.validate_epoch(model=model, dataloader_val=loader, epoch=1)
            self.assertAlmostEqual(trainer.val_loss_avg, 1.9)
            self.assertAlmostEqual(trainer.val_loss_step_or_epoch['model.pt.ep1'], 1.9)
            self.assertTrue(model.training)
            trainer.close()


if __name__ == '__main__':
    unittest.main()
