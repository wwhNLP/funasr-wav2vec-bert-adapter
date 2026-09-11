"""Run with python -m unittest discover -s tests -v (CUDA_VISIBLE_DEVICES=4)."""
import copy
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import random
import tempfile
import unittest
import wave

import torch
from omegaconf import OmegaConf
from funasr.utils.dynamic_import import import_module_from_path

ROOT = Path(__file__).resolve().parents[1]
import_module_from_path(str(ROOT / "register.py"))
from funasr_wav2vec_bert_adapter.models.w2vbert.model import W2VBertModel
from funasr_wav2vec_bert_adapter.models.wav2vec2.model import Wav2Vec2Model
from funasr_wav2vec_bert_adapter.datasets.large_audio_datasets import SelfSupervisedLargeAudioDataset
from funasr_wav2vec_bert_adapter.datasets.dataloader_entry import DataloaderIterable
from funasr_wav2vec_bert_adapter.schedulers import WarmupPolynomialDecayLR

spec = importlib.util.spec_from_file_location("smoke", ROOT / "scripts/smoke_forward.py")
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def test_both_models_train_eval_and_bfloat16(self):
        for kind in (W2VBertModel, Wav2Vec2Model):
            for mixed in (False, True):
                with self.subTest(model=kind.__name__, mixed=mixed):
                    cfg = smoke.small_config()
                    model = (kind(w2v2_config=cfg, num_bert_encoder_layers=2)
                             if kind is W2VBertModel else kind(**cfg)).to(self.device)
                    speech = torch.randn(2, 340, device=self.device)
                    lengths = torch.tensor([320, 300], device=self.device)
                    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
                    for _ in range(2):
                        opt.zero_grad()
                        with torch.autocast(self.device, dtype=torch.bfloat16, enabled=mixed):
                            loss, stats, weight = model(speech, lengths)
                        self.assertTrue(torch.isfinite(loss))
                        self.assertEqual(weight.item(), 2)
                        loss.backward()
                        quantizer = model.w2v2_model.quantizer if kind is W2VBertModel else model.quantizer
                        grad = quantizer.entry_proj.weight.grad
                        self.assertIsNotNone(grad)
                        self.assertTrue(torch.isfinite(grad).all())
                        self.assertGreater(grad.abs().sum().item(), 0)
                        opt.step()
                    updates = quantizer.num_updates.clone()
                    model.eval()
                    with torch.no_grad(), torch.autocast(self.device, dtype=torch.bfloat16, enabled=mixed):
                        valid_loss, _, _ = model(speech, lengths)
                    self.assertTrue(torch.isfinite(valid_loss))
                    self.assertEqual(quantizer.num_updates.item(), updates.item())

    def test_no_projection_alias_and_mask_padding(self):
        cfg = smoke.small_config()
        cfg['frontend_conf']['feature_dim'] = 16
        cfg['frontend_conf']['feature_extractor_conf']['layer_descs'] = [(16, 4, 2), (16, 3, 2)]
        cfg['quantizer_conf']['model_dim'] = 16
        model = Wav2Vec2Model(**cfg)
        speech = torch.randn(2, 320)
        lengths = torch.tensor([320, 200])
        features = model.run_frontend(speech, lengths)
        self.assertFalse(features.temporal_mask[~features.padding_mask].any())
        loss, _, _ = model(speech, lengths)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_short_audio_error(self):
        model = Wav2Vec2Model(**smoke.small_config())
        with self.assertRaisesRegex(ValueError, 'at least two masked'):
            model(torch.randn(1, 10), torch.tensor([10]))

    def test_remote_entry_from_other_directory(self):
        code = f'''from funasr.utils.dynamic_import import import_module_from_path
import_module_from_path({str(ROOT / 'register.py')!r})
from funasr.register import tables
from funasr.schedulers import scheduler_classes
assert 'W2VBertModel' in tables.model_classes
assert 'SelfSupervisedLargeAudioDataset' in tables.dataset_classes
assert 'WarmupPolynomialDecayLR' in scheduler_classes
'''
        subprocess.run([sys.executable, '-c', code], cwd='/tmp', check=True, capture_output=True)

    def make_shards(self, directory):
        paths = []
        for i in range(4):
            path = Path(directory) / f'{i}.tar'
            audio = io.BytesIO()
            with wave.open(audio, 'wb') as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(8000)
                stream.writeframes((torch.ones(400 + i * 40, dtype=torch.int16) * (i + 1) * 100).numpy().tobytes())
            with tarfile.open(path, 'w') as tar:
                info = tarfile.TarInfo(f'{i}.wav')
                info.size = len(audio.getvalue())
                tar.addfile(info, io.BytesIO(audio.getvalue()))
            paths.append(str(path))
        listing = Path(directory) / 'shards.list'
        listing.write_text('\n'.join(paths) + '\n\n')
        return str(listing)

    def test_rank_worker_sharding_resampling_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            listing = self.make_shards(directory)
            conf = dict(batch_num_epoch=5, valid_batch_num_epoch=2, batch_type='example',
                        batch_size=1, sort_size=2, shuffler_size=2, num_workers=2,
                        persistent_workers=False, min_wav_len=1, max_wav_len=2000)
            seen = []
            for rank in range(2):
                dataset = SelfSupervisedLargeAudioDataset(listing, is_training=False, **conf)
                dataset.rank, dataset.world_size = rank, 2
                batches = list(torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=2))
                seen.extend(int(b['speech_lengths'][0]) for b in batches)
            self.assertEqual(sorted(seen), [800, 880, 960, 1040])
            loader = DataloaderIterable(dataset='SelfSupervisedLargeAudioDataset',
                dataset_conf=conf, train_data_set_list=listing, valid_data_set_list=listing)
            train, valid = loader.build_iter(epoch=3)
            train.batch_sampler.set_epoch(3)
            valid.batch_sampler.set_epoch(3)
            full = [b['speech_lengths'].tolist() for b in train]
            resumed, _ = loader.build_iter(epoch=3, start_step=2)
            self.assertEqual([b['speech_lengths'].tolist() for b in resumed], full[2:])
            self.assertEqual(len(full), 5)
            self.assertEqual(len(list(valid)), 2)

    @unittest.skipUnless(torch.cuda.is_available(), "FunASR train_ds requires CUDA")
    def test_launcher_checkpoint_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            listing = self.make_shards(directory)
            cfg = OmegaConf.load(ROOT / 'configs/w2vbert_pretrain.yaml')
            cfg.model_conf = dict(num_bert_encoder_layers=2, num_target_codebooks=1,
                                  w2v2_config=smoke.small_config())
            cfg.train_conf.update(dict(max_epoch=1, accum_grad=1, resume=False, keep_nbest_models=1, avg_nbest_model=1))
            cfg.scheduler_conf = dict(warmup_steps=1, total_steps=6, power=1., end_lr=1e-6)
            cfg.dataset_conf.update(dict(batch_num_epoch=2, valid_batch_num_epoch=1,
                batch_type='example', batch_size=2, sort_size=2, num_workers=0,
                min_wav_len=1, max_wav_len=2000))
            cfg['disable_update'] = True
            OmegaConf.save(cfg, work / 'tiny.yaml')
            env = dict(os.environ, PYTHON=sys.executable, CONFIG_PATH=str(work / 'tiny.yaml'),
                       TRAIN_LIST=listing, VALID_LIST=listing, OUTPUT_DIR=str(work / 'exp'),
                       MASTER_PORT='29675')
            command = ['bash', str(ROOT / 'scripts/train_pretrain.sh')]
            for extra in ([], ['++train_conf.resume=true', '++train_conf.max_epoch=2']):
                result = subprocess.run(command + extra, cwd='/tmp', env=env,
                                        capture_output=True, text=True, timeout=180)
                self.assertEqual(result.returncode, 0, result.stdout[-6000:] + result.stderr[-3000:])
                self.assertTrue((work / 'exp/model.pt.best').exists())
            self.assertTrue((work / 'exp/model.pt.avg1').exists())
            checkpoint = torch.load(work / 'exp/model.pt', map_location='cpu', weights_only=False)
            self.assertEqual(checkpoint['epoch'], 2)

    def test_single_process_resume_ignores_global_random_state(self):
        with tempfile.TemporaryDirectory() as directory:
            listing = self.make_shards(directory)
            loader = DataloaderIterable(dataset='SelfSupervisedLargeAudioDataset',
                train_data_set_list=listing, valid_data_set_list=listing,
                dataset_conf=dict(batch_num_epoch=8, num_workers=0, batch_type='example',
                                  batch_size=1, sort_size=1, shuffler_size=3))
            train, _ = loader.build_iter(epoch=2)
            full = [b['speech_lengths'].tolist() for b in train]
            random.seed(999)
            resumed, _ = loader.build_iter(epoch=2, start_step=3)
            self.assertEqual([b['speech_lengths'].tolist() for b in resumed], full[3:])

    def test_scheduler_endpoints_and_resume(self):
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.AdamW([parameter], lr=0.1)
        scheduler = WarmupPolynomialDecayLR(optimizer, warmup_steps=2, total_steps=6, end_lr=0.01)
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 0.05)
        for _ in range(3):
            optimizer.step()
            scheduler.step()
        state = copy.deepcopy(scheduler.state_dict())
        restored = WarmupPolynomialDecayLR(optimizer, warmup_steps=2, total_steps=6, end_lr=0.01)
        restored.load_state_dict(state)
        self.assertEqual(restored.last_epoch, scheduler.last_epoch)
        for _ in range(5):
            optimizer.step()
            restored.step()
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 0.01)


if __name__ == '__main__':
    unittest.main()
