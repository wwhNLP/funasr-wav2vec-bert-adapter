"""Exercise repeated mid-epoch resumes through the actual FunASR trainer."""
from pathlib import Path
import tempfile
import unittest

import torch
from funasr.utils.dynamic_import import import_module_from_path

import_module_from_path(str(Path(__file__).resolve().parents[1] / "register.py"))
from funasr_wav2vec_bert_adapter.datasets.dataloader_entry import EpochDataLoader
from funasr_wav2vec_bert_adapter.training import PretrainingTrainer


class InterruptedTraining(Exception):
    pass


class Batches:
    is_training = False

    @property
    def dataset(self):
        return self

    def __iter__(self):
        for index in range(12):
            yield {"x": torch.full((index % 3 + 1, 1), float(index + 1))}


class Model(torch.nn.Module):
    def __init__(self, seen):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.seen = seen

    def forward(self, x):
        self.seen.append(int(x[0, 0]))
        count = torch.tensor(x.numel())
        return (self.weight * x).square().sum(), {"num_targets": count}, count


class ResumeTests(unittest.TestCase):
    def test_repeated_resumes_preserve_batches_and_optimizer_updates(self):
        for accumulation in (1, 2):
            with self.subTest(accum_grad=accumulation), tempfile.TemporaryDirectory() as work:
                def build(directory, seen):
                    trainer = PretrainingTrainer(
                        output_dir=str(directory), device="cpu", accum_grad=accumulation,
                        grad_clip=0, save_checkpoint_interval=2 * accumulation,
                        validate_interval=1000, log_interval=1000,
                        avg_keep_nbest_models_type="loss",
                    )
                    model = Model(seen)
                    optimizer = torch.optim.SGD(model.parameters(), lr=.001, momentum=.9)
                    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=.8)
                    return trainer, model, optimizer, scheduler

                def train(trainer, model, optimizer, scheduler):
                    trainer.train_epoch(
                        model=model, optim=optimizer, scheduler=scheduler, scaler=None,
                        dataloader_train=EpochDataLoader(Batches(), 12, trainer.start_step),
                        dataloader_val=None, epoch=0, start_step=trainer.start_step,
                    )

                expected_seen = []
                baseline = build(Path(work) / "baseline", expected_seen)
                try:
                    train(*baseline)
                    expected_weight = baseline[1].weight.detach().clone()
                    expected_momentum = baseline[2].state[baseline[1].weight]["momentum_buffer"].clone()
                    expected_scheduler = baseline[3].state_dict()
                finally:
                    baseline[0].close()

                actual_seen = []
                directory = Path(work) / "resumed"
                for interruption in range(3):
                    trainer, model, optimizer, scheduler = build(directory, actual_seen)
                    try:
                        if interruption:
                            trainer.resume_checkpoint(model=model, optim=optimizer, scheduler=scheduler)
                        if interruption < 2:
                            save_checkpoint = trainer.save_checkpoint

                            def save_then_interrupt(*args, **kwargs):
                                save_checkpoint(*args, **kwargs)
                                raise InterruptedTraining

                            trainer.save_checkpoint = save_then_interrupt
                            with self.assertRaises(InterruptedTraining):
                                train(trainer, model, optimizer, scheduler)
                            checkpoint = torch.load(directory / "model.pt", weights_only=False)
                            self.assertEqual(checkpoint["step"], (interruption + 1) * 2 * accumulation)
                        else:
                            train(trainer, model, optimizer, scheduler)
                            self.assertEqual(actual_seen, expected_seen)
                            torch.testing.assert_close(model.weight, expected_weight)
                            torch.testing.assert_close(
                                optimizer.state[model.weight]["momentum_buffer"], expected_momentum)
                            self.assertEqual(scheduler.state_dict(), expected_scheduler)
                            trainer.save_checkpoint(1, model=model, optim=optimizer, scheduler=scheduler)
                            checkpoint = torch.load(directory / "model.pt", weights_only=False)
                            self.assertIsNone(checkpoint["step"])
                    finally:
                        trainer.close()
