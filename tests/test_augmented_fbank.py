"""Keep augmented waveform views and their frame lengths through collation."""
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

import torch
from funasr.utils.dynamic_import import import_module_from_path

import_module_from_path(str(Path(__file__).resolve().parents[1] / "register.py"))
from funasr_wav2vec_bert_adapter.datasets.large_audio_datasets import (
    SelfSupervisedLargeAudioDatasetWithAugmentation,
)


def audio_sample(samples):
    signal = (10000 * torch.sin(torch.arange(samples) * .1)).to(torch.int16)
    payload = io.BytesIO()
    with wave.open(payload, 'wb') as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(signal.numpy().tobytes())
    return {"wav": payload.getvalue()}


class AugmentedFbankTests(unittest.TestCase):
    def test_fbank_views_have_independent_lengths_and_raw_sample_budgets(self):
        torch.manual_seed(7)
        with tempfile.TemporaryDirectory() as directory:
            listing = Path(directory) / 'shards.list'
            listing.write_text('')
            dataset = SelfSupervisedLargeAudioDatasetWithAugmentation(
                str(listing), batch_num_epoch=1, feature_type='fbank',
                noise_std=.02, time_stretch_range=[1.5, 1.5])
            with patch('torchaudio.load', side_effect=ImportError('use SoundFile fixture decoder')):
                samples = [dataset._process_sample(audio_sample(length)) for length in (3200, 4800)]
            self.assertEqual([sample['speech_lengths'] for sample in samples], [3200, 4800])
            self.assertTrue(all(sample['speech'].ndim == 1 for sample in samples))
            self.assertEqual(len(list(dataset._create_batches(samples, 8000, 'frame'))), 2)

            batch = dataset.collator(samples)
            self.assertEqual(batch['speech'].shape, (2, 28, 80))
            self.assertEqual(batch['speech_aug1'].shape, (2, 28, 80))
            self.assertEqual(batch['speech_aug2'].shape, (2, 43, 80))
            for key, lengths in [('speech_lengths', [18, 28]),
                                 ('speech_lengths_aug1', [18, 28]),
                                 ('speech_lengths_aug2', [28, 43])]:
                self.assertEqual(batch[key].tolist(), lengths)
                self.assertEqual(batch[key].dtype, torch.int32)
            self.assertFalse(torch.equal(batch['speech'], batch['speech_aug1']))
            self.assertEqual(batch['speech'][0, 18:].count_nonzero().item(), 0)
            self.assertEqual(batch['speech_aug2'][0, 28:].count_nonzero().item(), 0)

    def test_time_stretch_accepts_waveforms_and_legacy_features(self):
        with tempfile.TemporaryDirectory() as directory:
            listing = Path(directory) / 'shards.list'
            listing.write_text('')
            dataset = SelfSupervisedLargeAudioDatasetWithAugmentation(
                str(listing), batch_num_epoch=1, time_stretch_range=[1.5, 1.5])
            for shape in [(10,), (10, 4)]:
                with self.subTest(shape=shape):
                    output, lengths = dataset._time_stretch(torch.ones(shape), torch.tensor([10]))
                    self.assertEqual(output.shape, (15,) + shape[1:])
                    self.assertEqual(int(lengths), 15)
                    torch.testing.assert_close(output, torch.ones_like(output))
