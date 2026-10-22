"""Compare tar audio pipelines using real PCM decoding and resampling."""
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

import soundfile
import torch
import torchaudio
from funasr.utils.dynamic_import import import_module_from_path

import_module_from_path(str(Path(__file__).resolve().parents[1] / "register.py"))
from funasr_wav2vec_bert_adapter.datasets.large_audio_datasets import (
    LargeAudioDataset, SelfSupervisedLargeAudioDataset,
    SelfSupervisedLargeAudioDatasetWithAugmentation,
)


class WaveformFrontend(torch.nn.Module):
    fs = 16000

    def forward(self, waveform, lengths, **kwargs):
        return waveform.unsqueeze(-1), torch.tensor(lengths)


class Tokenizer:
    def encode(self, text):
        return [1]


def pcm_sample(sample_rate, seconds=1):
    time = torch.arange(round(sample_rate * seconds)) / sample_rate
    signal = (torch.sin(2 * torch.pi * 440 * time) * 12000).round().to(torch.int16)
    pcm = torch.stack((signal, signal // 2))
    payload = io.BytesIO()
    with wave.open(payload, 'wb') as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(pcm.T.contiguous().numpy().tobytes())
    return {"wav": payload.getvalue(), "txt": b"transcript"}, pcm.float() / 32768


def decode_pcm(source):
    data, sample_rate = soundfile.read(source, dtype="float32", always_2d=True)
    return torch.from_numpy(data.T.copy()), sample_rate


class AudioResamplingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def datasets(self, listing, **options):
        frontend = WaveformFrontend()
        yield LargeAudioDataset(str(listing), frontend=frontend, tokenizer=Tokenizer(), **options)
        yield SelfSupervisedLargeAudioDataset(str(listing), batch_num_epoch=1, **options)
        yield SelfSupervisedLargeAudioDatasetWithAugmentation(
            str(listing), frontend=frontend, batch_num_epoch=1,
            noise_std=0, time_stretch_range=[1, 1], **options)

    def test_all_tar_paths_resample_and_mix_down_before_the_frontend(self):
        with tempfile.TemporaryDirectory() as directory:
            listing = Path(directory) / 'shards.list'
            listing.write_text('')
            for sample_rate in (8000, 16000, 22050):
                sample, pcm = pcm_sample(sample_rate)
                expected = torchaudio.functional.resample(pcm, sample_rate, 16000).mean(0)
                for dataset in self.datasets(listing):
                    for decoder in ('available', 'missing'):
                        with self.subTest(dataset=type(dataset).__name__, sample_rate=sample_rate, decoder=decoder):
                            load = decode_pcm if decoder == 'available' else ImportError('optional decoder unavailable')
                            with patch('torchaudio.load', side_effect=load):
                                output = dataset._process_sample(sample)
                            self.assertIsNotNone(output)
                            self.assertEqual(int(output['speech_lengths']), 16000)
                            torch.testing.assert_close(output['speech'].reshape(-1), expected)

    def test_length_limits_use_target_sample_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            listing = Path(directory) / 'shards.list'
            listing.write_text('')
            short, _ = pcm_sample(8000, .75)
            long, _ = pcm_sample(8000, 1.25)
            for dataset in self.datasets(listing, min_wav_len=10000, max_wav_len=16000):
                with self.subTest(dataset=type(dataset).__name__), patch('torchaudio.load', side_effect=decode_pcm):
                    output = dataset._process_sample(short)
                    self.assertIsNotNone(output)
                    self.assertEqual(int(output['speech_lengths']), 12000)
                    self.assertIsNone(dataset._process_sample(long))
