"""Check lazy shard consumption without depending on an audio decoder."""
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import torch
from funasr.utils.dynamic_import import import_module_from_path

import_module_from_path(str(Path(__file__).resolve().parents[1] / "register.py"))
from funasr_wav2vec_bert_adapter.datasets.large_audio_datasets import (
    SelfSupervisedLargeAudioDataset, TarFileAndGroupDataPipe,
)


class StreamingTarTests(unittest.TestCase):
    def make_shard(self, directory, mode):
        path = Path(directory) / ("samples.tar.gz" if mode == "w:gz" else "samples.tar")
        with tarfile.open(path, mode) as archive:
            for name, content in [
                ("ignored.bin", b"metadata"), ("a.wav", b"first"),
                ("b.wav", b"second"), ("a.txt", b"transcript"),
                ("c.wav", b"third"), ("d.wav", b"fourth"),
            ]:
                info = tarfile.TarInfo(name)
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))
        return path

    def test_self_supervised_shards_yield_before_reading_later_members(self):
        for mode in ("w", "w:gz"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                path = self.make_shard(directory, mode)
                listing = Path(directory) / "shards.list"
                listing.write_text(str(path) + "\n")
                dataset = SelfSupervisedLargeAudioDataset(
                    str(listing), is_training=False, batch_num_epoch=4,
                    shuffler_size=2, sort_size=1, batch_size=1, batch_type="example")
                extracted, opened = [], []
                original = tarfile.TarFile.extractfile

                def track(archive, member):
                    opened.append(archive)
                    extracted.append(member.name)
                    return original(archive, member)

                with patch.object(tarfile.TarFile, "extractfile", track), \
                     patch("torchaudio.load", return_value=(torch.ones(1, 800), 16000)):
                    iterator = iter(dataset)
                    try:
                        self.assertEqual(next(iterator)["speech_lengths"].tolist(), [800])
                        self.assertEqual(extracted, ["a.wav", "b.wav"])
                        remaining = list(iterator)
                        self.assertEqual(len(remaining), 3)
                        self.assertEqual(extracted, ["a.wav", "b.wav", "c.wav", "d.wav"])
                    finally:
                        iterator.close()
                self.assertTrue(all(archive.closed for archive in opened))

    def test_grouped_reader_keeps_nonadjacent_transcripts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.make_shard(directory, "w")
            samples = list(TarFileAndGroupDataPipe(
                [{"path": str(path)}], shuffle=False, shuffler_size=1))
            self.assertEqual(samples[0], {"key": "a", "wav": b"first", "txt": b"transcript"})
            self.assertEqual(len(samples), 4)

    def test_audio_reader_continues_after_an_invalid_shard(self):
        with tempfile.TemporaryDirectory() as directory:
            bad = Path(directory) / "bad.tar"
            bad.write_bytes(b"invalid archive")
            good = self.make_shard(directory, "w")
            with self.assertLogs(level="WARNING"):
                samples = list(TarFileAndGroupDataPipe(
                    [{"path": str(bad)}, {"path": str(good)}],
                    shuffle=False, shuffler_size=1, audio_only=True))
            self.assertEqual([sample["key"] for sample in samples], ["a", "b", "c", "d"])
