"""Validation must cover every JSONL file even when training uses slices."""
import json
from pathlib import Path
import tempfile
import unittest

from funasr.utils.dynamic_import import import_module_from_path

import_module_from_path(str(Path(__file__).resolve().parents[1] / "register.py"))
from funasr_wav2vec_bert_adapter.datasets.self_supervised import (
    SelfSupervisedAudioDataset, SelfSupervisedAudioDatasetWithAugmentation,
)


class JsonlDatasetTests(unittest.TestCase):
    def test_validation_ignores_training_file_slices(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index in range(4):
                path = Path(directory) / f"{index}.jsonl"
                path.write_text(json.dumps({"source": f"sample-{index}.wav", "source_len": 100}) + "\n")
                paths.append(str(path))
            listing = Path(directory) / "files.list"
            listing.write_text("\n".join(paths) + "\n")

            for dataset_type in (SelfSupervisedAudioDataset, SelfSupervisedAudioDatasetWithAugmentation):
                for split in (0, 1):
                    with self.subTest(dataset=dataset_type.__name__, split=split):
                        options = dict(index_ds="SelfSupervisedIndexDSJsonl", data_split_num=2, data_split_i=split)
                        training = dataset_type(str(listing), is_training=True, **options)
                        validation = dataset_type(str(listing), is_training=False, **options)
                        self.assertEqual(
                            [training.index_ds[i]["source"] for i in range(len(training))],
                            [f"sample-{i}.wav" for i in range(2 * split, 2 * split + 2)],
                        )
                        self.assertEqual(
                            [validation.index_ds[i]["source"] for i in range(len(validation))],
                            [f"sample-{i}.wav" for i in range(4)],
                        )
