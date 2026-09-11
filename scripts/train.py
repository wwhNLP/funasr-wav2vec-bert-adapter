"""Register the adapter before FunASR resolves a YAML model_conf."""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parents[1] / "register.py"))

from funasr.bin import train_ds
from funasr_wav2vec_bert_adapter.training import PretrainingTrainer

train_ds.Trainer = PretrainingTrainer
main_hydra = train_ds.main_hydra

if __name__ == "__main__":
    main_hydra()
