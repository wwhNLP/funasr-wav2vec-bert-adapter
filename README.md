# FunASR wav2vec-BERT Adapter

This repository contains a FunASR remote-code adapter for wav2vec2 / w2v-BERT style self-supervised speech pretraining. It was built for low-resource ASR scenarios, especially Mongolian speech recognition, where unlabeled or weakly labeled speech can be used to learn acoustic representations before downstream ASR fine-tuning.

## What This Project Does

- Adapts fairseq2-style wav2vec2 / w2v-BERT pretraining modules to the FunASR training stack.
- Registers `Wav2Vec2Model`, `W2VBertModel`, `SDConformerEncoder`, self-supervised datasets, and iterable dataloaders through FunASR's `tables.register` mechanism.
- Supports raw-waveform self-supervised pretraining with:
  - CNN acoustic feature extraction
  - temporal masking
  - Conformer encoder
  - hard Gumbel vector quantization
  - wav2vec2 contrastive loss
  - w2v-BERT codebook prediction loss
- Provides streaming tar-shard dataset loading for large unlabeled speech corpora.

## Repository Layout

```text
.
├── __init__.py                     # FunASR remote-code entry
├── configuration.json              # Model metadata for FunASR-style loading
├── configs/
│   ├── w2vbert_pretrain.yaml       # Main pretraining config
│   └── w2vbert_pretrain_small.yaml # Smaller/debug config
├── datasets/
│   ├── dataloader_entry.py         # DataloaderIterable registration
│   ├── index_ds.py
│   ├── large_audio_datasets.py     # Streaming tar-shard self-supervised dataset
│   └── self_supervised.py          # JSONL/map-style self-supervised dataset
├── deepspeed_conf/
│   └── ds_stage1.json
├── encoder/
│   └── SDConformerEncoder.py       # Conformer encoder with hidden-state output
├── models/
│   ├── wav2vec2/                   # FunASR-compatible wav2vec2 components
│   └── w2vbert/                    # w2v-BERT wrapper
└── scripts/
    ├── smoke_forward.py            # Minimal forward/backward test
    └── train_pretrain.sh           # Training launcher template
```

The repository intentionally excludes datasets, checkpoints, training logs, tensorboard files, BPE models, CMVN files, and experiment outputs.

## Installation

Clone FunASR and this adapter side by side, or install FunASR in the current Python environment.

```bash
git clone https://github.com/modelscope/FunASR.git
git clone <your-repo-url> funasr-wav2vec-bert-adapter

cd funasr-wav2vec-bert-adapter
pip install -r requirements.txt
```

If FunASR is not installed as a package, pass its source path through `FUNASR_ROOT` when running scripts.

## Smoke Test

Run a small synthetic forward/backward test:

```bash
FUNASR_ROOT=/path/to/FunASR \
python scripts/smoke_forward.py
```

Expected output includes a finite loss and:

```text
quantizer_entry_proj_grad: True
```

This verifies that the hard Gumbel quantizer keeps the straight-through gradient path to `entry_proj`.

## Data Format

For large-scale self-supervised training, use a shard list file:

```text
/path/to/shard_000001.tar
/path/to/shard_000002.tar
/path/to/shard_000003.tar
```

Each tar shard should contain audio files such as `wav`, `flac`, `mp3`, `m4a`, `ogg`, `opus`, or `wma`. Text files are optional and ignored by `SelfSupervisedLargeAudioDataset`.

The dataset returns only:

```python
{
    "speech": Tensor[T],
    "speech_lengths": int
}
```

## Training

Set the required paths and launch:

```bash
export FUNASR_ROOT=/path/to/FunASR
export MODEL_DIR=/path/to/funasr-wav2vec-bert-adapter
export TRAIN_LIST=/path/to/train_shards.list
export VALID_LIST=/path/to/valid_shards.list
export OUTPUT_DIR=/path/to/exp/w2vbert_pretrain
export CUDA_VISIBLE_DEVICES=0,1

bash scripts/train_pretrain.sh
```

You can override config values with FunASR/Hydra style arguments:

```bash
bash scripts/train_pretrain.sh \
  ++train_conf.max_epoch=3 \
  ++dataset_conf.batch_size=800000 \
  ++train_conf.accum_grad=8 \
  ++optim_conf.lr=4e-5
```

The script calls:

```bash
torchrun ... ${FUNASR_ROOT}/funasr/bin/train_ds.py \
  ++model="${MODEL_DIR}" \
  ++trust_remote_code=true \
  ++remote_code="./"
```

## Technical Notes

The adapter follows fairseq2 w2v-BERT semantics:

- `run_frontend()` extracts masked wav2vec2 features and masked quantizer targets.
- The Conformer encoder returns all hidden states.
- An intermediate encoder layer is used for the wav2vec2 contrastive branch.
- The final encoder output is used for the BERT-style codebook prediction branch.
- The vector quantizer uses `entry_proj + entries + hard gumbel_softmax`, with temperature schedule `(max_temp, min_temp, decay)`.
- Diversity loss and feature penalty are scaled by the number of masked targets, matching fairseq2 wav2vec2 behavior.

## Typical Low-Resource ASR Workflow

1. Collect unlabeled or weakly labeled Mongolian speech.
2. Package audio into tar shards and prepare train/validation shard lists.
3. Run self-supervised w2v-BERT pretraining with this adapter.
4. Use the resulting checkpoint as initialization for downstream Mongolian ASR fine-tuning in FunASR.

## What Is Not Included

- No pretrained checkpoints.
- No speech datasets.
- No Mongolian tokenizer or BPE model.
- No downstream ASR fine-tuning recipe.

## License And Attribution

This adapter is built on FunASR and fairseq2 wav2vec2 / w2v-BERT implementations. Keep copyright headers in copied or adapted source files.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before publishing the repository. Add a top-level `LICENSE` file before public release, using a license compatible with FunASR and the adapted fairseq2 components.

## Suggested GitHub Push

```bash
cd funasr-wav2vec-bert-adapter
git init
git add .
git commit -m "Initial FunASR wav2vec-BERT adapter"
git branch -M main
git remote add origin git@github.com:<your-user>/funasr-wav2vec-bert-adapter.git
git push -u origin main
```
