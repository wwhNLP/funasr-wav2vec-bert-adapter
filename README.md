# FunASR wav2vec-BERT Adapter

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/Framework-FunASR-green.svg)](https://github.com/modelscope/FunASR)
[![Task](https://img.shields.io/badge/Task-Self--Supervised%20ASR-orange.svg)](#)

A FunASR remote-code adapter for wav2vec2 / w2v-BERT style self-supervised speech pretraining.

This project ports the core wav2vec2 / w2v-BERT pretraining path into the FunASR training ecosystem, including model registration, raw-waveform feature extraction, temporal masking, vector quantization, contrastive learning, codebook prediction, and streaming audio data loading.

## Highlights

- FunASR-compatible `Wav2Vec2Model` and `W2VBertModel`
- fairseq2-style wav2vec2 / w2v-BERT training semantics
- hard Gumbel vector quantizer with straight-through gradients
- Conformer encoder with hidden-state extraction
- wav2vec2 contrastive loss + w2v-BERT codebook prediction loss
- streaming tar-shard dataset support for large speech corpora
- plug-in style loading through `++trust_remote_code=true`
- minimal synthetic forward/backward smoke test

## Why This Exists

wav2vec2 / w2v-BERT pretraining is useful for learning acoustic representations from unlabeled or weakly labeled speech. FunASR provides a strong ASR training stack, but wav2vec-BERT style pretraining is not a native FunASR recipe.

This adapter bridges that gap by making the pretraining model and data pipeline loadable through FunASR's remote-code and registration mechanism.

## Architecture

```text
raw waveform
  -> CNN feature extractor
  -> post-extraction normalization
  -> projection
  -> temporal masking
  -> Conformer encoder
  -> wav2vec2 contrastive branch
  -> w2v-BERT codebook prediction branch
  -> FunASR trainer
```

The vector quantizer follows the fairseq2-style design:

```text
features -> entry_proj -> hard gumbel_softmax -> codebook entries -> quantized targets
```

## Repository Layout

```text
.
├── __init__.py                     # FunASR remote-code entry
├── configuration.json              # FunASR-style metadata
├── configs/
│   ├── w2vbert_pretrain.yaml       # Main pretraining config
│   └── w2vbert_pretrain_small.yaml # Smaller/debug config
├── datasets/
│   ├── dataloader_entry.py         # Dataloader registration
│   ├── index_ds.py
│   ├── large_audio_datasets.py     # Streaming tar-shard dataset
│   └── self_supervised.py          # JSONL/map-style dataset
├── deepspeed_conf/
│   └── ds_stage1.json
├── encoder/
│   └── SDConformerEncoder.py       # Conformer encoder with hidden states
├── models/
│   ├── wav2vec2/                   # wav2vec2 modules
│   └── w2vbert/                    # w2v-BERT wrapper
└── scripts/
    ├── smoke_forward.py            # Minimal forward/backward test
    └── train_pretrain.sh           # Training launcher template
```

## Quick Start

Clone FunASR and this adapter:

```bash
git clone https://github.com/modelscope/FunASR.git
git clone <your-repo-url> funasr-wav2vec-bert-adapter
cd funasr-wav2vec-bert-adapter
pip install -r requirements.txt
```

Run the smoke test:

```bash
FUNASR_ROOT=/path/to/FunASR \
python scripts/smoke_forward.py
```

Expected output:

```text
loss: ...
weight: 2
quantizer_entry_proj_grad: True
```

`quantizer_entry_proj_grad: True` confirms that the hard Gumbel quantizer keeps the straight-through gradient path to `entry_proj`.

## Data Format

For streaming pretraining, prepare a shard-list file:

```text
/path/to/shard_000001.tar
/path/to/shard_000002.tar
/path/to/shard_000003.tar
```

Each tar shard can contain audio files such as:

```text
wav, flac, mp3, m4a, ogg, opus, wma
```

Text files are optional and ignored by `SelfSupervisedLargeAudioDataset`.

The dataset yields:

```python
{
    "speech": Tensor[T],
    "speech_lengths": int,
}
```

## Training

Set paths:

```bash
export FUNASR_ROOT=/path/to/FunASR
export MODEL_DIR=/path/to/funasr-wav2vec-bert-adapter
export TRAIN_LIST=/path/to/train_shards.list
export VALID_LIST=/path/to/valid_shards.list
export OUTPUT_DIR=/path/to/exp/w2vbert_pretrain
export CUDA_VISIBLE_DEVICES=0,1
```

Launch:

```bash
bash scripts/train_pretrain.sh
```

Override config values with FunASR/Hydra arguments:

```bash
bash scripts/train_pretrain.sh \
  ++train_conf.max_epoch=3 \
  ++dataset_conf.batch_size=800000 \
  ++train_conf.accum_grad=8 \
  ++optim_conf.lr=4e-5
```

The launcher calls:

```bash
torchrun ... ${FUNASR_ROOT}/funasr/bin/train_ds.py \
  ++model="${MODEL_DIR}" \
  ++trust_remote_code=true \
  ++remote_code="./"
```

## Implementation Notes

The adapter is designed to preserve the important fairseq2 w2v-BERT behavior:

- `run_frontend()` extracts masked wav2vec2 features and masked quantizer targets.
- The Conformer encoder returns all hidden states.
- An intermediate encoder layer feeds the wav2vec2 contrastive branch.
- The final encoder output feeds the BERT-style codebook prediction branch.
- The quantizer uses `entry_proj + entries + hard gumbel_softmax`.
- The temperature schedule uses `(max_temp, min_temp, decay)`.
- Diversity loss and feature penalty are scaled by the number of masked targets.

## Workflow

1. Prepare unlabeled or weakly labeled speech.
2. Package audio into tar shards.
3. Create train and validation shard lists.
4. Run wav2vec2 / w2v-BERT self-supervised pretraining.
5. Use the checkpoint to initialize downstream ASR fine-tuning.

## Not Included

This repository does not include:

- pretrained checkpoints
- speech datasets
- tokenizer or BPE models
- CMVN statistics
- downstream ASR fine-tuning recipes
- experiment logs or TensorBoard outputs

## License

This project is released under the [MIT License](LICENSE).

It adapts code and implementation ideas from FunASR and fairseq2 wav2vec2 / w2v-BERT components. Keep copyright headers in copied or adapted source files.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution details.
