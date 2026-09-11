
# FunASR wav2vec-BERT Adapter

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/Framework-FunASR-green.svg)](https://github.com/modelscope/FunASR)
[![Task](https://img.shields.io/badge/Task-Self--Supervised%20Speech%20Pretraining-orange.svg)](#)

A FunASR-compatible custom adapter for wav2vec2 / w2v-BERT-style self-supervised speech pretraining.

This repository makes wav2vec2 and w2v-BERT pretraining components usable inside the FunASR training stack, including raw-waveform feature extraction, temporal masking, Gumbel vector quantization, contrastive learning, codebook prediction, and streaming tar-shard data loading.

## ✨ Highlights

- FunASR-compatible `Wav2Vec2Model` and `W2VBertModel`
- custom component registration through `++trust_remote_code=true`
- wav2vec2-style contrastive learning objective
- w2v-BERT-style codebook prediction objective
- Hard Gumbel vector quantizer with straight-through gradients
- Conformer encoder with hidden-state extraction
- Streaming tar-shard dataset for large-scale speech pretraining
- Synthetic smoke test for forward/backward validation

## 🧩 What This Repository Is

This repository is a **FunASR custom pretraining adapter**, not a standalone ASR toolkit.

It allows FunASR to load custom wav2vec2 / w2v-BERT-style model, encoder, dataset, and dataloader modules at runtime.

For FunASR remote-code loading, point to the registration file:

```bash
++model="${MODEL_DIR}"
++trust_remote_code=true
++remote_code="${MODEL_DIR}/register.py"
```

The training launcher uses `scripts/train.py` to register components before FunASR
reads `model_conf`. Keep the YAML model class name (`W2VBertModel`) when using
this entry point; overriding it with a directory bypasses normal construction in
FunASR versions that skip remote-code loading when `model_conf` is present.

After registration, FunASR can instantiate components such as:

* `W2VBertModel`
* `Wav2Vec2Model`
* `SDConformerEncoder`
* `SelfSupervisedLargeAudioDataset`
* `DataloaderIterable`

## 🏗️ Architecture

```text
raw waveform
  -> CNN feature extractor
  -> feature projection and normalization
  -> temporal masking
  -> Conformer encoder
  -> wav2vec2 contrastive branch
  -> w2v-BERT codebook prediction branch
  -> total pretraining loss
```

The vector quantizer follows a fairseq2-style design:

```text
features -> entry_proj -> hard gumbel_softmax -> codebook entries -> quantized targets
```

## 📁 Repository Layout

```text
.
├── register.py                    # FunASR file-based remote-code entry
├── schedulers.py                  # WarmupPolynomialDecayLR registration
├── __init__.py                     
├── configuration.json              
├── configs/
│   ├── w2vbert_pretrain.yaml       # Main pretraining config
│   └── w2vbert_pretrain_small.yaml # Alternate training config
├── datasets/
│   ├── dataloader_entry.py         # Dataloader registration
│   ├── index_ds.py
│   ├── large_audio_datasets.py     # Streaming tar-shard dataset
│   └── self_supervised.py          # JSONL/map-style dataset
├── deepspeed_conf/
│   └── ds_stage1.json
├── encoder/
│   └── SDConformerEncoder.py       # Conformer encoder
├── models/
│   ├── wav2vec2/                   # wav2vec2 modules
│   └── w2vbert/                    # w2v-BERT wrapper
└── scripts/
    ├── smoke_forward.py            # Forward/backward smoke test
    └── train_pretrain.sh           # Training launcher template
```

> Note: `w2vbert_pretrain_small.yaml` changes selected training settings. It is not necessarily a small model config unless `model_conf` is also modified.

## 🚀 Quick Start

Clone FunASR and this repository:

```bash
git clone https://github.com/modelscope/FunASR.git
git clone <your-repo-url> funasr-wav2vec-bert-adapter
cd funasr-wav2vec-bert-adapter
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Set paths:

```bash
export FUNASR_ROOT=/path/to/FunASR
export MODEL_DIR=/path/to/funasr-wav2vec-bert-adapter
```

Run the smoke test:

```bash
CUDA_VISIBLE_DEVICES=4 python scripts/smoke_forward.py --device cuda
CUDA_VISIBLE_DEVICES=4 python -m unittest discover -s tests -v
```

Expected output:

```text
loss: ...
weight: 2
quantizer_entry_proj_grad: True
```

The smoke test verifies forward/backward execution, quantizer gradients, and validation.
The regression suite also covers FP32/BF16, tar sharding and resampling, remote registration,
and CUDA training/checkpoint resume through the actual launcher. CPU runs skip the launcher test.

The launcher uses FunASR DDP by default for multiple GPUs. To enable DeepSpeed, install it
and pass `++train_conf.use_deepspeed=true`; ensure its gradient accumulation setting matches
`train_conf.accum_grad`. You can check your environment with:

```bash
python -c "import torch; print(torch.__version__)"
python -c "import deepspeed; print(deepspeed.__version__)"
```

## 📦 Data Preparation

Create a shard-list file:

```text
/path/to/shard_000001.tar
/path/to/shard_000002.tar
/path/to/shard_000003.tar
```

Each line points to one tar file. Inside each tar shard, audio files are treated as independent self-supervised training samples.

Supported audio formats:

```text
wav, flac, mp3, m4a, ogg, opus, wma
```

No transcript is required. Optional `.txt` files may exist in the tar shard, but `SelfSupervisedLargeAudioDataset` ignores text and only uses audio.

Important notes:

* Audio is loaded with `torchaudio.load`, with a SoundFile fallback when the optional decoder is unavailable. Supported formats depend on the installed decoding backend.
* Audio is resampled to 16 kHz before length filtering.
* Multi-channel audio is averaged to mono.
* `min_wav_len` and `max_wav_len` are measured in raw waveform samples.

The dataset yields:

```python
{
    "speech": Tensor[T],
    "speech_lengths": int,
}
```

## ⚙️ Training

The main config is:

```text
configs/w2vbert_pretrain.yaml
```

Key config fields:

* `model: W2VBertModel`
* `dataset: SelfSupervisedLargeAudioDataset`
* `dataset_conf.dataloader: DataloaderIterable`
* `dataset_conf.batch_num_epoch`
* `dataset_conf.batch_size`
* `dataset_conf.batch_type`
* `dataset_conf.sort_size`
* `dataset_conf.min_wav_len`
* `dataset_conf.max_wav_len`

Important configuration notes:

* `batch_num_epoch` is the exact number of training batches **per rank**. Shards repeat if exhausted, and longer streams are truncated at the step budget.
* `valid_batch_num_epoch` sets the validation budget (defaults to the training budget if omitted; supplied configs use 100).
* Each rank needs at least one shard with usable audio. Rank and worker partitions do not overlap within a pass.
* Resuming skips `start_step` batches in deterministic epoch order; keep the shard list, worker count, and batching settings unchanged. This restores data order, not bitwise model RNG state.
* Checkpoints are ranked by validation loss via `train_conf.avg_keep_nbest_models_type: loss`.
* `tokenizer` and external `frontend` are null: the model consumes raw waveforms through its internal CNN.
* `dataset_conf.batch_size` is measured in raw waveform samples, not seconds or fbank frames.
* `dataset_conf.batch_type: frame` keeps `max_raw_samples_in_batch * num_utterances <= batch_size`.
* `min_wav_len` and `max_wav_len` are also measured in raw waveform samples.

Set training paths:

```bash
export FUNASR_ROOT=/path/to/FunASR
export MODEL_DIR=/path/to/funasr-wav2vec-bert-adapter
export TRAIN_LIST=/path/to/train_shards.list
export VALID_LIST=/path/to/valid_shards.list
export OUTPUT_DIR=/path/to/exp/w2vbert_pretrain
export CUDA_VISIBLE_DEVICES=0,1
```

Launch training:

```bash
bash ${MODEL_DIR}/scripts/train_pretrain.sh
```

Override config values if needed:

```bash
bash ${MODEL_DIR}/scripts/train_pretrain.sh \
  ++train_conf.max_epoch=3 \
  ++dataset_conf.batch_size=1000000 \
  ++dataset_conf.batch_num_epoch=20000 \
  ++train_conf.accum_grad=8 \
  ++optim_conf.lr=4e-5
```

## 📊 Outputs

Training outputs are written to `OUTPUT_DIR`.

Typical FunASR outputs include:

```text
${OUTPUT_DIR}/train.log
${OUTPUT_DIR}/config.yaml
${OUTPUT_DIR}/model.pt
${OUTPUT_DIR}/model.pt.best
${OUTPUT_DIR}/model.pt.ep*
${OUTPUT_DIR}/tensorboard/
```

Exact checkpoint names depend on your FunASR trainer settings.

## 🔁 Downstream Usage

The produced checkpoint is a self-supervised pretraining checkpoint for this adapter model.

For downstream ASR fine-tuning, load it into a compatible architecture or write a key-mapping script for shared frontend/encoder modules.

This repository does not currently include a supervised ASR fine-tuning recipe.

## 📝 Implementation Notes

The adapter preserves important wav2vec2 / w2v-BERT-style pretraining behavior:

* `run_frontend()` extracts masked wav2vec2 features and masked quantizer targets.
* The Conformer encoder returns hidden states.
* An intermediate encoder layer feeds the wav2vec2 contrastive branch.
* The final encoder output feeds the BERT-style codebook prediction branch.
* The quantizer uses `entry_proj + entries + hard gumbel_softmax`.
* Diversity loss and feature penalty are scaled by the number of masked targets.

## 🚧 Scope and Limitations

This repository focuses on self-supervised speech pretraining inside FunASR.

It does not currently include:

* Prepared speech data
* Pre-built tar shards
* Pretrained model weights
* A supervised ASR fine-tuning recipe
* A checkpoint conversion script for arbitrary downstream ASR models

Users need to prepare speech data, package audio into tar shards, create train/validation shard lists, and run pretraining with the provided FunASR configuration.

## 📄 License and Attribution

This project is released under the [MIT License](LICENSE) for the original code in this repository.

Parts of the implementation are adapted from or inspired by FunASR and fairseq2 wav2vec2 / w2v-BERT components. Please keep original copyright headers in copied or modified source files.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution details.

## Debugging and review

See [REVIEW.md](REVIEW.md) for reproduced integration failures, fixes, and validation scope.
The supplied configurations use a raw-waveform CNN; they do not implement the
FBANK frontend of a pretrained w2v-BERT checkpoint. Successful execution does not
establish checkpoint compatibility or pretraining convergence.
