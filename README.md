# FunASR wav2vec-BERT Adapter

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/Framework-FunASR-green.svg)](https://github.com/modelscope/FunASR)
[![Task](https://img.shields.io/badge/Task-Self--Supervised%20Speech%20Pretraining-orange.svg)](#)

A FunASR remote-code adapter for wav2vec2 / w2v-BERT 2.0 style self-supervised speech pretraining.

This project makes wav2vec2 / w2v-BERT pretraining components usable inside the FunASR training stack. It provides model registration, raw-waveform feature extraction, temporal masking, hard Gumbel vector quantization, contrastive learning, codebook prediction, and streaming tar-shard data loading.

## Highlights

- FunASR-compatible `Wav2Vec2Model` and `W2VBertModel`
- fairseq2-style wav2vec2 / w2v-BERT training semantics
- hard Gumbel vector quantizer with straight-through gradients
- Conformer encoder with hidden-state extraction
- wav2vec2 contrastive loss + w2v-BERT codebook prediction loss
- streaming tar-shard dataset support for large speech corpora
- remote-code registration through `++trust_remote_code=true`
- synthetic model smoke test for forward/backward validation

## What This Adapter Is

FunASR loads custom model and dataset code through a remote-code mechanism. In this repository, `MODEL_DIR` points to the adapter directory. During training, FunASR receives:

```bash
++model="${MODEL_DIR}"
++trust_remote_code=true
++remote_code="${MODEL_DIR}"
```

FunASR imports the Python modules in this directory, which executes the registration code in `__init__.py`, `models/`, `encoder/`, and `datasets/`. After that, names such as `W2VBertModel`, `Wav2Vec2Model`, `SDConformerEncoder`, `SelfSupervisedLargeAudioDataset`, and `DataloaderIterable` are available through FunASR's registry.

This is not a FunASR plugin package. It is a remote-code model/dataset adapter loaded by the FunASR trainer.

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
│   └── w2vbert_pretrain_small.yaml # Alternate training config
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
    ├── smoke_forward.py            # Model-only forward/backward test
    └── train_pretrain.sh           # FunASR training launcher template
```

`w2vbert_pretrain_small.yaml` is an alternate training config. It is not a tiny model config: the model dimensions and layer count are still large unless you edit `model_conf`.

## Requirements

Use a recent FunASR checkout and a PyTorch/torchaudio environment that matches your CUDA runtime.

Minimal Python packages:

```bash
pip install -r requirements.txt
```

For real multi-GPU training with `funasr/bin/train_ds.py`, install DeepSpeed and verify that it matches your CUDA/PyTorch setup:

```bash
python -c "import torch; print(torch.__version__)"
python -c "import deepspeed; print(deepspeed.__version__)"
```

The smoke test does not require DeepSpeed. Training does.

## How To Use This Adapter

### 1. Clone FunASR And This Repository

```bash
git clone https://github.com/modelscope/FunASR.git
git clone <your-repo-url> funasr-wav2vec-bert-adapter
cd funasr-wav2vec-bert-adapter
```

Set paths:

```bash
export FUNASR_ROOT=/path/to/FunASR
export MODEL_DIR=/path/to/funasr-wav2vec-bert-adapter
```

`MODEL_DIR` must point to this adapter directory.

### 2. Run The Model Smoke Test

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

This is a model-only smoke test. It verifies local imports, model construction, forward/backward execution, and quantizer gradients. It does not validate FunASR remote-code loading or dataloader construction.

### 3. Prepare Tar-Shard Data

Create a shard-list file:

```text
/path/to/shard_000001.tar
/path/to/shard_000002.tar
/path/to/shard_000003.tar
```

Each line points to one tar file. Inside each tar shard, audio files are read as independent samples. Supported extensions include:

```text
wav, flac, mp3, m4a, ogg, opus, wma
```

Optional `.txt` files may exist in the same tar shard, but `SelfSupervisedLargeAudioDataset` ignores text and uses audio only.

Important data assumptions:

- Audio is loaded with `torchaudio.load`.
- The current dataset does not resample audio.
- Prepare audio at the sample rate expected by the model config, typically 16 kHz.
- Multi-channel audio is averaged to mono.
- `min_wav_len` and `max_wav_len` are measured in raw waveform samples.

The dataset yields:

```python
{
    "speech": Tensor[T],
    "speech_lengths": int,
}
```

### 4. Configure Training

The main config is:

```text
configs/w2vbert_pretrain.yaml
```

Key fields:

- `model: W2VBertModel`
- `dataset: SelfSupervisedLargeAudioDataset`
- `dataset_conf.dataloader: DataloaderIterable`
- `dataset_conf.batch_num_epoch`: estimated number of batches per epoch. Required because the dataset is iterable.
- `dataset_conf.batch_size`: dynamic-batch budget measured in raw waveform samples, not fbank frames.
- `dataset_conf.batch_type: frame`: the batcher keeps `max_raw_samples_in_batch * num_utterances <= batch_size`.
- `dataset_conf.sort_size`: buffer size used before length sorting and dynamic batching.
- `dataset_conf.min_wav_len` / `max_wav_len`: waveform length filters, measured in samples.

If you delete `batch_num_epoch`, the trainer will fail because it needs an estimated dataloader length.

### 5. Launch Training

Set paths:

```bash
export FUNASR_ROOT=/path/to/FunASR
export MODEL_DIR=/path/to/funasr-wav2vec-bert-adapter
export TRAIN_LIST=/path/to/train_shards.list
export VALID_LIST=/path/to/valid_shards.list
export OUTPUT_DIR=/path/to/exp/w2vbert_pretrain
export CUDA_VISIBLE_DEVICES=0,1
```

Launch from any directory:

```bash
bash ${MODEL_DIR}/scripts/train_pretrain.sh
```

The script calls:

```bash
torchrun ... ${FUNASR_ROOT}/funasr/bin/train_ds.py \
  --config-path ${MODEL_DIR}/configs \
  --config-name w2vbert_pretrain \
  ++model="${MODEL_DIR}" \
  ++trust_remote_code=true \
  ++remote_code="${MODEL_DIR}" \
  ++train_data_set_list="${TRAIN_LIST}" \
  ++valid_data_set_list="${VALID_LIST}"
```

Override config values with FunASR/Hydra arguments:

```bash
bash ${MODEL_DIR}/scripts/train_pretrain.sh \
  ++train_conf.max_epoch=3 \
  ++dataset_conf.batch_size=800000 \
  ++dataset_conf.batch_num_epoch=20000 \
  ++train_conf.accum_grad=8 \
  ++optim_conf.lr=4e-5
```

### 6. Check Outputs

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

Exact checkpoint names depend on your FunASR trainer settings such as `save_checkpoint_interval`, `keep_nbest_models`, and `best_model_criterion`.

### 7. Use The Checkpoint Downstream

The checkpoint is a pretraining checkpoint for the adapter model. For downstream ASR fine-tuning, load it only into a compatible architecture or write a key-mapping script for the shared encoder/frontend modules.

In FunASR configs, this is typically done through `init_param` for matching modules. This repository does not include a downstream ASR fine-tuning recipe.

## Implementation Notes

The adapter is designed to preserve the important fairseq2 w2v-BERT behavior:

- `run_frontend()` extracts masked wav2vec2 features and masked quantizer targets.
- The Conformer encoder returns all hidden states.
- An intermediate encoder layer feeds the wav2vec2 contrastive branch.
- The final encoder output feeds the BERT-style codebook prediction branch.
- The quantizer uses `entry_proj + entries + hard gumbel_softmax`.
- The temperature schedule uses `(max_temp, min_temp, decay)`.
- Diversity loss and feature penalty are scaled by the number of masked targets.

## Not Included

<<<<<<< HEAD
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

=======
This repository does not include:

- pretrained checkpoints
- speech datasets
- tokenizer or BPE models
- CMVN statistics
- downstream ASR fine-tuning recipes
- experiment logs or TensorBoard outputs

## License
This project is released under the [MIT License](LICENSE).
It adapts code and implementation ideas from FunASR and fairseq2 wav2vec2 / w2v-BERT 2.0 components. Keep copyright headers in copied or adapted source files.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution details.
