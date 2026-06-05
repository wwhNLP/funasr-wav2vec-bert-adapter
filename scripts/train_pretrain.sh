#!/usr/bin/env bash
set -euo pipefail

# Path to an installed or cloned FunASR repository.
FUNASR_ROOT="${FUNASR_ROOT:-/path/to/FunASR}"

# Path to this repository. Override when launching from another directory.
MODEL_DIR="${MODEL_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# Train/valid shard-list files. Each line should point to a WebDataset-style tar shard.
TRAIN_LIST="${TRAIN_LIST:-/path/to/train_shards.list}"
VALID_LIST="${VALID_LIST:-/path/to/valid_shards.list}"

OUTPUT_DIR="${OUTPUT_DIR:-${MODEL_DIR}/exp/w2vbert_pretrain}"
CONFIG_PATH="${CONFIG_PATH:-${MODEL_DIR}/configs/w2vbert_pretrain.yaml}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${MODEL_DIR}/deepspeed_conf/ds_stage1.json}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"

GPU_NUM="$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F "," '{print NF}')"
mkdir -p "${OUTPUT_DIR}"

DISTRIBUTED_ARGS=(
  --nnodes "${WORLD_SIZE:-1}"
  --nproc_per_node "${GPU_NUM}"
  --node_rank "${RANK:-0}"
  --master_addr "${MASTER_ADDR:-127.0.0.1}"
  --master_port "${MASTER_PORT:-26670}"
)

TRAIN_TOOL="${FUNASR_ROOT}/funasr/bin/train_ds.py"

torchrun "${DISTRIBUTED_ARGS[@]}" \
  "${TRAIN_TOOL}" \
  --config-path "$(dirname "${CONFIG_PATH}")" \
  --config-name "$(basename "${CONFIG_PATH}" .yaml)" \
  ++model="${MODEL_DIR}" \
  hydra.run.dir="${OUTPUT_DIR}/hydra_log" \
  ++trust_remote_code=true \
  ++remote_code="./" \
  ++train_data_set_list="${TRAIN_LIST}" \
  ++valid_data_set_list="${VALID_LIST}" \
  ++dataset_conf.data_split_num=1 \
  ++dataset_conf.dataloader="DataloaderIterable" \
  ++dataset_conf.batch_sampler=null \
  ++train_conf.deepspeed_config="${DEEPSPEED_CONFIG}" \
  ++output_dir="${OUTPUT_DIR}" \
  "$@" 2>&1 | tee "${OUTPUT_DIR}/train.log"
