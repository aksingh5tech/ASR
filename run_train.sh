#!/bin/bash

set -e
export HYDRA_FULL_ERROR=1  # Get full traceback on failure

MODEL_NAME=$1                # e.g., nvidia/parakeet-tdt-0.6b-v2
DATA_DIR=$2                 # e.g., datasets/Medical-ASR
GPUS=${3:-8}                # Default to 8 GPUs unless overridden

# Paths
CONFIG_NAME="${MODEL_NAME//\//_}-finetune.yaml"
CONFIG_PATH="config"
SCRIPT_PATH="scripts/speech_to_text_finetune.py"

# Validate inputs
if [ -z "$MODEL_NAME" ] || [ -z "$DATA_DIR" ]; then
  echo "Usage: bash run_train.sh <model_name> <data_dir> [gpu_count]"
  exit 1
fi

# Run training
echo "[🚀] Starting training with $GPUS GPU(s)..."
torchrun --nproc_per_node=$GPUS \
  $SCRIPT_PATH \
  --config-path $CONFIG_PATH \
  --config-name $CONFIG_NAME \
  model.train_ds.manifest_filepath=$DATA_DIR/train_manifest.json \
  model.validation_ds.manifest_filepath=${DATA_DIR}/val_manifest.json \
  exp_manager.exp_dir=nemo_experiments/${MODEL_NAME//\//_} \
  trainer.devices=$GPUS \
  trainer.strategy=ddp \
  trainer.precision=16 \
  trainer.max_epochs=20 \
  trainer.accumulate_grad_batches=1
