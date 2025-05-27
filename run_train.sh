#!/bin/bash

set -e  # Exit on any error

MODEL_NAME=$1                # e.g., nvidia/parakeet-tdt-0.6b-v2
TRAIN_MANIFEST_PATH=$2       # e.g., datasets/Medical-ASR/train_manifest.json
VAL_MANIFEST_PATH=$3         # optional (defaults to train_manifest)

if [ -z "$MODEL_NAME" ] || [ -z "$TRAIN_MANIFEST_PATH" ]; then
  echo "Usage: bash run_train.sh <model_name> <train_manifest_path> [val_manifest_path]"
  exit 1
fi

DATA_DIR=$(dirname "$TRAIN_MANIFEST_PATH")
VAL_MANIFEST_PATH=${VAL_MANIFEST_PATH:-$TRAIN_MANIFEST_PATH}
EXP_NAME="${MODEL_NAME//\//_}-finetune"
EXP_DIR="./nemo_experiments/${MODEL_NAME//\//_}"

# Launch training
torchrun --nproc_per_node=8 \
    examples/asr/speech_to_text_finetune.py \
    name=$EXP_NAME \
    model.train_ds.manifest_filepath=$TRAIN_MANIFEST_PATH \
    model.validation_ds.manifest_filepath=$VAL_MANIFEST_PATH \
    model.train_ds.batch_size=16 \
    model.validation_ds.batch_size=16 \
    model.optim.lr=1e-4 \
    model.optim.name=adam \
    model.tokenizer.dir=./tokenizer \
    model.tokenizer.type=bpe \
    trainer.devices=8 \
    trainer.strategy=ddp \
    trainer.precision=16 \
    trainer.max_epochs=20 \
    trainer.accumulate_grad_batches=1 \
    exp_manager.exp_dir=$EXP_DIR
