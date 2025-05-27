#!/bin/bash

MODEL_NAME=$1                # e.g., nvidia/parakeet-tdt-0.6b-v2
TRAIN_MANIFEST_PATH=$2       # e.g., datasets/jarvisx17_Medical-ASR-EN/train_manifest.json
VAL_MANIFEST_PATH=$3         # optional

DATA_DIR=$(dirname "$TRAIN_MANIFEST_PATH")

# Launch training using torchrun for multi-GPU DDP
torchrun --nproc_per_node=8 \
    examples/asr/speech_to_text_finetune.py \
    model.train_ds.manifest_filepath=${TRAIN_MANIFEST_PATH} \
    model.validation_ds.manifest_filepath=${VAL_MANIFEST_PATH:-$TRAIN_MANIFEST_PATH} \
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
    exp_manager.exp_dir=./nemo_experiments/${MODEL_NAME//\//_} \
    name=${MODEL_NAME//\//_}-finetune
