#!/bin/bash

# Set variables
DATA_DIR="datasets"
MODEL_NAME="nvidia/canary-1b-flash"
MANIFEST_PATH="${DATA_DIR}/jarvisx17_Medical-ASR-EN/train_manifest.json"
CONFIG_NAME="${MODEL_NAME//\//_}-finetune.yaml"

# Step 1: Run setup script
echo "Running Python setup..."
python train.py \
  --data_dir "$DATA_DIR" \
  --model_name "$MODEL_NAME" \
  --manifest_path "$MANIFEST_PATH"

# Step 2: Launch training
echo "Starting training..."
python scripts/speech_to_text_aed.py \
  --config-path=../config \
  --config-name="$CONFIG_NAME" \
  name="${MODEL_NAME//\//_}-finetune" \
  model.train_ds.manifest_filepath="$MANIFEST_PATH" \
  model.validation_ds.manifest_filepath="$MANIFEST_PATH" \
  model.test_ds.manifest_filepath="$MANIFEST_PATH" \
  exp_manager.exp_dir=canary_results \
  exp_manager.resume_ignore_no_checkpoint=true \
  trainer.max_steps=50 \
  trainer.log_every_n_steps=1
