#!/bin/bash

# Check for required arguments
if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <model_name> <manifest_path>"
  exit 1
fi

# Set variables from arguments
MODEL_NAME="$1"
MANIFEST_PATH="$2"

# Other constants
DATA_DIR="datasets"
CONFIG_NAME="${MODEL_NAME//\//_}-finetune.yaml"

# Step 1: Run setup script
echo "Running Training setup with model: $MODEL_NAME"
echo "Using manifest: $MANIFEST_PATH"
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
