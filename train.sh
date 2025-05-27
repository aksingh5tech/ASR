#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python /root/venv/lib/python3.10/site-packages/nemo/collections/asr/scripts/speech_to_text_finetune.py \
  --config-path=/path/to/config_dir \
  --config-name=parakeet_medical_config.yaml \
  model.train_ds.manifest_filepath=/path/to/train.json \
  model.validation_ds.manifest_filepath=/path/to/validation.json \
  exp_manager.exp_dir=results/parakeet_medical
