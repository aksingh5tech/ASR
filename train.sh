#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python speech_to_text_finetune.py \
  --config-path=./config \
  --config-name=nvidia_parakeet-tdt-0.6b-v2-finetune.yaml