#!/bin/bash
MODEL_NAME=$1
MANIFEST_PATH=$2

torchrun --nproc_per_node=8 examples/asr/speech_to_text_finetune.py \
  model.train_ds.manifest_filepath=$MANIFEST_PATH \
  model.validation_ds.manifest_filepath=${MANIFEST_PATH/train/val} \
  model.tokenizer.dir=./tokenizer \
  model.tokenizer.type=bpe \
  model.optim.lr=1e-4 \
  trainer.devices=8 \
  trainer.strategy=ddp \
  trainer.precision=16 \
  trainer.accumulate_grad_batches=1 \
  trainer.max_epochs=20 \
  exp_manager.exp_dir=./nemo_experiments/${MODEL_NAME//\//_} \
  ++model.init_from_pretrained_model=$MODEL_NAME
