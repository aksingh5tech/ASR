# Define paths
export BRANCH=main
export NEMO_DIR=./NeMo
export DATA_DIR=./data  # Adjust if needed

# Clone NeMo if not already done
git clone -b $BRANCH https://github.com/NVIDIA/NeMo $NEMO_DIR

# Split training data into 90% train / 10% val
python - <<EOF
import json, random, os
manifest_path = "$DATA_DIR/medical_asr_converted/train_manifest.json"
train_out = "$DATA_DIR/medical_asr_converted/train_split.json"
val_out = "$DATA_DIR/medical_asr_converted/val_split.json"

with open(manifest_path) as f:
    lines = f.readlines()

random.shuffle(lines)
split = int(0.9 * len(lines))

with open(train_out, "w") as f:
    f.writelines(lines[:split])
with open(val_out, "w") as f:
    f.writelines(lines[split:])
EOF

# Run NeMo fine-tuning
python $NEMO_DIR/examples/asr/speech_to_text_finetune.py \
  --config-path=/root/ASR/NeMo/examples/asr/conf/fastconformer/hybrid_transducer_ctc \
  --config-name=fastconformer_hybrid_transducer_ctc_bpe \
  +init_from_pretrained_model=stt_en_fastconformer_hybrid_large_pc \
  +model.train_ds.manifest_filepath="$DATA_DIR/medical_asr_converted/train_split.json" \
  +model.validation_ds.manifest_filepath="$DATA_DIR/medical_asr_converted/val_split.json" \
  +model.validation_ds.sample_rate=44100 \
  +model.tokenizer.dir=None \
  +model.tokenizer.type=char \
  +trainer.devices=1 \
  +trainer.max_epochs=1 \
  +trainer.precision=32 \
  +model.optim.name=adamw \
  +model.optim.lr=0.01 \
  +model.optim.weight_decay=0.001 \
  +model.optim.sched.warmup_steps=100 \
  +exp_manager.version=medical_test \
  +exp_manager.use_datetime_version=False \
  +exp_manager.exp_dir="$DATA_DIR/checkpoints"
