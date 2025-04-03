import os
from nemo.utils import wget_from_nemo

# Download NeMo training script and config
wget_from_nemo('examples/asr/speech_multitask/speech_to_text_aed.py')
wget_from_nemo('examples/asr/conf/speech_multitask/fast-conformer_aed.yaml', 'config')

# Define your LibriLight manifest
MANIFEST = os.path.abspath("datasets/LibriLight/train_manifest.json")

# Launch training
os.system(f"""
HYDRA_FULL_ERROR=1 python speech_to_text_aed.py \
  --config-path="config" \
  --config-name="fast-conformer_aed.yaml" \
  name="canary-small" \
  model.prompt_format="canary2" \
  model.train_ds.manifest_filepath={MANIFEST} \
  model.validation_ds.manifest_filepath={MANIFEST} \
  model.test_ds.manifest_filepath={MANIFEST} \
  model.tokenizer.langs.en.dir="tokenizers/en_libri1h_1024/tokenizer_spe_bpe_v1024" \
  model.tokenizer.langs.spl_tokens.dir="tokenizers/spl_tokens" \
  spl_tokens.model_dir="tokenizers/spl_tokens" \
  model.encoder.n_layers=2 \
  model.transf_decoder.config_dict.num_layers=2 \
  exp_manager.exp_dir="canary_results" \
  exp_manager.resume_ignore_no_checkpoint=true \
  trainer.max_steps=10 \
  trainer.log_every_n_steps=1
""")
