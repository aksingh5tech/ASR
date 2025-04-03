import os
import argparse
import librosa
import numpy as np
import subprocess
from omegaconf import OmegaConf
from nemo.collections.asr.models import EncDecMultiTaskModel
from helper import wget_from_nemo  # Import the helper method

class ASRModelTrainer:
    def __init__(self, data_root, model_name, manifest_path):
        self.data_root = data_root
        self.model_name = model_name
        self.manifest_path = manifest_path

        # Download required training and config scripts
        wget_from_nemo('examples/asr/speech_multitask/speech_to_text_aed.py')
        wget_from_nemo('examples/asr/conf/speech_multitask/fast-conformer_aed.yaml', 'config')

    def train_model(self):
        if 'canary_model' not in globals():
            canary_model = EncDecMultiTaskModel.from_pretrained(self.model_name)

        base_model_cfg = OmegaConf.load("config/fast-conformer_aed.yaml")
        base_model_cfg['name'] = f"{self.model_name.replace('/', '_')}-finetune"
        base_model_cfg.pop("init_from_nemo_model", None)
        base_model_cfg['init_from_pretrained_model'] = self.model_name

        canary_model.save_tokenizers('./canary_flash_tokenizers/')

        for lang in os.listdir('canary_flash_tokenizers'):
            base_model_cfg['model']['tokenizer']['langs'][lang] = {
                'dir': os.path.join('canary_flash_tokenizers', lang),
                'type': 'bpe'
            }

        base_model_cfg['spl_tokens']['model_dir'] = os.path.join('canary_flash_tokenizers', "spl_tokens")
        base_model_cfg['model']['prompt_format'] = canary_model._cfg['prompt_format']
        base_model_cfg['model']['prompt_defaults'] = canary_model._cfg['prompt_defaults']
        base_model_cfg['model']['model_defaults'] = canary_model._cfg['model_defaults']
        base_model_cfg['model']['preprocessor'] = canary_model._cfg['preprocessor']
        base_model_cfg['model']['encoder'] = canary_model._cfg['encoder']
        base_model_cfg['model']['transf_decoder'] = canary_model._cfg['transf_decoder']
        base_model_cfg['model']['transf_encoder'] = canary_model._cfg['transf_encoder']

        cfg = OmegaConf.create(base_model_cfg)
        config_path = f"config/{self.model_name.replace('/', '_')}-finetune.yaml"
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            OmegaConf.save(cfg, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an ASR model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset root')
    parser.add_argument('--model_name', type=str, required=True, help='Pretrained model name to use (e.g., nvidia/canary-1b-flash)')
    parser.add_argument('--manifest_path', type=str, required=True, help='Path to the training manifest file')
    args = parser.parse_args()

    trainer = ASRModelTrainer(args.data_dir, args.model_name, args.manifest_path)
    trainer.train_model()