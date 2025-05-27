import os
import argparse
import librosa
import numpy as np
import subprocess
from omegaconf import OmegaConf, open_dict
from nemo.collections.asr.models import EncDecMultiTaskModel
from helper import wget_from_nemo  # Import the helper method


class ASRModelTrainer:
    def __init__(self, data_root, model_name, manifest_path):
        self.data_root = data_root
        self.model_name = model_name
        self.manifest_path = manifest_path

        # Download training script and base config
        wget_from_nemo('examples/asr/speech_multitask/speech_to_text_aed.py')
        wget_from_nemo('examples/asr/conf/speech_multitask/fast-conformer_aed.yaml', 'config')

    def train_model(self):
        model = EncDecMultiTaskModel.from_pretrained(self.model_name)

        # Load and prepare base config
        base_config_path = "config/fast-conformer_aed.yaml"
        cfg = OmegaConf.load(base_config_path)

        with open_dict(cfg):
            cfg.name = f"{self.model_name.replace('/', '_')}-finetune"
            cfg.init_from_pretrained_model = self.model_name

            cfg.model.train_ds.manifest_filepath = os.path.join(self.data_root, "train_manifest.json")
            cfg.model.validation_ds.manifest_filepath = os.path.join(self.data_root, "val_manifest.json") if os.path.exists(os.path.join(self.data_root, "val_manifest.json")) else cfg.model.train_ds.manifest_filepath
            cfg.model.train_ds.batch_size = 16
            cfg.model.validation_ds.batch_size = 16

            tokenizer_dir = './canary_flash_tokenizers/'
            os.makedirs(tokenizer_dir, exist_ok=True)
            model.save_tokenizers(tokenizer_dir)

            # Register tokenizers per language (assumes multi-language structure)
            cfg.model.tokenizer.langs = {}
            for lang in os.listdir(tokenizer_dir):
                lang_path = os.path.join(tokenizer_dir, lang)
                if os.path.isdir(lang_path):
                    cfg.model.tokenizer.langs[lang] = {
                        'dir': lang_path,
                        'type': 'bpe'
                    }

            cfg.spl_tokens.model_dir = os.path.join(tokenizer_dir, "spl_tokens")
            cfg.model.prompt_format = model._cfg.get('prompt_format', None)
            cfg.model.prompt_defaults = model._cfg.get('prompt_defaults', None)
            cfg.model.model_defaults = model._cfg.get('model_defaults', None)
            cfg.model.preprocessor = model._cfg.get('preprocessor', None)
            cfg.model.encoder = model._cfg.get('encoder', None)
            cfg.model.transf_decoder = model._cfg.get('transf_decoder', None)
            cfg.model.transf_encoder = model._cfg.get('transf_encoder', None)

        # Save the updated config
        config_path = f"config/{self.model_name.replace('/', '_')}-finetune.yaml"
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        OmegaConf.save(config=cfg, f=config_path)
        print(f"[INFO] Fine-tuning config saved to: {config_path}")
        print(f"[INFO] You can now train using:")
        print(f"python examples/asr/speech_multitask/speech_to_text_aed.py --config-path config --config-name {os.path.basename(config_path)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune Canary ASR model')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing train/val manifests')
    parser.add_argument('--model_name', type=str, required=True, help='Model name from HuggingFace or NGC (e.g., nvidia/canary-1b-flash)')
    parser.add_argument('--manifest_path', type=str, required=True, help='Path to training manifest file (used to infer data root)')
    args = parser.parse_args()

    trainer = ASRModelTrainer(args.data_dir, args.model_name, args.manifest_path)
    trainer.train_model()
