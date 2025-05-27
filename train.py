import os
import argparse
import subprocess
from omegaconf import OmegaConf, open_dict
from nemo.collections.asr.models import EncDecCTCModel
from helper import wget_from_nemo  # Assumes you have a wget helper

class ASRModelTrainer:
    def __init__(self, data_root, model_name, manifest_path):
        self.data_root = data_root
        self.model_name = model_name
        self.manifest_path = manifest_path

        # Download fine-tuning script and config
        # Download fine-tuning script and config only if not already present
        if not os.path.exists('examples/asr/speech_to_text_finetune.py'):
            wget_from_nemo('examples/asr/speech_to_text_finetune.py')
        if not os.path.exists('config/parakeet_tdt.yaml'):
            wget_from_nemo('examples/asr/conf/ctc/parakeet_tdt.yaml', 'config')

    def train_model(self):
        # Load pretrained Parakeet model
        asr_model = EncDecCTCModel.from_pretrained(self.model_name)

        # Create tokenizer directory and save tokenizer
        tokenizer_dir = './tokenizer'
        os.makedirs(tokenizer_dir, exist_ok=True)
        try:
            asr_model.tokenizer.save_tokenizer(tokenizer_dir)
            print(f"[INFO] Tokenizer saved to {tokenizer_dir}")
        except AttributeError:
            print(f"[WARNING] This model does not support tokenizer saving.")

        # Load default Parakeet config
        config_path = 'config/parakeet_tdt.yaml'
        if not os.path.exists(config_path):
            print(f"[INFO] Downloading default parakeet config to {config_path}")
            wget_from_nemo('examples/asr/conf/ctc/parakeet_tdt.yaml', 'config')

        cfg = OmegaConf.load(config_path)

        # Update config fields for finetuning
        from omegaconf import open_dict
        with open_dict(cfg):
            cfg.name = f"{self.model_name.replace('/', '_')}-finetune"
            cfg.model.train_ds.manifest_filepath = os.path.join(self.data_root, 'train_manifest.json')
            cfg.model.validation_ds.manifest_filepath = os.path.join(self.data_root, 'val_manifest.json') \
                if os.path.exists(
                os.path.join(self.data_root, 'val_manifest.json')) else cfg.model.train_ds.manifest_filepath
            cfg.model.train_ds.batch_size = 16
            cfg.model.validation_ds.batch_size = 16
            cfg.model.tokenizer.dir = tokenizer_dir
            cfg.model.tokenizer.type = asr_model.tokenizer.tokenizer_type
            cfg.model.optim.name = 'adam'
            cfg.model.optim.lr = 1e-4
            cfg.trainer.devices = 8
            cfg.trainer.strategy = 'ddp'
            cfg.trainer.precision = 16
            cfg.trainer.max_epochs = 20
            cfg.trainer.accumulate_grad_batches = 1
            cfg.exp_manager.exp_dir = f'./nemo_experiments/{self.model_name.replace("/", "_")}'

        # Save finetune config
        output_config_path = f"config/{self.model_name.replace('/', '_')}-finetune.yaml"
        OmegaConf.save(config=cfg, f=output_config_path)
        print(f"[INFO] Updated training config saved to {output_config_path}")
        print(f"[INFO] Now run training using:")
        print(f"bash run_train.sh {self.model_name} {os.path.abspath(cfg.model.train_ds.manifest_filepath)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune Parakeet ASR model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset root')
    parser.add_argument('--model_name', type=str, default='nvidia/parakeet-tdt-0.6b-v2', help='Pretrained model name')
    parser.add_argument('--manifest_path', type=str, required=False, help='(Unused, required only by legacy scripts)')
    args = parser.parse_args()

    trainer = ASRModelTrainer(args.data_dir, args.model_name, args.manifest_path)
    trainer.train_model()
