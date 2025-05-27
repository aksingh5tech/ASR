import os
import argparse
from omegaconf import OmegaConf, open_dict
from nemo.collections.asr.models import EncDecRNNTBPEModel  # Use RNNT model class
from helper import wget_from_nemo  # You should ensure this exists

class ASRModelTrainer:
    def __init__(self, data_root, model_name):
        self.data_root = data_root
        self.model_name = model_name
        self.config_dir = 'config'
        os.makedirs(self.config_dir, exist_ok=True)

        # Download fine-tuning config if not present
        self.config_path = os.path.join(self.config_dir, 'parakeet_tdt.yaml')
        if not os.path.exists(self.config_path):
            wget_from_nemo('examples/asr/conf/transducer/parakeet_tdt.yaml', self.config_dir)

    def train_model(self):
        # Load pretrained Parakeet ASR model (Transducer)
        asr_model = EncDecRNNTBPEModel.from_pretrained(self.model_name)

        # Save tokenizer
        tokenizer_dir = './tokenizer'
        os.makedirs(tokenizer_dir, exist_ok=True)
        try:
            asr_model.tokenizer.save_tokenizer(tokenizer_dir)
            print(f"[INFO] Tokenizer saved to {tokenizer_dir}")
        except AttributeError:
            print(f"[WARNING] This model does not support tokenizer saving.")

        # Load training config
        cfg = OmegaConf.load(self.config_path)

        with open_dict(cfg):
            cfg.name = f"{self.model_name.replace('/', '_')}-finetune"
            cfg.model.train_ds.manifest_filepath = os.path.join(self.data_root, 'train_manifest.json')
            cfg.model.validation_ds.manifest_filepath = (
                os.path.join(self.data_root, 'val_manifest.json')
                if os.path.exists(os.path.join(self.data_root, 'val_manifest.json'))
                else cfg.model.train_ds.manifest_filepath
            )
            cfg.model.train_ds.batch_size = 16
            cfg.model.validation_ds.batch_size = 16
            cfg.model.tokenizer.dir = tokenizer_dir
            cfg.model.tokenizer.type = asr_model.tokenizer.tokenizer_type
            cfg.model.optim.name = 'adamw'
            cfg.model.optim.lr = 1e-4

            cfg.trainer.devices = 1  # Set as per your GPU count
            cfg.trainer.strategy = 'ddp' if cfg.trainer.devices > 1 else None
            cfg.trainer.precision = 16
            cfg.trainer.max_epochs = 20
            cfg.trainer.accumulate_grad_batches = 1

            cfg.exp_manager.exp_dir = f'./nemo_experiments/{self.model_name.replace("/", "_")}'

        output_config_path = os.path.join(self.config_dir, f"{self.model_name.replace('/', '_')}-finetune.yaml")
        OmegaConf.save(config=cfg, f=output_config_path)

        print(f"[INFO] Updated config saved to {output_config_path}")
        print(f"[INFO] Run training using:")
        print(f"bash run_train.sh {output_config_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune Parakeet TDT ASR model')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing NeMo manifests')
    parser.add_argument('--model_name', type=str, default='nvidia/parakeet-tdt-0.6b-v2', help='Hugging Face model name')
    args = parser.parse_args()

    trainer = ASRModelTrainer(args.data_dir, args.model_name)
    trainer.train_model()
