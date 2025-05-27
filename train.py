import os
import argparse
from omegaconf import OmegaConf, open_dict
from nemo.collections.asr.models import EncDecRNNTBPEModel
from helper import wget_from_nemo


class ParakeetTrainer:
    def __init__(self, data_root, model_name):
        self.data_root = data_root
        self.model_name = model_name
        self.script_dir = "scripts"
        self.config_dir = "config"
        os.makedirs(self.script_dir, exist_ok=True)
        os.makedirs(self.config_dir, exist_ok=True)

        # Download the appropriate training script and base config for Transducer
        wget_from_nemo("examples/asr/speech_to_text_finetune.py", local_dir=self.script_dir)
        wget_from_nemo("examples/asr/conf/transducer/parakeet_tdt.yaml", local_dir=self.config_dir)

    def train_model(self):
        # Load pretrained RNNT model
        model = EncDecRNNTBPEModel.from_pretrained(self.model_name)

        # Load and edit base config
        config_path = os.path.join(self.config_dir, "parakeet_tdt.yaml")
        cfg = OmegaConf.load(config_path)

        with open_dict(cfg):
            cfg.name = f"{self.model_name.replace('/', '_')}-finetune"
            cfg.init_from_pretrained_model = self.model_name

            # Dataset settings
            cfg.model.train_ds.manifest_filepath = os.path.join(self.data_root, "train_manifest.json")
            val_manifest = os.path.join(self.data_root, "val_manifest.json")
            cfg.model.validation_ds.manifest_filepath = val_manifest if os.path.exists(val_manifest) else cfg.model.train_ds.manifest_filepath

            cfg.model.train_ds.batch_size = 16
            cfg.model.validation_ds.batch_size = 16

            # Tokenizer (optional: use model default or save locally)
            tokenizer_dir = "./tokenizer"
            os.makedirs(tokenizer_dir, exist_ok=True)
            try:
                model.tokenizer.save_tokenizer(tokenizer_dir)
                cfg.model.tokenizer.dir = tokenizer_dir
                cfg.model.tokenizer.type = model.tokenizer.tokenizer_type or "bpe"
            except Exception as e:
                print(f"[WARNING] Tokenizer not saved: {e}")

            # Trainer settings
            cfg.trainer.devices = 8
            cfg.trainer.strategy = 'ddp'
            cfg.trainer.precision = 16
            cfg.trainer.max_epochs = 20
            cfg.trainer.accumulate_grad_batches = 1

            # Experiment output
            cfg.exp_manager.exp_dir = f"./nemo_experiments/{self.model_name.replace('/', '_')}"

        # Save final config
        output_config = os.path.join(self.config_dir, f"{self.model_name.replace('/', '_')}-finetune.yaml")
        OmegaConf.save(config=cfg, f=output_config)

        print(f"\n[✅] Fine-tuning config saved: {output_config}")
        print("[🚀] To train, run:")
        print(f"python scripts/speech_to_text_finetune.py --config-path {self.config_dir} --config-name {os.path.basename(output_config)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune Parakeet TDT ASR model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to folder containing train/val manifest files')
    parser.add_argument('--model_name', type=str, default='nvidia/parakeet-tdt-0.6b-v2', help='HuggingFace/NGC model name')
    args = parser.parse_args()

    trainer = ParakeetTrainer(args.data_dir, args.model_name)
    trainer.train_model()
