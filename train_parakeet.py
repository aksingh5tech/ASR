import os
import argparse
from omegaconf import OmegaConf, open_dict
from nemo.collections.asr.models import EncDecRNNTBPEModel
from helper import wget_from_nemo

"""
Usage:

python train_parakeet.py --dataset_name jarvisx17/Medical-ASR-EN

python scripts/speech_to_text_finetune.py \
  --config-path ../config \
  --config-name nvidia_parakeet-tdt-0.6b-v2-finetune.yaml

bash run_train.sh nvidia/parakeet-tdt-0.6b-v2 datasets/jarvisx17_Medical-ASR-EN
"""

class ParakeetTrainer:
    def __init__(self, dataset_name, model_name):
        self.dataset_name = dataset_name
        self.model_name = model_name

        self.data_dir = os.path.join("datasets", dataset_name.replace("/", "_"))
        self.train_manifest = os.path.join(self.data_dir, "train_manifest.json")
        self.val_manifest = os.path.join(self.data_dir, "val_manifest.json")
        self.config_dir = "config"
        self.script_dir = "scripts"

        os.makedirs(self.config_dir, exist_ok=True)
        os.makedirs(self.script_dir, exist_ok=True)

        # Download training script if not already present
        wget_from_nemo("examples/asr/speech_to_text_finetune.py", local_dir=self.script_dir)

    def train_model(self):
        print("[INFO] Loading pretrained model...")
        model = EncDecRNNTBPEModel.from_pretrained(self.model_name)

        print("[INFO] Building training config from model defaults...")
        cfg = OmegaConf.create({
            "name": f"{self.model_name.replace('/', '_')}-finetune",
            "init_from_pretrained_model": self.model_name,
            "model": model.cfg,
            "trainer": {},
            "exp_manager": {}
        })

        with open_dict(cfg):
            # Update train and validation settings
            with open_dict(cfg.model.train_ds):
                cfg.model.train_ds.manifest_filepath = self.train_manifest
                cfg.model.train_ds.batch_size = 16

            with open_dict(cfg.model.validation_ds):
                cfg.model.validation_ds.manifest_filepath = (
                    self.val_manifest if os.path.exists(self.val_manifest) else self.train_manifest
                )
                cfg.model.validation_ds.batch_size = 16

            # Tokenizer save (may be skipped if not supported)
            tokenizer_dir = "./tokenizer"
            os.makedirs(tokenizer_dir, exist_ok=True)
            try:
                model.tokenizer.save_tokenizer(tokenizer_dir)
                cfg.model.tokenizer.dir = tokenizer_dir
                cfg.model.tokenizer.type = model.tokenizer.tokenizer_type or "bpe"
            except Exception as e:
                print(f"[WARNING] Tokenizer not saved: {e}")

            # Trainer configuration
            with open_dict(cfg.trainer):
                cfg.trainer.devices = 8
                cfg.trainer.strategy = 'ddp'
                cfg.trainer.precision = 16
                cfg.trainer.max_epochs = 20
                cfg.trainer.accumulate_grad_batches = 1
                cfg.trainer.logger = False  # Disable Lightning logger
                cfg.trainer.enable_checkpointing = False  # ✅ Disable Lightning checkpointing

            # exp_manager settings (logger fix applied)
            with open_dict(cfg.exp_manager):
                cfg.exp_manager.exp_dir = f"./nemo_experiments/{self.model_name.replace('/', '_')}"
                cfg.exp_manager.create_tensorboard_logger = True  # ✅ Let exp_manager control logging
                cfg.exp_manager.create_wandb_logger = False
                cfg.exp_manager.create_mlflow_logger = False
                cfg.exp_manager.create_clearml_logger = False
                cfg.exp_manager.create_dllogger_logger = False

        config_output = os.path.join(self.config_dir, f"{self.model_name.replace('/', '_')}-finetune.yaml")
        OmegaConf.save(cfg, config_output)

        print(f"\n[✅] Config saved: {config_output}")
        print("[🚀] To start training, run:")
        print(f"python scripts/speech_to_text_finetune.py --config-path {self.config_dir} --config-name {os.path.basename(config_output)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Parakeet TDT model on a dataset.")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Dataset name (e.g., 'jarvisx17/Medical-ASR-EN')")
    parser.add_argument("--model_name", type=str, default="nvidia/parakeet-tdt-0.6b-v2",
                        help="Pretrained model to fine-tune")
    args = parser.parse_args()

    trainer = ParakeetTrainer(args.dataset_name, args.model_name)
    trainer.train_model()
