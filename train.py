import os
import argparse
from omegaconf import OmegaConf, open_dict
from nemo.collections.asr.models import EncDecRNNTBPEModel
from helper import wget_from_nemo

'''

python train_parakeet.py --dataset_name jarvisx17/Medical-ASR-EN

bash run_train.sh nvidia/parakeet-tdt-0.6b-v2 datasets/jarvisx17_Medical-ASR-EN

'''

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

        # Download necessary NeMo files
        wget_from_nemo("examples/asr/speech_to_text_finetune.py", local_dir=self.script_dir)
        wget_from_nemo("examples/asr/conf/transducer/parakeet_tdt.yaml", local_dir=self.config_dir)

    def train_model(self):
        model = EncDecRNNTBPEModel.from_pretrained(self.model_name)

        cfg_path = os.path.join(self.config_dir, "parakeet_tdt.yaml")
        cfg = OmegaConf.load(cfg_path)

        with open_dict(cfg):
            cfg.name = f"{self.model_name.replace('/', '_')}-finetune"
            cfg.init_from_pretrained_model = self.model_name

            cfg.model.train_ds.manifest_filepath = self.train_manifest
            cfg.model.validation_ds.manifest_filepath = (
                self.val_manifest if os.path.exists(self.val_manifest) else self.train_manifest
            )

            cfg.model.train_ds.batch_size = 16
            cfg.model.validation_ds.batch_size = 16

            tokenizer_dir = "./tokenizer"
            os.makedirs(tokenizer_dir, exist_ok=True)
            try:
                model.tokenizer.save_tokenizer(tokenizer_dir)
                cfg.model.tokenizer.dir = tokenizer_dir
                cfg.model.tokenizer.type = model.tokenizer.tokenizer_type or "bpe"
            except Exception as e:
                print(f"[WARNING] Tokenizer not saved: {e}")

            cfg.trainer.devices = 8
            cfg.trainer.strategy = 'ddp'
            cfg.trainer.precision = 16
            cfg.trainer.max_epochs = 20
            cfg.trainer.accumulate_grad_batches = 1

            cfg.exp_manager.exp_dir = f"./nemo_experiments/{self.model_name.replace('/', '_')}"

        config_output = os.path.join(self.config_dir, f"{self.model_name.replace('/', '_')}-finetune.yaml")
        OmegaConf.save(cfg, config_output)

        print(f"\n[✅] Config saved: {config_output}")
        print("[🚀] To train, run:")
        print(f"python scripts/speech_to_text_finetune.py --config-path config --config-name {os.path.basename(config_output)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Parakeet TDT on a prepared dataset.")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Dataset name (e.g., 'jarvisx17/Medical-ASR-EN')")
    parser.add_argument("--model_name", type=str, default="nvidia/parakeet-tdt-0.6b-v2",
                        help="Pretrained model to fine-tune")

    args = parser.parse_args()
    trainer = ParakeetTrainer(args.dataset_name, args.model_name)
    trainer.train_model()
