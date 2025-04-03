import os
import torch
import nemo.collections.asr as nemo_asr
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from nemo.collections.asr.parts.utils.manifest_utils import create_manifest


class ASRFineTuner:
    def __init__(
        self,
        model_name="QuartzNet15x5Base-En",
        output_dir="./fine_tuned_asr_model",
        vocabulary=list("abcdefghijklmnopqrstuvwxyz' "),
        batch_size=16,
        num_workers=4,
        max_epochs=10,
        lr=0.001,
        betas=(0.95, 0.5),
        weight_decay=1e-5,
        data_root="./librispeech_data"
    ):
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_epochs = max_epochs
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay

        self.train_manifest, self.val_manifest = self._prepare_librispeech(data_root)

        self.model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name=model_name)
        self.model.change_vocabulary(new_vocabulary=vocabulary)

        self.config = self.model.cfg
        self._update_config()

        self.trainer = Trainer(
            max_epochs=self.max_epochs,
            gpus=1 if torch.cuda.is_available() else 0,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
        )

    def _prepare_librispeech(self, data_root):
        print("Assuming LibriSpeech already downloaded and extracted manually...")

        train_dir = os.path.join(data_root, "train-clean-100")
        val_dir = os.path.join(data_root, "dev-clean")

        train_manifest = os.path.join(data_root, "train_manifest.json")
        val_manifest = os.path.join(data_root, "val_manifest.json")

        if not os.path.exists(train_manifest):
            print(f"Creating train manifest at: {train_manifest}")
            create_manifest(train_dir, train_manifest)

        if not os.path.exists(val_manifest):
            print(f"Creating validation manifest at: {val_manifest}")
            create_manifest(val_dir, val_manifest)

        return train_manifest, val_manifest

    def _update_config(self):
        self.config.train_ds.manifest_filepath = self.train_manifest
        self.config.train_ds.batch_size = self.batch_size
        self.config.train_ds.num_workers = self.num_workers

        self.config.validation_ds.manifest_filepath = self.val_manifest
        self.config.validation_ds.batch_size = self.batch_size
        self.config.validation_ds.num_workers = self.num_workers

        self.config.optim.lr = self.lr
        self.config.optim.betas = self.betas
        self.config.optim.weight_decay = self.weight_decay

    def fine_tune(self):
        self.model.setup_training_data(train_data_config=self.config.train_ds)
        self.model.setup_validation_data(val_data_config=self.config.validation_ds)

        print("Starting fine-tuning...")
        self.trainer.fit(self.model)
        print("Fine-tuning complete.")

        os.makedirs(self.output_dir, exist_ok=True)
        self.model.save_to(self.output_dir)
        print(f"Model saved to {self.output_dir}")

    def transcribe(self, audio_paths):
        print("Transcribing audio files...")
        return self.model.transcribe(audio_paths)

if __name__ == "__main__":
    tuner = ASRFineTuner()
    tuner.fine_tune()

    # Test transcription with sample file (or one from the downloaded dataset)
    results = tuner.transcribe(["./librispeech_data/dev-clean/84/121123/84-121123-0001.flac"])
    print("Transcription:", results)
