import os
import json
import glob
import tqdm
import librosa
import torch
import tarfile
import numpy as np
import wget
import nemo.collections.asr as nemo_asr
from pytorch_lightning import Trainer


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
        data_root="./datasets"
    ):
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_epochs = max_epochs
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay

        self.train_manifest = os.path.join(data_root, "LibriLight/train_manifest.json")
        self._prepare_librilight(data_root)

        self.model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name=model_name)
        self.model.change_vocabulary(new_vocabulary=vocabulary)

        self.config = self.model.cfg
        self._update_config()

        self.trainer = Trainer(
            max_epochs=self.max_epochs,
            gpus=1 if torch.cuda.is_available() else 0,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
        )

    def _prepare_librilight(self, data_root):
        libri_dir = os.path.join(data_root, "LibriLight")
        tgz_path = os.path.join(data_root, "librispeech_finetuning.tgz")

        if not os.path.exists(data_root):
            os.makedirs(data_root)

        if not os.path.exists(tgz_path):
            url = "https://dl.fbaipublicfiles.com/librilight/data/librispeech_finetuning.tgz"
            print("Downloading LibriLight dataset...")
            wget.download(url, out=tgz_path)
            print(f"\nDownloaded dataset to: {tgz_path}")

        if not os.path.exists(libri_dir):
            print("Extracting LibriLight dataset...")
            with tarfile.open(tgz_path) as tar:
                tar.extractall(path=libri_dir)
            print(f"Extracted to: {libri_dir}")

        # Build manifest from 1h finetuning subset
        if not os.path.exists(self.train_manifest):
            print("Building manifest from 1h subset...")
            self.build_manifest(data_root, self.train_manifest)
        else:
            print(f"Using existing manifest: {self.train_manifest}")

    def build_manifest(self, data_root, manifest_path):
        transcript_list = glob.glob(
            os.path.join(data_root, 'LibriLight/librispeech_finetuning/1h/**/*.txt'), recursive=True
        )
        tot_duration = 0
        with open(manifest_path, 'w') as fout:
            pass  # ensure new file

        for transcript_path in tqdm.tqdm(transcript_list):
            wav_dir = os.path.dirname(transcript_path)
            with open(transcript_path, 'r') as fin, open(manifest_path, 'a') as fout:
                for line in fin:
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue
                    file_id = parts[0]
                    transcript = ' '.join(parts[1:]).lower().strip()
                    audio_path = os.path.join(wav_dir, f'{file_id}.flac')
                    try:
                        duration = librosa.get_duration(path=audio_path)
                    except Exception as e:
                        print(f"Skipping {audio_path} due to error: {e}")
                        continue

                    tot_duration += duration
                    metadata = {
                        "audio_filepath": audio_path,
                        "duration": duration,
                        "text": transcript,
                        "target_lang": "en",
                        "source_lang": "en",
                        "pnc": "False"
                    }
                    json.dump(metadata, fout)
                    fout.write('\n')
        print(f'\n{np.round(tot_duration/3600, 2)} hours of audio ready for training')

    def _update_config(self):
        self.config.train_ds.manifest_filepath = self.train_manifest
        self.config.train_ds.batch_size = self.batch_size
        self.config.train_ds.num_workers = self.num_workers

        self.config.validation_ds.manifest_filepath = self.train_manifest  # reuse same for small 1h finetune
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

    # Example: transcribe an audio file from LibriLight dev set
    results = tuner.transcribe(["datasets/LibriLight/dev/84/121123/84-121123-0001.flac"])
    print("Transcription:", results)
