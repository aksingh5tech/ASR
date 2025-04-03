import os
import json
import librosa
import soundfile as sf
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

class MedicalASRDataHandler:
    def __init__(self, data_dir="datasets", split="train"):
        self.data_dir = data_dir
        self.split = split
        self.dataset_name = "jarvisx17/Medical-ASR-EN"
        self.output_dir = os.path.join(data_dir, "Medical-ASR-EN")
        self.audio_dir = os.path.join(self.output_dir, "audio")
        self.manifest_path = os.path.join(self.output_dir, "manifest.json")
        os.makedirs(self.audio_dir, exist_ok=True)

    def download_dataset(self):
        print(f"Loading dataset '{self.dataset_name}' split '{self.split}'...")
        dataset = load_dataset(self.dataset_name, split=self.split)
        print(f"Loaded {len(dataset)} samples.")
        return dataset

    def build_manifest(self):
        dataset = self.download_dataset()
        total_duration = 0

        with open(self.manifest_path, 'w') as fout:
            for i, sample in tqdm(enumerate(dataset), total=len(dataset)):
                audio_data = sample["audio"]
                text = sample["transcription"].strip().lower()

                sr = 16000
                if audio_data["sampling_rate"] != sr:
                    audio_array = librosa.resample(audio_data["array"], audio_data["sampling_rate"], sr)
                else:
                    audio_array = audio_data["array"]

                duration = len(audio_array) / sr
                total_duration += duration

                audio_filename = f"sample_{i}.wav"
                audio_path = os.path.join(self.audio_dir, audio_filename)
                sf.write(audio_path, audio_array, sr)

                metadata = {
                    "audio_filepath": audio_path,
                    "duration": duration,
                    "text": text,
                    "lang": "en",
                    "target_lang": "en",
                    "source_lang": "en",
                    "pnc": "False"
                }

                json.dump(metadata, fout)
                fout.write('\n')

        print(f"\nTotal duration: {np.round(total_duration / 3600, 2)} hours")
        print(f"Manifest created at: {self.manifest_path}")
