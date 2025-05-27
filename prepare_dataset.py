import os
import json
import librosa
import soundfile as sf
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import argparse


class MedicalASRDataHandler:
    def __init__(self, dataset_name, data_dir="datasets", split="train"):
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.split = split
        self.output_dir = os.path.join(data_dir, dataset_name.replace("/", "_"))
        self.audio_dir = os.path.join(self.output_dir, "audio")
        self.manifest_path = os.path.join(self.output_dir, f"{split}_manifest.json")
        os.makedirs(self.audio_dir, exist_ok=True)

    def download_dataset(self):
        print(f"[INFO] Loading dataset '{self.dataset_name}' (split='{self.split}') from Hugging Face...")
        dataset = load_dataset(self.dataset_name, split=self.split)
        print(f"[INFO] Loaded {len(dataset)} samples.")
        return dataset

    def build_manifest(self):
        dataset = self.download_dataset()
        total_duration = 0
        valid_samples = 0

        with open(self.manifest_path, 'w') as fout:
            for i, sample in tqdm(enumerate(dataset), total=len(dataset)):
                try:
                    audio_data = sample["audio"]
                    text = sample["transcription"].strip().lower()
                    if not text or text == "-":
                        continue

                    sr = 16000
                    original_sr = audio_data["sampling_rate"]
                    if original_sr != sr:
                        audio_array = librosa.resample(y=audio_data["array"], orig_sr=original_sr, target_sr=sr)
                    else:
                        audio_array = audio_data["array"]

                    duration = len(audio_array) / sr
                    if duration <= 0 or np.isnan(duration):
                        continue

                    audio_filename = f"{self.split}_sample_{i}.wav"
                    audio_path = os.path.join(self.audio_dir, audio_filename)
                    sf.write(audio_path, audio_array, sr)

                    metadata = {
                        "audio_filepath": audio_path,
                        "duration": duration,
                        "text": text
                    }

                    json.dump(metadata, fout)
                    fout.write('\n')
                    total_duration += duration
                    valid_samples += 1
                except Exception as e:
                    print(f"[WARNING] Skipping sample {i}: {e}")

        print(f"\n[✅] Manifest created: {self.manifest_path}")
        print(f"[📊] Total duration: {np.round(total_duration / 3600, 2)} hours")
        print(f"[📈] Valid samples: {valid_samples}/{len(dataset)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare NeMo-compatible manifest for ASR datasets.")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="The name of the dataset on HuggingFace Hub (e.g., 'jarvisx17/Medical-ASR-EN').")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split to use (e.g., 'train', 'validation', 'test')")
    parser.add_argument("--data_dir", type=str, default="datasets",
                        help="Directory to store processed data and audio files (default: 'datasets')")

    args = parser.parse_args()

    handler = MedicalASRDataHandler(
        dataset_name=args.dataset_name,
        data_dir=args.data_dir,
        split=args.split
    )
    handler.build_manifest()
