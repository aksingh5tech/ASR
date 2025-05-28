import os
import json
import librosa
from datasets import load_dataset
from tqdm import tqdm

# Define output directory
DATA_DIR = os.environ.get("DATA_DIR", "./data")
target_data_dir = os.path.join(DATA_DIR, "medical_asr_converted")
os.makedirs(target_data_dir, exist_ok=True)

def convert_to_manifest(dataset_split, manifest_path):
    """Convert dataset split to NeMo-compatible manifest."""
    with open(manifest_path, 'w') as fout:
        for sample in tqdm(dataset_split, desc=f"Creating manifest at {manifest_path}"):
            audio_path = sample["audio"]["path"]  # directly a file path
            transcript = sample["transcription"]
            duration = len(sample["audio"]["array"]) / sample["audio"]["sampling_rate"]

            metadata = {
                "audio_filepath": audio_path,
                "duration": duration,
                "text": transcript
            }

            json.dump(metadata, fout)
            fout.write("\n")

# Load dataset
dataset = load_dataset("jarvisx17/Medical-ASR-EN")

# Use the full train split
train_dataset = dataset["train"]
train_manifest_path = os.path.join(target_data_dir, "train_manifest.json")

# Generate manifest
convert_to_manifest(train_dataset, train_manifest_path)

print(f"✅ Manifest generated at: {train_manifest_path}")
