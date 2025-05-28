import os
import json
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
            audio_path = sample["audio"]["path"]
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

# Split dataset into train (80%), validation (10%), test (10%)
split_dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
train_val_split = split_dataset["train"].train_test_split(test_size=0.1111, seed=42)  # 10% of 90% ≈ 10%

train_dataset = train_val_split["train"]
val_dataset = train_val_split["test"]
test_dataset = split_dataset["test"]

# Paths
train_manifest_path = os.path.join(target_data_dir, "train_manifest.json")
val_manifest_path = os.path.join(target_data_dir, "val_manifest.json")
test_manifest_path = os.path.join(target_data_dir, "test_manifest.json")

# Create manifests
convert_to_manifest(train_dataset, train_manifest_path)
convert_to_manifest(val_dataset, val_manifest_path)
convert_to_manifest(test_dataset, test_manifest_path)

print(f"✅ All manifests generated in: {target_data_dir}")
