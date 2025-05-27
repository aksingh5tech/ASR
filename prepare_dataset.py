from datasets import load_dataset
import json
import os
from tqdm import tqdm

def convert_to_manifest(split_dataset, output_path, split_name):
    manifest = []
    for sample in tqdm(split_dataset):
        manifest.append({
            "audio_filepath": sample["audio"]["path"],
            "duration": sample["audio"]["duration"],
            "text": sample["transcription"]
        })

    with open(os.path.join(output_path, f"{split_name}.json"), "w") as f:
        for item in manifest:
            f.write(json.dumps(item) + "\n")

def main():
    output_dir = "./data"
    os.makedirs(output_dir, exist_ok=True)

    dataset = load_dataset("jarvisx17/Medical-ASR-EN")
    train_val = dataset["train"].train_test_split(test_size=0.1)

    convert_to_manifest(train_val["train"], output_dir, "train")
    convert_to_manifest(train_val["test"], output_dir, "validation")

if __name__ == "__main__":
    main()
