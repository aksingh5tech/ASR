from datasets import load_dataset
import json
import os
import argparse
from tqdm import tqdm

def convert_to_manifest(dataset, output_path, split_name):
    manifest = []
    for sample in tqdm(dataset):
        manifest.append({
            "audio_filepath": sample["audio"]["path"],
            "duration": sample["audio"]["duration"],
            "text": sample["transcription"]
        })
    with open(os.path.join(output_path, f"{split_name}.json"), "w") as f:
        for m in manifest:
            f.write(json.dumps(m) + "\n")

def main(output_path):
    os.makedirs(output_path, exist_ok=True)
    dataset = load_dataset("arvisx17/Medical-ASR-EN")
    for split in dataset.keys():
        convert_to_manifest(dataset[split], output_path, split)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    main(args.output_path)
