# 5tech-ASR

A lightweight Automatic Speech Recognition (ASR) system designed for fine-tuning and inference using custom datasets.


## 🛠 VENV SETUP

```bash
sudo apt update
sudo apt install python3.10 python3.10-venv -y

python -m venv asr_venv
source asr_venv/bin/activate
```

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/us-inc/5tech-ASR.git
cd 5tech-ASR

pip install .

```

## 🗂 Dataset Preparation

Prepare your dataset for training:

### 📄 Dataset Format

Your dataset **must** include the following two mandatory columns:

- `audio`: Path to the audio file or the audio data itself.
- `transcription`: The corresponding text transcription for the audio.

Make sure all audio files are accessible and properly formatted before running the script.

#### Example format (CSV or JSON):

| audio               | transcription                      |
|---------------------|------------------------------------|
| audio/001.wav       | Hello, how can I help you today?   |
| audio/002.wav       | Please take a deep breath.         |

> ⚠️ Note: If using Hugging Face datasets, ensure the dataset you reference includes these two fields.

This step will preprocess and organize the audio-text pairs for training.

```bash
python prepare_dataset.py --dataset_name jarvisx17/Medical-ASR-EN --split train --data_dir ./datasets
```

#### You'll get output like below:
```bash
Loading dataset 'jarvisx17/Medical-ASR-EN' split 'train'...
Loaded 6661 samples.
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6661/6661 [01:05<00:00, 101.62it/s]

Total duration: 8.47 hours
Manifest created at: ./datasets/jarvisx17_Medical-ASR-EN/train_manifest.json

```

## 🧠 Training
To start the model training process, follow the steps below.
1. Make the script executable (only needed once):
```bash
chmod +x run_train.sh
```

2. Run the training script:

```bash
bash run_train.sh nvidia/parakeet-tdt-0.6b-v2 datasets/jarvisx17_Medical-ASR-EN/train_manifest.json
```
###  bash run_train.sh <model_name> <manifest_path>
<model_name>: The name of the pretrained model you'd like to fine-tune. For example: nvidia/canary-1b-flash

<manifest_path>: Path to your training manifest JSON file created above in prepare dataset. For example: datasets/jarvisx17_Medical-ASR-EN/train_manifest.json



## 🔍 Inference

Run inference on new audio files:

```bash
python inference.py \
  --checkpoint canary_results/nvidia_canary-1b-flash-finetune/checkpoints/nvidia_canary-1b-flash-finetune.nemo \
  --audio datasets/jarvisx17_Medical-ASR-EN/audio/sample_990.wav

```

This will output the transcriptions for the input audio using the trained ASR model.

```
## 📬 Contact

For questions or contributions, feel free to open an issue or pull request.
