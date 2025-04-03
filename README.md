# 5tech-ASR

A lightweight Automatic Speech Recognition (ASR) system designed for fine-tuning and inference using custom datasets.

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/us-inc/5tech-ASR.git
cd 5tech-ASR
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

## 🧠 Training

Start the model training process:

```bash
python train.py
```

This script will initiate fine-tuning of the ASR model using the processed dataset.

## 🔍 Inference

Run inference on new audio files:

```bash
python inference.py
```

This will output the transcriptions for the input audio using the trained ASR model.

## 📁 Project Structure

```
5tech-ASR/
├── dataset.py         # Prepares and processes the dataset
├── train.py           # Trains the ASR model
├── inference.py       # Runs inference using the trained model
├── utils/             # Utility functions (if available)
└── models/            # Model definitions (if available)
```

## 🛠 Requirements

Make sure to install required dependencies. You can create a `requirements.txt` file or install manually:

```bash
pip install -r requirements.txt
```

## 📬 Contact

For questions or contributions, feel free to open an issue or pull request.
