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

```bash
python dataset.py
```

Make sure your dataset is structured properly before running the script. This step will preprocess and organize the audio-text pairs for training.

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
