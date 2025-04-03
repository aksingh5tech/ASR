# import libraries

import glob
import json
import librosa
import numpy as np
from omegaconf import OmegaConf, open_dict
import os
import soundfile as sf
import subprocess
import tarfile
import tqdm
import wget

import torch


def download_and_prepare_librilight_data(data_dir="datasets"):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    libri_data_dir = os.path.join(data_dir, 'LibriLight')
    libri_tgz_file = f'{data_dir}/librispeech_finetuning.tgz'

    if not os.path.exists(libri_tgz_file):
        url = "https://dl.fbaipublicfiles.com/librilight/data/librispeech_finetuning.tgz"
        libri_path = wget.download(url, data_dir, bar=None)
        print(f"Dataset downloaded at: {libri_path}")

    if not os.path.exists(libri_data_dir):
        tar = tarfile.open(libri_tgz_file)
        tar.extractall(path=libri_data_dir)

    print(f'LibriLight data is ready at {libri_data_dir}')
