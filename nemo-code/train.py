import os
import glob
import tqdm
import json
import librosa
import numpy as np
import subprocess
from omegaconf import OmegaConf
from nemo.collections.asr.models import EncDecMultiTaskModel
from nemo_helpers import wget_from_nemo  # Import the helper method

class ASRModelTrainer:
    def __init__(self, data_root, manifest_filename='train_manifest.json'):
        self.data_root = data_root
        self.manifest_path = os.path.join(data_root, 'LibriLight', manifest_filename)

    def build_manifest(self):
        transcript_list = glob.glob(
            os.path.join(self.data_root, 'LibriLight/librispeech_finetuning/1h/**/*.txt'),
            recursive=True
        )
        tot_duration = 0

        with open(self.manifest_path, 'w') as fout:
            pass

        for transcript_path in tqdm.tqdm(transcript_list):
            with open(transcript_path, 'r') as fin:
                wav_dir = os.path.dirname(transcript_path)
                with open(self.manifest_path, 'a') as fout:
                    for line in fin:
                        file_id = line.strip().split(' ')[0]
                        audio_path = os.path.join(wav_dir, f'{file_id}.flac')

                        transcript = ' '.join(line.strip().split(' ')[1:]).lower().strip()

                        duration = librosa.core.get_duration(path=audio_path)
                        tot_duration += duration

                        metadata = {
                            "audio_filepath": audio_path,
                            "duration": duration,
                            "text": transcript,
                            "target_lang": "en",
                            "source_lang": "en",
                            "pnc": "False"
                        }
                        json.dump(metadata, fout)
                        fout.write('\n')

        print(f"\n{np.round(tot_duration/3600)} hour audio data ready for training")
        print(f"Manifest created at {self.manifest_path}")

    def build_special_tokenizer(self):
        script_path = "scripts/speech_recognition/canary/build_canary_2_special_tokenizer.py"
        output_dir = "tokenizers/spl_tokens"

        wget_from_nemo(script_path)
        os.makedirs(output_dir, exist_ok=True)
        subprocess.run(["python", script_path, output_dir])
        print(f"Special tokenizer built at {output_dir}")

    def build_language_specific_tokenizer(self, lang='en', data='libri1h', vocab_size=1024):
        script_path = 'scripts/tokenizers/process_asr_text_tokenizer.py'
        wget_from_nemo(script_path)

        out_dir = f"tokenizers/{lang}_{data}_{vocab_size}"
        manifest_path = self.manifest_path
        train_text_path = os.path.join(self.data_root, 'LibriLight', 'train_text.lst')

        with open(manifest_path, "r") as f:
            data = [json.loads(line.strip()) for line in f.readlines()]

        with open(train_text_path, "w") as f:
            for line in data:
                f.write(f"{line['text']}\n")

        subprocess.run([
            "python", script_path,
            "--data_file", train_text_path,
            "--vocab_size", str(vocab_size),
            "--data_root", out_dir,
            "--tokenizer", "spe",
            "--spe_type", "bpe",
            "--spe_character_coverage", "1.0",
            "--no_lower_case",
            "--log"
        ])
        print(f"Language-specific tokenizer built at {out_dir}")

    def train_model(self):
        if 'canary_model' not in globals():
            canary_model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-180m-flash')

        base_model_cfg = OmegaConf.load("config/fast-conformer_aed.yaml")
        base_model_cfg['name'] = 'canary-180m-flash-finetune'
        base_model_cfg.pop("init_from_nemo_model", None)
        base_model_cfg['init_from_pretrained_model'] = "nvidia/canary-180m-flash"

        canary_model.save_tokenizers('./canary_flash_tokenizers/')

        for lang in os.listdir('canary_flash_tokenizers'):
            base_model_cfg['model']['tokenizer']['langs'][lang] = {
                'dir': os.path.join('canary_flash_tokenizers', lang),
                'type': 'bpe'
            }

        base_model_cfg['spl_tokens']['model_dir'] = os.path.join('canary_flash_tokenizers', "spl_tokens")
        base_model_cfg['model']['prompt_format'] = canary_model._cfg['prompt_format']
        base_model_cfg['model']['prompt_defaults'] = canary_model._cfg['prompt_defaults']
        base_model_cfg['model']['model_defaults'] = canary_model._cfg['model_defaults']
        base_model_cfg['model']['preprocessor'] = canary_model._cfg['preprocessor']
        base_model_cfg['model']['encoder'] = canary_model._cfg['encoder']
        base_model_cfg['model']['transf_decoder'] = canary_model._cfg['transf_decoder']
        base_model_cfg['model']['transf_encoder'] = canary_model._cfg['transf_encoder']

        cfg = OmegaConf.create(base_model_cfg)
        config_path = "config/canary-180m-flash-finetune.yaml"
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            OmegaConf.save(cfg, f)

        subprocess.run([
            "python", "scripts/speech_to_text_aed.py",
            "--config-path=../config",
            "--config-name=canary-180m-flash-finetune.yaml",
            f"name=canary-180m-flash-finetune",
            f"model.train_ds.manifest_filepath={self.manifest_path}",
            f"model.validation_ds.manifest_filepath={self.manifest_path}",
            f"model.test_ds.manifest_filepath={self.manifest_path}",
            "exp_manager.exp_dir=canary_results",
            "exp_manager.resume_ignore_no_checkpoint=true",
            "trainer.max_steps=10",
            "trainer.log_every_n_steps=1"
        ])
        print("Training launched for Canary model fine-tuning.")

# Example usage
if __name__ == '__main__':
    data_dir = "datasets"
    trainer = ASRModelTrainer(data_dir)
    trainer.build_manifest()
    trainer.build_special_tokenizer()
    trainer.build_language_specific_tokenizer()
    trainer.train_model()
