import os
import tarfile
import wget
import glob
import librosa
import soundfile as sf

class LibriLightDataHandler:
    def __init__(self, data_dir="datasets"):
        self.data_dir = data_dir
        self.librispeech_url = "https://dl.fbaipublicfiles.com/librilight/data/librispeech_finetuning.tgz"
        self.librispeech_tgz = os.path.join(self.data_dir, 'librispeech_finetuning.tgz')
        self.librilight_dir = os.path.join(self.data_dir, 'LibriLight')

        # Ensure the base data directory exists
        os.makedirs(self.data_dir, exist_ok=True)

    def download_dataset(self):
        if not os.path.exists(self.librispeech_tgz):
            print("Downloading LibriLight dataset...")
            wget.download(self.librispeech_url, self.data_dir, bar=None)
            print(f"\nDataset downloaded at: {self.librispeech_tgz}")
        else:
            print("Dataset archive already exists.")

    def extract_dataset(self):
        if not os.path.exists(self.librilight_dir):
            print("Extracting LibriLight dataset...")
            with tarfile.open(self.librispeech_tgz) as tar:
                tar.extractall(path=self.librilight_dir)
            print(f"Dataset extracted to: {self.librilight_dir}")
        else:
            print("Dataset already extracted.")

    def prepare_data(self):
        self.download_dataset()
        self.extract_dataset()
        print(f"LibriLight data is ready at: {self.librilight_dir}")

    def get_longform_audio_sample(self):
        libri_data_dir = self.librilight_dir
        audio_paths = glob.glob(os.path.join(
            libri_data_dir, 'librispeech_finetuning/1h/0/clean/3526/175658/3526-175658-*.flac'
        ))
        audio_paths.sort()  # sort by the utterance IDs

        if not audio_paths:
            print("No audio files found for longform sample.")
            return None

        write_path = os.path.join(
            libri_data_dir,
            'longform',
            '-'.join(os.path.basename(audio_paths[0]).split('-')[:2]) + '.wav'
        )
        os.makedirs(os.path.dirname(write_path), exist_ok=True)

        longform_audio_data = []
        for audio_path in audio_paths:
            data, sr = librosa.load(audio_path, sr=16000)
            longform_audio_data.extend(data)

        sf.write(write_path, longform_audio_data, sr)
        minutes, seconds = divmod(len(longform_audio_data) / sr, 60)
        print(f'{int(minutes)} min {int(seconds)} sec audio file saved at {write_path}')
        return write_path


if __name__ == "__main__":
    handler = LibriLightDataHandler()
    handler.prepare_data()
    longform_audio_path = handler.get_longform_audio_sample()

    # Optional: implement listen_to_audio if you're in an environment that supports audio playback
    # Example using IPython.display:
    # from IPython.display import Audio
    # if longform_audio_path:
    #     display(Audio(filename=longform_audio_path))
