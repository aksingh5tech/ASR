import os
import tarfile
import wget

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

if __name__ == "__main__":
    handler = LibriLightDataHandler()
    handler.prepare_data()
