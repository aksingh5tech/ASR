from pydub import AudioSegment
from IPython.display import Audio, display
import torch
from nemo.collections.asr.models import EncDecMultiTaskModel


class CanaryTranscriber:
    def __init__(self, model_name='nvidia/canary-180m-flash', device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading model on {self.device}...")
        self.model = EncDecMultiTaskModel.from_pretrained(model_name, map_location=self.device)

    def listen(self, audio_path, offset=0.0, duration=-1):
        """Play an audio segment."""
        audio = AudioSegment.from_file(audio_path)
        start_ms = int(offset * 1000)
        end_ms = int((offset + duration) * 1000) if duration != -1 else None
        segment = audio[start_ms:end_ms]
        audio = Audio(segment.export(format='wav').read())
        display(audio)

    def transcribe(self, audio_path, source_lang='en', target_lang='en', pnc=True):
        """Transcribe the audio using the ASR model."""
        result = self.model.transcribe(
            audio=[audio_path],
            batch_size=1,
            source_lang=source_lang,
            target_lang=target_lang,
            pnc=str(pnc)
        )
        return result[0].text


# Example usage:
if __name__ == "__main__":
    audio_file = "datasets/LibriLight/librispeech_finetuning/1h/0/clean/3526/175658/3526-175658-0000.flac"

    transcriber = CanaryTranscriber()

    print("\nPlaying audio...")
    transcriber.listen(audio_file)

    print("\nTranscription with PnC:")
    print(transcriber.transcribe(audio_file, pnc=True))

    print("\nTranscription without PnC:")
    print(transcriber.transcribe(audio_file, pnc=False))
