from pydub import AudioSegment
from IPython.display import Audio, display
import torch
from nemo.collections.asr.models import EncDecMultiTaskModel


class CanaryTranscriber:
    def __init__(self, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading model on {self.device}...")
        # self.model = EncDecMultiTaskModel.from_pretrained(model_name, map_location=self.device)
        self.model = EncDecMultiTaskModel.restore_from("canary_results/nvidia_canary-1b-flash-finetune/checkpoints# cd nvidia_canary-1b-flash-finetune.nemo")

    def listen(self, audio_path, offset=0.0, duration=-1):
        """Play an audio segment."""
        audio = AudioSegment.from_file(audio_path)
        start_ms = int(offset * 1000)
        end_ms = int((offset + duration) * 1000) if duration != -1 else None
        segment = audio[start_ms:end_ms]
        audio = Audio(segment.export(format='wav').read())
        display(audio)

    def transcribe(self, audio_path, source_lang='en', target_lang='en', pnc=True):
        """Transcribe the audio (ASR mode)."""
        result = self.model.transcribe(
            audio=[audio_path],
            batch_size=1,
            source_lang=source_lang,
            target_lang=target_lang,
            pnc=str(pnc)
        )
        return result[0].text

    def translate(self, audio_path, source_lang='en', target_lang='es', pnc=True):
        """Translate speech to text from one language to another."""
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
    # Path to audio file
    audio_file = "datasets/LibriLight/librispeech_finetuning/1h/0/clean/3526/175658/3526-175658-0000.flac"

    # Instantiate the transcriber
    transcriber = CanaryTranscriber()

    # Play the audio
    print("\nPlaying audio...")
    transcriber.listen(audio_file)

    # ASR transcription with punctuation and capitalization
    print("\nTranscription with PnC:")
    print(transcriber.transcribe(audio_file, pnc=True))

    # ASR transcription without punctuation and capitalization
    print("\nTranscription without PnC:")
    print(transcriber.transcribe(audio_file, pnc=False))

    # Translation example (English → Spanish)
    print("\n\nSpeech to text translation from English to Spanish with punctuation and capitalization:")
    translated_text = transcriber.translate(
        audio_path=audio_file,
        source_lang='en',
        target_lang='es',
        pnc=True
    )
    print(f'  "{translated_text}"')

    # Replay audio
    print("\nReplaying audio...")
    transcriber.listen(audio_file)
