import argparse
from pydub import AudioSegment
from IPython.display import Audio, display
import torch
from nemo.collections.asr.models import EncDecMultiTaskModel


class CanaryTranscriber:
    def __init__(self, checkpoint_path, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading model on {self.device} from checkpoint: {checkpoint_path}...")
        self.model = EncDecMultiTaskModel.restore_from(checkpoint_path, map_location=self.device)

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


def main():
    parser = argparse.ArgumentParser(description="Canary Transcriber")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the .nemo checkpoint file')
    parser.add_argument('--audio', type=str, required=True, help='Path to the audio file')
    args = parser.parse_args()

    # Instantiate the transcriber
    transcriber = CanaryTranscriber(checkpoint_path=args.checkpoint)

    # Play the audio
    print("\nPlaying audio...")
    transcriber.listen(args.audio)

    # ASR transcription with punctuation and capitalization
    print("\nTranscription with PnC:")
    print(transcriber.transcribe(args.audio, pnc=True))

    # ASR transcription without punctuation and capitalization
    print("\nTranscription without PnC:")
    print(transcriber.transcribe(args.audio, pnc=False))

    # Translation example (English → Spanish)
    print("\n\nSpeech to text translation from English to Spanish with punctuation and capitalization:")
    translated_text = transcriber.translate(
        audio_path=args.audio,
        source_lang='en',
        target_lang='es',
        pnc=True
    )
    print(f'  "{translated_text}"')

    # Replay audio
    print("\nReplaying audio...")
    transcriber.listen(args.audio)


if __name__ == "__main__":
    main()
