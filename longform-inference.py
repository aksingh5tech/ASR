import os
import json
import subprocess

# CONFIGURATION
USE_TIMESTAMPS = True  # Set to False if you don't want timestamps
MODEL_NAME = "nvidia/canary-180m-flash"
AUDIO_DIR = "datasets/LibriLight/longform/"
OUTPUT_FILE = "results/longform_output_with_timestamps.json" if USE_TIMESTAMPS else "results/longform_output.json"
CHUNK_LEN = 10.0 if USE_TIMESTAMPS else 40.0
BATCH_SIZE = 1
BEAM_SIZE = 1

# INFERENCE COMMAND
cmd = [
    "python", "scripts/speech_to_text_aed_chunked_infer.py",
    f'pretrained_name={MODEL_NAME}',
    f'audio_dir={AUDIO_DIR}',
    f'output_filename={OUTPUT_FILE}',
    f'chunk_len_in_secs={CHUNK_LEN}',
    f'batch_size={BATCH_SIZE}',
    f'decoding.beam.beam_size={BEAM_SIZE}',
    f'timestamps={str(USE_TIMESTAMPS)}'
]

# Run inference
print("Running inference...")
subprocess.run(cmd)

# Function placeholder: play audio (replace with actual implementation)
def listen_to_audio(audio_path, offset=0.0, duration=None):
    print(f"[Audio: {audio_path}, Offset: {offset:.2f}s, Duration: {duration if duration else 'full'}]")

# Read and print results
print("\nReading results...\n")
with open(OUTPUT_FILE, "r") as f:
    for line in f:
        pred_data = json.loads(line)
        listen_to_audio(pred_data["audio_filepath"])
        print("Transcript:\n", pred_data["pred_text"])

        if USE_TIMESTAMPS and "word" in pred_data:
            print('\nWord level timestamps:')
            for sample in pred_data['word']:
                word, start, end = sample['word'], sample['start'], sample['end']
                print(f'{word:<15}[{start:.2f}s, {end:.2f}s]')
                # To listen to each word segment, uncomment below:
                # listen_to_audio(pred_data["audio_filepath"], offset=start, duration=(end - start))
        print("\n" + "-"*40 + "\n")
