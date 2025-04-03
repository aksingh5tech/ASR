import os

BRANCH = 'main'

def wget_from_nemo(nemo_script_path, local_dir="scripts"):
    os.makedirs(local_dir, exist_ok=True)

    # Construct the URL and local filename
    script_url = f"https://raw.githubusercontent.com/NVIDIA/NeMo/{BRANCH}/{nemo_script_path}"
    script_name = os.path.basename(nemo_script_path)
    local_script_path = os.path.join(local_dir, script_name)

    # Download only if the file doesn't already exist
    if not os.path.exists(local_script_path):
        print(f"Downloading {script_name} to {local_dir}...")
        os.system(f"wget -O {local_script_path} {script_url}")
    else:
        print(f"{script_name} already exists in {local_dir}")

if __name__ == '__main__':

    # Example usage
    wget_from_nemo("examples/asr/asr_chunked_inference/aed/speech_to_text_aed_chunked_infer.py", local_dir="scripts")
