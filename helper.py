import os
import urllib.request

BRANCH = 'main'

def wget_from_nemo(nemo_script_path, local_dir=None):
    """
    Downloads a file from NeMo GitHub repo to a local directory, preserving subdirectory structure.
    If local_dir is None, downloads to the same relative path.
    """
    # Construct URL
    script_url = f"https://raw.githubusercontent.com/NVIDIA/NeMo/{BRANCH}/{nemo_script_path}"
    script_name = os.path.basename(nemo_script_path)

    # Set local path (preserving directory structure)
    if local_dir:
        local_script_path = os.path.join(local_dir, script_name)
    else:
        local_script_path = nemo_script_path

    local_dirname = os.path.dirname(local_script_path)
    os.makedirs(local_dirname, exist_ok=True)

    # Download only if the file doesn't already exist
    if not os.path.exists(local_script_path):
        try:
            print(f"Downloading {script_url} to {local_script_path}...")
            urllib.request.urlretrieve(script_url, local_script_path)
            print(f"✅ Downloaded: {local_script_path}")
        except Exception as e:
            print(f"❌ Failed to download {script_url}\nError: {e}")
    else:
        print(f"✔️ Already exists: {local_script_path}")

if __name__ == '__main__':
    # Example: download and preserve original path
    wget_from_nemo("examples/asr/speech_to_text_finetune.py")
