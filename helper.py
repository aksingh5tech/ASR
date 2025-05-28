import os
import urllib.request

BRANCH = "main"  # You can change this to tag/version if needed

def wget_from_nemo(nemo_script_path, local_dir="scripts"):
    """
    Download a file from the NeMo GitHub repository if it does not already exist.

    :param nemo_script_path: The path to the file within the NeMo repo (e.g. examples/asr/conf/fast_conformer/...).
    :param local_dir: The local directory to save the file into.
    """
    os.makedirs(local_dir, exist_ok=True)

    script_name = os.path.basename(nemo_script_path)
    local_path = os.path.join(local_dir, script_name)
    url = f"https://raw.githubusercontent.com/NVIDIA/NeMo/{BRANCH}/{nemo_script_path}"

    if os.path.exists(local_path):
        print(f"[✔️] File already exists: {local_path}")
        return

    print(f"[⬇️ ] Downloading {script_name} from NeMo GitHub...")
    try:
        urllib.request.urlretrieve(url, local_path)
        print(f"[✅] Saved to {local_path}")
    except Exception as e:
        print(f"[❌] Failed to download {script_name} from NeMo repo: {e}")
        print(f"URL tried: {url}")

if __name__ == '__main__':
    wget_from_nemo("examples/asr/asr_ctc/speech_to_text_ctc.py")