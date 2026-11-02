"""
Generate Pre-Cached Piper Character Audio WAV Files
===================================================

Generates high-fidelity WAV files for all Grade 1 characters (A-Z, 0-9,
math operators, and punctuation) using Piper Neural TTS (en_US-ryan-medium.onnx).
Applies 150ms leading digital silence (for MAX98357A I2S DAC clock lock)
and 200ms trailing digital silence (for ALSA DMA buffer drain).
Uses standard letters with period (A., B., C.) to ensure natural American prosody.
"""

import os
import sys
import wave
import paramiko

PI_HOST = os.getenv("PI_IP", "192.168.0.107")
PI_USER = os.getenv("PI_USER", "raspberrypi5")
PI_PASS = os.getenv("PI_PASS", "raspberrypi5")

CHARACTER_TEXT_MAP = {
    # Letters A-Z (natural letter names with period, no artificial phonetic respellings)
    "A": "A.",
    "B": "B.",
    "C": "C.",
    "D": "D.",
    "E": "E.",
    "F": "F.",
    "G": "G.",
    "H": "H.",
    "I": "I.",
    "J": "J.",
    "K": "K.",
    "L": "L.",
    "M": "M.",
    "N": "N.",
    "O": "O.",
    "P": "P.",
    "Q": "Q.",
    "R": "R.",
    "S": "S.",
    "T": "T.",
    "U": "U.",
    "V": "V.",
    "W": "W.",
    "X": "X.",
    "Y": "Y.",
    "Z": "Z.",
    # Digits 0-9
    "0": "zero.",
    "1": "one.",
    "2": "two.",
    "3": "three.",
    "4": "four.",
    "5": "five.",
    "6": "six.",
    "7": "seven.",
    "8": "eight.",
    "9": "nine.",
    # Math Operators
    "+": "plus.",
    "-": "minus.",
    "*": "times.",
    "/": "divided by.",
    "=": "equals.",
    # Punctuation
    ".": "period.",
    ",": "comma.",
    "?": "question mark.",
    "!": "exclamation mark.",
    "#": "number sign.",
}

CHARACTER_FILENAME_MAP = {
    "+": "char_plus.wav",
    "-": "char_minus.wav",
    "*": "char_times.wav",
    "/": "char_divide.wav",
    "=": "char_equals.wav",
    ".": "char_period.wav",
    ",": "char_comma.wav",
    "?": "char_question.wav",
    "!": "char_exclamation.wav",
    "#": "char_hash.wav",
}


def get_filename_for_char(char: str) -> str:
    if char in CHARACTER_FILENAME_MAP:
        return CHARACTER_FILENAME_MAP[char]
    return f"char_{char}.wav"


def generate_on_remote_pi():
    print(f"Connecting to Raspberry Pi 5 at {PI_HOST}...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(PI_HOST, username=PI_USER, password=PI_PASS, timeout=10)
    print("Connected!")

    remote_cache_dir = "/home/raspberrypi5/brailce/brailce/audio/characters"
    local_cache_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "brailce", "audio", "characters")
    )
    os.makedirs(local_cache_dir, exist_ok=True)

    # Clean existing cached files locally
    print("Wiping old local cached WAV files...")
    for f in os.listdir(local_cache_dir):
        if f.endswith(".wav"):
            try:
                os.remove(os.path.join(local_cache_dir, f))
            except Exception:
                pass

    # Clean existing cached files remotely
    print("Wiping old remote cached WAV files on Pi...")
    ssh.exec_command(f"rm -rf {remote_cache_dir}/*")
    ssh.exec_command(f"mkdir -p '{remote_cache_dir}'")

    # Build Python generator script to run on Pi
    remote_script = r'''
import os
import wave
import subprocess

PIPER_BIN = "/home/raspberrypi5/piper/piper/piper"
PIPER_MODEL = "/home/raspberrypi5/piper/en_US-ryan-medium.onnx"
OUT_DIR = "/home/raspberrypi5/brailce/brailce/audio/characters"

CHARACTER_MAP = {
'''
    for char, text in CHARACTER_TEXT_MAP.items():
        fname = get_filename_for_char(char)
        remote_script += f'    {repr(fname)}: {repr(text)},\n'

    remote_script += r'''
}

os.makedirs(OUT_DIR, exist_ok=True)

for fname, text in CHARACTER_MAP.items():
    out_path = os.path.join(OUT_DIR, fname)
    temp_wav = out_path + ".tmp.wav"
    
    # 1. Generate with Piper at normal deliberate speed (1.05)
    cmd = [
        PIPER_BIN,
        "--model", PIPER_MODEL,
        "--output_file", temp_wav,
        "--length-scale", "1.05",
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    p.stdin.write(text.encode("utf-8") + b"\n")
    p.stdin.close()
    p.wait()

    # 2. Prepend 150ms silence (I2S clock lock) and append 200ms silence (DMA buffer drain)
    try:
        with wave.open(temp_wav, "rb") as wf:
            params = wf.getparams()
            frames = wf.readframes(wf.getnframes())
            
        silence_pre_count = int(params.framerate * 0.15)
        silence_pre_bytes = b"\x00" * (silence_pre_count * params.nchannels * params.sampwidth)
        silence_post_count = int(params.framerate * 0.20)
        silence_post_bytes = b"\x00" * (silence_post_count * params.nchannels * params.sampwidth)
        
        with wave.open(out_path, "wb") as wf_out:
            wf_out.setparams(params)
            wf_out.writeframes(silence_pre_bytes + frames + silence_post_bytes)
            
        os.remove(temp_wav)
        print(f"Generated: {fname} -> '{text}'")
    except Exception as e:
        print(f"Error processing {fname}: {e}")
'''

    # Upload and execute the generator on the Pi
    sftp = ssh.open_sftp()
    with sftp.file("/home/raspberrypi5/gen_cache.py", "w") as f:
        f.write(remote_script)

    print("\n--- Generating Character Audio Cache on Pi using Piper Ryan (with Pre/Post Silence) ---")
    stdin, stdout, stderr = ssh.exec_command("python3 /home/raspberrypi5/gen_cache.py")
    out = stdout.read().decode("utf-8", errors="replace")
    print(out)
    err = stderr.read().decode("utf-8", errors="replace")
    if err:
        print("ERR:", err)

    # 3. Download the generated WAV files from Pi to local repository
    print("\n--- Synchronizing Cache to Local Repository ---")
    downloaded_count = 0
    for char in CHARACTER_TEXT_MAP:
        fname = get_filename_for_char(char)
        rem_file = f"{remote_cache_dir}/{fname}"
        loc_file = os.path.join(local_cache_dir, fname)
        try:
            sftp.get(rem_file, loc_file)
            downloaded_count += 1
        except Exception as exc:
            print(f"  Failed to download {fname}: {exc}")

    print(f"Successfully downloaded {downloaded_count}/{len(CHARACTER_TEXT_MAP)} character WAV files!")
    sftp.close()
    ssh.exec_command("rm -f /home/raspberrypi5/gen_cache.py")
    ssh.close()
    print("\nCharacter cache regeneration complete!")


if __name__ == "__main__":
    generate_on_remote_pi()
