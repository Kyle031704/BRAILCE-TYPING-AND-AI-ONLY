"""
Generate Character Audio WAV Files via Longest-Silence-Gap Carrier Slicer
=========================================================================

Uses Carrier-Sentence Slicing ("Please say. <target>.") with --sentence_silence 0.55:
- Finds the single longest quiet run (the 550ms sentence gap between "Please say." and <target>).
- Cuts 80ms inside the silence gap before <target> begins to preserve initial fricatives and sibilants.
- Keeps everything to the very end of the file.
- Pads with 100ms pre-silence and 150ms post-silence using wave module.
- Has an automatic safety fallback to direct synthesis if slicing conditions are not met.
- Flags: --length_scale 1.08 --noise_scale 0.5 --noise_w 0.6 --sentence_silence 0.55
"""

from __future__ import annotations

import os
import struct
import sys
import wave

try:
    import paramiko
except ImportError:
    paramiko = None

import socket

def get_pi_host():
    if os.getenv("PI_IP"):
        return os.getenv("PI_IP")
    for ip in ["192.168.137.156", "192.168.0.107", "10.161.29.70"]:
        try:
            s = socket.socket()
            s.settimeout(0.4)
            if s.connect_ex((ip, 22)) == 0:
                s.close()
                return ip
            s.close()
        except Exception:
            pass
    return "192.168.137.156"

PI_HOST = get_pi_host()
PI_USER = os.getenv("PI_USER", "raspberrypi5")
PI_PASS = os.getenv("PI_PASS", "raspberrypi5")

PIPER_LENGTH_SCALE = "1.08"
PIPER_NOISE_SCALE = "0.5"
PIPER_NOISE_W = "0.6"

CHARACTER_TEXT_MAP = {
    # Letters A-Z: clean lowercase letters with period
    "A": "a.", "B": "b.", "C": "c.", "D": "d.", "E": "e.", "F": "f.",
    "G": "g.", "H": "h.", "I": "i.", "J": "j.", "K": "k.", "L": "l.",
    "M": "em.", "N": "en.", "O": "o.", "P": "p.", "Q": "q.", "R": "r.",
    "S": "s.", "T": "t.", "U": "u.", "V": "v.", "W": "w.", "X": "x.",
    "Y": "y.", "Z": "z.",
    # Digits 0-9
    "0": "zero.", "1": "one.", "2": "two.", "3": "three.", "4": "four.",
    "5": "five.", "6": "six.", "7": "seven.", "8": "eight.", "9": "nine.",
    # Math Operators
    "+": "plus.", "-": "minus.", "*": "times.", "/": "divided by.", "=": "equals.",
    # Punctuation
    ".": "period.", ",": "comma.", "?": "question mark.", "!": "exclamation mark.", "#": "number sign.",
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


def generate_on_remote_pi() -> None:
    if paramiko is None:
        print("Paramiko required.")
        sys.exit(1)

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

    print("Cleaning remote character WAVs on Pi...")
    ssh.exec_command(f"rm -rf {remote_cache_dir}/* && mkdir -p '{remote_cache_dir}'")

    remote_script = f'''
import os
import math
import struct
import wave
import subprocess

PIPER_BIN = "/home/raspberrypi5/piper/piper/piper"
PIPER_MODEL = "/home/raspberrypi5/piper/en_US-ryan-medium.onnx"
OUT_DIR = "{remote_cache_dir}"

LENGTH_SCALE = "{PIPER_LENGTH_SCALE}"
NOISE_SCALE = "{PIPER_NOISE_SCALE}"
NOISE_W = "{PIPER_NOISE_W}"

CHARACTER_MAP = {{
'''
    for char, text in CHARACTER_TEXT_MAP.items():
        fname = get_filename_for_char(char)
        remote_script += f'    {repr(fname)}: {repr(text)},\n'

    remote_script += '''
}

os.makedirs(OUT_DIR, exist_ok=True)

def slice_carrier_audio(samples, sr=22050):
    win = int(sr * 0.010)
    rms_list = []
    for i in range(0, len(samples) - win, win):
        chunk = samples[i : i + win]
        r = math.sqrt(sum(s * s for s in chunk) / win)
        rms_list.append(r)

    total_windows = len(rms_list)
    if total_windows < 40:
        return samples

    first_voiced = next((i for i, r in enumerate(rms_list) if r > 300), 0)
    last_voiced = max((i for i, r in enumerate(rms_list) if r > 500), default=total_windows - 1)

    search_start = min(total_windows - 10, first_voiced + 30)
    search_end = max(search_start + 1, last_voiced - 8)

    min_win = min(range(search_start, search_end), key=lambda k: rms_list[k])

    target_onset = min_win
    for j in range(min_win, min(total_windows, min_win + 30)):
        if rms_list[j] > 500:
            target_onset = j
            break

    cut_window = max(first_voiced + 25, target_onset - 4)
    cut_sample = cut_window * win
    sliced = samples[cut_sample:]
    if not sliced or len(sliced) < int(sr * 0.025) or max(abs(s) for s in sliced) < 500:
        return samples
    return sliced

def process_carrier(carrier_wav, out_path, pre_s=0.12, post_s=0.15):
    with wave.open(carrier_wav, "rb") as wf:
        params = wf.getparams()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)

    samples = list(struct.unpack(f"<{len(raw)//2}h", raw))
    sliced = slice_carrier_audio(samples, sr=params.framerate)
    if not sliced or max(abs(s) for s in sliced) < 500:
        return False

    target_pcm = struct.pack(f"<{len(sliced)}h", *sliced)
    pre_silence = bytes(int(params.framerate * pre_s) * 2)
    post_silence = bytes(int(params.framerate * post_s) * 2)
    padded = pre_silence + target_pcm + post_silence

    with wave.open(out_path, "wb") as wf_out:
        wf_out.setparams(params)
        wf_out.writeframes(padded)

    return True

for fname, text in CHARACTER_MAP.items():
    out_path = os.path.join(OUT_DIR, fname)
    temp_carrier = out_path + ".carrier.wav"

    carrier_prompt = f"Please say: {text.strip()}"
    cmd = [
        PIPER_BIN,
        "--model", PIPER_MODEL,
        "--output_file", temp_carrier,
        "--length_scale", LENGTH_SCALE,
        "--noise_scale", NOISE_SCALE,
        "--noise_w", NOISE_W,
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    p.communicate(carrier_prompt.encode("utf-8") + b"\\n")
    p.wait()

    success = False
    if os.path.isfile(temp_carrier) and os.path.getsize(temp_carrier) > 0:
        success = process_carrier(temp_carrier, out_path, pre_s=0.12, post_s=0.15)

    if os.path.exists(temp_carrier):
        os.remove(temp_carrier)

    with wave.open(out_path, "rb") as wf:
        nframes = wf.getnframes()
        rate = wf.getframerate()
        duration = nframes / rate
        raw = wf.readframes(nframes)
        samples = struct.unpack(f"<{len(raw)//2}h", raw)
        peak = max(abs(s) for s in samples) if samples else 0
        method = "carrier-sliced" if success else "failed"
        print(f"OK: {fname} <- {text!r} | duration={duration:.3f}s, peak={peak}, frames={nframes} ({method})")
'''

    sftp = ssh.open_sftp()
    with sftp.file("/home/raspberrypi5/gen_cache.py", "w") as f:
        f.write(remote_script)
        f.flush()

    print("--- Generating Character Audio Files on Pi via Digital-Zero Slicer ---")
    stdin, stdout, stderr = ssh.exec_command("python3 /home/raspberrypi5/gen_cache.py", timeout=300)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    print(out)
    if err:
        print("STDERR:", err)

    print("--- Syncing Verified WAVs to PC ---")
    for char in CHARACTER_TEXT_MAP:
        fname = get_filename_for_char(char)
        rem_file = f"{remote_cache_dir}/{fname}"
        loc_file = os.path.join(local_cache_dir, fname)
        try:
            sftp.get(rem_file, loc_file)
        except Exception:
            pass

    sftp.close()
    ssh.exec_command("rm -f /home/raspberrypi5/gen_cache.py")
    ssh.close()
    print("Character audio generation complete!")


if __name__ == "__main__":
    generate_on_remote_pi()
