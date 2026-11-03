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

PI_HOST = os.getenv("PI_IP", "192.168.0.107")
PI_USER = os.getenv("PI_USER", "raspberrypi5")
PI_PASS = os.getenv("PI_PASS", "raspberrypi5")

PIPER_LENGTH_SCALE = "1.08"
PIPER_NOISE_SCALE = "0.5"
PIPER_NOISE_W = "0.6"
PIPER_SENTENCE_SILENCE = "0.55"

CHARACTER_TEXT_MAP = {
    # Letters A-Z: lowercase "a." through "z.", EXCEPT S -> "ess.", H -> "aitch.", X -> "eks."
    "A": "a.", "B": "b.", "C": "c.", "D": "d.", "E": "e.", "F": "f.",
    "G": "g.", "H": "aitch.", "I": "i.", "J": "j.", "K": "k.", "L": "l.",
    "M": "m.", "N": "n.", "O": "o.", "P": "p.", "Q": "q.", "R": "r.",
    "S": "ess.", "T": "t.", "U": "u.", "V": "v.", "W": "w.", "X": "eks.",
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
import struct
import wave
import subprocess

PIPER_BIN = "/home/raspberrypi5/piper/piper/piper"
PIPER_MODEL = "/home/raspberrypi5/piper/en_US-ryan-medium.onnx"
OUT_DIR = "{remote_cache_dir}"

LENGTH_SCALE = "{PIPER_LENGTH_SCALE}"
NOISE_SCALE = "{PIPER_NOISE_SCALE}"
NOISE_W = "{PIPER_NOISE_W}"
SENTENCE_SILENCE = "{PIPER_SENTENCE_SILENCE}"

CHARACTER_MAP = {{
'''
    for char, text in CHARACTER_TEXT_MAP.items():
        fname = get_filename_for_char(char)
        remote_script += f'    {repr(fname)}: {repr(text)},\n'

    remote_script += '''
}

os.makedirs(OUT_DIR, exist_ok=True)

def slice_longest_gap(carrier_wav, out_path, pre_s=0.10, post_s=0.15):
    """
    Longest-Silence-Gap Carrier Slicer:
    - Finds the 550ms silence gap between "Please say." and <target>.
    - Cuts 80ms inside the silence gap before <target> begins.
    - Slices to the very end of the file.
    - Prepend 100ms and append 150ms of silence using wave module.
    """
    try:
        with wave.open(carrier_wav, "rb") as wf:
            params = wf.getparams()
            nframes = wf.getnframes()
            raw = wf.readframes(nframes)

        if params.nchannels != 1 or params.sampwidth != 2 or nframes == 0:
            return False

        num_samples = len(raw) // 2
        samples = struct.unpack(f"<{num_samples}h", raw)

        win = int(params.framerate * 0.010)
        if win <= 0:
            win = 220

        energies = [max(abs(s) for s in samples[i : i + win]) for i in range(0, len(samples) - win, win)]
        num_win = len(energies)

        # 1. Find first voiced window (>1500) and last voiced window (>800)
        first_voiced = -1
        for i in range(num_win):
            if energies[i] > 1500:
                first_voiced = i
                break

        last_voiced = -1
        for i in range(num_win - 1, -1, -1):
            if energies[i] > 800:
                last_voiced = i
                break

        if first_voiced == -1 or last_voiced == -1 or (first_voiced + 15) >= (last_voiced - 10):
            return False

        # 2. Search between first_voiced + 15 and last_voiced - 10 for contiguous quiet windows (< 400)
        search_start = first_voiced + 15
        search_end = last_voiced - 10

        best_start = -1
        best_end = -1
        max_len = 0

        cur_start = -1
        for i in range(search_start, search_end):
            if energies[i] < 400:
                if cur_start == -1:
                    cur_start = i
            else:
                if cur_start != -1:
                    cur_len = i - cur_start
                    if cur_len > max_len:
                        max_len = cur_len
                        best_start = cur_start
                        best_end = i
                    cur_start = -1

        if cur_start != -1:
            cur_len = search_end - cur_start
            if cur_len > max_len:
                max_len = cur_len
                best_start = cur_start
                best_end = search_end

        # Safety check: gap must be at least 12 windows (120ms)
        if max_len < 12 or best_start == -1:
            return False

        # Cut 80ms inside the silence gap before <target> begins
        cut_window = max(best_start, best_end - 8)
        cut_sample = cut_window * win
        target_samples = samples[cut_sample:]

        # Safety check: target must have real voice amplitude (> 1500)
        if not target_samples or max(abs(s) for s in target_samples) < 1500:
            return False

        target_pcm = struct.pack(f"<{len(target_samples)}h", *target_samples)
        pre_silence = b"\\x00\\x00" * int(params.framerate * pre_s)
        post_silence = b"\\x00\\x00" * int(params.framerate * post_s)
        padded = pre_silence + target_pcm + post_silence

        with wave.open(out_path, "wb") as wf_out:
            wf_out.setparams(params)
            wf_out.writeframes(padded)

        return True
    except Exception as exc:
        print(f"Slicing exception: {exc}")
        return False

def direct_fallback(text, out_path, pre_s=0.10, post_s=0.15):
    temp_dir = out_path + ".fallback.wav"
    cmd = [
        PIPER_BIN,
        "--model", PIPER_MODEL,
        "--output_file", temp_dir,
        "--length_scale", LENGTH_SCALE,
        "--noise_scale", NOISE_SCALE,
        "--noise_w", NOISE_W,
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    p.communicate(text.encode("utf-8") + b"\\n")
    p.wait()

    with wave.open(temp_dir, "rb") as wf_in:
        params = wf_in.getparams()
        frames = wf_in.readframes(wf_in.getnframes())

    pre_silence = b"\\x00\\x00" * int(params.framerate * pre_s)
    post_silence = b"\\x00\\x00" * int(params.framerate * post_s)
    padded = pre_silence + frames + post_silence

    with wave.open(out_path, "wb") as wf_out:
        wf_out.setparams(params)
        wf_out.writeframes(padded)

    if os.path.exists(temp_dir):
        os.remove(temp_dir)

for fname, text in CHARACTER_MAP.items():
    out_path = os.path.join(OUT_DIR, fname)
    temp_carrier = out_path + ".carrier.wav"

    # Synthesize with carrier prompt: "Please say. <Target>"
    carrier_prompt = f"Please say. {text.strip().capitalize()}"
    cmd = [
        PIPER_BIN,
        "--model", PIPER_MODEL,
        "--output_file", temp_carrier,
        "--length_scale", LENGTH_SCALE,
        "--noise_scale", NOISE_SCALE,
        "--noise_w", NOISE_W,
        "--sentence_silence", SENTENCE_SILENCE,
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    p.communicate(carrier_prompt.encode("utf-8") + b"\\n")
    p.wait()

    success = False
    if os.path.isfile(temp_carrier) and os.path.getsize(temp_carrier) > 0:
        success = slice_longest_gap(temp_carrier, out_path, pre_s=0.10, post_s=0.15)

    if os.path.exists(temp_carrier):
        os.remove(temp_carrier)

    if not success:
        direct_fallback(text, out_path, pre_s=0.10, post_s=0.15)

    with wave.open(out_path, "rb") as wf:
        nframes = wf.getnframes()
        rate = wf.getframerate()
        duration = nframes / rate
        raw = wf.readframes(nframes)
        samples = struct.unpack(f"<{len(raw)//2}h", raw)
        peak = max(abs(s) for s in samples) if samples else 0
        method = "carrier-sliced" if success else "direct-fallback"
        print(f"OK: {fname} <- {text!r} | duration={duration:.3f}s, peak={peak}, frames={nframes} ({method})")
'''

    sftp = ssh.open_sftp()
    with sftp.file("/home/raspberrypi5/gen_cache.py", "w") as f:
        f.write(remote_script)

    print("--- Generating Character Audio Files on Pi via Longest-Gap Slicer ---")
    stdin, stdout, stderr = ssh.exec_command("python3 /home/raspberrypi5/gen_cache.py", timeout=300)
    out = stdout.read().decode("utf-8", errors="replace")
    print(out)

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
