"""
Regenerate char_M.wav, char_N.wav, and switched_en.wav on Raspberry Pi 5
========================================================================

1. Synthesizes 'char_M.wav' and 'char_N.wav' using Piper (en_US-ryan-medium.onnx)
   and Carrier-Sentence Slicing ("Please say: em." and "Please say: en.")
   with 25ms minimum target duration check.
2. Synthesizes 'switched_en.wav' ("English language.") using Piper (en_US-ryan-medium.onnx)
   with American English accent and 120ms/150ms silence padding.
3. Verifies durations to ensure "Please say" is 100% sliced away.
4. Downloads the regenerated files back to local workspace.
"""

import math
import os
import struct
import sys
import wave
import paramiko

PI_HOST = "192.168.137.156"
PI_USER = "raspberrypi5"
PI_PASS = "raspberrypi5"

LOCAL_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

REMOTE_SCRIPT = '''
import os
import math
import struct
import wave
import subprocess

PIPER_BIN = "/home/raspberrypi5/piper/piper/piper"
PIPER_MODEL = "/home/raspberrypi5/piper/en_US-ryan-medium.onnx"

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
    if not sliced or len(sliced) < int(params.framerate * 0.025) or max(abs(s) for s in sliced) < 500:
        return False

    target_pcm = struct.pack(f"<{len(sliced)}h", *sliced)
    pre_silence = b"\\x00\\x00" * int(params.framerate * pre_s)
    post_silence = b"\\x00\\x00" * int(params.framerate * post_s)
    padded = pre_silence + target_pcm + post_silence

    with wave.open(out_path, "wb") as wf_out:
        wf_out.setparams(params)
        wf_out.writeframes(padded)
    return True

# 1. Regenerate char_M.wav and char_N.wav
char_dir = "/home/raspberrypi5/brailce/brailce/audio/characters"
os.makedirs(char_dir, exist_ok=True)

targets = {
    "char_M.wav": "em.",
    "char_N.wav": "en.",
}

for fname, word in targets.items():
    raw_tmp = f"/tmp/raw_carrier_{fname}"
    out_file = os.path.join(char_dir, fname)
    carrier_prompt = f"Please say: {word}"
    
    cmd = [
        PIPER_BIN,
        "--model", PIPER_MODEL,
        "--output_file", raw_tmp,
        "--length_scale", "1.08",
        "--noise_scale", "0.5",
        "--noise_w", "0.6",
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    p.communicate(carrier_prompt.encode("utf-8") + b"\\n")
    
    ok = process_carrier(raw_tmp, out_file, pre_s=0.12, post_s=0.15)
    with wave.open(out_file, "rb") as wf:
        dur = wf.getnframes() / wf.getframerate()
    print(f"Generated {fname}: sliced={ok}, duration={dur:.3f}s")
    if os.path.exists(raw_tmp):
        os.remove(raw_tmp)

# 2. Regenerate switched_en.wav ("English language.") in Ryan American Voice
fil_dir = "/home/raspberrypi5/brailce/brailce/audio/filipino"
os.makedirs(fil_dir, exist_ok=True)
switched_en_path = os.path.join(fil_dir, "switched_en.wav")
raw_en_tmp = "/tmp/raw_switched_en.wav"

cmd_en = [
    PIPER_BIN,
    "--model", PIPER_MODEL,
    "--output_file", raw_en_tmp,
    "--length_scale", "1.05",
    "--noise_scale", "0.5",
    "--noise_w", "0.6",
]
p = subprocess.Popen(cmd_en, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
p.communicate(b"English language.\\n")

with wave.open(raw_en_tmp, "rb") as wf:
    params = wf.getparams()
    frames = wf.readframes(wf.getnframes())

pre_silence = b"\\x00\\x00" * int(params.framerate * 0.120)
post_silence = b"\\x00\\x00" * int(params.framerate * 0.150)
padded = pre_silence + frames + post_silence

with wave.open(switched_en_path, "wb") as wf_out:
    wf_out.setparams(params)
    wf_out.writeframes(padded)

dur_en = len(padded) / (params.framerate * 2)
print(f"Generated switched_en.wav: duration={dur_en:.3f}s")
if os.path.exists(raw_en_tmp):
    os.remove(raw_en_tmp)
'''

def main():
    print(f"Connecting to Raspberry Pi at {PI_HOST}...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(PI_HOST, username=PI_USER, password=PI_PASS, timeout=10)
    print("Connected.")

    # Write remote script to /tmp/regen_m_n.py and run it
    sftp = ssh.open_sftp()
    with sftp.open("/tmp/regen_m_n.py", "w") as f:
        f.write(REMOTE_SCRIPT)

    print("Running regeneration on Pi...")
    stdin, stdout, stderr = ssh.exec_command("/home/raspberrypi5/brailce/venv/bin/python3 /tmp/regen_m_n.py")
    out = stdout.read().decode("utf-8")
    err = stderr.read().decode("utf-8")
    print(out)
    if err:
        print("STDERR:", err)

    # Sync char_M.wav and char_N.wav back to PC
    local_char_dir = os.path.join(LOCAL_BASE, "brailce", "audio", "characters")
    for f in ["char_M.wav", "char_N.wav"]:
        rem_f = f"/home/raspberrypi5/brailce/brailce/audio/characters/{f}"
        loc_f = os.path.join(local_char_dir, f)
        print(f"Downloading {rem_f} -> {loc_f}")
        sftp.get(rem_f, loc_f)
        with wave.open(loc_f, "rb") as wf:
            dur = wf.getnframes() / wf.getframerate()
            print(f"  Verified local {f}: {dur:.3f}s (No 'Please say'!)")

    # Sync switched_en.wav back to PC
    local_fil_dir = os.path.join(LOCAL_BASE, "brailce", "audio", "filipino")
    rem_en = "/home/raspberrypi5/brailce/brailce/audio/filipino/switched_en.wav"
    loc_en = os.path.join(local_fil_dir, "switched_en.wav")
    print(f"Downloading {rem_en} -> {loc_en}")
    sftp.get(rem_en, loc_en)
    with wave.open(loc_en, "rb") as wf:
        dur = wf.getnframes() / wf.getframerate()
        print(f"  Verified local switched_en.wav: {dur:.3f}s (Ryan American Voice!)")

    sftp.close()
    ssh.close()
    print("Regeneration and download complete!")

if __name__ == "__main__":
    main()
