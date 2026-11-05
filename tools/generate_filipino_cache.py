"""
Generate 100% Offline Grade 1 Filipino Audio Bank
=================================================

Uses edge-tts (fil-PH-AngeloNeural) and miniaudio to generate
22050 Hz, 16-bit mono S16_LE WAV files padded with 120ms pre-silence
and 150ms post-silence into brailce/audio/filipino/.
"""

import asyncio
import os
import struct
import wave
import edge_tts
import miniaudio

VOICE = "fil-PH-AngeloNeural"
TARGET_SAMPLE_RATE = 22050

OUTPUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "brailce", "audio", "filipino")
)

ANNOUNCEMENTS = {
    "switched_fil.wav": "Wikang Filipino.",
    "switched_en.wav": "English language.",
}

# Top Grade 1 Filipino sight words, CVC/CVCV beginner words, pronouns, numbers, greetings
# (Strictly standard 26-letter A-Z spelling)
FILIPINO_WORDS = [
    # Pronouns & Demonstratives
    "ako", "ikaw", "siya", "kami", "tayo", "sila", "ito", "iyan", "iyon",
    # Markers & Particles
    "ang", "mga", "sa", "ng", "na", "at", "ay", "ba", "pa", "po", "opo", "oo", "hindi",
    # Family & People
    "bata", "aso", "pusa", "ibon", "isda", "baka", "baboy", "manok",
    "nanay", "tatay", "ate", "kuya", "bunso", "lolo", "lola", "guro", "kaibigan",
    # School & Home Objects
    "bahay", "paaralan", "silid", "mesa", "silya", "libro", "aklat", "lapis", "papel", "pisara", "bag",
    "damit", "sapatos",
    # Food & Drink
    "tubig", "kanin", "ulam", "gatas", "tinapay", "prutas", "gulay", "saging", "mangga", "mansanas",
    # Nature & Environment
    "araw", "buwan", "bituin", "ulan", "hangin", "ulap", "apoy", "lupa", "puno", "bulaklak", "dahon",
    # Body Parts
    "mata", "ilong", "bibig", "tainga", "kamay", "paa", "ulo", "buhok", "ngipin", "puso",
    # Common Grade 1 Actions / Verbs
    "kain", "inom", "tulog", "gising", "laro", "basa", "sulat", "lakad", "takbo", "upo", "bigay", "kuha",
    "bukas", "sara", "laba", "linis", "luto", "hugas",
    # Common Descriptors / Adjectives
    "mabait", "masaya", "malungkot", "maganda", "malaki", "maliit", "mahaba", "maikli", "malinis", "marumi",
    "mainit", "malamig", "bago", "luma", "mabilis", "mabagal", "malakas", "mahina", "tama", "mali",
    # Numbers 1-10
    "isa", "dalawa", "tatlo", "apat", "lima", "anim", "pito", "walo", "siyam", "sampu",
    # Polite Expressions & Greetings
    "salamat", "kumusta", "paalam", "tulong", "magandang", "umaga", "tanghali", "hapon", "gabi",
]


def trim_silence(samples: list[int], threshold: int = 200) -> list[int]:
    """Trim leading and trailing digital silence."""
    if not samples:
        return samples

    start = 0
    for i, s in enumerate(samples):
        if abs(s) > threshold:
            start = max(0, i - int(TARGET_SAMPLE_RATE * 0.020))
            break

    end = len(samples)
    for i in range(len(samples) - 1, -1, -1):
        if abs(samples[i]) > threshold:
            end = min(len(samples), i + int(TARGET_SAMPLE_RATE * 0.020))
            break

    if start >= end:
        return samples
    return samples[start:end]


def save_wav_with_padding(samples: list[int], out_path: str, pre_ms: int = 120, post_ms: int = 150) -> None:
    """Pad samples with exact silence and save as 22050Hz 16-bit mono WAV."""
    pre_silence = b"\x00\x00" * int(TARGET_SAMPLE_RATE * (pre_ms / 1000.0))
    post_silence = b"\x00\x00" * int(TARGET_SAMPLE_RATE * (post_ms / 1000.0))
    raw_pcm = struct.pack(f"<{len(samples)}h", *samples)
    payload = pre_silence + raw_pcm + post_silence

    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(TARGET_SAMPLE_RATE)
        wf.writeframes(payload)


async def generate_speech_wav(text: str, out_path: str) -> None:
    """Generate audio using edge-tts and decode to 22050Hz 16-bit mono WAV."""
    comm = edge_tts.Communicate(text, VOICE)
    mp3_bytes = b""
    async for chunk in comm.stream():
        if chunk["type"] == "audio":
            mp3_bytes += chunk["data"]

    decoded = miniaudio.decode(
        mp3_bytes,
        nchannels=1,
        sample_rate=TARGET_SAMPLE_RATE,
        output_format=miniaudio.SampleFormat.SIGNED16,
    )
    samples = list(decoded.samples)
    trimmed = trim_silence(samples)
    save_wav_with_padding(trimmed, out_path, pre_ms=120, post_ms=150)


async def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Target directory: {OUTPUT_DIR}")
    print(f"Voice: {VOICE} ({TARGET_SAMPLE_RATE} Hz, 16-bit mono)")

    # 1. Announcements
    print("\n--- 1. Generating Announcements ---")
    for fname, text in ANNOUNCEMENTS.items():
        out_path = os.path.join(OUTPUT_DIR, fname)
        await generate_speech_wav(text, out_path)
        with wave.open(out_path, "rb") as wf:
            dur = wf.getnframes() / wf.getframerate()
            print(f"  OK: {fname:20s} <- {text!r} ({dur:.2f}s)")

    # 2. Vocabulary words
    # Deduplicate while preserving order
    seen = set()
    unique_words = []
    for w in FILIPINO_WORDS:
        clean_w = w.strip().lower()
        if clean_w and clean_w not in seen:
            seen.add(clean_w)
            unique_words.append(clean_w)

    print(f"\n--- 2. Generating {len(unique_words)} Grade 1 Filipino Words ---")
    for i, word in enumerate(unique_words, 1):
        fname = f"word_{word}.wav"
        out_path = os.path.join(OUTPUT_DIR, fname)
        # Speak word cleanly (e.g. "aso.")
        tts_text = f"{word}."
        await generate_speech_wav(tts_text, out_path)
        with wave.open(out_path, "rb") as wf:
            dur = wf.getnframes() / wf.getframerate()
            print(f"[{i:3d}/{len(unique_words):3d}] OK: {fname:25s} ({dur:.2f}s)")

    print(f"\nSuccessfully generated {len(unique_words) + len(ANNOUNCEMENTS)} Filipino audio files!")


if __name__ == "__main__":
    asyncio.run(main())

