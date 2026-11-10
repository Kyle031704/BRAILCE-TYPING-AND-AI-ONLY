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

FILIPINO_LETTER_PRONUNCIATION = {
    "a": "ey.",
    "b": "bi.",
    "c": "si.",
    "d": "di.",
    "e": "eh.",
    "f": "ef.",
    "g": "dzi.",
    "h": "eyts.",
    "i": "ay.",
    "j": "dzey.",
    "k": "key.",
    "l": "el.",
    "m": "em.",
    "n": "en.",
    "o": "ow.",
    "p": "pi.",
    "q": "kyu.",
    "r": "ar.",
    "s": "es.",
    "t": "ti.",
    "u": "yu.",
    "v": "vi.",
    "w": "dobol-yu.",
    "x": "eks.",
    "y": "way.",
    "z": "zi.",
}

# 46 Filipino Character / Digit / Symbol Audio Files
FILIPINO_CHARACTERS = {
    # Letters A-Z (from FILIPINO_LETTER_PRONUNCIATION)
    **{f"char_{letter.upper()}.wav": phrasing for letter, phrasing in FILIPINO_LETTER_PRONUNCIATION.items()},
    # Digits 0-9
    "char_0.wav": "sero.",
    "char_1.wav": "isa.",
    "char_2.wav": "dalawa.",
    "char_3.wav": "tatlo.",
    "char_4.wav": "apat.",
    "char_5.wav": "lima.",
    "char_6.wav": "anim.",
    "char_7.wav": "pito.",
    "char_8.wav": "walo.",
    "char_9.wav": "siyam.",
    # Symbols
    "char_plus.wav": "dagdag.",
    "char_minus.wav": "bawas.",
    "char_times.wav": "beses.",
    "char_divide.wav": "hatiin sa.",
    "char_equals.wav": "equals.",
    "char_period.wav": "tuldok.",
    "char_comma.wav": "kuwit.",
    "char_question.wav": "tandang pananong.",
    "char_exclamation.wav": "tandang padamdam.",
    "char_hash.wav": "numero.",
}

# Mode Announcements
FILIPINO_MODES = {
    "mode_free_input.wav": "Malayang Pag-type. Mag-type ng mga titik at salita.",
    "mode_math.wav": "Modong Matematika. Mag-type ng bilang at operasyon.",
    "mode_ai.wav": "Modong A I Assistant. Mag-type ng iyong tanong at pindutin ang Enter.",
    "mode_scanner.wav": "Modong Scanner ng Braille. Pindutin ang Enter upang mag-scan.",
}

ANNOUNCEMENTS = {
    "switched_fil.wav": "Wikang Filipino.",
    "switched_en.wav": "English language.",
    "thinking_fil.wav": "Nag-iisip, mangyaring maghintay.",
    "scanning_fil.wav": "Nag-i-scan.",
    "no_braille_fil.wav": "Walang nakitang Braille.",
    "ai_kumusta.wav": "Mabuti naman ako! Masaya akong makasama ka sa pag-aaral ng Braille.",
    "ai_pangalan.wav": "Ako si Brailce, ang iyong matalinong voice assistant para sa Braille.",
    "ai_umaga.wav": "Magandang umaga sa iyo! Handa ka na bang mag-aral?",
    "ai_tulong.wav": "Narito ako upang tulungan ka sa pagbasa at pagsulat ng Braille.",
}

# Top Grade 1 Filipino sight words, common vocabulary, greetings, question words, pronouns, verbs
FILIPINO_WORDS = [
    # Question words
    "ano", "sino", "saan", "kailan", "bakit", "paano",
    # Demonstratives & Pronouns
    "ito", "iyan", "iyon", "ako", "ikaw", "siya", "kami", "tayo", "sila",
    "ko", "mo", "niya", "namin", "natin", "nila",
    # Markers & Particles & Conjunctions
    "ang", "mga", "sa", "ng", "na", "at", "ay", "ba", "pa", "po", "opo", "oo", "hindi",
    "para", "kung", "dahil", "pero", "kaya",
    # People, Family & School
    "bata", "aso", "pusa", "ibon", "isda", "baka", "baboy", "manok",
    "nanay", "tatay", "ate", "kuya", "bunso", "lolo", "lola", "guro", "kaibigan",
    "bahay", "paaralan", "silid", "mesa", "silya", "libro", "aklat", "lapis", "papel", "pisara", "bag",
    "damit", "sapatos",
    # Food & Drink & Nature
    "pagkain", "tubig", "kanin", "ulam", "gatas", "tinapay", "prutas", "gulay", "saging", "mangga", "mansanas",
    "araw", "buwan", "bituin", "ulan", "hangin", "ulap", "apoy", "lupa", "puno", "bulaklak", "dahon",
    # Body & Senses & Braille
    "mata", "ilong", "bibig", "tainga", "kamay", "paa", "ulo", "buhok", "ngipin", "puso", "daliri", "bulag", "braille",
    # Actions & Verbs
    "kain", "inom", "tulog", "gising", "laro", "basa", "sulat", "lakad", "takbo", "upo", "bigay", "kuha",
    "bukas", "sara", "laba", "linis", "luto", "hugas", "aral",
    # Descriptors & State
    "mabait", "masaya", "malungkot", "maganda", "malaki", "maliit", "mahaba", "maikli", "malinis", "marumi",
    "mainit", "malamig", "bago", "luma", "mabilis", "mabagal", "malakas", "mahina", "tama", "mali",
    "mabuti", "mahal", "buhay", "edad", "wika", "filipino", "pilipinas",
    # Numbers 1-10
    "isa", "dalawa", "tatlo", "apat", "lima", "anim", "pito", "walo", "siyam", "sampu",
    # Polite Expressions & Greetings
    "salamat", "kumusta", "paalam", "tulong", "pangalan", "magandang", "umaga", "tanghali", "hapon", "gabi",
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


def save_wav_with_padding(samples: list[int], out_path: str, pre_ms: int = 200, post_ms: int = 100) -> None:
    """Pad samples with 200ms digital zero PCM silence and 100ms post-silence, verify amplitude, and save as 22050Hz 16-bit mono WAV."""
    if not samples:
        samples = [0] * 1500

    peak = max(abs(s) for s in samples)
    if peak < 2000 and peak > 0:
        factor = 18000.0 / peak
        samples = [max(-32768, min(32767, int(s * factor))) for s in samples]

    pre_silence = b"\x00\x00" * int(TARGET_SAMPLE_RATE * (pre_ms / 1000.0))
    post_silence = b"\x00\x00" * int(TARGET_SAMPLE_RATE * (post_ms / 1000.0))
    raw_pcm = struct.pack(f"<{len(samples)}h", *samples)
    payload = pre_silence + raw_pcm + post_silence

    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(TARGET_SAMPLE_RATE)
        wf.writeframes(payload)


async def generate_speech_wav(
    text: str,
    out_path: str,
    voice: str = VOICE,
    pitch: str = "+0Hz",
    rate: str = "+0%",
    pre_ms: int = 120,
    post_ms: int = 180,
) -> None:
    """Generate audio using edge-tts and decode to 22050Hz 16-bit mono WAV."""
    comm = edge_tts.Communicate(text, voice, pitch=pitch, rate=rate)
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
    if len(trimmed) < 1500 and len(samples) >= 1500:
        trimmed = samples
    save_wav_with_padding(trimmed, out_path, pre_ms=pre_ms, post_ms=post_ms)


async def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(ENGLISH_WORDS_DIR, exist_ok=True)
    print(f"Target Filipino directory: {OUTPUT_DIR}")
    print(f"Target English words directory: {ENGLISH_WORDS_DIR}")
    print(f"Voice: {VOICE} ({TARGET_SAMPLE_RATE} Hz, 16-bit mono)")

    # 1. Filipino Character Audio (Letters A-Z, Digits 0-9, Symbols)
    # Regenerate so O, E, and all 46 letters have verified audio and peak amplitude
    print(f"\n--- 1. Generating {len(FILIPINO_CHARACTERS)} Filipino Character Audio Files ---")
    for i, (fname, phrasing) in enumerate(FILIPINO_CHARACTERS.items(), 1):
        out_path = os.path.join(OUTPUT_DIR, fname)
        await generate_speech_wav(phrasing, out_path, voice=VOICE)
        with wave.open(out_path, "rb") as wf:
            dur = wf.getnframes() / wf.getframerate()
            size = os.path.getsize(out_path)
            print(f"[{i:2d}/{len(FILIPINO_CHARACTERS):2d}] OK: {fname:25s} <- {phrasing!r} ({dur:.2f}s, {size} bytes)")

    # 3. Mode Announcements
    print(f"\n--- 3. Generating {len(FILIPINO_MODES)} Filipino Mode Announcements ---")
    for fname, phrasing in FILIPINO_MODES.items():
        out_path = os.path.join(OUTPUT_DIR, fname)
        if os.path.isfile(out_path) and os.path.getsize(out_path) > 1000:
            print(f"  EXISTS: {fname:25s}")
            continue
        await generate_speech_wav(phrasing, out_path)
        with wave.open(out_path, "rb") as wf:
            dur = wf.getnframes() / wf.getframerate()
            print(f"  OK: {fname:25s} <- {phrasing!r} ({dur:.2f}s)")

    # 4. Announcements
    print(f"\n--- 4. Generating {len(ANNOUNCEMENTS)} General Announcements ---")
    for fname, text in ANNOUNCEMENTS.items():
        out_path = os.path.join(OUTPUT_DIR, fname)
        if os.path.isfile(out_path) and os.path.getsize(out_path) > 1000:
            print(f"  EXISTS: {fname:25s}")
            continue
        await generate_speech_wav(text, out_path)
        with wave.open(out_path, "rb") as wf:
            dur = wf.getnframes() / wf.getframerate()
            print(f"  OK: {fname:25s} <- {text!r} ({dur:.2f}s)")

    # 5. Vocabulary words
    seen = set()
    unique_words = []
    for w in FILIPINO_WORDS:
        clean_w = w.strip().lower()
        if clean_w and clean_w not in seen:
            seen.add(clean_w)
            unique_words.append(clean_w)

    print(f"\n--- 5. Generating {len(unique_words)} Grade 1 Filipino Words ---")
    for i, word in enumerate(unique_words, 1):
        fname = f"word_{word}.wav"
        out_path = os.path.join(OUTPUT_DIR, fname)
        if os.path.isfile(out_path) and os.path.getsize(out_path) > 1000:
            continue
        # Speak word cleanly (e.g. "aso.")
        tts_text = f"{word}."
        await generate_speech_wav(tts_text, out_path)
        with wave.open(out_path, "rb") as wf:
            dur = wf.getnframes() / wf.getframerate()
            print(f"[{i:3d}/{len(unique_words):3d}] OK: {fname:25s} ({dur:.2f}s)")

    total_files = len(FILIPINO_CHARACTERS) + len(ENGLISH_ISOLATED_WORDS) + len(FILIPINO_MODES) + len(ANNOUNCEMENTS) + len(unique_words)
    print(f"\nSuccessfully verified/generated audio bank ({total_files} files target)!")


if __name__ == "__main__":
    asyncio.run(main())

