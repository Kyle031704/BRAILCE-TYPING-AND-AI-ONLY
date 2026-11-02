"""
Brailce Voice Audition Tool
===========================
Auditions several voice profiles (male, different pitches, natural speeds)
so the user can pick the exact sound they prefer.
"""

import subprocess
import time
import sys

PROFILES = [
    {
        "id": "1",
        "name": "Natural American Male (Smooth & Calm)",
        "voice": "en-us+m3",
        "speed": 160,
        "pitch": 44,
        "gap": 2,
        "sample": "Hello. I am Brailce, your assistive reading device. Ready to type."
    },
    {
        "id": "2",
        "name": "Deep American Male (Warm, reduced buzz)",
        "voice": "en-us+m7",
        "speed": 155,
        "pitch": 38,
        "gap": 1,
        "sample": "Free Input Mode active. Press any Braille key to hear its spoken letter."
    },
    {
        "id": "3",
        "name": "Crisp British Male (Clear pronunciation)",
        "voice": "en-gb+m3",
        "speed": 162,
        "pitch": 45,
        "gap": 1,
        "sample": "Braille to speech system online. Switched to number mode."
    },
    {
        "id": "4",
        "name": "Neutral Standard (Clear & Direct)",
        "voice": "en-us",
        "speed": 165,
        "pitch": 46,
        "gap": 0,
        "sample": "Letter B. Braille pattern dots one and two."
    },
    {
        "id": "5",
        "name": "Fast Interactive Male (Snappy for typing)",
        "voice": "en-us+m1",
        "speed": 175,
        "pitch": 42,
        "gap": 0,
        "sample": "Quick responsive typing feedback."
    }
]

def main():
    print("=" * 60)
    print("      BRAILCE VOICE PROFILE AUDITION (MALE / NATURAL)")
    print("=" * 60)
    print("Testing 5 different voice tunings live on your speaker...\n")

    for p in PROFILES:
        print(f"[{p['id']}] Auditioning: {p['name']}")
        print(f"    Voice: {p['voice']} | Speed: {p['speed']} wpm | Pitch: {p['pitch']}")
        
        cmd = [
            "espeak-ng",
            "-v", p["voice"],
            "-s", str(p["speed"]),
            "-p", str(p["pitch"]),
            "-g", str(p["gap"]),
            "-a", "100",
            p["sample"]
        ]
        try:
            subprocess.run(cmd, check=True)
        except Exception:
            cmd[0] = "espeak"
            try:
                subprocess.run(cmd, check=True)
            except Exception as e2:
                print(f"    Failed to run speech: {e2}")
        
        time.sleep(1.2)
        print()

    print("=" * 60)
    print("Which voice profile did you like best (1, 2, 3, 4, or 5)?")
    print("=" * 60)

if __name__ == "__main__":
    main()
