# Brailce Project Progress

## Status
- [x] Step 0: Working Folder & Branch Setup (BRAILCE ADJUST FOLDER, master branch verified)
- [x] Phase A: Filipino Letters (Respelling, consistent intonation, Blessica/Rosa carrier probe, char_O fix, ASR/F0 metrics, audition)
  - Measured: 26 letters standardized with active duration 0.31s-0.64s, median F0 ~170-190Hz, lead silence 15ms, tail 80ms. char_O fixed (voiced oh, F0 334Hz, ASR 'Oh.'). Tools generated: tools/letter_probe.py, tools/probe_results.json, tools/voice_audition/fil_letters/ (alphabet.wav, index.html).
- [ ] Phase B: Filipino Digits, Math Operators, Punctuation Names (tuldok, kuwit, tandang pananong, tandang padamdam; Blessica; consistent intonation)
- [ ] Phase C: Eliminate Deep Male Voice in Math & AI Assistant Echo (Tagalog numbers to 9999 + arithmetic phrases, F0 audit)
- [ ] Phase D: Filipino Mode Intros ("AI Mode") & Scanner Mode Speech (Whole-phrase clips, gap audit)
- [ ] Phase E: English Fixes (V vs B, digits 2 & 7, math/AI juncture, latency/silence trim, remove "Thinking, please wait.")
- [ ] Phase F: Scanner Mode (F1: Braille alphabet Grade 1 JSON dataset & synthetic generator; F2: Multi-condition presence gate; F3: Spell-out and pronounce pipeline)
- [ ] Phase G: LCD Screen (G1: Configurable SPI speed, 640x480 preview, threaded pipeline; G2: Layout wrap/no-overlap, missing-glyph fix)
- [ ] Phase H: Verification, Docs, Git & Push to master
