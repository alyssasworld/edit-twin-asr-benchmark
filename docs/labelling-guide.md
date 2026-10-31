# How I write reference transcripts

## Files
- `data/audio/<clip>.wav`: 16 kHz mono, 1–3 min.
  Make with: `ffmpeg -i in.mov -ss 00:01:00 -t 120 -ac 1 -ar 16000 data/audio/<clip>.wav`
- `data/refs/<clip>.txt`: what I actually said, word for word.
- `data/refs/<clip>.words.tsv`: word timings (only some clips), exported from
  Audacity: Label Track → one label per word → File > Export > Labels.
- Clip names: `reel_...`, `vlog_...`, `vo_...` (voiceover). Clips with no speech start
  with `silence_` (e.g. `silence_roomtone.wav`, `silence_music.wav`) and need no reference.

## Rules
1. Write every filler: um, uh, hmm. Use these spellings only.
2. Keep false starts and repeats exactly as said: "I was, I was going to".
3. Normal punctuation and capitals, as I'd want in a subtitle.
4. Numbers as I'd want them captioned ("$20", "3 days"). Be consistent.
5. Australian spelling (finalising, colour).
6. Unclear word: listen 3 times at 0.75x speed; if still unsure, write my best guess
   and note it at the bottom of this file.
7. Start from a Whisper transcript, then correct every word while listening.
