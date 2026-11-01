# Preparing the data

Four steps, in order. Everything lands in `data/` (private, git-ignored).

| Step | Output | Needed for |
|---|---|---|
| 1. Audio | `data/audio/<clip>.wav` | everything |
| 2. Reference transcript | `data/refs/<clip>.txt` | WER, CER, fillers |
| 3. Word timings (some clips) | `data/refs/<clip>.words.tsv` | timestamp error |
| 4. Silence clips | `data/audio/silence_*.wav` | hallucination |

Clip names: `reel_...`, `vlog_...`, `vo_...` (voiceover), `silence_...` (no speech).

## 1. Audio
16 kHz mono `.wav`, cut from raw (unedited) footage:

    ffmpeg -i in.mov -ss 00:01:00 -t 120 -ac 1 -ar 16000 data/audio/<clip>.wav

## 2. Reference transcripts
`data/refs/<clip>.txt`: what I actually said, word for word.

1. Write every filler: um, uh, hmm. Use these spellings only.
2. Keep false starts and repeats exactly as said: "I was, I was going to".
3. Normal punctuation and capitals, as I'd want in a subtitle.
4. Numbers as I'd want them captioned ("$20", "3 days"). Be consistent.
5. Australian spelling (finalising, colour).
6. Spell names and brands the same way every time ("Pop Mart", not sometimes "Popmart").
7. Unclear word: listen 3 times at 0.75x speed; if still unsure, write my best guess
   and note it at the bottom of this file.
8. Start from a Whisper transcript, then correct every word while listening.
   (Uncorrected Whisper words would make Whisper look better than it is.)

## 3. Word timings
`data/refs/<clip>.words.tsv`: the start and end of every word, for a few clips only
(currently the three `vlog_intro_day_*` clips, which hold almost all the fillers).
Labelling from scratch takes hours, so draft them first, then correct by ear.

**a. Draft.** `tools/draft_timings.py` aligns my reference text to the audio with
a neutral aligner (Meta's MMS, not one of the models being tested). It runs in its
own throwaway environment so the benchmark's torch version isn't changed. The first
run downloads a 1.2 GB model.

    uv run --no-project --with torch==2.8.0 --with torchaudio==2.8.0 \
        --with whisper-normalizer python tools/draft_timings.py <clip> [<clip> ...]

This writes `data/refs/<clip>.words.draft.tsv`. The scorer ignores `.draft` files.

**b. Correct in Audacity** (`brew install --cask audacity`):
1. File → Open `data/audio/<clip>.wav`.
2. File → Import → Labels… → `data/refs/<clip>.words.draft.tsv`.
3. Zoom in (⌘1), play each label, drag its edges to cover the word from its first
   sound to its last. Ends matter most (cuts land there). Watch short words like "i"
   (drafts can be too short) and um/uh.
4. File → Export → Export Labels… → `data/refs/<clip>.words.tsv` (no `.draft`).

If I edit a `.txt` reference later, redo the draft and the corrections for that clip,
so the words in both files still match.

## 4. Silence clips
Tests whether a model invents words when nobody is speaking. No reference needed.

- **Room tone:** ~60 s of nothing, same mic and room I film in.
- **Music:** ~60 s of a background track I use in videos, ideally no vocals.

      ffmpeg -i roomtone.m4a -t 60 -ac 1 -ar 16000 data/audio/silence_roomtone.wav
      ffmpeg -i music.mp3    -t 60 -ac 1 -ar 16000 data/audio/silence_music.wav

## After preparing
New or changed audio → re-run every model (`uv run python -m src.run <model>`), then
score. Changed references or timings only → `uv run python -m src.score`.
