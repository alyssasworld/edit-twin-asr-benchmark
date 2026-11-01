# Preparing the data

Four steps, in order. Everything lands in `data/` (private, git-ignored).

| Step | Output | Needed for |
|---|---|---|
| 1. Audio | `data/audio/<clip>.wav` | everything |
| 2. Reference transcript | `data/refs/<clip>.txt` | WER, CER, fillers |
| 3. Word timings (some clips) | `data/refs/<clip>.words.txt` | timestamp error |
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
`data/refs/<clip>.words.txt`: the start and end of every word, for a few clips only
(currently the three `vlog_intro_day_*` clips, which hold almost all the fillers).
Labelling from scratch takes hours, so draft them first, then correct by ear.

**a. Draft.** `tools/draft_timings.py` aligns my reference text to the audio with
a neutral aligner (Meta's MMS, not one of the models being tested). It runs in its
own throwaway environment so the benchmark's torch version isn't changed. The first
run downloads a 1.2 GB model.

    uv run --no-project --with torch==2.8.0 --with torchaudio==2.8.0 \
        --with whisper-normalizer python tools/draft_timings.py <clip> [<clip> ...]

This writes `data/refs/<clip>.words.draft.txt`. The scorer ignores `.draft` files.

**b. Fit to the waveform.** The aligner makes words too short (a word "ends" while I'm
still saying it). `tools/refine_timings.py` stretches each label's edges until the
sound falls back to the room's noise level, never overlapping the next word. Edges in
fast connected speech have no quiet to snap to, so they stay as the aligner put them.

    uv run --with matplotlib python tools/refine_timings.py <clip> [<clip> ...] --plot

This rewrites `<clip>.words.draft.txt`, keeps the untouched aligner output as
`<clip>.words.aligned.txt`, and saves `<clip>.draft.png` (orange = fitted labels,
blue = original aligner labels) for a quick look.

Why it matters: against the unfitted drafts Whisper's median end error was 46 ms;
against fitted labels it's 120 ms. Whisper ends words early by about the same amount
the aligner did, so unfitted labels hid the problem.

**c. Check in Audacity** (`brew install --cask audacity`, tested with Audacity 4.0):
1. File → Open `data/audio/<clip>.wav`.
2. File → Import → Labels… → `data/refs/<clip>.words.draft.txt`. (The label file
   must end in `.txt`, or Audacity won't show it.)
3. Zoom in (⌘1), click a label and press Space to play it. Only drag an edge if the
   word sounds cut off or includes part of the next word. Ends matter most (cuts land
   there). Never rename, add or delete labels.
4. File → Export → Export Labels… → `data/refs/<clip>.words.txt` (no `.draft`).

If I edit a `.txt` reference later, redo the draft and the corrections for that clip,
so the words in both files still match.

## 4. Silence clips
Tests whether a model invents words when nobody is speaking. No reference needed.

- **Room tone** (done): ~60 s of nothing, same mic and room I film in.
- **Music** (optional, not used yet): ~60 s of a background track, ideally no vocals.

Pressing record and stop makes handling noise at the start and end, which isn't
room tone. Find it by printing the loudest level of each second (look for numbers far
above the rest, e.g. -8 when the rest is about -55):

    ffmpeg -nostats -i roomtone.MP4 -vn -ac 1 -af "asetnsamples=48000,astats=metadata=1:reset=1,ametadata=print:key=lavfi.astats.Overall.Peak_level" -f null - 2>&1 | grep -o 'Peak_level=.*'

Then keep only the quiet stretch (`-ss` start, `-to` end, in seconds):

    ffmpeg -i roomtone.MP4 -ss 5 -to 58 -vn -ac 1 -ar 16000 data/audio/silence_roomtone.wav

## After preparing
New or changed audio → re-run every model (`uv run python -m src.run <model>`), then
score. Changed references or timings only → `uv run python -m src.score`.
