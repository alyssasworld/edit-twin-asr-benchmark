# Creator ASR Benchmark

Which speech-to-text model works best for cutting content creator videos? My editing tool (edit-twin-mcp) cuts on word timings and needs fillers (um/uh) transcribed, which public benchmarks don't measure. This benchmark tests models on my own footage.

## Models 
(currently tracking: https://huggingface.co/datasets/hf-audio/open-asr-leaderboard)
| Model | Status |
|---|---|
| Whisper large-v3-turbo (baseline, current production) | done |
| Parakeet TDT 0.6B v3 (NVIDIA) | done: leads (see [Results](#results)) |
| Qwen3-ASR 1.7B (Alibaba) | to do |

## Metrics and decision rule
Fixed before running anything: see [docs/metrics.md](docs/metrics.md).

## Dataset
| | |
|---|---|
| Clips | 5 (2 reels, 3 vlog clips; 1 voiceover) |
| Audio | ~11 min, 16 kHz mono, raw uncut footage |
| Reference words | 1,197 |
| Fillers (um/uh/hmm) | 20 |
| Speakers | 1 (me) |
| Word timings | 680 words over 3 vlog clips (automatic: aligner + waveform fit, spot-checked by eye) |
| Silence clips | 1 (53 s room tone) |

Clips and transcripts are my own unpublished footage and are not included in this
repo. References were transcribed verbatim by hand, starting from Whisper drafts; see
[docs/labelling-guide.md](docs/labelling-guide.md).

## How it works

```mermaid
flowchart LR
    subgraph inputs["Inputs (private, git-ignored)"]
        audio["data/audio/clip.wav<br/>raw footage, 16 kHz mono"]
        refs["data/refs/clip.txt + clip.words.txt<br/>my hand-checked answers"]
    end

    subgraph step1["1 · Transcribe"]
        models["models.py<br/>one function per model"]
        run["run.py<br/>warm-up + 3 timed runs per clip"]
    end

    hyps["data/hyps/model-name/clip.json<br/>words + timings, speed, memory"]

    subgraph step2["2 · Score"]
        textnorm["textnorm.py<br/>verbatim · clean · formatted views"]
        score["score.py<br/>WER, fillers, timestamps, speed"]
    end

    results["results/metrics/<br/>summary.csv · per_clip.csv"]
    edittwin[("edit-twin-mcp<br/>production prompt")]

    edittwin -.-> models
    audio --> run
    models --> run
    run --> hyps
    hyps --> score
    refs --> score
    textnorm --> score
    score --> results
```

1. **Transcribe:** `run.py` sends every clip through one model from `models.py` and
   saves the words, their timings and how long each run took.
2. **Score:** `score.py` compares each model's words with my references, after
   `textnorm.py` puts both into the same format, and writes one row per model.

Every model returns the same `[[word, start, end], ...]` format, so adding a model
only means adding one function to `models.py`.

## Preparing the data
Done once, before any model is compared. Each step says which tool to use and what it
produces. Transcript rules and Audacity click-by-click steps:
[docs/labelling-guide.md](docs/labelling-guide.md).

**1. Audio** (tool: `ffmpeg`). Cut each clip from raw footage into a 16 kHz mono `.wav`:

    ffmpeg -i footage.MP4 -ss 00:01:00 -t 120 -vn -ac 1 -ar 16000 data/audio/<clip>.wav

**2. Reference transcript** (tool: a text editor). Since I initially started with MCP creation and tested via editing my own videos, I already had some Whisper transcripts created. So, I started from a Whisper transcript and
correct every word while listening, keeping fillers and retakes. Then saved as
`data/refs/<clip>.txt`.

**3. Word timings**, for a few clips only. Timing every word by hand takes hours, so
the timings are made automatically and spot-checked. The tools were built with Claude
Code.
- a. **Draft** (`tools/draft_timings.py`): a neutral aligner (Meta's MMS, not one of
  the models being tested) finds a start and end time for every word of the reference.
  It writes `data/refs/<clip>.words.draft.txt`:

      uv run --no-project --with torch==2.8.0 --with torchaudio==2.8.0 \
          --with whisper-normalizer python tools/draft_timings.py <clip>

- b. **Fit to the waveform** (`tools/refine_timings.py`): the aligner makes words too
  short, so each label is stretched until the sound actually starts and stops. This
  rewrites the draft (the original is kept as `.words.aligned.txt`) and saves a plot
  to `data/refs/<clip>.draft.png`:

      uv run --with matplotlib python tools/refine_timings.py <clip> --plot

- c. **Spot-check the plot by eye**: each orange label should sit on a burst of sound
  and not spill into silence. If it looks right, copy the draft to
  `data/refs/<clip>.words.txt`, the file that gets scored. (Optional, more thorough:
  import the draft into Audacity as labels and listen to each word; see the guide.)

How it went. Opening the aligner's labels in Audacity showed the problem: short words
like "to" and "be" are slivers, much shorter than the sound above them.

![Aligner labels in Audacity, before fitting](docs/images/audacity-aligner-labels.png)

After fitting, the labels cover the sound (blue = aligner, orange = fitted). I
checked this plot of the first 12 s of `vlog_intro_day_1` by eye.

![Labels before and after fitting to the waveform](docs/images/timing-fit-before-after.png)

This changed the results: against the aligner's labels Whisper's median end error
looked like 46 ms, against the fitted labels it is 120 ms. Whisper ends words early
by about as much as the aligner did, so the unfitted labels hid the problem.

Limits: looking at a plot shows that a label sits on sound, not that it's the right
word, and inside fast connected speech there is no visible gap to fit to. No labels
were checked by ear. Every model is scored against the same labels, so comparisons
between models are fair; treat the exact millisecond values as approximate.

**4. Silence clip** (tools: a camera or mic, then `ffmpeg`). Record ~60 s of the empty
room where you film, with the same mic. Convert it, trimming off the seconds where you
touched the camera:

    ffmpeg -i roomtone.MP4 -ss 5 -to 58 -vn -ac 1 -ar 16000 data/audio/silence_roomtone.wav

**Then run the models** (next section). After new or changed audio, re-run `src.run`
for every model. After changed references or timings, re-running `src.score` is enough.

## How to run
    uv run python -m src.run whisper-turbo   # transcribe all clips -> data/hyps/
    uv run python -m src.run parakeet-tdt    # same for every model in src/models.py
    uv run python -m src.score               # -> results/metrics/summary.csv, per_clip.csv
    uv run pytest                            # fast checks on small examples
    uv run pytest -m slow                    # also run the real model on one clip

## Repo layout
    src/textnorm.py   normalise transcripts (verbatim, clean, formatted views)
    src/score.py      compute the metrics
    src/models.py     one function per model, all returning [[word, start, end], ...]
    src/run.py        transcribe every clip with one model, timing each run
    tools/draft_timings.py   draft word timings with a neutral aligner
    tools/refine_timings.py  fit those timings to the waveform and plot them for checking
    tests/            checks on small examples with known answers
    docs/             metrics, decision rule, labelling guide and README images
    data/             audio, references and model outputs (private, git-ignored)

## Results
Run 2026-11-01 on the Mac below. Full numbers: [results/metrics/](results/metrics/).

| | Parakeet TDT 0.6B v3 | Whisper large-v3-turbo |
|---|---|---|
| Verbatim WER (lower is better) | **3.8%** | 6.6% |
| Filler recall (um/uh caught) | **80%** (16 of 20) | 35% (7 of 20) |
| Filler precision | **94%** | 88% |
| End timestamp error, median | **100 ms** | 120 ms |
| Ends within 100 ms | **48%** | 42% |
| Start timestamp error, median | **100 ms** | 140 ms |
| Invented words on room tone | **0 / min** | 4.5 / min ("Thank you." ×2) |
| Formatted WER (case + punctuation) | 14.6% | **13.3%** |
| Speed (RTFx) | **95×** real time | 26× real time |
| Load time (first run) | **3.8 s** | 12.2 s |

**Decision rule:** Parakeet is not worse than Whisper on filler recall or end
timestamps, and has the lower verbatim WER, so **Parakeet wins** so far. Its one
weakness is punctuation and capitals, which matters for vlog subtitles.

Caveats:
- References started from Whisper drafts, which if anything favours Whisper.
- Parakeet runs in 30 s chunks. That setting was chosen after comparing options on
  these clips: 120 s chunks silently dropped 22 s of speech, and a single pass over the
  whole file was less accurate. It is tuned on the test set, so treat filler recall as
  roughly 80–90% (it varied with the chunk setting).
- Small dataset: 5 clips, ~11 min, one speaker, 20 fillers. Word timings cover 3 clips
  and were made automatically, spot-checked by eye on one plot, not by ear.
- Speed varies ~10% between runs on the same laptop; treat smaller gaps as a tie.

### Model setup notes
**Parakeet: how long audio is split matters.** Parakeet's first run used 120 s chunks
and scored 16% verbatim WER on `reel_navier_01`, much worse than Whisper. Listing its
errors showed ~45 words missing in one block rather than scattered mistakes, which
pointed to the code around the model, not the model. Its words had a 22 s gap
(179 s → 202 s): the library lost them when joining overlapping chunks. Comparing
settings on the two long clips (verbatim WER):

| Chunk size | `reel_navier_01` (4 min) | `vlog_intro_day_3` (6 min) |
|---|---|---|
| Whole file, one pass | 12.9% | 5.9% (and 8.4 GB GPU memory) |
| **30 s, 5 s overlap** (used) | **5.9%** | **1.7%** |
| 60 s, 15 s overlap | 9.1% | 1.9% |
| 120 s, 15 s overlap | 16.1% (22 s of speech lost) | 1.9% |

30 s was best on both clips and matches the 30 s window Whisper uses. Because it was
picked by looking at these test clips, it slightly flatters Parakeet; with only 5
clips there's no separate set to tune on. Lesson: when a model scores badly, check
how it's being run before blaming the model.

## Setup
- Apple M4 Pro, 24 GB RAM, macOS 15.6.1
- Python 3.13, managed with [uv](https://docs.astral.sh/uv/)
- Requires [edit-twin-mcp](https://github.com/alyssasworld/edit-twin-mcp) cloned next
  to this repo (`../edit-twin-mcp`): the Whisper baseline uses its production
  verbatim prompt.
- For preparing data: `ffmpeg` and Audacity (`brew install ffmpeg`,
  `brew install --cask audacity`)

      uv sync