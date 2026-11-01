# Creator ASR Benchmark

Which speech-to-text model works best for cutting content creator videos? My editing tool (edit-twin-mcp) cuts on word timings and needs fillers (um/uh) transcribed, which public benchmarks don't measure. This benchmark tests models on my own footage.

## Models 
(currently tracking: https://huggingface.co/datasets/hf-audio/open-asr-leaderboard)
| Model | Status |
|---|---|
| Whisper large-v3-turbo (baseline, current production) | in progress |
| Parakeet TDT 0.6B v3 (NVIDIA) | to do |
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
| Word timings | 680 words over 3 vlog clips (aligner + waveform fit; `vlog_intro_day_1` checked in Audacity) |
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

**3. Word timings**, for a few clips only (tools: `tools/draft_timings.py`,
`tools/refine_timings.py`, then Audacity).
- a. **Draft:** run the aligner to get a start and end time for every word of the
  reference. It writes `data/refs/<clip>.words.draft.txt`:

      uv run --no-project --with torch==2.8.0 --with torchaudio==2.8.0 \
          --with whisper-normalizer python tools/draft_timings.py <clip>

- b. **Fit to the waveform:** the aligner makes words too short, so stretch each label
  until the sound actually starts and stops. This rewrites the draft (the original is
  kept as `.words.aligned.txt`) and saves a picture to `data/refs/<clip>.draft.png`:

      uv run --with matplotlib python tools/refine_timings.py <clip> --plot

- c. **Check in Audacity** (`brew install --cask audacity`): open the `.wav`, import the
  draft as labels, play each label and fix any that sound wrong, then export the labels
  as `data/refs/<clip>.words.txt` (no `.draft`). Only this checked file is scored.

**4. Silence clip** (tools: a camera or mic, then `ffmpeg`). Record ~60 s of the empty
room where you film, with the same mic. Convert it, trimming off the seconds where you
touched the camera:

    ffmpeg -i roomtone.MP4 -ss 5 -to 58 -vn -ac 1 -ar 16000 data/audio/silence_roomtone.wav

**Then run the models** (next section). After new or changed audio, re-run `src.run`
for every model. After changed references or timings, re-running `src.score` is enough.

## How to run
    uv run python -m src.run whisper-turbo   # transcribe all clips -> data/hyps/
    uv run python -m src.score               # -> results/metrics/summary.csv, per_clip.csv
    uv run pytest                            # fast checks on small examples
    uv run pytest -m slow                    # also run the real model on one clip

## Repo layout
    src/textnorm.py   normalise transcripts (verbatim, clean, formatted views)
    src/score.py      compute the metrics
    src/models.py     one function per model, all returning [[word, start, end], ...]
    src/run.py        transcribe every clip with one model, timing each run
    tools/draft_timings.py   draft word timings with a neutral aligner
    tools/refine_timings.py  fit those timings to the waveform, before checking in Audacity
    tests/            checks on small examples with known answers
    docs/             metrics, decision rule and labelling guide
    data/             audio, references and model outputs (private, git-ignored)

## Results
TBD

## Setup
- Apple M4 Pro, 24 GB RAM, macOS 15.6.1
- Python 3.13, managed with [uv](https://docs.astral.sh/uv/)
- Requires [edit-twin-mcp](https://github.com/alyssasworld/edit-twin-mcp) cloned next
  to this repo (`../edit-twin-mcp`): the Whisper baseline uses its production
  verbatim prompt.
- For preparing data: `ffmpeg` and Audacity (`brew install ffmpeg`,
  `brew install --cask audacity`)

      uv sync