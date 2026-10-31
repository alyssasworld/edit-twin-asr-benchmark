# Creator ASR Benchmark

Which speech-to-text model works best for cutting content creator videos? My editing tool (edit-twin-mcp) cuts on word timings and needs fillers (um/uh) transcribed, which public benchmarks don't measure. This benchmark tests models on my own footage.

## Models
| Model | Status |
|---|---|
| Whisper large-v3-turbo (baseline, current production) | in progress |
| Model X | to do |
| Model Y | to do |

## Metrics and decision rule
Fixed before running anything: see [docs/metrics.md](docs/metrics.md).

## How to run
    uv run python -m src.run whisper-turbo   # transcribe all clips
    uv run python -m src.score               # write results/summary.csv
    uv run pytest                            # check the scoring code

## Data
Clips are my own videos and are not published. See
[docs/labelling-guide.md](docs/labelling-guide.md) for how references were made.

## Results
TBD

## Setup
Python 3.13, Apple Silicon Mac (model and RAM: fill in).
