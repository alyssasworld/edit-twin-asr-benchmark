"""
Transcribe every clip in data/audio with one model, timing each run.
Writes data/hyps/<model>/<clip>.json (words + timings) and data/hyps/<model>.meta.json
(load time, memory, which edit-twin-mcp version was used).

    uv run python -m src.run whisper-turbo
"""

import json  # save the outputs as .json files
import platform  # Python version, for the record
import resource  # peak memory of this process
import statistics  # median of the timed runs
import subprocess  # ask git which edit-twin-mcp commit is installed
import sys  # read the model name from the command line
import time  # time each transcription
import wave  # read the length of a .wav file
from datetime import datetime  # when the run happened
from pathlib import Path

from src.models import MODELS

AUDIO_DIR = Path("data/audio")  # input: my clips as 16 kHz mono .wav
HYPS_DIR = Path("data/hyps")  # output: one folder per model
RUNS_PER_CLIP = 3  # each clip is transcribed this many times; the median time is kept


def audio_duration_s(audio_path: Path) -> float:
    """Length of a .wav file in seconds (number of samples / samples per second)."""
    with wave.open(str(audio_path)) as wav_file:
        return wav_file.getnframes() / wav_file.getframerate()


def peak_memory_gb() -> dict:
    """The most memory this process has used so far, in GB.

    peak_rss_gb is normal (CPU) memory. MLX models also use GPU memory, which macOS
    may not count in that number, so MLX's own counter is saved as peak_gpu_gb.
    """
    # ru_maxrss is in bytes on macOS (it's kilobytes on Linux)
    peak_rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    try:
        import mlx.core as mx
        peak_gpu_gb = mx.get_peak_memory() / 1e9
    except Exception:  # not an MLX model, or an MLX version without this counter
        peak_gpu_gb = None
    return {"peak_rss_gb": peak_rss_gb, "peak_gpu_gb": peak_gpu_gb}


def edit_twin_commit() -> str | None:
    """The git commit of the installed edit-twin-mcp, so results record which
    production prompt they used. None if it can't be found."""
    try:
        import edit_twin_mcp
        # .../edit-twin-mcp/src/edit_twin_mcp/__init__.py -> .../edit-twin-mcp
        edit_twin_repo = Path(edit_twin_mcp.__file__).resolve().parents[2]
        git_output = subprocess.run(
            ["git", "-C", str(edit_twin_repo), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True)
        return git_output.stdout.strip()
    except Exception:
        return None


def timed_transcription(transcribe, audio_path: Path, runs: int) -> tuple[list, list[float]]:
    """Transcribe one clip `runs` times. Returns (words from the last run, seconds per run)."""
    run_times_s = []
    for _ in range(runs):
        start_time = time.perf_counter()  # a precise clock for measuring short durations
        timed_words = transcribe(str(audio_path))
        run_times_s.append(time.perf_counter() - start_time)
    return timed_words, run_times_s


def transcribe_all(model_name: str, transcribe, audio_dir: Path = AUDIO_DIR,
                   hyps_dir: Path = HYPS_DIR, runs: int = RUNS_PER_CLIP) -> None:
    """Transcribe every .wav in audio_dir and save the results for score.py."""
    audio_files = sorted(audio_dir.glob("*.wav"))  # sorted: same order every run
    if not audio_files:
        sys.exit(f"no .wav files in {audio_dir}")
    model_output_dir = hyps_dir / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    # warm-up: the first call loads the model into memory, which is much slower than
    # a normal run, so it's timed on its own and kept out of the speed numbers
    start_time = time.perf_counter()
    transcribe(str(audio_files[0]))
    first_run_s = time.perf_counter() - start_time
    print(f"warm-up (includes model load): {first_run_s:.1f}s")

    for audio_path in audio_files:
        timed_words, run_times_s = timed_transcription(transcribe, audio_path, runs)
        clip_audio_s = audio_duration_s(audio_path)
        median_s = statistics.median(run_times_s)  # middle value: ignores one-off slowdowns
        clip_output = {
            "words": timed_words,  # [[word, start_s, end_s], ...]
            "audio_s": clip_audio_s,  # clip length
            "median_s": median_s,  # what score.py uses for speed (RTFx)
            "runs_s": run_times_s,  # every run, in case one looks odd
        }
        (model_output_dir / f"{audio_path.stem}.json").write_text(json.dumps(clip_output))
        rtfx = clip_audio_s / median_s if median_s else float("inf")
        print(f"{audio_path.stem}: {clip_audio_s:.0f}s audio in {median_s:.1f}s (RTFx {rtfx:.0f})")

    # details about the whole run; score.py adds these to the summary table
    run_details = {
        "first_run_s": first_run_s,
        **peak_memory_gb(),
        "runs_per_clip": runs,
        "edit_twin_commit": edit_twin_commit(),
        "python": platform.python_version(),
        "run_at": datetime.now().isoformat(timespec="seconds"),
    }
    (hyps_dir / f"{model_name}.meta.json").write_text(json.dumps(run_details, indent=2))


def main():
    # sys.argv is the command split into words: ["src/run.py", "whisper-turbo"]
    model_name = sys.argv[1] if len(sys.argv) > 1 else ""
    if model_name not in MODELS:
        sys.exit(f"usage: python -m src.run <{'|'.join(MODELS)}>")
    transcribe_all(model_name, MODELS[model_name])

if __name__ == "__main__":
    main()
