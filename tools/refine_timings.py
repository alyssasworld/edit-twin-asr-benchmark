"""
Tighten drafted word timings using the waveform itself.

The aligner (draft_timings.py) often makes words too short: a word "ends" while my voice
is still sounding. This script fixes that the way you would by eye in Audacity:
  - a word's END is pushed right while the audio is still loud, until the sound falls
    back to the room's noise level (but never past the next word's start);
  - a word's START is pulled left while the audio is still loud, until it reaches the
    quiet before it (but never before the previous word's end).
Edges inside fast, connected speech have no quiet to snap to, so they barely move.

    uv run python tools/refine_timings.py vlog_intro_day_1 [...]

Reads and rewrites data/refs/<clip>.words.draft.txt (the original aligner output is
kept as <clip>.words.aligned.txt). Still a draft: spot-check it before trusting it.
Add --plot to save a picture of the waveform and labels to data/refs/<clip>.draft.png.
"""

import shutil
import sys
import wave
from pathlib import Path

import numpy as np

REFS = Path("data/refs")
AUDIO = Path("data/audio")
FRAME_S = 0.01  # loudness is measured in 10 ms steps
MAX_MOVE_S = 0.3  # never move an edge more than this (guards against runaway snaps)
ABOVE_FLOOR_DB = 15  # "sound" = this much louder than the room's noise floor


def loudness_db(audio_path: Path) -> np.ndarray:
    """Loudness of every 10 ms frame, in dB."""
    with wave.open(str(audio_path)) as w:
        rate = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16) / 32768
    hop = int(rate * FRAME_S)
    frames = pcm[: len(pcm) // hop * hop].reshape(-1, hop)
    return 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-9)


def refine(labels: list[tuple[float, float, str]], db: np.ndarray):
    voiced = db > np.percentile(db, 10) + ABOVE_FLOOR_DB  # True where there's sound
    last = len(voiced) - 1
    max_move = int(MAX_MOVE_S / FRAME_S)
    out = []
    for i, (start, end, word) in enumerate(labels):
        prev_end = out[-1][1] if out else 0.0
        next_start = labels[i + 1][0] if i + 1 < len(labels) else last * FRAME_S

        # END: walk right while there's still sound, stopping at the next word
        f = min(int(end / FRAME_S), last)
        limit = min(int(next_start / FRAME_S), f + max_move, last)
        while f < limit and voiced[f]:
            f += 1
        new_end = max(end, f * FRAME_S)

        # START: walk left while there's still sound, stopping at the previous word
        f = int(start / FRAME_S)
        limit = max(int(np.ceil(prev_end / FRAME_S)), f - max_move, 0)
        while f > limit and voiced[f - 1]:
            f -= 1
        new_start = min(start, f * FRAME_S)

        out.append((new_start, new_end, word))
    return out


def plot(clip: str, db: np.ndarray, labels, raw, seconds=(0, 12)):
    import matplotlib.pyplot as plt
    t = np.arange(len(db)) * FRAME_S
    lo, hi = seconds
    fig, ax = plt.subplots(figsize=(22, 4))
    m = (t >= lo) & (t <= hi)
    ax.plot(t[m], db[m], color="black", lw=0.6)
    for (s, e, w), (rs, re_, _) in zip(labels, raw):
        if e < lo or s > hi:
            continue
        ax.axvspan(rs, re_, ymin=0.0, ymax=0.08, color="tab:blue", alpha=0.6)
        ax.axvspan(s, e, ymin=0.1, ymax=1.0, color="tab:orange", alpha=0.25)
        ax.text((s + e) / 2, ax.get_ylim()[1], w, ha="center", va="bottom", fontsize=8)
    ax.set_xlim(lo, hi)
    ax.set_xlabel("seconds  (orange = refined label, blue strip = original aligner label)")
    ax.set_ylabel("loudness dB")
    dest = REFS / f"{clip}.draft.png"
    fig.savefig(dest, dpi=110, bbox_inches="tight")
    print(f"  plot -> {dest}")


def main(clip: str, make_plot: bool):
    draft = REFS / f"{clip}.words.draft.txt"
    aligned = REFS / f"{clip}.words.aligned.txt"
    if not aligned.exists():  # keep the untouched aligner output once
        shutil.copy(draft, aligned)
    raw = [(float(s), float(e), w) for s, e, w in
           (line.split("\t") for line in aligned.read_text().splitlines() if line.strip())]
    db = loudness_db(AUDIO / f"{clip}.wav")
    labels = refine(raw, db)
    draft.write_text("".join(f"{s:.3f}\t{e:.3f}\t{w}\n" for s, e, w in labels))
    short = sum(e - s < 0.08 for s, e, _ in labels)
    moved = np.median([abs(e - re_) for (_, e, _), (_, re_, _) in zip(labels, raw)]) * 1000
    print(f"{clip}: {len(labels)} words, median end moved {moved:.0f} ms, "
          f"{short} still shorter than 80 ms -> {draft}")
    if make_plot:
        plot(clip, db, labels, raw)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--plot"]
    for name in args:
        main(name, "--plot" in sys.argv)
