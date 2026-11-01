"""
Draft word timings for one clip from MY reference transcript, ready to check in Audacity.

It force-aligns data/refs/<clip>.txt to data/audio/<clip>.wav with a neutral aligner
(torchaudio's MMS_FA, wav2vec2), not any model under test, so no model gets a head start.
The output is a DRAFT: open it in Audacity and fix the boundaries by ear before trusting it.

Runs in its own throwaway environment so the benchmark's torch version is untouched:

    uv run --no-project --with torch==2.8.0 --with torchaudio==2.8.0 \
        --with whisper-normalizer python tools/draft_timings.py vlog_intro_day_1

Writes data/refs/<clip>.words.draft.txt (Audacity label format: start<TAB>end<TAB>word).
Rename it to <clip>.words.txt only after you've checked it, because score.py reads that name.
"""

import re
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `src` imports work

import torch
import torchaudio

from src.textnorm import verbatim_tokens

REFS = Path("data/refs")
AUDIO = Path("data/audio")


def load_wav(path: Path) -> torch.Tensor:
    """16 kHz mono .wav -> float tensor of shape (1, samples)."""
    with wave.open(str(path)) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1, "need 16 kHz mono"
        frames = w.readframes(w.getnframes())
    pcm = torch.frombuffer(bytearray(frames), dtype=torch.int16).float() / 32768
    return pcm.unsqueeze(0)


def main(clip: str):
    words = verbatim_tokens((REFS / f"{clip}.txt").read_text())
    # the aligner only knows letters and apostrophes; numbers ("2000s", "1") become a
    # wildcard that matches whatever audio is there
    align_words = ["*" if re.search(r"\d", w) else w for w in words]

    bundle = torchaudio.pipelines.MMS_FA
    model = bundle.get_model(with_star=True)
    dictionary = bundle.get_dict(star="*")
    waveform = load_wav(AUDIO / f"{clip}.wav")

    with torch.inference_mode():
        emission, _ = model(waveform)
    tokens = [dictionary[c] for w in align_words for c in w]
    aligned, scores = torchaudio.functional.forced_align(
        emission, torch.tensor([tokens], dtype=torch.int32), blank=0)
    spans = torchaudio.functional.merge_tokens(aligned[0], scores[0].exp())

    seconds_per_frame = waveform.shape[1] / emission.shape[1] / 16000
    out, i = [], 0
    for word, aw in zip(words, align_words):
        word_spans = spans[i:i + len(aw)]
        i += len(aw)
        start = word_spans[0].start * seconds_per_frame
        end = word_spans[-1].end * seconds_per_frame
        out.append(f"{start:.3f}\t{end:.3f}\t{word}")

    dest = REFS / f"{clip}.words.draft.txt"
    dest.write_text("\n".join(out) + "\n")
    print(f"{clip}: {len(out)} words -> {dest}")


if __name__ == "__main__":
    for name in sys.argv[1:]:
        main(name)
