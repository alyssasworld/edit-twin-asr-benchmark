"""
Check edit-twin-mcp's own Parakeet code against the benchmark, and measure what its
corrections (vocabulary + fixes) change.

1. Same words? Runs edit_twin_mcp.transcribe.transcribe_file(engine="parakeet") on every
   clip and compares its words and times with data/hyps/parakeet-tdt/ (the benchmark's
   Parakeet run). They should match, which shows the shipped code is the code that was
   scored.
2. Corrections. If data/vocabulary.txt exists (one name per line, git-ignored like the
   rest of data/), applies edit-twin's `correct` with it and prints verbatim, formatted
   WER and CER before and after. NOTE: a vocabulary written from these clips is tuned on
   the test set, so read the "after" numbers as a demo of the mechanism, not a result.

    uv run python -m tools.check_edit_twin
"""

import json
import sys
from pathlib import Path

import jiwer

from edit_twin_mcp import transcribe as tr
from src.textnorm import formatted_text, verbatim_tokens

AUDIO = Path("data/audio")
HYPS = Path("data/hyps/parakeet-tdt")
REFS = Path("data/refs")
CACHE = Path("data/edit_twin_cache")
VOCAB = Path("data/vocabulary.txt")


def flat(data: dict) -> list[list]:
    return [[w["w"], w["s"], w["e"]] for s in data["segments"] for w in s["words"]]


def main():
    vocab = [l.strip() for l in VOCAB.read_text().splitlines() if l.strip()] if VOCAB.exists() else []
    same = True
    texts = {"ref": [], "raw": [], "fixed": []}
    changes = []
    for audio in sorted(AUDIO.glob("*.wav")):
        clip = audio.stem
        data = tr.transcribe_file(str(audio), str(CACHE), engine="parakeet")
        got = flat(data)
        want = json.loads((HYPS / f"{clip}.json").read_text())["words"]
        words_match = [w[0] for w in got] == [w[0] for w in want]
        max_dt = max((abs(a[i] - b[i]) for a, b in zip(got, want) for i in (1, 2)), default=0.0)
        print(f"{clip}: {len(got)} words, same words: {words_match}, max time difference {max_dt * 1000:.0f} ms")
        same &= words_match and max_dt < 0.001
        ref_file = REFS / f"{clip}.txt"
        if vocab and ref_file.exists():
            fixed = tr.correct(data, vocab)
            changes += [(clip, c) for c in fixed.get("corrections", [])]
            texts["ref"].append(ref_file.read_text())
            texts["raw"].append(" ".join(w[0] for w in got))
            texts["fixed"].append(" ".join(w[0] for w in flat(fixed)))
    print("\nedit-twin's Parakeet matches the benchmark run" if same
          else "\n!! edit-twin's Parakeet output differs from the benchmark run")

    if vocab:
        print(f"\nVocabulary ({len(vocab)}): {', '.join(vocab)}")
        for clip, c in changes:
            print(f"  {clip} @ {c['at']:.1f}s: {c['from']!r} -> {c['to']!r}")
        ref_v = [" ".join(verbatim_tokens(t)) for t in texts["ref"]]
        ref_f = [formatted_text(t) for t in texts["ref"]]
        for name in ("raw", "fixed"):
            hyp_v = [" ".join(verbatim_tokens(t)) for t in texts[name]]
            hyp_f = [formatted_text(t) for t in texts[name]]
            print(f"  {name:5}  verbatim WER {jiwer.wer(ref_v, hyp_v):.2%}  "
                  f"formatted WER {jiwer.wer(ref_f, hyp_f):.2%}  CER {jiwer.cer(ref_v, hyp_v):.2%}")
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
