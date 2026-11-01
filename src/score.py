"""
Score every model in data/hyps against the references in data/refs,
writing tables to results/metrics.

    uv run python -m src.score
"""

import json 
import statistics  # median of the timing errors
from pathlib import Path  # file paths that work like objects: REFS / "clip.txt"

import jiwer  # WER/CER and the word alignment everything else builds on
import pandas as pd  # turn the results into tables and save them as CSV

from src.textnorm import clean_text, filler_words, formatted_text, timed_tokens, verbatim_tokens

REFS = Path("data/refs")  # my hand-corrected transcripts: <clip>.txt and <clip>.words.txt
HYPS = Path("data/hyps")  # model outputs written by run.py: <model>/<clip>.json
RESULTS = Path("results/metrics")  # where summary.csv and per_clip.csv are saved


def matched_pairs(ref_toks: list[str], hyp_toks: list[str]) -> list[tuple[int, int]]:
    """Return a list of matched token indices between reference and hypothesis tokens.
    Each tuple is (ref_index, hyp_index), one per word the model got exactly right.
    """
    # Nothing to match if either side is empty (and jiwer errors on an empty reference)
    if not ref_toks or not hyp_toks:
        return []
    # Use jiwer to compute the alignment between reference and hypothesis tokens
    out = jiwer.process_words(" ".join(ref_toks), " ".join(hyp_toks))
    pairs = []
    # alignments has one chunk list per sentence; we passed one sentence, so take [0]
    for c in out.alignments[0]:
        # keep only stretches where the model wrote exactly my words
        # (skip "delete", "insert" and "substitute")
        if c.type == "equal":
            # turn the stretch into pairs, e.g. ref 2-3 and hyp 1-2 -> (2, 1), (3, 2)
            # (end indexes are not included, like Python's range)
            pairs += zip(range(c.ref_start_idx, c.ref_end_idx),
                         range(c.hyp_start_idx, c.hyp_end_idx))
    return pairs


def filler_counts(ref_toks: list[str], hyp_toks: list[str]) -> tuple[int, int, int]:
    """Count filler words in reference and hypothesis tokens.
    Returns a tuple of (fillers_caught, fillers_in_reference, fillers_model_wrote).
    Counts, not percentages, so they can be added up across clips before dividing.
    """
    # a filler is caught only if the model wrote the same filler in the same place
    # (True counts as 1 and False as 0, so sum() counts the True ones)
    caught = sum(ref_toks[ref_i] in filler_words
                 for ref_i, hyp_i in matched_pairs(ref_toks, hyp_toks))
    in_ref = sum(t in filler_words for t in ref_toks)  # fillers I actually said
    in_hyp = sum(t in filler_words for t in hyp_toks)  # fillers the model wrote
    return caught, in_ref, in_hyp


def timing_errors(ref_tt, hyp_tt) -> tuple[list[float], list[float]]:
    """Compute the timing errors for matched words between reference and hypothesis.
    Takes [(word, start, end), ...] for both sides.
    Returns (start_errors, end_errors), each a list of errors in milliseconds.
    """
    # pair up the words first (times are only compared for words the model got right)
    ref_words = [word for word, start, end in ref_tt]
    hyp_words = [word for word, start, end in hyp_tt]
    pairs = matched_pairs(ref_words, hyp_words)
    # [1] is the start time, [2] the end time; abs() makes early and late both count,
    # and * 1000 turns seconds into milliseconds
    starts = [abs(hyp_tt[hyp_i][1] - ref_tt[ref_i][1]) * 1000 for ref_i, hyp_i in pairs]
    ends = [abs(hyp_tt[hyp_i][2] - ref_tt[ref_i][2]) * 1000 for ref_i, hyp_i in pairs]
    return starts, ends


def load_labels(path: Path):
    """Audacity label export: start<TAB>end<TAB>word per line."""
    # split each line on tabs; skip blank lines and Audacity's optional "\" frequency lines
    rows = [line.split("\t") for line in path.read_text().splitlines()
            if line.strip() and not line.startswith("\\")]
    # reorder to the models' [word, start, end] format, then clean the words the same way
    return timed_tokens([[r[2], float(r[0]), float(r[1])] for r in rows])


def share_within(errors: list[float], ms: float):
    """Fraction of errors at or under `ms`, e.g. 0.8 = 80% of words within 50 ms.
    None when there are no timings (clips without a .words.txt)."""
    return sum(e <= ms for e in errors) / len(errors) if errors else None


def score_model(model_dir: Path) -> tuple[dict, list[dict]]:
    """Pooled metrics for one model (summary), plus one row per clip (per_clip)."""
    # the three text views of every clip (mine and the model's), collected so WER is
    # computed once over all clips together
    ref_verbatim_texts, hyp_verbatim_texts = [], []
    ref_clean_texts, hyp_clean_texts = [], []
    ref_formatted_texts, hyp_formatted_texts = [], []
    # running filler totals across clips
    total_fillers_caught = total_fillers_in_ref = total_fillers_in_hyp = 0
    # timing errors from all labelled clips
    all_start_errors, all_end_errors = [], []
    per_clip = []  # one row per clip for per_clip.csv
    # for speed (RTFx) and hallucination
    total_audio_s = total_processing_s = 0.0
    silence_word_count = total_silence_s = 0.0

    # one .json per clip; sorted so the order is the same every run
    for hyp_file in sorted(model_dir.glob("*.json")):
        clip = hyp_file.stem  # file name without .json, e.g. "vlog_intro_day_3"
        hyp = json.loads(hyp_file.read_text())  # what run.py saved
        hyp_timed_words = hyp["words"]  # [[word, start, end], ...]
        hyp_text = " ".join(word for word, start, end in hyp_timed_words)  # plain text
        total_audio_s += hyp["audio_s"]  # seconds of audio
        total_processing_s += hyp["median_s"]  # seconds the model took (median of the runs)

        # clips with no speech: anything the model writes there is invented
        if clip.startswith("silence_"):
            silence_word_count += len(hyp_timed_words)
            total_silence_s += hyp["audio_s"]
            continue  # nothing else to score for these
        ref_file = REFS / f"{clip}.txt"
        if not ref_file.exists():
            print(f"  {clip}: no reference yet, skipped")
            continue

        ref_text = ref_file.read_text()  # my transcript
        ref_tokens = verbatim_tokens(ref_text)  # my words, verbatim
        hyp_tokens = verbatim_tokens(hyp_text)  # the model's words, verbatim
        ref_verbatim_texts.append(" ".join(ref_tokens))
        hyp_verbatim_texts.append(" ".join(hyp_tokens))
        ref_clean_texts.append(clean_text(ref_text))
        hyp_clean_texts.append(clean_text(hyp_text))
        ref_formatted_texts.append(formatted_text(ref_text))
        hyp_formatted_texts.append(formatted_text(hyp_text))

        # add this clip's filler counts to the running totals
        fillers_caught, fillers_in_ref, fillers_in_hyp = filler_counts(ref_tokens, hyp_tokens)
        total_fillers_caught += fillers_caught
        total_fillers_in_ref += fillers_in_ref
        total_fillers_in_hyp += fillers_in_hyp
        # one row for per_clip.csv (an empty model output counts as 100% wrong)
        clip_wer = jiwer.wer(" ".join(ref_tokens), " ".join(hyp_tokens)) if hyp_tokens else 1.0
        per_clip.append({"model": model_dir.name, "clip": clip, "ref_words": len(ref_tokens),
                         "wer_verbatim": clip_wer, "fillers_ref": fillers_in_ref,
                         "fillers_caught": fillers_caught, "fillers_written": fillers_in_hyp})

        # word timings only exist for the clips I labelled in Audacity
        labels_file = REFS / f"{clip}.words.txt"
        if labels_file.exists():
            start_errors, end_errors = timing_errors(load_labels(labels_file),
                                                     timed_tokens(hyp_timed_words))
            all_start_errors += start_errors
            all_end_errors += end_errors

    if not ref_verbatim_texts:  # no clip had a reference, so there's nothing to report
        return {}, []
    # load time and memory saved by run.py, if it's there
    meta_file = HYPS / f"{model_dir.name}.meta.json"
    meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
    # each "x if y else None" avoids dividing by zero when there's no data for a metric
    summary = {
        "model": model_dir.name,
        "clips": len(ref_verbatim_texts),
        # lists in = pooled over all clips
        "wer_verbatim": jiwer.wer(ref_verbatim_texts, hyp_verbatim_texts),
        "wer_clean": jiwer.wer(ref_clean_texts, hyp_clean_texts),
        "wer_formatted": jiwer.wer(ref_formatted_texts, hyp_formatted_texts),
        "cer": jiwer.cer(ref_verbatim_texts, hyp_verbatim_texts),
        # real fillers it caught
        "filler_recall": (total_fillers_caught / total_fillers_in_ref
                          if total_fillers_in_ref else None),
        # its fillers that were real
        "filler_precision": (total_fillers_caught / total_fillers_in_hyp
                             if total_fillers_in_hyp else None),
        "start_median_ms": statistics.median(all_start_errors) if all_start_errors else None,
        "end_median_ms": statistics.median(all_end_errors) if all_end_errors else None,
        "start_within_50ms": share_within(all_start_errors, 50),
        "end_within_50ms": share_within(all_end_errors, 50),
        "end_within_100ms": share_within(all_end_errors, 100),
        "timed_words": len(all_start_errors),  # how many words the timing numbers are based on
        "hallucinated_wpm": (silence_word_count / (total_silence_s / 60)
                             if total_silence_s else None),
        # 30 = 30x faster than real time
        "rtfx": total_audio_s / total_processing_s if total_processing_s else None,
        **meta,  # adds first_run_s, peak_rss_gb, peak_gpu_gb
    }
    return summary, per_clip


def main():
    # parents=True also creates results/ if it's missing
    RESULTS.mkdir(parents=True, exist_ok=True)
    summaries, clips = [], []
    # every folder in data/hyps is one model, e.g. data/hyps/whisper-turbo
    for model_dir in sorted(p for p in HYPS.iterdir() if p.is_dir()):
        summary, per_clip = score_model(model_dir)
        if summary:
            summaries.append(summary)
            clips += per_clip
    if not summaries:
        raise SystemExit("nothing to score: run src.run first and add references")
    pd.DataFrame(summaries).to_csv(RESULTS / "summary.csv", index=False)  # one row per model
    pd.DataFrame(clips).to_csv(RESULTS / "per_clip.csv", index=False)  # one row per model per clip
    # print models side by side (.T flips rows and columns), rounded to 3 decimals
    print(pd.DataFrame(summaries).set_index("model").T.round(3).to_string())


if __name__ == "__main__":
    main()
