"""Check the scoring on tiny examples where the right answer is known."""
import json

import pytest

import src.score as score
from src.score import filler_counts, load_labels, matched_pairs, share_within, timing_errors

# what I said, and a model that left out both fillers
REF_TOKENS = ["so", "um", "i", "was", "uh", "thinking"]
HYP_TOKENS = ["so", "i", "was", "thinking"]


# --- matched_pairs ---------------------------------------------------------------

def test_matched_pairs_skips_dropped_fillers():
    # the model skipped "um", so its words shift back by one: my "i" (2) is its 1
    assert matched_pairs(REF_TOKENS, HYP_TOKENS) == [(0, 0), (2, 1), (3, 2), (5, 3)]


def test_matched_pairs_empty_model_output():
    assert matched_pairs(REF_TOKENS, []) == []


def test_matched_pairs_wrong_word_is_not_a_match():
    # "was" -> "is" is a substitution, so position 1 doesn't pair up
    assert matched_pairs(["i", "was", "here"], ["i", "is", "here"]) == [(0, 0), (2, 2)]


# --- filler_counts ---------------------------------------------------------------
# returns (fillers caught, fillers in my reference, fillers the model wrote)

def test_filler_counts_when_fillers_dropped():
    assert filler_counts(REF_TOKENS, HYP_TOKENS) == (0, 2, 0)


def test_filler_counts_when_fillers_kept():
    assert filler_counts(REF_TOKENS, REF_TOKENS) == (2, 2, 2)


def test_filler_in_wrong_place_is_not_caught():
    # I said "um" at the start; the model wrote it at the end instead
    assert filler_counts(["um", "so", "i", "was"], ["so", "i", "was", "um"]) == (0, 1, 1)


def test_filler_one_word_off_still_lines_up():
    # a filler only one word out of place can still be lined up by the alignment,
    # so it counts as caught (jiwer pairs the two "um"s and marks "i" as an error)
    assert filler_counts(["so", "um", "i"], ["so", "i", "um"]) == (1, 1, 1)


# --- timing helpers --------------------------------------------------------------

def test_timing_errors_only_for_matched_words():
    ref_timed = [("so", 0.0, 0.2), ("um", 0.3, 0.5), ("thinking", 0.6, 1.0)]
    hyp_timed = [("so", 0.02, 0.25), ("thinking", 0.6, 0.9)]
    start_errors, end_errors = timing_errors(ref_timed, hyp_timed)
    # "um" was missed, so only "so" and "thinking" are compared (in milliseconds)
    assert start_errors == pytest.approx([20, 0])
    assert end_errors == pytest.approx([50, 100])


def test_share_within():
    assert share_within([10, 40, 60, 200], 50) == 0.5  # 2 of 4 errors are 50 ms or less
    assert share_within([], 50) is None  # no timings -> no number


def test_load_labels(tmp_path):
    # tmp_path is a temporary folder pytest makes for this test
    labels_file = tmp_path / "clip.words.tsv"
    labels_file.write_text("0.0\t0.2\tSo,\n0.3\t0.5\tum\n")
    # words are cleaned the same way as the model's ("So," -> "so")
    assert load_labels(labels_file) == [("so", 0.0, 0.2), ("um", 0.3, 0.5)]


# --- score_model (the whole pipeline on fake files) ------------------------------

@pytest.fixture
def fake_data(tmp_path, monkeypatch):
    """A tiny data/ folder: one clip with a reference and timings, one silence clip."""
    refs_dir = tmp_path / "refs"
    model_dir = tmp_path / "hyps" / "fake-model"
    refs_dir.mkdir()
    model_dir.mkdir(parents=True)
    # point score.py at the fake folders instead of the real data/ folder
    monkeypatch.setattr(score, "REFS", refs_dir)
    monkeypatch.setattr(score, "HYPS", tmp_path / "hyps")

    # my transcript and word timings
    (refs_dir / "clip1.txt").write_text("So um I was, uh, thinking.")
    (refs_dir / "clip1.words.tsv").write_text(
        "0.0\t0.2\tSo\n0.3\t0.5\tum\n0.6\t0.7\tI\n0.7\t0.9\twas\n1.0\t1.2\tuh\n1.3\t1.8\tthinking\n")
    # the model: dropped both fillers and the comma, ended "thinking" 100 ms late
    hyp_timed_words = [["So", 0.0, 0.2], ["I", 0.6, 0.7], ["was", 0.7, 0.9], ["thinking.", 1.3, 1.9]]
    (model_dir / "clip1.json").write_text(json.dumps(
        {"words": hyp_timed_words, "audio_s": 6.0, "median_s": 2.0}))
    # 30 s of room tone where the model invented 2 words
    (model_dir / "silence_roomtone.json").write_text(json.dumps(
        {"words": [["you", 1.0, 1.2], ["know", 1.2, 1.4]], "audio_s": 30.0, "median_s": 1.0}))
    return model_dir


def test_score_model_summary(fake_data):
    summary, per_clip = score.score_model(fake_data)
    assert summary["clips"] == 1  # the silence clip isn't counted as a scored clip
    assert summary["wer_verbatim"] == pytest.approx(2 / 6)  # 2 fillers missing of 6 words
    assert summary["wer_clean"] == 0  # fillers and punctuation don't count here
    assert summary["wer_formatted"] == pytest.approx(3 / 6)  # + "was," written as "was"
    assert summary["filler_recall"] == 0  # caught 0 of my 2 fillers
    assert summary["filler_precision"] is None  # it wrote no fillers, so nothing to check
    assert summary["timed_words"] == 4  # so, I, was, thinking
    assert summary["start_median_ms"] == pytest.approx(0)
    assert summary["end_within_50ms"] == 0.75  # 3 of 4 word ends within 50 ms
    assert summary["hallucinated_wpm"] == pytest.approx(4)  # 2 words in 0.5 min
    assert summary["rtfx"] == pytest.approx(12)  # 36 s of audio in 3 s


def test_score_model_per_clip_row(fake_data):
    summary, per_clip = score.score_model(fake_data)
    assert per_clip == [{"model": "fake-model", "clip": "clip1", "ref_words": 6,
                         "wer_verbatim": pytest.approx(2 / 6), "fillers_ref": 2,
                         "fillers_caught": 0, "fillers_written": 0}]


def test_score_model_without_references(fake_data):
    (score.REFS / "clip1.txt").unlink()  # delete my transcript
    assert score.score_model(fake_data) == ({}, [])
