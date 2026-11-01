"""Check the model wrappers.

The fast tests use fake model output. The slow test runs the real Whisper model on one
of my clips; it's skipped by default, run it with:  uv run pytest -m slow
"""
from pathlib import Path

import pytest

from src.models import MODELS, whisper_result_to_words, whisper_turbo

SHORTEST_CLIP = Path("data/audio/vlog_intro_day_1.wav")  # 16 s, quick to transcribe


def test_whisper_result_to_words_flattens_segments():
    # the shape mlx-whisper returns: words grouped into segments
    fake_whisper_result = {"segments": [
        {"words": [{"word": " So", "start": 0.0, "end": 0.32, "probability": 0.9},
                   {"word": " um,", "start": 0.4, "end": 0.6, "probability": 0.5}]},
        {"words": [{"word": " thinking.", "start": 1.0, "end": 1.5, "probability": 0.8}]},
    ]}
    assert whisper_result_to_words(fake_whisper_result) == [
        ["So", 0.0, 0.32], ["um,", 0.4, 0.6], ["thinking.", 1.0, 1.5]]


def test_whisper_result_to_words_skips_segments_without_words():
    fake_whisper_result = {"segments": [{"text": ""}, {"words": []}]}
    assert whisper_result_to_words(fake_whisper_result) == []


def test_every_model_is_a_function():
    # run.py calls MODELS[name](path), so every entry must be callable
    assert MODELS  # at least one model
    for model_name, transcribe in MODELS.items():
        assert callable(transcribe), model_name


@pytest.mark.slow
@pytest.mark.skipif(not SHORTEST_CLIP.exists(), reason="needs my clip in data/audio")
def test_whisper_turbo_on_a_real_clip():
    timed_words = whisper_turbo(str(SHORTEST_CLIP))
    assert len(timed_words) > 10  # the clip has ~30 words
    for word, start_s, end_s in timed_words:
        assert isinstance(word, str) and word  # a non-empty word
        assert 0 <= start_s <= end_s  # times make sense
