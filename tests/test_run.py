"""Check run.py with a fake model, so no real speech model is needed."""
import json
import wave

import pytest

from src.run import audio_duration_s, timed_transcription, transcribe_all

FAKE_WORDS = [["So", 0.0, 0.3], ["um", 0.4, 0.6]]


def write_silent_wav(path, seconds: float, sample_rate: int = 16000) -> None:
    """Make a silent 16 kHz mono .wav file of the given length."""
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)  # mono
        wav_file.setsampwidth(2)  # 16-bit samples
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * int(seconds * sample_rate))  # all zeros = silence


class FakeModel:
    """Stands in for a speech model: returns FAKE_WORDS and remembers each call."""

    def __init__(self):
        self.calls = []  # the audio paths it was asked to transcribe

    def __call__(self, audio_path: str) -> list:
        self.calls.append(audio_path)
        return FAKE_WORDS


def test_audio_duration_s(tmp_path):
    audio_path = tmp_path / "clip.wav"
    write_silent_wav(audio_path, seconds=2.5)
    assert audio_duration_s(audio_path) == pytest.approx(2.5)


def test_timed_transcription_runs_the_model_n_times():
    fake_model = FakeModel()
    timed_words, run_times_s = timed_transcription(fake_model, "clip.wav", runs=3)
    assert timed_words == FAKE_WORDS
    assert len(run_times_s) == 3  # one time per run
    assert len(fake_model.calls) == 3


@pytest.fixture
def audio_dir(tmp_path):
    """A fake data/audio folder with two short clips."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    write_silent_wav(audio_dir / "clip_a.wav", seconds=1.0)
    write_silent_wav(audio_dir / "clip_b.wav", seconds=2.0)
    return audio_dir


def test_transcribe_all_writes_one_json_per_clip(tmp_path, audio_dir):
    hyps_dir = tmp_path / "hyps"
    transcribe_all("fake-model", FakeModel(), audio_dir=audio_dir, hyps_dir=hyps_dir, runs=2)

    clip_output = json.loads((hyps_dir / "fake-model" / "clip_b.json").read_text())
    assert clip_output["words"] == FAKE_WORDS
    assert clip_output["audio_s"] == pytest.approx(2.0)
    assert len(clip_output["runs_s"]) == 2
    # these are the keys score.py reads
    assert {"words", "audio_s", "median_s"} <= clip_output.keys()
    assert sorted(p.name for p in (hyps_dir / "fake-model").iterdir()) == ["clip_a.json", "clip_b.json"]


def test_transcribe_all_warms_up_first(tmp_path, audio_dir):
    fake_model = FakeModel()
    transcribe_all("fake-model", fake_model, audio_dir=audio_dir, hyps_dir=tmp_path / "hyps", runs=2)
    # 1 warm-up call + 2 clips x 2 runs
    assert len(fake_model.calls) == 1 + 2 * 2


def test_transcribe_all_writes_run_details(tmp_path, audio_dir):
    hyps_dir = tmp_path / "hyps"
    transcribe_all("fake-model", FakeModel(), audio_dir=audio_dir, hyps_dir=hyps_dir, runs=1)
    run_details = json.loads((hyps_dir / "fake-model.meta.json").read_text())
    assert run_details["runs_per_clip"] == 1
    assert run_details["first_run_s"] >= 0
    assert run_details["peak_rss_gb"] > 0
    assert "edit_twin_commit" in run_details


def test_transcribe_all_stops_when_there_is_no_audio(tmp_path):
    empty_audio_dir = tmp_path / "audio"
    empty_audio_dir.mkdir()
    with pytest.raises(SystemExit):  # sys.exit() raises SystemExit
        transcribe_all("fake-model", FakeModel(), audio_dir=empty_audio_dir, hyps_dir=tmp_path / "hyps")
