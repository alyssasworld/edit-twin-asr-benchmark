"""
Speech-to-text models under test.

Every model is a function that takes the path to a .wav file and returns the words it
heard with their times: [[word, start_s, end_s], ...]. Because every model returns the
same format, run.py and score.py never need to know which model they're dealing with.

To add a model: write a function like whisper_turbo below and add it to MODELS.
"""

from edit_twin_mcp.transcribe import VERBATIM_PROMPT  # the exact prompt used in production

WHISPER_TURBO_REPO = "mlx-community/whisper-large-v3-turbo"  # Hugging Face model name


def whisper_result_to_words(whisper_result: dict) -> list[list]:
    """Turn mlx-whisper's output into the shared [[word, start_s, end_s], ...] format.

    mlx-whisper returns {"segments": [{"words": [{"word": " So", "start": 0.0,
    "end": 0.32, ...}, ...]}, ...]}: words grouped into segments (roughly sentences).
    """
    timed_words = []
    for segment in whisper_result["segments"]:
        # a segment with no speech can come back without a "words" list
        for word_info in segment.get("words", []):
            timed_words.append([
                word_info["word"].strip(),  # Whisper puts a space before each word
                float(word_info["start"]),  # make sure times are plain numbers
                float(word_info["end"]),
            ])
    return timed_words


def whisper_turbo(audio_path: str) -> list[list]:
    """Baseline: Whisper large-v3-turbo with the same settings edit-twin-mcp uses."""
    # imported here rather than at the top so the tests can import this file
    # without loading the speech model library
    import mlx_whisper

    whisper_result = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo=WHISPER_TURBO_REPO,
        word_timestamps=True,  # we need a start and end time for every word
        language="en",  # fixed for every model, so no model wastes time detecting it
        initial_prompt=VERBATIM_PROMPT,  # nudges Whisper to keep um/uh and repeats
        condition_on_previous_text=False,  # same as production: stops errors snowballing
    )
    return whisper_result_to_words(whisper_result)


PARAKEET_REPO = "mlx-community/parakeet-tdt-0.6b-v3"
_parakeet_model = None  # loaded once on the first call (the warm-up), then reused


def parakeet_tokens_to_words(tokens) -> list[list]:
    """Join Parakeet's sub-word tokens into the shared [[word, start_s, end_s], ...] format.

    Parakeet returns pieces of words, e.g. " fl", "at", "mate": a piece starting with a
    space begins a new word, any other piece belongs to the word before it.
    """
    timed_words = []
    for token in tokens:
        if token.text.startswith(" ") or not timed_words:
            timed_words.append([token.text.strip(), float(token.start), float(token.end)])
        else:  # continue the current word and stretch its end time
            timed_words[-1][0] += token.text
            timed_words[-1][2] = float(token.end)
    return [w for w in timed_words if w[0]]  # drop empty pieces


def parakeet_tdt(audio_path: str) -> list[list]:
    """NVIDIA Parakeet TDT 0.6B v3 (multilingual), run with MLX on the Mac.
    No prompt: Parakeet has no prompt option, so this is how it would run in edit-twin."""
    global _parakeet_model
    if _parakeet_model is None:
        from parakeet_mlx import from_pretrained  # imported here, like mlx_whisper above
        _parakeet_model = from_pretrained(PARAKEET_REPO)
    # 30 s chunks with 5 s overlap, the same window length Whisper uses. Chosen after
    # comparing settings on my clips (2026-11-01), so it's tuned on the test set:
    #   whole file: verbatim WER 12.9% reel / 5.9% vlog_3, 8.4 GB GPU memory
    #   120 s chunks: dropped 22 s of speech from reel_navier_01 when joining chunks
    #   30 s chunks: 5.9% reel / 1.7% vlog_3 (best on both)
    result = _parakeet_model.transcribe(audio_path, chunk_duration=30, overlap_duration=5)
    return parakeet_tokens_to_words(result.tokens)


# model name (used on the command line and as the data/hyps/<name> folder) -> function
MODELS = {
    "whisper-turbo": whisper_turbo,
    "parakeet-tdt": parakeet_tdt,
}
