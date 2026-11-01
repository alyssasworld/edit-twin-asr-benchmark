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


# model name (used on the command line and as the data/hyps/<name> folder) -> function
MODELS = {
    "whisper-turbo": whisper_turbo,
}
