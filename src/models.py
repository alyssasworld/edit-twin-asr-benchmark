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


QWEN_ASR_REPO = "mlx-community/Qwen3-ASR-1.7B-bf16"  # bf16: full precision, like the others
QWEN_ALIGNER_REPO = "mlx-community/Qwen3-ForcedAligner-0.6B-bf16"
QWEN_CHUNK_S = 30  # longest piece of audio handled at once (the aligner's limit is ~5 min)
# Loop guard. Qwen writes text like a chatbot, one token after another, and on silence
# it can get stuck repeating ("oh, oh, oh, ..." 4,096 times, 190 s on a 24 s piece).
# Real speech on my clips needed at most 3.9 tokens per second of audio, so each piece
# is allowed 10 per second; output that uses up the whole allowance never stopped by
# itself, so it's a loop and is thrown away. (Whisper's own loop rule, text compressing
# more than 2.4x, would also delete my genuine retakes: those score up to 2.78.)
QWEN_TOKENS_PER_S = 10
_qwen_models = {}  # {"asr": ..., "aligner": ...}, loaded once on the first call


def attach_times(text: str, aligned_items, is_word) -> list[list]:
    """Pair each word of `text` (punctuation kept) with the aligner's times.

    The aligner splits text on spaces and strips punctuation, so it returns "today" for
    "today."; pieces that are only punctuation (e.g. "-") get no time at all. Walking
    both lists together keeps the written form, which formatted WER needs.
    `is_word(piece)` says whether the aligner kept that piece.
    """
    pieces = [p for p in text.split() if is_word(p)]
    if len(pieces) != len(aligned_items):  # should never happen; fail loudly if it does
        raise ValueError(f"{len(pieces)} words but {len(aligned_items)} timings")
    return [[piece, float(item.start_time), float(item.end_time)]
            for piece, item in zip(pieces, aligned_items)]


def qwen3_asr(audio_path: str) -> list[list]:
    """Alibaba Qwen3-ASR 1.7B, run with MLX (mlx-audio) on the Mac.

    Qwen3-ASR only outputs text, so word times come from Qwen's own companion model,
    Qwen3-ForcedAligner, as Qwen recommends. Both models are timed together.
    No prompt, so this is how it would run out of the box.
    """
    if not _qwen_models:
        from mlx_audio.stt import load  # imported here, like the other models
        _qwen_models["asr"] = load(QWEN_ASR_REPO)
        _qwen_models["aligner"] = load(QWEN_ALIGNER_REPO)
    asr, aligner = _qwen_models["asr"], _qwen_models["aligner"]
    from mlx_audio.stt.models.qwen3_asr.qwen3_asr import split_audio_into_chunks
    from mlx_audio.stt.utils import load_audio
    sample_rate = 16000
    audio = load_audio(audio_path)  # 16 kHz samples
    is_word = lambda p: bool(aligner.aligner_processor.clean_token(p))

    timed_words = []
    # 1) cut the clip into pieces of up to 30 s, at quiet moments (Qwen's own splitter)
    for piece, start in split_audio_into_chunks(audio, sample_rate,
                                                chunk_duration=QWEN_CHUNK_S):
        # 2) transcribe the piece with a token allowance; drop it if it never stopped
        allowance = int(QWEN_TOKENS_PER_S * len(piece) / sample_rate) + 20
        result = asr.generate(piece, language="English", max_tokens=allowance)
        text = result.text.strip()
        if not text or result.generation_tokens >= allowance:
            continue
        # 3) align the piece's text to the same piece of audio, then shift the times
        #    from "seconds into the piece" to "seconds into the clip"
        aligned = aligner.generate(audio=piece, text=text, language="English")
        start = float(start)  # plain number (it comes back as a numpy float)
        for word, s, e in attach_times(text, aligned.items, is_word):
            timed_words.append([word, start + s, start + e])
    return timed_words


# model name (used on the command line and as the data/hyps/<name> folder) -> function
MODELS = {
    "whisper-turbo": whisper_turbo,
    "parakeet-tdt": parakeet_tdt,
    "qwen3-asr": qwen3_asr,
}
