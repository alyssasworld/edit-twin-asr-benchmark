"""Purpose: Turn transcripts into comparable word lists (see docs/metrics.md).
  WHY?
  - ASR model outputs can vary in punctuation, case, and filler spellings.
  i.e. I said: "So, um, I think it's uh, you know, a good idea."
  meanwhile Whisper can give: So, umm... I think it's uhm, you know, a good idea."

  We want a comparabile representation of the words, so we can compute metrics like WER and CER.
"""

import re
from whisper_normalizer.english import EnglishTextNormalizer # OpenAI's standard text cleaner for English from the whisper-normalizer package

filler_words = {"uh", "um", "hmm"}
# Map alternative filler spellings to a canonical form, so we can compute metrics properly
spellings = {"umm": "um", "uhm": "um", "erm": "um", "uhh": "uh", "er": "uh",
             "mm": "hmm", "hm": "hmm", "hmmm": "hmm"}

# Initialise the English text normaliser from the whisper-normalizer package
_clean = EnglishTextNormalizer()


def verbatim_tokens(text: str) -> list[str]:
    """Every word turned lowercase, including fillers, with no punctuation or capitals
      To be used for: verbatim WER, CER and filler recall metrics.
      Limitations: Does not currently take into account of accented letters, and splits hyphenated words into two tokens."
    """
    text = text.lower().replace("\u2019", "'") # normalise apostrophes

    # Finds all words (alphanumeric + apostrophes) and replaces any alternative filler spellings with the canonical form
    return [spellings.get(w,w) for w in re.findall(r"[a-z0-9']+", text)]

def clean_text(text: str) -> str:
    """Removes fillers, US spelling.
      To be used for: clean WER (compariblae to public numbers)
    """
    # Unify filler spellings first
    text = re.sub(r"[A-Za-z0-9']+", lambda m: spellings.get(m.group().lower(), m.group()), text)
    
    return _clean(text)

def formatted_text(text: str) -> str:
    """The text as written, punctuation and capitals preserved
      To be used for: formatted WER
    """
    return " ".join(text.split())

def timed_tokens(words: list) -> list[tuple[str, float, float]]:
    """ Return a list of (token, start_time, end_time) tuples
      Input: Model output format: (word, start_time, end_time)
      Output: [(token, start_time, end_time), ...]

      I.e. zero tokens if the word is a filler or punctuation, otherwise one token per word, with the same start and end times as the original word.
      To be used for: timestamp metrics
    """
    return [(t,s,e) for w,s,e in words for t in verbatim_tokens(w)]
            
# Quick test run commands
# uv run python -c "from src.textnorm import verbatim_tokens; print(verbatim_tokens('So, umm, I love the colour!'))"
# uv run python -c "from src.textnorm import clean_text; print(clean_text('So, umm, I love the colour!'))"
# uv run python -c "from src.textnorm import formatted_text; print(formatted_text('So, umm, I love the colour!'))"
# uv run python -c "from src.textnorm import timed_tokens; print(timed_tokens([['So,', 0.0, 0.2], ['…', 0.2, 0.3], ['well-known', 0.3, 0.8]]))"