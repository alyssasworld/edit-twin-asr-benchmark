"""
Score every model in data/hyps against the references in results/
uv run python -m src.score
"""

import json
import statistics
from pathlib import Path
import jiwer
import pandas as pd
from src.textnorm import clean_text, filler_words, formatted_text, timed_tokens, verbatim_tokens

REFS = Path("results") # Reference transcripts - using past works from open ~/"Movies/Edit Twin/.transcripts/"
HYPS = Path("data/hyps")
RESULTS = Path("results/metrics")

