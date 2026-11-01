"""
Write the data file the dashboard reads: results/metrics/*.csv -> docs/results.json.

This does NOT build the page. docs/index.html is written by hand; when it opens, it
loads docs/results.json and uses its two parts, "summary" (one row per model, from
summary.csv) and "per_clip" (one row per model per clip, from per_clip.csv). The
figures in the page's findings cards (46 -> 120 ms, the chunk table, ...) are typed
into index.html itself, because they come from one-off investigations, not the scorer.

GitHub Pages only serves the docs/ folder, so the page can't read results/metrics/
directly. The Pages workflow (.github/workflows/pages.yml) runs this on every push of
new results, so docs/results.json is never committed. To preview the page locally:

    uv run python tools/build_dashboard.py
    cd docs && python3 -m http.server 8000   # then open http://localhost:8000
"""

import json
import math
from pathlib import Path

import pandas as pd

RESULTS = Path("results/metrics")
DEST = Path("docs/results.json")


def clean(records: list[dict]) -> list[dict]:
    """NaN (an empty cell) isn't valid JSON, so turn it into null."""
    return [{k: (None if isinstance(v, float) and math.isnan(v) else v)
             for k, v in row.items()} for row in records]


def main():
    summary = pd.read_csv(RESULTS / "summary.csv").to_dict("records")
    per_clip = pd.read_csv(RESULTS / "per_clip.csv").to_dict("records")
    DEST.write_text(json.dumps({"summary": clean(summary), "per_clip": clean(per_clip)},
                               indent=1))
    print(f"{len(summary)} models, {len(per_clip)} clip rows -> {DEST}")


if __name__ == "__main__":
    main()
