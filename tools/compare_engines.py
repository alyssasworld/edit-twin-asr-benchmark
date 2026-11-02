"""
Before/after page for demos: Whisper (what edit-twin used) vs Parakeet (what it uses
now, with the vocabulary corrections), clip by clip, against my reference transcripts.

For every clip it shows each um/uh/hmm I said and whether each engine caught it, and both
transcripts side by side with word errors, missed words and name corrections marked.
Click any word or filler to hear that moment of the clip.

Writes data/compare/index.html. It plays the private clips from data/audio/ and shows my
private transcripts, so it lives in data/ (git-ignored): open it locally, don't publish it.

    uv run python -m tools.compare_engines        # needs data/hyps/whisper-turbo first
    open data/compare/index.html
"""

import html
import json
from pathlib import Path

import jiwer

from edit_twin_mcp import transcribe as tr
from src.textnorm import filler_words, verbatim_tokens

AUDIO = Path("data/audio")
REFS = Path("data/refs")
WHISPER = Path("data/hyps/whisper-turbo")
CACHE = Path("data/edit_twin_cache")
VOCAB = Path("data/vocabulary.txt")
OUT = Path("data/compare/index.html")


def parakeet_words(audio: Path, vocab: list[str]) -> tuple[list[list], set[float]]:
    """edit-twin's own Parakeet transcript with the vocabulary applied, plus the start
    times of the words that were corrected."""
    data = tr.correct(tr.transcribe_file(str(audio), str(CACHE), engine="parakeet"), vocab)
    words = [[w["w"], w["s"], w["e"]] for s in data["segments"] for w in s["words"]]
    return words, {c["at"] for c in data.get("corrections", [])}


def align(ref_toks: list[str], words: list[list]) -> dict:
    """Mark each of the model's words ok / wrong / extra, find the reference words it
    left out, and which reference fillers it caught (and when)."""
    toks, owner = [], []  # every token, and which displayed word it came from
    for i, w in enumerate(words):
        for t in verbatim_tokens(w[0]):
            toks.append(t)
            owner.append(i)
    status = ["ok"] * len(words)
    missing = {}  # word index -> reference words left out just before it
    caught = {}  # reference token index -> time the model wrote it
    out = jiwer.process_words(" ".join(ref_toks), " ".join(toks)) if toks else None
    errors = len(ref_toks) if out is None else out.substitutions + out.deletions + out.insertions
    for c in (out.alignments[0] if out else []):
        if c.type == "equal":
            for r, h in zip(range(c.ref_start_idx, c.ref_end_idx), range(c.hyp_start_idx, c.hyp_end_idx)):
                if ref_toks[r] in filler_words:
                    caught[r] = words[owner[h]][1]
        elif c.type in ("substitute", "insert"):
            for h in range(c.hyp_start_idx, c.hyp_end_idx):
                status[owner[h]] = "wrong" if c.type == "substitute" else "extra"
        else:  # delete: show the left-out words just before the next word the model wrote
            at = owner[c.hyp_start_idx] if c.hyp_start_idx < len(toks) else len(words)
            missing.setdefault(at, []).extend(ref_toks[c.ref_start_idx:c.ref_end_idx])
    return {"status": status, "missing": missing, "caught": caught, "errors": errors}


def word_html(words, a, fixed=frozenset()) -> str:
    parts = []
    for i, (w, s, _e) in enumerate(words + [["", None, None]]):
        if i in a["missing"]:
            parts.append(f'<del title="left out">{html.escape(" ".join(a["missing"][i]))}</del>')
        if s is None:
            break
        cls = [a["status"][i]]
        if verbatim_tokens(w) and verbatim_tokens(w)[0] in filler_words:
            cls.append("filler")
        if round(s, 2) in fixed:
            cls.append("fixed")
        tip = {"wrong": "misheard", "extra": "not said"}.get(a["status"][i], "")
        if "fixed" in cls:
            tip = "spelling corrected from the vocabulary"
        parts.append(f'<span class="{" ".join(cls)}" data-t="{s:.2f}" title="{tip}">{html.escape(w)}</span>')
    return " ".join(parts)


def clip_section(clip: str, ref_text: str, whisper, parakeet, fixed) -> tuple[str, dict]:
    ref = verbatim_tokens(ref_text)
    aw, ap = align(ref, whisper), align(ref, parakeet)
    fillers = [i for i, t in enumerate(ref) if t in filler_words]
    rows = []
    for i in fillers:
        t = aw["caught"].get(i, ap["caught"].get(i))
        ctx = " ".join(ref[max(0, i - 4):i]) + f" <b>{ref[i]}</b> " + " ".join(ref[i + 1:i + 5])
        mark = lambda a: '<span class="yes">caught</span>' if i in a["caught"] else '<span class="no">missed</span>'
        when = f'<button class="t" data-t="{t:.2f}">{t:.1f}s</button>' if t is not None else "–"
        rows.append(f"<tr><td>{when}</td><td>{ctx}</td><td>{mark(aw)}</td><td>{mark(ap)}</td></tr>")
    stats = {"words": len(ref), "fillers": len(fillers),
             "w_caught": len(aw["caught"]), "p_caught": len(ap["caught"]),
             "w_errors": aw["errors"], "p_errors": ap["errors"], "fixed": len(fixed)}
    table = (f'<table><thead><tr><th>When</th><th>What I said</th><th>Whisper</th><th>Parakeet</th></tr></thead>'
             f'<tbody>{"".join(rows)}</tbody></table>') if rows else '<p class="muted">No fillers in this clip.</p>'
    sec = f"""
<section class="clip" id="{clip}">
  <h2>{clip.replace("_", " ")}</h2>
  <p class="muted">{stats["words"]} words · {stats["fillers"]} fillers ·
    word errors: Whisper {stats["w_errors"]}, Parakeet {stats["p_errors"]} ·
    fillers caught: Whisper {stats["w_caught"]}/{stats["fillers"]}, Parakeet {stats["p_caught"]}/{stats["fillers"]}</p>
  <audio controls preload="none" src="../audio/{clip}.wav"></audio>
  {table}
  <div class="cols">
    <div><h3>Before: Whisper</h3><p class="tx">{word_html(whisper, aw)}</p></div>
    <div><h3>After: Parakeet + vocabulary</h3><p class="tx">{word_html(parakeet, ap, fixed)}</p></div>
  </div>
</section>"""
    return sec, stats


CSS = """
:root { --bg:#fbfaf8; --fg:#1d1d1f; --muted:#6b6b70; --line:#e4e2dd; --panel:#fff;
  --ok:#1f7a4d; --bad:#b3261e; --badbg:#fde7e4; --fill:#fff1c2; --fix:#d8f0e3; --accent:#3b5bdb; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#141416; --fg:#ececee; --muted:#9a9aa1;
  --line:#2c2c31; --panel:#1c1c20; --ok:#5fd39a; --bad:#ff8a80; --badbg:#4a1f1c; --fill:#4a3d12; --fix:#1d3e2d; --accent:#8ea2ff; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.55 -apple-system, system-ui, sans-serif; }
main { max-width:1100px; margin:0 auto; padding:32px 16px 64px; }
h1 { font-size:28px; margin:0 0 4px; } h2 { font-size:19px; margin:0 0 4px; text-transform:capitalize; }
h3 { font-size:13px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); margin:0 0 8px; }
.muted { color:var(--muted); font-size:13px; }
.tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:12px; margin:20px 0 28px; }
.tile { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:14px 16px; }
.tile .l { font-size:13px; color:var(--muted); } .tile .v { font-size:26px; font-weight:650; }
.tile .v small { font-size:14px; color:var(--muted); font-weight:500; }
.clip { background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:20px; margin:0 0 20px; }
audio { width:100%; margin:10px 0 14px; }
table { width:100%; border-collapse:collapse; font-size:14px; margin-bottom:16px; }
th, td { text-align:left; padding:6px 8px; border-bottom:1px solid var(--line); vertical-align:top; }
th { font-size:12px; color:var(--muted); font-weight:600; }
.yes { color:var(--ok); font-weight:600; } .no { color:var(--bad); font-weight:600; }
.cols { display:grid; grid-template-columns:1fr 1fr; gap:20px; }
@media (max-width:760px) { .cols { grid-template-columns:1fr; } }
.tx { margin:0; line-height:1.9; }
.tx span { cursor:pointer; border-radius:4px; padding:1px 2px; }
.tx span:hover { outline:1px solid var(--accent); }
.wrong, .extra { background:var(--badbg); color:var(--bad); }
.extra { text-decoration:underline wavy; }
.filler { background:var(--fill); font-weight:600; }
.fixed { background:var(--fix); color:var(--ok); }
del { color:var(--bad); opacity:.75; font-size:13px; }
button.t { font:inherit; font-size:13px; border:1px solid var(--line); background:transparent; color:var(--accent);
  border-radius:6px; padding:1px 6px; cursor:pointer; }
.legend { display:flex; flex-wrap:wrap; gap:14px; font-size:13px; color:var(--muted); margin:0 0 20px; }
.legend span span { padding:1px 5px; border-radius:4px; }
"""

JS = """
document.querySelectorAll('[data-t]').forEach((el) => el.addEventListener('click', () => {
  const audio = el.closest('.clip').querySelector('audio');
  audio.currentTime = Math.max(0, parseFloat(el.dataset.t) - 1.0);  // a second of lead-in
  audio.play();
}));
"""


def main():
    vocab = [l.strip() for l in VOCAB.read_text().splitlines() if l.strip()] if VOCAB.exists() else []
    sections, totals = [], {}
    for ref_file in sorted(REFS.glob("*.txt")):
        clip = ref_file.name.removesuffix(".txt")
        if "." in clip or not (AUDIO / f"{clip}.wav").exists():
            continue  # skips the .words.txt timing files
        whisper = json.loads((WHISPER / f"{clip}.json").read_text())["words"]
        parakeet, fixed = parakeet_words(AUDIO / f"{clip}.wav", vocab)
        sec, stats = clip_section(clip, ref_file.read_text(), whisper, parakeet, fixed)
        sections.append(sec)
        for k, v in stats.items():
            totals[k] = totals.get(k, 0) + v
        print(f"{clip}: fillers Whisper {stats['w_caught']}/{stats['fillers']}, "
              f"Parakeet {stats['p_caught']}/{stats['fillers']}; errors {stats['w_errors']} vs {stats['p_errors']}")
    t = totals
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Whisper vs Parakeet</title>
<style>{CSS}</style></head><body><main>
<h1>Whisper → Parakeet</h1>
<p class="muted">The same {t["words"]} words of my footage, transcribed by edit-twin before (Whisper large-v3-turbo,
verbatim prompt) and after (Parakeet TDT 0.6B v3 with my vocabulary: {html.escape(", ".join(vocab)) or "none"}).
Checked against my hand-made transcripts. Click any word to hear it.</p>
<div class="tiles">
  <div class="tile"><div class="l">Fillers caught (um/uh/hmm)</div>
    <div class="v">{t["w_caught"]} → {t["p_caught"]} <small>of {t["fillers"]}</small></div></div>
  <div class="tile"><div class="l">Word errors</div>
    <div class="v">{t["w_errors"]} → {t["p_errors"]} <small>of {t["words"]} words</small></div></div>
  <div class="tile"><div class="l">Names fixed by vocabulary</div><div class="v">{t["fixed"]}</div></div>
</div>
<div class="legend"><span><span class="filler">um</span> filler</span><span><span class="wrong">word</span> misheard</span>
  <span><span class="extra">word</span> not said</span><span><del>word</del> left out</span>
  <span><span class="fixed">Name</span> corrected from vocabulary</span></div>
{"".join(sections)}
</main><script>{JS}</script></body></html>"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
