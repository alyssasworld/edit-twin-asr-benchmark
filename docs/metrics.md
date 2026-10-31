### PRIMARY METRICS

- Verbatim Word Error Rate (WER)
    - (Substitution + Deletion + Insertions) / Number of reference words, where N = number of words in MY reference transcript
    - lowercase, no punctuation, fillers KEPT, filler spellings unified
    - pooled across all clips (jiwer.wer(list, list)), lower is better, can exceed 100%
- Filler recall and precision
    - fillers = um, uh, hmm (+ spellings); "like"/"so" are NOT fillers
    - recall = real fillers caught / real fillers; precision = real fillers caught / fillers output
    - learn.py and the rough-cut workflow depend on these
- Word timestamp error
    - start and end separately, only correctly recognised words
    - median ms, % within 50 ms (~1.5 frames), % within 100 ms
    - cuts and captions land on these times

### SECONDARY METRICS

- Clean WER: Whisper EnglishTextNormalizer (fillers removed), comparable to leaderboards
- Formatted WER: case + punctuation kept → matters for vlog subtitles
- CER: character-level, catches near-miss spellings and names
- Hallucination: words per minute on room tone / music (target 0)

### PRACTICAL METRICS

- RTFx = audio seconds / processing seconds (higher is better), load time excluded,
1 warm-up + median of 3 runs, plugged in, same Mac
- Peak memory: /usr/bin/time -l (max RSS) + MLX peak memory for MLX models
- Load time and download size

### DECISION RULE (fixed before running)

- Must not be worse than Whisper turbo on filler recall or end-timestamp error
- Then lowest verbatim WER; RTFx breaks near-ties