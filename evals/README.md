# Retrieval eval

Reproducible evaluation of the `triage_agent.rag` retrieval layer. Nothing
here modifies existing modules or `local_kb/`.

| Path | What |
|---|---|
| `corpus/*.tex` | 5 synthetic, topic-distinct drafts (no results in them) |
| `eval_manifest.json` | Registers the drafts; ids `peft`, `rl_trading`, `gnn`, `diffusion`, `fedlearn` |
| `golden.jsonl` | 20 real arXiv queries (15 answerable, 5 rejection), human-reviewed |
| `build_golden.py` | Re-fetches the golden set from pinned, versioned arXiv ids |
| `metrics.py` | Recall@k, RR@k, rejection; scoring rules in the docstring |
| `rerankers.py` | Reranker hook: `none` (identity); `jev` reserved, not implemented |
| `run_eval.py` | Runner: `--mode vector bm25 hybrid`, `--reranker none` |
| `results/` | Timestamped JSON (config snapshot + per-query rankings) and Markdown tables |

## Run

```bash
python -m evals.run_eval            # needs OPENAI_API_KEY in env or .env
pytest tests evals/tests            # pyproject testpaths only covers tests/
```

The runner rebuilds an eval-only Chroma index in `evals/.index/` (gitignored)
on every run, so the snapshot always matches the chunk parameters used.

## Scoring

- Ranking unit is the chunk, mapped to its draft id; labels are per draft.
- Recall@5: distinct relevant drafts in the top 5 chunks / relevant drafts.
- MRR@10: mean of 1/rank of the first relevant chunk in the top 10 (0 if none).
- Rejection queries (empty labels) are excluded from Recall/MRR means and
  scored as rejection rate: correct iff the retriever returns nothing.
- Latency: wall clock per query, warm-up excluded; vector and hybrid include
  the OpenAI query-embedding round trip.

## Known caveats

- q03 (ElaLoRA) is an easy query: its method closely matches the `peft` draft.
- None of the modes has a score threshold, so a rejection rate of 0 is the
  expected behaviour of the current retriever, not a runner bug.
