# Retrieval eval

Reproducible evaluation of the `triage_agent.rag` retrieval layer. Nothing
here modifies existing modules or `local_kb/`.

| Path | What |
|---|---|
| `corpus/*.tex` | 8 synthetic drafts (no results in them): 5 topic-distinct + 3 near-topic distractors |
| `eval_manifest.json` | Registers the drafts; ids `peft`, `rl_trading`, `gnn`, `diffusion`, `fedlearn`, distractors `llm_quant`, `gan`, `marl` |
| `golden.jsonl` | 32 real arXiv queries (27 answerable, 5 rejection), human-reviewed |
| `build_golden.py` | Re-fetches the golden set from pinned, versioned arXiv ids |
| `metrics.py` | Recall@k, RR@k, rejection; scoring rules in the docstring |
| `rerankers.py` | Reranker hook: `none` (identity); `jev` reserved, not implemented |
| `run_eval.py` | Runner: `--mode vector bm25 hybrid`, `--reranker none` |
| `compare.py` | Side-by-side table of two result files, incl. the new run on the old query set |
| `rescore.py` | Recomputes all metrics offline from stored rankings → `*.rescored.json` |
| `analysis.py` | Distractor-intrusion counts and per-query RRF traces from stored rankings |
| `bootstrap.py` | Paired bootstrap 95% CIs for Precision@5 and bm25 − vector |
| `results/` | Timestamped JSON (config snapshot + per-query rankings) and Markdown tables |
| `reports/` | Written reports; every table is pasted from the scripts above |

## Corpus rounds

| Round | Corpus | Golden | Commit |
|---|---|---|---|
| 1 | 5 drafts (57 chunks) | q01–q20 | `b293214` |
| 2 | + `llm_quant`, `gan`, `marl` (93 chunks) | + q21–q32 boundary queries | `3778173` |

Boundary labelling rules (q21–q32, one label each):

- peft vs llm_quant: LoRA fine-tuning combined with quantization → `peft`;
  training-free quantization, even with low-rank terms → `llm_quant`.
- diffusion vs gan: adversarial distillation of a diffusion model → `diffusion`;
  GAN generators positioned against diffusion → `gan`.
- rl_trading vs marl: multi-agent RL in financial markets → `rl_trading`;
  multi-agent RL in games / coordination → `marl`.

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
- Precision@5: relevant chunks in the top 5 / 5.
- nDCG@5: binary gain; ideal = min(5, indexed chunks of the relevant draft).
- Rejection queries (empty labels) are excluded from Recall/MRR means and
  scored as rejection rate: correct iff the retriever returns nothing.
- Latency: wall clock per query, warm-up excluded; vector and hybrid include
  the OpenAI query-embedding round trip.

## Known caveats

- q03 (ElaLoRA) is an easy query: its method closely matches the `peft` draft.
- q27 (StyleGAN-T) is an easy query: the `gan` draft cites StyleGAN-T by name.
- q01-q20 labels predate the distractors and were kept unchanged; q01
  (QDyLoRA) and q12 (IMM) now also border `llm_quant` / `gan`.
- None of the modes has a score threshold, so a rejection rate of 0 is the
  expected behaviour of the current retriever, not a runner bug.
