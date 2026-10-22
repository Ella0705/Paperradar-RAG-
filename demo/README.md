# Demo memos

Two real end-to-end runs of `triage 2402.10462` (QDyLoRA), same day, same
settings (`LLM_BACKEND=auto`, only `OPENAI_API_KEY` set). The only difference
is the local knowledge base.

| File | Local KB | LocalOverlap | Relevance / decision |
|---|---|---|---|
| `triage_2402.10462.md` | empty manifest (no papers, no sources) | skipped, no local section | LOW / skip |
| `triage_2402.10462_with_local_kb.md` | 8 synthetic demo drafts, index rebuilt (93 chunks) | matches `peft` (relevance 0.9, "extends your work") | HIGH / read in full |

With an empty manifest the LocalOverlap agent returns before any retrieval,
so the memo can only say there is no overlap with your work. With the demo
drafts indexed, the vector + BM25 hybrid retrieval finds the PEFT draft's
chunks and the agent labels the relationship.

## Demo corpus

`local_kb/local_manifest.json` currently registers the 8 synthetic drafts
from `evals/corpus/` (5 topic drafts + 3 near-topic distractors). They exist
for this demo and the retrieval eval only. Replace them with your real
Overleaf / GitHub material: source enrollment + sync rewrites the manifest.
Then rebuild the index:

```bash
python -m triage_agent.rag.build_index   # needs OPENAI_API_KEY in the environment
```

## Known degradation: no related-papers list

Both memos have no related-papers list. The Retriever agent queries
Semantic Scholar, which returned HTTP 429 in both runs. **Without a
`SEMANTIC_SCHOLAR_API_KEY`, the related-papers list degrades to empty; this
is known rate-limit behaviour, not a bug in the pipeline.** Set the key in
`.env` to get the list back.

## Reproduce

```bash
# .env: OPENAI_API_KEY=..., LLM_BACKEND=auto
dotenv run -- python -m triage_agent.rag.build_index
triage 2402.10462 --output demo/triage_2402.10462_with_local_kb.md
```

LLM output is not deterministic, so wording and scores will vary between runs.
