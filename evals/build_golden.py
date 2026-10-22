"""Build evals/golden.jsonl from a pinned list of real arXiv papers.

Candidates were discovered with ArxivClient.search_papers; the chosen papers
are pinned below by versioned arXiv id so that re-running this script fetches
the exact same titles and abstracts. Labels (relevant_draft_ids) are the
proposed annotation and must be human-reviewed before any eval is run.

Usage:
    python -m evals.build_golden            # writes evals/golden.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

from triage_agent.api.arxiv import ArxivClient

GOLDEN_PATH = Path(__file__).resolve().parent / "golden.jsonl"

# arXiv asks API clients to wait at least 3 seconds between requests.
_ARXIV_DELAY_S = 3.5

# (query_id, versioned arXiv id, relevant_draft_ids).
# Empty relevant_draft_ids = unrelated to every draft (rejection query).
PINNED: list[tuple[str, str, list[str]]] = [
    ("q01", "2402.10462v1", ["peft"]),        # QDyLoRA
    ("q02", "2405.17357v3", ["peft"]),        # DoRA: dynamic rank distribution
    ("q03", "2504.00254v1", ["peft"]),        # ElaLoRA
    ("q04", "1811.07522v3", ["rl_trading"]),  # Practical DRL for stock trading
    ("q05", "1907.01503v1", ["rl_trading"]),  # Adaptive DRL portfolio allocation
    ("q06", "2511.20678v1", ["rl_trading"]),  # Crypto portfolio with SAC / DDPG
    ("q07", "2006.13318v1", ["gnn"]),         # A note on over-smoothing for GNNs
    ("q08", "2507.05263v1", ["gnn"]),         # Over-smoothing via Anderson localization
    ("q09", "2211.10579v1", ["gnn"]),         # Tired of over-smoothing?
    ("q10", "2310.04378v1", ["diffusion"]),   # Latent Consistency Models
    ("q11", "2303.01469v2", ["diffusion"]),   # Consistency Models
    ("q12", "2503.07565v7", ["diffusion"]),   # Inductive Moment Matching
    ("q13", "2008.03371v1", ["fedlearn"]),    # LotteryFL
    ("q14", "2504.08872v1", ["fedlearn"]),    # Personalized FL, hierarchical edge
    ("q15", "2312.15148v1", ["fedlearn"]),    # Personalized FL, client selection
    ("q16", "1908.10115v1", []),              # astro-ph: galaxy mergers
    ("q17", "1004.1078v1", []),               # math.NT: gaps between primes
    ("q18", "1310.2110v1", []),               # q-bio.PE: scaling in ecology
    ("q19", "2306.11935v1", []),              # cond-mat: superconducting diode
    ("q20", "1912.13313v1", []),              # hep-ph: neutrino decoherence
]


def format_query(title: str, abstract: str) -> str:
    """Same query layout as triage_agent.rag.retriever.retrieve_for_paper_query."""
    return f"Title: {title}\n\nAbstract:\n{abstract}"


async def fetch_golden() -> list[dict]:
    fetched_at = datetime.now(UTC).date().isoformat()
    rows: list[dict] = []
    async with ArxivClient() as client:
        for i, (query_id, arxiv_id, relevant) in enumerate(PINNED):
            if i > 0:
                await asyncio.sleep(_ARXIV_DELAY_S)
            paper = await client.fetch_paper(arxiv_id)
            rows.append(
                {
                    "query_id": query_id,
                    "query": format_query(paper.title, paper.abstract),
                    "relevant_draft_ids": relevant,
                    "arxiv_id": arxiv_id,
                    "title": paper.title,
                    "fetched_at": fetched_at,
                }
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=GOLDEN_PATH)
    args = parser.parse_args(argv)

    rows = asyncio.run(fetch_golden())
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} queries to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
