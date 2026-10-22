| mode | reranker | Recall@5 | MRR@10 | rejection rate | mean latency (ms) | p50 latency (ms) |
|---|---|---|---|---|---|---|
| vector | none | 1.000 | 1.000 | 0.000 | 211.1 | 207.3 |
| bm25 | none | 1.000 | 0.981 | 0.000 | 1.8 | 1.8 |
| hybrid | none | 1.000 | 1.000 | 0.000 | 202.1 | 200.1 |

Recall@5 / MRR@10 are means over 27 answerable queries; rejection rate is over 5 rejection queries (correct iff nothing is returned).
