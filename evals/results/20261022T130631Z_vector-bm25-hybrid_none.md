| mode | reranker | Recall@5 | MRR@10 | rejection rate | mean latency (ms) | p50 latency (ms) |
|---|---|---|---|---|---|---|
| vector | none | 1.000 | 1.000 | 0.000 | 245.0 | 180.4 |
| bm25 | none | 1.000 | 1.000 | 0.000 | 1.4 | 1.3 |
| hybrid | none | 1.000 | 1.000 | 0.000 | 179.4 | 172.1 |

Recall@5 / MRR@10 are means over 15 answerable queries; rejection rate is over 5 rejection queries (correct iff nothing is returned).
