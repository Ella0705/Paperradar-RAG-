# 检索评测报告：第 1 轮 vs 第 2 轮（加入近主题干扰草稿）

所有数字均由已保存的结果文件离线计算，本报告未重跑检索：

- 第 1 轮：`evals/results/20261022T130631Z_vector-bm25-hybrid_none.json`
- 第 2 轮：`evals/results/20261022T133248Z_vector-bm25-hybrid_none.json`
- Precision@5 / nDCG@5 由 `python -m evals.rescore` 从两轮各自保存的 top-10 排序重算，
  输出为同名 `*.rescored.json`；重算时逐题复现了原有的 Recall@5 / RR@10，全部一致。
- 表格分别由 `python -m evals.compare` 和 `python -m evals.analysis` 生成，原样粘贴。

## 1. 两轮语料构成

| 轮次 | 语料 | chunk 数（各草稿） | 评测集 |
|---|---|---|---|
| 第 1 轮 | 5 份主题区分明显的草稿：`peft` `rl_trading` `gnn` `diffusion` `fedlearn` | 57（11–12/份） | q01–q20：15 有答案 + 5 拒识 |
| 第 2 轮 | 第 1 轮 + 3 份近主题干扰草稿：`llm_quant`（对 `peft`）、`gan`（对 `diffusion`）、`marl`（对 `rl_trading`） | 93（11–13/份） | q01–q32：27 有答案 + 5 拒识（新增 q21–q32 共 12 道边界题） |

两轮检索配置相同：chunk 800 / overlap 120，k=10，fetch_k=20，MMR λ=0.5，RRF k=60，
`text-embedding-3-small`，reranker = none。q01–q20 的题面与标注两轮完全相同。

边界题标注规则（单标签）：

- peft vs llm_quant：LoRA 微调与量化结合 → `peft`；免训练量化（即使含低秩项）→ `llm_quant`。
- diffusion vs gan：对扩散模型的对抗蒸馏 → `diffusion`；与扩散对标的 GAN 生成器 → `gan`。
- rl_trading vs marl：金融市场中的多智能体 RL → `rl_trading`；游戏 / 协作中的多智能体 RL → `marl`。

已知简单样例：q03（ElaLoRA，方法与 `peft` 草稿高度吻合）、q27（StyleGAN-T，`gan` 草稿相关工作中显式引用了该论文）。

## 2. 指标定义（摘要，完整规则见 `evals/metrics.py`）

- 排序单位是 chunk，按其所属草稿判定相关。
- Recall@5：前 5 个 chunk 覆盖到的相关草稿占比。MRR@10：前 10 中首个相关 chunk 排名的倒数均值。
- Precision@5：前 5 个 chunk 中来自正确草稿的占比（分母恒为 5）。
- nDCG@5：二元增益、log2(rank+1) 折扣；理想排序为 min(5, 该草稿 chunk 数) 个相关 chunk 置顶。每份草稿至少 11 个 chunk，所以理想值总是 5 个相关 chunk。
- 拒识题不计入上述均值，单列拒识率（返回为空才算正确）。

## 3. 两轮并列（全部指标）

| mode | round | corpus | queries (answerable+rejection) | Recall@5 | MRR@10 | Precision@5 | nDCG@5 | rejection rate | mean latency (ms) |
|---|---|---|---|---|---|---|---|---|---|
| vector | old | 5 drafts / 57 chunks | 15+5 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 245.0 |
| vector | new | 8 drafts / 93 chunks | 27+5 | 1.000 | 1.000 | 0.904 | 0.927 | 0.000 | 211.1 |
| vector | new (old set) | 8 drafts / 93 chunks | 15+5 | 1.000 | 1.000 | 0.973 | 0.982 | 0.000 | 219.5 |
| bm25 | old | 5 drafts / 57 chunks | 15+5 | 1.000 | 1.000 | 0.987 | 0.986 | 0.000 | 1.4 |
| bm25 | new | 8 drafts / 93 chunks | 27+5 | 1.000 | 0.981 | 0.859 | 0.887 | 0.000 | 1.8 |
| bm25 | new (old set) | 8 drafts / 93 chunks | 15+5 | 1.000 | 1.000 | 0.947 | 0.958 | 0.000 | 2.0 |
| hybrid | old | 5 drafts / 57 chunks | 15+5 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 179.4 |
| hybrid | new | 8 drafts / 93 chunks | 27+5 | 1.000 | 1.000 | 0.904 | 0.930 | 0.000 | 202.1 |
| hybrid | new (old set) | 8 drafts / 93 chunks | 15+5 | 1.000 | 1.000 | 0.987 | 0.991 | 0.000 | 209.7 |

"new (old set)" = 第 2 轮语料上只对 q01–q20 计分，用于把"语料变难"和"题目变难"分开。

读法：

- Recall@5 / MRR@10 在两轮几乎都饱和，看不出干扰草稿的影响；Precision@5 / nDCG@5 能看出来。
- 只看原 20 题（语料变了，题没变），三种模式的 Precision@5 都下降，即干扰草稿本身就会挤进旧题的前 5：vector 1.000→0.973，bm25 0.987→0.947，hybrid 1.000→0.987。
- 加上边界题后，三种模式都继续下降，bm25 最低（P@5 0.859，nDCG@5 0.887）。vector 与 hybrid 非常接近（P@5 同为 0.904，nDCG@5 0.927 vs 0.930）。在 27 题上，一个 chunk 的差别就是 1/135 ≈ 0.007，这个差距不足以说明谁更好。
- 拒识率两轮均为 0：检索没有分数阈值，每个拒识题都返回了 10 个 chunk。
- 延迟：vector / hybrid 以 OpenAI 查询 embedding 的网络往返为主，两轮间的差异属于网络波动；bm25 随 chunk 数增加略升（1.4→1.8 ms）。

## 4. 干扰 chunk 挤入前 5 的次数统计

"对手草稿"按第 1 节的三组配对定义。只统计相关草稿有对手的有答案题：边界题 12 道，原题中 9 道（q01–q06、q10–q12）。

第 2 轮：

| mode | query group | queries | rival chunks in top 5 / slots | queries with >=1 rival in top 5 | rival at rank 1 |
|---|---|---|---|---|---|
| vector | boundary (q21-q32) | 12 | 11 / 60 | 8 | 0 |
| vector | original with a rival (q01-q20) | 9 | 2 / 45 | 2 | 0 |
| bm25 | boundary (q21-q32) | 12 | 15 / 60 | 11 | 1 |
| bm25 | original with a rival (q01-q20) | 9 | 3 / 45 | 2 | 0 |
| hybrid | boundary (q21-q32) | 12 | 12 / 60 | 11 | 0 |
| hybrid | original with a rival (q01-q20) | 9 | 1 / 45 | 1 | 0 |

第 1 轮对照（干扰草稿尚未入库）：三种模式在这 9 道原题上均为 0 / 45。

要点：

- 边界题上，干扰 chunk 占据前 5 名位置的 18%（vector 11/60）到 25%（bm25 15/60）；12 题中有 8–11 题至少被挤入 1 个。
- 被挤到第 1 名只发生了 1 次：bm25 的 q23，见第 5 节。
- hybrid 挤入的题数和 bm25 一样多（11 题），总数（12）接近 vector（11）。也就是说，RRF 把 bm25 的干扰也带进了前 5 的后几位，但没让它们排到第 1。
- 原题中被挤入的具体是：vector 的 q10、q12，bm25 的 q04（2 个 `marl`）、q12，hybrid 的 q10，每处各 1 个对手 chunk，q04 例外。

## 5. 案例：q23（LQER）上 bm25 排错、hybrid 排对

q23 是 LQER（免训练的低秩量化误差重建），标注为 `llm_quant`。bm25 把 `peft` 草稿的
"Interaction with Quantization" 小节排在第 1，RR@10 = 0.5；vector 和 hybrid 都把
`llm_quant` 的 "Related Work" 排在第 1。

由 `python -m evals.analysis <第 2 轮 rescored.json> --trace q23` 重建的 RRF 分数：

| chunk (draft / section) | vector rank | bm25 rank | RRF score | exact? | hybrid rank |
|---|---|---|---|---|---|
| llm_quant / Related Work | 1 | 4 | 0.032018 | yes | 1 |
| peft / Interaction with Quantization | 4 | 1 | 0.032018 | yes | 2 |
| llm_quant / (preamble) | 3 | 3 | 0.031746 | yes | 3 |
| llm_quant / Introduction | 5 | 5 | 0.030769 | yes | 4 |
| peft / Introduction | 8 | 7 | 0.029631 | yes | 6 |

这个案例需要如实说明：**hybrid 排对，不是 RRF 判断出了哪个更相关，而是一次平分的决胜结果。**

- 两个 chunk 的名次正好互换（1 与 4、4 与 1），RRF 分数都是 1/61 + 1/64 = 0.032018，完全相同。
- `rrf_merge`（`triage_agent/rag/hybrid.py`）用 Python 的稳定排序 `sorted(..., reverse=True)` 按分数排序。分数相同时保持插入顺序，而 `retriever.py` 先传入向量列表、后传入 BM25 列表，所以平分时向量列表里排名更前的 chunk 胜出。
- 换句话说，这一题 hybrid 的正确，来自"平分时偏向向量结果"这一实现细节。如果向量列表排序略有不同，或两个列表的传入顺序调换，hybrid 在 q23 上就会和 bm25 一样排错。

**重建的前提**：表中分数用的是第 2 轮 vector 与 bm25 两次独立运行各自保存的排序。hybrid 运行时内部会再做一次查询 embedding；bm25 是确定性的，但无法离线证明 hybrid 内部的向量排序与 vector 那次运行完全一致。重建出的顺序（1、2、3、4、6）与 hybrid 实际保存的名次逐一吻合，支持这一前提，但这只是佐证，不是证明。

bm25 为什么会排错：LQER 的摘要里 "low-rank"、"quantization"、"LLMs" 这几个词很密集，`peft` 草稿那一节恰好同时讨论低秩适配器和量化后的骨干网络，关键词重合度高；向量检索则把 `llm_quant` 的相关工作小节排在第 1。

## 6. Bootstrap 置信区间（Precision@5）

由 `python -m evals.bootstrap <两轮 rescored.json>` 生成：

- 对有答案题有放回重采样，共 10,000 次，固定种子 20261022，取百分位区间。
- 每次重采样，三种模式共用同一组题，所以 bm25 − vector 是配对差值。
- 拒识题没有 Precision@5，不参与重采样。

| 20261022T130631Z (n=15) | mean | 95% CI |
|---|---|---|
| vector | +1.000 | [+1.000, +1.000] |
| bm25 | +0.987 | [+0.960, +1.000] |
| hybrid | +1.000 | [+1.000, +1.000] |
| bm25 - vector | -0.013 | [-0.040, +0.000] |

| 20261022T133248Z (n=27) | mean | 95% CI |
|---|---|---|
| vector | +0.904 | [+0.852, +0.956] |
| bm25 | +0.859 | [+0.800, +0.911] |
| hybrid | +0.904 | [+0.859, +0.941] |
| bm25 - vector | -0.044 | [-0.111, +0.015] |

读法：

- 第 2 轮 bm25 − vector 的点估计是 −0.044，方向是 bm25 更差，但 95% 区间 [−0.111, +0.015] 包含 0。按这组数据，不能说两者的差异在统计上成立。
- 三种模式的区间大幅重叠，vector 与 hybrid 的点估计相同（0.904），与"持平"的判断一致。
- 第 1 轮 vector 和 hybrid 的区间退化为 [1, 1]：15 题全部 Precision@5 = 1，没有方差可供重采样，这反映的是指标在这一轮饱和，并不说明估计很精确。bm25 − vector 的上界恰为 0，原因相同（vector 每题都是 1，差值不可能为正）。

## 7. 结论

- 加入干扰草稿后，评测集开始能分出差别，但只有 Precision@5 / nDCG@5 能看出来；Recall@5 / MRR@10 只关心有没有命中、第一个命中在哪，在单标签加强主题的设置下仍然饱和。
- 方向上，三种模式中 bm25 最容易被近主题干扰带偏：Precision@5 最低，挤入的干扰 chunk 最多，唯一一次第 1 名错误也是它。但第 6 节的置信区间表明，这一差异在当前样本量下不显著。
- vector 与 hybrid 在这 27 题上持平。
- hybrid 在 q23 上的唯一"纠错"是平分决胜的结果，不能当作 RRF 优于 bm25 的证据。
- 拒识能力为 0，原因是检索层没有阈值（按约定本轮不改检索层）。

## 8. 局限

- **样本量**：第 2 轮只有 27 道计分题（第 1 轮 15 道），只够支持方向性结论，不够支持显著性结论。第 6 节中 bm25 − vector 的 95% 区间包含 0，就是直接体现。题目之间也不完全独立：每份草稿只有 3–4 道题，同一草稿的题往往一起对、一起错。
- **合成语料与真实草稿的差异**：8 份草稿都是为评测写的合成 LaTeX，每份 540–690 词，只描述计划中的实验，结构统一（Introduction / Related Work / Method / Experimental Plan / Limitations），主题边界是刻意设计的。真实研究草稿篇幅更长，有公式、图表、实验结果和未完成的段落，主题之间的重叠也不受控制。另外，部分草稿在相关工作中直接点名了查询论文（q27），q03 的方法和草稿高度吻合。因此这里的绝对分数不能外推到真实知识库。
- **延迟含 OpenAI 网络往返**：vector 与 hybrid 的延迟包括每次查询的 OpenAI embedding 请求，以网络和 API 排队时间为主，随时间和地点波动；两轮之间几十毫秒的差异不代表检索本身变快或变慢。bm25 只在本地计算，它的延迟和另外两种模式不可直接比较。
- **标注**：每题只有一个标签，由我提议、你逐条审核。边界题的归属依赖第 1 节的三条规则，规则本身带有判断。
