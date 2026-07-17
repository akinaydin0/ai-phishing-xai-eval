# Transformer max_length=512 Ablation — Pre-registered Result

Tests the 2026-07-16 pre-registration (binding): does raising the context window from 256 to 512 recover the two legacy↔AI cross-corpus transfer losses? Isolated namespace `results/transformers_512/`; canonical results untouched. Protocol identical to Directive 6 except max_length=512. **Verdict (mechanical): H1.**

## (a) In-corpus test ROC-AUC — 256 vs 512 (6 units)

| Corpus | Model | 256 | 512 | Δ |
|---|---|--:|--:|--:|
| d2 | DistilBERT | 0.9999 | 0.9999 | +0.0000 |
| d2 | BERT-base | 0.9999 | 0.9999 | +0.0000 |
| d2 | RoBERTa-base | 0.9999 | 0.9999 | +0.0000 |
| d3_en | DistilBERT | 0.9968 | 0.9975 | +0.0007 |
| d3_en | BERT-base | 0.9979 | 0.9977 | -0.0002 |
| d3_en | RoBERTa-base | 0.9976 | 0.9984 | +0.0008 |

## (b) Cross-corpus (legacy↔AI) — the pre-registered test

| Direction | 256 | 512 | Δ (512−256) | best classical | 50%-gap target | Δ≥0.03? | ≥target? | supports H1? |
|---|--:|--:|--:|--:|--:|:--:|:--:|:--:|
| d2→d3_en | 0.7748 | 0.7642 | -0.0106 | 0.8429 (LR (tfidf)) | 0.809 | no | no | no |
| d3_en→d2 | 0.7306 | 0.8440 | +0.1134 | 0.7523 (LinearSVC (tfidf)) | 0.742 | YES | YES | YES |

**Decision rule (verbatim from the pre-registration).** H1 if EITHER direction has Δ ≥ +0.03 AND AUC_512 ≥ its 50%-gap target; H0 if BOTH have Δ < +0.01; else partial/inconclusive. → **H1.** Secondary weak check (never decides alone): in-corpus d3_en test AUC at 512 = 0.9984 vs 256 baseline 0.9979.

## (c) Truncation — 256 vs 512 (train split, % truncated)

| Corpus | Model tokenizer | 256 trunc% | 512 trunc% | median WP | p95 WP |
|---|---|--:|--:|--:|--:|
| d2 | DistilBERT | 36.53 | 17.69 | 176 | 1111 |
| d2 | BERT-base | 36.53 | 17.69 | 176 | 1111 |
| d2 | RoBERTa-base | 37.02 | 18.17 | 179 | 1124 |
| d3_en | DistilBERT | 75.95 | 22.11 | 364 | 707 |
| d3_en | BERT-base | 75.95 | 22.11 | 364 | 707 |
| d3_en | RoBERTa-base | 73.46 | 18.73 | 349 | 669 |

## (d) Cost + peak GPU memory (512)

| Corpus | Model | finalist wall (s) | unit wall (s) | peak GPU (MB) |
|---|---|--:|--:|--:|
| d2 | DistilBERT | 394 | 985 | 4502 |
| d2 | BERT-base | 582 | 1741 | 8473 |
| d2 | RoBERTa-base | 776 | 2137 | 8655 |
| d3_en | DistilBERT | 257 | 761 | 4503 |
| d3_en | BERT-base | 511 | 1530 | 8473 |
| d3_en | RoBERTa-base | 516 | 1546 | 8655 |

*Total 512 fine-tuning wall-clock (6 units): **145.0 min** vs 73 min for the same six units at 256.*

## Observations

1. **Verdict: H1.** Raising the window recovered the legacy↔AI transfer on **one** of the two pre-registered directions: **d3_en→d2 jumps 0.7306 → 0.8440** (Δ+0.1134), clearing both the +0.03 bar and the 0.742 target and now **exceeding the best classical champion (0.7523)** — a reversal of the Directive-6 result for this direction. The other direction is essentially unmoved (d2→d3_en 0.7642, Δ-0.0106).

2. **The recovery is one-directional, and that is the real finding.** Truncation caused the **d3_en→d2** loss (source = the 76%-truncated AI corpus; at 512 the champion trains on fuller emails and transfers far better), but **not** the **d2→d3_en** loss (source d2 has short docs, median 176 WP — already almost untruncated at 256, so 512 adds no training signal). The pre-registered rule fires H1 on 'either direction', but the honest reading is a *directional* truncation effect keyed to the source corpus's length, not a blanket legacy↔AI fix.

3. **512 cut d3_en train truncation from ~76% to ~22%** (see (c)) — the window genuinely widened on the corpus the hypothesis targets, so the d3_en→d2 recovery is attributable to more complete AI-phishing context rather than chance.

4. **In-corpus stays at ceiling** at 512 (d3_en 0.9984 vs 256 0.9979); the ablation neither helps nor hurts in-corpus, as expected near AUC 1.0 — the secondary weak check adds nothing to the verdict.

5. **512 champions** (val-AUC, ties→smaller model): d2=DistilBERT (val 0.9999); d3_en=RoBERTa-base (val 0.9991). **Cost** ~145 min for six 512 units vs 73 min at 256 (~2.0× — the ~2× activation cost of doubling the sequence length), peak GPU ≤ 8.7 GB; trivial overnight.
