# EDA v2 — Findings on Cleaned Corpora

Recomputed on the deduplicated, cleaned corpora (`data/cache/clean_*.pkl`). Reproducible cold via `python -m src.eda_v2`; seed 42. D1 statistics use the **subject+body** natural-text view (51,839 rows) for like-for-like comparability with D2/D3 and the interim report; see the heterogeneity note in §5.

## 1. Token length per stratum (cleaned)

| Stratum | n | median | p95 |
|---|--:|--:|--:|
| modern phishing | 28,062 | 62 | 542 |
| modern legit | 23,777 | 199 | 1177 |
| Enron ham | 15,406 | 169 | 939 |
| Enron spam | 13,818 | 132 | 951 |
| AI phishing | 6,432 | 229 | 394 |
| AI spam | 6,228 | 282 | 510 |
| AI valid | 6,504 | 269 | 477 |

## 2. Pairwise Jaccard, top-500 non-stopword tokens (cleaned)

| Pair | Jaccard (clean) | Jaccard (raw, interim) |
|---|--:|--:|
| legacy↔modern | 0.353 | 0.351 |
| legacy↔AI | 0.248 | 0.250 |
| modern↔AI | 0.256 | 0.263 |

## 3. D1 per-class pattern means (subject+body, cleaned)

| Metric | Legitimate | Phishing |
|---|--:|--:|
| URLs / email | 2.847 | 3.141 |
| Email addresses / email | 1.292 | 2.003 |
| OOV tokens / email (5k/class sample, seed 42) | 97.7 | 148.3 |

*Scope: computed on the decontaminated D1 (Enron excluded). URL/email density exceeds the interim §2.5 full-set reference (1.71/2.07 URLs, 0.78/1.31 emails) because the removed Enron corporate mail carried almost no URLs — a scope difference, not a cleaning effect. The directional finding (phishing > legitimate on all three metrics) holds.*

## 4. RAW vs CLEAN — do the interim findings still hold?

| Interim finding | Raw | Clean | Verdict |
|---|--:|--:|:--|
| Modern phishing median (uniquely short) | 62 | 62 | HOLDS |
| Modern legit median (≈ Enron ham) | 199 | 199 | HOLDS |
| Enron ham median | 176 | 169 | HOLDS |
| Enron spam median | 138 | 132 | HOLDS |
| AI phishing median (longest phish) | 232 | 229 | HOLDS |
| Jaccard legacy↔modern | 0.351 | 0.353 | HOLDS |
| Jaccard legacy↔AI | 0.250 | 0.248 | HOLDS |
| Jaccard modern↔AI | 0.263 | 0.256 | HOLDS |

**Derived qualitative findings (all recomputed on cleaned data):**
- *Modern phishing is uniquely short* — 62 tok, the shortest stratum. **HOLDS.**
- *Modern legit ≈ Enron ham* — 199 vs 169 tok. **HOLDS.**
- *AI phishing is the longest phish (~4× modern phish)* — 229 vs 62 tok (3.7×). **HOLDS.**
- *AI distribution is the tightest* — AI-phish p95 394 < modern-phish p95 542 < Enron-spam p95 951. **HOLDS.**
- *AI valid/phishing/spam nearly indistinguishable by length* — 269/229/282 tok. **HOLDS.**
- *Three distinct lexical clusters* — Jaccard 0.353/0.248/0.256. **HOLDS.**

### Duplicate rates (raw observation → resolved by cleaning)

| Corpus | Raw dup % | After cleaning | Note |
|---|--:|--:|---|
| D2 Enron-Spam | 9.6% (exact-raw) / 11.64% (post-norm) | 0% | removed by within-corpus dedup |
| D1 Kaggle | 0.5% | 0% | removed by dedup |
| D3 PhishFuzzer | 0.3% **(full 23,100-corpus, seeds incl.)** | 0% | variants-only raw dup = **0.0%**; 0% after cleaning |

*D3 English filter: 19,799 cleaned variants → 19,164 English (635 non-English removed). The interim '742 non-English' was the full 23,100-corpus figure; the variants-only equivalent is 636 raw / 635 post-cleaning.*

## 5. D1 text-representation heterogeneity

D1 ships two text representations, both pinned as determinism fingerprints:

- **subject+body** (natural text, reconstructed from the 5 non-Enron per-source CSVs): **51,839** rows, medians legit 199 / phish 62 tok.
- **text_combined** (the Kaggle-preprocessed unified field): **52,704** rows, medians legit 118 / phish 45 tok.

(a) `text_combined` is **third-party-preprocessed** by the Kaggle author (stop-words + punctuation stripped, ~half the tokens), so it cannot support Pipeline A's stop-word ablation and is not length-comparable to D2/D3.
(b) Cleaned **subject+body reproduces the interim medians 199/62 exactly**, so the interim length findings **hold like-for-like** after cleaning.
(c) The shorter 118/45 medians on `text_combined` are a **representation artifact**, not a cleaning effect.
(d) The two universes differ by **865 rows** (52,704 text_combined − 51,839 subject+body). Accounting: subject+body cleaning removed 624 empties + 256 duplicates = 880, vs text_combined's 2 + 13 = 15 (= 622 more empties + 243 more duplicates; the raw per-source CSVs are less curated than the unified field). The two universes must never appear in one table without labels.

> **Modelling note.** The representation used for modelling is deferred to the baselines directive (working hypothesis: subject+body, for consistency with D2/D3 and to keep the stop-word ablation feasible). Supervisor-meeting talking point. See `notes/decisions.md`.

## Figures

- `figures/eda_v2/fig_class_dist_d1.{png,pdf}` — D1 class distribution
- `figures/eda_v2/fig_text_length_d1.{png,pdf}` — D1 class-conditional length
- `figures/eda_v2/fig_cross_corpus_length.{png,pdf}` — three-way cross-corpus length
- `figures/eda_v2/fig_d1_representation.{png,pdf}` — supplementary D1 representation gap
