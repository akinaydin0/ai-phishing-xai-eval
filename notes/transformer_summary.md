# Transformer Fine-Tuning — Summary

Fine-tuned DistilBERT, BERT-base and RoBERTa-base on the three frozen corpora (subject+body, max_length=256). Per (model, corpus) grid: learning-rate ∈ {2e-5, 3e-5, 5e-5}, up to 4 epochs, batch 32 + fp16, weight_decay 0.01, warmup_ratio 0.1; val ROC-AUC each epoch, early-stop patience 1, best-val-AUC checkpoint kept. **The lr is selected by validation AUC and the TEST split is evaluated exactly once per finalist.** Determinism: seeds=42, cudnn.deterministic, pinned versions (recorded per run json); GPU fine-tuning is not bit-exact (see notes/decisions.md). Full log: `results/results.csv`.

## (a) In-corpus: transformers vs. classical champions (test set)

| Corpus | DistilBERT | BERT-base | RoBERTa-base | Best transformer | Classical champion | Δ AUC (T−C) |
|---|--:|--:|--:|---|---|--:|
| d1_sb | 0.9999 | 0.9998 | 0.9998 | DistilBERT (0.9999) | LinearSVC (0.9998) | +0.0001 |
| d2 | 0.9999 | 0.9999 | 0.9999 | DistilBERT (0.9999) | MultinomialNB (0.9996) | +0.0003 |
| d3_en | 0.9968 | 0.9979 | 0.9976 | BERT-base (0.9979) | LinearSVC (0.9993) | -0.0014 |

*Values are test ROC-AUC. The classical champion is the in-corpus leaderboard #1 (tuned classical/dense) from the HPO phase.*

| Corpus | Model | sel. lr | val AUC | test F1 | test AUC | best epoch |
|---|---|--:|--:|--:|--:|--:|
| d1_sb | DistilBERT | 3e-05 | 0.9999 | 0.9961 | 0.9999 | 2 |
| d1_sb | BERT-base | 2e-05 | 0.9999 | 0.9956 | 0.9998 | 1 |
| d1_sb | RoBERTa-base | 2e-05 | 0.9999 | 0.9975 | 0.9998 | 1 |
| d2 | DistilBERT | 3e-05 | 0.9999 | 0.9944 | 0.9999 | 2 |
| d2 | BERT-base | 5e-05 | 0.9998 | 0.9949 | 0.9999 | 2 |
| d2 | RoBERTa-base | 2e-05 | 0.9998 | 0.9937 | 0.9999 | 2 |
| d3_en | DistilBERT | 5e-05 | 0.9976 | 0.9864 | 0.9968 | 3 |
| d3_en | BERT-base | 5e-05 | 0.9985 | 0.9852 | 0.9979 | 3 |
| d3_en | RoBERTa-base | 5e-05 | 0.9986 | 0.9851 | 0.9976 | 3 |

## (b) Cross-corpus: transformer champions vs. classical champions

Best transformer per **source** (by val AUC) applied to the other corpora's TEST splits. Compared against the best classical champion (tfidf/ensemble/dense) for the same directed pair.

| train → test | Transformer champ | T AUC | T F1 | Classical best | C AUC | Δ AUC (T−C) |
|---|---|--:|--:|---|--:|--:|
| d1_sb → d2 | DistilBERT | 0.9434 | 0.8551 | LogisticRegression (tfidf) | 0.9091 | +0.0343 |
| d1_sb → d3_en | DistilBERT | 0.9243 | 0.7955 | LogisticRegression (tfidf) | 0.8965 | +0.0278 |
| d2 → d1_sb | DistilBERT | 0.9199 | 0.8162 | LogisticRegression (tfidf) | 0.9080 | +0.0119 |
| d2 → d3_en | DistilBERT | 0.7748 | 0.8021 | LogisticRegression (tfidf) | 0.8429 | -0.0681 |
| d3_en → d1_sb | RoBERTa-base | 0.9659 | 0.7630 | LightGBM (ensemble) | 0.8888 | +0.0771 |
| d3_en → d2 | RoBERTa-base | 0.7306 | 0.5735 | LinearSVC (tfidf) | 0.7523 | -0.0217 |

**Headline (AI→real, d3_en→d2).** Classical champions spanned AUC 0.675–0.752; the transformer champion (RoBERTa-base) reaches **0.731** — inside that band but **does not close** the AI-to-real gap (Δ -0.022 vs the best classical, LinearSVC 0.752). See observations 2–3 for the full cross-corpus picture.

## (c) Truncation at max_length=256 (% of documents truncated)

WordPiece = BERT-base & DistilBERT (shared uncased vocab, identical lengths); BPE = RoBERTa.

| Corpus | Split | WordPiece trunc% | BPE trunc% | median len (WP) | p95 len (WP) | max len (WP) |
|---|---|--:|--:|--:|--:|--:|
| d1_sb | train | 48.92 | 45.5 | 247 | 1505 | 228292 |
| d1_sb | val | 48.86 | 45.54 | 248 | 1474 | 33662 |
| d1_sb | test | 48.78 | 45.32 | 245 | 1475 | 2901223 |
| d2 | train | 36.53 | 37.02 | 176 | 1111 | 37945 |
| d2 | val | 35.7 | 36.27 | 171 | 1051 | 48690 |
| d2 | test | 35.42 | 36.06 | 171 | 1091 | 21912 |
| d3_en | train | 75.95 | 73.46 | 364 | 707 | 2568 |
| d3_en | val | 74.85 | 72.14 | 364 | 696 | 2210 |
| d3_en | test | 76.0 | 73.53 | 364 | 718 | 2569 |

## (d) Computational cost — fine-tune minutes vs. sklearn seconds

| Corpus | Model | finalist wall (s) | unit wall (3 lr, s) | peak GPU (MB) | vs classical champ fit (s) |
|---|---|--:|--:|--:|--:|
| d1_sb | DistilBERT | 353 | 790 | 2724 | LinearSVC 0.35s |
| d1_sb | BERT-base | 512 | 1549 | 5004 | LinearSVC 0.35s |
| d1_sb | RoBERTa-base | 520 | 1223 | 5184 | LinearSVC 0.35s |
| d2 | DistilBERT | 199 | 596 | 2724 | MultinomialNB 0.05s |
| d2 | BERT-base | 390 | 778 | 5004 | MultinomialNB 0.05s |
| d2 | RoBERTa-base | 395 | 1089 | 5184 | MultinomialNB 0.05s |
| d3_en | DistilBERT | 131 | 386 | 2726 | LinearSVC 0.41s |
| d3_en | BERT-base | 257 | 770 | 5004 | LinearSVC 0.41s |
| d3_en | RoBERTa-base | 262 | 784 | 5184 | LinearSVC 0.41s |

*Canonical fine-tuning compute: **132.8 min** (2.21 h) on one RTX 3090 — the sum of the nine units' training wall-clock (the *unit wall* column above: the three-lr training loops per unit, including per-epoch validation scoring). This is the number comparable to the classical champions' sklearn fit times. The **end-to-end grid-script wall-clock was 143.7 min**; the ~11-min difference is not training — it is the per-unit tokenization + truncation-statistics passes (a full *untruncated* tokenisation of every split for the table in (c)), best-checkpoint `save_pretrained` I/O, and the single finalist TEST evaluations. The tuned classical champions fit in **seconds** — the accuracy-vs-compute trade-off the dissertation quantifies.*

*The 9 finalist checkpoints total **3.6 GB** on disk (`results/transformers/checkpoints/`, git-ignored as regenerable artifacts); they are reused by the cross-corpus transfer above and by the upcoming XAI phase.*

## Observations

1. **In-corpus, fine-tuned transformers match but do not beat the tuned linear TF-IDF champions — at ~100–1000× the compute.** d1_sb: best transformer DistilBERT 0.9999 vs LinearSVC 0.9998; d2: best transformer DistilBERT 0.9999 vs MultinomialNB 0.9996; d3_en: best transformer BERT-base 0.9979 vs LinearSVC 0.9993. On these near-ceiling corpora (classical AUC ≥ 0.999) contextual modelling buys no in-corpus discrimination; its promise is on the harder cross-corpus transfer, tested next.

2. **The headline AI→real gap (d3_en→d2) is NOT closed by contextual modelling.** Classical champions transferred d3_en→d2 at AUC 0.675–0.752; the transformer champion (RoBERTa-base) reaches 0.731 — inside that band and -0.022 below the best classical (LinearSVC 0.752). On this specific legacy-Enron target, contextual token representations do not beat a tuned linear model — the hardest transfer stays hard.

3. **But transformers win every transfer touching the modern-real corpus, and lose only the two legacy-Enron↔AI directions — reframing the asymmetry.** Transformer champions beat the best classical champion on 4/6 directed pairs; all four pairs involving d1_sb (modern real) improve — most strikingly d3_en→d1_sb 0.966 vs 0.889 (+0.077). The only two losses are d2→d3_en 0.775, d3_en→d2 0.731 — i.e. Enron↔AI in both directions. The residual hard case is legacy↔AI transfer, not AI→real in general; anchoring on modern-real data is where contextual modelling clearly helps.

4. **Truncation at 256 bites hardest on the AI corpus, not the legacy one — AI-generated emails are the longest.** d3_en truncates ~75.95% of train documents at 256 tokens (median 364 WordPiece tokens — the longest of the three), versus d1_sb ~48.92% (median 247) and d2 ~36.53% (median 176, the shortest). The 256 cap discards the most context on the AI corpus, yet its champion (RoBERTa) still tops d3_en in-corpus and transfers best to d1_sb — the discriminative signal survives in the first 256 tokens. Extreme outliers exist (d1_sb max >228,292 WP tokens — giant HTML/quoted threads), so median/p95 are the honest summaries.

5. **Compute cost is the transformers' real liability; the payoff is cross-corpus robustness on modern-real data, not in-corpus accuracy.** The 9-grid took 133 min on the RTX 3090 versus seconds for the tuned linear champions, with no in-corpus gain (all ≥ 0.997 AUC). Their value shows only in transfer to/from the modern-real corpus (up to +0.077 AUC over the best classical); for pure in-corpus deployment the accuracy-per-FLOP verdict still favours linear TF-IDF. The saved finalist checkpoints feed the next phase (XAI: contextual explanations).
