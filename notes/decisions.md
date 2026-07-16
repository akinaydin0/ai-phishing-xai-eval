# Decision Log

Append-only record of methodological decisions. Newest at the bottom. Each entry:
**date · decision · rationale · links.** Created 2026-07-06, consolidating earlier
project decisions.

---

## 2026-07-05 · D3 binary label mapping (id: C2)

**Decision.** For PhishFuzzer (D3), collapse the three-class label
(Phishing / Spam / Valid) to binary: **Phishing + Spam → 1, Valid → 0.**

**Rationale.** Matches the binary convention already used by D1 and D2 (where label 1
covers phishing *and* spam / unsolicited mail, label 0 covers legitimate/ham), so the
three corpora share one label semantics for cross-corpus training and evaluation. The raw
three-class label is retained alongside the binary one (`type_3class`) for any finer-grained
analysis.

**Links.** `src/data_loader.py` (`load_phishfuzzer`, `_PHISHFUZZER_BINARY_MAP`); `CLAUDE.md`
(Datasets → D3).

---

## 2026-07-05 · D2 full dedup + pinned determinism fingerprints

**Decision.** Deduplicate every corpus fully before any split (non-negotiable, prevents
train/test leakage). Do **not** retain duplicates to preserve a class ratio. Replace the
generic "±2 pp class-balance shift" gate — which trips on D2 — with **exact determinism
fingerprints** (`CLEANING_EXPECTED`) pinning each frozen corpus's final row count and
per-class counts; any drift hard-fails the pipeline and forces a conscious re-pin recorded
in the commit message. Unpinned corpora fall back to the ±2 pp shift as a WARN.

**Rationale.** D2's duplicates are 74 % spam (spam-campaign repetition), so mandatory full
dedup legitimately shifts its spam share 50.93 % → 47.28 % (−3.65 pp), exceeding ±2 pp.
That is an expected data property, not a defect; keeping duplicate spam to hold a ratio
would re-introduce leakage. Two duplicate definitions are kept distinct so figures never look
contradictory: 9.6 % exact-raw (interim figure) vs 11.64 % after normalisation.

**Links.** `notes/cleaning_report.md`; `src/cleaning.py` (`CLEANING_EXPECTED`,
`run_sanity_asserts`); `CLAUDE.md` (Known data facts).

---

## 2026-07-06 · D1 dual text representation

**Decision.** EDA v2 reports D1 in **both** representations, with **subject+body as
primary**:
- **subject+body** (natural text, 51,839 rows after cleaning) drives all cross-corpus and
  lexical statistics — the three-way length figure, pairwise Jaccard, and D1 per-class
  pattern means — because every interim reference number and both D2/D3 are natural text.
  Cached as a first-class pinned corpus, `d1_decontaminated_sb`.
- **text_combined** (the Kaggle-preprocessed field, 52,704 rows) is reported only as a
  secondary heterogeneity comparison (medians ~half the length: legit 118 / phish 45 vs
  199 / 62).

The **modelling** representation is **deferred to the baselines directive** (working
hypothesis: subject+body, for representation consistency with D2/D3 and because Pipeline A's
stop-word ablation is impossible on the pre-stripped text_combined field). **Supervisor-meeting
talking point.**

**Rationale.** The cached `clean_d1_decontaminated` pkl holds `text_combined`, which the
Kaggle author already stripped of stop-words/punctuation (~half the tokens). Measuring D1
length on it would make "modern legit ≈ Enron ham" appear broken (118 vs 169) purely from
the representation change. Reconstructed cleaned subject+body reproduces the interim medians
199 / 62 **exactly**, confirming the interim findings hold like-for-like. The two D1 universes
(52,704 vs 51,839; an 865-row delta = 622 more empties + 243 more duplicates in the less-curated
per-source CSVs) must never appear in one table without labels.

**Links.** `notes/eda_v2_summary.md` (§5); `src/eda_v2.py`; `src/cleaning.py`
(`d1_decontaminated_sb`, `CLEANING_EXPECTED`).

---

## 2026-07-06 · D1 modelling representation = subject+body (evidence-backed)

**Decision.** Use **subject+body** (`d1_decontaminated_sb`) as the D1 modelling
representation. This upgrades the [D1 dual text representation] working hypothesis to a
decision, now backed by the baseline ablation. **Still a supervisor-meeting talking point.**

**Evidence (representation ablation, Run B vs Run A; frozen splits, seed 42).** In-corpus
TF-IDF + Logistic Regression:
- subject+body (`d1_decontaminated_sb`, test n=7,776): **F1 0.9906**, ROC-AUC 0.9995.
- text_combined (`d1_decontaminated`, test n=7,906): **F1 0.9927**, ROC-AUC 0.9992.

The gap is **0.21 pp F1** (text_combined marginally higher — its stop-word/punctuation
stripping mildly denoises the bag-of-words). So the representation choice costs ~nothing in
accuracy.

**Rationale.** subject+body wins on the two grounds that matter downstream, at negligible
accuracy cost: (a) representation consistency with D2/D3 (both natural text), which is
required for clean cross-corpus evaluation — a primary reported outcome; and (b) it
preserves stop-words/punctuation, so Pipeline A's stop-word-removal ablation and
URL/email placeholder substitution are actually possible (both are impossible on the
pre-stripped text_combined field).

**Links.** `notes/baseline_summary.md` (§Results Run A/B, Observation 4);
`results/results.csv`; `src/experiments/run_baselines.py`.

---

## 2026-07-06 · Evaluate the shortcut floor on ROC-AUC + cross-corpus, not in-corpus F1

**Decision.** Judge the length-only shortcut floor — and models generally — on **ROC-AUC
and cross-corpus behaviour**, not in-corpus F1. `results/results.csv` now logs trivial
`all_positive` / `majority_class` references and a `token_count`-only monotonic probe as the
honest floors every model must clear.

**Evidence (baselines addendum).** The 3-feature length floor's in-corpus F1
(0.694/0.636/0.777, d1_sb/d2/d3_en) is **at or below** the all-positive reference
(0.702/0.642/0.796) — F1 is dominated by the class prior and makes a near-signal-free model
look competent. The floor's weak real signal shows only in ROC-AUC (0.729/0.635/0.745, d2
almost at chance). A monotonic token_count-only probe shows the **pure-length direction is
universal** (coef sign − / *shorter→positive* in all three corpora; cross-corpus AUC
source-independent, all > 0.5), whereas the 3-feature floor's cross-corpus **inversion**
(d2→d1_sb AUC 0.341) is a **chars-per-token / character-density style signal**, not length —
confirmed because the pure-length probe d2→d1_sb = 0.722 does **not** invert.

**Correction to prior framing.** The earlier "d3_en has the highest length signal / AI is
structurally uniform" note overstated it: d3_en's high *floor* AUC (0.745) is largely the
style component, not token length (token-only AUC 0.560). The refined shortcut narrative:
length direction transfers; the style contrast the floor adds does not.

**Links.** `notes/baseline_summary.md` (Observations 3 & 6, trivial + probe tables);
`results/results.csv`; `results/baseline_coefs.json`; `src/experiments/run_baselines.py`.

---

## 2026-07-06 · Pin d3_en as a split-time determinism fingerprint

**Decision.** Add `DERIVED_FINGERPRINT` in `src/splits.py` pinning `d3_en` (n=19,164;
counts {0: 6504, 1: 12660}). `load_corpus` hard-fails on drift.

**Rationale.** `d3_en` is `clean_d3` filtered to English and is **not** covered by
`src.cleaning.CLEANING_EXPECTED` (which pins the full 19,799-row d3). The scoped leakage
review flagged the realistic path where clean_d3 is re-pinned/regenerated and the English
subset silently changes without the splits being rebuilt. The fingerprint + the earlier
`load_splits` coverage guard now make that impossible to miss.

**Links.** `src/splits.py` (`DERIVED_FINGERPRINT`, `load_corpus`).

---

## 2026-07-07 · XGBoost GPU episode (reportable engineering finding)

**Context.** Supervisor item #3 HPO runs the XGBoost search on the RTX 3090 GPU (device
`cuda`) for a ~100× speedup. Getting a correct, in-memory GPU search took four diagnosed
steps — worth documenting in the implementation chapter (and a candidate upstream XGBoost
issue).

**(a) CUDA path silently degenerate on scipy sparse CSR.** Feeding a sparse CSR matrix to
`XGBClassifier(device='cuda')` (or `QuantileDMatrix(sparse)`) produces **degenerate models**
— `best_iteration=0`, val AUC ≈ 0.69 (no early stopping) / 0.43 — with **no error**. The
identical params on CPU give AUC 0.995. Caught by the 2-trial smoke test (val_auc 0.70 for
XGB vs ~0.99 for every other family), which is exactly why the smoke test exists.

**(b) Dense-on-GPU OOMs at 50k.** Densifying the TF-IDF matrix fixes correctness (cuda+dense
AUC 0.996) but a 36,287×50,000 float32 matrix needs ~27 GB of GPU histogram memory → OOM on
the 24 GB card.

**(c) Fix: `QuantileDMatrix` from a dense array.** Quantising the dense matrix to uint8
(~1.8 GB) is both correct and memory-efficient. GPU search uses this; CPU finalisation keeps
sparse (works, and is the deterministic authority).

**(d) 24 GB ceiling binds at 50k × depth-10.** Even quantised, depth-10 histograms over 50k
features exceed 24 GB, and XGBoost's caching allocator retains memory across trials. Resolution
(Akın-approved): **XGB tunable `max_features` capped to {10k, 20k}**; the CPU-trained tree
families (**LightGBM keeps {10k, 20k, 50k}**, as do DT/RF) provide the empirical 50k check.
GPU hygiene: booster/QuantileDMatrix references are dropped + `gc.collect()` after every trial
so allocator retention cannot compound. All finalists re-set `n_estimators` via early stopping
on the val split **on CPU** (the GPU search's `best_iteration` is not reused).

**Links.** `src/experiments/hpo.py` (`_xgb_gpu_fit`, `XGB_MAXFEAT`, `finalize`);
`results/probe_timings.json`.

---

## 2026-07-07 · HPO methodology + headline modelling findings

**Decisions.** (1) One Optuna study per (family, corpus), **objective = validation ROC-AUC**
(tie-break val F1 via +1e-6·F1); the **test split is never touched during tuning**, the val
split is the sole optimisation surface. (2) **Representation-appropriate feature spaces**:
linear families (LR/NB/SVC) use 100k TF-IDF (LR/NB also tune ngram/min_df/sublinear);
tree/boosting families **tune** max_features ∈ {10k, 20k, 50k} (XGB {10k,20k}). (3) Boosting
fixes n_estimators=800 with **early_stopping_rounds=50 on val AUC** (finalists re-set it on
CPU). (4) Dense baselines: Word2Vec (trained on train only, mean-pooled) and pretrained GloVe,
classified by LR and an MLP that **early-stops on val**. (5) Champions (best tfidf / ensemble /
dense by val AUC per source) evaluated across all directed pairs.

**Findings (evidence for the write-up).**
- **Tuned linear TF-IDF is the in-corpus champion** on every corpus (LinearSVC/LR/MNB, AUC
  0.9993–0.9998); LightGBM is the best ensemble but does not beat linear; dense (W2V/GloVe-MLP)
  is competitive only on the AI corpus. Ensembles do **not** justify their cost on high-dim
  sparse text.
- **HPO is near-ceiling**: ΔF1 ≈ +0.002…+0.018, largest on weak defaults (NB); LinearSVC is
  optimal by default. **DecisionTree tuning trades F1 for AUC** (objective is AUC) — the 0.5
  threshold must be set separately from the ranking model.
- **Trees prefer ~20k features** (LightGBM 20k on all three) — the tuned-dimension decision
  was correct; the fair comparison is each family at its own best representation.
- **Champion cross-corpus AUC 0.68–0.91, asymmetric, AI→real weakest** (d3_en→d2 AUC 0.71–0.75)
  — reaffirms the dataset-aware-evaluation mandate for the tuned champions across families.

**Links.** `notes/hpo_summary.md`; `results/results.csv`; `results/hpo/*.json`;
`results/champions.json`; `figures/models/cross_corpus_*_heatmap.*`.

---

## 2026-07-16 · Transformer fine-tuning — determinism policy + methodology + findings

**Decisions.** (1) Fine-tune DistilBERT, BERT-base and RoBERTa-base on the three frozen
corpora (subject+body, `max_length=256`, truncation on). Per (model, corpus) grid: lr ∈
{2e-5, 3e-5, 5e-5}, up to 4 epochs, batch 32 + fp16 (grad-accum on OOM), weight_decay 0.01,
warmup_ratio 0.1. **Validation ROC-AUC each epoch; early-stop patience 1; best-val-AUC
checkpoint kept; the best lr is selected by val AUC only.** The **TEST split is evaluated
exactly once per finalist**, after selection. (2) Cross-corpus: champion per source = model
with the highest **val** AUC; ties at 4 d.p. (the corpora saturate near 1.0) broken toward the
**smaller/cheaper** model (distilbert < bert < roberta). Each champion's fine-tuned checkpoint
is applied unchanged to the other corpora's TEST splits (6 directed pairs); the diagonal reuses
the single official in-corpus test eval. (3) Environment: `torch==2.11.0+cu128` (the driver is
CUDA 13.1-capable, so any 12.x runtime runs on the RTX 3090), `transformers==4.57.6`,
`accelerate==1.14.0`.

**Determinism policy (not bit-exact — deliberately).** GPU fine-tuning here is **not
bit-reproducible**. The concrete cause: the memory-efficient / SDPA attention path used by
these encoders has **no deterministic CUDA kernel** for its backward pass — with
`torch.use_deterministic_algorithms(True)` it *raises*; we therefore run it in `warn_only=True`
mode. Mitigations that make runs *stable and auditable* rather than bit-identical:
- **Fixed seeds = 42** (`random`, `numpy`, `torch`, `torch.cuda`) set before every model init,
  so classifier-head init and data-shuffle order are reproducible and lr is the only difference.
- **`torch.backends.cudnn.deterministic = True`, `benchmark = False`** (deterministic cuDNN
  where a kernel exists; no autotuner nondeterminism).
- **`CUBLAS_WORKSPACE_CONFIG=":4096:8"`** pinned *before* the CUDA context (deterministic cuBLAS
  GEMMs).
- **Pinned library/CUDA versions**, recorded in every `results/transformers/*.json` (`versions`
  block: python/torch/transformers/cuda_build/cudnn/gpu).
- **Saved finalist checkpoints are the reproducibility anchor** — the exact evaluated weights are
  persisted (`save_pretrained`), so every reported number is re-verifiable from the checkpoint
  regardless of run-to-run FP noise. Checkpoints (~3.6 GB) are git-ignored as regenerable; only
  code, JSONs, figures and summaries are committed.

**Findings (evidence for the write-up).**
- **In-corpus at ceiling; transformers do not beat tuned linear TF-IDF.** Best-transformer vs
  classical champion test AUC: d1_sb 0.9999 vs 0.9998, d2 0.9999 vs 0.9996, d3_en 0.9979 vs
  0.9993. All ≥ 0.997; no meaningful in-corpus gain for ~1000× the compute.
- **Cross-corpus is a mixed, informative result — NOT a blanket transformer win.** Transformer
  champions beat the best classical champion on **4/6** directed pairs — **every pair involving
  the modern-real corpus d1_sb** (e.g. d3_en→d1_sb 0.966 vs 0.889, +0.077) — and **lose the two
  legacy-Enron↔AI pairs** (d2→d3_en 0.775 vs 0.843; d3_en→d2 0.731 vs 0.752).
- **The headline AI→real gap (d3_en→d2) is NOT closed** by contextual modelling: 0.731 lands
  inside the classical band (0.675–0.752) but −0.022 below the best classical (LinearSVC 0.752).
  The residual hard case is **legacy↔AI** transfer specifically, not AI→real in general.
- **Truncation @256 bites hardest on the AI corpus, not the legacy one** — AI-generated emails are
  the longest (d3_en train 75.95% truncated, median 364 WordPiece tokens) vs d1_sb 48.92%/247 and
  d2 36.53%/176. RoBERTa still tops d3_en in-corpus, so the signal survives the first 256 tokens.
- **Cost:** full 9-grid = **132.8 min** on one RTX 3090 (peak ≤ 5.2 GB, effective batch 32
  throughout, no OOM) vs seconds for the tuned linear champions.

**Links.** `src/experiments/transformers_ft.py`, `transformer_cross.py`, `transformer_report.py`;
`notes/transformer_summary.md`; `results/results.csv`; `results/transformers/*.json`;
`results/transformer_champions.json`; `figures/models/transformer_cross_*_heatmap.*`.
