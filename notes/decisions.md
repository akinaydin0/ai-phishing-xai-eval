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

**Cost figure note (canonical).** Two wall-clock numbers appear for the 9-grid: **132.8 min is
canonical** — the sum of the nine units' *training* wall-clock (three-lr loops + per-epoch val
scoring), the figure comparable to the classical champions' sklearn fit seconds. The **143.7 min**
end-to-end grid-script time additionally includes the per-unit tokenization + full untruncated
truncation-statistics passes, best-checkpoint `save_pretrained` I/O, and the single finalist test
evals — none of which are training. Stated in `notes/transformer_summary.md` §(d).

---

## 2026-07-16 · PRE-REGISTERED ablation (ablation week) · max_length=512 vs the legacy↔AI transfer losses

**Status: NOT YET RUN.** Registered *before* execution so the outcome cannot be reinterpreted
post-hoc. To be run in the ablation week.

**Hypothesis (falsifiable).** The transformers' *only* two cross-corpus losses vs the best
classical champion are the legacy-Enron↔AI directions — d2→d3_en (0.7748 vs 0.8429) and d3_en→d2
(0.7306 vs 0.7523) — and d3_en is by far the most truncated corpus at `max_length=256` (75.95% of
train docs truncated; median 364, p95 707 WordPiece tokens). **H1:** the 256-token window discards
discriminative content and *causes* the legacy↔AI transfer loss; raising the window to 512 (which
covers d3_en's median 364 and cuts truncation from ~76% to the fraction >512, well under half) will
recover a material share of the lost transfer AUC. **H0 (null):** the loss is representational /
distributional (1990s–2000s corporate Enron vs LLM-generated phishing are genuinely far apart in
token space — cf. pairwise Jaccard legacy↔AI **0.248 (cleaned)**, the lowest of the three
(legacy↔modern 0.353, modern↔AI 0.256)), not a context-window artifact, and 512 will not recover it.

**Method.** Re-run the affected units at `max_length=512`, everything else identical to the 256 run
(same frozen splits, seeds=42, lr grid {2e-5,3e-5,5e-5}, ≤4 epochs, val-AUC selection, single test
eval, resumable). Because train/eval tokenisation must stay identical (leakage-review Surface 4), a
512 evaluation of a champion requires that champion to have been *trained* at 512 — so the ablation
re-runs **both the d2 and d3_en units** at 512 (six units; d1_sb is untouched — all its transfers
already win and it is not part of H1), then re-evaluates the two legacy↔AI cross directions:
d3_en→d2 (512 d3_en champion on d2 test@512) and d2→d3_en (512 d2 champion on d3_en test@512).

**Pre-specified decision rule.** Comparing 512 vs the committed 256 numbers:
- **Supports H1** if *either* legacy↔AI direction recovers **≥ +0.03 AUC** at 512 **and** closes
  **≥ 50%** of its gap to the best classical champion (d3_en→d2 gap 0.0217 → target ≥ 0.742;
  d2→d3_en gap 0.0681 → target ≥ 0.809).
- **Supports H0 (refutes truncation-cause)** if *both* directions move **< +0.01 AUC** at 512.
- Between those bands = partial/inconclusive; report as such, no reinterpretation.
- Secondary, weak check: in-corpus d3_en test AUC at 512 (already ceiling ~0.997, little headroom —
  a null there does not by itself refute H1 for the cross directions).

**Cost / feasibility.** 512 ≈ 2–4× the per-step compute and activation memory of 256; the six
d2+d3_en units cost ~73 min at 256, so expect ~2.5–3 h at 512. Peak GPU ~2× (well under the 24 GB
ceiling; effective batch 32 retained, grad-accum fallback already implemented).

**Implementation notes (do at ablation time).** Parametrise `MAX_LEN` as a `--max-length` CLI arg in
`transformers_ft.py` / `transformer_cross.py`, and write to an **isolated namespace**
(`results/transformers_512/`, checkpoints under it, run_id suffix `__finetune512` /
`__xcorpus_transformer512`) so the canonical 256 results and `results/results.csv` rows are **not**
overwritten — the deliverable is a 512-vs-256 side-by-side, not a replacement.

**Links.** `notes/transformer_summary.md` §(b)/(c); `results/results.csv` (256 baseline rows);
this hypothesis tests the "legacy↔AI, not AI→real in general" finding logged in the entry above.

---

## 2026-07-16 · PRE-REGISTERED XAI evaluation protocol (Directive 7) — written BEFORE any results

**Status when written: NO XAI RESULTS EXIST YET.** This protocol, and the fixed cue lexicon
(`src/xai/cue_lexicon.py`, 206 terms across 6 categories), are committed *before* any attribution or
metric is computed, so faithfulness/stability/plausibility numbers cannot be reverse-fitted.

**Units (12).** Family champions per corpus (in-corpus, 9) + transformer transfer (cross, 3):
- **tfidf** (val-selected): d1_sb LogisticRegression, d2 LogisticRegression, d3_en **LinearSVC** —
  rebuilt from the stored HPO `best_params` (`results/hpo/*.json`), fit on the SOURCE train split.
- **ensemble**: tuned **LightGBM** per corpus (rebuilt from HPO best_params, early-stopped on val).
- **transformer** (saved checkpoints, `results/transformers/checkpoints/`): d1_sb DistilBERT,
  d2 DistilBERT, d3_en RoBERTa.
- **cross (transformers only, n=100):** d3_en champion (RoBERTa) on **d1_sb** test (success axis);
  d2 champion (DistilBERT) on **d3_en** test and d3_en champion (RoBERTa) on **d2** test (hard
  legacy↔AI axis). Question: does explanation quality degrade with transfer, and more on the hard axis?

**Model score `p`.** Faithfulness uses `p = p(predicted class)` — the probability the model assigns
its own predicted class on the ORIGINAL doc (class fixed from the original prediction). `predict_proba`
for LogisticRegression / LightGBM / transformers (softmax); **LinearSVC has no `predict_proba`, so
`p = sigmoid(decision_function)`** — a monotonic pseudo-probability (documented; comprehensiveness /
sufficiency are *differences* of the same monotone score, so calibration is immaterial, only monotonicity).

**Attribution granularity + the identical removal mechanic.** All attributions are reduced to the
raw-text **word** level under one fixed segmentation (`\w+`, unicode, with character spans). Removal
blanks the chosen words' spans in the raw text and **re-runs the full pipeline** (re-vectorise for
tfidf/LightGBM, re-tokenise for transformers) — so the mechanic is byte-identical across families.
Method→word reduction: (i) **tfidf/LightGBM** SHAP feature values (unigram+bigram vocab) are
distributed **equally to the constituent word positions** that generated each feature; (ii)
**transformer** sub-word attributions (SHAP text tokens / IG sub-tokens) are **summed within each
word span**. Top-k = the k word positions with the highest attribution **toward the predicted class**
(for methods giving class-1 attribution, oriented by `sign = +1 if pred==1 else −1`).

**Explanation set (per unit, TEST split ONLY).** n=200 in-corpus (100 per TRUE class); include ALL
misclassified docs up to **30** (by the unit's champion prediction), fill the remainder with correctly
classified docs holding the 100/100 balance; **seed 42**. Cross units: n=100 (50/class), misclassified
cap **15** (proportional to 30/200). Selected doc ids (sha1 body-hash keys) persisted to
`results/xai/explained_ids_<unit>.json`; the **same docs feed every method** on that unit.

**(1) Faithfulness — token-removal.** For k ∈ {1,5,10,20}, on `p = p(pred class)`:
- **comprehensiveness@k** = `p_orig − p(top-k attributed words REMOVED)` (higher = more faithful).
- **sufficiency@k** = `p_orig − p(ONLY top-k words kept)` (nearer 0 = more faithful).
- **AOPC** = mean over k=1..20 of comprehensiveness@k (per doc, then averaged).
- **MANDATORY random-attribution baseline** through the identical pipeline (random word ranking,
  seed 42): every faithfulness number is reported beside its random-baseline equivalent.

**(2) Stability.** Per doc, **5 perturbed variants**: randomly delete **5% of the NON-top-10 tokens**
(seeded). Metrics: mean **top-10 Jaccard** and mean **Spearman** of shared-token attributions
(original vs perturbed). Plus **LIME seed-agreement**: LIME with two seeds (42, 43), top-10 Jaccard.

**(3) Plausibility (heuristic proxy, NOT human annotation — stated as such in the summary).**
Cue-hit rate within the **top-10** attributed words on **phishing-PREDICTED** docs (pred==1), using
the fixed `cue_lexicon.py`; a word hits if it is any category term, or (url_email) its span overlaps a
URL/e-mail match. Reported against the random-attribution baseline.

**Method hyperparameters (fixed here).**
- **SHAP**: LinearExplainer (masker = Independent over a 1,000-row train-feature background) for
  linear; TreeExplainer (tree_path_dependent, exact) for LightGBM; **Partition/Text explainer** for
  transformers (word masker aligned to the removal `\w+` tokenizer, `max_evals=500`).
- **LIME**: `LimeTextExplainer(bow=True)`, `num_samples=1000`, split_expression `\w+`; main
  attribution seed 42; seed-agreement uses seeds {42, 43}.
- **IG** (transformers only): `LayerIntegratedGradients` on the input-embedding layer, **n_steps=32**,
  **baseline = PAD-token embedding** (choice recorded), target = predicted-class logit; sub-token
  attributions summed to words.
- Determinism: seed 42 for all sampling/perturbation; transformer forward passes batched on GPU
  (not bit-exact — same SDPA caveat as Directive 6).

**Cost control.** After the first (transformer, SHAP) unit, project full-plan wall-clock; if any
method projects **beyond ~6 h**, checkpoint Akın with options rather than silently trimming
(e.g. reduce SHAP `max_evals` / LIME `num_samples`, recorded if changed). Incremental per-unit JSON
persistence with resume, exactly like Directives 5–6.

**Outputs.** `results/xai_metrics.csv` (unit, corpus, family, method, n_explained, aopc,
comp@{1,5,10,20}, suff@{1,5,10,20}, stability_jaccard, stability_spearman, lime_seed_agreement,
plausibility@10, random-baseline equivalents, runtime_s); per-unit JSON `results/xai/<unit>.json`;
figures (deletion curves; method×family bars for comp@10/stability/plausibility; 6 qualitative
examples) PNG+PDF; `notes/xai_summary.md` with the RQ3 answer + 5 observations.

**Links.** `src/xai/` (`cue_lexicon.py`, `champions.py`, `explainers.py`, `metrics.py`, `run_xai.py`,
`report_xai.py`, `figures_xai.py`); `results/xai/`; `results/xai_metrics.csv`; `notes/xai_summary.md`.

---

## 2026-07-17 · XAI phase — results + headline findings (RQ3)

**Executed** the pre-registered protocol above: 12 units × their methods = **30 (unit, method)** runs,
n=200 in-corpus / 100 cross, on the RTX 3090 in **306 min** (5.1 h — within the ~6 h budget, so no
trimming; the trim-checkpoint was not triggered). Log: `results/xai_metrics.csv` (30 rows); figures in
`figures/xai/`; report `notes/xai_summary.md`. Per-unit attribution JSONs are **git-ignored** (18 MB,
regenerable via `python -m src.xai.run_xai --all`); the CSV + `explained_ids_*.json` are the committed
anchors.

**Headline findings (evidence for Ch6 / RQ3).**
- **SHAP is the only demonstrably faithful method; LIME ≈ the random baseline.** In-corpus AOPC:
  ensemble·SHAP **0.281**, tfidf·SHAP 0.161, transformer·SHAP 0.107 ≈ transformer·IG 0.102 — while
  every LIME (0.014–0.037) sits at the random-attribution floor (~0.017). The deletion curves show it:
  SHAP climbs, LIME hugs random.
- **Faithfulness–stability trade-off; no method wins both.** LIME is the *most stable* (top-10 Jaccard
  ~0.99, seed-agreement 1.00) but only because its bag-of-words fit is insensitive to which tokens
  matter; SHAP/IG are faithful but less stable (transformer·SHAP Jaccard 0.51). Report the two together.
- **Transformers are the hardest to explain faithfully** (SHAP/IG AOPC ~0.11, half the linear
  champions', a third of LightGBM's; SHAP Jaccard 0.51, the lowest). Diffuse contextual signal + softmax
  saturation. Among transformer explainers **IG ≈ SHAP faithfulness at ~4× lower cost + higher
  stability** → IG is the transformer method of choice.
- **Plausibility tracks faithfulness** (heuristic cue-lexicon proxy, *not* human annotation): SHAP
  cue-hit@10 beats random by +0.06…+0.16 on tfidf/ensemble; LIME sits at/below random. Densest cues on
  d3_en (AI phishing).
- **Cross-corpus:** apparent faithfulness *rises* on transfer (0.21 vs 0.11) — a low-confidence
  artifact (OOD predictions are unsaturated, so removals move p more), not better explanations. The
  honest signal is plausibility, which **collapses on the hard d3_en→d2 axis (0.059)** — legacy Enron
  carries few lexical cues — mirroring the Directive-6 legacy↔AI transfer breakdown.
- **RQ3 answer: linear TF-IDF + SHAP** — best (tied) in-corpus accuracy, second-most-faithful and
  joint-most-plausible explanations, at the **lowest cost of any combination** (tfidf·SHAP 6 s/unit vs
  transformer·SHAP 686 s, transformer·LIME 2408 s). LightGBM+SHAP is the most faithful but slightly less
  accurate and ~15× costlier; transformers+IG only where cross-corpus robustness forces a contextual model.

**Methodology notes.** (1) `p = p(predicted class)`; **LinearSVC uses sigmoid(decision_function)** (no
`predict_proba`) — monotone, so comp/suff (differences of the same score) are unaffected. (2) **Negative
sufficiency** for SHAP on tfidf/LightGBM is expected and *supports* faithfulness: keeping ONLY the top-10
words drives p(pred) *above* the full-document value (the rest of the doc dilutes the concentrated cue
signal). (3) Near-ceiling models saturate, compressing absolute comp — hence **comp-vs-random lift**, not
absolute comp, is the discriminating quantity. (4) Word-level removal is identical across families; SHAP
feature (uni/bigram) values distributed to constituent word positions, transformer sub-tokens summed to
words.

**Links.** `notes/xai_summary.md`; `results/xai_metrics.csv`; `figures/xai/xai_deletion_curves.*`,
`xai_method_family_bars.*`, `xai_qualitative_examples.*`; `src/xai/*`.

---

## 2026-07-17 · Independent XAI verification (PASS) + plausibility wording refinements

**Verification (PASS).** `results/xai_metrics.csv` was independently re-verified against the summary
tables and the raw log. Scope + results: **row accounting exact**; **288/288** summary-table cells
consistent at 3 dp — the 11 flagged cells are all exact 0.0005 half-boundary roundings of the 4 dp CSV
values; **all derived means exact**; **comp@k monotone** on every SHAP/IG row (only sub-0.009 wiggles on
the no-signal LIME/random rows); **calibration lift over random**: SHAP comp@10 min **+0.058 on 9/9**
units, IG min **+0.068 on 3/3**, LIME **−0.003 to +0.020**; **suff@10 ≤ random on 30/30** rows; runtime
sum **299.5 min** vs **306 min** wall clock (difference = per-unit build/tokenisation overhead outside the
timed metric blocks). No numeric defect found; the CSV stands.

**Two prose statements over-stated their own tables and were refined at source** (in
`src/xai/report_xai.py`, then `notes/xai_summary.md` regenerated; **tables unchanged**). These are now the
**canonical Ch6 phrasing**:
- **LIME plausibility is NOT "at or below random everywhere."** It is at/below random on d1_sb and d2, but
  on the cue-dense **d3_en** corpus LIME's cue-hit@10 **exceeds** random (+0.085…+0.111, and transformer
  LIME 0.253 even exceeds transformer SHAP 0.236) — while LIME stays **unfaithful** on d3_en (AOPC at the
  random floor). Framed as **"plausibility without faithfulness"**: LIME surfaces cue words because
  AI-generated text is saturated with them, but removing them does not move the model. This is the exact
  failure mode the three-metric design is built to expose, so it **strengthens** the argument.
- **"SHAP beats random on every family" holds at the family-mean level** (transformer 0.162 vs 0.121); the
  single per-corpus exception is **d1_sb transformer·SHAP at 0.133, 0.027 below** its 0.159 random base rate.
- **Cross-corpus d3_en→d2 collapse is two-layered:** absolute cue-hit falls to **0.059** largely because
  legacy Enron's **random cue base rate is only 0.026**; independently, SHAP's **lift over random shrinks
  from +0.094 in-corpus to +0.033 on transfer**. Both layers are reported together.

**Links.** `notes/xai_summary.md` (RQ3 bullet, Observations 4–5); `src/xai/report_xai.py`;
`results/xai_metrics.csv`.

---

## 2026-07-17 · Supervisor feedback (weekly report 1) logged

Recorded from the first weekly supervisor report so the actions are tracked and dated.

- **Overfitting concern → queued as Directive 10 (plausibility *validation*).** The supervisor
  suggested guarding against over-reading the heuristic cue-lexicon plausibility proxy. Planned
  response: a **method-blind human plausibility annotation** of the explanation top-10 sets
  (SHAP / LIME / IG / random, **~20 docs per corpus** drawn from the *existing*
  `results/xai/explained_ids_*.json` **TEST** sets), plus an **LLM-judge cross-check under the same
  blind protocol**. The annotation guide + agreement metrics (inter-rater + human-vs-LLM) will be
  **pre-registered in this log BEFORE any rating happens** — same discipline as the XAI
  pre-registration. This directly tests whether the lexicon proxy tracks human judgement.
- **LSEP (Legal/Social/Ethical/Professional): stays as Chapter 1 §1.4.** Supervisor approved either
  placement; keeping it in Ch1 §1.4, to be anchored in **relevant, DOI-verified** literature during
  the writing phase (honours supervisor feedback items 1 + 4).
- **Conference confirmed: DeSE 2026** — 19th International Conference on Developments in eSystems
  Engineering, **16–18 Dec 2026, Selangor Darul Ehsan, Malaysia**; IEEE Xplore proceedings.
  **Full-paper deadline 15/09/2026; camera-ready + author registration 15/10/2026.** Plan: distil the
  paper from the full dissertation draft in **August** (before the 31 Aug dissertation deadline gives
  a draft to compress).
- **Viva:** the online-viva question was redirected by the supervisor to programme leader **[name redacted]**;
  email sent, **answer pending** (follow up if no reply within a week).

**Links.** Directive 10 (to be pre-registered before ratings); `CLAUDE.md` (Supervisor feedback);
`results/xai/explained_ids_*.json` (source docs for the annotation subset).

---

## 2026-07-17 · 512 ablation result — verdict: **H1 (supported)**

Executed the binding 2026-07-16 pre-registration (hypothesis, method, decision rule fixed there;
no trimming, no reinterpretation). Six units {distilbert, bert-base, roberta-base} × {d2, d3_en} at
`max_length=512`, isolated namespace `results/transformers_512/` (**canonical `results/results.csv`
byte-identical, 124 rows unchanged**). Determinism policy identical to Directive 6. The script
(`transformer_512_report.py`) applies the decision rule **mechanically** and prints the verdict.

**Result (script-computed, verbatim rule).**

| Direction | 256 | 512 | Δ | 50%-gap target | best classical | Δ≥+0.03 | ≥target | H1? |
|---|--:|--:|--:|--:|--:|:--:|:--:|:--:|
| d2→d3_en | 0.7748 | 0.7642 | −0.0106 | 0.809 | 0.8429 | no | no | no |
| d3_en→d2 | 0.7306 | **0.8440** | **+0.1134** | 0.742 | 0.7523 | **yes** | **yes** | **YES** |

Rule: H1 if EITHER direction has Δ ≥ +0.03 AND AUC_512 ≥ its target. **d3_en→d2 satisfies both →
verdict = H1.** (H0 would need BOTH Δ < +0.01; not met. Secondary weak check: in-corpus d3_en test
AUC 0.9984 @512 vs 0.9979 @256 — negligible, does not decide.)

**Interpretation (the finding, not a rule override).** The recovery is **one-directional and keyed to
the SOURCE corpus's truncation**: **d3_en→d2 jumps 0.731 → 0.844** — now **above the best classical
champion (0.752)**, reversing the Directive-6 loss — because at 512 the d3_en champion trains on
**fuller AI-phishing emails** (d3_en train truncation **76% → 22%**). **d2→d3_en does not move**
(−0.011): d2's training docs are short (median 176 WP, already ~untruncated at 256), so 512 adds no
training signal in that direction. So the Directive-6 "legacy↔AI transfer breaks down" is, for the
d3_en→d2 half, **substantially a 256-token truncation artifact of the long AI corpus**, not purely
distributional — a nuance the dissertation should state precisely (H1 holds, but directionally).

**Cost.** 6 units = 145 min on the RTX 3090 (~2.0× the 73 min at 256), peak ≤ 8.7 GB, no OOM.
Checkpoints git-ignored (`results/transformers_512/checkpoints/`).

**Links.** `notes/transformer_512_summary.md`; `results/results_512.csv`;
`results/transformers_512/*.json`; `results/transformer_champions_512.json`;
`figures/models/transformer_512_cross_comparison.*`; pre-registration entry above (2026-07-16).

---

## 2026-07-17 · PRE-REGISTERED Directive 9 — statistics + imbalance + preprocessing ablation

**Status when written: NO D9 RESULT EXISTS.** Committed alone, before any statistic/metric, as the
integrity anchor. Champion identities + expected test AUCs are **verbatim from the committed records**
(`results/results.csv`, `results/results_512.csv`, this log) — **nothing is reselected**. CIs/p-values
**annotate** the already-committed comparisons; **no champion, threshold, or claim direction may be
introduced or reversed on their basis.**

### A. The seven statistical comparisons (exact; no additions afterwards)

Model-a vs model-b, paired on the stated TEST split. Committed champion identities/AUCs:

| id | test split | model_a | AUC_a | model_b | AUC_b |
|---|---|---|--:|---|--:|
| C1a | d1_sb test | LinearSVC (tfidf, in-corpus classical champ) | 0.9998 | DistilBERT (256 in-corpus champ) | 0.9999 |
| C1b | d2 test | MultinomialNB (tfidf, in-corpus classical champ) | 0.9996 | DistilBERT (256 in-corpus champ) | 0.9999 |
| C1c | d3_en test | LinearSVC (tfidf, in-corpus classical champ) | 0.9993 | RoBERTa (256 in-corpus champ) | 0.9976 |
| C2 | d2 test | RoBERTa 512 champ (d3_en→d2) | 0.8440 | RoBERTa 256 champ (d3_en→d2) | 0.7306 |
| C3 | d2 test | RoBERTa 512 champ (d3_en→d2) | 0.8440 | LinearSVC tfidf classical cross champ | 0.7523 |
| C4 | d3_en test | LogisticRegression tfidf classical cross champ (d2→d3_en) | 0.8429 | DistilBERT 256 champ (d2→d3_en) | 0.7748 |
| C5 | d3_en test | DistilBERT 512 champ (d2→d3_en) | 0.7642 | DistilBERT 256 champ (d2→d3_en) | 0.7748 |

The in-corpus classical champions are the committed in-corpus leaderboard #1 (LinearSVC / MultinomialNB /
LinearSVC, test AUC 0.9998 / 0.9996 / 0.9993 — `notes/transformer_summary.md` §(a)); the 256 transformer
in-corpus champions are the committed val-selected finalists (`transformer_champions.json`: d1_sb/d2
DistilBERT, d3_en RoBERTa); cross champions are the committed rows in results(_512).csv. **C5's CI is
expected to include 0** (512 is 0.011 *below* 256 in that direction).

### B. Statistical methods (fixed)

- **Paired stratified bootstrap** for AUC differences: **B=10,000**; test indices resampled **within class**
  (positive and negative counts preserved), **percentile 95% CI** on Δ=AUC_a−AUC_b; base **seed 42**; the
  **identical resampled index set** is used for both models of a pair (same cached score vectors).
- **Single-AUC 95% CIs** (same bootstrap config) for the **12 headline numbers**: the 6 in-corpus finalists
  (classical + transformer × {d1_sb, d2, d3_en}) and the 6 cross numbers (256/512 transformer × {d3_en→d2,
  d2→d3_en} = 4, plus the 2 classical cross champions).
- **McNemar exact binomial, two-sided**, at the fixed operating point: discordant counts **n01** (a wrong,
  b correct) and **n10** (a right, b wrong); raw p = exact two-sided binomial(min(n01,n10); n01+n10, 0.5);
  **Holm-adjusted** across the **7-test family** {C1a,C1b,C1c,C2,C3,C4,C5}.
- **Operating point:** probability-like score **≥ 0.5**. Probabilistic models use `predict_proba[:,1]`;
  margin models (LinearSVC) use `sigmoid(decision_function)` (≥0.5 ⇔ decision_function ≥ 0) — the identical
  mapping pre-registered in the XAI phase.
- **Interpretation rule (binding):** CIs and p-values **annotate**; they do not reselect champions,
  thresholds, or claim directions.

### C. Imbalance stress test

- **Eval-side prevalence stress** on **cached test scores** (no new model runs). Ratios **1:9, 1:19, 1:49**
  (phish:legit → 10% / 5% / 2% prevalence). Per draw: keep **ALL** legitimate test docs, subsample positives
  **without replacement** to the ratio; **50 draws** per (corpus, ratio), **seeds 42..91**. Families: the
  in-corpus **classical finalist** and the **256 transformer finalist** per corpus.
- **Metrics per draw** (fixed list): PR-AUC (average precision), ROC-AUC (sanity), precision/recall/F1 at the
  **0.5** operating point, **precision at 95% recall**, and **false positives per 1,000 legitimate emails** at
  the per-draw threshold achieving ≥95% recall. Report **mean** and **2.5 / 97.5 percentiles** over the 50 draws.
- **Train-side check, classical only:** retrain the classical finalist per corpus with **TRAIN positives
  subsampled** to each ratio (champion **hyperparameters frozen — no re-tuning**, a stated limitation),
  evaluate **ONCE** on the natural TEST split per (corpus, ratio) = **9 pre-registered single-use test
  evaluations**, no selection among them. Transformers excluded from train-side (cost) — recorded as future work.

### D. Pipeline A preprocessing ablation — VALIDATION ONLY (zero test contact)

- **Universe:** the classical champion representation per corpus (LinearSVC/MNB/LinearSVC + their committed
  tuned TF-IDF vectorizer params), **hyperparameters frozen** — the ablation cannot change any champion.
- **Variants** (deterministic; pinned **symspellpy 6.10.0**, **wordninja 2.0.0**, **scikit-learn 1.8.0**):
  - **V0** canonical cleaned text (baseline).
  - **V1** = V0 + **stopword removal** (sklearn `ENGLISH_STOP_WORDS`; drop tokens whose lowercase ∈ the set).
  - **V2** = V0 + **placeholders**, applied in order on the string: URL `https?://\S+|www\.\S+`→`urltoken`;
    email `[\w.+-]+@[\w-]+\.[\w.-]+`→`emailtoken`; standalone number `\b\d+\b`→`numtoken`.
  - **V3** = V0 + **spelling normalisation**: SymSpell `max_edit_distance=1`, pinned default frequency dict
    `frequency_dictionary_en_82_765.txt`; only tokens that are **alphabetic and length ≥ 4** are looked up
    (`lookup(..., Verbosity.TOP, max_edit_distance=1, include_unknown=True)`), replaced by the top suggestion.
  - **V4** = V0 + **segmentation**: `wordninja.split` on tokens of **length ≥ 12**, rejoined with spaces;
    shorter tokens unchanged.
  - **V5** = combined, applied in the fixed order **V2 → V4 → V3 → V1** (placeholders on the string; then on
    whitespace tokens: segment ≥12, spell ≥4, drop stopwords; rejoined with single spaces).
- **Protocol:** train the frozen champion on the TRAIN split transformed by the variant; evaluate on **VAL
  only** — own-corpus val **plus the other two corpora's val** — applying the **same** variant transform to the
  eval text. **No test split is loaded anywhere in Part E.**
- **Endpoint:** **val ROC-AUC Δ vs V0** per (train corpus, eval corpus, variant); descriptive, **no
  significance testing**. **Directional hypothesis (recorded now): placeholders (V2) help cross-corpus transfer.**

### Reproduction + isolation (Parts B/F)

Per-doc TEST scores for every model in the 7 comparisons are recomputed from stored artefacts (classical:
stored HPO params → deterministic CPU retrain; transformers: saved 256/512 checkpoints) and each recomputed
test ROC-AUC is **asserted == its committed value to 4 dp (HARD-FAIL otherwise)**. Score vectors cached under
`results/stats/scores/` (git-ignored); only aggregates committed. **Nothing is written to
`results/results.csv` or `results/results_512.csv`.** All D9 outputs live in `results/stats/`,
`results/imbalance/`, `results/preproc_ablation/`.

**Links.** `src/experiments/d9_*` (to be written); `results/stats/`, `results/imbalance/`,
`results/preproc_ablation/`; `notes/d9_stats_robustness_summary.md`.
