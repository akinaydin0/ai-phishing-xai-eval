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

---

## 2026-07-17 · D9 result — statistics + imbalance + preprocessing (annotations, no reselection)

Executed the binding pre-registration above. All 12 score vectors recomputed from stored artefacts and
**asserted == committed test AUC to 4 dp (12/12 PASS)**. Canonical `results/results.csv` (124) and
`results_512.csv` (8) **byte-identical**; all D9 output in `results/stats|imbalance|preproc_ablation/`.

**Statistics (7 comparisons, paired stratified bootstrap B=10,000 + Holm-McNemar).**
- Sign convention **Δ = classical − transformer** (pre-reg model_a=classical for C1a/b/c). **C1a d1_sb**:
  Δ −0.0001, **CI [−0.0002,+0.0000] includes 0** — indistinguishable. **C1b d2**: Δ −0.0003, CI
  [−0.0005,−0.0001] excludes 0 (a ~0.03 pp edge to the transformer; Holm-McNemar p=0.066). **C1c d3_en:
  Δ +0.0017, CI [+0.0006,+0.0030] SEPARATED (Holm p=0.010) — the classical LinearSVC significantly *beats*
  RoBERTa on the AI corpus.** So D6's "match but don't beat" is now precise: at the 0.999 ceiling the
  in-corpus gaps are ≤0.0003 either way, and the only material significant one favours the classical model.
- **C2** (512 vs 256, d3_en→d2): Δ **+0.1134** [+0.1042,+0.1229], Holm p≈0. **C3** (512 vs best classical
  cross): Δ **+0.0917** [+0.0784,+0.1052], Holm p≈0. The Directive-8 H1 recovery **survives paired
  inference** and significantly beats the classical cross champion.
- **C4** (classical cross vs 256 transformer, d2→d3_en): Δ +0.0681 [+0.0534,+0.0832] separated (classical
  wins). **C5** (512 vs 256, d2→d3_en): Δ −0.0106, **CI [−0.0223,+0.0009] includes 0 — the pre-registered
  null held** (Holm-McNemar p=0.030 flags a 0.5-threshold difference but the AUC ranking is not separated).

**Imbalance (eval-side 50 draws + train-side 9 single-use).** PR-AUC stays ≥0.93 but the operational
**FP-per-1,000-legit @95% recall** balloons as phishing gets rarer (train-side d3_en 19.5→**103.6** from
10%→2%). The more-robust family is **corpus-dependent**: transformer far cleaner on d2 (0.1 vs classical
1.7 FP/1,000 @2%), classical cleaner on d3_en (8.3 vs 14.7) — consistent with C1c. The FP/1000 framing is
the deployment signal ROC-AUC (prevalence-inflated) hides.

**Preprocessing ablation (VALIDATION only, frozen champions — cannot change any champion).** In-corpus Δ
vs V0 ≈ 0 (ceiling). Cross-corpus: **placeholders (V2) HELD the pre-registered hypothesis — mean Δ
+0.0343 over the 6 cross pairs, 5/6 improved** (up to +0.077 d2→d1_sb); **stopword removal (V1) HURTS
transfer** (mean −0.039, e.g. d2→d1_sb −0.111); V3/V4 ~neutral; V5 mixed (dragged by V1). Placeholders
normalise corpus-specific URL/email/number surface forms that don't transfer — a concrete,
validation-supported lead for a future retrain (not done here; frozen-champion limitation stated).

**Interpretation rule honoured:** every CI/p-value annotates a pre-committed comparison; no champion,
threshold or claim direction was introduced or reversed. Scoped review (reviewer + confirmer) PASS on
all 5 surfaces (no selection from test stats; pre-reg precedes results; Part E train+val only;
results.csv/results_512.csv byte-identical; 12/12 reproduction asserts).

**Links.** `notes/d9_stats_robustness_summary.md`; `results/stats/stats_summary.csv`;
`results/imbalance/imbalance_{eval,train}.csv`; `results/preproc_ablation/preproc_ablation.csv`;
`figures/stats/imbalance_curves.*`; `src/experiments/d9_{scores,stats,imbalance,preproc,report}.py`.

---

## 2026-07-18 · Independent D9 verification (PASS) + precision fixes

**Verification (PASS).** D9 was independently re-verified: **7/7 McNemar exact p-values** and **7/7
Holm-adjusted values** reproduce; **all 26 AUC anchors == committed** (12 score-vector AUCs + the 7×2
comparison endpoints); **imbalance counts == floor(frozen-split legit / ratio) on 9/9**; **preproc deltas
internally consistent 54/54**. No arithmetic defect. Three precision refinements below are now the
**canonical Ch6 phrasing** (they supersede any looser wording in the 2026-07-17 D9 result entry).

**1. C1a confidence interval — do NOT call it "CI includes 0".** The 4-dp CSV stored ci_high as `-0.0`,
which is a *rounded tiny negative*. The **raw bootstrap upper bound is −2.3111694852304143e-05** (ci_low
−0.00021880386609292036); `stats_summary.csv` now carries **6 dp** (C1a CI **[−0.000219, −0.000023]**).
Canonical C1a verdict: **nominally separated at the AUC level by ~1 bp (upper bound just below zero, so the
95% CI excludes 0 at full precision), NOT separated at the threshold level (Holm-McNemar p=0.174),
practically indistinguishable at the 0.999 ceiling.**

**2. Separate the AUC-level and threshold-level tests wherever they diverge** (a McNemar p is never cited
inside a claim it does not support):
- **C5:** *AUC-level* — CI [−0.0223, +0.0009] includes 0, the pre-registered null holds at AUC level.
  *Threshold-level* — Holm-McNemar **p=0.030 IS significant, favouring the 256 model** (n01=266 vs n10=209)
  at the fixed 0.5 operating point.
- **C1b:** *AUC-level* — CI [−0.0005, −0.0001] excludes 0 (a ~3 bp edge to the transformer). *Threshold-level*
  — Holm-McNemar p=0.066, not significant.

**3. Champion identities (C1a/C1b/C1c) — quoted + disclosed.** (a) The D9 pre-registration fixed the classical
side as the committed in-corpus leaderboard #1: *"C1a … LinearSVC (tfidf, in-corpus classical champ) 0.9998
… C1b … MultinomialNB (tfidf, in-corpus classical champ) 0.9996 … C1c … LinearSVC (tfidf, in-corpus classical
champ) 0.9993"*. (b) Committed definitions: the `svc` family is **`sklearn.svm.LinearSVC`** (linear SVM, NOT
kernel SVC — `hpo.py` `make_model('svc') → LinearSVC(...)`); per-corpus tuned classical, **test** AUC
(results.csv): d1_sb LinearSVC 0.9998 / LR 0.9997 / NB 0.9994; d2 **MultinomialNB 0.9996** / LR 0.9995 /
LinearSVC 0.9994; d3_en LinearSVC 0.9993 / LR 0.9982; **val** AUC (hpo_summary tuned): d1_sb LR 0.9997 =
LinearSVC 0.9997 > NB 0.9994; **d2 LR 0.9991 = LinearSVC 0.9991 > NB 0.9990**; d3_en LinearSVC 0.9989 > LR
0.9975. So the pre-registration deliberately used the **strongest committed classical (test champion)** on
the classical side — material for **d2** (NB is the test champion but not val-selected) — while the
transformer side is val-selected. **Disclosure (added to the summary):** *"Classical side = strongest
committed classical result per corpus (test champion); transformer side = val-selected finalist. This
asymmetry biases the in-corpus comparison toward the classical model, making the transformer's d1_sb/d2
parity conservative; C1c is unaffected (d3_en's val- and test-selected classical coincide)."* No champion
was re-run/reselected — the pre-registration and the interpretation rule (annotate, never reselect) stand;
the asymmetry is disclosed, and it only makes the transformer's near-parity **more** conservative.

**Links.** `results/stats/stats_summary.csv` (6-dp CIs); `notes/d9_stats_robustness_summary.md` §(a)/(b)/obs
1,3; `src/experiments/d9_{stats,report}.py`.

---

## 2026-07-18 · Directive 10 PRE-REGISTRATION — HITL + LLM-judge plausibility validation

**Committed FIRST and ALONE (this entry + `notes/hitl_annotation_guide.md` + `results/hitl/selected_ids.json`
+ the Part-A scripts `src/experiments/d10_{anchor,select}.py`), before any rating exists and before any
judge call.** Purpose: validate the cue-lexicon plausibility proxy — the XAI phase's one stated limitation —
with (i) method-blind human annotation by the author and (ii) an LLM judge under the identical blind
protocol. **Interpretation rule (binding): the bands and expectations below are REPORTING LANGUAGE ONLY —
nothing in the committed XAI results (`results/xai_metrics.csv`, 30 rows) is reselected, reversed, or
re-run by this phase.**

**Guard (Part A.0, PASS 2026-07-18):** `fd274b9` ancestor of HEAD; tree clean; canonical row counts
`results.csv`=124, `results_512.csv`=8, `xai_metrics.csv`=30.

**Regeneration integrity anchor (Part A.1, PASS 2026-07-18) — `src/experiments/d10_anchor.py`.** The
git-ignored attribution JSONs are regenerable; proven BEFORE any subset was extracted. Regenerated
attributions with the committed XAI parameters (SHAP `max_evals=500`, LIME `num_samples=1000`, IG 32 steps,
seed 42) for two anchor units on their FULL committed `explained_ids` sets and recomputed
`plausibility@10`, asserting equality with `results/xai_metrics.csv` to 4 dp: **`tfidf_d1_sb·shap`
0.2338==0.2338; `transformer_d3_en·shap` 0.2360==0.2360; `transformer_d3_en·ig` 0.2180==0.2180 — all
PASS** (only attributions + plausibility are recomputed; the canonical JSONs/CSV are left untouched).

### A. Sample (fixed) — `src/experiments/d10_select.py` → `results/hitl/selected_ids.json`
From the committed `explained_ids` TEST sets only. **Per corpus (d1_sb, d2, d3_en): 20 phishing-PREDICTED
docs = 15 true positives + up to 5 false positives (fill with TPs if fewer FPs exist). 60 docs.**
Deterministic: candidates **sorted by doc id (sha1 body hash) ascending, stratified first-N**; the underlying
`explained_ids` were built with the global **seed 42** (recorded for provenance — no further stochastic step).
`doc_uid = "{corpus}_{hash[:12]}"`.
- **d1_sb / d2:** source = the tfidf champion's `explained_ids` (`tfidf_d1_sb`, `tfidf_d2`); phishing-PRED =
  tfidf pred==1. Only the **tfidf** family is annotated.
- **d3_en:** source = the **INTERSECTION** of `tfidf_d3_en` and `transformer_d3_en` `explained_ids`,
  restricted to docs **phishing-PRED by BOTH** champions. Both families (tfidf, transformer) are annotated,
  so every d3_en doc is a committed-explained, phishing-predicted doc for BOTH families — keeping every
  generated set a top-10-toward-phishing set on a committed-explained doc (surface S1). Realised counts:
  every corpus 15 TP + 5 FP = 20; 60 globally-unique docs.

### B. Sets per doc (fixed)
- **tfidf family, all 3 corpora:** SHAP top-10, LIME top-10, RANDOM top-10 → **3 sets/doc, 180 sets.**
- **transformer family, d3_en only** (AI corpus + the lexicon's LIME-above-random anomaly + IG coverage):
  SHAP, LIME, IG, RANDOM → **4 sets/doc, 80 sets.**
- **260 sets total, ≤2,600 token judgments.** A "top-10" set = the word tokens at
  `textops.top_k_positions(attr, pred=1, min(10, n_word_tokens))` (predicted class is phishing for every
  selected doc), in most-attributed-first order. **RANDOM** = `min(10, n)` distinct word positions drawn
  **uniformly over the doc's `\w+` word tokens** via `numpy.random.default_rng([42, int(hash[:8],16),
  FAMILY_CODE])` with `FAMILY_CODE`={tfidf:0, transformer:1} (a per-(doc,family) seed-42-anchored,
  processing-order-independent draw). Set tokens = the raw-text word strings at those positions.

### C. Blinding (fixed)
Within each **(doc, family)** group the sets are shuffled via `numpy.random.default_rng([42,
int(hash[:8],16), FAMILY_CODE, 99])` and labelled **Set A/B/C(/D)**. The method key is deterministic and
committed at generation time in **`results/hitl/key.json`** (per `set_uid`: `doc_uid, corpus, family,
method, label, positions[], tokens[]`); the annotator must **not** open it until `human_ratings.csv` is
committed. **`set_uid = "{doc_uid}__{family}__{label}"`** (family is a grouping, not blinded; the METHOD is
blinded). **Single-annotator (the author) + procedural-only blinding are DISCLOSED LIMITATIONS.**

### D. Annotation guide
`notes/hitl_annotation_guide.md` (committed with this entry). A "genuine cue" = a token that, in the context
of this specific email, is plausible evidence of phishing intent (six **concept-level** families only —
urgency/pressure, credentials/verification, money/lure, links/technical artifacts, impersonation/authority,
secrecy). The guide **does NOT reproduce the 206 lexicon terms**, so human ratings are not a re-application
of the lexicon. Ratings: per token `is_cue`∈{0,1}; per set one holistic score *"does this token set explain
why the email is phishing?"*∈{0,1,2}.

### E. LLM judge (fixed) — model pinned BEFORE any call
- **Model id (pinned): `claude-opus-4-8`** (alias). Mechanism: Anthropic API, **temperature 0**, one call
  per set, JSON output. Rationale: an independent, maximally-capable second annotator; ~260 short calls.
- **Judge prompt (verbatim).** *System prompt* = the full text of `notes/hitl_annotation_guide.md`, verbatim.
  *User message* (`{email_text}` = the same subject+body text the tokens were attributed from; tokens
  presented 1..N in set order):
  ```
  EMAIL (subject and body):
  <<<
  {email_text}
  >>>

  TOKEN SET (rate each token in order):
  1. {token_1}
  2. {token_2}
  ... (through N)

  Return ONLY a JSON object, no prose, with exactly these keys:
  {"tokens": [<is_cue for token 1>, ..., <is_cue for token N>], "holistic": <0, 1, or 2>}
  where each is_cue is 0 or 1 and the "tokens" array has exactly N entries in the given order.
  ```
  (The directive's `{token: 0/1}` is realised as an index-aligned `tokens[]` array to disambiguate repeated
  token strings; this is the faithful, more-precise form.)
- **Additions baked in at the author's direction (2026-07-18), part of this pre-registration:**
  1. **Determinism caveat (DISCLOSED):** temperature 0 is greedy decoding but the API does not guarantee
     bit-exact reproducibility across calls/time. **All raw judge responses are committed verbatim**
     (`results/hitl/judge_ratings.csv` + a raw JSONL `results/hitl/judge_raw.jsonl` of the full responses),
     so every downstream number is exactly reproducible from committed artifacts; only re-querying the judge
     is subject to residual nondeterminism / model-alias drift. **The resolved model id (from the API
     response metadata) and call timestamps are recorded** in `judge_raw.jsonl`.
  2. **Isolation:** each call contains ONLY the fixed system prompt (the guide verbatim) + one blind set — no
     repo context, no lexicon, no `key.json`, no other sets. **Retry policy (fixed):** on invalid JSON, up to
     **2 retries** with the identical prompt; if still invalid, the set is recorded as **`judge_missing`**
     (excluded from P3, counted and disclosed).
  3. **Provenance disclosure (DISCLOSED LIMITATION):** the judge shares a model family with the tooling that
     authored the cue lexicon; **the human annotator is the independent reference and P2 (lexicon vs human)
     is the primary validation metric.**
  4. **Cost/PII:** ~260 calls on public research corpora (Kaggle/Enron/PhishFuzzer); no personal data beyond
     what the public datasets already contain.
- **Runtime rule (fixed):** if the API **rejects the pinned alias** `claude-opus-4-8` at runtime, **STOP and
  checkpoint** — no other model id is substituted without the author's approval.
- The annotator must not open `judge_ratings.csv` before his ratings are committed; the judge never sees
  human ratings.

### F. Metrics (computed only AFTER the ratings commit)
- **P1 (primary)** — per (corpus, family, method): **human cue-rate@10** (mean over the set's tokens of
  `is_cue`) with **paired bootstrap 95% CIs over docs** (B=10,000, seed 42); **ranking concordance vs the
  lexicon** = sign agreement on **SHAP-vs-RANDOM, LIME-vs-RANDOM, SHAP-vs-LIME**, and the **IG pairs**
  (IG-vs-RANDOM, IG-vs-SHAP, IG-vs-LIME on d3_en transformer).
- **P2** — token-level **lexicon-vs-human** agreement: **percent agreement + Cohen's κ**, per corpus and
  overall (lexicon label per token = `cue_lexicon.is_cue(word)` OR the token's char span overlaps a
  URL/e-mail span, identical to the committed metric's `_cue_hits`).
- **P3** — **human-vs-LLM:** token-level **κ**; **set-level Spearman** on cue-rates and on holistic scores.
- **P4** — holistic means per method vs RANDOM.
- **Interpretation bands (fixed now, Landis–Koch):** κ **≥ 0.4** → *"moderately validated proxy"*; **0.2–0.4**
  → *"fair — proxy retained as a heuristic with the limitation restated"*; **< 0.2** → *"poor — proxy
  reported as weak"*.
- **Directional expectations (fixed):** **E1** human SHAP cue-rate > RANDOM on every (corpus, family);
  **E2** lexicon–human κ ≥ 0.4.
- **Exploratory (no gate):** does the d3_en LIME-above-random anomaly replicate in human ratings?

### Package / gate / analysis / review (fixed plan)
- **Part B:** `src/experiments/d10_package.py` builds `results/hitl/{annotate.html, blind_sheets.csv,
  key.json}` (attributions regenerated from the anchored pipeline for the 60 selected docs; asserts every
  `doc_uid` ⊂ committed TEST `explained_ids`). `src/experiments/d10_judge.py` runs the pinned judge →
  `judge_ratings.csv` + `judge_raw.jsonl`.
- **Part C (human gate, STOP):** the author reads the guide, annotates in `annotate.html` (~1.5–2 h; does
  NOT open `key.json`/`judge_ratings.csv`/`cue_lexicon.py`), exports `human_ratings.csv` → schema
  `(doc_uid, set_uid, token_idx, token, is_cue, holistic)`. On resume: verify completeness, **commit the
  ratings file untouched BEFORE any analysis**.
- **Part D:** join ratings + judge + key + lexicon labels; compute P1–P4 under the fixed bands; write
  `notes/hitl_summary.md` + one figure `figures/hitl/hitl_cue_rates.(png+pdf)` (grouped bars: human vs
  lexicon vs LLM cue-rate per (corpus, method)).
- **Part E (review + commit):** scoped adversarial review (1 reviewer + 1 confirmer) on four surfaces —
  **S1** every annotated doc ⊂ committed TEST `explained_ids`; **S2** this pre-registration commit precedes
  the ratings commit and every judge call (git + timestamps); **S3** the analysis consumed the committed
  `human_ratings.csv` byte-identically; **S4** metrics/bands/expectations match this pre-registration
  verbatim.

**Links.** `notes/hitl_annotation_guide.md`; `results/hitl/selected_ids.json`;
`src/experiments/d10_{anchor,select}.py`; `src/experiments/d10_{package,judge}.py`;
`results/hitl/{annotate.html,key.json,blind_sheets.csv}`.

---

## 2026-07-24 · Directive 10 — pre-data amendments (BEFORE any judge call)

Disclose-don't-rewrite: the pre-registration entry above (§A–F + fixed plan) is **untouched**. Both
amendments below were committed **before any successful judge call and before any rating exists**
(git order: these precede the first `judge_raw.jsonl` line and `human_ratings.csv`). Deviations are
recorded as separate dated notes, never edited into the protocol. **Bookkeeping disclosure (Part E
review finding):** the pre-registration entry's trailing `**Links.**` file-pointer line *was* edited
in place at `9a32bb7` (dropping the then-stale parenthetical "(Part B, uncommitted at this point)" and
adding the new artifact paths). Verified scope: that line and the section separator are the **only**
differences in §A–F between `8ef0937` and HEAD — no sample rule, set definition, blinding rule, judge
prompt, metric, band or expectation was touched. Future amendments append only. Timeline: Parts A + B.3 and
Amendment 1 were done **2026-07-18**; the session then paused at the API-key handover and resumed
**2026-07-24**, when Amendment 2 and the judge run occurred (git commit timestamps confirm the
6-day gap; the first successful judge call `req_ts` 2026-07-24T12:20:53Z follows the Amendment 2
commit `3f6edc8` at 12:20:37Z).

### Amendment 1 — information-parity / blinding fix (2026-07-18, committed `9a32bb7`)
Author-directed pre-flight: confirm the human (annotate.html) and the LLM judge rate under
**identical information conditions** — the email subject+body plus the one blind token set (the
guide's cue definition is context-dependent).
- **Judge prompt template CONTAINS the email text** (the named failure did NOT occur): the committed
  template (§E) fills `{email_text}` from the same `text` field the human sees. Verified empirically
  across all **60 docs / 260 sets**: 0 email-text and 0 token-set mismatches between the judge source
  (`key.json`) and the human source (the `annotate.html` payload); a rendered sample confirms
  subject+body + the blind set are both present. **No template change.**
- **Conformance fix to annotate.html.** The check surfaced that annotate.html additionally displayed
  a **ground-truth stratum badge ("true positive"/"false positive") and a corpus badge** —
  information the isolated judge does NOT receive, which breaks identical-information-conditions and
  would **bias** the human on the 5 FPs/corpus (corrupting P3). The pre-registered spec (§D / Part
  B.3) says annotate.html shows only *"each email's subject+body and its blind sets"*, so the badges
  were a deviation. **Fix:** `stratum`/`corpus` removed from the annotate.html DATA and UI; the human
  now sees email text + blind sets only, ground-truth-blind like the judge. `key.json` and
  `blind_sheets.csv` regenerate **byte-identical** (fix isolated to the human tool; sets/key/
  blinding/attributions untouched).
- **Disclosed residual asymmetry (pre-registered, not a defect):** the judge rates **one isolated set
  per call** (§E addition 2); the human sees an email's sibling sets together. A token's cue status
  depends on the token + email context (not sibling sets); recorded as a limitation.

### Amendment 2 — pinned model retained; temperature parameter deprecated (2026-07-24, committed `3f6edc8`)
At the first call the Anthropic API **accepted** the pinned model **`claude-opus-4-8`** but
**rejected the pre-registered `temperature 0`**, returning verbatim:
```
HTTP 400  invalid_request_error  —  message: "`temperature` is deprecated for this model."
```
Per the author's decision (2026-07-24) the pinned model is **retained — do NOT substitute** (a
harness/sub-agent judge is also rejected: only direct API calls satisfy the §E isolation clause with
a pinnable model id), and the judge is called with **NO sampling parameters sent at all**, one call
per set exactly as pre-registered.
- **Rationale:** reproducibility was already anchored by Addition 1 (§E) — every raw response is
  committed verbatim, so every downstream number reproduces from `judge_ratings.csv` +
  `judge_raw.jsonl` regardless of sampling. The temperature knob was redundant; the judge is a second
  annotator and annotator stochasticity is standard. The committed single run is the analysis artifact.
- **Cost:** the constant system block (the guide) is sent with `cache_control` (prompt caching) across
  the 260 sequential calls; call structure otherwise unchanged (guide + one blind set).
- **Limitations line (to appear in `notes/hitl_summary.md`):** *judge ratings are a single stochastic
  draw at the model default; re-querying could differ, but all reported analyses reproduce exactly
  from the committed `judge_ratings.csv` + `judge_raw.jsonl`.*
- No rating and no successful judge call existed at amendment time.

---

## 2026-07-25 · Directive 10 — Amendment 3: annotation-integrity disclosure (post-judge, BEFORE annotation)

Original pre-registration (§A–F) and Amendments 1–2 untouched. Disclosed by the author before any
human annotation exists (`human_ratings.csv` not yet created).

**What happened.** During the Part B.4 judge run, the orchestrator's (Claude's) progress reports to
the single annotator (the author) inadvertently disclosed, **before any human annotation**, for **one
document — `d1_sb_017547c4cef2`** (the "CNN.com Daily Top 10" newsletter; d1_sb, stratum TP): (a) the
**method identity of all three of its tfidf sets** (Set A = RANDOM, Set B = LIME, Set C = SHAP), and
(b) the **LLM judge's all-zero verdict** on two of its sets. This **procedurally broke the
single-annotator method-blinding for that one document** (1 of 60). It was a reporting slip during a
sanity-check of the judge output, **not** a protocol change; the other **59/60 documents retain intact
method-blinding**.

**Also disclosed (lesser; no method leak).** The content type (obfuscated pharma/adult spam) of the
**3 fully-refused documents** (`d1_sb_0c55e64c8f49`, `d2_0b48feee9474`, `d2_22f7325368b4`) was
described. Their **methods were NOT revealed**, and they carry **no LLM-judge ratings** (`judge_missing`),
so they are already outside P3.

**Handling (fixed now, before annotation — no committed result is reselected):**
1. **This note** (the disclosure; identifies the doc uid).
2. **Part D sensitivity analysis** (`d10_analyze.py` → `sensitivity_exclude_compromised`): **P1–P3
   recomputed with AND without `d1_sb_017547c4cef2`**, both reported. No other change.
3. **Limitations lines (to appear in `notes/hitl_summary.md`):** *(i) method-blinding was procedurally
   broken for 1 of 60 documents (`d1_sb_017547c4cef2`) by the orchestrator's progress reporting; a
   sensitivity analysis excluding it is reported alongside the primary result. (ii) The 9 LLM-judge
   refusals are content-correlated (3 obfuscated-spam documents), so P3 coverage is slightly biased
   away from that content type.*

The single-annotator + procedural-only-blinding limitations were already disclosed in §C; this note
records the concrete break and its bounded, pre-annotation mitigation.

---

## 2026-07-26 · Directive 10 RESULT — HITL + LLM-judge plausibility validation

Human ratings committed **untouched** (`51c30b6`, sha256 `4568450…33d0e`) **before** any analysis ran;
completeness verified against `key.json` (2,558/2,558 rows, 260/260 sets, 0 missing/extra/duplicate,
0 token-string or doc_uid edits). Judge: `claude-opus-4-8`, **251 ok / 9 refusals** (260 sets).
Full report: `notes/hitl_summary.md`; figure `figures/hitl/hitl_cue_rates.(png|pdf)`.

> **⚠ Read with Amendment 4 (below).** The Part E review found that **every LIME attribution in the
> XAI phase is identically zero** (upstream inverted `split_expression`), so the arm annotated as
> "LIME" is actually the document's **first 10 word tokens** (`FIRST10†`). All LIME-specific
> conclusions are **withdrawn**; SHAP/IG/RANDOM, E1, P2 and P3 are unaffected.

**Headline verdict — the plausibility proxy is directionally validated but only "fair" in calibration.**
- **P2 (primary) lexicon–human token κ = 0.396 → band "fair — proxy retained as a heuristic with the
  limitation restated"** (84.5% agreement; per corpus d1_sb 0.434 / d2 0.421 *moderate*, d3_en 0.368
  *fair*). **E2 (κ ≥ 0.4) does NOT hold** — short by 0.004, i.e. **on the fair/moderate boundary**;
  reported as "fair", not rounded up.
- **E1 (human SHAP > RANDOM in every cell) HOLDS 4/4**, statistically separated in **3/4** (d2
  +0.060 [−0.005, 0.140] directional only). **Ranking concordance: 6/6 of the contrasts that do not
  involve the void arm agree in sign** (raw all-contrast count 12/15, and **all three disagreements
  involve the void `FIRST10†` arm**, each a near-zero human difference with a CI spanning 0). **No
  statistically separated human contrast contradicts the lexicon's sign.**
- **P3 human–judge κ = 0.702 (moderate, 93.1% agreement)**, set-level ρ 0.818 (cue-rate) / 0.734
  (holistic) — two independent raters on the same concept guide agree far better than either agrees
  with the term list, locating the gap in the lexicon's operationalisation, not the cue construct.
- **P4:** every method **matches or beats** RANDOM in every cell for both raters, with **one exact tie**
  (d1_sb·tfidf judge: FIRST10† = RANDOM = 0.158); SHAP beats RANDOM 4/4 for both raters. Absolute
  holistic levels are low (max 0.85 / 2.0).

**The proxy is CONSERVATIVE, not inflationary — this answers supervisor item 6 (plausibility
over-reading) directly.** In **9/9** method-vs-random contrasts the **human** separation exceeds the
lexicon's (e.g. d3_en·transformer IG−RANDOM human **+0.185** vs lexicon +0.060). The lexicon's random
baseline is inflated by context-free matches, compressing the gap; the committed plausibility column
therefore **understates** the human-visible separation. Correct usage: **ordinal, not absolute** — it
ranks explainers reliably and must not be quoted as "the fraction of tokens a human would call a cue".

**Exploratory question — VOID as posed (see Amendment 4).** The pre-registered question ("does the
d3_en LIME-above-random anomaly replicate in human ratings?") **cannot be answered**: the committed
d3_en "LIME" plausibility was computed from zero attributions, so the anomaly was never a property of
LIME. Withdrawn. What the data *does* show, under its correct name: on the AI corpus the **first 10
word tokens** are markedly cue-dense (human 0.225 vs RANDOM 0.060, +0.165 [0.090, 0.250], separated;
judge 0.200 vs 0.045) because those tokens are the LLM-generated **subject line** — a property of
AI-generated phishing text, not of an explainer. Notably a trivial "read the subject line" baseline is
**competitive with SHAP on d3_en** (0.225 vs 0.210), so d3_en plausibility discriminates weakly between
methods. (One observation, not two: the tfidf and transformer FIRST10† sets are identical in 20/20 docs.)

**Structural finding (not acted on).** The proxy's misses concentrate on **impersonation/authority**
and call-to-action phrasing (`payguard`, `fedex`, `digitalcoin`, `attached`, `action`, `required`):
the annotation guide's six concept families include impersonation and secrecy, but the committed
lexicon's six categories (urgency, credential, financial, imperative, greeting, url_email) contain
neither. A future revision should add them; **the committed 206-term lexicon is NOT changed here** —
it was pre-registered before results and any edit now would be outcome-tuned.

**Sensitivity (Amendment 3, pre-committed).** Excluding the one blinding-compromised document
(`d1_sb_017547c4cef2`): P2 κ 0.396 → **0.409** (band flips fair → moderate, E2 would hold); d2/d3_en κ
unchanged; P3 κ 0.702 → 0.701; E1 still 4/4. **The primary 60-document result governs** (κ = 0.396,
"fair", E2 fails); the sensitivity shows the *band assignment*, not the substance, is knife-edge
across the 0.4 boundary. Removing a document compromised *toward* the annotator is not the headline.

**Interpretation rule honoured:** every number here annotates a pre-committed metric; **no committed
XAI result was reselected, re-run or reversed** (`results/xai_metrics.csv` untouched, 30 rows), and no
champion, threshold or claim direction changed. Limitations disclosed in `notes/hitl_summary.md`:
single annotator + procedural blinding; **1/60 docs' method-blinding broken** (with sensitivity);
**9 refusals content-correlated** (3 obfuscated-spam docs ⇒ P3 coverage biased away from that content
type); judge = single stochastic draw at model default (Amendment 2) but fully reproducible from
committed artifacts; judge shares a model family with the lexicon's authoring tooling ⇒ **human is the
independent reference, P2 primary**; 60 docs / phishing-predicted / top-10 only.

**Links.** `notes/hitl_summary.md`; `figures/hitl/hitl_cue_rates.(png|pdf)`;
`results/hitl/{human_ratings,judge_ratings,hitl_token_level,hitl_set_level}.csv`,
`results/hitl/{hitl_metrics.json,judge_raw.jsonl,key.json,selected_ids.json,annotate.html}`;
`src/experiments/d10_{anchor,select,package,judge,analyze,figures}.py`.

---

## 2026-07-26 · Directive 10 — Amendment 4: **the LIME arm is void project-wide** (found in Part E review)

Found by the Part E scoped review (reviewer + independent confirmer, converging independently) and
then **verified directly**. Disclosed, not silently corrected.

**Defect.** `src/xai/explainers.py:152` passes `split_expression=r"\w+"` to `LimeTextExplainer`. In
LIME's API `split_expression` is the **separator** pattern (default `r"\W+"`) and is used as the
*non-word* test — `lime.lime_text.IndexedString` does `splitter = re.compile(r'(%s)|$' % split_expression)`
then `non_word = splitter.match`. Passing `\w+` therefore **inverts** the contract: every real word
matches and is discarded as a "non-word", leaving only whitespace/punctuation in the perturbation
vocabulary.

**Verified consequences (own checks, this repo):**
- **2,100 / 2,100** committed LIME attribution vectors are **identically zero** across **all 12 LIME
  units** (`results/xai/*__lime.json`, global max |attr| = **0.0**). SHAP 0/2,100 zero; IG 0/900 zero.
- Standalone reproduction: `IndexedString(text, split_expression=r"\w+").num_words()` = **2** (`' '`,
  `'.'`) vs **10** with `r"\W+"`; with `\w+` LIME returns only separator features (~1e-31), with
  `\W+` it correctly ranks `verify` first (0.756).
- `textops.top_k_positions` stable-argsorts a zero vector ⇒ the "top-k" is `[0…k−1]`. Hence **all
  80/80 D10 "LIME" sets are the document's first 10 word tokens** (SHAP 1/80 coincidentally, IG 0/20,
  RANDOM 0/80), and the d3_en tfidf-"LIME" and transformer-"LIME" sets are **identical in 20/20 docs**
  (SHAP: 0/20) because the void arm cannot depend on the model.
- The defect's signature was already visible in the committed, previously "verified PASS" log: **all 12
  LIME rows** of `results/xai_metrics.csv` carry `stability_spearman = nan` (Spearman of a constant
  vector), `lime_seed_agreement = 1.000`, and `stability_jaccard` 0.990–0.998 — the highest "stability"
  of any method, which is what a constant vector looks like.

**Provenance.** **Pre-existing, not introduced by D10**: `explainers.py` is byte-identical to pre-D10
`fd274b9` (behaviourally — only an explanatory comment has now been added). The `\w+` value was itself
pre-registered in the Directive 7 entry, which is likely why it was never re-examined. D10's own
protocol executed correctly against its pre-registered set definition; the **regeneration anchor (§A.1)
covered SHAP and IG only**, so it structurally could not catch a LIME-only failure.

**Effect on D10 (handled in `notes/hitl_summary.md`, no numbers changed):** 80/260 sets (30.8%) and
786/2,558 token judgments (30.7%) are **mislabelled** — they measure a first-10-tokens positional
baseline, now written **`FIRST10†`** throughout. **Withdrawn:** the exploratory d3_en "LIME anomaly
replicates" answer and the "LIME is plausible yet unfaithful" conclusion. **Unaffected:** E1, all
SHAP/IG/RANDOM cells, P2, P3, P4 (agreement statistics over the tokens actually shown are valid
regardless of which procedure selected them); ranking concordance is now reported as **6/6 valid
(LIME-free) contrasts**, with all three raw disagreements traced to the void arm.

**Effect on the committed XAI phase (NOT corrected here — explicit decision required).** The following
committed claims are artifacts of a constant attribution vector and must not be cited as properties of
LIME: **"LIME ≈ random"**, **"faithfulness–stability trade-off (LIME most stable, least faithful)"**,
and the **12 LIME rows** of `results/xai_metrics.csv` (plus the LIME series in `figures/xai/*` and the
LIME passages of `notes/xai_summary.md`). The **SHAP and IG** results, the RQ3 answer **"linear TF-IDF +
SHAP"**, and every non-XAI result (`results.csv`, `results_512.csv`, D9 statistics) are **unaffected** —
no LIME quantity enters them.

**Deliberately NOT done (and why).** `explainers.py` is **not** behaviourally fixed and nothing is
re-run: changing that line would silently invalidate the committed `xai_metrics.csv` that this exact
line produced, which the project's integrity rules forbid doing implicitly. A defect banner has been
added at the call site pointing here (comment only — verified as a comment-only diff, so every committed
number remains exactly reproducible). **Open decision for the author:** either (a) re-run the LIME arm
of Directive 7 with `split_expression=r"\W+"` as a new, pre-registered, isolated-namespace ablation
(and, if the corrected LIME sets are to be validated by humans, a fresh annotation round — the existing
FIRST10† ratings cannot be reused), or (b) report the corpus-level finding as-is and confine LIME to a
documented "explainer harness defect" limitation. Both are legitimate; (a) restores an RQ3 comparison,
(b) costs no compute. Not chosen unilaterally.

**Methodological lesson (for the dissertation).** A random baseline is not a sufficient control: the
void arm's metrics looked like a *publishable trade-off* rather than a bug. An explainer pipeline needs
a **positive control** — assert attributions are non-zero and non-degenerate, and that a planted
decisive token is recovered — and the regeneration/integrity anchor must cover **every** method, not a
subset. This is now the thesis's strongest concrete argument for treating XAI tooling as code under
test rather than trusted library output.

---

## 2026-07-26 · Directive 11 PRE-REGISTRATION — corrected-LIME isolated re-run (resolves Amendment 4)

**Committed FIRST and ALONE** (this entry + the known-answer suite + the one-argument fix), before any
corrected-LIME result exists. Resolves the D10 Amendment 4 open defect by option **(a)**: re-run the
LIME arm properly in an isolated namespace.

**Guard (Part A.0, PASS 2026-07-26):** `12d60e6` ancestor of HEAD; tree clean; `results.csv` 124,
`results_512.csv` 8, `xai_metrics.csv` 30 rows **including the 12 void LIME rows** (all carrying the
void signature `stability_spearman=nan`, `lime_seed_agreement=1.0`); erratum present in
`notes/xai_summary.md`.

### Scope and the fix
- **Scope: LIME only.** No SHAP/IG re-run, no metric redefinition, no re-selection of any unit or doc.
- **Fix:** `src/xai/explainers.py` `lime_attr` no longer passes `split_expression`, so LIME applies its
  documented default separator semantics (`r"\W+"`). This is the single behavioural change. The
  defective value is retained as the module constant `LEGACY_VOID_SPLIT_EXPRESSION` and reachable via
  `lime_attr(..., split_expression=...)` / `attribute(..., lime_split_expression=...)` **for audit
  reproduction only**.
- **Consequences recorded deliberately (not side effects):** (i) the 12 LIME rows of the canonical
  `results/xai_metrics.csv` are **no longer reproducible from current code** — they are the void arm's
  output, retained as the **FIRST10 position-baseline** record and superseded by the fix namespace;
  (ii) re-running `src/experiments/d10_package.py` would now build corrected-LIME sets, so the
  **committed** D10 `key.json` / `blind_sheets.csv` / `human_ratings.csv` remain the artifacts of
  record for D10's FIRST10† arm; (iii) `run_xai.run_unit_method` now also returns its `payload`
  (purely additive, no behaviour change) so the isolated runner can persist elsewhere while provably
  sharing the identical computation path.

### Units and parameters (committed D7 configuration verbatim, except the fixed argument)
- **12 units:** 9 in-corpus (`tfidf|ensemble|transformer` × `d1_sb|d2|d3_en`) + 3 transformer cross
  (`xfer_d3_en__d1_sb`, `xfer_d2__d3_en`, `xfer_d3_en__d2`).
- `num_samples=1000`, **seed 42**, n=**200** in-corpus / **100** cross, and the **SAME committed
  `explained_ids` sets** loaded from `results/xai/` — identical documents, so corrected vs void is a
  paired comparison.
- **Identical metric pipeline:** comp@k / suff@k (k=1,5,10,20), AOPC over k=1..20, stability (top-10
  Jaccard **and** the now-meaningful Spearman), LIME seed-agreement (seeds 42/43), plausibility@10,
  runtime — each with the **identical random-attribution baseline** through the same pipeline. Same CSV
  schema as the canonical log for direct comparison.
- **Isolation:** results written ONLY to `results/xai_lime_fix/<unit>__lime.json` and
  `results/xai_metrics_limefix.csv`. The canonical `results/xai/*`, `results/xai_metrics.csv`,
  `figures/xai/*`, `results.csv`, `results_512.csv` and all D9/D10 artifacts stay **byte-untouched**.
  Note the isolation is of *results*: the corrected explainer code is shared (that is the point of a
  fix), so the void arm is reproducible only via the legacy constant above.

### Known-answer harness (Part A.1) — `src/xai/test_known_answer.py`, **PASSED (41 checks)**
A permanent positive-control regression test; the control whose absence let the D7 defect survive. Two
deterministic synthetic documents (neutral lead → planted phishing paragraph → neutral tail) are run
through **all three explainers on a champion of every family** (`tfidf_d1_sb`, `ensemble_d1_sb`,
`transformer_d1_sb`; IG transformer-only):
- **T0 precondition** — the champion must actually *predict* phishing on each planted doc (p≥0.5);
  realised p = 0.999/0.989 (tfidf), 0.999/0.976 (ensemble), 0.998/0.661 (transformer).
- **T1 planted-cue localisation** — a planted token must appear in the top-**10** *distinct* words by
  class-1 attribution. Top-10 is the project's native granularity (plausibility@10, comp@10, suff@10);
  distinct words because LIME is bag-of-words. The planted set is derived **objectively** (cue-paragraph
  tokens minus neutral tokens), so it cannot be tuned per method.
- **T1b attribution-mass localisation** — planted-region mean signed attribution > non-planted mean.
  **Gating for SHAP and LIME; diagnostic-only for IG**, with the measured justification recorded in the
  code: IG's signed attributions are diffuse and net-negative, so on the lower-confidence doc
  (p=0.661) IG's planted mean (−0.031) sits just under the non-planted mean (−0.025) while it still
  satisfies T1. The `|attr|` "relevance-mass" variant was tested and **rejected** (it fails SHAP and
  LIME too, being diluted by the long planted paragraph). IG stays fully gated by T0/T1/T4.
- **T2 non-degeneracy on real data** — corrected LIME is non-constant on 5 real TEST docs per family
  (5/5 for all three).
- **T3 defect reproduction** — the legacy value still yields identically-zero LIME (2/2 per family),
  confirming the documented defect model.
- **T4 self-validation** — a **zero** attribution vector must FAIL T1 (2/2 per family), proving the
  suite detects the D7 defect; the neutral lead is placed first precisely so a first-k baseline lands
  on neutral tokens.
- **Honest disclosure:** the suite was iterated *during construction, before any corrected-LIME result
  existed*. Two earlier drafts were rejected on diagnosis, not convenience: (1) scoring raw positions
  (bag-of-words repetition filled the top-k) and hand-curated cue lists; (2) filler-dominated documents
  the champions classified as **ham** (p=0.0004), which made the class-1 question ill-posed and produced
  a *spurious* IG failure — this is why T0 exists. IG's word-level output was checked against its raw
  sub-token attributions and matches, so **IG has no code defect**; only its per-token sharpness is
  weaker than the perturbation methods'.

### Expectation, interpretation rule, out of scope
- **Descriptive expectation (no gate):** corrected LIME should show **non-trivial faithfulness** (a
  standalone probe ranked `verify` first at 0.756 where the defective call returned only separator
  features at ~1e-31). Nothing is gated on the outcome; the numbers stand as measured.
- **Interpretation rule:** corrected-LIME numbers are reported **beside** the void arm's numbers
  (relabelled **FIRST10**, a position baseline) and beside the committed SHAP/IG numbers. The committed
  **RQ3 answer is re-examined honestly** — LIME's cost is re-measured and **no conclusion is
  protected**. If corrected LIME changes the RQ2/RQ3 picture, that is reported as the finding.
- **Cost:** projected **~3.8 h** for all 12 units from the committed void runtimes (the void arm still
  executed 1000 perturbation queries per call, so those timings are a valid basis). Part B.3 confirms
  by probe from the most expensive unit; if the projection exceeds ~6 h, checkpoint the author with
  **scheduling options only** — the design is never trimmed.
- **Out of scope, stated:** human re-annotation of corrected-LIME sets (**deferred** — not required for
  RQ2; the D10 FIRST10† ratings cannot be reused for corrected LIME); any edit to
  `src/xai/cue_lexicon.py`; any change to the SHAP or IG code paths.

**Links.** `src/xai/test_known_answer.py`; `src/xai/explainers.py` (`lime_attr`,
`LEGACY_VOID_SPLIT_EXPRESSION`); `src/xai/run_lime_fix.py` (Part B, uncommitted at this point);
`results/xai_metrics_limefix.csv` + `results/xai_lime_fix/` (to be produced);
`notes/lime_fix_summary.md` (to be written).

---

## 2026-07-26 · Directive 11 RESULT — corrected LIME: **Amendment 4 defect RESOLVED**

12/12 units re-run in the isolated namespace in **2.75 h** (projection 3.8 h). Canonical
`results/xai_metrics.csv` (30 rows), `results/xai/*`, `results.csv` (124), `results_512.csv` (8),
`figures/xai/*` and every D9/D10 artifact are **byte-untouched** (verified). Full report:
`notes/lime_fix_summary.md`; log `results/xai_metrics_limefix.csv`; figures `figures/xai_lime_fix/*`.

**Headline: the void arm had inverted the LIME finding on BOTH axes.**
- **Faithfulness restored.** Corrected LIME AOPC **0.071–0.295** vs its random baseline 0.0001–0.073 —
  **above random on 12/12 units, by at least 4×** (per-unit ratios 4.0×–2357×; the top end is an
  artifact of `ensemble_d1_sb`'s 0.0001 denominator, so the floor is the meaningful figure). The void
  arm was **−0.0016 to 0.0753** ≈ random. "**LIME ≈ random**" is **withdrawn**, and so is "**SHAP is
  the only demonstrably faithful method**".
- **The stability claim is inverted.** Corrected LIME is the **least stable** method on **11/12 units**
  (per-unit Jaccard 0.355–0.700; seed-agreement 0.354–0.711; the exception is `xfer_d3_en__d2`, where
  IG 0.4907 < LIME 0.5019) — the committed "LIME most stable" (Jaccard 0.99, seed-agreement **1.000**,
  Spearman **nan** on all 12 rows) was purely the signature of a constant vector. A
  faithfulness–stability trade-off **does** exist, but with LIME at the **faithful-but-unstable** end,
  the exact opposite of the committed wording.
- **5 of 12 units flip** to corrected LIME as most faithful: `transformer_d1_sb` (0.1233 vs SHAP
  0.0931), `transformer_d2` (0.1753 vs 0.1372) and **all three cross-corpus transfer units**
  (0.2180/0.2818/0.2945 vs SHAP 0.1844/0.2125/0.2227). The six sklearn units do **not** flip — SHAP
  stays most faithful there. On `transformer_d3_en` IG still edges LIME (0.1288 vs 0.1265).
- **RQ2 revision (transformers).** Corrected LIME is the most faithful (0.1417 vs SHAP 0.1069, IG
  0.1024) **and** most plausible (0.1845 vs 0.1615, 0.1476) transformer explainer; IG keeps the best
  stability (0.7028 vs 0.4091) and is ~8× cheaper, so **IG remains the pragmatic transformer default**
  while LIME is the faithfulness-first choice. "Transformers are hardest to explain" still holds in
  absolute terms.
- **RQ3 answer HOLDS — on replaced grounds.** For linear TF-IDF (family means), SHAP beats corrected
  LIME on AOPC 0.1613 vs 0.1462, stability 0.9091 vs 0.6508, plausibility 0.2183 vs 0.1781 (though
  **not** on `tfidf_d1_sb`: LIME 0.2357 vs 0.2338), and on **attribution cost 0.6 s vs 39.6 s per 200
  documents (64.5×)**. **Cost basis corrected:** the `runtime_s` column is the whole metric pipeline
  and includes a **LIME-only** seed-agreement step, so it must not be quoted as explanation cost;
  attribution time (`timing.attr`) is the like-for-like measure and makes the tfidf gap *larger*
  (64.5×) and the transformer gaps *smaller* (LIME 181.1 s vs IG 27.4 s = 6.6×; vs SHAP 115.9 s =
  1.6×). The old justification ("LIME ≈ random") was false and is replaced by "SHAP edges LIME on
  faithfulness, is markedly more stable, is more plausible on average, and is ~64× cheaper to compute".
  LIME is **viable, not broken** — merely dominated for this family. The 0.0151 AOPC margin is a point
  estimate; **no significance test was run**.
- **Residual failure mode disclosed.** On **84/2,100 documents (4.0%)** corrected LIME returns
  numerically negligible attributions (max |attr| ≈ 1e-6, stored as 0 at 5-dp) — `transformer_d3_en`
  22.5%, `xfer_d3_en__d1_sb` 21.0%, `transformer_d1_sb` 9.0%, **zero on all six sklearn units**. These
  are near-saturated predictions where 1,000 perturbations do not move the output, so the surrogate has
  no signal; **SHAP and IG are unaffected on the same documents**. **80/84 (95%) are ham-predicted**, so
  **plausibility@10 (phishing-predicted docs only) is essentially unaffected**; **faithfulness and
  stability iterate over all explained docs and are affected** — biasing those units' AOPC down and
  stability up (conservative for the claims above) but leaving the closest ordering,
  `transformer_d3_en` IG 0.1288 vs LIME 0.1265, **unresolved**. Not ham-exclusive: a fresh sample of
  phishing-predicted transformer docs showed 2/5 below a 1e-5 floor. **The positive control was
  strengthened in response** (magnitude floor instead of exact zero, a <50% rate gate since the D7
  defect was 100% degenerate, phishing-predicted sampling, plus a non-gating `T2b` saturation
  diagnostic); re-run **41 checks PASS**.

**Positive control now permanent.** `src/xai/test_known_answer.py` (**41 checks, PASS** over
tfidf/ensemble/transformer champions): T0 predicted-phishing precondition, T1 planted-cue in the top-10
distinct words, T1b attribution-mass localisation (gating SHAP/LIME, diagnostic for IG), T2
non-degeneracy on real docs, T3 the legacy value still reproduces the void arm, **T4 a ZERO vector must
fail T1** — i.e. the suite provably detects the original defect. Every future explainer must pass it.

**Interpretation rule honoured:** corrected numbers are reported **beside** the void arm (relabelled
FIRST10) and the committed SHAP/IG numbers; the RQ3 answer was re-examined with LIME's cost re-measured
and **no conclusion protected** — one committed claim was withdrawn, one inverted, one revised, one
upheld on new grounds. Nothing was reselected, and no SHAP/IG/cue-lexicon code path was touched.

**Limitations.** No human validation of the corrected arm (D10's ratings were collected on FIRST10† and
cannot be reused; re-annotation deferred, out of scope) ⇒ corrected plausibility is the lexicon proxy
only, which D10 showed is "fair" (κ=0.396) and conservative ⇒ ordinal, not absolute. The canonical
`xai_metrics.csv` LIME rows are deliberately no longer reproducible from current code (reproducible via
`explainers.LEGACY_VOID_SPLIT_EXPRESSION`). Single seed (42); part of LIME's measured instability is its
own Monte-Carlo variance at the inherited `num_samples=1000`. **Significance testing was initially
absent**; Directive 11b (below) subsequently added paired bootstrap CIs for the two knife-edge
orderings — all other comparisons here remain point estimates. **The stored corrected-LIME JSONs are a lossy
record**: `attrs` are rounded to 5 dp, which re-orders LIME's tied bag-of-words weights, so
`plausibility@10` re-derived from the stored files matches the CSV on only 7/12 units (30/30 for the
canonical SHAP/IG/void rows) — **the CSV is authoritative** (a full-precision live re-run reproduces it
exactly), but D10's regeneration-anchor property does not hold for this arm from the stored files.
**Positive-control caveat:** reviewers showed by Monte Carlo that **T1 alone is weak** (the objectively
derived planted set spans 53–60% of each document's vocabulary, so a random attribution passes at
p≈0.9998); T1's power is against *degenerate* vectors (T4), and **T1b is the discriminating gate**
(random passes at p≈0.5) — from which IG is exempt. The suite is a degeneracy detector, not a proof of
attribution quality, and should be strengthened (magnitude floor instead of exact zero, tighter planted
set) before further reliance.

**Links.** `notes/lime_fix_summary.md`; `results/xai_metrics_limefix.csv`; `figures/xai_lime_fix/*`;
`src/xai/{test_known_answer,run_lime_fix,report_lime_fix,figures_lime_fix}.py`.

---

## 2026-07-26 · D10 one-line correction — the `greeting` false-positive share

`notes/hitl_summary.md` observation 3 said the lexicon's `greeting` category "generates 18% of its
false positives (`dear`+`hello`+`greetings` = 38/209)". That 18.2% was an **ad-hoc subset**: only the
three greeting tokens that happened to appear in the truncated top-15 false-positive list, silently
omitting `valued` 7, `customer` 7, `friend` 3, `user` 3, `welcome` 2, `client` 1. **Corrected to simple
membership: 61/209 = 29.2%** of proxy false positives are tokens whose lowercase form is in the
`greeting` set (definition now stated in the observation). Recomputed from the committed
`results/hitl/hitl_token_level.csv`; no re-run, no other D10 number affected (the qualitative point —
that the guide negates non-manipulative greetings while the lexicon counts them — is unchanged and
strengthened). Flagged by independent verification as the only number in the D10 package that failed
to reproduce.

---

## 2026-07-26 · Directive 11b — paired CIs for the two knife-edge AOPC orderings

Guard PASS (`9d63505` ancestor of HEAD; tree clean; `xai_metrics.csv` 30 rows, `xai_metrics_limefix.csv`
12 rows, `results.csv` 124, `results_512.csv` 8 all byte-identical to `9d63505`). Two pre-stated
comparisons only; **CIs annotate, nothing is reselected**. `src/experiments/d11b_ci.py` →
`results/xai_lime_fix/knife_edge_ci.json`.

**Method.** Directive 9's paired stratified bootstrap reused verbatim (`d9_stats._resamples`):
**B=10,000** within-class resamples over documents (positive/negative counts preserved), **seed 42**,
the **same resample indices applied to both arms** so the difference is paired; CI = 2.5/97.5
percentiles of the bootstrap mean-delta. The statistic is the **per-document AOPC**, recomputed here
because only the aggregate was stored: per document, the deletion curve `p_pred(orig) − p_pred(top-k
removed)` averaged over k=1..20. SHAP/IG attributions were read from the committed
`results/xai/*.json`; **corrected LIME was regenerated live at full precision** because the stored fix
JSONs are 5-dp lossy (D11 limitation 8). **Validation: all 8 recomputed aggregates reproduce the
committed AOPC to 4 dp** (tfidf d1_sb 0.1989/0.1865, d2 0.1913/0.1812, d3_en 0.0936/0.0708;
transformer_d3_en IG 0.1288 / LIME 0.1265), confirming both the per-document decomposition and the
determinism of the live LIME regeneration.

**K1 — tfidf family, SHAP − corrected LIME (the RQ3 margin): SEPARATED, 4/4.**

| universe | Δ AOPC | 95% CI | excludes 0 |
|---|---|---|---|
| d1_sb | +0.0124 | [+0.006597, +0.018744] | **yes** |
| d2 | +0.0101 | [+0.004997, +0.015628] | **yes** |
| d3_en | +0.0228 | [+0.020257, +0.025366] | **yes** |
| **pooled (n=600)** | **+0.0151** | **[+0.012288, +0.017951]** | **yes** |

So SHAP's faithfulness edge over corrected LIME on the linear TF-IDF family is **statistically
separated**, per corpus and pooled — the 0.0151 point estimate is not noise, and the **RQ3 answer's
faithfulness leg is now inferentially supported** (it already held on stability and the 64.5×
attribution-cost gap).

**K2 — `transformer_d3_en`, IG − corrected LIME: NOT SEPARATED.** Δ = **+0.0023**, 95% CI
**[−0.021086, +0.024304]** — **includes 0**. This confirms the D11 disclosure: IG's nominal lead on the
one non-flipping transformer unit is **formally unresolved** and must not be reported as a difference.
It also means the count "5/12 units flip to corrected LIME" is best stated as *5 flips plus one unit
where the ordering cannot be resolved*.

**Supersedes** the D11 limitation "no significance testing" for these two comparisons only; every other
D11 comparison remains a point estimate. `notes/lime_fix_summary.md` §(c)/§(e) carry the two-line
addenda.

---

## 2026-08-17 · Ch6 presentation-only figure regeneration (DATA UNCHANGED)

Three committed figures were regenerated for the dissertation's Chapter 6 with **in-image text only**
changed: internal directive/amendment references, file paths and em dashes removed. **No metric
artifact was read for anything other than plotting, none was modified, and no model was re-run.**

**Files replaced (png + pdf each):** `figures/xai_lime_fix/xai_lime_fix_deletion_curves`,
`figures/xai_lime_fix/xai_lime_fix_bars`, `figures/hitl/hitl_cue_rates`.

**Text changes.** (1) Deletion curves: suptitle now *"Faithfulness deletion curves with the corrected
LIME arm (higher = more faithful)"*; the method-comparison line is retained verbatim; FIRST10 is
described as *"the void arm relabelled as a position baseline (the document's first 10 word tokens, not
LIME)"* with no path and no amendment pointer. (2) Bars: suptitle now *"Corrected LIME vs SHAP and
Integrated Gradients vs the FIRST10 position baseline (in-corpus family means; black ticks = random
baseline)"* plus the single FIRST10 note. (3) HITL: suptitle now *"Human vs cue-lexicon vs LLM-judge
plausibility (cue-rate@10) on the committed TEST explanation sets"*; all counts unchanged (60 docs /
260 sets / 2,558 judgments / judge ok 251/260, 9 refusals); the void-arm disclosure is kept without a
path.

**One label deliberately not read from the artifact.** The human-judge kappa band is printed as
**"0.702 (substantial)"** — the full Landis-Koch label (0.61-0.80). `hitl_metrics.json` stores
`"moderate"` because the **pre-registered** three-band scheme collapses everything ≥0.4 to "moderate";
that pre-registered value is untouched in the artifact and in `notes/hitl_summary.md`, and the P3 table
there already recorded both readings. The figure and the thesis text now use the full scale
consistently. `p3['band']` is therefore intentionally not interpolated in `d10_figures.py` (comment at
the call site records why). The lexicon-human band ("fair") is still read from the artifact.

**Verification.** The two figure scripts' diffs touch only suptitle strings and comments (no data,
axis, legend or colour code). An md5 over `xai_metrics.csv`, `xai_metrics_limefix.csv`,
`hitl_metrics.json`, `hitl_set_level.csv`, `results/xai_lime_fix/*.json` and `results/xai/*__shap.json`
is **identical before and after** regeneration (`3a5fce09deb65c04655f0331ddbacb2c`). All plotted values
are therefore provably unchanged. Canonical `figures/xai/*` were **not** regenerated; the erratum at the
top of `notes/xai_summary.md` continues to govern them.

---

## 2026-08-17 · Confusion matrices for the six Table 6.1 finalists (regenerated, GATE PASSED 6/6)

POLISH-1 established that **no committed artifact stores TN/FP/FN/TP**. The cells are therefore
regenerated here under the thesis's **reproducibility-by-regeneration policy** (Ch5 artefact policy:
detail outputs are regenerable from deterministic scripts plus committed anchors) for the appendix.
**No training and no tuning.** Script `src/experiments/confusion_matrices.py`; artifact
`results/confusion_matrices.json`.

**Method.** Each finalist is scored **exactly once on its OWN frozen TEST partition** at the committed
**0.5** operating point. Sklearn finalists go through the committed champions inference module
(`src.xai.champions._SklearnChampion`), which rebuilds the model from its committed HPO `best_params`
and re-fits it on the frozen TRAIN split — the identical inference path that produced the committed XAI
results — with **LinearSVC mapped through the committed `expit(decision_function)`** margin mapping
(monotone, so the 0.5 cut equals a zero-margin cut). Transformer finalists **load their saved
fine-tuned checkpoints** (no fitting) at the committed `MAX_LEN=256`. Disclosure: the sklearn step is a
deterministic **re-fit from committed hyperparameters**, not re-training or re-tuning; no pickled fitted
estimator exists in the repo, and this is the same path all downstream XAI work uses.

**Matrices (TN / FP / FN / TP; row sums equal the frozen test n).**

| finalist | corpus | TN | FP | FN | TP | n_test |
|---|---|--:|--:|--:|--:|--:|
| LinearSVC | d1_sb | 3,548 | 19 | 24 | 4,185 | 7,776 |
| MultinomialNB | d2 | 2,294 | 17 | 23 | 2,050 | 4,384 |
| LinearSVC | d3_en | 948 | 27 | 8 | 1,892 | 2,875 |
| DistilBERT-256 | d1_sb | 3,552 | 15 | 18 | 4,191 | 7,776 |
| DistilBERT-256 | d2 | 2,308 | 3 | 20 | 2,053 | 4,384 |
| RoBERTa-256 | d3_en | 936 | 39 | 18 | 1,882 | 2,875 |

**VERIFICATION GATE: PASSED 6/6 (24/24 metric comparisons exact).** Accuracy, precision, recall and F1
recomputed from every matrix equal the committed values at their stored precision (4 dp), against
`results/results.csv` `__tuned` rows for the classical finalists and `results/transformers/<model>__<corpus>.json`
`test` blocks for the transformers. All gate deltas are 0.0000; nothing was reconciled. Notably the
transformer matrices reproduce the committed metrics exactly despite fp16 autocast inference, so no
borderline sample flipped at the 0.5 cut.

**Isolation.** `results.csv` (124), `results_512.csv` (8), `xai_metrics.csv` (30),
`xai_metrics_limefix.csv` (12) and `results/transformers/*.json` are **byte-untouched**; only the new
artifact and script were added. `CLAUDE.md` deliberately unchanged (no phase change).

---

## 2026-08-17 · AUDIT-D3 — external-review response (POST-HOC DIAGNOSTICS; committed results unchanged)

Read-only audit answering three external-review concerns. **No retraining, no tuning, no reselection**;
every committed number stands as published. All figures below are **post-hoc diagnostics**. Analysis
script was run from the scratchpad and is not committed; every input is a committed artifact or the
git-ignored raw corpus.

### A. d3_en seed-family leakage — CONFIRMED, and it is total

**A1. The split is keyed PER DOCUMENT, not grouped by seed family.** `src/splits.py:108-113` verbatim:
`train_idx, temp_idx = train_test_split(idx, test_size=0.30, random_state=seed, stratify=y)` then
`val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=seed, stratify=y[temp_idx])`,
with the key documented at `src/splits.py:8` as *"keyed by the **sha1 hash of the cleaned body text**"*.
Stratification is on the binary label only. There is **no `groups=` argument, no `GroupShuffleSplit`, and
no `Original_ID` reference anywhere in `src/splits.py`** — the sole `Original_ID` mention in `src/` is
`data_loader.py:339` (`"original_id": raw["Original_ID"].astype(int)`). PhishFuzzer emits **6 variants per
seed template**, so variant siblings are free to land in different splits.

**A2. Join is exact and needs no reconstruction.** `data/cache/clean_d3.pkl` already carries
`original_id` (int64, 0 nulls): **19,164/19,164 d3_en documents mapped, 100% coverage**.

**A3. Family overlap (2,875 test documents, 3,194 families).** Families per split: train 3,191 / val
1,963 / test 2,004.
- TEST documents sharing a family with ≥1 **TRAIN** document: **2,865 / 2,875 = 99.65%**
- TEST documents sharing a family with ≥1 **VAL** document: 1,567 / 2,875 = 54.50%
- TEST documents sharing a family with train **or** val: **2,875 / 2,875 = 100.00%**
- VAL documents sharing a family with ≥1 TRAIN document: 2,867 / 2,875 = 99.72%

**A4. NOT COMPUTABLE — the clean-test subset is EMPTY (n = 0).** Every d3_en test document has a variant
sibling in train or val, so no leakage-free holdout exists inside the frozen split and the
evaluation-only sensitivity cannot be run. The laxer "no TRAIN sibling" variant yields **n = 10, all of
class 0** — ROC-AUC undefined. Consequence: **the in-corpus d3_en scores (classical 0.9993, RoBERTa
0.9976) cannot be leakage-corrected by evaluation alone**; quantifying the effect requires a
family-grouped (`GroupShuffleSplit` on `original_id`) re-split **and retraining**, which is out of scope
here and should be a separate pre-registered directive.

### B. D1↔D3 provenance overlap

**B5. Seed-level provenance EXISTS (not ABSENT), but only in the raw seeds file.** The cleaned variant
cache is uniform — `clean_d3.pkl.source` is `'Synthetic Entity-Seed Expansion 2026'` for all 19,164 rows
and `created_by` is `'LLM'` for all — so provenance must be joined from
`data/raw/phishfuzzer/PhishFuzzer_emails_original_seed_v1.json` field `Source` via `Original_ID`.
Seed buckets (n=3,300): Nazario 1,015 · WarrantedSpamDataset 842 · CEAS_08 839 · private "Manual" 300 ·
SpamAssassin 183 · Spam Archive 2024/2025 121. **Nazario, CEAS_08 and SpamAssassin are three of D1's six
sub-corpora**, so **62.4% of d3_en (11,964 documents, 1,994 families) descends from seeds drawn from D1's
own constituent corpora** — Nazario 30.5%, CEAS_08 26.2%, SpamAssassin 5.7%. Of the 2,875 test documents,
**1,775 (61.7%) are D1-provenance**.

**B6a. D1→D3 (DistilBERT-256 trained on d1_sb, evaluated on d3_en TEST). Committed 0.9243.**

| evaluation slice | n | pos share | ROC-AUC |
|---|--:|--:|--:|
| full test (reproduction anchor) | 2,875 | 0.661 | **0.9243** (= committed) |
| D1-provenance families only | 1,775 | 0.496 | **0.9948** |
| **excluding D1-provenance families** | 1,100 | 0.927 | **0.3970** |

**CONFOUND, stated up front:** provenance is strongly correlated with class — the non-D1 slice is 92.7%
positive with only **80 negatives**, the D1 slice is balanced (49.6%). AUC is rank-based so balance does
not bias it in expectation, but the estimate on 80 negatives is noisy and the two slices are different
populations (Manual/Warranted/Archive spam vs Nazario/CEAS_08/SpamAssassin). The diagnostic therefore
**does not bound the leakage**; what it does show is that the committed 0.9243 is **highly non-homogeneous
across seed provenance** — near-perfect on D3 documents descended from D1's own sub-corpora, at or below
chance on documents of independent provenance. Resolving it needs a provenance-stratified re-split with
retraining.

**B6b. D3→D1 (RoBERTa-256 trained on d3_en, evaluated on d1_sb TEST). Committed 0.9659. NULL RESULT.**
The literal instruction (exclude D3 documents on the evaluation side) is vacuous here — D3 is the
*training* side, and evaluation-only exclusion cannot address training contamination. The informative
symmetric variant is to drop the D1 test documents belonging to the sub-corpora D3's seeds were drawn
from (`Nazario`, `CEAS_08`, `SpamAssasin`), leaving `Nigerian_Fraud` + `Ling`:

| evaluation slice | n | ROC-AUC |
|---|--:|--:|
| full test (reproduction anchor) | 7,776 | **0.9659** (= committed) |
| excluding D3-seed sub-corpora | 939 | **0.9898** |

AUC **rises** on the provenance-clean slice, so the committed D3→D1 transfer result is **not** explained
by seed-provenance overlap. Both full-test anchors reproduce the committed values exactly, validating the
evaluation path.

### C. Table 6.5 comparator provenance — the "4/6" claim is pool-dependent

**C7. The committed rule, verbatim.** `notes/transformer_summary.md` §(a): *"Best transformer per
**source** (by val AUC) applied to the other corpora's TEST splits. Compared against the best classical
champion (tfidf/ensemble/dense) for the same directed pair."* And `notes/decisions.md` (D9 champion
definitions): *"cross champions are the committed rows in results(_512).csv"*.

**Does the rule predate observation of target test scores? Partly.** Each classical family's *model* was
selected on **source validation** AUC before the target test was seen — `results.csv` notes confirm it,
e.g. `'cross-corpus tfidf champion (val_auc=0.9997)'`. But the choice of **which family** to quote as
"best classical" per direction is a **post-hoc maximum over already-observed target test AUCs**, and no
committed entry pre-specifies it. The comparator pool is also implicitly restricted to the
`__xcorpus_*` rows (2026-07-07), excluding the earlier baseline-phase cross rows (2026-07-06,
`notes='cross-corpus'`, default hyperparameters). That exclusion is material:

| direction | T AUC | classical max, **xcorpus pool** | Δ (committed) | classical max, **ALL committed rows** | Δ |
|---|--:|--:|--:|--:|--:|
| d1_sb→d2 | 0.9434 | 0.9091 LR | +0.0343 | 0.9152 LR | +0.0282 |
| d1_sb→d3_en | 0.9243 | 0.8965 LR | +0.0278 | **0.9263 LR** | **−0.0020** |
| d2→d1_sb | 0.9199 | 0.9080 LR | +0.0119 | **0.9469 LR** | **−0.0270** |
| d2→d3_en | 0.7748 | 0.8429 LR | −0.0681 | 0.8559 LR | −0.0811 |
| d3_en→d1_sb | 0.9659 | 0.8888 LightGBM | +0.0771 | 0.9225 LR | +0.0434 |
| d3_en→d2 | 0.7306 | 0.7523 LinearSVC | −0.0217 | 0.7967 LR | −0.0661 |

Under the committed (xcorpus-pool) reading transformers win **4/6**; under a literal
"strongest committed classical transfer result" reading they win **2/6** — `d1_sb→d3_en` and `d2→d1_sb`
flip. The restriction is defensible (compare val-selected champions, not a max over every run), but the
**phrase "strongest committed classical" is literally inaccurate for 3 of 6 directions** and an external
reviewer recomputing from `results.csv` will land on 2/6. Chapter 6 should state the comparator pool
explicitly. Note also the direction of the bias: taking a post-hoc max over three classical families
makes transformer wins *harder* (conservative for the 4/6 claim) but makes the two classical wins
(`d2→d3_en`, `d3_en→d2`) *easier* — so the "transformers do not close the AI→real gap" statement is the
anti-conservative one.

**Nothing in `results.csv`, `results_512.csv`, `xai_metrics.csv`, `xai_metrics_limefix.csv`,
`confusion_matrices.json` or any figure was modified by this audit. `CLAUDE.md` untouched.**

---

## 2026-08-17 · Amendment 5 — Group-aware sensitivity analysis for d3_en (pre-registered)

**Motivation.** AUDIT-D3 (commit `66bcdd2`) confirmed the frozen d3_en split is **per-document**
(`src/splits.py` L108-113: stratified `train_test_split`, no `groups=`), so PhishFuzzer variants of the
same seed email appear on both sides of the train/test boundary — **100% of test seed families have a
train or val sibling** (99.65% have a train sibling). The in-corpus d3 absolutes are therefore
optimistic. This amendment pre-registers a group-aware sensitivity analysis to quantify that.

**Pre-registered design (fixed BEFORE any result is seen).**
- **Group key:** `Original_ID` from `data/cache/clean_d3.pkl` (100% join verified in AUDIT-D3:
  19,164/19,164, 0 nulls, 3,194 distinct families).
- **Split:** two-stage `GroupShuffleSplit`, seed 42 — stage 1
  `GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=42)` over all d3_en groups; stage 2
  `GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=42)` over the held-out groups → val/test.
  **No stratification** (group splits cannot stratify); achieved per-part sizes and positive shares will
  be reported (corpus positive share 0.6605; deviations > 2 pp warned, **report-only, not a gate**).
  Persisted to `data/splits/d3_en_grouped.json` in the same hash-keyed format with an extra field
  `"variant": "grouped_by_original_id"`. **`data/splits/d3_en.json` is NOT overwritten.**
- **Classical (FROZEN, no re-tuning, no re-search):** LinearSVC + TF-IDF refit on the grouped train with
  the committed d3_en hyperparameters read from `results/hpo/d3_en_svc.json` —
  **`best_params = {"C": 9.587479255847708}`** — and the committed vectorizer configuration for the
  `svc` family, `hpo.DEFAULT_LINEAR_VEC = {"ngram": "1_2", "min_df": 2, "sublinear": True,
  "max_features": 100000}` (`hpo._vec_from_best` returns `dict(DEFAULT_LINEAR_VEC)` for `svc`).
  Decision scores via the committed `expit(decision_function)` margin mapping.
- **Transformer (FROZEN, ONE run):** `roberta-base` fine-tuned **once** on the grouped train with the
  committed d3_en run's hyperparameters from `results/transformers/roberta-base__d3_en.json` and
  `src/experiments/transformers_ft.py` — **`selected_lr = 5e-05`** (the committed val-selected value; the
  3-lr grid is **not** re-run), `MAX_EPOCHS = 4`, `PATIENCE = 1`, `BATCH = 32` (effective, grad
  accumulation on OOM), `MAX_LEN = 256`, `WEIGHT_DECAY = 0.01`, `WARMUP_RATIO = 0.1`, AdamW + linear
  warmup schedule, fp16, `SEED = 42`, `set_determinism`, same pinned environment
  (`torch 2.11.0+cu128`, `transformers 4.57.6`, CUDA build 12.8, RTX 3090). The committed run used the
  **validation split for per-epoch ROC-AUC early stopping and best-epoch checkpoint selection**
  (`selection: val_roc_auc`), so the **grouped val is used for exactly that and nothing else** — no
  hyperparameter is chosen on it here. Training reuses the committed `transformers_ft.train_lr` loop
  verbatim.
- **Reporting scope.** STEP 2 in-corpus sensitivity (LinearSVC, RoBERTa, C1c direction); STEP 3
  evaluation-only cross-corpus onto the grouped test plus the provenance decomposition (subsets with
  < 100 negatives flagged unstable); STEP 4 grouped-trained LinearSVC outward to the committed D1/D2
  tests (those splits unchanged). All numbers to `results/group_sens.json` at 4 dp beside their
  committed counterparts.
- **Status of the committed results.** The committed headline values remain the **pre-registered results
  of record**; this analysis is reported as a **sensitivity, not a replacement**. Nothing is reselected.

**Status: pre-registered before execution; results to be appended on completion.**

---

## 2026-08-17 · Amendment 5 RESULTS — executed exactly as pre-registered (`2bad9cc`)

Appended as a new dated block (decisions.md is append-only; the pre-registration entry above is left
untouched). Total runtime **4.3 min**. Artifacts: `data/splits/d3_en_grouped.json`,
`results/group_sens.json`, script `src/experiments/group_sens.py`. **`data/splits/d3_en.json` and every
committed result are unmodified; the committed headlines remain the results of record.**

### STEP A0 — scope confirmations
**(1) The H1 512 ablation trio is evaluated on the D2 TEST split, not on d3_en test.** The committed row
is `train_corpus=d3_en, test_corpus=d2, RoBERTa-base, auc=0.844, n_test=4384` in `results_512.csv`
(n_test = the D2 test size 4,384), against the 256 value 0.7306 and the classical comparator 0.7523 —
all three on D2 test. Training data is the frozen per-document d3_en **train** split (n_train=13,414) at
`max_length=512`. **Therefore d3_en's family leakage cannot inflate the H1 result**: the evaluation
corpus is D2, which shares no documents and no seed families with D3 (D3's seed buckets contain no Enron
material). H1 stands unaffected.

**(2) FLAG — the exposure is broader than the expected trio.** Every committed row that trains on d3_en
and evaluates on d3_en test sits on the leaked boundary, not just SVC 0.9993 / RoBERTa 0.9976 / C1c:
the **whole in-corpus d3 leaderboard** (LR 0.9982, NB 0.9870, LinearSVC 0.9993, DecisionTree 0.9232,
RandomForest 0.9901, XGBoost 0.9957, LightGBM 0.9980, plus the untuned variants), the **dense-embedding
rows** (LR-w2v 0.9853, MLP-w2v 0.9942, LR-glove 0.9760, MLP-glove 0.9961), **all three 256 transformers**
(DistilBERT 0.9968, BERT 0.9979, RoBERTa 0.9976) **and all three 512 in-corpus transformers in
`results_512.csv`** (DistilBERT 0.9975, BERT 0.9977, RoBERTa 0.9984). Also downstream: the D9 **d3_en
imbalance stress numbers**, and the **XAI/HITL d3_en in-corpus units** (`tfidf_d3_en`, `ensemble_d3_en`,
`transformer_d3_en`), whose explanation sets are drawn from the leaked test split. Cross-corpus rows
(other-trained on d3 test, or d3-trained on other tests) are **not** affected by family leakage.

### STEP 1 — grouped split asserts
- **(a) no `Original_ID` in more than one part: PASS** (shared families: 0)
- **(b) pairwise disjoint, covering all 19,164 rows exactly once: PASS** (n=19,164, unique=19,164)
- **(c) achieved sizes / positive shares** (corpus 0.6605; families 2,235 train / 480 val / 479 test):

| part | n | share of corpus | pos share | deviation |
|---|--:|--:|--:|--:|
| train | 13,410 | 69.97% | 0.6622 | +0.17 pp |
| val | 2,874 | 15.00% | 0.6117 | **−4.88 pp (WARN, report-only)** |
| test | 2,880 | 15.03% | 0.7021 | **+4.16 pp (WARN, report-only)** |

The two warnings are the expected cost of grouping: `GroupShuffleSplit` cannot stratify, and class is
strongly correlated with seed family. Pre-registered as report-only, not a gate.

### STEP 2 — in-corpus sensitivity (the wounded trio)

| metric | committed (per-document split) | grouped | Δ |
|---|--:|--:|--:|
| LinearSVC + TF-IDF, d3_en test ROC-AUC | 0.9993 | **0.9928** | **−0.0065** |
| RoBERTa-256, d3_en test ROC-AUC | 0.9976 | **0.9877** | **−0.0099** |

**C1c direction under the grouped split: the classical model is still higher** (LinearSVC 0.9928 vs
RoBERTa 0.9877) — the same direction as the committed comparison (0.9993 vs 0.9976). **The C1c
conclusion survives group-aware splitting.**

**Interpretation.** Family leakage inflates the d3_en in-corpus absolutes by roughly **0.7–1.0 pp**, not
by the order of magnitude the 100%-overlap statistic might suggest. Both families remain above 0.98 on a
fully family-disjoint test set, so d3_en in-corpus separability is real; the committed absolutes are
mildly optimistic and should be quoted with this sensitivity. Whether the ~1 pp gap is statistically
separated was **not** tested (no CIs pre-registered here).

### STEP 3 — cross-corpus onto the grouped d3 test (evaluation only)

| direction | committed | onto grouped d3 test | Δ |
|---|--:|--:|--:|
| D1→D3 transformer (DistilBERT) | 0.9243 | 0.9160 | −0.0083 |
| D1→D3 classical (LR tfidf) | 0.8965 | 0.8822 | −0.0143 |
| D2→D3 transformer (DistilBERT) | 0.7748 | 0.8045 | +0.0297 |
| D2→D3 classical (LR tfidf) | 0.8429 | 0.8631 | +0.0202 |

Cross-corpus numbers move by ≤ 0.03 in both directions, i.e. the grouped test is not systematically
harder for externally-trained models — as expected, since a model that never saw any D3 document cannot
benefit from D3-internal family overlap. The residual movement reflects the different test composition
(2,880 vs 2,875 documents, positive share 0.7021 vs 0.6605).

**(c) Provenance decomposition on the grouped test** (D1→D3 DistilBERT scores):

| subset | n | pos | neg | ROC-AUC |
|---|--:|--:|--:|--:|
| D1-descended families | 1,680 | 924 | 756 | **0.9926** |
| independent provenance | 1,200 | 1,098 | 102 | **0.4218** |

The AUDIT-D3 finding **replicates on the family-disjoint split**: a D1-trained model is near-perfect on
D3 documents descended from D1's own sub-corpora and near/below chance on independent-provenance ones.
Neither subset trips the pre-registered `< 100 negatives` instability flag, but the independent subset is
**marginal at 102 negatives** and remains class-skewed (91.5% positive), so it still bounds nothing
precisely. **Conclusion: seed-provenance overlap, not intra-D3 family leakage, is the larger threat to
the D1↔D3 transfer numbers** — family grouping does not address it.

### STEP 4 — outward transfer from the grouped-trained LinearSVC (committed D1/D2 tests unchanged)

| direction | grouped-trained LinearSVC | committed LinearSVC (like-for-like) | Δ | committed transformer |
|---|--:|--:|--:|--:|
| D3→D1 (n=7,776) | **0.9007** | 0.8783 | **+0.0224** | 0.9659 |
| D3→D2 (n=4,384) | **0.7467** | 0.7523 | **−0.0056** | 0.7306 |

Training on a family-disjoint D3 split does **not** degrade outward transfer — it is slightly better to
D1 and essentially unchanged to D2. So the committed cross-corpus story (including the AI→real
difficulty on D3→D2) is **robust to the leakage**: the per-document split did not manufacture the
transfer results.

### Deviations from the directive
1. **New file added** (`src/experiments/group_sens.py`) alongside the two artifacts, for reproducibility;
   "no other files changed" is otherwise honoured — no committed result, figure, split or thesis text was
   modified.
2. **STEP 3 "committed D1/D2 champion" was ambiguous** (each direction has a transformer champion and a
   classical champion). Both are reported rather than one being chosen silently.
3. **STEP 4 comparator**: the like-for-like committed value is the classical LinearSVC cross row
   (0.8783 / 0.7523); the committed transformer values are shown alongside for context, since 0.9659 is a
   RoBERTa number and not comparable to a LinearSVC.
4. **Results appended as a new dated block** rather than edited into the Amendment 5 entry, to honour the
   append-only rule.
5. No CIs were computed (none pre-registered), so all Δ values here are point estimates.

## 2026-10-12 · AUDIT-PAPER — DeSE paper number check: corrections and undisclosed details (NO RESULTS CHANGED)

**Scope.** A read-only audit of the DeSE 2026 paper draft (`paper/main.tex`, untracked snapshot)
against the committed artefacts at `be7ed67`: 264 sub-claims, of which 257 match, 6 mismatch and 1 is
not found.
- No committed number, table, interval or p-value changes.
- Recomputed from committed files: AUCs (from score vectors), McNemar/Holm, κ, Spearman, family means,
  ratios and counts.
- Taken as committed: the bootstrap intervals (numpy RNG) and the notes-only items.
- Not re-run: the positive-control suite, since this machine had no Python environment.

The items below correct or qualify earlier entries, which stay untouched (append-only).

1. **Explanation sets exceed the stated error quota (D7 design, ~line 351).**
   - **Mechanism.** `src/xai/champions.py:170-174` caps only the *sampled* errors. The fill at
     `:181` draws from same-class documents not in that capped subset, so errors that were not sampled
     can re-enter. The `# pred==y for these` comment holds only when the cap does not bind.
   - **Realised errors** (y_true ≠ pred in `results/xai/explained_ids_<unit>.json`):
     - ensemble_d1_sb 33, ensemble_d2 32, ensemble_d3_en 31 and transformer_d3_en 31, against a quota of 30;
     - xfer_d3_en__d1_sb 30, xfer_d2__d3_en 38 and xfer_d3_en__d2 43 of 100, against a quota of 15;
     - the other five sets have 30 / 30 / 30 / 30 / 23. In total 73 of 2,100 documents are errors
       beyond the sampled ones.
   - **Stored field.** `n_misclassified` records only the sampled subset.
   - **Impact.** Every metric was computed on these sets, so nothing changes numerically. Describe the
     sets by their realised error counts. The transfer sets are 30–43% errors, not ≤15%.
2. **Truncation attribution.**
   - **Where the error is.** Line 529 and the generated `notes/transformer_512_summary.md` items 2–3.
     That text is hard-coded in `src/experiments/transformer_512_report.py:209,215`.
   - **Correction.** "76% → 22%" is the DistilBERT/BERT WordPiece train truncation on d3_en
     (75.95% → 22.11%). The d3_en champion at both windows is RoBERTa-base, whose figure is
     **73.46% → 18.73%** (`results/transformers{,_512}/roberta-base__d3_en.json`,
     `truncation.train.truncated_pct`). Table (c) of the same summary lists both rows. The H1 verdict
     is unaffected.
3. **Seed-family sibling wording.**
   - **Where.** The Amendment 5 motivation (~line 1561) and the `src/experiments/group_sens.py:5`
     docstring restate AUDIT-D3 A3 at family level.
   - **What A3 measured.** It counted *documents*: 100% of test documents have a train-or-val
     sibling, and 99.65% (2,865/2,875) have a train sibling.
   - **Consequence.** 10 test documents (class 0) have none in train. Train covers 3,191 of the 3,194
     families, so 2–3 test families have no training member. "Every test family has a training
     sibling" is therefore not supported.
4. **Judge calls (D10).** There were 278 API calls for 260 sets. 251 sets needed one call each; the
   9 refused sets were tried 3 times each under the pre-registered `MAX_ATTEMPTS = 3`
   (`src/experiments/d10_judge.py:49`).
5. **Undisclosed deviations from the D7 pre-registration (method hyperparameters, ~lines 374 and 380).**
   - **SHAP LinearExplainer background:** 200 seeded train rows (`src/xai/explainers.py:82-88`), not 1,000.
   - **IG target:** `target=1`, applied to the class-1 *logit* (`explainers.py:210-219`), with the sign
     flipped for ham-predicted documents when ranking (`src/xai/textops.py:54-66`). The pre-registration
     said "target = predicted-class logit".
   - Neither deviation was recorded before. Both are reported here without re-running anything.
6. **Typos and clarifications.**
   - **Line 1237, "IG ~8× cheaper".** This is the `runtime_s` basis. On the `timing.attr` basis used
     everywhere else (line 1246, `notes/lime_fix_summary.md`, the dissertation), IG is 6.6× cheaper
     than corrected LIME.
   - **Line 1634, "480 val / 479 test" families.** `results/group_sens.json` and
     `data/splits/d3_en_grouped.json` give val 479 / test 480.
   - **`fp_per_1000_legit@95recall` (D9).** It is read from the precision–recall curve: the threshold
     with the highest precision among those with recall ≥ 0.95 (`src/experiments/d9_imbalance.py:66-75`).
     It is not the ROC false-positive rate at exactly 95% TPR.
   - **Duplicate rates.** 9.6% raw and 11.64% normalised are both D2 figures; `data/cache/clean_d2_report.json`
     gives `dup_pct_observed` 11.6443.
7. **D3 attribution verified (`CONTEXT.md` caveat 5).**
   - arXiv 2511.21448 (Toth, Bisztray, Gruschka) names PhishFuzzer and links
     github.com/DataPhish/PhishFuzzer (23,100 variants).
   - CC BY 4.0 is the paper's licence. The dataset repository has no licence (GitHub API
     `license: null`, 2026-10-12).

## 2026-10-12 · RECONSTRUCTION: D3 family-audit script

The AUDIT-D3 family-overlap analysis (2026-08-17, section A) was run from a scratchpad and never
committed. For reproducibility it is reconstructed after the fact as `src/experiments/d3_family_audit.py`,
with output `results/stats/d3_family_audit.json`.
- **Inputs:** only frozen artefacts.
  - The split `data/splits/d3_en.json`.
  - The family key of the Appendix B.4 grouped split: `original_id` from `data/cache/clean_d3.pkl`
    (English rows), joined to `load_corpus("d3_en")` exactly as in `src/experiments/group_sens.py`.
    The reconstruction adds a guard that checks row alignment by sha1 body hash.
- **What it does not do:** run a model or read a score.

**It reproduces every previously logged figure exactly (13/13; the script hard-fails otherwise):**
- 19,164 documents in 3,194 families, all of size 6;
- families per split: train 3,191 / val 1,963 / test 2,004;
- test documents with a train sibling: 2,865 / 2,875 = 99.65%;
- with a val sibling: 1,567 = 54.50%;
- with a train-or-val sibling: 2,875 = 100.00%;
- val documents with a train sibling: 2,867 = 99.72%;
- 10 test documents without a train sibling, all class 0, and 0 without a train-or-val sibling.

**New figures (not logged before):**
- The 10 test documents without a train sibling belong to exactly **3** families. So 3 of the 2,004
  test families have no training member, which settles the "2–3" bound in the AUDIT-PAPER entry above.
- **2,001** families straddle the train/test boundary.
- The full partition-overlap matrix is in the JSON (e.g. 56.20% of train documents have a test sibling).

This is the family key that the family-cluster bootstrap uses. Nothing else changes: the per-document
split remains the result of record, and earlier entries are untouched.
