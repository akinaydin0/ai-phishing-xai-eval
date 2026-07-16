# MSc Thesis Project — Context

## Project

**Title:** An Explainable Machine Learning Framework for Detecting AI-Generated Phishing Emails
**Author:** Akın Aydın, MSc Machine Learning, Liverpool John Moores University
**Final deadline:** 31 August 2026 (dissertation). Interim report submitted May 2026.
**Hardware:** Local workstation with NVIDIA RTX 3090 (24 GB VRAM).

The project designs and evaluates an **explainable** phishing-detection framework, with
particular emphasis on **AI-generated** phishing. It compares three model families and
integrates explainability as a first-class, quantitatively-evaluated component:

- **Classical models:** Logistic Regression, Naïve Bayes, SVM, Decision Tree, Random Forest.
- **Ensembles:** XGBoost, LightGBM.
- **Transformers:** BERT-base, DistilBERT, RoBERTa-base.
- **XAI:** SHAP, LIME, Integrated Gradients — evaluated quantitatively for
  **faithfulness, stability, and plausibility** (not treated as post-hoc decoration).

## Current Phase

**Modelling — transformer fine-tuning complete (HPO + classical/ensemble + dense done earlier).**
Fine-tuned **DistilBERT, BERT-base, RoBERTa-base** on the frozen 70/15/15 splits (subject+body,
`max_length=256`) via `src/experiments/transformers_ft.py` (lr ∈ {2e-5,3e-5,5e-5}, ≤4 epochs,
batch 32 + fp16, wd 0.01, warmup 0.1; **val-AUC early-stop patience 1, best-val checkpoint kept,
best lr by val AUC, TEST evaluated once per finalist**). Cross-corpus champions (best val AUC per
source; ties→smaller model) in `transformer_cross.py` → `figures/models/transformer_cross_*_heatmap.*`;
reporting in `notes/transformer_summary.md`; per-run val curves in `results/transformers/*.json`;
9 finalist checkpoints in `results/transformers/checkpoints/` (git-ignored, ~3.6 GB); log in
`results/results.csv` (124 rows).
**Findings:** in-corpus **at ceiling — transformers match but don't beat tuned linear TF-IDF**
(best-T vs classical AUC: d1_sb 0.9999/0.9998, d2 0.9999/0.9996, d3_en 0.9979/0.9993). Cross-corpus
**mixed**: transformers beat best classical on **4/6** pairs — every transfer touching modern-real
d1_sb (d3_en→d1_sb 0.966 vs 0.889) — but **lose both legacy-Enron↔AI pairs**; the **AI→real
(d3_en→d2) gap is NOT closed** (0.731 vs classical 0.752). Truncation@256 hardest on d3_en (76%,
median 364 WordPiece). Full 9-grid **132.8 min** on the RTX 3090 (peak ≤5.2 GB); GPU fine-tuning is
**not bit-exact** (SDPA attention has no deterministic CUDA kernel) — mitigated by fixed seeds,
cudnn.deterministic, CUBLAS workspace pin, pinned versions, and saved checkpoints as the
reproducibility anchor (see `notes/decisions.md` "Transformer fine-tuning").
**Next:** XAI (SHAP/LIME/IG) — reuses the saved finalist checkpoints.

> **Standing rule:** every task's final commit updates this Current Phase section.

## Datasets (authoritative counts — do not change)

### D1 — Kaggle multi-source phishing dataset
- **Source:** Kaggle — `naserabdullahalam/phishing-email-dataset`
- **URL:** https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset
- **Path:** `data/raw/phishing_email_dataset/` (Kaggle CLI needs `kaggle.json`; if absent,
  download the ZIP manually and unzip into this folder).
- **Counts:** 82,486 emails aggregated from 6 sub-corpora (CEAS_08, Enron, Ling-Spam,
  Nazario, Nigerian Fraud, SpamAssassin); ~52% phishing (label 1).
- **Decontaminated:** 52,719 emails — the 29,767-row `Enron.csv` sub-collection removed,
  because those rows duplicate the D2 legacy corpus and would contaminate cross-corpus
  comparison. Use `load_phishing_dataset(decontaminate=True)`.

### D2 — Enron-Spam (Metsis et al. 2006)
- **Source:** AUEB (Androutsopoulos lab), pre-processed ham/spam folder structure.
- **URL:** http://www2.aueb.gr/users/ion/data/enron-spam/  (6 `.tar.gz`: enron1–enron6)
- **Path:** `data/raw/enron-spam/`
- **Counts:** 33,716 emails — 16,545 ham (label 0) / 17,171 spam (label 1).

### D3 — PhishFuzzer (Tóth et al. 2025)
- **Source:** GitHub — `DataPhish/PhishFuzzer` (paper: arXiv 2511.21448; CC BY 4.0).
- **Path:** `data/raw/phishfuzzer/` (two JSON files: original seeds + entity-rephrased variants).
- **Counts:** 23,100 emails = 3,300 seeds + 19,800 LLM variants.
- **Usage rule:** use **only the 19,800 variants** (`load_phishfuzzer(variant="variants")`).
  Exclude non-English variants from cross-corpus comparisons.
  - ⚠️ **Count correction:** the "742 non-English" figure refers to the *full 23,100-row
    corpus*. Among the **19,800 variants only**, non-English count is **636** (635 after
    cleaning). Use 636/635 when filtering variants; 742 applies only if seeds are included.
- **Labels (binary):** Phishing + Spam → 1, Valid → 0 (see `notes/decisions.md`).

## Known data facts (raw, from EDA — for sanity-checking downstream work)

- **Exact-duplicate rates — two definitions, kept distinct** so dissertation figures
  never look contradictory:
  - **9.6% exact-raw** on the untouched loaded text (D2; ≈3,253 rows). D1 ~0.5%, D3 ~0.3%.
    This is the interim-report figure.
  - **11.64% after normalisation** (D2; 3,926 rows) — header-stripping + whitespace-collapse
    make additional near-identical bodies hash-identical. The cleaning pipeline removes this
    larger set. See `notes/cleaning_report.md`.
- **D2 class-balance finding:** 74% of D2's duplicates (2,918 of 3,926) are spam —
  spam-campaign repetition. Mandatory full within-corpus dedup therefore shifts D2's spam
  share **50.93% → 47.28% (−3.65pp)**. This is an expected consequence of the dedup rule,
  not a defect; duplicates are never retained to preserve a ratio. Post-cleaning D2 is still
  healthy (15,406 ham / 13,818 spam). The ±2pp class-shift guard is replaced for the three
  frozen corpora by exact **determinism fingerprints** (`CLEANING_EXPECTED` in
  `src/cleaning.py`): any drift hard-fails the pipeline and forces a conscious re-pin.
- **Token-length medians:** modern phishing 62, modern legitimate 199, Enron ham 176,
  Enron spam 138, AI-generated phishing 232.
- **Pairwise Jaccard (top-500 tokens):** legacy↔modern 0.351, legacy↔AI 0.250,
  modern↔AI 0.263 — three distinct lexical clusters; substantiates dataset-aware evaluation.

## NON-NEGOTIABLE evaluation rules

1. **Dedup BEFORE any split.** Exact-deduplicate within each corpus before train/val/test
   partitioning — duplicates spanning splits leak labels and inflate scores.
2. **Decontamination as above.** D1 cross-corpus work uses the decontaminated 52,719-row
   set; the Enron sub-collection lives only in D2.
3. **Dataset-aware evaluation.** Cross-corpus degradation (train on one corpus, test on
   another, both directions) is a **primary reported outcome**, not an afterthought.
4. **No test-set leakage in tuning.** All hyperparameter optimisation happens on a
   validation split of the **training** corpus only — never touch test sets during tuning.
5. **Global seed = 42** for every stochastic step (splits, model init, resampling).
6. **Results log.** Every experiment appends exactly one row to `results/results.csv`.

## Conventions

- **Directory map:**
  ```
  msc-thesis/
  ├── CLAUDE.md
  ├── data/
  │   ├── raw/{phishing_email_dataset, enron-spam, phishfuzzer}/   (git-ignored)
  │   ├── processed/                                               (loader caches)
  │   └── cache/            clean_<id>.pkl + clean_<id>_report.json (cleaned corpora)
  ├── src/       data_loader.py, cleaning.py, eda_v2.py
  ├── notebooks/ 01_eda.ipynb (+ build script)
  ├── figures/   PNG + PDF (eda_v2/ = EDA-on-cleaned-data figures)
  ├── notes/     eda_findings.md, cleaning_report.md, eda_v2_summary.md, decisions.md
  ├── results/   results.csv
  └── report/
  ```
- **Environment:** Python 3.11.9 venv at `./venv/`. Freeze after any install:
  `./venv/Scripts/python.exe -m pip freeze > requirements.txt`.
- **Figures:** save as **both PNG and PDF** (PNG for the report, PDF for LaTeX/vector).
- **Scripts:** runnable **headless from the repo root** (e.g. `python -m src.cleaning --all`,
  `python -m src.eda_v2`).
- **Corpus ids:** `d1_full`, `d1_decontaminated`, `d1_decontaminated_sb` (subject+body
  natural-text D1 for cross-corpus EDA), `d2`, `d3`.
- **Phase log:** every task's final commit updates the **Current Phase** section above.
- **Figure provenance:** every figure title/caption names its universe — corpus id + row
  count (e.g. "d1_decontaminated_sb, test n=7,776").
- **Frozen splits:** all experiments consume `data/splits/<corpus>.json` via
  `src.splits.load_splits` (sha1-body-keyed, seed 42); never re-split ad hoc.
- **Transformer finalist checkpoints:** the 9 fine-tuned models live at
  `results/transformers/checkpoints/<model>__<corpus>/` (HF `save_pretrained` format:
  `model.safetensors` + `config.json` + tokenizer), where `<model>` ∈
  {`distilbert-base-uncased`, `bert-base-uncased`, `roberta-base`} and `<corpus>` ∈
  {`d1_decontaminated_sb`, `d2`, `d3_en`}. **Git-ignored** (~3.6 GB, regenerable via
  `python -m src.experiments.transformers_ft --all`). The **XAI phase reuses these**:
  `AutoModelForSequenceClassification.from_pretrained(<dir>)` + `AutoTokenizer.from_pretrained(<dir>)`;
  per-model metadata (selected lr, val curves, truncation, versions) in
  `results/transformers/<model>__<corpus>.json`.

## Supervisor feedback (5 Jul 2026 — honour throughout)

1. Use **peer-reviewed** sources.
2. **Clean before EDA** (data cleaning precedes exploratory analysis).
3. Perform **hyperparameter optimisation** for all models.
4. Keep references **consistent and with valid DOIs**.
5. Complete the **STARD-AI checklist** on completion.

## Reference Errata (Important — do not cite these)

Three references in the original `proposal.docx` were found to be fabricated (LLM
hallucination). DO NOT cite them in any output:

- Aljofey et al. (2022) "IEEE Access 10, 15321-15333" — does not exist. Real paper:
  Aljofey et al. (2022), *Scientific Reports* 12, 8842, DOI 10.1038/s41598-022-10841-5.
- Almeida et al. (2020) "Stopword removal, ESWA 153, 113434" — does not exist.
- Alsharnouby et al. (2020) "IJHCS 141, 102432" — does not exist. Real paper:
  Alsharnouby et al. (2015), IJHCS 82.

## Working Style

- Plan before executing; checkpoint at methodological decisions before committing to them.
- Flag inference vs cited fact. Don't fabricate statistics — if data isn't present, say so.
- Use the existing venv. Commit at logical milestones with descriptive messages.
