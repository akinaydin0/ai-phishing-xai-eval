# MSc Thesis Project — Context

> ### ▶ START HERE: read `CONTEXT.md` first
> `CONTEXT.md` (repo root) is the project handover: current state, headline results, where every
> artefact lives (the 6.12 GB of heavy artefacts are not in this repository), and — most important —
> the **six caveats** that must not be re-derived or cited around (void LIME numbers, d3_en
> seed-family leakage, the Table 6.5 comparator pool, D1/D3 provenance overlap, fabricated
> references, the explanation sets' realised error counts). This file below holds the *operating rules*; `CONTEXT.md` holds the *state*.
> The append-only canonical log is `notes/decisions.md`.
> Submitted dissertation: 85 pp, 7 chapters + STARD-AI and group-sensitivity appendices. Its PDF is
> not in this repository: heavy artefacts and private material are available from the author on request.

## Project

**Title:** An Explainable Machine Learning Framework for Detecting AI-Generated Phishing Emails
**Author:** Akın Aydın, MSc Machine Learning, Liverpool John Moores University
**Final deadline:** 31 August 2026 (dissertation). Interim report submitted May 2026.
**Hardware:** Local workstation with NVIDIA RTX 3090 (24 GB VRAM), where all training ran. A second
machine (cloned 2026-09-02) has an AMD RX 5600 XT, which has no CUDA, so use CPU torch there. Nothing
needs retraining; see `CONTEXT.md` §5.

The project designs and evaluates an **explainable** phishing-detection framework, with
particular emphasis on **AI-generated** phishing. It compares three model families and
integrates explainability as a first-class, quantitatively-evaluated component:

- **Classical models:** Logistic Regression, Naïve Bayes, SVM, Decision Tree, Random Forest.
- **Ensembles:** XGBoost, LightGBM.
- **Transformers:** BERT-base, DistilBERT, RoBERTa-base.
- **XAI:** SHAP, LIME, Integrated Gradients — evaluated quantitatively for
  **faithfulness, stability, and plausibility** (not treated as post-hoc decoration).

## Current Phase

**XAI complete — the framework is end-to-end (EDA → cleaning → HPO/classical/ensemble → transformers → XAI).**
SHAP / LIME / Integrated Gradients on the 9 family champions + 3 transformer transfer units, evaluated
quantitatively for **faithfulness / stability / plausibility** (each vs a random-attribution baseline)
via `src/xai/` (`cue_lexicon.py` committed *before* results; `champions.py`, `explainers.py`,
`metrics.py`, `run_xai.py`, `report_xai.py`, `figures_xai.py`). Pre-registered protocol + fixed
206-term lexicon in `notes/decisions.md`. Explanation docs are **TEST-split only** (n=200 in-corpus /
100 cross, 100/class, seed 42; ids in `results/xai/explained_ids_*.json`). Each set holds up to 30
(15 cross) deliberately sampled misclassified docs plus a same-class fill that can add more errors.
The **realised** error counts are 23–33 per in-corpus set and 30–43 per cross set (`CONTEXT.md` caveat 6).
Log `results/xai_metrics.csv` (30 rows, **independently verified PASS** 2026-07-17 — row accounting
exact, 288/288 table cells consistent at 3 dp, means exact, comp@k monotone on SHAP/IG, suff@10 ≤ random
30/30); figures `figures/xai/*`; report `notes/xai_summary.md`.
Per-unit attribution JSONs (18 MB) were git-ignored at the time and have been **committed since
2026-09-01**. Run: 30 (unit,method) in **306 min**.
**Findings (RQ3):** ⚠️ **[SUPERSEDED — every LIME claim in this paragraph is VOID. See Directive 11
below and `CONTEXT.md` caveat 1. Do not cite: "LIME ≈ random", the "LIME most stable, least faithful"
trade-off (inverted), "IG is their best explainer" (revised), or the per-unit cost figures (wrong
basis — quote `timing.attr`, not `runtime_s`). Corrected numbers:
`results/xai_metrics_limefix.csv`.]** **SHAP is the only demonstrably faithful method; LIME ≈ random** (in-corpus AOPC
ensemble·SHAP 0.281 / tfidf·SHAP 0.161 / transformer·SHAP 0.107 ≈ IG 0.102 ≫ LIME ~0.02). **Faithfulness–
stability trade-off** (LIME most stable, least faithful). **Transformers hardest to explain** (IG is
their best explainer). **RQ3 answer: linear TF-IDF + SHAP** — top accuracy, faithful+plausible, cheapest
(6 s/unit vs transformer 686–2408 s). Prior transformer findings unchanged (in-corpus at ceiling;
cross-corpus win 4/6, **AI→real d3_en→d2 NOT closed** 0.731 vs 0.752; results.csv 124 rows;
`results/transformers/checkpoints/` git-ignored ~3.6 GB).
**Done:** pre-registered `max_length=512` ablation (Directive 8; isolated `results/transformers_512/`,
canonical results byte-identical). **Verdict H1 (directional):** raising the window recovers
**d3_en→d2 0.731→0.844** (above best classical 0.752 — reverses the D6 loss) because the d3_en
champion now trains on fuller AI emails (train truncation 73%→19% for the RoBERTa champion; the
often-quoted 76%→22% is the DistilBERT/BERT WordPiece figure; see decisions.md 2026-10-12); **d2→d3_en unmoved** (−0.011, d2
docs already short). So the legacy↔AI break is, for the d3_en→d2 half, largely a 256-token truncation
artifact — not purely distributional. 6 units = 145 min (~2× the 256 cost). See
`notes/transformer_512_summary.md`, `results/results_512.csv`,
`figures/models/transformer_512_cross_comparison.*`, decisions.md (2026-07-17 result).
**Also done (Directive 9, pre-registered):** statistical inference + imbalance + Pipeline A ablation
(`src/experiments/d9_*`; `results/stats|imbalance|preproc_ablation/`; `notes/d9_stats_robustness_summary.md`).
12/12 scores reproduced == committed to 4 dp; canonical CSVs byte-identical. **7 paired comparisons**
(bootstrap B=10k + Holm-McNemar): in-corpus **classical significantly beats RoBERTa on d3_en (C1c)**,
tied at ceiling on d1_sb/d2; **512 recovery C2/C3 strongly significant** (Δ+0.113/+0.092, p≈0) and beats
the classical cross champion; **C5 null held** (CI incl 0). Imbalance: **FP-per-1,000-legit @95% recall**
balloons at 2% prevalence (d3_en 19.5→103.6), robust family is corpus-dependent. **Preproc (val-only):
placeholders V2 help transfer (+0.034, 5/6; hypothesis HELD), stopword-removal V1 hurts**. All annotate,
no reselection; review PASS 5/5.
**Also done (Directive 10, pre-registered): HITL + LLM-judge plausibility validation** — the response to
supervisor item 6 (plausibility over-reading). Pre-registration `8ef0937` (committed ALONE; guard +
regeneration anchor PASS: plaus@10 reproduced to 4 dp on `tfidf_d1_sb·shap` 0.2338 /
`transformer_d3_en·shap` 0.2360 / `·ig` 0.2180, proving the git-ignored attribution JSONs regenerable);
3 dated pre-data amendments (`9a32bb7` info-parity/blinding fix, `3f6edc8` temperature deprecated for
`claude-opus-4-8` → no sampling params, `3b2a155` 1-doc blinding break + pre-committed sensitivity);
ratings committed untouched `51c30b6` before analysis. **60 TEST docs** (20/corpus, 15 TP + 5 FP,
phishing-predicted), **260 blind sets / 2,558 token judgments**; judge `claude-opus-4-8` 251 ok / 9
refusals. **Verdict: the cue-lexicon proxy is directionally validated, "fair" in calibration** —
**P2 lexicon–human κ=0.396 (fair, on the 0.4 boundary; E2 FAILS by 0.004)**, per corpus 0.434/0.421
(moderate) / 0.368; **E1 holds 4/4** (separated 3/4); **concordance 6/6 valid (LIME-free) contrasts**;
**P3 human–judge κ=0.702 (93.1%; pre-registered band "moderate", which is "substantial" on the full
Landis–Koch scale used in the dissertation and paper)**, ρ 0.818/0.734. **The proxy is CONSERVATIVE: human
separation > lexicon in 9/9 method-vs-random contrasts** → the committed plausibility column
**understates** the effect and is **ordinal, not absolute**. Lexicon gap (not acted on): no
impersonation/secrecy category; `greeting` drives 29.2% of its false positives. Sensitivity excl. the
1 blinding-compromised doc: κ→0.409 (band flips, but 10/60 leave-one-out exclusions do the same ⇒
ordinary jitter) — **primary 60-doc result governs**. Nothing reselected; `xai_metrics.csv` untouched.
See `notes/hitl_summary.md`, `figures/hitl/hitl_cue_rates.*`, `results/hitl/`, decisions.md
(2026-07-26 result). Review PASS 4/4 (S1–S4).

**Also done (Directive 11, pre-registered): corrected-LIME isolated re-run — Amendment 4 defect RESOLVED**
(option (a)). Pre-reg `f3d5e3a` (committed ALONE with the fix + a permanent **positive-control suite**
`src/xai/test_known_answer.py`, **41 checks PASS**: T0 predicted-phishing precondition, T1 planted cue in
top-10 distinct words, T1b mass localisation (gates SHAP/LIME, diagnostic for IG), T2 non-degeneracy on
real docs, T3 legacy value still reproduces the void arm, **T4 a ZERO vector must fail T1** ⇒ the suite
provably detects the original defect). Fix = `split_expression` omitted (LIME default `r"\W+"`); the
defective value kept as `explainers.LEGACY_VOID_SPLIT_EXPRESSION` for audit reproduction. **12/12 units
re-run in 2.75 h** on the SAME committed `explained_ids`, committed D7 params verbatim; isolated
`results/xai_metrics_limefix.csv` + `results/xai_lime_fix/` (its JSONs were git-ignored at the time and have been committed since 2026-09-01); **canonical `xai_metrics.csv`
(30), `results.csv` (124), `results_512.csv` (8), `figures/xai/*` byte-untouched**.
**Findings — the void arm had inverted LIME on BOTH axes.** Corrected LIME AOPC **0.071–0.295, above its
random baseline on 12/12 units by at least 4×** (ratios 4.0×–2357×; top end is a 0.0001-denominator
artifact) ⇒ "**LIME ≈ random**" and "**SHAP is the only faithful method**" are **WITHDRAWN**; corrected
LIME is the **LEAST stable** method on **11/12** units (per-unit Jaccard 0.355–0.700, seed-agree
0.354–0.711; exception `xfer_d3_en__d2` where IG 0.4907 < LIME 0.5019) ⇒ the "faithfulness–stability
trade-off" is **INVERTED** (LIME = faithful-but-unstable; the committed Jaccard 0.99 / seed-agree 1.000 /
Spearman nan were constant-vector artifacts). **5/12 units flip** to LIME as most faithful
(`transformer_d1_sb` 0.1233 vs SHAP 0.0931, `transformer_d2` 0.1753 vs 0.1372, **all 3 cross units**
0.218/0.282/0.295 vs 0.184/0.213/0.223); the 6 sklearn units do not flip. **RQ2 revised:** on family
means corrected LIME is the most faithful + most plausible transformer explainer (0.1417 / 0.1845 vs SHAP
0.1069 / 0.1615, IG 0.1024 / 0.1476) — not universal per unit — but **IG stays the pragmatic transformer
default** (best stability 0.7028, 6.6× cheaper to attribute). **RQ3 answer HOLDS on replaced grounds:**
for linear TF-IDF, SHAP beats corrected LIME on AOPC 0.1613 vs 0.1462, stability 0.9091 vs 0.6508,
plausibility 0.2183 vs 0.1781 (not on `tfidf_d1_sb`), and **attribution cost 0.6 s vs 39.6 s (64.5×)**.
**Cost basis matters:** `runtime_s` is the whole pipeline incl. a LIME-only seed-agreement step — quote
`timing.attr`, not `runtime_s`. LIME is **viable, not broken** — just dominated for that family.
**Disclosed residual:** on **84/2,100 docs (4.0%)** corrected LIME returns numerically negligible
attributions (max|attr|≈1e-6) on near-saturated predictions — `transformer_d3_en` 22.5%,
`xfer_d3_en__d1_sb` 21%, `transformer_d1_sb` 9%, none on sklearn units; SHAP/IG unaffected on the same
docs. Biases those units' AOPC down / stability up (conservative), but leaves `transformer_d3_en`
IG 0.1288 vs LIME 0.1265 **unresolved**. **Not done (stated):** human re-annotation of corrected-LIME
sets (deferred) ⇒ plausibility is the lexicon proxy only; stored fix JSONs are lossy at 5 dp (CSV
authoritative). **D11b added paired CIs (D9 bootstrap verbatim, B=10k, seed 42) for the two knife-edge
orderings: K1 the RQ3 margin is SEPARATED** (pooled tfidf SHAP−LIME +0.0151, CI [+0.0123, +0.0180];
excludes 0 on 3/3 corpora too) — the RQ3 faithfulness leg is now inferentially supported; **K2
`transformer_d3_en` IG−LIME is NOT separated** (+0.0023, CI [−0.0211, +0.0243]) ⇒ that ordering is
formally unresolved, so "5/12 flip" = 5 flips + 1 unresolved. All 8 recomputed AOPC aggregates
reproduced the committed values to 4 dp. Other comparisons remain point estimates. The canonical `xai_metrics.csv` LIME rows are deliberately no
longer reproducible from current code (use `LEGACY_VOID_SPLIT_EXPRESSION`); `notes/xai_summary.md` keeps
its erratum. Review PASS 5/5 (S1–S5) with corrections applied. See `notes/lime_fix_summary.md`,
decisions.md (2026-07-26 pre-reg + result).

**Status (2026-10-12):** dissertation **submitted** (LaTeX compile "main 18", 1 Sep 2026; the PDF is
not in this repository; STARD-AI = Appendix A, so supervisor item 5 is done) and **viva completed**.
**DeSE 2026 paper in progress**, written in OpenAI Prism with a Cowork agent. The conference paper
(source and PDF) lives outside git. A read-only audit of
that draft against this repo matched 257 of 264 sub-claims. The mismatches, plus three undisclosed
implementation details, are logged in decisions.md (2026-10-12 AUDIT-PAPER) and in `CONTEXT.md`
caveat 6. Heavy artefacts are restored on the second machine from the verified `_release` zips.

**Reviewer response (2026-10-12):**
- The D3 family audit was reconstructed: `src/experiments/d3_family_audit.py` reproduces 13/13 logged figures.
- A pre-registered family-cluster bootstrap was run for C1c / C4 / C5 (`results/stats/cluster_bootstrap_*.json`).
  C1c and C4 still exclude 0 and C5 still includes 0. Design effects are 0.96 / 1.29 / 1.27.

**Public version (2026-10-12):** this repository is built from a rewritten history (private material
purged, two redactions, tool-generated trailers removed). See decisions.md (2026-10-12 HISTORY
REWRITE) and `notes/history_rewrite_2026-10-12_commit_map.txt` for old → new commit hashes.

**Next:** DeSE paper finalisation (deadlines: item 8 below); public release once README, LICENSE,
DATA.md and CITATION.cff exist.

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
- **Source:** GitHub — `DataPhish/PhishFuzzer`. Paper: Toth, Bisztray & Gruschka, arXiv 2511.21448,
  **verified 2026-10-12**: its abstract names PhishFuzzer and links this repo. CC BY 4.0 is the
  *paper's* licence; the dataset repo itself has **no licence** (GitHub API `license: null`).
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
- **Token-length medians (RAW):** modern phishing 62, modern legitimate 199, Enron ham 176,
  Enron spam 138, AI-generated phishing 232.
  - The dissertation and paper use the **cleaned** values: Enron ham 169, Enron spam 132,
    AI phishing 229 (`notes/eda_v2_summary.md`).
  - The "D2 median 176 WordPiece tokens" in the 512 discussion is a different statistic: the D2
    train split under the DistilBERT tokenizer. It only coincides with the raw ham 176.
- **Pairwise Jaccard (top-500 tokens) — cleaned (primary):** legacy↔modern **0.353**,
  legacy↔AI **0.248**, modern↔AI **0.256** (post-cleaning; matches `notes/decisions.md`).
  Raw (pre-cleaning) values were legacy↔modern 0.351, legacy↔AI 0.250, modern↔AI 0.263.
  Either way, three distinct lexical clusters (legacy↔AI lowest); substantiates dataset-aware
  evaluation. **Always label which triple (cleaned vs raw) — never cite either unlabelled.**

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

### Weekly report 1 (17 Jul 2026 — see `notes/decisions.md`)

6. **Overfitting/plausibility over-reading** → **Directive 10**: method-blind human plausibility
   annotation (~20 docs/corpus from the existing TEST `explained_ids`) + LLM-judge cross-check;
   annotation guide + agreement metrics **pre-registered before any rating**.
7. **LSEP** stays **Chapter 1 §1.4** (supervisor approved either placement); DOI-anchored at writing.
8. **Conference: DeSE 2026** (16–18 Dec 2026, Malaysia; IEEE Xplore) — **full paper 15/09/2026**,
   camera-ready + registration 15/10/2026; distil from the dissertation draft in August.
9. **Viva** question redirected to programme leader **[name redacted]**. Resolved: **viva completed** (confirmed 2026-10-12).

## Reference Errata (Important — do not cite these)

Three references in the original project proposal (`proposal.docx`, not in this repository) were found to be fabricated (LLM
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

## Git

Never add Co-Authored-By, Claude-Session or 'Generated with Claude Code' lines to commit messages or pull request descriptions. Commits are authored by the repository owner only.

## Heavy artefacts (added 2026-09-01)
- Files over GitHub's 100 MB limit (transformer checkpoints, data caches, GloVe, raw data, preproc cache) are NOT in git: they live as assets of GitHub Release `artefacts-v1` of the author's private repository, not of this one; heavy artefacts and private material are available from the author on request. See ARTEFACTS.md; create with tools/archive_artefacts.ps1, restore with tools/restore_artefacts.ps1 (verifies SHA-256; needs access to the private release).
- XAI attribution JSONs, stats scores and HPO logs are committed directly (about 32 MB). `_release/` is the ignored staging folder.
- Restoring without `gh`: if the release zips are already local (e.g. a cloud backup of `_release/`), check them against `MANIFEST.sha256`, then run `tar.exe -xf <zip>` from the repo root. Every restored path is git-ignored. The second machine was restored this way on 2026-10-12 (22/22 hashes matched).
- The viva package, the dissertation PDFs, the project proposal and the conference-paper source are not in this repository: heavy artefacts and private material are available from the author on request.
