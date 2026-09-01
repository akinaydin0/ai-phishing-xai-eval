# PROJECT CONTEXT — read this first

**If you are an AI assistant opening a chat in this folder: read this file completely before doing
anything.** It is the single-page handover for the whole project. `CLAUDE.md` holds the operating
rules; this file holds the *state of the work*, the *verdicts*, and — most importantly — the
**caveats that must not be re-discovered the hard way**.

- **Project:** An Explainable Machine Learning Framework for Detecting AI-Generated Phishing Emails
- **Author:** Akın Aydın — MSc Machine Learning, Liverpool John Moores University
- **Final dissertation PDF (submitted version):** `AKIN-AYDIN-DISSERTATION.pdf` (repo root).
  ⚠️ It is **password/permission protected**, so its text could not be extracted programmatically.
  Everything below is grounded in the repo's own artefacts, not in the PDF.
- **Earlier draft with supervisor feedback:** `Dissertation-full-draft-Akin-AydinFeedback.pdf`
- **Canonical decision log:** `notes/decisions.md` — append-only, ~124 KB, every pre-registration,
  amendment and result entry in chronological order. **This is the ground truth.**

---

## 1. What the project is

Three model families are compared for phishing detection across three corpora, with explainability
evaluated *quantitatively* (faithfulness / stability / plausibility) rather than as decoration.

| family | members |
|---|---|
| classical | LogisticRegression, MultinomialNB, LinearSVC, DecisionTree, RandomForest (+ w2v/GloVe dense) |
| ensemble | XGBoost, LightGBM |
| transformer | DistilBERT, BERT-base, RoBERTa-base (256 and 512 token windows) |
| XAI | SHAP, LIME, Integrated Gradients — each vs a random-attribution baseline |

**Corpora** (authoritative counts; never "fix" these):

| id | source | cleaned n | legit / phish | train / val / test |
|---|---|--:|---|---|
| `d1_decontaminated_sb` (d1_sb) | Kaggle multi-source phishing (Enron sub-corpus removed) | 51,839 | 23,777 / 28,062 | 36,287 / 7,776 / 7,776 |
| `d2` | Enron-Spam (Metsis et al. 2006, AUEB) | 29,224 | 15,406 / 13,818 | 20,456 / 4,384 / 4,384 |
| `d3_en` | PhishFuzzer LLM variants, English only | 19,164 | 6,504 / 12,660 | 13,414 / 2,875 / 2,875 |

Splits are frozen, sha1-body-hash-keyed, seed 42, consumed **only** via `src.splits.load_splits`.

---

## 2. Headline results (all committed, all verified)

- **In-corpus:** everything is at ceiling (AUC >= 0.997). Tuned linear TF-IDF matches or beats
  transformers; on d3_en the classical model **significantly** beats RoBERTa (D9 comparison C1c).
- **Cross-corpus is the real story.** Transformers win 4/6 directed pairs *under the committed
  comparator pool* — see caveat 3. The AI-to-real direction (`d3_en->d2`) was **not** closed at 256
  tokens (0.731 vs classical 0.752).
- **512-token ablation (H1, directional):** raising the window recovers `d3_en->d2` to **0.844**,
  above the best classical 0.752, because the d3_en champion finally trains on fuller AI emails
  (train truncation 76% -> 22%). `d2->d3_en` unmoved. So that half of the legacy/AI break was largely
  a **truncation artefact**, not purely distributional.
- **RQ3 answer: linear TF-IDF + SHAP** — top accuracy, faithful, more stable, more plausible, and
  ~64x cheaper to attribute than LIME (0.6 s vs 39.6 s per 200 documents). This survived a serious
  correction (caveat 1) and is now supported by a paired bootstrap CI
  (pooled tfidf SHAP-LIME +0.0151, 95% CI [+0.0123, +0.0180], excludes 0).
- **Plausibility proxy validated (D10):** the cue-lexicon proxy is *directionally* valid but only
  **"fair"** (lexicon-human kappa = 0.396, on the 0.4 boundary; E2 fails by 0.004). Human-judge
  kappa = 0.702. The proxy is **conservative** — humans separate methods *more* than the lexicon does
  (9/9 contrasts), so the committed plausibility column **understates** the effect and is **ordinal,
  not absolute**.

Result tables: `results/results.csv` (124 rows), `results_512.csv` (8), `xai_metrics.csv` (30),
`xai_metrics_limefix.csv` (12), `confusion_matrices.json`, `group_sens.json`.
Narrative summaries live in `notes/*_summary.md` (13 files).

---

## 3. CAVEATS — do not re-derive these, and do not cite around them

### 1. The committed LIME numbers are VOID; use the corrected file
`src/xai/explainers.py` originally passed `split_expression=r"\w+"` to `LimeTextExplainer`, but that
argument is LIME's **separator** pattern (default `r"\W+"`) used as the *non-word* test. Every real
word was discarded, so **all 2,100 LIME attribution vectors were identically zero**, and each "LIME
top-k" was really the document's **first k word tokens**.

- **Cite `results/xai_metrics_limefix.csv`, never the 12 LIME rows of `xai_metrics.csv`** (kept
  deliberately as the "FIRST10" position-baseline record; reproducible only via
  `explainers.LEGACY_VOID_SPLIT_EXPRESSION`).
- **Withdrawn:** "LIME ~ random"; "SHAP is the only faithful method".
  **Inverted:** the faithfulness-stability trade-off — corrected LIME is *faithful but least stable*
  (11/12 units), the opposite of the committed wording.
- Corrected LIME beats SHAP on 5/12 units (both non-d3_en transformer units + all 3 cross units).
- **Lesson (now a thesis argument):** a random baseline is a *negative* control and cannot catch a
  dead explainer. `src/xai/test_known_answer.py` is the permanent **positive control** (41 checks);
  its T4 proves a zero vector fails. Any new explainer must pass it.
- Residual, disclosed: on 84/2,100 docs (4.0%; 22.5% of `transformer_d3_en`) corrected LIME returns
  ~1e-6 attributions under prediction saturation; SHAP/IG are fine on the same docs.

### 2. d3_en has total seed-family leakage (quantified, bounded, disclosed)
`src/splits.py` splits **per document**, not grouped by PhishFuzzer `Original_ID`. Since 6 variants
share each seed, **100% of d3_en test documents have a train/val sibling** (99.65% a train sibling).
There is **no clean holdout inside the frozen split** (clean subset n = 0).

- **Amendment 5 measured it** with a group-aware re-split (`data/splits/d3_en_grouped.json`,
  `results/group_sens.json`): LinearSVC 0.9993 -> **0.9928**, RoBERTa 0.9976 -> **0.9877**. So the
  inflation is only **~0.7-1.0 pp** and **C1c's direction survives** (classical still higher).
- Outward transfer is unaffected (grouped-trained LinearSVC -> D1 0.9007 vs committed 0.8783).
- Exposure is **broader than the headline trio**: every d3-trained-on-d3-test row (whole leaderboard,
  dense rows, all 256 *and* 512 in-corpus transformers, D9 imbalance, XAI/HITL d3_en units).
- **H1 is safe** — it evaluates on the **D2** test split, not d3_en.

### 3. Table 6.5's "4/6" depends on the comparator pool — state the pool
"Best classical champion per directed pair" is taken over the `__xcorpus_*` val-selected champions.
Over **all** committed classical rows the count becomes **2/6** (`d1_sb->d3_en` -0.0020,
`d2->d1_sb` -0.0270). The within-family model choice predates the target test, but the *cross-family*
pick is a post-hoc max over observed target-test AUCs. Defensible, but must be stated explicitly.

### 4. D1/D3 provenance overlap (unresolved)
62.4% of d3_en descends from seeds drawn from **D1's own sub-corpora** (Nazario 30.5%, CEAS_08 26.2%,
SpamAssassin 5.7%). D1->D3 DistilBERT scores 0.9948 on D1-descended documents vs **0.3970** on
independent-provenance ones (confounded: that slice is 92.7% positive with only 80 negatives, so it
bounds nothing). Replicates on the grouped split too. D3->D1 is **not** affected (AUC *rises* to
0.9898 on the provenance-clean slice). Fixing this needs a provenance-stratified re-split **with
retraining**.

### 5. Reference hygiene
- **Never cite** these three fabricated references (LLM hallucinations found in the original
  proposal): Aljofey et al. 2022 "IEEE Access 10, 15321"; Almeida et al. 2020 "ESWA 153, 113434";
  Alsharnouby et al. 2020 "IJHCS 141, 102432". Real: Aljofey et al. 2022 *Sci. Rep.* 12, 8842,
  DOI 10.1038/s41598-022-10841-5; Alsharnouby et al. **2015**, IJHCS 82.
- **D3 attribution is unverified at source:** the PhishFuzzer repo has **no licence at all**
  (`license: null`) and its citation block is literally `[Placeholder for citation information]`.
  "CC BY 4.0", "Tóth et al. 2025" and "arXiv 2511.21448" appear **nowhere** in the source.
  The dataset's own name there is *"The Machine Wars Email Dataset"*.
- The D3 variant generator **is** pinned: `expand.py:25` gives `MODEL_ID = "google/gemini-2.5-flash"`,
  temperature 0.7, via OpenRouter.
- Lexicon caveat: the `greeting` category drives **29.2%** (61/209) of the proxy's false positives.

---

## 4. Methodology rules that were followed (keep following them)

1. **Pre-register before computing.** Every directive's design was committed *alone*, before any
   result existed (D10 `8ef0937`, D11 `f3d5e3a`, Amendment 5 `2bad9cc`). Deviations are recorded as
   **separate dated amendments** — `notes/decisions.md` is **append-only**, earlier entries are never
   edited.
2. **Isolate ablations.** New arms write to their own namespace (`results/xai_lime_fix/`,
   `results/transformers_512/`, `d3_en_grouped.json`); canonical files stay **byte-identical**, and
   that is verified (md5 / `git diff`) and stated in the commit.
3. **Quote committed champions; never reselect.** CIs and diagnostics *annotate*, they never change a
   result.
4. **Hard-fail gates.** Reproduction asserts exit non-zero (e.g. the confusion-matrix gate re-derives
   accuracy/precision/recall/F1 from each matrix and must match to 4 dp — it passed 6/6).
5. **Positive controls, not just baselines** (see caveat 1).
6. Seed 42 everywhere; dedup before splitting; no test-set contact during tuning; every experiment
   appends exactly one row to `results/results.csv`.

---

## 5. Where things are

```
CLAUDE.md                  operating rules + Current Phase (read after this file)
CONTEXT.md                 this handover
notes/decisions.md         APPEND-ONLY canonical log (pre-registrations, amendments, results)
notes/*_summary.md         13 narrative reports (xai, lime_fix, hitl, transformer, d9, eda, cleaning)
src/                       cleaning, splits, features, experiments/, xai/
results/                   metric tables, hpo/ + transformers/ finalist params, hitl/, stats/, xai*/
data/splits/               frozen split JSONs (+ d3_en_grouped.json)
figures/                   PNG + PDF per phase (xai_lime_fix/ supersedes xai/ for LIME)
viva/                      deck, Q&A, presentation, build scripts
tools/                     archive_artefacts.ps1 / restore_artefacts.ps1
ARTEFACTS.md               what lives in the GitHub Release and how to restore it
```

**Heavy artefacts are NOT in git.** They are GitHub Release **`artefacts-v1`** on
`akinaydin0/msc-thesis` (private): 24 assets, **6.12 GB** — 15 checkpoint zips (9x256 + 6x512),
`data_raw`, `data_cache`, `data_processed`, GloVe, preproc cache, plus `MANIFEST.sha256`.
Verified 2026-09-01: every asset's byte size matches its local zip, and a fresh download reproduced
its manifest SHA-256 and extracted to the correct paths.

### Restore on a new machine

```bash
git clone https://github.com/akinaydin0/msc-thesis.git
cd msc-thesis
gh auth login
powershell -ExecutionPolicy Bypass -File ./tools/restore_artefacts.ps1
python -m venv venv
```

Use **forward slashes** for the script path in bash/Git Bash — backslashes get swallowed.

`requirements.txt` pins `torch==2.11.0+cu128`, which needs an NVIDIA GPU + CUDA wheel. On a GPU-less
machine install CPU torch instead
(`pip install torch --index-url https://download.pytorch.org/whl/cpu`) — nothing needs retraining;
checkpoint loading, inference, XAI and figure regeneration all work on CPU.
`results/hitl/.env` (Anthropic API key) is git-ignored by design and does **not** transfer; it is
only needed to re-run the LLM judge, whose outputs are already committed.

---

## 6. Open items

- **STARD-AI checklist** (supervisor item 5) — not started.
- **DeSE 2026 paper**, full paper deadline 15/09/2026 — distil from the dissertation.
- Ch6 text should state the Table 6.5 comparator pool (caveat 3) and carry the d3_en leakage
  sensitivity (caveat 2).
- Optional future work: provenance-stratified D1/D3 re-split *with* retraining (caveat 4); human
  re-annotation of the corrected-LIME sets (deferred — the D10 ratings were collected on the void
  arm and cannot be reused); adding impersonation/secrecy categories to the cue lexicon.
