# An Explainable Machine Learning Framework for Detecting AI-Generated Phishing Emails: Leakage-Controlled Evaluation Across Legacy, Modern and AI-Generated Corpora

This repository is the code and artefact record of an MSc dissertation (Liverpool John Moores University, 2026) and of a paper prepared for submission to DeSE 2026, the 19th International Conference on Developments in eSystems Engineering. The authors are Akın Aydın and Dhiya Al-Jumeily. Contact: akinaydin0@outlook.com.

## What is here

- `src/`: the code for cleaning, the frozen splits, the classical, ensemble and transformer experiments, the statistics, and the explanation pipeline with its evaluation.
- `data/splits/`: the frozen split identifiers. Each document is identified by the SHA-1 hash of its cleaned body; no text is stored.
- `notes/decisions.md`: the append-only decision log, with the pre-registered designs and comparisons and the dated amendments. The `notes/*_summary.md` files are the narrative reports of each phase.
- Results tables:
  - `results/results.csv` (124 rows, one per experiment run) and `results/results_512.csv` (8 rows, the 512-token ablation);
  - `results/xai_metrics.csv` (30 rows) and `results/xai_metrics_limefix.csv` (12 rows, the corrected LIME arm);
  - `results/stats/stats_summary.csv`, `results/stats/cluster_bootstrap_*.json`, `results/stats/d3_family_audit.json` and `results/group_sens.json`;
  - the supporting files under `results/`: tuned hyperparameters, transformer run metadata, cached test scores, the cluster-bootstrap inputs and the confusion matrices.
- Explanation outputs: `results/xai/` (attribution vectors and the identifiers of the explained documents) and `results/xai_lime_fix/` (the corrected LIME arm).
- The human and LLM-judge validation record, `results/hitl/`. It includes the method key without the email text (`key.json`) and the raw judge replies (`judge_raw.jsonl`).
- The planted-cue positive-control suite, `src/xai/test_known_answer.py`.
- Figures in `figures/`, as PNG and PDF.
- `CONTEXT.md` (project handover, including the caveats that qualify the results), `CLAUDE.md` (the operating rules used with the AI coding assistant) and `ARTEFACTS.md` (the heavy-artefact archives).

Not here, and why:

- The three corpora. They remain under their providers' terms; see [DATA.md](DATA.md).
- The 15 fine-tuned transformer checkpoints, the GloVe vectors and the caches (cleaned corpora and preprocessing). They are too large for git.
- The API key used for the LLM judge. It is private.
- The dissertation and the conference paper.

Heavy artefacts are available from the author on request.

## Environment

The code uses Python 3.11.9 (see `.python-version`). Create a virtual environment, activate it, and install one of the two pinned requirement files:

```bash
python -m venv venv
# with an NVIDIA GPU:
pip install -r requirements.txt
# without one:
pip install -r requirements-cpu.txt
```

- `requirements.txt` is for a machine with an NVIDIA GPU. Its header adds the PyTorch CUDA 12.8 index for `torch==2.11.0+cu128`.
- `requirements-cpu.txt` is for a machine without one. It has the same pins with `torch==2.11.0+cpu`.

Hardware:

- **Training** ran on a single NVIDIA RTX 3090 (24 GB).
- **On CPU:** the statistics scripts, the explanation evaluation on the classical models and the positive-control suite all run on CPU. On a CPU-only machine with network access blocked, the suite passed 41 of 41 checks in 952 s.

Some steps fetch public resources on first use:

- `transformers_ft` and `group_sens` download the base models from the Hugging Face Hub.
- `src.eda_v2` downloads the NLTK stopwords and words lists.
- The dense baselines read the GloVe vectors from `data/embeddings/glove.6B.300d.txt` (glove.6B, Stanford NLP).

## Data

The corpora are not included. Obtain them from their sources and place them where `src/data_loader.py` expects them:

| Corpus | Source | Place under |
|---|---|---|
| D1, Kaggle multi-source phishing dataset | https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset (download the archive; a free Kaggle account is needed) | `data/raw/phishing_email_dataset/`, containing `phishing_email.csv`, `Enron.csv`, `CEAS_08.csv`, `Ling.csv`, `Nazario.csv`, `Nigerian_Fraud.csv` and `SpamAssasin.csv` (spelt so in the archive) |
| D2, Enron-Spam | http://www2.aueb.gr/users/ion/data/enron-spam/ (the six preprocessed archives `enron1` to `enron6`) | `data/raw/enron-spam/enron1/` to `enron6/`, each with `ham/` and `spam/` |
| D3, PhishFuzzer | `git clone https://github.com/DataPhish/PhishFuzzer data/raw/phishfuzzer`, then `git -C data/raw/phishfuzzer checkout 1e21dd4` | `data/raw/phishfuzzer/`; the file used is `PhishFuzzer_emails_entity_rephrased_v1.json` |

Then rebuild the cleaned corpora and the splits:

```bash
python -m src.cleaning --all
python -m src.splits --all
git diff --exit-code -- data/splits
```

- **`src.cleaning`** writes the cleaned corpora to `data/cache/` and regenerates `notes/cleaning_report.md`. It stops with an error if a corpus does not reproduce its pinned row and class counts (`CLEANING_EXPECTED` in `src/cleaning.py`).
- **`src.splits`** rewrites the four split files in `data/splits/`.
- **Checking the result.** Each split lists the SHA-1 hashes of the cleaned bodies, so an empty `git diff` shows that the rebuilt splits match the committed ones document for document.

[DATA.md](DATA.md) gives the licences, the cleaning rules and the counts.

## Reproduction

Run every command from the repository root. Several scripts resume and skip work whose outputs already exist, for example `transformers_ft`, `run_xai`, `run_lime_fix` and `d10_judge`; each module's docstring describes its behaviour.

| Artefact | Commands, in order | Notes |
|---|---|---|
| Frozen splits, `data/splits/*.json` | `python -m src.cleaning --all`<br>`python -m src.splits --all` | `d3_en_grouped.json` comes from `group_sens`, below. |
| `results/results.csv` (124 rows) | `python -m src.experiments.run_baselines`<br>`python -m src.experiments.hpo --all`<br>`python -m src.experiments.dense --all`<br>`python -m src.experiments.cross_corpus`<br>`python -m src.experiments.transformers_ft --all`<br>`python -m src.experiments.transformer_cross` | **Order:** `run_baselines` starts a fresh `results.csv` (it deletes the existing file) and writes its rows directly. The other five append their rows through `src/experiments/reslog.py`, in the order the decision log records them (2026-07-07 to 2026-07-16). **Inputs:** `dense` needs the GloVe vectors. **Summaries:** `python -m src.experiments.report_hpo` and `python -m src.experiments.transformer_report`. |
| `results/results_512.csv` (8 rows) | `python -m src.experiments.transformers_ft --corpora d3_en d2 --max-length 512`<br>`python -m src.experiments.transformer_cross --max-length 512 --corpora d2 d3_en`<br>`python -m src.experiments.transformer_512_report` | Runs six units (DistilBERT, BERT-base and RoBERTa-base on D2 and D3) at 512 tokens. The outputs go to the isolated `results/transformers_512/` and `results/results_512.csv`. The report applies the pre-registered decision rule. |
| `results/xai_metrics.csv` (30 rows) | `python -m src.xai.run_xai --all`<br>`python -m src.xai.report_xai` | **LIME rows:** the 12 LIME rows are the void-LIME FIRST10 arm that the paper discloses. They reproduce only when LIME is called with `explainers.LEGACY_VOID_SPLIT_EXPRESSION` (the `lime_split_expression` argument of `explainers.attribute`), because the current code uses the corrected explainer. **Checking without writing:** `python -m src.xai.run_xai --all --no-persist` recomputes and prints the metrics without writing them. **Figures:** `python -m src.xai.figures_xai`. |
| Corrected LIME, `results/xai_lime_fix/` and `results/xai_metrics_limefix.csv` (12 rows) | `python -m src.xai.run_lime_fix`<br>`python -m src.xai.report_lime_fix`<br>`python -m src.xai.figures_lime_fix`<br>`python -m src.experiments.d11b_ci` | Uses the same documents and parameters as the committed run; only the LIME argument is corrected. `d11b_ci` writes `results/xai_lime_fix/knife_edge_ci.json`. |
| Statistics, `results/stats/stats_summary.csv` | `python -m src.experiments.d9_scores`<br>`python -m src.experiments.d9_stats`<br>`python -m src.experiments.d9_report` | `d9_scores` caches the test scores in `results/stats/scores/` and needs the 256- and 512-token checkpoints. `d9_report` writes `notes/d9_stats_robustness_summary.md`. It also reads the imbalance and preprocessing results, which come from `python -m src.experiments.d9_imbalance` and `python -m src.experiments.d9_preproc`. |
| D3 family audit, `results/stats/d3_family_audit.json` | `python -m src.experiments.d3_family_audit` | Stops with an error unless it reproduces the 13 previously logged figures. |
| Family-cluster bootstrap, `results/stats/cluster_bootstrap_{C1c,C4,C5}.json` | `python -m src.experiments.cluster_bootstrap_inputs`, then the three commands below the table | The inputs are committed in `results/stats/cluster_inputs/`. The committed deltas are +0.0017 (C1c), +0.0681 (C4) and -0.0106 (C5). |
| Grouped split and sensitivity, `data/splits/d3_en_grouped.json` and `results/group_sens.json` | `python -m src.experiments.group_sens` | The first run downloads `roberta-base` from the Hugging Face Hub. |
| Human and LLM-judge validation, `results/hitl/` | `python -m src.experiments.d10_restore_text`<br>`python -m src.experiments.d10_analyze`<br>`python -m src.experiments.d10_figures` | **The committed key** (`key.json`) is the historical package that the annotators rated; its LIME sets are the void FIRST10 sets. **Restoring the text:** `d10_restore_text` refills the email text by hash from the cleaned test splits; do not commit the restored key. **The judge:** `python -m src.experiments.d10_judge` needs `ANTHROPIC_API_KEY` and the model `claude-opus-4-8`. It resumes from `judge_raw.jsonl` and is not bit-reproducible. **Do not use `d10_package` to rebuild the key:** `python -m src.experiments.d10_package` builds a new package with the current explainers, does not reproduce the committed one, and overwrites `key.json`, `annotate.html` and `blind_sheets.csv`. |
| Positive-control suite | `python -m src.xai.test_known_answer` | Exit code 0 means every check passed (41 of 41 on CPU in 952 s). `--families tfidf` runs a quick subset. The suite needs the checkpoints and the cleaned corpora. |

The three family-cluster bootstrap commands:

```bash
python src/experiments/cluster_bootstrap_d3.py --csv results/stats/cluster_inputs/c1c.csv --name C1c --expected-delta 0.0017 --out results/stats/cluster_bootstrap_C1c.json
python src/experiments/cluster_bootstrap_d3.py --csv results/stats/cluster_inputs/c4.csv --name C4 --expected-delta 0.0681 --out results/stats/cluster_bootstrap_C4.json
python src/experiments/cluster_bootstrap_d3.py --csv results/stats/cluster_inputs/c5.csv --name C5 --expected-delta -0.0106 --out results/stats/cluster_bootstrap_C5.json
```

Other entry points:

- `python -m src.eda_v2`: the exploratory analysis of the cleaned corpora.
- `python -m src.experiments.confusion_matrices`: the confusion matrices.
- `python -m src.experiments.d10_select`: selects the validation documents.
- `python -m src.experiments.d10_anchor`: checks that the SHAP and IG attributions regenerate the committed plausibility to four decimal places.

## Pre-registration and the decision log

`notes/decisions.md` is chronological and append-only. Each entry is headed by its date and a title, and later entries correct or qualify earlier ones without editing them.

- **Pre-registration.** Designs were pre-registered in entries committed before any of their results existed, for example Directives 7 to 11 and Amendment 5.
- **Amendments.** Changes to a registered design are recorded as dated amendments. Those that changed what would be computed were committed before that computation ran, for example Amendment 5 and the 2026-10-12 amendment for the family-cluster bootstrap.
- **Defects found afterwards** are recorded as such, for example Amendment 4, the void LIME arm.
- **Commit hashes.** Those cited in the log belong to the original history. The history was rewritten for public release, and every old hash maps to its new one through `notes/history_rewrite_2026-10-12_commit_map.txt`. The entry [2026-10-12 · HISTORY REWRITE: public-release preparation](notes/decisions.md#2026-10-12--history-rewrite-public-release-preparation) describes the rewrite.

## Corrections to the dissertation text

The code verification for the paper found four statements in the dissertation that the decision log (entries of 12 October 2026) and the committed results correct. The RoBERTa champion truncated 73.5 per cent of the D3 training messages at 256 tokens and 18.7 per cent at 512, not 76 and 22 per cent (that is the DistilBERT and BERT figure). 99.65 per cent (2,865 of 2,875) of the D3 test documents have a seed-family sibling in the training split, and 10 do not. The explanation sets take up to 30 misclassified documents by design, but the realised counts are 23 to 33 per in-corpus unit. The 84 documents with negligible corrected-LIME attributions (4.0 per cent of 2,100) all belong to transformer units, where they are 9.3 per cent of the 900 documents (counted from `results/xai_lime_fix/`).

## Tooling

Generative AI coding assistants (Anthropic Claude Code) were used for implementation and checking under the authors' direction, as declared in the paper; all research decisions, analyses and verification are the authors' own.

## Licence

- **Code:** MIT licence ([LICENSE](LICENSE)).
- **Results tables, figures and documentation:** Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)).
- **Third-party corpus content** is excluded and stays under its sources' terms. This includes the short excerpts in the HITL token lists and in the qualitative figure; see [DATA.md](DATA.md).

## Citation

The citation metadata is in [CITATION.cff](CITATION.cff). Until the paper is accepted, please cite it as prepared for submission to DeSE 2026:

Akın Aydın and Dhiya Al-Jumeily. An Explainable Machine Learning Framework for Detecting AI-Generated Phishing Emails: Leakage-Controlled Evaluation Across Legacy, Modern and AI-Generated Corpora. Prepared for submission to the 19th International Conference on Developments in eSystems Engineering (DeSE 2026), 2026.

Dissertation: Akın Aydın. An Explainable Machine Learning Framework for Detecting AI-Generated Phishing Emails. MSc dissertation, Liverpool John Moores University, 2026.
