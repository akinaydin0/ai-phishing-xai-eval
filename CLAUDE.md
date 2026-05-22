# MSc Thesis Project — Context

## Project
**Title:** An Explainable Machine Learning Framework for Detecting AI-Generated Phishing Emails
**Author:** Akın Aydın, MSc Machine Learning, Liverpool John Moores University
**Deadline:** Interim Report 22 May 2026 (very soon); Final dissertation late August 2026
**Hardware:** Local workstation with NVIDIA RTX 3090 (24 GB VRAM)

## Current Task
**Phase: Exploratory Data Analysis (EDA)**

Build a clean Jupyter notebook (`notebooks/01_eda.ipynb`) and supporting modules under `src/` to:
1. Download two phishing email datasets (instructions in "Datasets" below)
2. Load, clean, and characterise them
3. Produce three figures (PNG, 200 dpi) saved under `figures/` for inclusion in the interim report
4. Output a summary statistics table (`figures/table_dataset_summary.csv`)
5. Provide a written EDA findings summary (200-300 words) as `notes/eda_findings.md`

The interim report submission this week needs *real numbers from this EDA*, not placeholders.

## Repository Structure (already exists, do not recreate)
msc-thesis/
├── CLAUDE.md                 (this file)
├── proposal.docx             (full project proposal)
├── interim_slides.pptx       (university template / structure requirements)
├── data/
│   ├── raw/
│   │   ├── phishing_email_dataset/   (Dataset 1 lands here)
│   │   └── enron-spam/               (Dataset 2 lands here)
│   └── processed/
├── notebooks/
├── src/
├── figures/
├── notes/
└── report/

## Datasets

### Dataset 1: Phishing Email Dataset (multi-source curated)
- **Source:** Kaggle — naserabdullahalam/phishing-email-dataset
- **URL:** https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset
- **Target path:** `data/raw/phishing_email_dataset/`
- **Note:** Kaggle CLI requires kaggle.json credentials. If unavailable, surface a `curl`/`wget` command or instruct me (Akın) to drop the ZIP into the folder manually.

### Dataset 2: Enron-Spam (Metsis et al., 2006)
- **Source:** AUEB (Androutsopoulos lab)
- **URL:** http://www2.aueb.gr/users/ion/data/enron-spam/
- **Files:** 6 `.tar.gz` archives (enron1.tar.gz through enron6.tar.gz) plus pre-processed versions. Use the pre-processed ham/spam folder structure.
- **Target path:** `data/raw/enron-spam/`
- **Note:** Direct `wget`/`curl` from AUEB should work — try it.

## Environment
- Python 3.11 expected. A venv may already exist at `./venv/`; check before creating.
- Required packages: pandas, numpy, matplotlib, seaborn, wordcloud, scikit-learn, jupyter, nltk
- After install: `pip freeze > requirements.txt`

## Methodology Context (informs analysis choices)

The literature review (already conducted) identified several methodological revisions to the original proposal. These affect what EDA should surface:

1. **Pipeline A extension:** Classical preprocessing will add spelling correction, word segmentation, URL/email/phone placeholder substitution (Kulal et al. 2025). EDA should report counts of: tokens with potential misspellings, URLs, email addresses, phone numbers per class.

2. **Stopword removal as ablation variable:** Don't assume stopword removal is harmless. Report top tokens per class both with and without stopword filtering.

3. **Dataset mixing risk (CRITICAL):** The original plan to combine Enron (2000s) with modern phishing is high-risk — a model can learn "old vs new language" rather than phishing intent. The EDA must produce a **cross-dataset comparison figure** (text length distribution overlaid) that visually demonstrates this temporal/domain shift. This figure will support the §4 (Difficulties) section of the interim report.

4. **Class imbalance:** Report imbalance ratios for both datasets. Plan mentions class-ratio experiments at 1:9, 1:19, 1:49.

## Reference Errata (Important — do not cite these)

During literature review, 3 references in `proposal.docx` were found to be fabricated (likely LLM hallucination). DO NOT cite them in any output you generate:

- Aljofey et al. (2022) "IEEE Access 10, 15321-15333" — does not exist. Real paper is Aljofey et al. (2022) in *Scientific Reports* 12, 8842, DOI 10.1038/s41598-022-10841-5.
- Almeida et al. (2020) "Stopword removal, ESWA 153, 113434" — does not exist.
- Alsharnouby et al. (2020) "IJHCS 141, 102432" — does not exist. Real paper is Alsharnouby et al. (2015) in IJHCS 82.

## Output Expectations

1. **Notebook:** `notebooks/01_eda.ipynb` — clean, sectioned with markdown headers, all cells re-runnable from a fresh kernel.
2. **Three figures (PNG, 200 dpi) under `figures/`:**
   - `fig_class_dist_phishing.png` — class distribution of Dataset 1
   - `fig_text_length_phishing.png` — text length by class for Dataset 1
   - `fig_cross_dataset_length.png` — overlaid length distributions of both datasets (CRITICAL — supports §4 of interim report)
3. **Summary CSV:** `figures/table_dataset_summary.csv` with columns: Dataset, Source, Total emails, Positive class count, Imbalance ratio.
4. **Written findings:** `notes/eda_findings.md` — 200-300 words covering: dataset sizes, class balance, length characteristics, evidence of temporal/domain shift, any data quality issues noticed.
5. **Reproducibility:** add `src/data_loader.py` with two functions: `load_phishing_dataset()` and `load_enron_spam()`. The notebook imports from these rather than embedding loading logic.

## Working Style

- Plan before executing. Show me the plan, get confirmation, then execute.
- If a download fails, surface the exact `curl`/`wget` URL so I can run it manually.
- Don't fabricate dataset statistics — if a dataset isn't downloaded yet, say so and stop.
- Use the existing venv if present (`source venv/bin/activate`).
- Commit at logical milestones with descriptive messages.