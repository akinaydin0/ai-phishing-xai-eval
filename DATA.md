# Data statement

This repository does not redistribute any corpus text. The work uses three third-party corpora, which have to be obtained from their sources and placed at the paths listed at the end of this file. The frozen splits, and every per-document output in the repository, identify documents by the SHA-1 hash of the cleaned body (40 hexadecimal characters).

## Sources and terms

The licence statements follow Table C.1 of the dissertation (EU FAIR data criteria mapping).

### D1: Kaggle multi-source phishing dataset

- **Source:** naserabdullahalam/phishing-email-dataset, https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset
- **Terms:** licensed CC BY-SA 4.0, as stated on the dataset page (accessed 26 August 2026).
  - The dataset aggregates six sub-corpora: CEAS_08, Enron, Ling-Spam, Nazario, Nigerian Fraud and SpamAssassin.
  - The sub-corpora's own terms are not recorded.
- **Size:** 82,486 emails in `phishing_email.csv`.
- **Labels:** 1 = phishing or spam, 0 = legitimate.

### D2: Enron-Spam

- **Source:** Metsis, Androutsopoulos and Paliouras (2006), distributed by the Athens University of Economics and Business (AUEB): http://www2.aueb.gr/users/ion/data/enron-spam/ (the six preprocessed archives `enron1` to `enron6`).
- **Terms:** no licence is stated by the distributor. Per-subset summaries (`Summary.txt`) document the provenance.
- **Size:** 33,716 emails (16,545 ham and 17,171 spam).
- **Labels:** 1 = spam, 0 = ham.

### D3: PhishFuzzer

- **Source:** https://github.com/DataPhish/PhishFuzzer at commit `1e21dd4`. The accompanying paper is Tóth, Bisztray and Gruschka, arXiv:2511.21448.
- **Terms:**
  - The repository has no licence file, although its paper describes the framework and dataset as fully open source.
  - The CC BY 4.0 licence of the arXiv paper does not cover the data.
  - 300 of the 3,300 seed emails are a de-identified private collection.
- **Size:** 23,100 emails, made up of 3,300 seeds and 19,800 LLM-generated variants (six per seed). Only the variants are used.
- **Labels:** Phishing and Spam = 1, Valid = 0.

## What the repository contains

Identifiers, labels and model outputs:

- `data/splits/*.json`: for each corpus, the SHA-1 hashes of the cleaned bodies in the training, validation and test splits, with counts and class shares.
- Hashes, labels, predictions, scores and metrics in `results/`.
- The attribution vectors in `results/xai/` and `results/xai_lime_fix/`. These are numbers only, with no tokens.

Short excerpts:

- **HITL token lists.** These are the rated top-ten token lists of the human and LLM-judge validation files (`results/hitl/key.json`, `blind_sheets.csv`, `human_ratings.csv`, `judge_ratings.csv` and `hitl_token_level.csv`), at most ten words per set.
  - For the void-LIME FIRST10 sets, a list is the first ten words of an email.
  - Four very short emails, of three to nine words, appear in full.
- **The qualitative figure** (`figures/xai/xai_qualitative_examples.png` and `.pdf`). It shows the first words of six emails, at most 52 words each, with the top-ten SHAP tokens boxed.
  - The D1 excerpts are reproduced as word tokens from the Kaggle dataset naserabdullahalam/phishing-email-dataset (https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset) under CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/).
  - The D2 and D3 excerpts are short quotations that stay under their sources' terms.

A sampled ten-word shingle check of every committed file against the three cleaned corpora (2026-10-12), which reliably detects verbatim passages of about 30 words or more, found no other such passage.

## Cleaning, deduplication and decontamination

The rules are implemented in `src/data_loader.py`, `src/cleaning.py` and `src/splits.py`. `notes/cleaning_report.md` records the row flow.

1. **Decontamination.** D1 contains the Enron.csv sub-collection (29,767 rows), which overlaps the D2 Enron material, so it is removed and 52,719 rows remain. The primary D1 view (subject and body) is read from the five non-Enron source files.
2. **Cleaning, per corpus, in this order:**
   - HTML stripping;
   - normalisation: encoding repair with ftfy, removal of residual email header lines from a fixed allowlist, and whitespace collapse;
   - removal of bodies with fewer than five tokens;
   - exact deduplication within the corpus by the SHA-1 hash of the normalised body, keeping the first occurrence.

   There is no deduplication across corpora.
3. **D3.** Only the 19,800 variants are used, and modelling uses the English subset; 635 non-English records are excluded after cleaning.
4. **Pinned counts.** The final row and class counts are pinned in `src/cleaning.py` (`CLEANING_EXPECTED`) and, for the English D3 subset, in `src/splits.py` (`DERIVED_FINGERPRINT`). Any drift stops the pipeline.
5. **Splits.** Each corpus is split 70/15/15, stratified, with seed 42 and keyed by the SHA-1 body hash. `data/splits/d3_en_grouped.json` is the family-disjoint sensitivity split, produced by `python -m src.experiments.group_sens`.

| Corpus id | Content | Rows in | Rows after cleaning | Class 0 / class 1 | Train / validation / test |
|---|---|--:|--:|---|---|
| `d1_decontaminated_sb` (primary D1) | D1 subject and body, Enron excluded | 52,719 | 51,839 | 23,777 / 28,062 | 36,287 / 7,776 / 7,776 |
| `d1_decontaminated` (ablation) | D1 `text_combined` field, Enron excluded | 52,719 | 52,704 | 23,798 / 28,906 | 36,892 / 7,906 / 7,906 |
| `d2` | D2 | 33,716 | 29,224 | 15,406 / 13,818 | 20,456 / 4,384 / 4,384 |
| `d3_en` | D3 variants, English | 19,800 | 19,164 (19,799 before the language filter) | 6,504 / 12,660 | 13,414 / 2,875 / 2,875 |

The full D1 file (82,486 rows, 81,970 after cleaning) is cleaned and pinned but not used for modelling.

## Expected raw-data paths

```
data/raw/phishing_email_dataset/
    phishing_email.csv, Enron.csv, CEAS_08.csv, Ling.csv, Nazario.csv,
    Nigerian_Fraud.csv, SpamAssasin.csv   (file names as in the Kaggle archive)
data/raw/enron-spam/
    enron1/ to enron6/, each with ham/*.txt and spam/*.txt
data/raw/phishfuzzer/
    a clone of DataPhish/PhishFuzzer at commit 1e21dd4, containing
    PhishFuzzer_emails_entity_rephrased_v1.json (the variants, used) and
    PhishFuzzer_emails_original_seed_v1.json (the seeds)
data/embeddings/glove.6B.300d.txt   (dense baselines only)
```

These folders are ignored by git. `src.cleaning` writes the cleaned corpora to `data/cache/`, and the D2 loader caches its parsed files in `data/processed/`.
