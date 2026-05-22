# EDA Findings — Phishing Email Datasets

Supporting document for §4.2 of the interim report. Numbers reproducible from a cold kernel via `notebooks/01_eda.ipynb`.

## 1. Dataset summaries

**D1 — Kaggle multi-source phishing dataset** (`naserabdullahalam/phishing-email-dataset`). 82,486 emails from six aggregated corpora (CEAS_08, Enron, Ling-Spam, Nazario, Nigerian Fraud, SpamAssassin); 51.9% phishing. The unified `phishing_email.csv` is lowercased, punctuation-stripped, and stopword-filtered by the Kaggle author, so pattern counts and stopword-ablation analyses must run on raw per-source `subject + body`. Post-Enron-decontamination: 52,719 rows.

**D2 — Enron-Spam (Metsis et al. 2006, AUEB).** 33,716 emails across six subsets — `enron1–3` ham-dominant, `enron4–6` spam-dominant. Aggregate balance 1.038 : 1.

**D3 — PhishFuzzer (Tóth et al. 2025).** 23,100 emails: 3,300 seeds (300 private "Manual" + 3,000 from Kaggle/SpamAssassin) and 19,800 LLM-paraphrased variants (Gemini family per the README; precise model version — likely Gemini-2.5-Flash based on the Label Enrichment step that used it for URL/filename completion — needs paper-level verification before the final dissertation). <!-- TODO(dissertation): confirm exact Gemini version used for variant generation against the arXiv paper. --> Three-class labels (Phishing / Spam / Valid) collapsed to binary `Phishing + Spam = 1, Valid = 0` per Decision C2 (both retained in the loader). The variant grid is an exact 2 × 3 (`entity_type × length_type`), 3,300 emails per cell.

## 2. Cross-dataset comparison (fig 3)

The three-way token-length figure shows three distinct profiles:

- **Modern phishing is uniquely short** (62 tokens median); modern legitimate ≈ Enron ham (199 vs 176 tokens). A model could learn "phishing-by-brevity" as a shortcut that fails to transfer to legacy data where spam length ≈ ham length.
- **AI-generated phishing is the longest phish anywhere** (232 tokens, ~4× the human-written modern phish). LLMs elaborate; human attackers truncate.
- **The AI-generated curve is the tightest distribution** (p95 = 402 tokens vs 540 modern, 924 Enron) — LLMs produce structurally consistent output while real corpora have long tails.
- **AI Valid / Phishing / Spam medians (269 / 232 / 283) are nearly indistinguishable** — a length-based shortcut won't transfer to AI-generated data.

Pairwise Jaccard of top-500 non-stopword tokens: **legacy ↔ modern 0.351, legacy ↔ AI 0.250, modern ↔ AI 0.263** — AI has its own lexical cluster (modern-services vocabulary: *ai, amazon, app, banking, cloud, …*).

## 3. Methodological observations

Two **decontamination steps** are deliberate findings, not implementation details:

- **Kaggle ↔ Enron.** The Kaggle bundle's 29,767-row `Enron.csv` re-imports the AUEB legacy corpus. Retained on the modern side it would place identical emails on both sides of the legacy-vs-modern comparison and dilute the temporal-shift signal. The Kaggle author re-labeled 174 borderline cases (0.58%); the divergence is surfaced via `warnings.warn` in the loader.
- **PhishFuzzer ↔ Kaggle.** 3,000 of the 3,300 PhishFuzzer seeds derive from Kaggle/SpamAssassin per the README. Fig 3 therefore uses variants only (19,800); seeds are characterised in §3.5 of the notebook but excluded from the cross-dataset curve.

742 non-English PhishFuzzer rows (3.2%) are also filtered out of fig 3 so the token-count overlay stays like-for-like with the English-only D1 and D2.

## 4. Data quality

- **Enron-Spam: 9.6% exact duplicates** (3,253 rows; 2,618 spam vs 635 ham) — spam-campaign repetition. Deduplication required before train/test splits to avoid label leakage.
- Kaggle D1: 0.5% duplicates (408 rows). PhishFuzzer: 0.3% (74 rows, mostly Spam).
- One source-data typo (`'Huamn'` on a Spam seed row) silently normalised to `'Human'` in the PhishFuzzer loader.
- Kaggle's `text_combined` returns 0 URLs / 0 email addresses because punctuation was stripped — surfaced in §2.5; raw per-source text used as workaround.

## 5. Implications for §5 (Anticipated Progress)

Three concrete experimental-design commitments follow:

1. **Per-source confusion matrices** — the six Kaggle sources are linguistically heterogeneous; aggregate metrics hide corpus-specific shortcuts.
2. **Cross-corpus held-out evaluation** — pairwise Jaccard 0.25–0.35 implies ~30% novel vocabulary at cross-corpus test time; explicit cross-corpus testing is required to substantiate any claim of generalisation across the three-way landscape.
3. **Length-controlled ablation** — the 4× gap between modern phish (62 tok) and AI phish (232 tok) makes a length-only logistic regression a credible non-trivial baseline. We will compute it as a "shortcut floor" against which real models must improve.
