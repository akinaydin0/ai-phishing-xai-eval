# Classical Baseline Summary

TF-IDF (1–2 gram, min_df=2, sublinear, ≤100k feats) + Logistic Regression (`class_weight='balanced'`) and Multinomial NB, on the frozen 70/15/15 splits (seed 42). Positive class = label 1 (phishing/spam). Vectorizers/scalers fit on the source **train** split only; models evaluated on the **test** split; the val split is untouched. Full log: `results/results.csv`.

## Results (model × text_field × train→test × F1)

| Run | Model | Features | text_field | train → test | F1 | ROC-AUC |
|---|---|---|---|---|--:|--:|
| A | LogReg | tfidf | subject_body | d1_sb → d1_sb | 0.9906 | 0.9995 |
| A | NB | tfidf | subject_body | d1_sb → d1_sb | 0.9779 | 0.9985 |
| A | LogReg | tfidf | subject_body | d2 → d2 | 0.9876 | 0.9992 |
| A | NB | tfidf | subject_body | d2 → d2 | 0.9860 | 0.9989 |
| A | LogReg | tfidf | subject_body | d3_en → d3_en | 0.9751 | 0.9949 |
| A | NB | tfidf | subject_body | d3_en → d3_en | 0.9442 | 0.9720 |
| B | LogReg | tfidf | text_combined | d1_tc → d1_tc | 0.9927 | 0.9992 |
| C | LogReg | length3 | subject_body | d1_sb → d1_sb | 0.6937 | 0.7285 |
| C | LogReg | length3 | subject_body | d2 → d2 | 0.6358 | 0.6353 |
| C | LogReg | length3 | subject_body | d3_en → d3_en | 0.7769 | 0.7452 |
| D | LogReg | tfidf | subject_body | d1_sb → d2 | 0.8224 | 0.9152 |
| D | LogReg | length3 | subject_body | d1_sb → d2 | 0.4932 | 0.5380 |
| D | LogReg | tfidf | subject_body | d1_sb → d3_en | 0.8575 | 0.9263 |
| D | LogReg | length3 | subject_body | d1_sb → d3_en | 0.1696 | 0.5506 |
| D | LogReg | tfidf | subject_body | d2 → d1_sb | 0.7960 | 0.9469 |
| D | LogReg | length3 | subject_body | d2 → d1_sb | 0.6956 | 0.3411 |
| D | LogReg | tfidf | subject_body | d2 → d3_en | 0.8501 | 0.8559 |
| D | LogReg | length3 | subject_body | d2 → d3_en | 0.7960 | 0.3356 |
| D | LogReg | tfidf | subject_body | d3_en → d1_sb | 0.5735 | 0.9225 |
| D | LogReg | length3 | subject_body | d3_en → d1_sb | 0.6929 | 0.5071 |
| D | LogReg | tfidf | subject_body | d3_en → d2 | 0.6177 | 0.7967 |
| D | LogReg | length3 | subject_body | d3_en → d2 | 0.6304 | 0.4135 |

## Observations

1. **In-corpus levels match the literature.** TF-IDF + LR reaches F1 0.991 (d1_sb), 0.988 (d2), 0.975 (d3_en) with ROC-AUC ≥ 0.995 — the 0.95–0.99 band reported for linear models on Enron-Spam and curated phishing corpora. LR edges out Multinomial NB everywhere (e.g. d3_en 0.975 vs 0.944). No inflated or degenerate scores, so the splits and pipeline are behaving.

2. **Cross-corpus degradation is large and asymmetric.** In-corpus ≈0.98 F1 falls to 0.574–0.858 across the 6 directed pairs. Training on AI-generated d3_en transfers worst (d3_en→d1_sb 0.574, d3_en→d2 0.618), whereas real corpora detect AI phishing well (d1_sb→d3_en 0.858, d2→d3_en 0.850); d1_sb↔d2 is roughly symmetric (0.822 / 0.796). AI-generated data is a weak training source for real-world detection — direct support for the dataset-aware evaluation mandate.

3. **The length-only shortcut floor is real but beatable in-corpus, and collapses out-of-corpus.** In-corpus the floor reaches F1 0.694/0.636/0.777 (d1_sb/d2/d3_en); TF-IDF beats it by 0.30/0.35/0.20 F1, so models learn far more than length. d3_en has the highest floor (0.777) — structurally uniform LLM output carries the most length signal, a mild shortcut flag for AI data. Cross-corpus, the floor **inverts**: several directions post ROC-AUC < 0.5 (d2→d3_en 0.336, d2→d1_sb 0.341), i.e. the length–label relationship reverses between corpora. Length is a non-transferable, corpus-specific shortcut.

4. **subject+body vs text_combined: a negligible in-corpus gap → subject+body for modelling.** In-corpus D1 LR F1 is 0.9906 (subject_body) vs 0.9927 (text_combined) — a 0.21 pp difference (text_combined marginally higher; its stop-word/punctuation stripping mildly denoises the bag-of-words). The representation choice therefore costs ~nothing in accuracy, and subject+body is preferred for modelling because it (a) is representation-consistent with D2/D3 (both natural text), enabling clean cross-corpus experiments, and (b) preserves stop-words/punctuation so Pipeline A's stop-word-removal ablation is possible (impossible on the pre-stripped text_combined). **Upgraded from working hypothesis to evidence-backed decision.**

5. **Anomaly — ranking transfers but the threshold does not.** Several cross-corpus directions keep high ROC-AUC while F1 drops sharply (d2→d1_sb AUC 0.947 at F1 0.796; d3_en→d1_sb AUC 0.922 at F1 0.574). The model still separates the classes, but the fixed 0.5 threshold is miscalibrated for the target's class prior — a calibration artifact, not a feature failure. Cross-corpus deployment needs target-domain threshold recalibration, and AUC must be reported alongside F1.

## Confusion matrices

Per in-corpus LR run, under `figures/baselines/` (PNG + PDF): `cm_d1_decontaminated_sb`, `cm_d1_decontaminated` (text_combined ablation), `cm_d2`, `cm_d3_en`.