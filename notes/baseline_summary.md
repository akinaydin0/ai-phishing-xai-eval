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

## Trivial reference baselines (no training)

| Corpus | all_positive F1 | majority_class F1 | ROC-AUC |
|---|--:|--:|--:|
| d1_sb | 0.7024 | 0.7024 | 0.50 |
| d2 | 0.6421 | 0.0000 | 0.50 |
| d3_en | 0.7958 | 0.7958 | 0.50 |

*Constant predictors → ROC-AUC 0.50. `majority_class` predicts label 0 on d2 (ham-majority) so its positive-class F1 = 0; on d1_sb/d3_en it coincides with `all_positive`.*

## Monotonic length probe — token_count-only LR (ROC-AUC)

| Source (coef sign) | → d1_sb | → d2 | → d3_en |
|---|--:|--:|--:|
| d1_sb (-) | **0.722** | 0.536 | 0.560 |
| d2 (-) | 0.722 | **0.536** | 0.560 |
| d3_en (-) | 0.722 | 0.536 | **0.560** |

*Diagonal = in-corpus. All three source coef signs are **−** (model learns *shorter→positive*), so every source produces the same ranking and each column is source-independent — pure length **direction** is universal across the three corpora; only its strength varies.*

## 3-feature floor coefficient signs (tok / char / log-tok)

| Source | tok | char | log-tok |
|---|:--:|:--:|:--:|
| d1_sb | - | + | - |
| d2 | - | + | + |
| d3_en | + | - | - |

## Observations

1. **In-corpus levels match the literature.** TF-IDF + LR reaches F1 0.991 (d1_sb), 0.988 (d2), 0.975 (d3_en) with ROC-AUC ≥ 0.995 — the 0.95–0.99 band reported for linear models on Enron-Spam and curated phishing corpora. LR edges out Multinomial NB everywhere (e.g. d3_en 0.975 vs 0.944). No inflated or degenerate scores, so the splits and pipeline are behaving.

2. **Cross-corpus degradation is large and asymmetric.** In-corpus ≈0.98 F1 falls to 0.574–0.858 across the 6 directed pairs. Training on AI-generated d3_en transfers worst (d3_en→d1_sb 0.574, d3_en→d2 0.618), whereas real corpora detect AI phishing well (d1_sb→d3_en 0.858, d2→d3_en 0.850); d1_sb↔d2 is roughly symmetric (0.822 / 0.796). AI-generated data is a weak training source for real-world detection — direct support for the dataset-aware evaluation mandate.

3. **The 3-feature 'length floor' barely clears the trivial all-positive baseline on F1 — judge it on ROC-AUC.** In-corpus the floor's F1 is 0.694/0.636/0.777 (d1_sb/d2/d3_en) — at or just **below** the all-positive reference 0.702/0.642/0.796, because both are dominated by the positive-class prior, so F1 alone makes the floor look competent when it has learned almost nothing. Its real (weak) signal is in ROC-AUC 0.729/0.635/0.745 — only modestly above the 0.5 chance line (d2 nearly at chance), vs TF-IDF LR's ≥0.995. Cross-corpus the floor's AUC drops below 0.5 (d2→d1_sb 0.341, d2→d3_en 0.336) — what signal it has does not transfer. **Takeaway:** any model whose F1 merely matches ~0.70/0.64/0.80 has learned nothing beyond the class prior; models must be judged on AUC and cross-corpus behaviour, not in-corpus F1 alone.

4. **subject+body vs text_combined: a negligible in-corpus gap → subject+body for modelling.** In-corpus D1 LR F1 is 0.9906 (subject_body) vs 0.9927 (text_combined) — a 0.21 pp difference (text_combined marginally higher; its stop-word/punctuation stripping mildly denoises the bag-of-words). The representation choice therefore costs ~nothing in accuracy, and subject+body is preferred for modelling because it (a) is representation-consistent with D2/D3 (both natural text), enabling clean cross-corpus experiments, and (b) preserves stop-words/punctuation so Pipeline A's stop-word-removal ablation is possible (impossible on the pre-stripped text_combined). **Upgraded from working hypothesis to evidence-backed decision.** *Caveat: subject+body (51,839) and text_combined (52,704) are different post-dedup universes (865-row delta), so the 0.21 pp gap is indicative, not a paired test on identical emails.*

5. **Anomaly — ranking transfers but the threshold does not.** Several cross-corpus directions keep high ROC-AUC while F1 drops sharply (d2→d1_sb AUC 0.947 at F1 0.796; d3_en→d1_sb AUC 0.922 at F1 0.574). The model still separates the classes, but the fixed 0.5 threshold is miscalibrated for the target's class prior — a calibration artifact, not a feature failure. Cross-corpus deployment needs target-domain threshold recalibration, and AUC must be reported alongside F1.

6. **Transfer direction, made exact by a monotonic (token_count-only) probe — with a correction.** A single-feature LR is monotonic in length, so its cross-corpus ROC-AUC exactly encodes whether the source's learned length direction holds in the target. The fitted coef sign is **negative in all three source corpora** — the model learns **shorter→positive**, not longer→positive as the directive anticipated (correction: phishing/spam skews shorter in every corpus here — short modern-phishing and Enron-spam medians). Because all three signs agree, each source produces the same ranking, so the cross-corpus probe AUC is source-independent: →d1_sb 0.722, →d2 0.536, →d3_en 0.560 (all > 0.5). Pure length **direction is universal** across the three corpora; only its strength varies (strong in d1_sb, near-chance in d2/d3_en).

### Why the 3-feature floor inverts (d2→d1_sb AUC 0.341)

The floor fits mixed-sign coefficients on the collinear [token, char, log-token] features — d1_sb -/+/-, d2 -/+/+, d3_en +/-/-. For d2 the token(−)/char(+) contrast on near-collinear features cancels the pure-length component and leaves a **chars-per-token / character-density style signal**, not length. That style signal is corpus-specific and reverses on d1_sb, giving d2→d1_sb floor AUC 0.341 (inverted). The monotonic probe confirms the mechanism: d2→d1_sb on token_count alone is 0.722 (**not** inverted) — so the inversion is the style component the extra features introduce, exactly the hypothesised mechanism. In-corpus that same style component *helps* (the floor's AUC 0.745 on d3_en far exceeds token-only 0.560), which is precisely why it fails to transfer.

## Confusion matrices

Per in-corpus LR run, under `figures/baselines/` (PNG + PDF): `cm_d1_decontaminated_sb`, `cm_d1_decontaminated` (text_combined ablation), `cm_d2`, `cm_d3_en`.