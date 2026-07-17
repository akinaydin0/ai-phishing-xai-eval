# XAI Phase — Attribution Methods + Quantitative Evaluation

SHAP / LIME / Integrated Gradients on the family champions (tfidf-linear, LightGBM, transformer) per corpus, evaluated for **faithfulness** (token-removal comprehensiveness / sufficiency / AOPC), **stability** (perturbation Jaccard + Spearman, LIME seed-agreement) and **plausibility** (cue-hit@10). Explanation docs are drawn from each corpus's TEST split only (n=200 in-corpus, 100/class, all misclassified up to 30). Every faithfulness and plausibility number is reported beside a **random-attribution baseline** through the identical pipeline. **Plausibility is a heuristic cue-lexicon proxy, not human annotation.** Pre-registration + fixed lexicon: `notes/decisions.md`, `src/xai/cue_lexicon.py`. Full log: `results/xai_metrics.csv`.

## (a) Faithfulness — comprehensiveness / sufficiency / AOPC (test set)

Higher comp@k = removing the top-k attributed tokens drops the predicted-class probability more (more faithful). Lower suff@k = the top-k alone reproduce the prediction. `rnd` = random-attribution baseline.

| Corpus | Family | Method | AOPC | rnd AOPC | comp@10 | rnd | suff@10 | rnd |
|---|---|---|--:|--:|--:|--:|--:|--:|
| d1_sb | tfidf | shap | 0.199 | 0.012 | 0.213 | 0.010 | -0.029 | 0.169 |
| d1_sb | tfidf | lime | 0.016 | 0.012 | 0.017 | 0.010 | 0.097 | 0.169 |
| d1_sb | ensemble | shap | 0.281 | 0.000 | 0.306 | 0.002 | -0.009 | 0.354 |
| d1_sb | ensemble | lime | -0.002 | 0.000 | -0.002 | 0.002 | 0.225 | 0.354 |
| d1_sb | transformer | shap | 0.093 | 0.012 | 0.093 | 0.015 | 0.008 | 0.134 |
| d1_sb | transformer | lime | 0.030 | 0.012 | 0.035 | 0.015 | 0.069 | 0.134 |
| d1_sb | transformer | ig | 0.080 | 0.012 | 0.087 | 0.015 | 0.020 | 0.134 |
| d2 | tfidf | shap | 0.191 | 0.024 | 0.189 | 0.020 | -0.045 | 0.146 |
| d2 | tfidf | lime | 0.026 | 0.024 | 0.020 | 0.020 | 0.115 | 0.146 |
| d2 | ensemble | shap | 0.330 | 0.040 | 0.353 | 0.038 | -0.027 | 0.205 |
| d2 | ensemble | lime | 0.041 | 0.040 | 0.044 | 0.038 | 0.169 | 0.205 |
| d2 | transformer | shap | 0.137 | 0.024 | 0.128 | 0.032 | -0.003 | 0.173 |
| d2 | transformer | lime | 0.043 | 0.024 | 0.039 | 0.032 | 0.059 | 0.173 |
| d2 | transformer | ig | 0.098 | 0.024 | 0.099 | 0.032 | 0.037 | 0.173 |
| d3_en | tfidf | shap | 0.094 | 0.004 | 0.093 | 0.004 | -0.153 | 0.174 |
| d3_en | tfidf | lime | 0.006 | 0.004 | 0.005 | 0.004 | 0.128 | 0.174 |
| d3_en | ensemble | shap | 0.232 | 0.002 | 0.226 | 0.002 | -0.022 | 0.281 |
| d3_en | ensemble | lime | 0.002 | 0.002 | 0.000 | 0.002 | 0.226 | 0.281 |
| d3_en | transformer | shap | 0.090 | 0.031 | 0.096 | 0.037 | 0.299 | 0.408 |
| d3_en | transformer | lime | 0.037 | 0.031 | 0.047 | 0.037 | 0.152 | 0.408 |
| d3_en | transformer | ig | 0.129 | 0.031 | 0.130 | 0.037 | 0.193 | 0.408 |

## (b) Stability — perturbation robustness + LIME seed-agreement

| Corpus | Family | Method | top-10 Jaccard | Spearman | LIME seed-agree |
|---|---|---|--:|--:|--:|
| d1_sb | tfidf | shap | 0.924 | 0.993 | nan |
| d1_sb | tfidf | lime | 0.992 | nan | 1.000 |
| d1_sb | ensemble | shap | 0.922 | 0.958 | nan |
| d1_sb | ensemble | lime | 0.994 | nan | 1.000 |
| d1_sb | transformer | shap | 0.584 | 0.762 | nan |
| d1_sb | transformer | lime | 0.993 | nan | 1.000 |
| d1_sb | transformer | ig | 0.827 | 0.930 | nan |
| d2 | tfidf | shap | 0.935 | 0.993 | nan |
| d2 | tfidf | lime | 0.998 | nan | 1.000 |
| d2 | ensemble | shap | 0.951 | 0.972 | nan |
| d2 | ensemble | lime | 0.996 | nan | 1.000 |
| d2 | transformer | shap | 0.547 | 0.763 | nan |
| d2 | transformer | lime | 0.998 | nan | 1.000 |
| d2 | transformer | ig | 0.793 | 0.930 | nan |
| d3_en | tfidf | shap | 0.868 | 0.989 | nan |
| d3_en | tfidf | lime | 0.997 | nan | 1.000 |
| d3_en | ensemble | shap | 0.881 | 0.942 | nan |
| d3_en | ensemble | lime | 0.996 | nan | 1.000 |
| d3_en | transformer | shap | 0.404 | 0.624 | nan |
| d3_en | transformer | lime | 0.996 | nan | 1.000 |
| d3_en | transformer | ig | 0.489 | 0.619 | nan |

## (c) Plausibility — cue-hit@10 on phishing-predicted docs vs random (heuristic proxy)

| Corpus | Family | Method | plausibility@10 | random | lift |
|---|---|---|--:|--:|--:|
| d1_sb | tfidf | shap | 0.234 | 0.166 | +0.068 |
| d1_sb | tfidf | lime | 0.100 | 0.166 | -0.066 |
| d1_sb | ensemble | shap | 0.192 | 0.175 | +0.017 |
| d1_sb | ensemble | lime | 0.077 | 0.175 | -0.098 |
| d1_sb | transformer | shap | 0.133 | 0.159 | -0.027 |
| d1_sb | transformer | lime | 0.085 | 0.159 | -0.074 |
| d1_sb | transformer | ig | 0.131 | 0.159 | -0.029 |
| d2 | tfidf | shap | 0.131 | 0.073 | +0.058 |
| d2 | tfidf | lime | 0.065 | 0.073 | -0.008 |
| d2 | ensemble | shap | 0.186 | 0.064 | +0.122 |
| d2 | ensemble | lime | 0.070 | 0.064 | +0.006 |
| d2 | transformer | shap | 0.116 | 0.061 | +0.054 |
| d2 | transformer | lime | 0.071 | 0.061 | +0.010 |
| d2 | transformer | ig | 0.094 | 0.061 | +0.033 |
| d3_en | tfidf | shap | 0.290 | 0.127 | +0.163 |
| d3_en | tfidf | lime | 0.236 | 0.127 | +0.109 |
| d3_en | ensemble | shap | 0.304 | 0.139 | +0.165 |
| d3_en | ensemble | lime | 0.224 | 0.139 | +0.085 |
| d3_en | transformer | shap | 0.236 | 0.142 | +0.094 |
| d3_en | transformer | lime | 0.253 | 0.142 | +0.111 |
| d3_en | transformer | ig | 0.218 | 0.142 | +0.076 |

## (d) Three-metric comparison by family (mean over in-corpus corpora)

| Family | best method (by AOPC) | AOPC | stability Jaccard | plausibility@10 | mean XAI runtime/unit (s) |
|---|---|--:|--:|--:|--:|
| tfidf | shap | 0.161 | 0.909 | 0.218 | 6 |
| ensemble | shap | 0.281 | 0.918 | 0.227 | 87 |
| transformer | shap | 0.107 | 0.512 | 0.162 | 686 |

## (e) Cross-corpus explanation quality — does it degrade with transfer?

Transformer champions explained on a DIFFERENT corpus's test split. Compared to the same champion's in-corpus explanation quality.

| Transfer | Method | AOPC | comp@10 | stability Jaccard | plausibility@10 |
|---|---|--:|--:|--:|--:|
| d2 → d3_en | ig | 0.076 | 0.071 | 0.637 | 0.230 |
| d2 → d3_en | lime | 0.055 | 0.066 | 0.998 | 0.210 |
| d2 → d3_en | shap | 0.212 | 0.231 | 0.440 | 0.253 |
| d3_en → d1_sb | ig | 0.145 | 0.140 | 0.493 | 0.123 |
| d3_en → d1_sb | lime | 0.068 | 0.065 | 0.990 | 0.091 |
| d3_en → d1_sb | shap | 0.184 | 0.180 | 0.456 | 0.159 |
| d3_en → d2 | ig | 0.165 | 0.175 | 0.491 | 0.041 |
| d3_en → d2 | lime | 0.075 | 0.077 | 0.995 | 0.041 |
| d3_en → d2 | shap | 0.223 | 0.248 | 0.525 | 0.059 |

## RQ3 — which family gives the best accuracy–explanation–cost balance?

Combining the modelling results (Directive 6: tuned linear TF-IDF has the top in-corpus accuracy, AUC ≥ 0.999; transformers match but do not beat it, at ~1000× the training cost) with these XAI results:

- **Faithfulness** (AOPC): ensemble·SHAP 0.281 > tfidf·SHAP 0.161 > transformer·SHAP 0.107 ≈ transformer·IG 0.102 ≫ every LIME (0.016/0.014/0.037, at the random-baseline floor ~0.017).
- **Plausibility** (cue-hit@10): SHAP beats the random baseline at the family-mean level (tfidf 0.218 vs 0.122; ensemble 0.227 vs 0.126). LIME is at or below random on d1_sb and d2, but on the cue-dense d3_en corpus LIME *exceeds* random (+0.085…+0.111) — a case of **plausibility without faithfulness**: LIME surfaces cue words because AI-generated text is saturated with them, yet its d3_en AOPC stays at the random floor, so removing those words does not move the model.
- **Cost** to explain one 200-doc unit: tfidf·SHAP **6 s**, ensemble·SHAP 87 s, transformer·SHAP 686 s, transformer·LIME 2408 s — a 100–400× spread.

**Answer: linear TF-IDF + SHAP.** It pairs the best (tied) in-corpus accuracy with the second-most-faithful and joint-most-plausible explanations, at the **lowest cost of any combination** (6 s/unit). Ensemble (LightGBM) + SHAP is the single most *faithful* explainer but its accuracy is a shade lower and it costs ~15× more; transformers are the hardest and most expensive to explain faithfully and are justified only where cross-corpus robustness forces a contextual model (Directive 6), with **IG** as their explainer of choice (as faithful as SHAP there, ~4× cheaper, and more stable).

## Observations

1. **SHAP is the only method that is demonstrably faithful; LIME is barely above random on every family.** Removing the top-10 SHAP tokens drops the predicted-class probability by AOPC 0.281 (ensemble), 0.161 (tfidf) and 0.107 (transformer), whereas LIME's AOPC (0.016/0.014/0.037) is indistinguishable from the random-attribution baseline (~0.017). The deletion curves make this visual: SHAP climbs, LIME hugs the random line.

2. **There is a faithfulness–stability trade-off, and no method wins both.** LIME is the most *stable* (top-10 Jaccard ~1.00 across families, LIME seed-agreement 1.00) precisely because its local bag-of-words fit is insensitive to which tokens matter; SHAP and IG are faithful but less stable (transformer·SHAP Jaccard 0.51, IG 0.70). Stability without faithfulness is a stable *wrong* answer — the two must be reported together.

3. **Transformers are the hardest models to explain faithfully.** Their SHAP/IG AOPC (~0.11) is half the linear champions' and a third of LightGBM's, and their SHAP top-10 Jaccard (0.51) is the lowest anywhere. Contextual models spread the decision across many tokens and their softmax saturates on confident cases, so no small token set is individually decisive — the attribution is real but diffuse. Among transformer explainers **IG ≈ SHAP on faithfulness at ~4× lower cost and higher stability**, so IG is the transformer method of choice.

4. **Plausibility tracks faithfulness — with two honest caveats.** At the **family-mean** level SHAP's top tokens are phishing cues above chance: SHAP plausibility@10 exceeds random by +0.096 (tfidf), +0.101 (ensemble) and +0.041 (transformer, 0.162 vs 0.121) — the one per-corpus exception being d1_sb transformer·SHAP at 0.133, -0.027 below its 0.159 random base rate. And LIME is at or below random on d1_sb and d2 but **exceeds it on the cue-dense d3_en corpus (+0.085…+0.111; transformer·LIME 0.253 even tops transformer·SHAP 0.236)** — yet LIME's d3_en AOPC (0.037) stays at the random floor. This is **plausibility without faithfulness**: LIME surfaces the cue words AI-generated text is saturated with, but removing them does not move the model — exactly the failure mode the three-metric design exists to expose. *(Plausibility is a heuristic lexicon proxy, not human annotation.)* The qualitative panels show the faithful case (advance-fee terms 'International/Assets/beneficiaries/facilitate' boxed on a d3_en true positive).

5. **Cross-corpus: apparent faithfulness *rises* on transfer (a low-confidence artifact), while cue-plausibility falls on the hard legacy axis — in two layers.** Transferred transformer AOPC (0.207 SHAP) is *higher* than in-corpus (0.107) — not because explanations improve, but because out-of-distribution predictions are less saturated, so token removal moves p more. The telling metric is plausibility on the hardest axis d3_en→d2 (AI→legacy-Enron), and it must be read two ways: (i) the **absolute** SHAP cue-hit falls to 0.059 largely because legacy Enron's **random cue base rate is only 0.026** (Enron carries few lexical phishing cues); (ii) SHAP's **lift over random** shrinks from +0.094 in-corpus to +0.033 on transfer. Both layers point the same way — the legacy↔AI axis is where transfer genuinely breaks down, mirroring the Directive-6 accuracy finding.
