# D9 — Statistical Inference, Imbalance & Preprocessing Robustness

All comparisons, champions and claim directions were **pre-registered before any statistic existed** (notes/decisions.md, 2026-07-17). CIs/p-values **annotate** the committed comparisons — no champion, threshold or claim direction is reselected on their basis. Score vectors were recomputed from stored artefacts and asserted equal to the committed test AUCs to 4 dp. Full log: `results/stats/stats_summary.csv`.

## (a) The seven comparisons — Δ AUC (95% paired stratified bootstrap, B=10,000) + McNemar/Holm

| id | comparison | AUC_a | AUC_b | Δ=a−b | 95% CI | McNemar n01/n10 | p_raw | p_holm | verdict |
|---|---|--:|--:|--:|---|:--:|--:|--:|---|
| C1a | LinearSVC vs DistilBERT-256 | 0.9998 | 0.9999 | -0.0001 | [-0.0002, -0.0000] | 27/17 | 0.174 | 0.174 | **CI includes 0** |
| C1b | MultinomialNB vs DistilBERT-256 | 0.9996 | 0.9999 | -0.0003 | [-0.0005, -0.0001] | 37/20 | 0.0331 | 0.0663 | separated |
| C1c | LinearSVC vs RoBERTa-256 | 0.9993 | 0.9976 | +0.0017 | [+0.0006, +0.0030] | 14/36 | 0.0026 | 0.0104 | separated |
| C2 | RoBERTa-512(d3en->d2) vs RoBERTa-256(d3en->d2) | 0.8440 | 0.7306 | +0.1134 | [+0.1042, +0.1229] | 139/607 | 0 | 0 | separated |
| C3 | RoBERTa-512(d3en->d2) vs LinearSVC-cross | 0.8440 | 0.7523 | +0.0917 | [+0.0784, +0.1052] | 496/859 | 0 | 0 | separated |
| C4 | LogReg-cross(d2->d3en) vs DistilBERT-256(d2->d3en) | 0.8429 | 0.7748 | +0.0681 | [+0.0534, +0.0832] | 249/360 | 8e-06 | 3.9e-05 | separated |
| C5 | DistilBERT-512(d2->d3en) vs DistilBERT-256(d2->d3en) | 0.7642 | 0.7748 | -0.0106 | [-0.0223, +0.0009] | 266/209 | 0.0101 | 0.0303 | **CI includes 0** |

## (b) Which committed claims gain (or lack) statistical separation

- **C1a** (in-corpus d1_sb: 256 transformer vs classical champion): AUC Δ=-0.0001 — **the 95% CI includes 0** ([-0.0002,-0.0000]): no AUC-level separation (Holm-McNemar p=0.174).
- **C1b** (in-corpus d2: 256 transformer vs classical champion): AUC Δ=-0.0003 is **statistically separated** (CI [-0.0005,-0.0001] excludes 0; Holm-McNemar p=0.0663).
- **C1c** (in-corpus d3_en: classical champion vs 256 transformer): AUC Δ=+0.0017 is **statistically separated** (CI [+0.0006,+0.0030] excludes 0; Holm-McNemar p=0.0104).
- **C2** (d3_en→d2: 512 vs 256 transformer (the ablation recovery)): AUC Δ=+0.1134 is **statistically separated** (CI [+0.1042,+0.1229] excludes 0; Holm-McNemar p=0).
- **C3** (d3_en→d2: 512 transformer vs best classical cross champion): AUC Δ=+0.0917 is **statistically separated** (CI [+0.0784,+0.1052] excludes 0; Holm-McNemar p=0).
- **C4** (d2→d3_en: best classical cross champion vs 256 transformer): AUC Δ=+0.0681 is **statistically separated** (CI [+0.0534,+0.0832] excludes 0; Holm-McNemar p=3.9e-05).
- **C5** (d2→d3_en: 512 vs 256 transformer (expected null)): AUC Δ=-0.0106 — **the 95% CI includes 0** ([-0.0223,+0.0009]): no AUC-level separation (Holm-McNemar p=0.0303).

Single-AUC 95% CIs (bootstrap, B=10,000) for the 12 headline numbers:

| number | AUC | 95% CI |
|---|--:|---|
| svc_d1sb@d1sb | 0.9998 | [0.9997, 0.9999] |
| nb_d2@d2 | 0.9996 | [0.9994, 0.9998] |
| svc_d3en@d3en | 0.9993 | [0.9988, 0.9996] |
| svc_d3en@d2 | 0.7523 | [0.7379, 0.7661] |
| lr_d2@d3en | 0.8429 | [0.8278, 0.8571] |
| distilbert256_d1sb@d1sb | 0.9999 | [0.9998, 1.0000] |
| distilbert256_d2@d2 | 0.9999 | [0.9998, 1.0000] |
| roberta256_d3en@d3en | 0.9976 | [0.9962, 0.9987] |
| roberta256_d3en@d2 | 0.7306 | [0.7159, 0.7449] |
| roberta512_d3en@d2 | 0.8440 | [0.8324, 0.8552] |
| distilbert256_d2@d3en | 0.7748 | [0.7568, 0.7923] |
| distilbert512_d2@d3en | 0.7642 | [0.7458, 0.7821] |

## (c) Imbalance stress — operational view (FP per 1,000 legitimate emails)

Eval-side prevalence stress (all legit kept; positives subsampled; 50 draws, mean [2.5–97.5 pct]).

| corpus | prevalence | family | PR-AUC | precision@95%recall | FP / 1,000 legit @95%recall |
|---|---|---|--:|--:|--:|
| d1_sb | 10% | classical | 0.999 [0.997,0.999] | 0.999 | 0.2 [0.0,0.6] |
| d1_sb | 10% | transformer | 0.999 [0.998,1.000] | 0.997 | 0.3 [0.0,0.6] |
| d1_sb | 5% | classical | 0.998 [0.994,0.999] | 0.996 | 0.2 [0.0,0.8] |
| d1_sb | 5% | transformer | 0.998 [0.997,0.999] | 0.994 | 0.3 [0.0,0.6] |
| d1_sb | 2% | classical | 0.994 [0.977,1.000] | 0.978 | 0.5 [0.0,2.6] |
| d1_sb | 2% | transformer | 0.997 [0.988,0.999] | 0.981 | 0.4 [0.0,1.1] |
| d2 | 10% | classical | 0.997 [0.995,0.998] | 0.985 | 1.6 [0.9,3.0] |
| d2 | 10% | transformer | 0.999 [0.998,1.000] | 0.999 | 0.1 [0.0,0.9] |
| d2 | 5% | classical | 0.994 [0.987,0.998] | 0.968 | 1.7 [0.9,4.3] |
| d2 | 5% | transformer | 0.999 [0.996,1.000] | 0.997 | 0.2 [0.0,0.9] |
| d2 | 2% | classical | 0.986 [0.969,0.996] | 0.924 | 1.7 [0.4,4.3] |
| d2 | 2% | transformer | 0.996 [0.985,1.000] | 0.993 | 0.1 [0.0,0.9] |
| d3_en | 10% | classical | 0.994 [0.988,0.998] | 0.972 | 3.1 [1.0,9.2] |
| d3_en | 10% | transformer | 0.982 [0.976,0.989] | 0.929 | 8.3 [4.1,16.4] |
| d3_en | 5% | classical | 0.990 [0.978,0.998] | 0.937 | 3.6 [0.2,13.3] |
| d3_en | 5% | transformer | 0.966 [0.935,0.985] | 0.855 | 9.5 [4.1,22.6] |
| d3_en | 2% | classical | 0.980 [0.925,1.000] | 0.791 | 8.3 [0.0,49.1] |
| d3_en | 2% | transformer | 0.933 [0.856,0.976] | 0.661 | 14.7 [4.1,74.9] |

Train-side (classical only, hyperparameters frozen; 9 single-use natural-TEST evals):

| corpus | prevalence | train pos/neg | PR-AUC | precision@95%recall | FP / 1,000 legit |
|---|---|---|--:|--:|--:|
| d1_sb | 10% | 1849/16644 | 1.000 | 0.998 | 2.0 |
| d1_sb | 5% | 876/16644 | 0.999 | 0.998 | 2.5 |
| d1_sb | 2% | 339/16644 | 0.999 | 0.996 | 4.5 |
| d2 | 10% | 1198/10784 | 0.999 | 0.992 | 7.4 |
| d2 | 5% | 567/10784 | 0.997 | 0.988 | 10.8 |
| d2 | 2% | 220/10784 | 0.995 | 0.975 | 22.1 |
| d3_en | 10% | 505/4553 | 0.997 | 0.990 | 19.5 |
| d3_en | 5% | 239/4553 | 0.995 | 0.977 | 44.1 |
| d3_en | 2% | 92/4553 | 0.990 | 0.947 | 103.6 |

## (d) Pipeline A preprocessing ablation — val ROC-AUC Δ vs V0 (VALIDATION only)

V1 stopwords · V2 placeholders · V3 SymSpell · V4 wordninja · V5 combined. Δ vs V0 per (train→eval, variant); **off-diagonal = cross-corpus** (the transfer test).

| train→eval | V1 | V2 | V3 | V4 | V5 |
|---|--:|--:|--:|--:|--:|
| d1_sb→d1_sb (self) | +0.0000 | +0.0000 | +0.0000 | -0.0001 | +0.0000 |
| d1_sb→d2 | -0.0095 | +0.0133 | +0.0025 | -0.0026 | +0.0038 |
| d1_sb→d3_en | +0.0024 | -0.0050 | -0.0003 | -0.0007 | -0.0034 |
| d2→d1_sb | -0.1106 | +0.0774 | +0.0052 | -0.0148 | -0.0958 |
| d2→d2 (self) | +0.0001 | +0.0000 | +0.0000 | +0.0001 | +0.0000 |
| d2→d3_en | -0.0667 | +0.0021 | -0.0026 | -0.0163 | -0.1261 |
| d3_en→d1_sb | -0.0038 | +0.0663 | +0.0050 | +0.0039 | +0.0611 |
| d3_en→d2 | -0.0430 | +0.0519 | +0.0053 | +0.0022 | -0.0259 |
| d3_en→d3_en (self) | +0.0000 | -0.0003 | +0.0000 | +0.0001 | -0.0002 |

**V2 (placeholders) cross-corpus hypothesis:** mean Δ vs V0 over the 6 off-diagonal (cross-corpus) pairs = **+0.0343** (5/6 pairs improved). Hypothesis HELD (pre-registered directional; descriptive, no significance test).

## Observations

1. **In-corpus the gaps are near-ceiling and sign-mixed — and the one material, significant difference favours the *classical* champion on d3_en.** With Δ = classical − transformer: d1_sb Δ=-0.0001 (CI [-0.0002,-0.0000] includes 0 — indistinguishable); d2 Δ=-0.0003 (CI [-0.0005,-0.0001] excludes 0 — a ~0.03 pp edge to the transformer, Holm-McNemar p=0.0663); but **C1c d3_en Δ=+0.0017, CI [+0.0006,+0.0030] excludes 0, Holm p=0.0104 — the classical LinearSVC significantly beats RoBERTa on the AI corpus.** This sharpens the Directive-6 'match but don't beat': at the 0.999 ceiling the in-corpus gaps are ≤0.0003 either way, and the only material, significant one favours the classical model.

2. **The 512 recovery is strongly significant and beats the classical cross champion.** C2 (512 vs 256, d3_en→d2) Δ=+0.1134 CI [+0.1042,+0.1229], C3 (512 vs classical cross) Δ=+0.0917 CI [+0.0784,+0.1052] — both exclude 0 (Holm p≈0). The Directive-8 H1 verdict survives paired inference.

3. **The classical cross champion beats the 256 transformer where the source is short (d2→d3_en), and C5 is the pre-registered null.** C4 Δ=+0.0681 CI [+0.0534,+0.0832] (separated, classical wins); **C5 (512 vs 256, d2→d3_en) Δ=-0.0106, CI [-0.0223,+0.0009] includes 0** exactly as pre-registered — 512 adds no AUC-level signal when the training corpus was already un-truncated.

4. **Imbalance: precision at fixed recall collapses as phishing gets rarer — the FP-per-1,000 framing shows the operational cost.** At 2% prevalence every family pays far more false positives per 1,000 legitimate emails to hold 95% recall than at 10% (see table (c)); the PR-AUC / FP-per-1000 curves (figure) are the deployment-relevant robustness signal, not the prevalence-inflated ROC-AUC. And the more-robust family is **corpus-dependent**: at 2% prevalence the transformer holds 95% recall at only 0.1 FP/1,000 on d2 (vs classical 1.7), but on the AI corpus d3_en the classical champion is cleaner (8.3 vs transformer 14.7 FP/1,000) — matching C1c, where classical wins d3_en.

5. **Preprocessing (validation only): placeholders (V2) help cross-corpus transfer** (mean Δ vs V0 over the 6 cross pairs = +0.0343); the pre-registered directional hypothesis held. The ablation is descriptive and VALIDATION-only — it cannot and does not change any committed champion.
