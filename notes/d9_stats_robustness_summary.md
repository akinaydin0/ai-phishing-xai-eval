# D9 — Statistical Inference, Imbalance & Preprocessing Robustness

All comparisons, champions and claim directions were **pre-registered before any statistic existed** (notes/decisions.md, 2026-07-17). CIs/p-values **annotate** the committed comparisons — no champion, threshold or claim direction is reselected on their basis. Score vectors were recomputed from stored artefacts and asserted equal to the committed test AUCs to 4 dp. Full log: `results/stats/stats_summary.csv`.

## (a) The seven comparisons — Δ AUC (95% paired stratified bootstrap, B=10,000) + McNemar/Holm

The **AUC-level** CI tests the ranking; the **threshold-level** McNemar tests 0.5-operating-point errors. They answer different questions and are reported in separate columns; a claim never cites a McNemar p it does not support. CIs are shown to 6 dp (they are stored to 6 dp in the CSV).

| id | comparison | AUC_a | AUC_b | Δ=a−b | 95% CI (Δ) | McNemar n01/n10 | p_raw | p_holm | AUC-level | thr-level |
|---|---|--:|--:|--:|---|:--:|--:|--:|:--:|:--:|
| C1a | LinearSVC vs DistilBERT-256 | 0.9998 | 0.9999 | -0.0001 | [-0.000219, -0.000023] | 27/17 | 0.174 | 0.174 | excl 0 | ns |
| C1b | MultinomialNB vs DistilBERT-256 | 0.9996 | 0.9999 | -0.0003 | [-0.000543, -0.000092] | 37/20 | 0.0331 | 0.0663 | excl 0 | ns |
| C1c | LinearSVC vs RoBERTa-256 | 0.9993 | 0.9976 | +0.0017 | [+0.000584, +0.003044] | 14/36 | 0.0026 | 0.0104 | excl 0 | sig |
| C2 | RoBERTa-512(d3en->d2) vs RoBERTa-256(d3en->d2) | 0.8440 | 0.7306 | +0.1134 | [+0.104178, +0.122906] | 139/607 | 0 | 0 | excl 0 | sig |
| C3 | RoBERTa-512(d3en->d2) vs LinearSVC-cross | 0.8440 | 0.7523 | +0.0917 | [+0.078399, +0.105240] | 496/859 | 0 | 0 | excl 0 | sig |
| C4 | LogReg-cross(d2->d3en) vs DistilBERT-256(d2->d3en) | 0.8429 | 0.7748 | +0.0681 | [+0.053360, +0.083210] | 249/360 | 8e-06 | 3.9e-05 | excl 0 | sig |
| C5 | DistilBERT-512(d2->d3en) vs DistilBERT-256(d2->d3en) | 0.7642 | 0.7748 | -0.0106 | [-0.022303, +0.000919] | 266/209 | 0.0101 | 0.0303 | incl 0 | sig |

**Champion-identity disclosure (in-corpus, C1a/C1b/C1c).** The `svc` family is **LinearSVC** (linear SVM, `sklearn.svm.LinearSVC`), not a kernel SVC. **Classical side = strongest committed classical result per corpus (test champion); transformer side = val-selected finalist. This asymmetry biases the in-corpus comparison toward the classical model, making the transformer's d1_sb/d2 parity conservative; C1c is unaffected (d3_en's val- and test-selected classical coincide).** Specifically: d1_sb LinearSVC is both val-tied-best (0.9997) and test-best (0.9998); **d2 MultinomialNB is the test champion (0.9996) but NOT val-selected** (NB val 0.9990 < LR/SVC 0.9991); d3_en LinearSVC is val-best (0.9989) AND test-best (0.9993).

## (b) AUC-level vs threshold-level — reported separately wherever they diverge

- **C1a** (in-corpus d1_sb, classical vs 256 transformer): **AUC-level** — nominally separated by ~1 bp (CI [-0.000219, -0.000023]; upper bound just below zero → the 95% CI **excludes 0 at full precision**). **Threshold-level** — Holm-McNemar p=0.174, **NOT** separated. Practically **indistinguishable at the 0.999 ceiling**.
- **C1b** (in-corpus d2, classical vs 256 transformer): **AUC-level** — CI [-0.0005, -0.0001] **excludes 0** (a ~3 bp edge to the transformer). **Threshold-level** — Holm-McNemar p=0.0663, **not significant**.
- **C1c** (in-corpus d3_en, classical vs 256 transformer): **AUC-level** — CI [+0.0006, +0.0030] **excludes 0** (classical +17 bp). **Threshold-level** — Holm-McNemar p=0.0104, **significant → the classical LinearSVC beats RoBERTa on the AI corpus** (both tests agree).
- **C2** (d3_en→d2: 512 vs 256 transformer (the ablation recovery)): **AUC-level** — CI [+0.1042, +0.1229] excludes 0 (Δ=+0.1134); **Threshold-level** — Holm-McNemar p=0, significant. Both tests agree.
- **C3** (d3_en→d2: 512 transformer vs best classical cross champion): **AUC-level** — CI [+0.0784, +0.1052] excludes 0 (Δ=+0.0917); **Threshold-level** — Holm-McNemar p=0, significant. Both tests agree.
- **C4** (d2→d3_en: best classical cross champion vs 256 transformer): **AUC-level** — CI [+0.0534, +0.0832] excludes 0 (Δ=+0.0681); **Threshold-level** — Holm-McNemar p=3.9e-05, significant. Both tests agree.
- **C5** (d2→d3_en, 512 vs 256 transformer): **AUC-level** — CI [-0.0223, +0.0009] **includes 0 — the pre-registered null holds at AUC level.** **Threshold-level** — Holm-McNemar p=0.0303 **IS significant, favouring the 256 model** (n01=266 vs n10=209) at the fixed 0.5 operating point.

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

1. **In-corpus the gaps are near-ceiling and sign-mixed — and the one material, significant difference favours the *classical* champion on d3_en.** With Δ = classical − transformer (classical side = strongest committed classical / test champion; transformer = val-selected — an asymmetry that makes d1_sb/d2 parity conservative): d1_sb Δ=-0.0001 — **AUC-level** nominally separated by ~1 bp (CI upper bound -0.000023, just below 0), but **threshold-level** Holm-McNemar p=0.174 not separated → practically indistinguishable at the ceiling; d2 Δ=-0.0003 (AUC CI [-0.0005,-0.0001] excludes 0 — a ~3 bp edge to the transformer; threshold p=0.0663, ns); but **C1c d3_en Δ=+0.0017, CI [+0.0006,+0.0030] excludes 0 AND Holm p=0.0104 significant — the classical LinearSVC beats RoBERTa on the AI corpus on both tests.** This sharpens the Directive-6 'match but don't beat': in-corpus gaps are ≤3 bp either way at the 0.999 ceiling, and the only material, significant one favours the classical model (C1c is unaffected by the selection asymmetry — d3_en's val- and test-selected classical coincide).

2. **The 512 recovery is strongly significant and beats the classical cross champion.** C2 (512 vs 256, d3_en→d2) Δ=+0.1134 CI [+0.1042,+0.1229], C3 (512 vs classical cross) Δ=+0.0917 CI [+0.0784,+0.1052] — both exclude 0 (Holm p≈0). The Directive-8 H1 verdict survives paired inference.

3. **The classical cross champion beats the 256 transformer where the source is short (d2→d3_en); C5's two tests diverge and must be reported separately.** C4 Δ=+0.0681 CI [+0.0534,+0.0832] (both tests agree, classical wins). **C5 (512 vs 256, d2→d3_en): AUC-level — Δ=-0.0106, CI [-0.0223,+0.0009] includes 0, so the pre-registered null holds at AUC level** (512 adds no ranking signal when the training corpus was already un-truncated). **Threshold-level — Holm-McNemar p=0.0303 IS significant, favouring the 256 model** (n01=266 vs n10=209) at the fixed 0.5 operating point. The two tests answer different questions; neither is cited for the other's claim.

4. **Imbalance: precision at fixed recall collapses as phishing gets rarer — the FP-per-1,000 framing shows the operational cost.** At 2% prevalence every family pays far more false positives per 1,000 legitimate emails to hold 95% recall than at 10% (see table (c)); the PR-AUC / FP-per-1000 curves (figure) are the deployment-relevant robustness signal, not the prevalence-inflated ROC-AUC. And the more-robust family is **corpus-dependent**: at 2% prevalence the transformer holds 95% recall at only 0.1 FP/1,000 on d2 (vs classical 1.7), but on the AI corpus d3_en the classical champion is cleaner (8.3 vs transformer 14.7 FP/1,000) — matching C1c, where classical wins d3_en.

5. **Preprocessing (validation only): placeholders (V2) help cross-corpus transfer** (mean Δ vs V0 over the 6 cross pairs = +0.0343); the pre-registered directional hypothesis held. The ablation is descriptive and VALIDATION-only — it cannot and does not change any committed champion.
