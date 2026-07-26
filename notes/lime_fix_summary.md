# Directive 11 — Corrected-LIME isolated re-run (resolves the Amendment 4 defect)

**Pre-registered** in `notes/decisions.md` (2026-07-26, commit `f3d5e3a`, committed alone with the
known-answer suite and the one-argument fix, before any corrected-LIME result existed).

**What was wrong.** `src/xai/explainers.py` passed `split_expression=r"\w+"` to `LimeTextExplainer`,
but that argument is LIME's **separator** pattern (default `r"\W+"`), used as the *non-word* test. The
inversion discarded every real word from the perturbation vocabulary, so **all 2,100 committed LIME
attribution vectors were identically zero** and every "LIME top-k" was really the document's **first k
word tokens** (Amendment 4). That arm is written **FIRST10** throughout this report.

**What was done.** LIME only, re-run over the **same 12 committed units** and the **same committed
`explained_ids` documents**, with the committed Directive 7 parameters verbatim (`num_samples=1000`,
seed 42, n=200 in-corpus / 100 cross) and the identical metric pipeline and random baseline. Results
were written to an isolated namespace; **`results/xai_metrics.csv`, `results/xai/*`, `results.csv`,
`results_512.csv` and every D9/D10 artifact are byte-untouched** (verified). Total cost **2.75 h**
(under the 3.8 h projection). A permanent positive-control regression suite
(`src/xai/test_known_answer.py`, **41 checks, PASS**) now guards every explainer.

---

## (a) Corrected LIME vs the void FIRST10 arm vs the committed SHAP / IG

### Faithfulness — AOPC (higher = more faithful)

| unit | AOPC corrected LIME | AOPC FIRST10 (void) | AOPC SHAP | AOPC IG | AOPC random |
|---|---|---|---|---|---|
| `tfidf_d1_sb` | **0.1865** | 0.0159 | 0.1989 | — | 0.0116 |
| `tfidf_d2` | **0.1812** | 0.0260 | 0.1913 | — | 0.0242 |
| `tfidf_d3_en` | **0.0708** | 0.0056 | 0.0936 | — | 0.0038 |
| `ensemble_d1_sb` | **0.2357** | −0.0016 | 0.2808 | — | 0.0001 |
| `ensemble_d2` | **0.2920** | 0.0409 | 0.3300 | — | 0.0399 |
| `ensemble_d3_en` | **0.1903** | 0.0023 | 0.2317 | — | 0.0024 |
| `transformer_d1_sb` | **0.1233** | 0.0300 | 0.0931 | 0.0801 | 0.0116 |
| `transformer_d2` | **0.1753** | 0.0435 | 0.1372 | 0.0984 | 0.0241 |
| `transformer_d3_en` | **0.1265** | 0.0373 | 0.0904 | 0.1288 | 0.0313 |
| `xfer_d3_en__d1_sb` | **0.2180** | 0.0680 | 0.1844 | 0.1451 | 0.0427 |
| `xfer_d2__d3_en` | **0.2818** | 0.0551 | 0.2125 | 0.0765 | 0.0218 |
| `xfer_d3_en__d2` | **0.2945** | 0.0753 | 0.2227 | 0.1647 | 0.0730 |

### Stability and plausibility (corrected vs void)

| unit | Jaccard corr. | Jaccard void | Spearman corr. | Spearman void | seed-agree corr. | seed-agree void | plaus@10 corr. | plaus@10 void | plaus@10 random |
|---|---|---|---|---|---|---|---|---|---|
| `tfidf_d1_sb` | 0.5774 | 0.9924 | 0.5777 | nan | 0.5888 | 1.0000 | 0.2357 | 0.0998 | 0.1659 |
| `tfidf_d2` | 0.7004 | 0.9975 | 0.7143 | nan | 0.7111 | 1.0000 | 0.0941 | 0.0653 | 0.0730 |
| `tfidf_d3_en` | 0.6745 | 0.9973 | 0.8644 | nan | 0.6929 | 1.0000 | 0.2044 | 0.2360 | 0.1272 |
| `ensemble_d1_sb` | 0.5134 | 0.9938 | 0.4686 | nan | 0.4906 | 1.0000 | 0.2829 | 0.0771 | 0.1752 |
| `ensemble_d2` | 0.5569 | 0.9956 | 0.5078 | nan | 0.5549 | 1.0000 | 0.1385 | 0.0697 | 0.0635 |
| `ensemble_d3_en` | 0.4298 | 0.9964 | 0.3472 | nan | 0.4332 | 1.0000 | 0.1954 | 0.2237 | 0.1389 |
| `transformer_d1_sb` | 0.4296 | 0.9925 | 0.4047 | nan | 0.4130 | 1.0000 | 0.2094 | 0.0849 | 0.1594 |
| `transformer_d2` | 0.4427 | 0.9978 | 0.4466 | nan | 0.4495 | 1.0000 | 0.0964 | 0.0711 | 0.0614 |
| `transformer_d3_en` | 0.3549 | 0.9960 | 0.2627 | nan | 0.3936 | 1.0000 | 0.2477 | 0.2532 | 0.1423 |
| `xfer_d3_en__d1_sb` | 0.4338 | 0.9898 | 0.4467 | nan | 0.4658 | 1.0000 | 0.1500 | 0.0909 | 0.0864 |
| `xfer_d2__d3_en` | 0.3555 | 0.9975 | 0.3188 | nan | 0.3541 | 1.0000 | 0.2529 | 0.2100 | 0.1386 |
| `xfer_d3_en__d2` | 0.5019 | 0.9949 | 0.5295 | nan | 0.5195 | 1.0000 | 0.0963 | 0.0407 | 0.0259 |

The void arm's `Spearman = nan` and `seed-agreement = 1.0000` on **all 12** units were the signature of
a constant vector. Both are now finite and informative.

### In-corpus family means

| family | metric | corrected LIME | FIRST10 (void) | SHAP | IG | random |
|---|---|---|---|---|---|---|
| tfidf | aopc | **0.1462** | 0.0158 | 0.1613 | — | 0.0132 |
| tfidf | comp@10 | **0.1496** | 0.0139 | 0.1648 | — | 0.0114 |
| tfidf | stability_jaccard | **0.6508** | 0.9957 | 0.9091 | — | — |
| tfidf | plausibility@10 | **0.1781** | 0.1337 | 0.2183 | — | 0.1220 |
| ensemble | aopc | **0.2393** | 0.0139 | 0.2808 | — | 0.0141 |
| ensemble | comp@10 | **0.2537** | 0.0142 | 0.2951 | — | 0.0138 |
| ensemble | stability_jaccard | **0.5000** | 0.9953 | 0.9179 | — | — |
| ensemble | plausibility@10 | **0.2056** | 0.1235 | 0.2273 | — | 0.1259 |
| transformer | aopc | **0.1417** | 0.0369 | 0.1069 | 0.1024 | 0.0223 |
| transformer | comp@10 | **0.1452** | 0.0404 | 0.1056 | 0.1055 | 0.0280 |
| transformer | stability_jaccard | **0.4091** | 0.9954 | 0.5118 | 0.7028 | — |
| transformer | plausibility@10 | **0.1845** | 0.1364 | 0.1615 | 0.1476 | 0.1210 |

### Most-faithful method per unit — before vs after

| unit | most faithful BEFORE (committed) | most faithful AFTER (corrected LIME included) | flip? |
|---|---|---|---|
| `tfidf_d1_sb` | shap (0.1989) | shap (0.1989) | no |
| `tfidf_d2` | shap (0.1913) | shap (0.1913) | no |
| `tfidf_d3_en` | shap (0.0936) | shap (0.0936) | no |
| `ensemble_d1_sb` | shap (0.2808) | shap (0.2808) | no |
| `ensemble_d2` | shap (0.3300) | shap (0.3300) | no |
| `ensemble_d3_en` | shap (0.2317) | shap (0.2317) | no |
| `transformer_d1_sb` | shap (0.0931) | **lime_corrected (0.1233)** | **YES** |
| `transformer_d2` | shap (0.1372) | **lime_corrected (0.1753)** | **YES** |
| `transformer_d3_en` | ig (0.1288) | ig (0.1288) | no |
| `xfer_d3_en__d1_sb` | shap (0.1844) | **lime_corrected (0.2180)** | **YES** |
| `xfer_d2__d3_en` | shap (0.2125) | **lime_corrected (0.2818)** | **YES** |
| `xfer_d3_en__d2` | shap (0.2227) | **lime_corrected (0.2945)** | **YES** |

**5 of 12 units flip** — both non-d3_en transformer units and **all three cross-corpus transfer
units**. The six sklearn units do not flip: SHAP remains the most faithful there.

---

## (b) Committed claims — restored, revised, or withdrawn

| committed claim | status | evidence |
|---|---|---|
| "**LIME ≈ random**" | **WITHDRAWN — reversed** | corrected AOPC 0.071–0.295 vs random 0.0001–0.073; **above its random baseline on 12/12 units, by at least 4×** (per-unit ratio range 4.0×–2357×; the 2357× is an artifact of `ensemble_d1_sb`'s near-zero denominator 0.0001, so the ratio is reported as a floor, not a range). comp@k monotone 12/12 and suff@10 ≤ random 12/12 |
| "**SHAP is the only demonstrably faithful method**" | **WITHDRAWN** | corrected LIME clears its random baseline on 12/12 units and beats SHAP on 5 |
| "**faithfulness–stability trade-off: LIME most stable, least faithful**" | **INVERTED, but a real trade-off survives** | LIME is the **least stable** method on **11/12 units** (per-unit Jaccard 0.355–0.700 vs SHAP 0.404–0.951; the exception is `xfer_d3_en__d2`, where IG 0.4907 < LIME 0.5019) and among the **most faithful**. The trade-off exists — LIME sits at the faithful-but-unstable end, the exact opposite of the committed wording, which was an artifact of a constant vector scoring Jaccard 0.99 / seed-agreement 1.000 |
| "**transformers are hardest to explain; IG is their best explainer**" | **REVISED** | on **family means** corrected LIME is the most faithful (0.1417 vs SHAP 0.1069, IG 0.1024) **and** most plausible (0.1845 vs 0.1615, 0.1476) transformer explainer. Per unit this is not universal: SHAP is more plausible on `transformer_d2` (0.1157 vs 0.0964) and `xfer_d3_en__d1_sb` (0.1591 vs 0.1500), and on **d3_en** IG still edges LIME on AOPC (0.1288 vs 0.1265 — a 0.0023 gap on the unit with the most negligible-attribution documents, see Limitation 6, so this ordering should be treated as unresolved). IG keeps the best stability (0.7028 vs 0.4091) and is ~6.6× cheaper to run |
| "**RQ3 answer: linear TF-IDF + SHAP**" | **HOLDS — on revised grounds** | see §(c) |
| SHAP and IG numbers, D9 statistics, all transformer/classical results, D10 E1/P2/P3/P4 | **UNAFFECTED** | no LIME quantity enters them; canonical CSVs byte-identical |
| D10's exploratory "d3_en LIME anomaly" | **remains withdrawn** | it was measured on the void arm; the corrected arm was **not** human-annotated (deferred, out of scope) |

---

## (c) Revised RQ3 paragraph (with re-measured LIME cost)

**The RQ3 answer — a linear TF-IDF model explained with SHAP — survives, but the reasoning that
supported it was partly false and is replaced.** The old justification leaned on "LIME ≈ random",
which was an artifact. On the linear TF-IDF family the corrected comparison is:

| axis (tfidf family mean) | SHAP | corrected LIME | verdict |
|---|---|---|---|
| faithfulness (AOPC) | **0.1613** | 0.1462 | SHAP ahead, but the gap is now narrow (0.015, not 10×) — and **untested for significance** (Limitation 7) |
| stability (Jaccard) | **0.9091** | 0.6508 | SHAP markedly more stable (holds on all 3 tfidf units) |
| plausibility@10 | **0.2183** | 0.1781 | SHAP more plausible **on the family mean**, but **not on `tfidf_d1_sb`** (LIME 0.2357 vs SHAP 0.2338); both above random 0.1220 |
| **attribution cost** per 200 docs | **0.6 s** | 39.6 s | **SHAP ≈ 64.5× cheaper** |

**Cost basis (corrected).** The figures above are **attribution time only** (`timing.attr` in the
per-unit payloads) — the like-for-like quantity. The `runtime_s` column is *not* comparable across
methods: it covers the whole metric pipeline (attribution + faithfulness + stability + plausibility)
and for LIME additionally includes `lime_seed_agreement`, a **LIME-only** step that is never computed
for SHAP or IG (≈79 s of the tfidf LIME total). Quoting `runtime_s` (6.1 s vs 311.4 s ⇒ "51×") would
mislabel a pipeline cost as an explanation cost; on the correct basis the tfidf gap is **larger**
(64.5×) and the transformer gaps are smaller.

So SHAP wins on **faithfulness, stability and cost** for the linear model, and on plausibility at the
family level (with one per-unit exception). The cost gap is the decisive practical argument: **0.6 s
versus 39.6 s** to explain the same 200 documents. The honest revision is that **LIME is a viable
explainer, not a broken one** — it is simply dominated for this model family. For **transformers** the
picture changes: on attribution cost LIME (181.1 s) is **6.6×** IG (27.4 s) and only **1.6×** SHAP
(115.9 s), while LIME leads on faithfulness and IG leads on stability — so **IG remains the pragmatic
transformer default**, with LIME the choice when faithfulness is the priority and its instability and
cost are acceptable.

---

## (d) Figures

`figures/xai_lime_fix/xai_lime_fix_deletion_curves.(png|pdf)` — comprehensiveness vs k (1..20), one
panel per in-corpus corpus, corrected LIME against committed SHAP/IG, the void **FIRST10** arm and the
random baseline. `figures/xai_lime_fix/xai_lime_fix_bars.(png|pdf)` — comp@10 / stability / plausibility
per family. FIRST10 is drawn hatched and labelled as a **position baseline**, not LIME. The canonical
`figures/xai/*` are deliberately **untouched**; the erratum at the top of `notes/xai_summary.md`
governs them.

---

## (e) Observations

1. **A one-character argument error inverted a headline finding in both directions.** The committed
   phase concluded LIME was the least faithful and most stable explainer; corrected, it is among the
   **most faithful** and the **least stable**. Both halves of the "trade-off" were artifacts of a
   constant vector — yet the numbers looked like a coherent, publishable methodological story, which
   is precisely why nobody questioned them.
2. **The random baseline could not catch it, and that is the transferable lesson.** A zero vector
   trivially fails to beat random, so the mandatory negative control was *satisfied* by a broken
   explainer. What was missing was a **positive control**: `src/xai/test_known_answer.py` now asserts
   planted-cue recovery, non-degeneracy on real documents, and (T4) that a zero vector would fail —
   i.e. the suite provably detects the original defect. Any future explainer added to this framework
   must pass it.
3. **Perturbation-based explainers gain the most where the model is hardest to explain.** Corrected
   LIME beats SHAP on all three **cross-corpus transfer** units (0.218/0.282/0.295 vs
   0.184/0.213/0.223) and on the two non-d3_en transformer units, while losing on all six sklearn
   units. LIME's model-agnostic local surrogate appears better suited to the transformer's non-linear
   decision surface than SHAP's partition approximation at a fixed `max_evals=500` budget.
4. **Faithfulness and stability genuinely trade off — with LIME on the opposite end from the committed
   claim.** Corrected LIME's seed-agreement is 0.35–0.71: two runs with different seeds share only
   about half their top-10. An explainer that is faithful but seed-dependent is a real deployment
   hazard (an analyst re-running an explanation sees a different rationale), and this is now an
   evidence-backed argument rather than an artifact.
5. **The cost ordering is unchanged and remains decisive for the framework recommendation.** Linear
   TF-IDF + SHAP costs **0.6 s** of attribution per 200 documents against **39.6 s** for LIME — a
   **64.5×** gap that dwarfs the 0.015 AOPC difference. The corrected result therefore *strengthens*
   the practical recommendation while *removing* the false claim it used to rest on.
6. **LIME degenerates on model-saturated documents — a residual, method-specific failure mode.** On
   **84 / 2,100 documents (4.0%)** the corrected LIME attributions are numerically negligible
   (max |attr| ≈ 1e-6, stored as exactly zero at the pipeline's 5-dp rounding), concentrated on
   `transformer_d3_en` (45/200 = 22.5%), `xfer_d3_en__d1_sb` (21/100 = 21.0%) and `transformer_d1_sb`
   (18/200 = 9.0%); **zero occurrences on the six sklearn units**. These are near-saturated
   predictions where 1,000 perturbations leave the output unmoved, so the local surrogate has no
   signal to fit — **SHAP and IG produce normal-magnitude attributions on the very same documents**.
   **Where it bites:** 80 of the 84 (95%) are **ham**-predicted and only 4 are phishing-predicted, so
   **plausibility@10 — computed on phishing-predicted documents only — is essentially unaffected**,
   whereas **faithfulness and stability iterate over all explained documents and are affected**
   (biasing those units' AOPC down and stability up). The rate is population-dependent: a fresh
   random sample of *phishing-predicted* transformer TEST documents showed 2/5 below a 1e-5 floor, so
   this is not confined to the ham side. It is a genuine limitation of perturbation-based explanation
   under prediction saturation, not a coding defect — but a minority of LIME explanations do silently
   collapse to the first-k-token ordering, the very failure mode this directive set out to remove.
   The positive control has been strengthened in response (magnitude floor instead of exact zero, a
   rate gate, and a `T2b` saturation diagnostic that reports the rate on every run).

---

## Limitations

1. **No human validation of the corrected arm.** Directive 10's plausibility ratings were collected on
   the FIRST10† sets and **cannot be reused**; re-annotation was explicitly deferred (out of scope).
   Corrected-LIME plausibility is therefore the **lexicon proxy only** — and D10 established that proxy
   is "fair" (κ = 0.396) and *conservative*, so these plausibility numbers are ordinal, not absolute.
2. **The canonical `xai_metrics.csv` LIME rows are no longer reproducible from current code** — by
   design. They are the void arm's record, retained deliberately; the historical behaviour can still be
   reproduced by passing `explainers.LEGACY_VOID_SPLIT_EXPRESSION`.
3. **Scope.** LIME only. SHAP and IG were not re-run and their code paths were not touched, so all
   SHAP/IG numbers quoted here are the committed ones. `cue_lexicon.py` is unchanged.
4. **Single seed for the main run** (42, as pre-registered). LIME's seed-agreement (0.35–0.71) is
   measured across seeds 42/43, so the per-unit AOPC values themselves carry seed variance that is not
   quantified by a confidence interval. Part of LIME's measured *instability* is also its own
   Monte-Carlo sampling variance at the inherited `num_samples=1000` budget, not model sensitivity; no
   budget-sensitivity analysis was run.
5. **IG diagnostic, and the limits of the new positive control.** The known-answer suite gates IG on
   T0/T1/T4 but records T1b as a diagnostic, because IG's signed attributions are diffuse; IG was
   verified to have **no code defect** (its word-level output matches its raw sub-token ranking).
   Reviewers established by Monte Carlo that **T1 alone is weak** — because the objectively-derived
   planted set covers 53–60% of each document's vocabulary, a random attribution passes T1 at
   p≈0.9998. T1's real power is against **degenerate** vectors (T4 proves a zero vector fails), which
   is exactly the D7 defect class; **T1b is the discriminating gate** (random passes at p≈0.5), and IG
   is exempt from it. The suite should therefore be read as a *degeneracy* detector, not a proof of
   attribution quality. **Strengthened in response to the review** (and re-run, still 41 checks PASS):
   T2 now uses a **magnitude floor** (1e-5) instead of an exact-zero test, gates on a **rate**
   (< 50% of sampled documents, since the D7 defect was 100% degenerate while saturation degeneracy is
   a genuine minority), samples **phishing-predicted** documents, and is accompanied by a non-gating
   **T2b** saturation diagnostic. A tighter planted set for T1 remains outstanding work.
6. **Residual negligible-attribution documents.** 84/2,100 (4.0%) corrected-LIME vectors are
   numerically negligible (Observation 6), up to 22.5% on `transformer_d3_en`. These documents enter
   the AOPC and stability means (faithfulness iterates over all explained docs), biasing that unit's
   AOPC **down** and its stability **up**. The direction is conservative for this report's headline
   claims, but it makes the single closest ordering — `transformer_d3_en` IG 0.1288 vs LIME 0.1265 —
   **unresolved**.
7. **No significance testing.** All comparisons here are point estimates. The project already has
   paired-bootstrap machinery (Directive 9), and the RQ3 margin (0.0151 AOPC) and the d3_en ordering
   (0.0023) are small enough that CIs are warranted before either is asserted in the dissertation.
8. **Stored corrected-LIME attributions are a lossy record.** The per-unit JSONs round `attrs` to 5 dp
   (inherited pipeline behaviour). For LIME's bag-of-words weights this creates ties that re-order the
   top-10, so `plausibility@10` re-derived from the stored JSONs matches the committed CSV on only
   7/12 units (it matches 30/30 for the canonical SHAP/IG/void rows). **The CSV is authoritative** — a
   live re-run at full precision reproduces its values exactly — but the regeneration-anchor property
   used in Directive 10 does **not** hold for this arm from the stored files.

## Artifacts

`results/xai_metrics_limefix.csv` (12 rows) · `results/xai_lime_fix/<unit>__lime.json` (git-ignored,
regenerable) · `figures/xai_lime_fix/*` · `src/xai/{run_lime_fix,report_lime_fix,figures_lime_fix,
test_known_answer}.py` · fix in `src/xai/explainers.py` (`lime_attr`, `LEGACY_VOID_SPLIT_EXPRESSION`).
