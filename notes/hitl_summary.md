# Directive 10 — Human-in-the-loop + LLM-judge plausibility validation

**Pre-registered** in `notes/decisions.md` (2026-07-18, commit `8ef0937`) with four dated amendments
(`9a32bb7`, `3f6edc8`, `3b2a155`, and Amendment 4 below); human ratings committed untouched (`51c30b6`)
**before** this analysis ran. Purpose: validate the **cue-lexicon plausibility proxy** — the XAI phase's
one stated limitation, and the supervisor's "plausibility over-reading" concern (weekly report 1,
item 6) — against (i) method-blind human annotation by the author and (ii) an LLM judge under the
identical blind protocol.

> ## ⚠ READ FIRST — the arm labelled "LIME" is not LIME (upstream defect, Amendment 4)
> The scoped review (Part E) discovered that **every LIME attribution vector in the entire XAI phase
> is identically zero** (2,100/2,100 documents across all 12 LIME units; global max |attr| = 0.0),
> caused by one inverted argument in `src/xai/explainers.py:152` —
> `LimeTextExplainer(split_expression=r"\w+")`. In LIME's API `split_expression` is the **separator**
> pattern (default `\W+`) and is used as the *non-word* test (`non_word = splitter.match`), so passing
> `\w+` classifies every real word as a non-word: the perturbation vocabulary collapses to whitespace
> and punctuation. Verified standalone: `IndexedString(text, split_expression=r"\w+").num_words()` = 2
> (`' '`, `'.'`) vs 10 with `\W+`, and LIME returns only separator features with ~1e-31 weights.
> Because `top_k_positions` stable-argsorts a zero vector, **every "LIME top-10" set is simply the
> document's first 10 word tokens** (verified: 80/80 D10 LIME sets == positions `[0…n−1]`).
>
> **Scope.** The defect is **pre-existing** (`explainers.py` is byte-identical to pre-D10 `fd274b9`);
> D10 inherited it and did not cause it. **D10's own protocol executed correctly** — the human and the
> judge rated exactly the token sets they were shown, so all agreement statistics stand. What is
> invalid is the **label**: 80/260 sets (30.8%) and 786/2,558 token judgments (30.7%) measure a
> **"first-10-tokens" positional baseline**, not LIME. Throughout this report that arm is written
> **`FIRST10†`** and every LIME-specific conclusion has been withdrawn. The regeneration anchor
> (§A.1) covered SHAP and IG only, so it could not have caught this.
>
> **Upstream consequence (outside D10's remit, flagged for decision):** the committed RQ3 claims
> "LIME ≈ random", "faithfulness–stability trade-off (LIME most stable, least faithful)" and the 12
> LIME rows of `results/xai_metrics.csv` are artifacts of a constant attribution vector — the
> tell-tale signatures are visible in the committed log itself (all 12 LIME rows:
> `stability_spearman = nan`, `lime_seed_agreement = 1.000`, `stability_jaccard` 0.990–0.998, the
> highest of any method). **Nothing has been re-run or silently corrected**; SHAP and IG results are
> unaffected.

**Universe.** 60 documents from the **committed TEST `explained_ids`** only (20 per corpus: 15 true
positives + 5 false positives, all phishing-**predicted**; d3_en drawn from the tfidf∩transformer
intersection so both families are annotated on the same docs). **260 blind sets** (tfidf
SHAP/FIRST10†/RANDOM on all 3 corpora = 180; transformer SHAP/FIRST10†/IG/RANDOM on d3_en = 80),
**2,558 token judgments**. Human: sets blind-shuffled to A/B/C(/D), ground-truth-blind. Judge:
`claude-opus-4-8`, one isolated call per set, **251 ok / 9 refusals**.

**Interpretation rule (binding).** Everything below is **reporting language only**. No committed XAI
result (`results/xai_metrics.csv`, 30 rows) is reselected, re-run, or reversed by this phase.

---

## (a) P1 — human cue-rate@10 vs the lexicon proxy

Mean over 20 docs per cell; 95% bootstrap CI over docs (B=10,000, seed 42; differences use a shared
resample index, i.e. paired). Human and lexicon columns cover all 20 docs per cell; the **judge**
column covers judge-rated sets only (d1_sb 19/20 docs, d2 18/20, d3_en 20/20 — limitation 4).
**† `FIRST10` is the arm generated as "LIME"; it is the document's first 10 word tokens (see above).**

| corpus · family | method | **human** cue-rate@10 (95% CI) | lexicon | LLM judge |
|---|---|---|---|---|
| d1_sb · tfidf | SHAP | **0.219** [0.111, 0.344] | 0.293 | 0.159 |
| | FIRST10† | 0.079 [0.020, 0.151] | 0.138 | 0.037 |
| | RANDOM | 0.069 [0.030, 0.117] | 0.168 | 0.049 |
| d2 · tfidf | SHAP | **0.140** [0.058, 0.240] | 0.098 | 0.078 |
| | FIRST10† | 0.095 [0.033, 0.180] | 0.053 | 0.117 |
| | RANDOM | 0.080 [0.037, 0.130] | 0.058 | 0.054 |
| d3_en · tfidf | SHAP | **0.210** [0.100, 0.335] | 0.185 | 0.185 |
| | FIRST10† | 0.225 [0.120, 0.335] | 0.215 | 0.200 |
| | RANDOM | 0.100 [0.035, 0.175] | 0.140 | 0.095 |
| d3_en · transformer | SHAP | 0.185 [0.090, 0.290] | 0.160 | 0.115 |
| | FIRST10† | 0.225 [0.120, 0.335] | 0.215 | 0.200 |
| | **IG** | **0.245** [0.125, 0.375] | 0.170 | 0.185 |
| | RANDOM | 0.060 [0.030, 0.095] | 0.110 | 0.045 |

*The two `FIRST10†` rows on d3_en are the **same 20 sets** (identical positions in 20/20 docs, identical
human labels and holistic scores), because the void arm ignores the model. They are printed once per
family for table symmetry and must **not** be read as two independent observations.*

**E1 (pre-registered: human SHAP > RANDOM in every cell) — HOLDS, 4/4** (unaffected by the defect;
SHAP and RANDOM are both genuine), statistically separated in **3/4** cells: d1_sb +0.150
[0.045, 0.275], d3_en·tfidf +0.110 [0.030, 0.195], d3_en·transformer +0.125 [0.060, 0.205]; **d2
+0.060 [−0.005, 0.140] is directionally correct but not separated**.

### Ranking concordance (sign agreement, human vs lexicon)

Restricted to the **6 contrasts that do not involve the void arm** — the only interpretable ones:

| cell | SHAP−RANDOM | IG−RANDOM | IG−SHAP |
|---|---|---|---|
| d1_sb · tfidf | ✔ | — | — |
| d2 · tfidf | ✔ | — | — |
| d3_en · tfidf | ✔ | — | — |
| d3_en · transformer | ✔ | ✔ | ✔ |

**6/6 valid contrasts agree in sign.** The raw all-contrast count is 12/15, and **all three
disagreements involve the void `FIRST10†` arm** (FIRST10−RANDOM on d1_sb and d2; IG−FIRST10 on
d3_en) — each also a near-zero human difference whose CI spans 0. **No statistically separated human
contrast contradicts the lexicon's sign.** (Caveat: 8/15 raw contrasts have human CIs spanning 0, so
a raw "12/15" would be computed largely over unresolved comparisons; the 6/6 figure above and the
"no separated contrast disagrees" statement are the defensible forms.)

---

## (b) P2 / P3 — agreement tables and band verdicts

Both remain **fully valid**: they are agreement statistics between raters over the tokens actually
shown, independent of which procedure selected those tokens. Note only that the token universe is
30.7% `FIRST10†` tokens (by design of the original set plan).

### P2 — lexicon vs human (token level) · **PRIMARY validation metric**

| universe | n tokens | % agreement | Cohen's κ | **band verdict** |
|---|---|---|---|---|
| **overall** | 2,558 | 84.5% | **0.396** | **fair — proxy retained as a heuristic with the limitation restated** |
| d1_sb | 591 | 84.6% | 0.434 | moderate |
| d2 | 567 | 91.2% | 0.421 | moderate |
| d3_en | 1,400 | 81.8% | 0.368 | fair |

**E2 (pre-registered: lexicon–human κ ≥ 0.4) — DOES NOT HOLD** overall (κ = 0.396, short by 0.004),
though it holds in 2 of 3 corpora separately. Per the bands fixed before any rating, the governing
verdict is **"fair"**. The result sits **on the fair/moderate boundary** and is reported as such — not
rounded up.

**Error structure** (why κ is only fair): the lexicon marks 397 tokens as cues and the human 375 —
nearly the same *quantity*, but only about half the same *tokens* (precision 0.474, recall 0.501; 209
proxy false-positives, 187 misses). The 84.5% raw agreement is carried by the large non-cue majority,
which is exactly what κ corrects for.
- **Proxy false positives** (lexicon=cue, human=no), full top-10: `cnn`(29 — URL-span hits inside
  `http://www.cnn.com`), `dear`(18), `hello`(15), `email`(10), `unlock`(9), `com`(8), `valued`(7),
  `review`(7), `customer`(7), `greetings`(5). Overwhelmingly **context-free matches**: impersonal
  salutations and imperatives in benign contexts.
- **Proxy misses** (human=cue, lexicon=no), full top-10: `attached`(12), `action`(11), `required`(10),
  `cnn`(8), `exclusive`(8), `opportunity`(7), `financial`(6), `payguard`(6), `fedex`(6),
  `digitalcoin`(6). Dominated by **impersonation and call-to-action phrasing the fixed lexicon has no
  category for**. The annotation guide's six concept families include **impersonation/authority** and
  **secrecy**; the committed lexicon's six categories (urgency, credential, financial, imperative,
  greeting, url_email) contain **neither** — a structural, not random, source of disagreement. Note
  `cnn` appears on *both* lists: it is a cue to the human in some documents (brand impersonation) and
  a false positive in the genuine CNN newsletter — precisely the context-dependence a term list cannot
  express.

### P3 — human vs LLM judge

| metric | value | n | note |
|---|---|---|---|
| token-level Cohen's κ | **0.702** | 2,468 | pre-registered ≥0.4 band ("moderate"); on the full Landis–Koch scale 0.61–0.80 is **"substantial"** |
| set-level Spearman ρ (cue-rate) | **0.818** | 251 | strong |
| set-level Spearman ρ (holistic) | **0.734** | 251 | strong |

The judge is **conservative but precise** against the human reference (precision 0.841, recall 0.662:
46 false positives vs 124 misses; 93.1% agreement) and agrees with the human far better (κ 0.702) than
the lexicon does (κ 0.396). Two independent raters applying the same concept-level guide converge; the
*term-list proxy* is the weaker link — evidence that the gap lies in the lexicon's operationalisation,
not in ambiguity of the cue construct.

---

## (c) Exploratory question — **VOID as posed**, and what actually replicates

The pre-registered exploratory question was whether the *"d3_en LIME-above-random anomaly"* replicates
in human ratings. **It cannot be answered**: the committed d3_en "LIME" plausibility was computed from
zero attributions, so the anomaly was never a property of LIME. The question is withdrawn.

What the data *does* show — a real finding under its correct name: on the AI corpus the **first 10 word
tokens** of a phishing email are markedly cue-dense, and this is confirmed by both independent raters
(human 0.225 vs RANDOM 0.060, +0.165 [0.090, 0.250], separated; judge 0.200 vs 0.045). Inspection shows
why: in d3_en the opening tokens are the LLM-generated **subject line** (e.g. *"Urgent: Verify Your
TransactFlow Account Details to Avoid Service Interruption"*), which front-loads urgency and
credential cues. **This is a property of AI-generated phishing text, not of any explainer** — and it is
a useful positional baseline to report *because* it is competitive with SHAP on d3_en (0.225 vs 0.210),
i.e. on this corpus a trivial "read the subject line" heuristic matches a real attribution method on
plausibility. (This single observation, not two: the tfidf and transformer FIRST10† rows are the same
20 sets.)

The genuine faithfulness/plausibility dissociation claimed in the earlier draft of this report **has
been withdrawn** — it rested on the void LIME arm.

---

## (d) P4 — holistic scores ("does this set explain why the email is phishing?", 0–2)

| corpus · family | SHAP | FIRST10† | IG | RANDOM |
|---|---|---|---|---|
| d1_sb · tfidf | **0.60** / 0.37 | 0.30 / 0.16 | — | 0.20 / 0.16 |
| d2 · tfidf | **0.50** / 0.17 | 0.30 / 0.28 | — | 0.25 / 0.11 |
| d3_en · tfidf | 0.70 / 0.45 | **0.85** / 0.55 | — | 0.40 / 0.25 |
| d3_en · transformer | 0.70 / 0.30 | **0.85** / 0.55 | 0.80 / 0.40 | 0.10 / 0.05 |

*(human / judge means.)* Every method **matches or beats** RANDOM in every cell for both raters, with
**one exact tie**: on d1_sb·tfidf the judge scores FIRST10† and RANDOM identically (0.158 = 3/19 in
both). SHAP beats RANDOM in 4/4 cells for both raters. Absolute levels are low (max 0.85 of 2.0): even
the best explanations were rarely judged to *clearly* explain the prediction.

---

## (e) Observations

1. **The proxy's direction is validated; its calibration is not.** The pre-registered directional
   claim survives (E1 holds 4/4; 6/6 valid sign-concordances), but token-level agreement is only
   **fair (κ = 0.396)**. The committed XAI plausibility column should be read as **ordinal, not
   absolute**: it ranks explainers reliably and must not be quoted as "the fraction of tokens a human
   would call a cue".
2. **The proxy is conservative, not inflationary — which addresses the over-reading concern
   directly.** In **9/9** method-vs-random contrasts the **human** separation exceeds the lexicon's
   (e.g. d3_en·transformer IG−RANDOM: human **+0.185** vs lexicon +0.060; SHAP−RANDOM: human +0.125
   vs lexicon +0.050). The lexicon's random baseline is inflated by context-free matches, compressing
   the gap. The committed plausibility findings therefore **understate** the human-visible separation.
3. **Two independent raters agree with each other far better than either agrees with the term list**
   (human–judge κ 0.702 vs lexicon–human κ 0.396), and the proxy's misses are structural — it has no
   **impersonation/authority** or **secrecy** category, so brand-spoofing tokens (`payguard`, `fedex`,
   `digitalcoin`) and "action required" phrasing are invisible to it, while its `greeting` category
   generates **29.2% of its false positives** (61/209 tokens whose lowercase form is in the lexicon's
   `greeting` category — `dear` 18, `hello` 15, `valued` 7, `customer` 7, `greetings` 5, `friend` 3,
   `user` 3, `welcome` 2, `client` 1) on a construct the guide
   explicitly negates ("greetings that carry no manipulative force"). A future revision should add the
   two missing categories and context-gate greetings; **the committed lexicon is not changed here** (it
   was pre-registered before results; editing it now would be outcome-tuned).
4. **SHAP leads on the legacy corpora; IG leads on the AI corpus.** Human ratings put SHAP clearly
   first on d1_sb (0.219 vs RANDOM 0.069) and d2, but on d3_en the real methods are statistically
   indistinguishable from one another (IG 0.245, SHAP 0.185, overlapping CIs) with **IG highest** —
   consistent with the committed finding that IG is the transformers' best explainer. A trivial
   first-10-tokens baseline is also competitive there (§c), so **d3_en plausibility discriminates
   weakly between methods**.
5. **A silent explainer failure survived a full quantitative XAI evaluation, a "verified PASS" review
   and a pre-registered human validation.** The LIME arm produced identically zero attributions for
   2,100 documents; its metrics looked *publishable* rather than broken (LIME appeared "most stable"
   — Jaccard 0.99 — and "least faithful", which reads as a plausible trade-off), and the giveaway
   (`stability_spearman = nan`, `lime_seed_agreement = 1.000` on all 12 rows) sat unexamined in the
   committed log. **Methodological lesson for the dissertation:** a random baseline is not enough —
   an explainer pipeline needs a *positive control* (e.g. assert non-zero, non-degenerate attributions
   and that a planted decisive token is recovered) and the regeneration anchor must cover **every**
   method, not a subset. This is now the strongest concrete argument in the thesis for treating XAI
   tooling as code under test rather than as trusted library output.

---

## Sensitivity analysis (Amendment 3) — excluding the blinding-compromised document

Method-blinding was procedurally broken for **one** of the 60 documents (`d1_sb_017547c4cef2`) by the
orchestrator's progress reporting during the judge run, *before* annotation (disclosed in
`notes/decisions.md` Amendment 3, committed before any rating existed). P1–P3 were therefore
**pre-committed** to be recomputed without it:

| metric | primary (60 docs) | excluding `d1_sb_017547c4cef2` (59 docs) |
|---|---|---|
| P2 lexicon–human κ (overall) | **0.396 — fair** | 0.409 — moderate |
| P2 κ d1_sb | 0.434 | 0.490 |
| P2 κ d2 / d3_en | 0.421 / 0.368 | unchanged |
| P3 human–judge token κ | 0.702 | 0.701 |
| P3 set ρ (cue-rate / holistic) | 0.818 / 0.734 | 0.816 / 0.733 |
| E1 (SHAP > RANDOM, 4/4) | holds | holds |
| E2 (κ ≥ 0.4) | **does not hold** | holds |

**The primary 60-document result governs**: κ = 0.396, band **"fair"**, E2 does not hold. Two further
facts keep this honest: (i) the band flip is **ordinary small-sample jitter, not a property of that
document** — the full leave-one-document-out κ ranges 0.372–0.418 and **10 of 60** single-doc
exclusions push κ above 0.40, with the compromised document only 4th largest; (ii) removing a document
that was compromised *toward* the annotator could never be the headline. P3 and E1 are essentially
untouched. Honest summary: **"fair, at the fair/moderate boundary"**.

---

## Limitations (disclosed)

1. **The "LIME" arm is void** (Amendment 4, above): an upstream inverted `split_expression` zeroed
   every LIME attribution project-wide, so 80/260 sets and 30.7% of token judgments measure a
   first-10-tokens positional baseline. All LIME-specific conclusions are withdrawn here; the
   committed XAI phase's LIME rows and RQ3 LIME claims are affected and **not** corrected by this
   phase. SHAP, IG, RANDOM, E1, P2 and P3 are unaffected.
2. **Single annotator, procedural blinding only.** One rater (the author); no inter-human agreement is
   computable, and blinding was procedural (A/B/C/D labels + sealed key), not third-party enforced.
   Pre-registered in §C. The LLM judge partially substitutes for a second human (P3), but see 6.
3. **Method-blinding was broken for 1 of 60 documents** (`d1_sb_017547c4cef2`) by the orchestrator's
   progress reporting before annotation; the sensitivity analysis above reports P1–P3 with and without
   it, and 59/60 documents retain intact blinding.
4. **The 9 LLM-judge refusals are content-correlated** — all fall on 3 obfuscated pharma/adult-spam
   documents (`d1_sb_0c55e64c8f49`, `d2_0b48feee9474`, `d2_22f7325368b4`) that the model declined on
   content grounds — so **P3 coverage is slightly biased away from that content type** (judge coverage
   d1_sb 19/20 docs, d2 18/20, d3_en 20/20). Refusals were whole-document, so P3 pairing stays balanced.
5. **Judge ratings are a single stochastic draw at the model default** (temperature is deprecated for
   `claude-opus-4-8`; Amendment 2). Re-querying could differ, but every reported analysis reproduces
   exactly from the committed `judge_ratings.csv` + `judge_raw.jsonl`.
6. **Judge provenance.** The judge shares a model family with the tooling that authored the cue
   lexicon, so **the human annotation is the independent reference and P2 is the primary validation
   metric**; P3 is triangulation, not confirmation.
7. **P2 independence is imperfect (conservative direction).** 22 of the 206 lexicon terms appear
   verbatim in the guide's prose as cue *examples* (`account`, `verify`-family, `click`, `login`, …).
   The pre-registration's commitment — not reproducing the term *list* — holds, but naming some terms
   can only **inflate** lexicon–human κ, so the "fair" verdict is if anything generous to the proxy.
8. **Scope and power.** 60 documents, phishing-predicted only, top-10 tokens only; 20 docs per cell,
   hence wide CIs (8/15 raw contrasts unresolved). No ham-predicted explanations were rated. The
   d3_en sample is a **near-census** of its eligible pool (23 docs: 15 of 16 TP-eligible, 5 of 7
   FP-eligible), so d3_en cells are effectively the whole tfidf∩transformer intersection rather than a
   random sample; d1_sb/d2 drew from pools of 106/118.
9. **Terminology.** The per-cell CIs are ordinary bootstrap CIs (the pre-registration's word "paired"
   applies meaningfully only to the *differences*, which do share a resample index).

---

## Artifacts

`results/hitl/` — `human_ratings.csv` (2,558 rows, committed untouched at `51c30b6`, sha256
`4568450…33d0e`), `judge_ratings.csv` (2,468 rows) + `judge_raw.jsonl` (260 verbatim records, resolved
model id + timestamps), `key.json`, `annotate.html`, `blind_sheets.csv`, `selected_ids.json`,
`hitl_token_level.csv`, `hitl_set_level.csv`, `hitl_metrics.json`.
Figure: `figures/hitl/hitl_cue_rates.(png|pdf)`.
Code: `src/experiments/d10_{anchor,select,package,judge,analyze,figures}.py`.

**Review (Part E).** Reviewer + independent confirmer, four surfaces: **S1 PASS** (all 60 docs ⊂
committed TEST `explained_ids`, `pred==1` in every source unit, TEST-split membership re-derived,
counts exact); **S2 PASS** (pre-registration `8ef0937` precedes the first judge call by 6 days and the
ratings commit; §A–F byte-identical between `8ef0937` and HEAD; amendments are separate appended
notes); **S3 PASS** (working-tree `human_ratings.csv` byte-identical to `51c30b6`; every P1–P4 number,
all CIs, the error structure and the sensitivity values independently recomputed from raw artifacts
with two separate implementations — zero mismatches); **S4 PASS after correction** (definitions, bands
and E1/E2 match §F verbatim; the primary verdict leads with κ = 0.396 "fair"/E2-fails, not the
sensitivity value). The reviewers' substantive findings — the void LIME arm, the P4 tie, the P3 band
label, the truncated illustrative lists, the d3_en double-count — are all incorporated above.
