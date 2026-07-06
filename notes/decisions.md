# Decision Log

Append-only record of methodological decisions. Newest at the bottom. Each entry:
**date · decision · rationale · links.** Created 2026-07-06, consolidating earlier
project decisions.

---

## 2026-07-05 · D3 binary label mapping (id: C2)

**Decision.** For PhishFuzzer (D3), collapse the three-class label
(Phishing / Spam / Valid) to binary: **Phishing + Spam → 1, Valid → 0.**

**Rationale.** Matches the binary convention already used by D1 and D2 (where label 1
covers phishing *and* spam / unsolicited mail, label 0 covers legitimate/ham), so the
three corpora share one label semantics for cross-corpus training and evaluation. The raw
three-class label is retained alongside the binary one (`type_3class`) for any finer-grained
analysis.

**Links.** `src/data_loader.py` (`load_phishfuzzer`, `_PHISHFUZZER_BINARY_MAP`); `CLAUDE.md`
(Datasets → D3).

---

## 2026-07-05 · D2 full dedup + pinned determinism fingerprints

**Decision.** Deduplicate every corpus fully before any split (non-negotiable, prevents
train/test leakage). Do **not** retain duplicates to preserve a class ratio. Replace the
generic "±2 pp class-balance shift" gate — which trips on D2 — with **exact determinism
fingerprints** (`CLEANING_EXPECTED`) pinning each frozen corpus's final row count and
per-class counts; any drift hard-fails the pipeline and forces a conscious re-pin recorded
in the commit message. Unpinned corpora fall back to the ±2 pp shift as a WARN.

**Rationale.** D2's duplicates are 74 % spam (spam-campaign repetition), so mandatory full
dedup legitimately shifts its spam share 50.93 % → 47.28 % (−3.65 pp), exceeding ±2 pp.
That is an expected data property, not a defect; keeping duplicate spam to hold a ratio
would re-introduce leakage. Two duplicate definitions are kept distinct so figures never look
contradictory: 9.6 % exact-raw (interim figure) vs 11.64 % after normalisation.

**Links.** `notes/cleaning_report.md`; `src/cleaning.py` (`CLEANING_EXPECTED`,
`run_sanity_asserts`); `CLAUDE.md` (Known data facts).

---

## 2026-07-06 · D1 dual text representation

**Decision.** EDA v2 reports D1 in **both** representations, with **subject+body as
primary**:
- **subject+body** (natural text, 51,839 rows after cleaning) drives all cross-corpus and
  lexical statistics — the three-way length figure, pairwise Jaccard, and D1 per-class
  pattern means — because every interim reference number and both D2/D3 are natural text.
  Cached as a first-class pinned corpus, `d1_decontaminated_sb`.
- **text_combined** (the Kaggle-preprocessed field, 52,704 rows) is reported only as a
  secondary heterogeneity comparison (medians ~half the length: legit 118 / phish 45 vs
  199 / 62).

The **modelling** representation is **deferred to the baselines directive** (working
hypothesis: subject+body, for representation consistency with D2/D3 and because Pipeline A's
stop-word ablation is impossible on the pre-stripped text_combined field). **Supervisor-meeting
talking point.**

**Rationale.** The cached `clean_d1_decontaminated` pkl holds `text_combined`, which the
Kaggle author already stripped of stop-words/punctuation (~half the tokens). Measuring D1
length on it would make "modern legit ≈ Enron ham" appear broken (118 vs 169) purely from
the representation change. Reconstructed cleaned subject+body reproduces the interim medians
199 / 62 **exactly**, confirming the interim findings hold like-for-like. The two D1 universes
(52,704 vs 51,839; an 865-row delta = 622 more empties + 243 more duplicates in the less-curated
per-source CSVs) must never appear in one table without labels.

**Links.** `notes/eda_v2_summary.md` (§5); `src/eda_v2.py`; `src/cleaning.py`
(`d1_decontaminated_sb`, `CLEANING_EXPECTED`).

---

## 2026-07-06 · D1 modelling representation = subject+body (evidence-backed)

**Decision.** Use **subject+body** (`d1_decontaminated_sb`) as the D1 modelling
representation. This upgrades the [D1 dual text representation] working hypothesis to a
decision, now backed by the baseline ablation. **Still a supervisor-meeting talking point.**

**Evidence (representation ablation, Run B vs Run A; frozen splits, seed 42).** In-corpus
TF-IDF + Logistic Regression:
- subject+body (`d1_decontaminated_sb`, test n=7,776): **F1 0.9906**, ROC-AUC 0.9995.
- text_combined (`d1_decontaminated`, test n=7,906): **F1 0.9927**, ROC-AUC 0.9992.

The gap is **0.21 pp F1** (text_combined marginally higher — its stop-word/punctuation
stripping mildly denoises the bag-of-words). So the representation choice costs ~nothing in
accuracy.

**Rationale.** subject+body wins on the two grounds that matter downstream, at negligible
accuracy cost: (a) representation consistency with D2/D3 (both natural text), which is
required for clean cross-corpus evaluation — a primary reported outcome; and (b) it
preserves stop-words/punctuation, so Pipeline A's stop-word-removal ablation and
URL/email placeholder substitution are actually possible (both are impossible on the
pre-stripped text_combined field).

**Links.** `notes/baseline_summary.md` (§Results Run A/B, Observation 4);
`results/results.csv`; `src/experiments/run_baselines.py`.
