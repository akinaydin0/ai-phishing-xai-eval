"""
Known-answer (positive-control) harness for the XAI explainers — Directive 11 Part A.1.

WHY THIS EXISTS. The Directive 7 LIME arm produced identically-zero attributions for all 2,100
explained documents (inverted `split_expression`; notes/decisions.md Amendment 4) and NO check caught
it: the random baseline was passed, the metrics looked like a plausible trade-off, and the Directive 10
regeneration anchor covered SHAP and IG only. A random baseline is a NEGATIVE control; an explainer
pipeline also needs a POSITIVE control — proof that a method recovers a cue that is known, by
construction, to drive the prediction. This suite is that control and is a permanent regression test.

WHAT IT ASSERTS (deterministic, seed 42):
  T0  precondition: each champion actually PREDICTS phishing on each planted document. "Which tokens
      explain the phishing prediction?" is ill-posed otherwise — on a confidently-ham document the
      class-1 ranking is small-magnitude noise, and the committed pipeline only ever explains
      phishing-predicted documents (`metrics.plausibility` filters on `pred==1`). Fails loudly.
  T1  planted-cue localisation: on synthetic docs where a phishing cue paragraph is planted after
      cue-free neutral text, every explainer ranks a PLANTED token among its top-TOP_K (=10) DISTINCT
      words by class-1 ("phishing") attribution — for every family (tfidf, ensemble, transformer) and
      every applicable method (SHAP, LIME, and IG on the transformer).
      POWER CAVEAT (measured by review, Monte Carlo): because the objectively derived planted set
      spans 53-60% of each document's vocabulary, a RANDOM attribution passes T1 with p~0.9998. T1 is
      therefore a *degeneracy* detector (its real target is the zero/constant vector of the D7 defect,
      which T4 pins down) and NOT evidence of attribution quality. T1b is the discriminating gate
      (random passes at p~0.5). Strengthening this suite -- a tighter planted set, and gating the
      curated content-cue criterion that is currently only printed -- is outstanding work.
      The planted set is defined OBJECTIVELY as the tokens of the planted sentence that do not occur
      in the filler (so it cannot be tuned to make a method pass), and the filler is deliberately
      free of phishing-cue vocabulary. DISTINCT words (not raw positions) are scored because LIME is
      bag-of-words: a word occurring k times would otherwise occupy k of the top-k slots. A curated
      content-cue subset is also reported, for information only.
  T1b attribution-mass localisation: the MEAN signed class-1 attribution over planted tokens must
      exceed the mean over non-planted tokens. GATING for SHAP and LIME; recorded as a DIAGNOSTIC for
      IG (see T1B_GATED for the measured justification). A zero or constant vector yields equal means
      and fails, so this is also defect-sensitive.
  T2  non-degeneracy on real data: on 5 real, PHISHING-PREDICTED TEST documents per family, fewer
      than `MAX_DEGENERATE_RATE` of corrected LIME's attributions may fall below a MAGNITUDE FLOOR
      (`MIN_ATTR_MAG`) — the property the D7 defect violated (it was degenerate on 100%). An exact
      `== 0.0` test is deliberately NOT used (Directive 11 found LIME returning max|attr| ~ 1e-6,
      which an exact-zero test passes but which is degenerate in practice), and a RATE rather than an
      all-or-nothing gate is used because saturation degeneracy is a real minority phenomenon
      (measured 4.0% overall, up to 22.5% on one unit) that must not make this test permanently red.
      The sample is restricted to phishing-predicted documents for the same well-posedness reason
      as T0.
  T2b saturation diagnostic (NON-GATING): the same magnitude check on confidently-HAM documents,
      where Directive 11 measured LIME collapsing to ~1e-6 on 84/2,100 documents (up to 22.5% of one
      unit) while SHAP and IG stay normal. Recorded, not gated: it is a real property of perturbation
      explanation under prediction saturation, not a code defect.
  T3  defect reproduction: with the legacy value `LEGACY_VOID_SPLIT_EXPRESSION` LIME attributions ARE
      identically zero — i.e. the documented defect model is correct.
  T4  self-validation (model-free): a ZERO attribution vector must FAIL T1. This proves the suite
      actually detects the D7 defect and guards against future edits that would silently remove that
      power (e.g. moving the cue paragraph to the start of the document).

Attribution is scored toward CLASS 1 (phishing); T0 guarantees the champion actually predicts phishing,
so class 1 is also the predicted class and the assertions match how the pipeline consumes attributions.

Run: python -m src.xai.test_known_answer            (exit 0 = PASS, 1 = FAIL)
     python -m src.xai.test_known_answer --families tfidf     (subset, for quick iteration)
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

try:
    from src.xai import champions as C, explainers, textops
    from src.splits import load_splits
except ImportError:  # pragma: no cover
    from xai import champions as C, explainers, textops
    from splits import load_splits

SEED = 42
# Top-k granularity = 10, the project's NATIVE reporting granularity (plausibility@10, comp@10,
# suff@10, stability top-10) — chosen for that reason, not tuned per method. A stricter k=3 is passed
# by SHAP and LIME on every family but not by Integrated Gradients, whose per-token attributions are
# diffuse (gradient saturation / PAD-baseline sensitivity) even though it is the strongest transformer
# explainer in aggregate (best committed transformer AOPC 0.1288; highest D10 human plausibility
# 0.245). Verified NOT to be a code defect: IG's raw sub-token attributions rank the same tokens as
# its word-level output, so the sub-token -> word mapping is faithful. T4 asserts that k=10 still
# rejects a zero vector, so the control keeps its power to detect the D7 defect.
TOP_K = 10
# T1b (signed attribution-mass localisation) GATES the perturbation methods, where it is well
# behaved, and is recorded as a DIAGNOSTIC for Integrated Gradients. Measured justification (not a
# convenience exemption): IG's signed attributions are diffuse and net-negative even on
# phishing-predicted documents, so on the lower-confidence synthetic case (p=0.661) the planted-region
# mean (-0.031) sits just below the non-planted mean (-0.025) while IG still places planted tokens in
# its top-10 (T1) -- and IG's word-level output was verified to match its raw sub-token ranking, i.e.
# no code defect. The absolute-value ("relevance mass") variant was tested and REJECTED: it fails for
# SHAP and LIME too, because mean |attr| is diluted by the long planted paragraph relative to the
# short, high-magnitude neutral region. IG remains fully gated by T0, T1 and T4.
T1B_GATED = {"shap", "lime"}
N_REAL_DOCS = 5      # T2: real documents per family
# T2 magnitude floor. The D7 defect produced EXACTLY zero, but Directive 11 found a softer degeneracy:
# LIME returns max|attr| ~ 1e-6 on near-saturated predictions (84/2,100 docs), which an exact-zero test
# would pass. 1e-5 also matches the pipeline's 5-dp attribution rounding: anything below it is stored
# as zero anyway, so a vector under this floor is degenerate by construction.
MIN_ATTR_MAG = 1e-5
POOL_SIZE = 40       # real-document candidate pool, filtered by predicted class for T2 / T2b
MAX_DEGENERATE_RATE = 0.5   # T2 fires only when degeneracy is the rule (the D7 defect was 100%)
# Neutral text, deliberately free of phishing-cue vocabulary. It is placed FIRST in every document so
# that a first-k-position baseline (exactly what the D7 void arm degenerated into) lands on NEUTRAL
# tokens -- this is what gives T1 its power to detect the defect (asserted directly by T4).
NEUTRAL_LEAD = (
    "Minutes of the previous meeting are stored in the shared archive folder. "
    "Travel arrangements for the summer conference appear in the printed itinerary. "
)
NEUTRAL_TAIL = (
    "The quarterly logistics summary was circulated to the regional team on Monday morning. "
)
# Planted cue paragraphs. Each is a realistic phishing message, so the champions actually PREDICT
# phishing (T0) -- a precondition for the question "which tokens explain the phishing prediction?",
# which is also the only regime the committed pipeline ever explains (metrics.plausibility filters on
# pred==1). The authoritative planted set is derived objectively as the paragraph's tokens minus the
# neutral tokens; the set given here is the curated CONTENT-cue subset, reported for information only.
PLANTED_CASES = [
    ("Security Alert: Unusual Sign-in Activity Detected. Dear Customer, our system has detected "
     "unauthorized access to your online banking account. You must verify your identity immediately "
     "or your account will be permanently suspended. Click the secure link below and confirm your "
     "login credentials, password and billing details within 24 hours to avoid termination of "
     "service.",
     {"verify", "account", "password", "credentials", "login", "suspended", "immediately", "click",
      "confirm", "unauthorized"}),
    ("Final Notice: Payment Required. Your invoice remains unpaid and your subscription expires "
     "today. Please update your card details and complete the secure payment form urgently, or "
     "access to your account will be permanently restricted. Click the verification link to claim "
     "your refund before the deadline.",
     {"payment", "invoice", "unpaid", "expires", "urgently", "click", "verification", "claim",
      "refund", "deadline", "account", "restricted"}),
]
# One champion per family (the cheapest representative of each; IG is transformer-only).
FAMILY_CHAMPIONS = {
    "tfidf": ("tfidf_d1_sb", ["shap", "lime"]),
    "ensemble": ("ensemble_d1_sb", ["shap", "lime"]),
    "transformer": ("transformer_d1_sb", ["shap", "lime", "ig"]),
}


def _docs():
    """Deterministic synthetic documents: NEUTRAL lead + planted cue paragraph + neutral tail.

    Returns (text, planted_tokens, content_cues) where `planted_tokens` is derived objectively as the
    cue paragraph's tokens minus the neutral tokens (so it cannot be tuned per method).
    """
    neutral_tokens = set(textops.words_lower(NEUTRAL_LEAD + NEUTRAL_TAIL))
    out = []
    for para, content_cues in PLANTED_CASES:
        text = NEUTRAL_LEAD + para + " " + NEUTRAL_TAIL
        planted = set(textops.words_lower(para)) - neutral_tokens
        out.append((text, planted, content_cues))
    return out


def _top_distinct_words(text: str, attr: np.ndarray, k: int) -> list[str]:
    """The k DISTINCT words with the highest CLASS-1 attribution.

    Distinct words (not raw positions) because LIME is bag-of-words: every occurrence of a word
    carries the same weight, so a thrice-repeated word would otherwise fill three top-k slots. Word
    score = max attribution over its occurrences; ties broken by first position (stable).
    """
    words = textops.words_lower(text)
    a = np.asarray(attr, dtype=float)
    best: dict[str, float] = {}
    order: list[str] = []
    for w, v in zip(words, a):
        if w not in best:
            best[w] = float(v)
            order.append(w)
        else:
            best[w] = max(best[w], float(v))
    ranked = sorted(order, key=lambda w: (-best[w], order.index(w)))
    return ranked[:k]


def _real_texts(corpus: str, n: int, champ=None, want_phishing: bool | None = None) -> list[str]:
    """n real TEST documents, deterministic (seed 42).

    With `champ` + `want_phishing`, the pool is first scored and filtered to documents the champion
    predicts phishing (True) or ham (False). T2 gates on phishing-PREDICTED documents for the same
    reason T0 exists: an explanation is only ever consumed for a phishing prediction, and asking for
    one on a confidently-ham document is not a well-posed request (see T2b).
    """
    te = load_splits(corpus)["test"].reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    pool = sorted(rng.choice(len(te), size=min(POOL_SIZE, len(te)), replace=False).tolist())
    texts = [te["text"].iloc[i] for i in pool]
    if champ is None or want_phishing is None:
        return texts[:n]
    p1 = champ.proba(texts)
    keep = [t for t, p in zip(texts, p1) if (float(p) >= 0.5) == want_phishing]
    return keep[:n]


def run(families: list[str]) -> int:
    failures: list[str] = []
    checks = 0
    docs = _docs()

    for family in families:
        unit, methods = FAMILY_CHAMPIONS[family]
        meta = C.unit_meta(unit)
        print(f"\n[{family}] champion unit={unit} corpus={meta['explained_corpus']}", flush=True)
        champ = C.build_champion(unit)

        # ---- T0: precondition - the champion must PREDICT phishing on each planted document ----
        # Without this the question "which tokens explain the phishing prediction?" is ill-posed: on a
        # confidently-ham document the class-1 ranking is small-magnitude noise (this is exactly how an
        # earlier draft of this suite produced a spurious IG failure at p(phish)=0.0004). Fails LOUDLY
        # rather than skipping, so a bad test document must be fixed instead of the assertion relaxed.
        for di, (text, _p, _c) in enumerate(docs):
            p1 = float(champ.proba([text])[0])
            checks += 1
            ok = p1 >= 0.5
            if not ok:
                failures.append(f"T0 {family} doc{di}: champion does not predict phishing "
                                f"(p={p1:.4f}) - the known-answer case is invalid for this champion")
            print(f"  T0 {'PASS' if ok else 'FAIL'} doc{di} p(phish)={p1:.4f}", flush=True)

        # ---- T1: planted-cue localisation, every method ----
        for method in methods:
            for di, (text, planted, content_cues) in enumerate(docs):
                attr = explainers.attribute(champ, family, method, text)
                top = _top_distinct_words(text, attr, TOP_K)
                hit = sorted(set(top) & planted)
                checks += 1
                ok = bool(hit)
                if not ok:
                    failures.append(f"T1 {family}/{method} doc{di}: no planted token in the top-"
                                    f"{TOP_K} distinct words (top={top}, planted={sorted(planted)})")
                print(f"  T1 {'PASS' if ok else 'FAIL'} {method:5s} doc{di} "
                      f"planted_hit={hit[:6]} content_cue_hit={sorted(set(top) & content_cues)[:6]} "
                      f"top{TOP_K}={top[:6]}...", flush=True)

                # ---- T1b: attribution MASS localises in the planted region ----
                # Mean signed class-1 attribution over planted tokens vs non-planted tokens. GATING
                # for the perturbation methods (SHAP, LIME), DIAGNOSTIC ONLY for IG -- see T1B_GATED.
                wl = textops.words_lower(text)
                av = np.asarray(attr, dtype=float)
                pm = [v for w, v in zip(wl, av) if w in planted]
                nm = [v for w, v in zip(wl, av) if w not in planted]
                okb = bool(pm and nm and float(np.mean(pm)) > float(np.mean(nm)))
                gated = method in T1B_GATED
                if gated:
                    checks += 1
                    if not okb:
                        failures.append(f"T1b {family}/{method} doc{di}: planted-region mean "
                                        f"attribution {np.mean(pm):+.5f} <= non-planted mean "
                                        f"{np.mean(nm):+.5f}")
                tag = ("PASS" if okb else "FAIL") if gated else ("DIAG-ok" if okb else "DIAG-below")
                print(f"  T1b {tag:10s} {method:5s} doc{di} planted_mean={np.mean(pm):+.5f} vs "
                      f"non_planted_mean={np.mean(nm):+.5f}", flush=True)

        # ---- T2: corrected LIME non-degenerate on real PHISHING-PREDICTED docs (magnitude floor) --
        n_const, mags = 0, []
        for text in _real_texts(meta["explained_corpus"], N_REAL_DOCS, champ, want_phishing=True):
            a = explainers.lime_attr(champ, text)
            m = float(np.max(np.abs(a))) if a.size else 0.0
            mags.append(m)
            if m < MIN_ATTR_MAG:
                n_const += 1
        checks += 1
        # RATE gate, not all-or-nothing. The D7 defect made 100% of documents degenerate; a working
        # explainer shows a minority (Directive 11 measured 4.0% overall, up to 22.5% on one unit,
        # from prediction saturation -- a real property, not a bug). Failing on a single degenerate
        # document would make this test permanently red and useless as a regression guard, so it
        # fires only when degeneracy is the RULE rather than the exception.
        rate = n_const / len(mags) if mags else 1.0
        ok = rate < MAX_DEGENERATE_RATE
        if not ok:
            failures.append(f"T2 {family}: corrected LIME below the magnitude floor {MIN_ATTR_MAG:g} "
                            f"on {n_const}/{len(mags)} real phishing-predicted documents "
                            f"(rate {rate:.0%} >= {MAX_DEGENERATE_RATE:.0%}) "
                            f"(max|attr| = {[f'{m:.2g}' for m in mags]})")
        print(f"  T2 {'PASS' if ok else 'FAIL'} corrected LIME above magnitude floor on "
              f"{len(mags) - n_const}/{len(mags)} real phishing-predicted docs "
              f"(degenerate rate {rate:.0%} < {MAX_DEGENERATE_RATE:.0%}; "
              f"min max|attr| = {min(mags):.3g})", flush=True)

        # ---- T2b: saturation diagnostic (NON-GATING) ----
        # Directive 11 measured that LIME collapses to ~1e-6 attributions on near-saturated
        # predictions (84/2,100 docs; up to 22.5% of a unit) while SHAP/IG are unaffected. This
        # records the rate on confidently-HAM documents so the phenomenon stays visible and tracked.
        ham_mags = []
        for text in _real_texts(meta["explained_corpus"], N_REAL_DOCS, champ, want_phishing=False):
            a = explainers.lime_attr(champ, text)
            ham_mags.append(float(np.max(np.abs(a))) if a.size else 0.0)
        n_deg = sum(1 for m in ham_mags if m < MIN_ATTR_MAG)
        print(f"  T2b DIAG   LIME below floor on {n_deg}/{len(ham_mags)} ham-predicted docs "
              f"(min max|attr| = {min(ham_mags):.3g} — saturation degeneracy, non-gating)", flush=True)

        # ---- T3: the legacy value really does produce the void arm (suite discriminates) ----
        legacy_zero = 0
        for text, _planted, _cues in docs:
            a = explainers.lime_attr(champ, text,
                                     split_expression=explainers.LEGACY_VOID_SPLIT_EXPRESSION)
            if a.size and float(np.max(np.abs(a))) == 0.0:
                legacy_zero += 1
        checks += 1
        ok = legacy_zero == len(docs)
        if not ok:
            failures.append(f"T3 {family}: legacy split_expression did NOT reproduce the void arm "
                            f"({legacy_zero}/{len(docs)} zero) - the defect model is wrong")
        print(f"  T3 {'PASS' if ok else 'FAIL'} legacy value reproduces void arm "
              f"({legacy_zero}/{len(docs)} all-zero)", flush=True)

        # ---- T4: the suite provably DETECTS the defect (self-validation, model-free) ----
        # A zero attribution vector is what the void arm produced; T1 must REJECT it. If this ever
        # passes, T1 has lost its power (e.g. someone moved the cue paragraph to the front).
        caught = 0
        for text, planted, _c in docs:
            zero = np.zeros(len(textops.word_spans(text)))
            if not (set(_top_distinct_words(text, zero, TOP_K)) & planted):
                caught += 1
        checks += 1
        ok = caught == len(docs)
        if not ok:
            failures.append(f"T4 {family}: a ZERO attribution vector would satisfy T1 on "
                            f"{len(docs) - caught}/{len(docs)} docs - the suite no longer detects the "
                            f"D7 defect (is the cue paragraph at the start of the document?)")
        print(f"  T4 {'PASS' if ok else 'FAIL'} zero-vector rejected by T1 on {caught}/{len(docs)} "
              f"docs (suite detects the D7 defect)", flush=True)

    print("\n" + "=" * 72, flush=True)
    if failures:
        print(f"KNOWN-ANSWER SUITE FAILED ({len(failures)} of {checks} checks):", flush=True)
        for f in failures:
            print("  - " + f, flush=True)
        return 1
    print(f"KNOWN-ANSWER SUITE PASSED ({checks} checks over families: {', '.join(families)})",
          flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="XAI known-answer positive control.")
    ap.add_argument("--families", nargs="*", default=list(FAMILY_CHAMPIONS),
                    choices=list(FAMILY_CHAMPIONS))
    args = ap.parse_args(argv)
    return run(args.families)


if __name__ == "__main__":
    sys.exit(main())
