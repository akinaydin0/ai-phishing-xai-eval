"""
D9 Part C — the seven pre-registered comparisons + 12 single-AUC CIs.

Paired stratified bootstrap (B=10,000, resample within class, seed 42, identical resamples for
both models of a pair) for Δ=AUC_a−AUC_b and for each single AUC; McNemar exact two-sided binomial
at the 0.5 operating point (n01/n10, raw p), Holm-adjusted across the 7-test family. Reads the
cached, reproduction-asserted score vectors from d9_scores. Writes results/stats/stats_summary.csv.

Run: python -m src.experiments.d9_scores && python -m src.experiments.d9_stats
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

try:
    from src.experiments import d9_scores
except ImportError:  # pragma: no cover
    import d9_scores

_ROOT = Path(__file__).resolve().parent.parent.parent
OUT = _ROOT / "results" / "stats" / "stats_summary.csv"
B = 10_000
SEED = 42

# comparison_id -> (score_id_a, score_id_b, test_corpus, label_a, label_b)
# model_a/model_b order is VERBATIM from the pre-registration table (decisions.md 2026-07-17):
# C1a/C1b/C1c list the classical champion as a, transformer as b (Δ = classical − transformer);
# C2/C3 list the 512 transformer as a; C4 the classical cross champion as a; C5 the 512 transformer as a.
COMPARISONS = {
    "C1a": ("svc_d1sb@d1sb", "distilbert256_d1sb@d1sb", "d1_sb", "LinearSVC", "DistilBERT-256"),
    "C1b": ("nb_d2@d2", "distilbert256_d2@d2", "d2", "MultinomialNB", "DistilBERT-256"),
    "C1c": ("svc_d3en@d3en", "roberta256_d3en@d3en", "d3_en", "LinearSVC", "RoBERTa-256"),
    "C2":  ("roberta512_d3en@d2", "roberta256_d3en@d2", "d2", "RoBERTa-512(d3en->d2)", "RoBERTa-256(d3en->d2)"),
    "C3":  ("roberta512_d3en@d2", "svc_d3en@d2", "d2", "RoBERTa-512(d3en->d2)", "LinearSVC-cross"),
    "C4":  ("lr_d2@d3en", "distilbert256_d2@d3en", "d3_en", "LogReg-cross(d2->d3en)", "DistilBERT-256(d2->d3en)"),
    "C5":  ("distilbert512_d2@d3en", "distilbert256_d2@d3en", "d3_en", "DistilBERT-512(d2->d3en)", "DistilBERT-256(d2->d3en)"),
}


def _resamples(y: np.ndarray, seed: int = SEED, b: int = B):
    """b within-class bootstrap index arrays (positive & negative counts preserved)."""
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    return [np.concatenate([rng.choice(pos, len(pos), replace=True),
                            rng.choice(neg, len(neg), replace=True)]) for _ in range(b)]


def _auc_boot(scores, y, resamples):
    return np.array([roc_auc_score(y[ix], scores[ix]) for ix in resamples])


def _mcnemar(pred_a, pred_b, y):
    a_correct = pred_a == y
    b_correct = pred_b == y
    n01 = int(np.sum(~a_correct & b_correct))   # a wrong, b right
    n10 = int(np.sum(a_correct & ~b_correct))   # a right, b wrong
    n = n01 + n10
    p = binomtest(min(n01, n10), n, 0.5, alternative="two-sided").pvalue if n > 0 else 1.0
    return n01, n10, float(p)


def _holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for rank, (cid, p) in enumerate(items):
        val = min(1.0, (m - rank) * p)
        running = max(running, val)   # Holm is monotone non-decreasing in sorted order
        adj[cid] = round(running, 6)
    return adj


def run():
    # ---- gather cached scores, group by test corpus, build shared resamples (seed 42) ----
    scores_of, y_of, tc_of = {}, {}, {}
    for sid in d9_scores.SPECS:
        s, y, _ = d9_scores.load(sid)
        scores_of[sid], y_of[sid] = s, y
        tc_of[sid] = d9_scores.SPECS[sid][2]
    resamples_by_tc = {}
    for tc in set(tc_of.values()):
        ref = next(sid for sid in tc_of if tc_of[sid] == tc)
        resamples_by_tc[tc] = _resamples(y_of[ref])
        # assert every score vector on this test corpus shares identical labels/order
        for sid in tc_of:
            if tc_of[sid] == tc:
                assert np.array_equal(y_of[sid], y_of[ref]), f"label misalignment on {tc}: {sid}"

    auc_boot = {sid: _auc_boot(scores_of[sid], y_of[sid], resamples_by_tc[tc_of[sid]])
                for sid in d9_scores.SPECS}
    single_ci = {sid: (round(float(roc_auc_score(y_of[sid], scores_of[sid])), 4),
                       round(float(np.percentile(auc_boot[sid], 2.5)), 4),
                       round(float(np.percentile(auc_boot[sid], 97.5)), 4))
                 for sid in d9_scores.SPECS}

    # ---- seven comparisons ----
    raw_p = {}
    rows = []
    for cid, (a, b, tc, la, lb) in COMPARISONS.items():
        ya = y_of[a]
        delta = auc_boot[a] - auc_boot[b]                      # identical resamples → paired
        auc_a = round(float(roc_auc_score(ya, scores_of[a])), 4)
        auc_b = round(float(roc_auc_score(ya, scores_of[b])), 4)
        ci_lo = round(float(np.percentile(delta, 2.5)), 4)
        ci_hi = round(float(np.percentile(delta, 97.5)), 4)
        n01, n10, p = _mcnemar((scores_of[a] >= 0.5).astype(int),
                               (scores_of[b] >= 0.5).astype(int), ya)
        raw_p[cid] = p
        rows.append({"comparison_id": cid, "test_corpus": tc, "model_a": la, "model_b": lb,
                     "auc_a": auc_a, "auc_b": auc_b, "delta": round(auc_a - auc_b, 4),
                     "ci_low": ci_lo, "ci_high": ci_hi, "n01": n01, "n10": n10,
                     "mcnemar_p_raw": round(p, 6), "B": B, "seed": SEED})
    holm = _holm(raw_p)
    for r in rows:
        r["mcnemar_p_holm"] = holm[r["comparison_id"]]

    # ---- write: comparisons first, then the 12 single-AUC CI rows ----
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fields = ["comparison_id", "test_corpus", "model_a", "model_b", "auc_a", "auc_b", "delta",
              "ci_low", "ci_high", "n01", "n10", "mcnemar_p_raw", "mcnemar_p_holm", "B", "seed"]
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
        for sid, (auc, lo, hi) in single_ci.items():
            w.writerow({"comparison_id": f"AUC_CI:{sid}", "test_corpus": tc_of[sid],
                        "model_a": sid, "model_b": "", "auc_a": auc, "auc_b": "",
                        "delta": "", "ci_low": lo, "ci_high": hi, "n01": "", "n10": "",
                        "mcnemar_p_raw": "", "mcnemar_p_holm": "", "B": B, "seed": SEED})

    print("=== D9 seven comparisons (Δ=auc_a−auc_b, 95% paired-bootstrap CI, McNemar+Holm) ===")
    for r in rows:
        sep = "separated" if (r["ci_low"] > 0 or r["ci_high"] < 0) else "CI INCLUDES 0"
        print(f"  {r['comparison_id']:4s} {r['model_a'][:22]:22s} vs {r['model_b'][:22]:22s} | "
              f"Δ={r['delta']:+.4f} [{r['ci_low']:+.4f},{r['ci_high']:+.4f}] {sep:14s} | "
              f"McNemar n01/n10={r['n01']}/{r['n10']} p_raw={r['mcnemar_p_raw']:.4g} "
              f"p_holm={r['mcnemar_p_holm']:.4g}")
    print(f"\nwrote {OUT}")
    return rows, single_ci


if __name__ == "__main__":
    run()
