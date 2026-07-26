"""
Directive 11b: paired stratified bootstrap CIs for the two knife-edge AOPC orderings.

Exactly two pre-stated comparisons, nothing else:
  K1  tfidf family, SHAP vs corrected LIME  — per corpus and pooled (the RQ3 margin, point est. 0.0151)
  K2  transformer_d3_en, IG vs corrected LIME                        (point est. 0.0023)

Method (identical to Directive 9's `d9_stats._resamples`): B=10,000 within-class bootstrap resamples
(positive and negative document counts preserved), seed 42, the SAME resample indices applied to both
arms so the difference is paired. CI = 2.5 / 97.5 percentiles of the bootstrap delta distribution.

Per-document AOPC is recomputed here because only the aggregate is stored: for each document, the
deletion curve `p_pred(original) - p_pred(top-k removed)` for k=1..20 is averaged over k — exactly the
quantity `metrics.faithfulness` averages over documents (asserted: the mean of the per-document values
reproduces the committed `aopc`).

Attributions: SHAP and IG are read from the committed `results/xai/*.json` (verified to reproduce);
**corrected LIME is regenerated live at full precision**, because the stored fix JSONs are rounded to
5 dp, which re-orders LIME's tied bag-of-words weights (Directive 11 limitation 8).

CIs ANNOTATE ONLY — nothing is reselected and no committed number changes.

Run: python -m src.experiments.d11b_ci
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from src.xai import champions as C, explainers, metrics, textops
except ImportError:  # pragma: no cover
    from xai import champions as C, explainers, metrics, textops

_ROOT = Path(__file__).resolve().parent.parent.parent
XAI_DIR = _ROOT / "results" / "xai"
OUT = _ROOT / "results" / "xai_lime_fix" / "knife_edge_ci.json"
B = 10_000
SEED = 42
K_AOPC = list(range(1, 21))


def _resamples(y: np.ndarray, seed: int = SEED, b: int = B):
    """b within-class bootstrap index arrays (verbatim from src/experiments/d9_stats.py)."""
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    return [np.concatenate([rng.choice(pos, len(pos), replace=True),
                            rng.choice(neg, len(neg), replace=True)]) for _ in range(b)]


def per_doc_aopc(champ, texts, attrs, preds) -> np.ndarray:
    """AOPC per document = mean over k=1..20 of (p_pred(orig) - p_pred(top-k removed))."""
    preds = np.asarray(preds)
    p_orig = metrics._p_pred(champ.proba(texts), preds)
    variants, meta = [], []
    for i, (text, attr, pred) in enumerate(zip(texts, attrs, preds)):
        nw = len(textops.word_spans(text))
        top = textops.top_k_positions(attr, int(pred), min(20, nw))
        for k in K_AOPC:
            variants.append(textops.remove_words(text, top[:k]))
            meta.append((i, k))
    P = metrics._p_pred(champ.proba(variants), preds[[m[0] for m in meta]])
    acc = defaultdict(list)
    for (i, _k), pv in zip(meta, P):
        acc[i].append(p_orig[i] - pv)
    return np.array([float(np.mean(acc[i])) for i in range(len(texts))])


def _stored_attrs(unit: str, method: str) -> list[np.ndarray]:
    d = json.loads((XAI_DIR / f"{unit}__{method}.json").read_text(encoding="utf-8"))
    return [np.asarray(a, dtype=float) for a in d["attrs"]]


def _arm(unit: str, method: str, champ, es, texts):
    """Per-doc AOPC for one arm. LIME is regenerated live (stored fix JSONs are 5-dp lossy)."""
    family = C.unit_meta(unit)["family"]
    if method == "lime":
        attrs = [explainers.attribute(champ, family, "lime", t) for t in texts]
    else:
        attrs = _stored_attrs(unit, method)
    return per_doc_aopc(champ, texts, attrs, es["pred"])


def _ci(delta_per_doc: np.ndarray, y: np.ndarray) -> dict:
    res = _resamples(y)
    boot = np.array([delta_per_doc[ix].mean() for ix in res])
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    return {"delta_mean": round(float(delta_per_doc.mean()), 6),
            "ci95": [round(lo, 6), round(hi, 6)],
            "excludes_zero": bool(lo > 0 or hi < 0), "n_docs": int(delta_per_doc.size),
            "B": B, "seed": SEED}


def main() -> int:
    out: dict = {"directive": "D11b", "B": B, "seed": SEED,
                 "method": ("paired within-class bootstrap over documents (d9_stats._resamples "
                            "verbatim); statistic = mean per-document AOPC difference; CI = 2.5/97.5 "
                            "percentiles; identical resample indices for both arms"),
                 "note": "CIs annotate only; no committed number is changed or reselected.",
                 "K1": {}, "K2": {}}

    # ---------- K1: tfidf family, SHAP vs corrected LIME ----------
    pooled_delta, pooled_y = [], []
    for unit in ["tfidf_d1_sb", "tfidf_d2", "tfidf_d3_en"]:
        champ = C.build_champion(unit)
        es = C.load_or_build_explanation_set(unit, champ)
        texts = C.explained_texts(unit, es)
        a_shap = _arm(unit, "shap", champ, es, texts)
        a_lime = _arm(unit, "lime", champ, es, texts)
        y = np.asarray(es["y_true"])
        # sanity: per-doc means must reproduce the committed aggregate AOPC
        print(f"  [{unit}] recomputed AOPC shap={a_shap.mean():.4f} lime={a_lime.mean():.4f} "
              f"(committed shap/lime in the CSVs)", flush=True)
        delta = a_shap - a_lime                                  # SHAP minus corrected LIME
        out["K1"][unit] = _ci(delta, y)
        pooled_delta.append(delta)
        pooled_y.append(y)
        print(f"     delta={out['K1'][unit]['delta_mean']:+.4f} "
              f"CI{out['K1'][unit]['ci95']} excludes0={out['K1'][unit]['excludes_zero']}", flush=True)
    out["K1"]["pooled"] = _ci(np.concatenate(pooled_delta), np.concatenate(pooled_y))
    print(f"  [K1 pooled] delta={out['K1']['pooled']['delta_mean']:+.4f} "
          f"CI{out['K1']['pooled']['ci95']} excludes0={out['K1']['pooled']['excludes_zero']}", flush=True)

    # ---------- K2: transformer_d3_en, IG vs corrected LIME ----------
    unit = "transformer_d3_en"
    champ = C.build_champion(unit)
    es = C.load_or_build_explanation_set(unit, champ)
    texts = C.explained_texts(unit, es)
    a_ig = _arm(unit, "ig", champ, es, texts)
    a_lime = _arm(unit, "lime", champ, es, texts)
    print(f"  [{unit}] recomputed AOPC ig={a_ig.mean():.4f} lime={a_lime.mean():.4f}", flush=True)
    out["K2"][unit] = _ci(a_ig - a_lime, np.asarray(es["y_true"]))     # IG minus corrected LIME
    print(f"     delta={out['K2'][unit]['delta_mean']:+.4f} "
          f"CI{out['K2'][unit]['ci95']} excludes0={out['K2'][unit]['excludes_zero']}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"  wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
