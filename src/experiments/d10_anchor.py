"""
Directive 10 - Part A.1: regeneration integrity anchor.

The git-ignored XAI attribution JSONs are regenerable. This proves it BEFORE any HITL subset is
extracted: regenerate attributions with the committed XAI parameters (SHAP max_evals=500,
LIME num_samples=1000, IG 32 steps, seed 42) for two anchor units on their FULL committed
explained_ids sets, recompute plausibility@10, and ASSERT equality with results/xai_metrics.csv
to 4 dp. HARD-FAIL (non-zero exit) on any mismatch.

Anchors: (d1_sb, tfidf) shap ; (d3_en, transformer) shap + ig.

Only attributions + plausibility@10 are recomputed (the metric the HITL phase validates); the
expensive faithfulness/stability steps are not re-run - plausibility depends only on the
attribution ranking. Attributions are recomputed with the committed pipeline; nothing is written
to results/xai/ (the canonical JSONs/CSV are left untouched).

Run: python -m src.experiments.d10_anchor
"""

from __future__ import annotations

import csv as _csv
import sys
from pathlib import Path

try:
    from src.xai import champions as C, explainers, metrics
except ImportError:  # pragma: no cover
    from xai import champions as C, explainers, metrics

_ROOT = Path(__file__).resolve().parent.parent.parent
CSV_PATH = _ROOT / "results" / "xai_metrics.csv"

# (unit, method) anchors; shap on both, ig on the transformer unit (per the pre-registration).
ANCHORS = [("tfidf_d1_sb", "shap"), ("transformer_d3_en", "shap"), ("transformer_d3_en", "ig")]


def _committed_plaus(unit: str, method: str) -> str:
    with open(CSV_PATH, encoding="utf-8", newline="") as f:
        for r in _csv.DictReader(f):
            if r["unit"] == unit and r["method"] == method:
                return r["plausibility@10"]
    raise KeyError(f"{unit}/{method} not in {CSV_PATH}")


def regenerate_plaus(unit_id: str, method: str, champ=None, es=None) -> float:
    m = C.unit_meta(unit_id)
    family = m["family"]
    if champ is None:
        champ = C.build_champion(unit_id)
    if es is None:
        es = C.load_or_build_explanation_set(unit_id, champ)
    texts = C.explained_texts(unit_id, es)
    preds = es["pred"]
    attrs = [explainers.attribute(champ, family, method, tx) for tx in texts]
    return float(metrics.plausibility(texts, attrs, preds)["plausibility@10"])


def main() -> int:
    print("Directive 10 - Part A.1 regeneration integrity anchor", flush=True)
    print("=" * 68, flush=True)
    ok = True
    champ_cache: dict[str, object] = {}
    es_cache: dict[str, dict] = {}
    for unit, method in ANCHORS:
        if unit not in champ_cache:
            champ_cache[unit] = C.build_champion(unit)
            es_cache[unit] = C.load_or_build_explanation_set(unit, champ_cache[unit])
        got = regenerate_plaus(unit, method, champ_cache[unit], es_cache[unit])
        want_str = _committed_plaus(unit, method)
        want = float(want_str)
        match = round(got, 4) == round(want, 4)
        ok = ok and match
        flag = "PASS" if match else "FAIL"
        print(f"  [{flag}] {unit:20s} {method:5s}  regenerated plaus@10={got:.4f}  "
              f"committed={want_str} (== {want:.4f})", flush=True)
    print("=" * 68, flush=True)
    if not ok:
        print("HARD-FAIL: at least one anchor plausibility@10 does not reproduce to 4 dp.",
              flush=True)
        return 1
    print("ALL ANCHORS REPRODUCE to 4 dp. Attribution regenerability proven.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
