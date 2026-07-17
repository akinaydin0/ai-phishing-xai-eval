"""
XAI orchestrator (Directive 7): compute SHAP / LIME / IG attributions and the faithfulness /
stability / plausibility metrics (+ random baseline) per (unit, method), with incremental
persistence and resume, exactly like Directives 5–6.

Per (unit, method): results/xai/<unit>__<method>.json (metrics, deletion curve, per-doc word
attributions for the figures) + one row in results/xai_metrics.csv. Resume skips a (unit, method)
whose json AND csv row both exist. Transformer forward passes are batched on the GPU inside
`champ.proba`.

Run: python -m src.xai.run_xai --all
     python -m src.xai.run_xai --units transformer_d3_en --methods shap --limit 25   # cost probe
"""

from __future__ import annotations

import argparse
import csv as _csv
import json
import time
from pathlib import Path

import numpy as np

try:
    from src.xai import champions as C, explainers, metrics
except ImportError:  # pragma: no cover
    from xai import champions as C, explainers, metrics

_ROOT = Path(__file__).resolve().parent.parent.parent
XAI_DIR = _ROOT / "results" / "xai"
CSV_PATH = _ROOT / "results" / "xai_metrics.csv"
SHORT = {"d1_decontaminated_sb": "d1_sb", "d2": "d2", "d3_en": "d3_en"}
SEED = 42

FIELDNAMES = [
    "unit", "corpus", "source_corpus", "family", "method", "kind", "n_explained",
    "aopc", "comp@1", "comp@5", "comp@10", "comp@20", "suff@1", "suff@5", "suff@10", "suff@20",
    "stability_jaccard", "stability_spearman", "lime_seed_agreement", "plausibility@10",
    "rnd_aopc", "rnd_comp@1", "rnd_comp@5", "rnd_comp@10", "rnd_comp@20",
    "rnd_suff@1", "rnd_suff@5", "rnd_suff@10", "rnd_suff@20", "rnd_plausibility@10",
    "runtime_s", "seed",
]


def _source_corpus(unit_id: str) -> str:
    # transformer champions carry their source; sklearn champions' source == explained corpus
    for u, fam, builder, exc, kind, n, cap in C.UNIT_SPECS:
        if u == unit_id:
            if kind == "cross":
                return unit_id.split("__")[0].replace("xfer_", "")
            return exc
    return ""


def _read_csv():
    if not CSV_PATH.exists():
        return []
    with open(CSV_PATH, encoding="utf-8", newline="") as f:
        return list(_csv.DictReader(f))


def _upsert(row: dict):
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    key = (row["unit"], row["method"])
    kept = [r for r in _read_csv() if (r["unit"], r["method"]) != key]
    out = kept + [row]
    with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        for r in out:
            w.writerow({k: r.get(k, "") for k in FIELDNAMES})


def _done(unit_id, method) -> bool:
    j = (XAI_DIR / f"{unit_id}__{method}.json").exists()
    row = any(r["unit"] == unit_id and r["method"] == method for r in _read_csv())
    return j and row


def run_unit_method(unit_id: str, method: str, champ=None, es=None, limit: int | None = None,
                    persist: bool = True) -> dict:
    m = C.unit_meta(unit_id)
    family = m["family"]
    if champ is None:
        champ = C.build_champion(unit_id)
    if es is None:
        es = C.load_or_build_explanation_set(unit_id, champ)
    texts = C.explained_texts(unit_id, es)
    preds = es["pred"]
    y_true = es["y_true"]
    if limit is not None:
        texts, preds, y_true = texts[:limit], preds[:limit], y_true[:limit]

    t = {}
    t0 = time.time()
    attrs = [explainers.attribute(champ, family, method, tx) for tx in texts]
    t["attr"] = time.time() - t0

    t0 = time.time(); faith = metrics.faithfulness(champ, texts, attrs, preds); t["faith"] = time.time() - t0
    t0 = time.time(); stab = metrics.stability(champ, family, method, texts, attrs, preds); t["stab"] = time.time() - t0
    t0 = time.time(); plaus = metrics.plausibility(texts, attrs, preds); t["plaus"] = time.time() - t0
    lime_agree = float("nan")
    if method == "lime":
        t0 = time.time(); lime_agree = metrics.lime_seed_agreement(champ, texts, preds); t["lime_agree"] = time.time() - t0

    runtime = round(sum(t.values()), 1)
    row = {"unit": unit_id, "corpus": SHORT[m["explained_corpus"]],
           "source_corpus": SHORT.get(_source_corpus(unit_id), _source_corpus(unit_id)),
           "family": family, "method": method, "kind": m["kind"], "n_explained": len(texts),
           "aopc": faith["aopc"], "stability_jaccard": stab["stability_jaccard"],
           "stability_spearman": stab["stability_spearman"],
           "lime_seed_agreement": round(lime_agree, 4) if lime_agree == lime_agree else "",
           "plausibility@10": plaus["plausibility@10"], "rnd_aopc": faith["rnd_aopc"],
           "rnd_plausibility@10": plaus["rnd_plausibility@10"], "runtime_s": runtime, "seed": SEED}
    for k in metrics.K_REPORT:
        row[f"comp@{k}"] = faith[f"comp@{k}"]; row[f"suff@{k}"] = faith[f"suff@{k}"]
        row[f"rnd_comp@{k}"] = faith[f"rnd_comp@{k}"]; row[f"rnd_suff@{k}"] = faith[f"rnd_suff@{k}"]

    payload = {**row, "timing": {k: round(v, 2) for k, v in t.items()},
               "curve_comp": faith["_curve_comp"], "curve_rnd": faith["_curve_rnd"],
               "cue_categories": plaus["_cue_categories"], "n_phishing_pred": plaus["n_phishing_pred"],
               "hashes": es["hashes"][:len(texts)], "preds": preds, "y_true": y_true,
               "attrs": [[round(float(a), 5) for a in arr] for arr in attrs]}
    if persist and limit is None:
        XAI_DIR.mkdir(parents=True, exist_ok=True)
        (XAI_DIR / f"{unit_id}__{method}.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
        _upsert(row)
    return {"row": row, "timing": t, "runtime": runtime}


def main(argv=None):
    ap = argparse.ArgumentParser(description="XAI orchestrator.")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--units", nargs="*", default=[u[0] for u in C.UNIT_SPECS])
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-persist", action="store_true")
    args = ap.parse_args(argv)

    grand = time.time()
    for unit_id in args.units:
        m = C.unit_meta(unit_id)
        methods = args.methods or m["methods"]
        methods = [mm for mm in methods if mm in m["methods"]]
        if not methods:
            continue
        champ = C.build_champion(unit_id)
        es = C.load_or_build_explanation_set(unit_id, champ)
        for method in methods:
            if args.limit is None and not args.no_persist and _done(unit_id, method):
                print(f"[{unit_id:22s} {method:4s}] SKIP (done)", flush=True)
                continue
            r = run_unit_method(unit_id, method, champ, es, limit=args.limit,
                                persist=not args.no_persist)
            tb = " ".join(f"{k}={v:.0f}s" for k, v in r["timing"].items())
            print(f"[{unit_id:22s} {method:4s}] n={r['row']['n_explained']} "
                  f"aopc={r['row']['aopc']} comp@10={r['row']['comp@10']} "
                  f"stab_j={r['row']['stability_jaccard']} plaus={r['row']['plausibility@10']} "
                  f"| {tb} total={r['runtime']:.0f}s", flush=True)
    print(f"\nXAI done in {(time.time() - grand) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
