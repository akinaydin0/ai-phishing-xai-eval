"""
Directive 11 Part B: isolated re-run of the LIME arm with the corrected `split_expression`.

Scope: **LIME only**, all 12 committed XAI units (9 in-corpus + 3 transformer cross). Everything else
is the committed Directive 7 configuration verbatim — `num_samples=1000`, seed 42, n=200 in-corpus /
100 cross, the **same committed `explained_ids` sets** (loaded from `results/xai/`, so the corrected
and void arms are measured on identical documents), the identical metric pipeline (comp@k / suff@k /
AOPC, stability Jaccard + Spearman, LIME seed-agreement, plausibility@10) and the identical random
baseline. The single change is the fixed argument in `explainers.lime_attr` (Amendment 4).

Isolation: writes ONLY to `results/xai_lime_fix/<unit>__lime.json` and
`results/xai_metrics_limefix.csv`. The canonical `results/xai/*` and `results/xai_metrics.csv` are
never touched (the void LIME rows there are retained as the FIRST10 position-baseline record).

The metric computation is delegated to `run_xai.run_unit_method(..., persist=False)`, i.e. the exact
same code path that produced the committed numbers — this runner only redirects persistence.

Run: python -m src.xai.run_lime_fix                    # all 12, cheap families first, resumable
     python -m src.xai.run_lime_fix --units tfidf_d1_sb
     python -m src.xai.run_lime_fix --units transformer_d3_en --limit 10   # cost probe (no persist)
"""

from __future__ import annotations

import argparse
import csv as _csv
import json
import time
from pathlib import Path

try:
    from src.xai import champions as C, run_xai
except ImportError:  # pragma: no cover
    from xai import champions as C, run_xai

_ROOT = Path(__file__).resolve().parent.parent.parent
FIX_DIR = _ROOT / "results" / "xai_lime_fix"
FIX_CSV = _ROOT / "results" / "xai_metrics_limefix.csv"
METHOD = "lime"
FIELDNAMES = run_xai.FIELDNAMES          # same schema as the canonical log, for direct comparison

# cheap families first (tfidf -> ensemble -> transformer in-corpus -> transformer cross)
ORDER = ["tfidf_d1_sb", "tfidf_d2", "tfidf_d3_en",
         "ensemble_d1_sb", "ensemble_d2", "ensemble_d3_en",
         "transformer_d1_sb", "transformer_d2", "transformer_d3_en",
         "xfer_d3_en__d1_sb", "xfer_d2__d3_en", "xfer_d3_en__d2"]


def _read_csv() -> list[dict]:
    if not FIX_CSV.exists():
        return []
    with open(FIX_CSV, encoding="utf-8", newline="") as f:
        return list(_csv.DictReader(f))


def _upsert(row: dict) -> None:
    FIX_CSV.parent.mkdir(parents=True, exist_ok=True)
    kept = [r for r in _read_csv() if (r["unit"], r["method"]) != (row["unit"], row["method"])]
    out = kept + [row]
    # stable order = ORDER, so the file is deterministic regardless of run/resume sequence
    rank = {u: i for i, u in enumerate(ORDER)}
    out.sort(key=lambda r: rank.get(r["unit"], 99))
    with open(FIX_CSV, "w", encoding="utf-8", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        for r in out:
            w.writerow({k: r.get(k, "") for k in FIELDNAMES})


def _done(unit_id: str) -> bool:
    j = (FIX_DIR / f"{unit_id}__{METHOD}.json").exists()
    row = any(r["unit"] == unit_id and r["method"] == METHOD for r in _read_csv())
    return j and row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Corrected-LIME isolated re-run (Directive 11).")
    ap.add_argument("--units", nargs="*", default=ORDER)
    ap.add_argument("--limit", type=int, default=None, help="cost probe only; never persists")
    args = ap.parse_args(argv)

    grand = time.time()
    for unit_id in args.units:
        if args.limit is None and _done(unit_id):
            print(f"[{unit_id:22s} lime] SKIP (done)", flush=True)
            continue
        champ = C.build_champion(unit_id)
        es = C.load_or_build_explanation_set(unit_id, champ)   # the COMMITTED explained_ids set
        r = run_xai.run_unit_method(unit_id, METHOD, champ, es, limit=args.limit, persist=False)
        row, payload = r["row"], r["payload"]
        if args.limit is None:
            FIX_DIR.mkdir(parents=True, exist_ok=True)
            (FIX_DIR / f"{unit_id}__{METHOD}.json").write_text(json.dumps(payload, indent=1),
                                                               encoding="utf-8")
            _upsert(row)
        tb = " ".join(f"{k}={v:.0f}s" for k, v in r["timing"].items())
        tag = f" [PROBE limit={args.limit}, not persisted]" if args.limit is not None else ""
        print(f"[{unit_id:22s} lime] n={row['n_explained']} aopc={row['aopc']} "
              f"comp@10={row['comp@10']} stab_j={row['stability_jaccard']} "
              f"stab_sp={row['stability_spearman']} seed_agree={row['lime_seed_agreement']} "
              f"plaus={row['plausibility@10']} | {tb} total={r['runtime']:.0f}s{tag}", flush=True)
    print(f"\ncorrected-LIME arm done in {(time.time() - grand) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
