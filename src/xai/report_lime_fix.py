"""
Directive 11: generate the corrected-LIME comparison tables (markdown) for notes/lime_fix_summary.md.

Emits, from the committed CSVs only (no recomputation):
  (1) per-unit corrected LIME vs the void FIRST10 arm vs committed SHAP/IG — faithfulness, stability,
      plausibility, runtime;
  (2) in-corpus family means;
  (3) the ranking flips (which method is most faithful per unit, before vs after);
  (4) cost comparison (runtime per unit, corrected vs void, and vs SHAP/IG).

Run: python -m src.xai.report_lime_fix            # prints markdown
     python -m src.xai.report_lime_fix --check    # assert fix CSV completeness only
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent.parent
CSV_CANON = _ROOT / "results" / "xai_metrics.csv"
CSV_FIX = _ROOT / "results" / "xai_metrics_limefix.csv"
UNITS = ["tfidf_d1_sb", "tfidf_d2", "tfidf_d3_en", "ensemble_d1_sb", "ensemble_d2", "ensemble_d3_en",
         "transformer_d1_sb", "transformer_d2", "transformer_d3_en",
         "xfer_d3_en__d1_sb", "xfer_d2__d3_en", "xfer_d3_en__d2"]


def _load():
    canon = pd.read_csv(CSV_CANON)
    fix = pd.read_csv(CSV_FIX)
    return canon, fix


def _get(df, unit, method, col):
    s = df[(df.unit == unit) & (df.method == method)]
    return None if s.empty else s.iloc[0][col]


def _f(v, nd=4):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def table_per_unit(canon, fix) -> str:
    out = ["| unit | AOPC corrected LIME | AOPC FIRST10 (void) | AOPC SHAP | AOPC IG | AOPC random |",
           "|---|---|---|---|---|---|"]
    for u in UNITS:
        out.append("| `%s` | **%s** | %s | %s | %s | %s |" % (
            u, _f(_get(fix, u, "lime", "aopc")), _f(_get(canon, u, "lime", "aopc")),
            _f(_get(canon, u, "shap", "aopc")), _f(_get(canon, u, "ig", "aopc")),
            _f(_get(fix, u, "lime", "rnd_aopc"))))
    return "\n".join(out)


def table_stability_plaus(canon, fix) -> str:
    out = ["| unit | Jaccard corr. | Jaccard void | Spearman corr. | Spearman void | seed-agree corr. "
           "| seed-agree void | plaus@10 corr. | plaus@10 void | plaus@10 random |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for u in UNITS:
        out.append("| `%s` | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            u,
            _f(_get(fix, u, "lime", "stability_jaccard")), _f(_get(canon, u, "lime", "stability_jaccard")),
            _f(_get(fix, u, "lime", "stability_spearman")), _f(_get(canon, u, "lime", "stability_spearman")),
            _f(_get(fix, u, "lime", "lime_seed_agreement")), _f(_get(canon, u, "lime", "lime_seed_agreement")),
            _f(_get(fix, u, "lime", "plausibility@10")), _f(_get(canon, u, "lime", "plausibility@10")),
            _f(_get(fix, u, "lime", "rnd_plausibility@10"))))
    return "\n".join(out)


def table_family_means(canon, fix) -> str:
    ci, fi = canon[canon.kind == "in"], fix[fix.kind == "in"]
    out = ["| family | metric | corrected LIME | FIRST10 (void) | SHAP | IG | random |",
           "|---|---|---|---|---|---|---|"]
    for fam in ["tfidf", "ensemble", "transformer"]:
        for metric, rnd in [("aopc", "rnd_aopc"), ("comp@10", "rnd_comp@10"),
                            ("stability_jaccard", None), ("plausibility@10", "rnd_plausibility@10")]:
            f_ = fi[fi.family == fam][metric].mean()
            v_ = ci[(ci.family == fam) & (ci.method == "lime")][metric].mean()
            s_ = ci[(ci.family == fam) & (ci.method == "shap")][metric].mean()
            g_ = ci[(ci.family == fam) & (ci.method == "ig")][metric].mean()
            r_ = fi[fi.family == fam][rnd].mean() if rnd else None
            out.append(f"| {fam} | {metric} | **{_f(f_)}** | {_f(v_)} | {_f(s_)} | "
                       f"{'—' if pd.isna(g_) else _f(g_)} | {_f(r_)} |")
    return "\n".join(out)


def table_ranking(canon, fix) -> str:
    """Most-faithful method per unit, before (committed) vs after (with corrected LIME)."""
    out = ["| unit | most faithful BEFORE (committed) | most faithful AFTER (corrected LIME included) "
           "| flip? |", "|---|---|---|---|"]
    for u in UNITS:
        cand_before = {m: _get(canon, u, m, "aopc") for m in ("shap", "lime", "ig")}
        cand_before = {k: float(v) for k, v in cand_before.items() if v is not None and not pd.isna(v)}
        after = dict(cand_before)
        after.pop("lime", None)
        fl = _get(fix, u, "lime", "aopc")
        if fl is not None and not pd.isna(fl):
            after["lime_corrected"] = float(fl)
        b = max(cand_before, key=cand_before.get) if cand_before else "—"
        a = max(after, key=after.get) if after else "—"
        bl = f"{b} ({cand_before.get(b, float('nan')):.4f})" if cand_before else "—"
        al = f"{a} ({after.get(a, float('nan')):.4f})" if after else "—"
        out.append(f"| `{u}` | {bl} | {al} | {'**YES**' if a != b else 'no'} |")
    return "\n".join(out)


def table_cost(canon, fix) -> str:
    out = ["| unit | runtime corrected LIME (s) | runtime FIRST10 void (s) | runtime SHAP (s) | "
           "runtime IG (s) |", "|---|---|---|---|---|"]
    tf = tv = 0.0
    for u in UNITS:
        f_ = _get(fix, u, "lime", "runtime_s")
        v_ = _get(canon, u, "lime", "runtime_s")
        tf += float(f_) if f_ is not None else 0.0
        tv += float(v_) if v_ is not None else 0.0
        out.append("| `%s` | %s | %s | %s | %s |" % (
            u, _f(f_, 1), _f(v_, 1), _f(_get(canon, u, "shap", "runtime_s"), 1),
            _f(_get(canon, u, "ig", "runtime_s"), 1)))
    out.append(f"| **TOTAL** | **{tf:.0f} s = {tf/3600:.2f} h** | {tv:.0f} s = {tv/3600:.2f} h | | |")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    canon, fix = _load()
    missing = [u for u in UNITS if fix[(fix.unit == u) & (fix.method == "lime")].empty]
    print(f"fix CSV rows: {len(fix)} / 12 | missing units: {missing or 'none'}", flush=True)
    if args.check:
        return 1 if missing else 0
    print("\n### (1) Faithfulness (AOPC) per unit\n")
    print(table_per_unit(canon, fix))
    print("\n### (2) Stability + plausibility per unit\n")
    print(table_stability_plaus(canon, fix))
    print("\n### (3) In-corpus family means\n")
    print(table_family_means(canon, fix))
    print("\n### (4) Most-faithful method per unit, before vs after\n")
    print(table_ranking(canon, fix))
    print("\n### (5) Cost\n")
    print(table_cost(canon, fix))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
