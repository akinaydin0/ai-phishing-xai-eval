"""
Directive 11 figures: figures/xai_lime_fix/ (PNG + PDF), corrected LIME beside the void FIRST10 arm.

  1. xai_lime_fix_deletion_curves — one panel per in-corpus corpus; corrected LIME vs the committed
     SHAP / IG curves vs the void arm (FIRST10, dotted) vs the random baseline.
  2. xai_lime_fix_bars — comprehensiveness@10 / stability (top-10 Jaccard) / plausibility@10 per
     family, corrected LIME beside SHAP, IG and FIRST10.

Reads the canonical `results/xai/<unit>__<method>.json` + `results/xai_metrics.csv` (SHAP/IG and the
void LIME arm) and the isolated `results/xai_lime_fix/<unit>__lime.json` +
`results/xai_metrics_limefix.csv` (corrected LIME). The canonical `figures/xai/*` are NOT touched --
the erratum in notes/xai_summary.md already governs them.

Run: python -m src.xai.figures_lime_fix
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent.parent
XAI_DIR = _ROOT / "results" / "xai"
FIX_DIR = _ROOT / "results" / "xai_lime_fix"
CSV_CANON = _ROOT / "results" / "xai_metrics.csv"
CSV_FIX = _ROOT / "results" / "xai_metrics_limefix.csv"
FIG_DIR = _ROOT / "figures" / "xai_lime_fix"

CORPORA = ["d1_sb", "d2", "d3_en"]
N_UNIVERSE = {"d1_sb": 7776, "d2": 4384, "d3_en": 2875}
FAMILIES = ["tfidf", "ensemble", "transformer"]
# series label -> (colour, linestyle)
STYLE = {"lime_fixed": ("#d62728", "-"), "shap": ("#1f77b4", "-"), "ig": ("#2ca02c", "-"),
         "first10": ("#9467bd", (0, (4, 2))), "random": ("grey", (0, (1, 1)))}
LABEL = {"lime_fixed": "LIME (corrected)", "shap": "SHAP", "ig": "IG",
         "first10": "FIRST10 (void arm)", "random": "random baseline"}


def _load(d: Path, unit: str, method: str):
    p = d / f"{unit}__{method}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _save(fig, stem):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)
    print(f"  wrote {FIG_DIR / stem}.png + .pdf", flush=True)


def deletion_curves():
    ks = list(range(1, 21))
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), sharey=True)
    for ax, sc in zip(axes, CORPORA):
        rnd = []
        for family in FAMILIES:
            unit = f"{family}_{sc}"
            series = [("lime_fixed", _load(FIX_DIR, unit, "lime")),
                      ("shap", _load(XAI_DIR, unit, "shap")),
                      ("first10", _load(XAI_DIR, unit, "lime"))]
            if family == "transformer":
                series.append(("ig", _load(XAI_DIR, unit, "ig")))
            for key, d in series:
                if not d:
                    continue
                colour, ls = STYLE[key]
                # one line style per method; family distinguished by alpha/width
                ax.plot(ks, d["curve_comp"], color=colour, linestyle=ls,
                        linewidth=2.4 if family == "transformer" else 1.5,
                        alpha=1.0 if family == "transformer" else 0.55,
                        label=f"{LABEL[key]} · {family}")
                rnd.append(d["curve_rnd"])
        if rnd:
            colour, ls = STYLE["random"]
            ax.plot(ks, np.mean(rnd, axis=0), color=colour, linestyle=ls, linewidth=2.2,
                    label=LABEL["random"])
        ax.set_title(f"{sc} (TEST n={N_UNIVERSE[sc]:,}; explained n=200)")
        ax.set_xlabel("k tokens removed")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=6.5, ncol=2)
    axes[0].set_ylabel("comprehensiveness (drop in p_predicted)")
    fig.suptitle("Directive 11 — faithfulness deletion curves with the CORRECTED LIME arm "
                 "(higher = more faithful)\n"
                 "corrected LIME (red) vs committed SHAP (blue) / IG (green) vs the void arm "
                 "FIRST10 (purple dashed) vs random (grey dotted); bold = transformer, faded = "
                 "tfidf/ensemble", fontsize=11.5, y=1.06)
    fig.tight_layout()
    _save(fig, "xai_lime_fix_deletion_curves")


def bars():
    canon = pd.read_csv(CSV_CANON)
    fix = pd.read_csv(CSV_FIX)
    canon = canon[canon.kind == "in"]
    fix = fix[fix.kind == "in"]
    panels = [("comp@10", "rnd_comp@10", "comprehensiveness@10 (faithfulness)"),
              ("stability_jaccard", None, "stability (top-10 Jaccard)"),
              ("plausibility@10", "rnd_plausibility@10", "plausibility@10 (cue-hit proxy)")]
    series = ["lime_fixed", "shap", "ig", "first10"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    x = np.arange(len(FAMILIES))
    w = 0.2
    for ax, (metric, rnd, title) in zip(axes, panels):
        for si, key in enumerate(series):
            vals, rnds = [], []
            for fam in FAMILIES:
                if key == "lime_fixed":
                    sub = fix[fix.family == fam]
                elif key == "first10":
                    sub = canon[(canon.family == fam) & (canon.method == "lime")]
                else:
                    sub = canon[(canon.family == fam) & (canon.method == key)]
                vals.append(sub[metric].mean() if len(sub) else np.nan)
                rnds.append(sub[rnd].mean() if (rnd and len(sub)) else np.nan)
            colour, _ls = STYLE[key]
            ax.bar(x + (si - 1.5) * w, vals, w, label=LABEL[key], color=colour,
                   hatch="//" if key == "first10" else None, alpha=0.95)
            if rnd:
                ax.plot(x + (si - 1.5) * w, rnds, "k_", markersize=9, markeredgewidth=2)
        ax.set_xticks(x)
        ax.set_xticklabels(FAMILIES)
        ax.set_title(title)
        ax.grid(alpha=0.25, axis="y")
        ax.set_axisbelow(True)
        ax.legend(fontsize=7.5)
    fig.suptitle("Directive 11 — corrected LIME vs committed SHAP / IG vs the void FIRST10 arm "
                 "(mean over in-corpus d1_sb / d2 / d3_en; black ticks = random baseline)\n"
                 "FIRST10 (hatched) is the void arm relabelled: it is the document's first 10 word "
                 "tokens, not LIME — see notes/decisions.md Amendment 4", fontsize=11.5, y=1.04)
    fig.tight_layout()
    _save(fig, "xai_lime_fix_bars")


if __name__ == "__main__":
    deletion_curves()
    bars()
