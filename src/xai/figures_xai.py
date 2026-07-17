"""
XAI figures (Directive 7), PNG + PDF, universe in every title:
  1. deletion curves — one panel per corpus, (family, method) curves overlaid + random baseline;
  2. method-by-family bars — comp@10 / stability / plausibility (mean over in-corpus corpora) vs random;
  3. six qualitative examples — TP + FN across corpora, top-10 attributed words highlighted.

Reads results/xai/<unit>__<method>.json and results/xai_metrics.csv.
Run: python -m src.xai.figures_xai
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

try:
    from src.xai import textops
except ImportError:  # pragma: no cover
    from xai import textops

_ROOT = Path(__file__).resolve().parent.parent.parent
XAI_DIR = _ROOT / "results" / "xai"
FIG_DIR = _ROOT / "figures" / "xai"
CSV_PATH = _ROOT / "results" / "xai_metrics.csv"

CORPORA = ["d1_decontaminated_sb", "d2", "d3_en"]
SHORT = {"d1_decontaminated_sb": "d1_sb", "d2": "d2", "d3_en": "d3_en"}
N_UNIVERSE = {"d1_sb": 7776, "d2": 4384, "d3_en": 2875}   # test-split sizes
FAMILY_COLOR = {"tfidf": "#1f77b4", "ensemble": "#2ca02c", "transformer": "#d62728"}
METHOD_STYLE = {"shap": "-", "lime": "--", "ig": ":"}
FAMILY_UNIT = {"tfidf": "tfidf", "ensemble": "ensemble", "transformer": "transformer"}


def _load(unit, method):
    p = XAI_DIR / f"{unit}__{method}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _save(fig, stem):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 1. Deletion curves
# --------------------------------------------------------------------------- #
def deletion_curves():
    ks = list(range(1, 21))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, corpus in zip(axes, CORPORA):
        sc = SHORT[corpus]
        rnd_curves = []
        for family in ["tfidf", "ensemble", "transformer"]:
            for method in ["shap", "lime", "ig"]:
                unit = f"{FAMILY_UNIT[family]}_{sc}"
                d = _load(unit, method)
                if not d:
                    continue
                ax.plot(ks, d["curve_comp"], color=FAMILY_COLOR[family],
                        linestyle=METHOD_STYLE[method], linewidth=1.8,
                        label=f"{family}·{method}")
                rnd_curves.append(d["curve_rnd"])
        if rnd_curves:
            ax.plot(ks, np.mean(rnd_curves, axis=0), color="grey", linestyle=(0, (1, 1)),
                    linewidth=2.2, label="random baseline")
        ax.set_title(f"{sc} (test n={N_UNIVERSE[sc]:,})")
        ax.set_xlabel("k tokens removed"); ax.set_ylabel("comprehensiveness (Δ p_pred)")
        ax.grid(alpha=0.25); ax.legend(fontsize=7, ncol=2)
    fig.suptitle("XAI faithfulness — deletion curves (higher = more faithful); "
                 "family = colour, method = line style", fontsize=12)
    fig.tight_layout()
    _save(fig, "xai_deletion_curves")


# --------------------------------------------------------------------------- #
# 2. Method-by-family bars (mean over in-corpus corpora) vs random
# --------------------------------------------------------------------------- #
def method_family_bars():
    df = pd.read_csv(CSV_PATH)
    inc = df[df.kind == "in"]
    families = ["tfidf", "ensemble", "transformer"]
    methods = ["shap", "lime", "ig"]
    panels = [("comp@10", "rnd_comp@10", "comprehensiveness@10"),
              ("stability_jaccard", None, "stability (top-10 Jaccard)"),
              ("plausibility@10", "rnd_plausibility@10", "plausibility@10 (cue-hit)")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(families)); w = 0.25
    for ax, (metric, rnd, title) in zip(axes, panels):
        for mi, method in enumerate(methods):
            vals, rnds = [], []
            for fam in families:
                sub = inc[(inc.family == fam) & (inc.method == method)]
                vals.append(sub[metric].mean() if len(sub) else np.nan)
                rnds.append(sub[rnd].mean() if (rnd and len(sub)) else np.nan)
            ax.bar(x + (mi - 1) * w, vals, w, label=method, color=f"C{mi}")
            if rnd:
                ax.plot(x + (mi - 1) * w, rnds, "k_", markersize=10, markeredgewidth=2)
        ax.set_xticks(x); ax.set_xticklabels(families)
        ax.set_title(title); ax.grid(alpha=0.25, axis="y"); ax.legend(fontsize=8)
    fig.suptitle("XAI method × family (mean over in-corpus d1_sb / d2 / d3_en); "
                 "black ticks = random baseline", fontsize=12)
    fig.tight_layout()
    _save(fig, "xai_method_family_bars")


# --------------------------------------------------------------------------- #
# 3. Six qualitative examples (TP + FN per corpus), top-10 highlighted
# --------------------------------------------------------------------------- #
def _pick_examples(method="shap", max_words=90):
    """One TP (true1/pred1) + one FN (true1/pred0) per corpus from the transformer unit."""
    picks = []
    for corpus in CORPORA:
        sc = SHORT[corpus]
        d = _load(f"transformer_{sc}", method)
        if not d:
            continue
        preds, y = d["preds"], d["y_true"]
        tp = next((i for i in range(len(preds)) if y[i] == 1 and preds[i] == 1), None)
        fn = next((i for i in range(len(preds)) if y[i] == 1 and preds[i] == 0), None)
        for tag, i in (("TP", tp), ("FN", fn)):
            if i is not None:
                picks.append((sc, tag, i, d))
    return picks[:6]


def qualitative_examples(method="shap", per_line=13, max_lines=4):
    """Grid layout (monospace) so nothing overlaps; top-10 attributed words boxed in red."""
    from src.splits import load_splits
    max_words = per_line * max_lines
    picks = _pick_examples(method, max_words)
    if not picks:
        return
    fig, axes = plt.subplots(len(picks), 1, figsize=(14, 1.7 * len(picks)))
    if len(picks) == 1:
        axes = [axes]
    dx = 1.0 / per_line
    for ax, (sc, tag, i, d) in zip(axes, picks):
        corpus = [c for c in CORPORA if SHORT[c] == sc][0]
        te = load_splits(corpus)["test"].reset_index(drop=True)
        pos = {h: r for r, h in enumerate(te["_hash"].to_numpy())}
        text = te["text"].iloc[pos[d["hashes"][i]]]
        attr = np.asarray(d["attrs"][i]); pred = d["preds"][i]
        top = set(textops.top_k_positions(attr, pred, 10))
        ax.axis("off")
        colour = "#2ca02c" if tag == "TP" else "#d62728"
        ax.text(0.0, 1.06, f"{sc} · {tag} · true={d['y_true'][i]} pred={pred}",
                fontsize=9, fontweight="bold", color=colour, transform=ax.transAxes)
        wlist = textops.words(text)[:max_words]
        for wi, w in enumerate(wlist):
            row, col = divmod(wi, per_line)
            x = col * dx
            y = 0.86 - row * (0.92 / max_lines)
            hot = wi in top
            ax.text(x, y, w[:14], fontsize=8, family="monospace", va="top", transform=ax.transAxes,
                    color="white" if hot else "#333",
                    bbox=dict(boxstyle="square,pad=0.15", fc=colour, ec="none") if hot else None)
    fig.suptitle(f"XAI qualitative — transformer champions, top-10 {method.upper()} tokens boxed "
                 f"(TP = green, FN = red; first {max_words} words)", fontsize=11, y=1.005)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    _save(fig, "xai_qualitative_examples")


def main():
    deletion_curves()
    method_family_bars()
    qualitative_examples("shap")
    print(f"XAI figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
