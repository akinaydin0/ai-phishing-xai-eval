"""
Directive 10 figure (Part D): figures/hitl/hitl_cue_rates.(png+pdf).

Grouped bars: HUMAN vs LEXICON vs LLM-JUDGE cue-rate@10 per (corpus, family, method), one panel per
(corpus, family) cell, with the human 95% bootstrap CI drawn on the human bar. Reads the committed
results/hitl/{hitl_metrics.json, hitl_set_level.csv}.

Run: python -m src.experiments.d10_figures
"""

from __future__ import annotations

import csv as _csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent.parent
HITL = _ROOT / "results" / "hitl"
FIG_DIR = _ROOT / "figures" / "hitl"

CELLS = ["d1_sb|tfidf", "d2|tfidf", "d3_en|tfidf", "d3_en|transformer"]
METHOD_ORDER = ["shap", "lime", "ig", "random"]
# The arm generated as "LIME" is void (upstream inverted split_expression zeroed every LIME
# attribution => top-10 == the doc's first 10 word tokens). It is plotted under its true identity.
# See notes/decisions.md Amendment 4 / notes/hitl_summary.md.
METHOD_LABEL = {"shap": "SHAP", "lime": "FIRST10†", "ig": "IG", "random": "RANDOM"}
RATER_COLOR = {"human": "#1f77b4", "lexicon": "#ff7f0e", "judge": "#2ca02c"}
RATER_LABEL = {"human": "Human (author, method-blind)", "lexicon": "Cue lexicon (proxy)",
               "judge": "LLM judge (claude-opus-4-8, blind)"}


def _judge_rates():
    """mean judge cue-rate per (corpus|family, method) over judge-ok sets."""
    acc = defaultdict(list)
    with open(HITL / "hitl_set_level.csv", encoding="utf-8", newline="") as f:
        for r in _csv.DictReader(f):
            if r["judge_status"] == "ok" and r["judge_cuerate"] != "":
                acc[(f"{r['corpus']}|{r['family']}", r["method"])].append(float(r["judge_cuerate"]))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def cue_rate_figure():
    m = json.loads((HITL / "hitl_metrics.json").read_text(encoding="utf-8"))
    jr = _judge_rates()
    acc = m["accounting"]
    p2 = m["P2"]["overall"]
    p3 = m["P3"]["token_kappa"]

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.8), sharey=True)
    for ax, cell in zip(axes, CELLS):
        cr = m["P1"]["cue_rates"][cell]
        methods = [mm for mm in METHOD_ORDER if mm in cr]
        x = np.arange(len(methods))
        w = 0.27
        for j, rater in enumerate(["human", "lexicon", "judge"]):
            vals, errs = [], [[], []]
            for mm in methods:
                if rater == "human":
                    v = cr[mm]["human_mean"]
                    lo, hi = cr[mm]["ci95"]
                    errs[0].append(max(0.0, v - lo)); errs[1].append(max(0.0, hi - v))
                elif rater == "lexicon":
                    v = cr[mm]["lexicon_mean"]
                    errs[0].append(0.0); errs[1].append(0.0)
                else:
                    v = jr.get((cell, mm), float("nan"))
                    errs[0].append(0.0); errs[1].append(0.0)
                vals.append(v)
            ax.bar(x + (j - 1) * w, vals, w, color=RATER_COLOR[rater],
                   label=RATER_LABEL[rater] if cell == CELLS[0] else None,
                   yerr=(errs if rater == "human" else None), capsize=3, ecolor="#333333")
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABEL[mm] for mm in methods], fontsize=10)
        corpus, family = cell.split("|")
        ax.set_title(f"{corpus} · {family}\n(n=20 docs, {len(methods)} sets/doc)", fontsize=11)
        ax.grid(axis="y", alpha=0.3, linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("cue-rate@10 (mean over docs)")
    fig.legend(loc="lower center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.09))
    fig.suptitle(
        "Directive 10 — human vs lexicon-proxy vs LLM-judge plausibility (cue-rate@10) on the "
        "committed TEST explanation sets\n"
        f"60 docs (20/corpus: 15 TP + 5 FP, phishing-predicted) · {acc['n_sets']} blind sets · "
        f"{acc['n_tokens']} token judgments · judge ok {acc['n_judge_ok']}/{acc['n_sets']} "
        f"({acc['n_judge_missing']} refusals) · lexicon–human κ={p2['kappa']:.3f} ({p2['band']}) · "
        f"human–judge κ={p3['kappa']:.3f} ({p3['band']})\n"
        "† the arm generated as \"LIME\" is void (upstream inverted split_expression zeroed every LIME "
        "attribution) and is in fact the document's first 10 word tokens — see Amendment 4",
        fontsize=11.5, y=1.14)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "hitl_cue_rates.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG_DIR / "hitl_cue_rates.pdf", bbox_inches="tight",
                metadata={"CreationDate": None})
    plt.close(fig)
    print(f"  wrote {FIG_DIR / 'hitl_cue_rates.png'} + .pdf", flush=True)


if __name__ == "__main__":
    cue_rate_figure()
