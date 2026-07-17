"""
D9 Part D — imbalance stress test.

Eval-side (no new model runs): on the cached test scores of the classical finalist and the 256
transformer finalist per corpus, stress prevalence to 1:9 / 1:19 / 1:49 (10/5/2%) by keeping ALL
legitimate test docs and subsampling positives without replacement; 50 draws per (corpus, ratio),
seeds 42..91. Train-side (classical only): retrain the frozen-hyperparameter classical finalist
with TRAIN positives subsampled to each ratio and evaluate ONCE on the natural TEST split (9
single-use evaluations). Operational framing: false positives per 1,000 legitimate emails.

Outputs: results/imbalance/imbalance_eval.csv, imbalance_train.csv;
         figures/stats/imbalance_curves.(png|pdf)
Run: python -m src.experiments.d9_scores && python -m src.experiments.d9_imbalance
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.special import expit  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score, f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score,
)

try:
    from src.experiments import d9_scores, hpo
    from src.splits import load_splits
except ImportError:  # pragma: no cover
    from experiments import d9_scores, hpo
    from splits import load_splits

_ROOT = Path(__file__).resolve().parent.parent.parent
IMB_DIR = _ROOT / "results" / "imbalance"
FIG_DIR = _ROOT / "figures" / "stats"
HPO_DIR = _ROOT / "results" / "hpo"
D1 = "d1_decontaminated_sb"

RATIOS = {"1:9": 9, "1:19": 19, "1:49": 49}   # phish:legit ; prevalence 10/5/2%
SEEDS = list(range(42, 92))                    # 50 draws
# corpus -> (classical score_id, classical family, transformer score_id)
CORPORA = {
    "d1_sb": ("svc_d1sb@d1sb", ("svc", D1), "distilbert256_d1sb@d1sb"),
    "d2":    ("nb_d2@d2", ("nb", "d2"), "distilbert256_d2@d2"),
    "d3_en": ("svc_d3en@d3en", ("svc", "d3_en"), "roberta256_d3en@d3en"),
}
METRICS = ["pr_auc", "roc_auc", "precision@0.5", "recall@0.5", "f1@0.5",
           "precision@95recall", "fp_per_1000_legit@95recall"]


def draw_metrics(y, score) -> dict:
    y = np.asarray(y); score = np.asarray(score)
    pred = (score >= 0.5).astype(int)
    m = {"pr_auc": average_precision_score(y, score),
         "roc_auc": roc_auc_score(y, score) if len(np.unique(y)) == 2 else float("nan"),
         "precision@0.5": precision_score(y, pred, zero_division=0),
         "recall@0.5": recall_score(y, pred, zero_division=0),
         "f1@0.5": f1_score(y, pred, zero_division=0)}
    prec, rec, thr = precision_recall_curve(y, score)
    mask = rec >= 0.95
    if mask.any():
        idx = np.where(mask)[0]
        best = idx[np.argmax(prec[idx])]                 # max precision among recall>=0.95
        m["precision@95recall"] = float(prec[best])
        t = float(thr[best]) if best < len(thr) else float(thr[-1] if len(thr) else 0.5)
        n_legit = int(np.sum(y == 0))
        fp = int(np.sum((score >= t) & (y == 0)))
        m["fp_per_1000_legit@95recall"] = fp / n_legit * 1000 if n_legit else float("nan")
    else:
        m["precision@95recall"] = float("nan")
        m["fp_per_1000_legit@95recall"] = float("nan")
    return m


# --------------------------------------------------------------------------- #
# Eval-side
# --------------------------------------------------------------------------- #
def eval_side() -> list[dict]:
    rows = []
    for corpus, (sk_id, _fam, tf_id) in CORPORA.items():
        for fam, sid in (("classical", sk_id), ("transformer", tf_id)):
            scores, y, _ = d9_scores.load(sid)
            pos_idx = np.where(y == 1)[0]; neg_idx = np.where(y == 0)[0]
            for rname, k in RATIOS.items():
                n_pos = max(1, len(neg_idx) // k)         # positives = legit / k
                per_draw = {mtr: [] for mtr in METRICS}
                for seed in SEEDS:
                    rng = np.random.default_rng(seed)
                    take = rng.choice(pos_idx, size=min(n_pos, len(pos_idx)), replace=False)
                    idx = np.concatenate([neg_idx, take])
                    dm = draw_metrics(y[idx], scores[idx])
                    for mtr in METRICS:
                        per_draw[mtr].append(dm[mtr])
                for mtr in METRICS:
                    arr = np.array(per_draw[mtr], dtype=float)
                    rows.append({"corpus": corpus, "ratio": rname, "family": fam, "metric": mtr,
                                 "mean": round(float(np.nanmean(arr)), 4),
                                 "p2_5": round(float(np.nanpercentile(arr, 2.5)), 4),
                                 "p97_5": round(float(np.nanpercentile(arr, 97.5)), 4),
                                 "draws": len(SEEDS)})
    return rows


# --------------------------------------------------------------------------- #
# Train-side (classical only): 9 single-use natural-test evals
# --------------------------------------------------------------------------- #
def _classical_proba(model, vec, texts):
    X = vec.transform(list(texts))
    return model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else expit(model.decision_function(X))


def train_side() -> list[dict]:
    rows = []
    for corpus, (_sk_id, (family, source), _tf_id) in CORPORA.items():
        bp = json.loads((HPO_DIR / f"{source}_{family}.json").read_text(encoding="utf-8"))["best_params"]
        model_params = {kk: vv for kk, vv in bp.items() if kk not in hpo._VEC_KEYS}
        parts = load_splits(source)
        tr, te = parts["train"], parts["test"]
        pos = tr[tr.label == 1]; neg = tr[tr.label == 0]
        yte = te["label"].to_numpy()
        for rname, k in RATIOS.items():
            n_pos = max(1, len(neg) // k)
            pos_sub = pos.sample(n=min(n_pos, len(pos)), random_state=42)
            tr_sub = pd.concat([neg, pos_sub])
            vec = hpo._build_vec(hpo._vec_from_best(family, bp))
            Xtr = vec.fit_transform(tr_sub["text"])
            model = hpo.make_model(family, model_params, device="cpu")
            model.fit(Xtr, tr_sub["label"].to_numpy())            # frozen hyperparams, no early stop
            score = _classical_proba(model, vec, te["text"])       # single-use natural-TEST eval
            dm = draw_metrics(yte, score)
            rows.append({"corpus": corpus, "ratio": rname, "family": "classical",
                         "n_train_pos": len(pos_sub), "n_train_neg": len(neg),
                         **{mtr: round(float(dm[mtr]), 4) for mtr in METRICS}})
            print(f"  train-side {corpus:6s} {rname:5s} pr_auc={dm['pr_auc']:.4f} "
                  f"fp/1000@95rec={dm['fp_per_1000_legit@95recall']:.1f}", flush=True)
    return rows


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def figure(eval_rows):
    df = pd.DataFrame(eval_rows)
    prev = {"1:9": 10, "1:19": 5, "1:49": 2}
    corpora = ["d1_sb", "d2", "d3_en"]
    fams = ["classical", "transformer"]
    colour = {"classical": "#1f77b4", "transformer": "#d62728"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for j, corpus in enumerate(corpora):
        for mi, metric in enumerate(["pr_auc", "fp_per_1000_legit@95recall"]):
            ax = axes[mi, j]
            for fam in fams:
                sub = df[(df.corpus == corpus) & (df.family == fam) & (df.metric == metric)]
                sub = sub.set_index("ratio").reindex(["1:9", "1:19", "1:49"])
                xs = [prev[r] for r in sub.index]
                ax.plot(xs, sub["mean"], "o-", color=colour[fam], label=fam)
                ax.fill_between(xs, sub["p2_5"], sub["p97_5"], color=colour[fam], alpha=0.15)
            ax.set_xticks([2, 5, 10]); ax.invert_xaxis()
            ax.set_xlabel("phishing prevalence (%)")
            ax.set_ylabel("PR-AUC" if metric == "pr_auc" else "FP per 1,000 legit @95% recall")
            ax.set_title(f"{corpus} — {'PR-AUC' if metric=='pr_auc' else 'FP/1000-legit'}")
            ax.grid(alpha=0.25); ax.legend(fontsize=8)
    fig.suptitle("Imbalance stress — eval-side prevalence 10/5/2% (all legit kept, positives "
                 "subsampled; 50 draws, band = 2.5–97.5 pct)", fontsize=12)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "imbalance_curves.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG_DIR / "imbalance_curves.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)


def run():
    IMB_DIR.mkdir(parents=True, exist_ok=True)
    ev = eval_side()
    with open(IMB_DIR / "imbalance_eval.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["corpus", "ratio", "family", "metric", "mean", "p2_5", "p97_5", "draws"])
        w.writeheader(); w.writerows(ev)
    tr = train_side()
    tr_fields = ["corpus", "ratio", "family", "n_train_pos", "n_train_neg"] + METRICS
    with open(IMB_DIR / "imbalance_train.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=tr_fields)
        w.writeheader(); w.writerows(tr)
    figure(ev)
    print(f"\nwrote {IMB_DIR}/imbalance_eval.csv ({len(ev)} rows), imbalance_train.csv ({len(tr)} rows); figure written")
    return ev, tr


if __name__ == "__main__":
    run()
