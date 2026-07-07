"""
Generate notes/hpo_summary.md from the HPO / dense / cross-corpus artifacts:
  results/results.csv, results/hpo/<corpus>_<family>.json, results/champions.json,
  results/probe_timings.json.

Sections: (a) best-params, (b) tuned-vs-default deltas, (c) in-corpus leaderboard,
(d) champion cross-corpus, (e) computational cost (GPU column for XGBoost),
per-family tuned max_features winners, and 5 observations.

Run: python -m src.experiments.report_hpo
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = _PROJECT_ROOT / "results"
HPO_DIR = RESULTS / "hpo"
SUMMARY = _PROJECT_ROOT / "notes" / "hpo_summary.md"

CORPORA = ["d1_decontaminated_sb", "d2", "d3_en"]
SHORT = {"d1_decontaminated_sb": "d1_sb", "d2": "d2", "d3_en": "d3_en"}
FAMILIES = ["lr", "nb", "svc", "dt", "rf", "xgb", "lgbm"]
NAME = {"lr": "LogisticRegression", "nb": "MultinomialNB", "svc": "LinearSVC",
        "dt": "DecisionTree", "rf": "RandomForest", "xgb": "XGBoost", "lgbm": "LightGBM"}
GROUPS = {"tfidf": ["lr", "nb", "svc", "dt"], "ensemble": ["rf", "xgb", "lgbm"]}


def _load_hpo():
    out = {}
    for c in CORPORA:
        for f in FAMILIES:
            p = HPO_DIR / f"{c}_{f}.json"
            if p.exists():
                out[(c, f)] = json.loads(p.read_text(encoding="utf-8"))
    return out


def _fmt_params(bp: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in bp.items())


def build():
    df = pd.read_csv(RESULTS / "results.csv")
    hpo = _load_hpo()
    probe = json.loads((RESULTS / "probe_timings.json").read_text(encoding="utf-8"))
    champ_path = RESULTS / "champions.json"
    champions = json.loads(champ_path.read_text(encoding="utf-8")) if champ_path.exists() else {}

    def rid(suffix_model, feat, tr, te, tag):
        r = df[df.run_id == f"{suffix_model}__{feat}__{tr}__{te}__{tag}"]
        return r.iloc[0] if len(r) else None

    L = ["# HPO + Classical/Ensemble Grid + Dense Embeddings — Summary\n"]
    L.append("Optuna HPO (val ROC-AUC objective, tie-break val F1; test untouched during "
             "tuning) for 7 families × 3 corpora, plus dense-embedding baselines and champion "
             "cross-corpus transfer. Linear families use 100k TF-IDF; tree/boosting families "
             "**tune** max_features ∈ {10k, 20k, 50k}. XGB/LGBM fix n_estimators=800 with "
             "early_stopping_rounds=50 on val AUC. XGBoost search ran on GPU (RTX 3090); every "
             "finalist retrain + single test eval ran on CPU (seed 42, deterministic). "
             "Full log: `results/results.csv`.\n")

    # (a) best params
    L.append("## (a) Best hyperparameters per (family, corpus)\n")
    L.append("| Corpus | Family | val AUC | search | max_feat | best params |")
    L.append("|---|---|--:|:--:|--:|---|")
    for c in CORPORA:
        for f in FAMILIES:
            pl = hpo.get((c, f))
            if not pl:
                continue
            bp = dict(pl["best_params"])
            mf = bp.pop("tfidf_max_features", "100k" if f in ("lr", "nb", "svc") else "?")
            bi = f" best_iter={pl['best_iteration']}" if pl.get("best_iteration") else ""
            L.append(f"| {SHORT[c]} | {NAME[f]} | {pl['best_val_auc']:.4f} | "
                     f"{pl['search_device']} | {mf} | {_fmt_params(bp)}{bi} |")
    L.append("")

    # (b) tuned vs default deltas
    L.append("## (b) Tuned vs. untuned-default (test set) — tuning delta\n")
    L.append("| Corpus | Family | default F1 | tuned F1 | ΔF1 | default AUC | tuned AUC | ΔAUC |")
    L.append("|---|---|--:|--:|--:|--:|--:|--:|")
    for c in CORPORA:
        for f in FAMILIES:
            t = rid(NAME[f], "tfidf", c, c, "tuned")
            d = rid(NAME[f], "tfidf", c, c, "default")
            if t is None or d is None:
                continue
            L.append(f"| {SHORT[c]} | {NAME[f]} | {d.f1:.4f} | {t.f1:.4f} | {t.f1 - d.f1:+.4f} | "
                     f"{d.roc_auc:.4f} | {t.roc_auc:.4f} | {t.roc_auc - d.roc_auc:+.4f} |")
    L.append("")

    # (c) in-corpus leaderboard
    L.append("## (c) In-corpus leaderboard (tuned models + dense), by test ROC-AUC\n")
    for c in CORPORA:
        inc = df[(df.train_corpus == c) & (df.test_corpus == c)
                 & (df.run_id.str.endswith("__tuned") | df.run_id.str.contains("__dense"))].copy()
        if not len(inc):
            continue
        inc = inc.sort_values("roc_auc", ascending=False)
        L.append(f"**{SHORT[c]}**\n")
        L.append("| Rank | Model | Features | F1 | ROC-AUC |")
        L.append("|--:|---|---|--:|--:|")
        for i, (_, r) in enumerate(inc.iterrows(), 1):
            L.append(f"| {i} | {r.model} | {r.features} | {r.f1:.4f} | {r.roc_auc:.4f} |")
        L.append("")

    # (d) champion cross-corpus
    L.append("## (d) Champion cross-corpus (train → test)\n")
    xc = df[df.run_id.str.contains("__xcorpus_")]
    if len(xc):
        L.append("| Group | train → test | Model | F1 | ROC-AUC |")
        L.append("|---|---|---|--:|--:|")
        for _, r in xc.iterrows():
            grp = r.run_id.split("__xcorpus_")[1]
            L.append(f"| {grp} | {SHORT.get(r.train_corpus, r.train_corpus)} → "
                     f"{SHORT.get(r.test_corpus, r.test_corpus)} | {r.model} | {r.f1:.4f} | {r.roc_auc:.4f} |")
        L.append("")
        L.append("Champions per source (by val AUC): "
                 + "; ".join(f"{k}={v.get('family', v.get('config'))}" for k, v in champions.items()) + "\n")

    # (e) computational cost
    L.append("## (e) Computational cost (per-fit calibration, results/probe_timings.json)\n")
    cpu = probe["cpu_100k_features_single_fit_s"]
    gpu = probe["xgboost_gpu_vs_cpu_20k_features"]
    L.append("| Model | CPU single-fit @100k (s) | GPU (XGB @20k) | study runtime (sum 3 corpora, s) |")
    L.append("|---|--:|--:|--:|")
    for f in FAMILIES:
        rt = sum(hpo[(c, f)]["runtime_s"] for c in CORPORA if (c, f) in hpo)
        key = {"lr": "LogisticRegression", "nb": "MultinomialNB", "svc": "LinearSVC",
               "dt": "DecisionTree_depth40", "rf": "RandomForest_400_depth40_full",
               "xgb": "XGBoost_500_hist", "lgbm": "LightGBM_500_hist"}[f]
        gpu_cell = f"{gpu['gpu_cuda_s']}s ({gpu['speedup_x']}× vs {gpu['cpu_s']}s CPU)" if f == "xgb" else "—"
        L.append(f"| {NAME[f]} | {cpu.get(key, '—')} | {gpu_cell} | {rt:.0f} |")
    L.append("")
    L.append("*XGBoost is ~106× faster on the RTX 3090 GPU at 20k features (3.4 s vs 360 s CPU); "
             "without the GPU the XGB search alone would exceed 10 h. This feeds the dissertation's "
             "accuracy-vs-computational-cost analysis (to be extended with transformer timings).*\n")
    for f in ("xgb", "lgbm"):
        wins = [str(hpo[(c, f)]["best_params"].get("tfidf_max_features", "?"))
                for c in CORPORA if (c, f) in hpo]
        tag = (" — capped to {10k, 20k} by the 24 GB GPU ceiling at 50k×depth-10 (see "
               "notes/decisions.md)" if f == "xgb" else " — full {10k, 20k, 50k} on CPU, the "
               "empirical 50k check")
        if wins:
            L.append(f"- **{NAME[f]}** tuned TF-IDF max_features (d1_sb / d2 / d3_en): "
                     f"{' / '.join(wins)}{tag}.")
    L.append("")

    # tuned max_features winners
    L.append("## Tuned TF-IDF max_features winner per tree/boosting family\n")
    L.append("| Corpus | DecisionTree | RandomForest | XGBoost | LightGBM |")
    L.append("|---|--:|--:|--:|--:|")
    for c in CORPORA:
        cells = []
        for f in ("dt", "rf", "xgb", "lgbm"):
            pl = hpo.get((c, f))
            cells.append(str(pl["best_params"].get("tfidf_max_features", "?")) if pl else "—")
        L.append(f"| {SHORT[c]} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")
    L.append("")
    L.append("*Fairness note: linear models use 100k features, tree/boosting models use their "
             "tuned max_features — the deployment-relevant comparison is each family at its best "
             "configuration, not a single shared representation.*\n")

    # ---- observations (data-driven) ----
    def incbest(c):
        inc = df[(df.train_corpus == c) & (df.test_corpus == c)
                 & (df.run_id.str.endswith("__tuned") | df.run_id.str.contains("__dense"))]
        r = inc.loc[inc.roc_auc.idxmax()]
        return r.model, r.features, r.roc_auc

    def delta(c, fam, metric):
        t = rid(NAME[fam], "tfidf", c, c, "tuned")
        d0 = rid(NAME[fam], "tfidf", c, c, "default")
        return (t[metric] - d0[metric]) if (t is not None and d0 is not None) else float("nan")

    best = {c: incbest(c) for c in CORPORA}
    nb_deltas = [delta(c, "nb", "f1") for c in CORPORA]
    svc_deltas = [delta(c, "svc", "f1") for c in CORPORA]
    dt_f1 = [delta(c, "dt", "f1") for c in CORPORA]
    dt_auc = [delta(c, "dt", "roc_auc") for c in CORPORA]
    lgbm_mf = [hpo[(c, "lgbm")]["best_params"].get("tfidf_max_features") for c in CORPORA if (c, "lgbm") in hpo]

    L.append("## Observations\n")
    L.append(f"1. **Tuned linear TF-IDF is the in-corpus champion; the ensembles and dense "
             f"models do not beat it.** The top model by test ROC-AUC is a linear TF-IDF model "
             f"on every corpus — {best['d1_decontaminated_sb'][0]} ({best['d1_decontaminated_sb'][2]:.4f}) "
             f"on d1_sb, {best['d2'][0]} ({best['d2'][2]:.4f}) on d2, {best['d3_en'][0]} "
             f"({best['d3_en'][2]:.4f}) on d3_en — with LightGBM/XGBoost a hair behind and the best "
             f"dense model (GloVe-MLP) only reaching the top-5 on the AI corpus. On high-dimensional "
             f"sparse TF-IDF, linear models remain both strongest and cheapest — a standard result "
             f"reaffirmed here after full HPO.")
    L.append("")
    L.append(f"2. **HPO gains are small and near-ceiling — largest where the default is weak.** "
             f"Tuning lifts test F1 by roughly +0.002 to +0.018, with the biggest gains on the "
             f"weakest defaults (MultinomialNB ΔF1 {nb_deltas[0]:+.4f}/{nb_deltas[1]:+.4f}/"
             f"{nb_deltas[2]:+.4f}). LinearSVC is already at its optimum by default "
             f"(ΔF1 {svc_deltas[0]:+.4f}/{svc_deltas[1]:+.4f}/{svc_deltas[2]:+.4f}). These corpora "
             f"sit near the performance ceiling (in-corpus AUC ≥ 0.99), so HPO buys little in-corpus; "
             f"its value is expected to show on harder data and cross-corpus.")
    L.append("")
    L.append(f"3. **Tuning for AUC can trade away threshold-F1 — DecisionTree is the tell.** DT is "
             f"the one family where tuning *lowers* F1 (ΔF1 {dt_f1[0]:+.4f}/{dt_f1[1]:+.4f}/"
             f"{dt_f1[2]:+.4f}) while *raising* AUC sharply (ΔAUC {dt_auc[0]:+.4f}/{dt_auc[1]:+.4f}/"
             f"{dt_auc[2]:+.4f}). The val-AUC objective selects trees that rank better but whose "
             f"fixed-0.5-threshold F1 falls — reinforcing the baselines' finding that the operating "
             f"threshold must be set separately from the ranking model.")
    L.append("")
    L.append(f"4. **Tree/boosting models prefer ~20k features, validating the tuned-dimension "
             f"decision.** Given {{10k, 20k, 50k}} (XGB capped to {{10k, 20k}}), the winners cluster "
             f"at 20k — LightGBM chose 20k on all three corpora ({'/'.join(map(str, lgbm_mf))}) — and "
             f"never approach the linear models' 100k. Histogram methods genuinely cannot exploit "
             f"ultra-high-dimensional sparse text, so tuning max_features per family (rather than "
             f"sharing the 100k default) was the correct, fairer design; each family is compared at "
             f"its own best representation.")
    L.append("")
    xc = df[df.run_id.str.contains("__xcorpus_")]
    if len(xc):
        xmin, xmax = xc.roc_auc.min(), xc.roc_auc.max()
        to_d3 = xc[xc.test_corpus == "d3_en"].roc_auc.mean()
        from_d3 = xc[xc.train_corpus == "d3_en"].roc_auc.mean()
        L.append(f"5. **Champion cross-corpus transfer echoes the baselines: strong AUC but "
                 f"asymmetric, AI→real weakest.** Champion cross-corpus ROC-AUC spans "
                 f"{xmin:.3f}–{xmax:.3f}. Real→AI transfer is strong (mean AUC to d3_en "
                 f"{to_d3:.3f}) while AI→real is weaker (mean AUC from d3_en {from_d3:.3f}) — the "
                 f"same asymmetry the TF-IDF LR baselines showed, now confirmed for the tuned "
                 f"champions across model families. The diagonal (in-corpus) stays ≥0.99; the "
                 f"off-diagonal degradation is the dataset-shift signal the framework is built to "
                 f"surface.")
    else:
        L.append("5. *(cross-corpus champions pending — run `python -m src.experiments.cross_corpus`.)*")
    L.append("")

    SUMMARY.write_text("\n".join(L), encoding="utf-8")
    return SUMMARY


if __name__ == "__main__":
    print(build())
