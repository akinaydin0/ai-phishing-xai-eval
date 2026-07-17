"""
Generate notes/xai_summary.md from results/xai_metrics.csv (+ per-unit JSONs).

Sections: faithfulness, stability, plausibility (each vs the random baseline), cross-corpus
explanation-quality degradation, the explicit RQ3 answer, and 5 house-style observations.
Run: python -m src.xai.report_xai
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent.parent
CSV_PATH = _ROOT / "results" / "xai_metrics.csv"
XAI_DIR = _ROOT / "results" / "xai"
SUMMARY = _ROOT / "notes" / "xai_summary.md"
CORD = ["d1_sb", "d2", "d3_en"]
FAM_ORD = ["tfidf", "ensemble", "transformer"]


def _fmt(x, d=3):
    try:
        return f"{float(x):.{d}f}"
    except (TypeError, ValueError):
        return "—"


def build():
    df = pd.read_csv(CSV_PATH)
    inc = df[df.kind == "in"].copy()
    L = ["# XAI Phase — Attribution Methods + Quantitative Evaluation\n"]
    L.append("SHAP / LIME / Integrated Gradients on the family champions (tfidf-linear, LightGBM, "
             "transformer) per corpus, evaluated for **faithfulness** (token-removal comprehensiveness / "
             "sufficiency / AOPC), **stability** (perturbation Jaccard + Spearman, LIME seed-agreement) "
             "and **plausibility** (cue-hit@10). Explanation docs are drawn from each corpus's TEST split "
             "only (n=200 in-corpus, 100/class, all misclassified up to 30). Every faithfulness and "
             "plausibility number is reported beside a **random-attribution baseline** through the "
             "identical pipeline. **Plausibility is a heuristic cue-lexicon proxy, not human annotation.** "
             "Pre-registration + fixed lexicon: `notes/decisions.md`, `src/xai/cue_lexicon.py`. "
             "Full log: `results/xai_metrics.csv`.\n")

    # ---- (a) Faithfulness ----
    L.append("## (a) Faithfulness — comprehensiveness / sufficiency / AOPC (test set)\n")
    L.append("Higher comp@k = removing the top-k attributed tokens drops the predicted-class "
             "probability more (more faithful). Lower suff@k = the top-k alone reproduce the "
             "prediction. `rnd` = random-attribution baseline.\n")
    L.append("| Corpus | Family | Method | AOPC | rnd AOPC | comp@10 | rnd | suff@10 | rnd |")
    L.append("|---|---|---|--:|--:|--:|--:|--:|--:|")
    for c in CORD:
        for fam in FAM_ORD:
            for meth in ["shap", "lime", "ig"]:
                r = inc[(inc.corpus == c) & (inc.family == fam) & (inc.method == meth)]
                if not len(r):
                    continue
                r = r.iloc[0]
                L.append(f"| {c} | {fam} | {meth} | {_fmt(r['aopc'])} | {_fmt(r['rnd_aopc'])} | "
                         f"{_fmt(r['comp@10'])} | {_fmt(r['rnd_comp@10'])} | {_fmt(r['suff@10'])} | "
                         f"{_fmt(r['rnd_suff@10'])} |")
    L.append("")

    # ---- (b) Stability ----
    L.append("## (b) Stability — perturbation robustness + LIME seed-agreement\n")
    L.append("| Corpus | Family | Method | top-10 Jaccard | Spearman | LIME seed-agree |")
    L.append("|---|---|---|--:|--:|--:|")
    for c in CORD:
        for fam in FAM_ORD:
            for meth in ["shap", "lime", "ig"]:
                r = inc[(inc.corpus == c) & (inc.family == fam) & (inc.method == meth)]
                if not len(r):
                    continue
                r = r.iloc[0]
                L.append(f"| {c} | {fam} | {meth} | {_fmt(r['stability_jaccard'])} | "
                         f"{_fmt(r['stability_spearman'])} | {_fmt(r.get('lime_seed_agreement'))} |")
    L.append("")

    # ---- (c) Plausibility ----
    L.append("## (c) Plausibility — cue-hit@10 on phishing-predicted docs vs random (heuristic proxy)\n")
    L.append("| Corpus | Family | Method | plausibility@10 | random | lift |")
    L.append("|---|---|---|--:|--:|--:|")
    for c in CORD:
        for fam in FAM_ORD:
            for meth in ["shap", "lime", "ig"]:
                r = inc[(inc.corpus == c) & (inc.family == fam) & (inc.method == meth)]
                if not len(r):
                    continue
                r = r.iloc[0]
                lift = float(r["plausibility@10"]) - float(r["rnd_plausibility@10"])
                L.append(f"| {c} | {fam} | {meth} | {_fmt(r['plausibility@10'])} | "
                         f"{_fmt(r['rnd_plausibility@10'])} | {lift:+.3f} |")
    L.append("")

    # ---- (d) three-metric family comparison (mean over corpora, best method per family) ----
    L.append("## (d) Three-metric comparison by family (mean over in-corpus corpora)\n")
    L.append("| Family | best method (by AOPC) | AOPC | stability Jaccard | plausibility@10 | mean XAI runtime/unit (s) |")
    L.append("|---|---|--:|--:|--:|--:|")
    fam_summary = {}
    for fam in FAM_ORD:
        sub = inc[inc.family == fam]
        if not len(sub):
            continue
        by_m = sub.groupby("method").agg(aopc=("aopc", "mean"), jacc=("stability_jaccard", "mean"),
                                         plaus=("plausibility@10", "mean"), rt=("runtime_s", "mean"))
        best_m = by_m["aopc"].idxmax()
        row = by_m.loc[best_m]
        fam_summary[fam] = {"method": best_m, **row.to_dict()}
        L.append(f"| {fam} | {best_m} | {_fmt(row['aopc'])} | {_fmt(row['jacc'])} | "
                 f"{_fmt(row['plaus'])} | {row['rt']:.0f} |")
    L.append("")

    # ---- cross-corpus degradation ----
    cross = df[df.kind == "cross"].copy()
    if len(cross):
        L.append("## (e) Cross-corpus explanation quality — does it degrade with transfer?\n")
        L.append("Transformer champions explained on a DIFFERENT corpus's test split. Compared to the "
                 "same champion's in-corpus explanation quality.\n")
        L.append("| Transfer | Method | AOPC | comp@10 | stability Jaccard | plausibility@10 |")
        L.append("|---|---|--:|--:|--:|--:|")
        for _, r in cross.sort_values(["unit", "method"]).iterrows():
            src_tgt = r["unit"].replace("xfer_", "").replace("__", " → ")
            L.append(f"| {src_tgt} | {r['method']} | {_fmt(r['aopc'])} | {_fmt(r['comp@10'])} | "
                     f"{_fmt(r['stability_jaccard'])} | {_fmt(r['plausibility@10'])} |")
        L.append("")

    # ---- RQ3 + observations ----
    _observations(L, inc, cross, fam_summary)
    SUMMARY.write_text("\n".join(L), encoding="utf-8")
    return SUMMARY


def _observations(L, inc, cross, fam_summary):
    g = inc.groupby(["family", "method"]).agg(
        aopc=("aopc", "mean"), rnd=("rnd_aopc", "mean"), comp=("comp@10", "mean"),
        jacc=("stability_jaccard", "mean"), plaus=("plausibility@10", "mean"),
        rplaus=("rnd_plausibility@10", "mean"), rt=("runtime_s", "mean")).round(3)

    def gv(fam, meth, col):
        try:
            return float(g.loc[(fam, meth), col])
        except KeyError:
            return float("nan")

    def pcv(corpus, fam, meth, col):
        r = inc[(inc.corpus == corpus) & (inc.family == fam) & (inc.method == meth)]
        return float(r[col].iloc[0]) if len(r) else float("nan")

    # --- RQ3 ---
    L.append("## RQ3 — which family gives the best accuracy–explanation–cost balance?\n")
    L.append(f"Combining the modelling results (Directive 6: tuned linear TF-IDF has the top "
             f"in-corpus accuracy, AUC ≥ 0.999; transformers match but do not beat it, at ~1000× the "
             f"training cost) with these XAI results:\n")
    L.append(f"- **Faithfulness** (AOPC): ensemble·SHAP {gv('ensemble','shap','aopc'):.3f} > "
             f"tfidf·SHAP {gv('tfidf','shap','aopc'):.3f} > transformer·SHAP "
             f"{gv('transformer','shap','aopc'):.3f} ≈ transformer·IG {gv('transformer','ig','aopc'):.3f} "
             f"≫ every LIME ({gv('tfidf','lime','aopc'):.3f}/{gv('ensemble','lime','aopc'):.3f}/"
             f"{gv('transformer','lime','aopc'):.3f}, at the random-baseline floor ~"
             f"{g['rnd'].mean():.3f}).")
    L.append(f"- **Plausibility** (cue-hit@10): SHAP beats the random baseline at the family-mean level "
             f"(tfidf {gv('tfidf','shap','plaus'):.3f} vs {gv('tfidf','shap','rplaus'):.3f}; ensemble "
             f"{gv('ensemble','shap','plaus'):.3f} vs {gv('ensemble','shap','rplaus'):.3f}). LIME is at or "
             f"below random on d1_sb and d2, but on the cue-dense d3_en corpus LIME *exceeds* random "
             f"(+0.085…+0.111) — a case of **plausibility without faithfulness**: LIME surfaces cue words "
             f"because AI-generated text is saturated with them, yet its d3_en AOPC stays at the random "
             f"floor, so removing those words does not move the model.")
    L.append(f"- **Cost** to explain one 200-doc unit: tfidf·SHAP **{gv('tfidf','shap','rt'):.0f} s**, "
             f"ensemble·SHAP {gv('ensemble','shap','rt'):.0f} s, transformer·SHAP "
             f"{gv('transformer','shap','rt'):.0f} s, transformer·LIME "
             f"{gv('transformer','lime','rt'):.0f} s — a 100–400× spread.")
    L.append("")
    L.append(f"**Answer: linear TF-IDF + SHAP.** It pairs the best (tied) in-corpus accuracy with the "
             f"second-most-faithful and joint-most-plausible explanations, at the **lowest cost of any "
             f"combination** ({gv('tfidf','shap','rt'):.0f} s/unit). Ensemble (LightGBM) + SHAP is the "
             f"single most *faithful* explainer but its accuracy is a shade lower and it costs ~15× more; "
             f"transformers are the hardest and most expensive to explain faithfully and are justified "
             f"only where cross-corpus robustness forces a contextual model (Directive 6), with **IG** as "
             f"their explainer of choice (as faithful as SHAP there, ~4× cheaper, and more stable).\n")

    # --- observations ---
    L.append("## Observations\n")
    L.append(f"1. **SHAP is the only method that is demonstrably faithful; LIME is barely above "
             f"random on every family.** Removing the top-10 SHAP tokens drops the predicted-class "
             f"probability by AOPC {gv('ensemble','shap','aopc'):.3f} (ensemble), "
             f"{gv('tfidf','shap','aopc'):.3f} (tfidf) and {gv('transformer','shap','aopc'):.3f} "
             f"(transformer), whereas LIME's AOPC ({gv('tfidf','lime','aopc'):.3f}/"
             f"{gv('ensemble','lime','aopc'):.3f}/{gv('transformer','lime','aopc'):.3f}) is "
             f"indistinguishable from the random-attribution baseline (~{g['rnd'].mean():.3f}). The "
             f"deletion curves make this visual: SHAP climbs, LIME hugs the random line.")
    L.append("")
    L.append(f"2. **There is a faithfulness–stability trade-off, and no method wins both.** LIME is the "
             f"most *stable* (top-10 Jaccard ~{gv('tfidf','lime','jacc'):.2f} across families, LIME "
             f"seed-agreement 1.00) precisely because its local bag-of-words fit is insensitive to which "
             f"tokens matter; SHAP and IG are faithful but less stable (transformer·SHAP Jaccard "
             f"{gv('transformer','shap','jacc'):.2f}, IG {gv('transformer','ig','jacc'):.2f}). Stability "
             f"without faithfulness is a stable *wrong* answer — the two must be reported together.")
    L.append("")
    L.append(f"3. **Transformers are the hardest models to explain faithfully.** Their SHAP/IG AOPC "
             f"(~{gv('transformer','shap','aopc'):.2f}) is half the linear champions' and a third of "
             f"LightGBM's, and their SHAP top-10 Jaccard ({gv('transformer','shap','jacc'):.2f}) is the "
             f"lowest anywhere. Contextual models spread the decision across many tokens and their softmax "
             f"saturates on confident cases, so no small token set is individually decisive — the "
             f"attribution is real but diffuse. Among transformer explainers **IG ≈ SHAP on faithfulness "
             f"at ~4× lower cost and higher stability**, so IG is the transformer method of choice.")
    L.append("")
    d1t_shap = pcv("d1_sb", "transformer", "shap", "plausibility@10")
    d1t_rnd = pcv("d1_sb", "transformer", "shap", "rnd_plausibility@10")
    L.append(f"4. **Plausibility tracks faithfulness — with two honest caveats.** At the **family-mean** "
             f"level SHAP's top tokens are phishing cues above chance: SHAP plausibility@10 exceeds random "
             f"by +{gv('tfidf','shap','plaus') - gv('tfidf','shap','rplaus'):.3f} (tfidf), "
             f"+{gv('ensemble','shap','plaus') - gv('ensemble','shap','rplaus'):.3f} (ensemble) and "
             f"+{gv('transformer','shap','plaus') - gv('transformer','shap','rplaus'):.3f} (transformer, "
             f"{gv('transformer','shap','plaus'):.3f} vs {gv('transformer','shap','rplaus'):.3f}) — the "
             f"one per-corpus exception being d1_sb transformer·SHAP at {d1t_shap:.3f}, "
             f"{d1t_shap - d1t_rnd:+.3f} below its {d1t_rnd:.3f} random base rate. And LIME is at or below "
             f"random on d1_sb and d2 but **exceeds it on the cue-dense d3_en corpus (+0.085…+0.111; "
             f"transformer·LIME {pcv('d3_en','transformer','lime','plausibility@10'):.3f} even tops "
             f"transformer·SHAP {pcv('d3_en','transformer','shap','plausibility@10'):.3f})** — yet LIME's "
             f"d3_en AOPC ({gv('transformer','lime','aopc'):.3f}) stays at the random floor. This is "
             f"**plausibility without faithfulness**: LIME surfaces the cue words AI-generated text is "
             f"saturated with, but removing them does not move the model — exactly the failure mode the "
             f"three-metric design exists to expose. *(Plausibility is a heuristic lexicon proxy, not human "
             f"annotation.)* The qualitative panels show the faithful case (advance-fee terms "
             f"'International/Assets/beneficiaries/facilitate' boxed on a d3_en true positive).")
    L.append("")
    if len(cross):
        cr = cross.groupby("method")["aopc"].mean()
        hard = cross[cross.unit == "xfer_d3_en__d2"]
        hs = hard[hard.method == "shap"].iloc[0]
        hard_abs, hard_rnd = float(hs["plausibility@10"]), float(hs["rnd_plausibility@10"])
        inc_lift = pcv("d3_en", "transformer", "shap", "plausibility@10") - \
            pcv("d3_en", "transformer", "shap", "rnd_plausibility@10")
        L.append(f"5. **Cross-corpus: apparent faithfulness *rises* on transfer (a low-confidence "
                 f"artifact), while cue-plausibility falls on the hard legacy axis — in two layers.** "
                 f"Transferred transformer AOPC ({cr.get('shap', float('nan')):.3f} SHAP) is *higher* than "
                 f"in-corpus ({gv('transformer','shap','aopc'):.3f}) — not because explanations improve, "
                 f"but because out-of-distribution predictions are less saturated, so token removal moves p "
                 f"more. The telling metric is plausibility on the hardest axis d3_en→d2 (AI→legacy-Enron), "
                 f"and it must be read two ways: (i) the **absolute** SHAP cue-hit falls to {hard_abs:.3f} "
                 f"largely because legacy Enron's **random cue base rate is only {hard_rnd:.3f}** (Enron "
                 f"carries few lexical phishing cues); (ii) SHAP's **lift over random** shrinks from "
                 f"+{inc_lift:.3f} in-corpus to +{hard_abs - hard_rnd:.3f} on transfer. Both layers point "
                 f"the same way — the legacy↔AI axis is where transfer genuinely breaks down, mirroring the "
                 f"Directive-6 accuracy finding.")
        L.append("")


if __name__ == "__main__":
    print(build())
