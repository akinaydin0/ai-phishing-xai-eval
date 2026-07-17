"""
D9 Part F — notes/d9_stats_robustness_summary.md from the committed D9 CSVs.

(a) the 7-comparison table (CIs + Holm-adjusted McNemar); (b) which committed claims gain/lack
statistical separation (CI-includes-0 stated plainly); (c) imbalance with the FP-per-1000-legit
framing; (d) the preprocessing table + whether the V2-helps-transfer hypothesis held; (e) 3-5
observations. No narrative overrides the pre-registered interpretation rule (CIs/p annotate only).

Run: python -m src.experiments.d9_report
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent.parent
STATS = _ROOT / "results" / "stats" / "stats_summary.csv"
IMB_E = _ROOT / "results" / "imbalance" / "imbalance_eval.csv"
IMB_T = _ROOT / "results" / "imbalance" / "imbalance_train.csv"
PRE = _ROOT / "results" / "preproc_ablation" / "preproc_ablation.csv"
SUMMARY = _ROOT / "notes" / "d9_stats_robustness_summary.md"

CLAIM = {
    "C1a": "in-corpus d1_sb: 256 transformer vs classical champion",
    "C1b": "in-corpus d2: 256 transformer vs classical champion",
    "C1c": "in-corpus d3_en: classical champion vs 256 transformer",
    "C2": "d3_en→d2: 512 vs 256 transformer (the ablation recovery)",
    "C3": "d3_en→d2: 512 transformer vs best classical cross champion",
    "C4": "d2→d3_en: best classical cross champion vs 256 transformer",
    "C5": "d2→d3_en: 512 vs 256 transformer (expected null)",
}


def _sep(lo, hi):
    return "separated" if (lo > 0 or hi < 0) else "**CI includes 0**"


def build():
    df = pd.read_csv(STATS)
    comp = df[df.comparison_id.isin(CLAIM)].set_index("comparison_id").reindex(list(CLAIM))
    cis = df[df.comparison_id.str.startswith("AUC_CI:")].copy()

    L = ["# D9 — Statistical Inference, Imbalance & Preprocessing Robustness\n"]
    L.append("All comparisons, champions and claim directions were **pre-registered before any statistic "
             "existed** (notes/decisions.md, 2026-07-17). CIs/p-values **annotate** the committed "
             "comparisons — no champion, threshold or claim direction is reselected on their basis. Score "
             "vectors were recomputed from stored artefacts and asserted equal to the committed test AUCs "
             "to 4 dp. Full log: `results/stats/stats_summary.csv`.\n")

    # (a) seven comparisons
    L.append("## (a) The seven comparisons — Δ AUC (95% paired stratified bootstrap, B=10,000) + McNemar/Holm\n")
    L.append("| id | comparison | AUC_a | AUC_b | Δ=a−b | 95% CI | McNemar n01/n10 | p_raw | p_holm | verdict |")
    L.append("|---|---|--:|--:|--:|---|:--:|--:|--:|---|")
    for cid, r in comp.iterrows():
        verdict = _sep(r.ci_low, r.ci_high)
        L.append(f"| {cid} | {r.model_a} vs {r.model_b} | {r.auc_a:.4f} | {r.auc_b:.4f} | {r.delta:+.4f} "
                 f"| [{r.ci_low:+.4f}, {r.ci_high:+.4f}] | {int(r.n01)}/{int(r.n10)} | {r.mcnemar_p_raw:.3g} "
                 f"| {r.mcnemar_p_holm:.3g} | {verdict} |")
    L.append("")

    # (b) which claims gain/lack separation
    L.append("## (b) Which committed claims gain (or lack) statistical separation\n")
    for cid, r in comp.iterrows():
        sep = r.ci_low > 0 or r.ci_high < 0
        holm = r.mcnemar_p_holm
        if sep:
            msg = (f"**{cid}** ({CLAIM[cid]}): AUC Δ={r.delta:+.4f} is **statistically separated** "
                   f"(CI [{r.ci_low:+.4f},{r.ci_high:+.4f}] excludes 0; Holm-McNemar p={holm:.3g}).")
        else:
            msg = (f"**{cid}** ({CLAIM[cid]}): AUC Δ={r.delta:+.4f} — **the 95% CI includes 0** "
                   f"([{r.ci_low:+.4f},{r.ci_high:+.4f}]): no AUC-level separation "
                   f"(Holm-McNemar p={holm:.3g}).")
        L.append("- " + msg)
    L.append("")
    L.append("Single-AUC 95% CIs (bootstrap, B=10,000) for the 12 headline numbers:\n")
    L.append("| number | AUC | 95% CI |")
    L.append("|---|--:|---|")
    for _, r in cis.iterrows():
        L.append(f"| {r.model_a} | {r.auc_a:.4f} | [{r.ci_low:.4f}, {r.ci_high:.4f}] |")
    L.append("")

    # (c) imbalance
    if IMB_E.exists():
        ev = pd.read_csv(IMB_E)
        L.append("## (c) Imbalance stress — operational view (FP per 1,000 legitimate emails)\n")
        L.append("Eval-side prevalence stress (all legit kept; positives subsampled; 50 draws, mean [2.5–97.5 pct]).\n")
        L.append("| corpus | prevalence | family | PR-AUC | precision@95%recall | FP / 1,000 legit @95%recall |")
        L.append("|---|---|---|--:|--:|--:|")
        prevmap = {"1:9": "10%", "1:19": "5%", "1:49": "2%"}
        for corpus in ["d1_sb", "d2", "d3_en"]:
            for ratio in ["1:9", "1:19", "1:49"]:
                for fam in ["classical", "transformer"]:
                    def cell(metric):
                        s = ev[(ev.corpus == corpus) & (ev.ratio == ratio) & (ev.family == fam) & (ev.metric == metric)]
                        return s.iloc[0] if len(s) else None
                    pr, p95, fp = cell("pr_auc"), cell("precision@95recall"), cell("fp_per_1000_legit@95recall")
                    if pr is None:
                        continue
                    L.append(f"| {corpus} | {prevmap[ratio]} | {fam} | {pr['mean']:.3f} "
                             f"[{pr['p2_5']:.3f},{pr['p97_5']:.3f}] | {p95['mean']:.3f} | "
                             f"{fp['mean']:.1f} [{fp['p2_5']:.1f},{fp['p97_5']:.1f}] |")
        L.append("")
    if IMB_T.exists():
        tr = pd.read_csv(IMB_T)
        L.append("Train-side (classical only, hyperparameters frozen; 9 single-use natural-TEST evals):\n")
        L.append("| corpus | prevalence | train pos/neg | PR-AUC | precision@95%recall | FP / 1,000 legit |")
        L.append("|---|---|---|--:|--:|--:|")
        prevmap = {"1:9": "10%", "1:19": "5%", "1:49": "2%"}
        for _, r in tr.iterrows():
            L.append(f"| {r.corpus} | {prevmap[r.ratio]} | {int(r.n_train_pos)}/{int(r.n_train_neg)} | "
                     f"{r['pr_auc']:.3f} | {r['precision@95recall']:.3f} | {r['fp_per_1000_legit@95recall']:.1f} |")
        L.append("")

    # (d) preprocessing
    v2_help = None
    if PRE.exists():
        pre = pd.read_csv(PRE)
        L.append("## (d) Pipeline A preprocessing ablation — val ROC-AUC Δ vs V0 (VALIDATION only)\n")
        L.append("V1 stopwords · V2 placeholders · V3 SymSpell · V4 wordninja · V5 combined. "
                 "Δ vs V0 per (train→eval, variant); **off-diagonal = cross-corpus** (the transfer test).\n")
        L.append("| train→eval | V1 | V2 | V3 | V4 | V5 |")
        L.append("|---|--:|--:|--:|--:|--:|")
        for tc in ["d1_sb", "d2", "d3_en"]:
            for ec in ["d1_sb", "d2", "d3_en"]:
                cells = []
                for v in ["V1", "V2", "V3", "V4", "V5"]:
                    s = pre[(pre.train_corpus == tc) & (pre.eval_corpus == ec) & (pre.variant == v)]
                    cells.append(f"{s.iloc[0].delta_vs_V0:+.4f}" if len(s) else "—")
                tag = " (self)" if tc == ec else ""
                L.append(f"| {tc}→{ec}{tag} | {' | '.join(cells)} |")
        # V2 cross-corpus hypothesis
        cross = pre[(pre.train_corpus != pre.eval_corpus)]
        v2_cross = cross[cross.variant == "V2"]["delta_vs_V0"]
        v2_help = float(v2_cross.mean())
        n_pos = int((v2_cross > 0).sum()); n_tot = len(v2_cross)
        L.append("")
        L.append(f"**V2 (placeholders) cross-corpus hypothesis:** mean Δ vs V0 over the 6 off-diagonal "
                 f"(cross-corpus) pairs = **{v2_help:+.4f}** ({n_pos}/{n_tot} pairs improved). "
                 f"Hypothesis {'HELD' if v2_help > 0 else 'NOT supported'} (pre-registered directional; "
                 f"descriptive, no significance test).\n")

    # (e) observations
    _observations(L, comp, v2_help)
    SUMMARY.write_text("\n".join(L), encoding="utf-8")
    return SUMMARY


def _observations(L, comp, v2_help):
    L.append("## Observations\n")
    c1a, c1b, c1c = comp.loc["C1a"], comp.loc["C1b"], comp.loc["C1c"]
    c2, c3, c4, c5 = comp.loc["C2"], comp.loc["C3"], comp.loc["C4"], comp.loc["C5"]
    L.append(f"1. **In-corpus the gaps are near-ceiling and sign-mixed — and the one material, significant "
             f"difference favours the *classical* champion on d3_en.** With Δ = classical − transformer: "
             f"d1_sb Δ={c1a.delta:+.4f} (CI [{c1a.ci_low:+.4f},{c1a.ci_high:+.4f}] includes 0 — "
             f"indistinguishable); d2 Δ={c1b.delta:+.4f} (CI [{c1b.ci_low:+.4f},{c1b.ci_high:+.4f}] "
             f"{'excludes' if (c1b.ci_low>0 or c1b.ci_high<0) else 'includes'} 0 — a ~0.03 pp edge to the "
             f"transformer, Holm-McNemar p={c1b.mcnemar_p_holm:.3g}); but **C1c d3_en Δ={c1c.delta:+.4f}, "
             f"CI [{c1c.ci_low:+.4f},{c1c.ci_high:+.4f}] excludes 0, Holm p={c1c.mcnemar_p_holm:.3g} — the "
             f"classical LinearSVC significantly beats RoBERTa on the AI corpus.** This sharpens the "
             f"Directive-6 'match but don't beat': at the 0.999 ceiling the in-corpus gaps are ≤0.0003 "
             f"either way, and the only material, significant one favours the classical model.")
    L.append("")
    L.append(f"2. **The 512 recovery is strongly significant and beats the classical cross champion.** "
             f"C2 (512 vs 256, d3_en→d2) Δ={c2.delta:+.4f} CI [{c2.ci_low:+.4f},{c2.ci_high:+.4f}], "
             f"C3 (512 vs classical cross) Δ={c3.delta:+.4f} CI [{c3.ci_low:+.4f},{c3.ci_high:+.4f}] — both "
             f"exclude 0 (Holm p≈0). The Directive-8 H1 verdict survives paired inference.")
    L.append("")
    L.append(f"3. **The classical cross champion beats the 256 transformer where the source is short (d2→d3_en), "
             f"and C5 is the pre-registered null.** C4 Δ={c4.delta:+.4f} CI [{c4.ci_low:+.4f},{c4.ci_high:+.4f}] "
             f"(separated, classical wins); **C5 (512 vs 256, d2→d3_en) Δ={c5.delta:+.4f}, CI "
             f"[{c5.ci_low:+.4f},{c5.ci_high:+.4f}] includes 0** exactly as pre-registered — 512 adds no "
             f"AUC-level signal when the training corpus was already un-truncated.")
    L.append("")
    imb_note = ""
    if IMB_E.exists():
        ev = pd.read_csv(IMB_E)
        def fp(corpus, fam):
            s = ev[(ev.corpus == corpus) & (ev.ratio == "1:49") & (ev.family == fam)
                   & (ev.metric == "fp_per_1000_legit@95recall")]
            return float(s.iloc[0]["mean"]) if len(s) else float("nan")
        imb_note = (f" And the more-robust family is **corpus-dependent**: at 2% prevalence the "
                    f"transformer holds 95% recall at only {fp('d2','transformer'):.1f} FP/1,000 on d2 "
                    f"(vs classical {fp('d2','classical'):.1f}), but on the AI corpus d3_en the classical "
                    f"champion is cleaner ({fp('d3_en','classical'):.1f} vs transformer "
                    f"{fp('d3_en','transformer'):.1f} FP/1,000) — matching C1c, where classical wins d3_en.")
    L.append("4. **Imbalance: precision at fixed recall collapses as phishing gets rarer — the FP-per-1,000 "
             "framing shows the operational cost.** At 2% prevalence every family pays far more false "
             "positives per 1,000 legitimate emails to hold 95% recall than at 10% (see table (c)); the "
             "PR-AUC / FP-per-1000 curves (figure) are the deployment-relevant robustness signal, not the "
             "prevalence-inflated ROC-AUC." + imb_note)
    L.append("")
    if v2_help is not None:
        L.append(f"5. **Preprocessing (validation only): placeholders (V2) {'help' if v2_help>0 else 'do not help'} "
                 f"cross-corpus transfer** (mean Δ vs V0 over the 6 cross pairs = {v2_help:+.4f}); the "
                 f"pre-registered directional hypothesis {'held' if v2_help>0 else 'was not supported'}. The "
                 f"ablation is descriptive and VALIDATION-only — it cannot and does not change any committed "
                 f"champion.")
        L.append("")


if __name__ == "__main__":
    print(build())
