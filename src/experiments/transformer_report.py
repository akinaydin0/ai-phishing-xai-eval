"""
Generate notes/transformer_summary.md from the transformer artifacts:
  results/results.csv, results/transformers/<model>__<corpus>.json,
  results/transformer_champions.json.

Sections: (a) in-corpus vs classical champions, (b) cross-corpus vs classical champions
(the AI→real headline), (c) truncation @256, (d) computational cost (fine-tune minutes vs
sklearn seconds), and 5 house-style observations.

Run: python -m src.experiments.transformer_report
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = _PROJECT_ROOT / "results"
OUT_DIR = RESULTS / "transformers"
SUMMARY = _PROJECT_ROOT / "notes" / "transformer_summary.md"

CORPORA = ["d1_decontaminated_sb", "d2", "d3_en"]
SHORT = {"d1_decontaminated_sb": "d1_sb", "d2": "d2", "d3_en": "d3_en"}
MODELS = ["distilbert-base-uncased", "bert-base-uncased", "roberta-base"]
DISPLAY = {"distilbert-base-uncased": "DistilBERT", "bert-base-uncased": "BERT-base",
           "roberta-base": "RoBERTa-base"}


def _safe(m):
    return m.replace("/", "_")


def _finalist(model_name, corpus):
    p = OUT_DIR / f"{_safe(model_name)}__{corpus}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _classical_champion(df, corpus):
    """In-corpus leaderboard #1 among tuned classical + dense (matches hpo_summary)."""
    inc = df[(df.train_corpus == corpus) & (df.test_corpus == corpus)
             & (df.run_id.str.endswith("__tuned") | df.run_id.str.contains("__dense"))]
    r = inc.loc[inc.roc_auc.idxmax()]
    return r.model, float(r.f1), float(r.roc_auc)


def _classical_cross(df, src, tgt):
    """Best classical champion AUC/F1 for a directed pair, across tfidf/ensemble/dense; + range.

    Excludes the transformer cross rows (`__xcorpus_transformer`) so the comparison bar is the
    classical champions only, not the transformer against itself.
    """
    xc = df[(df.train_corpus == src) & (df.test_corpus == tgt)
            & df.run_id.str.contains("__xcorpus_")
            & ~df.run_id.str.contains("__xcorpus_transformer")]
    if not len(xc):
        return None
    best = xc.loc[xc.roc_auc.idxmax()]
    grp = best.run_id.split("__xcorpus_")[1]
    return {"auc": float(best.roc_auc), "f1": float(best.f1), "model": best.model,
            "group": grp, "auc_min": float(xc.roc_auc.min()), "auc_max": float(xc.roc_auc.max())}


def _transformer_cross(df, src, tgt):
    xc = df[(df.train_corpus == src) & (df.test_corpus == tgt)
            & df.run_id.str.contains("__xcorpus_transformer")]
    if not len(xc):
        return None
    r = xc.iloc[0]
    return {"auc": float(r.roc_auc), "f1": float(r.f1), "model": r.model}


def build():
    df = pd.read_csv(RESULTS / "results.csv")
    champ_path = RESULTS / "transformer_champions.json"
    champions = json.loads(champ_path.read_text(encoding="utf-8")) if champ_path.exists() else {}
    fin = {(m, c): _finalist(m, c) for m in MODELS for c in CORPORA}

    L = ["# Transformer Fine-Tuning — Summary\n"]
    L.append("Fine-tuned DistilBERT, BERT-base and RoBERTa-base on the three frozen corpora "
             "(subject+body, max_length=256). Per (model, corpus) grid: learning-rate ∈ "
             "{2e-5, 3e-5, 5e-5}, up to 4 epochs, batch 32 + fp16, weight_decay 0.01, "
             "warmup_ratio 0.1; val ROC-AUC each epoch, early-stop patience 1, best-val-AUC "
             "checkpoint kept. **The lr is selected by validation AUC and the TEST split is "
             "evaluated exactly once per finalist.** Determinism: seeds=42, cudnn.deterministic, "
             "pinned versions (recorded per run json); GPU fine-tuning is not bit-exact (see "
             "notes/decisions.md). Full log: `results/results.csv`.\n")

    # ---------------- (a) in-corpus vs classical champions ----------------
    L.append("## (a) In-corpus: transformers vs. classical champions (test set)\n")
    L.append("| Corpus | DistilBERT | BERT-base | RoBERTa-base | Best transformer | Classical champion | Δ AUC (T−C) |")
    L.append("|---|--:|--:|--:|---|---|--:|")
    for c in CORPORA:
        cells, best_auc, best_name = [], -1, None
        for m in MODELS:
            fj = fin[(m, c)]
            if fj:
                a = fj["test"]["roc_auc"]
                cells.append(f"{a:.4f}")
                if a > best_auc:
                    best_auc, best_name = a, DISPLAY[m]
            else:
                cells.append("—")
        cm, cf1, ca = _classical_champion(df, c)
        delta = best_auc - ca if best_auc > 0 else float("nan")
        L.append(f"| {SHORT[c]} | {cells[0]} | {cells[1]} | {cells[2]} | {best_name} ({best_auc:.4f}) "
                 f"| {cm} ({ca:.4f}) | {delta:+.4f} |")
    L.append("")
    L.append("*Values are test ROC-AUC. The classical champion is the in-corpus leaderboard #1 "
             "(tuned classical/dense) from the HPO phase.*\n")

    # per-model F1/AUC + selected lr detail
    L.append("| Corpus | Model | sel. lr | val AUC | test F1 | test AUC | best epoch |")
    L.append("|---|---|--:|--:|--:|--:|--:|")
    for c in CORPORA:
        for m in MODELS:
            fj = fin[(m, c)]
            if not fj:
                continue
            lr = fj["lr_results"][repr(fj["selected_lr"])]
            L.append(f"| {SHORT[c]} | {DISPLAY[m]} | {fj['selected_lr']:g} | {fj['best_val_auc']:.4f} | "
                     f"{fj['test']['f1']:.4f} | {fj['test']['roc_auc']:.4f} | {lr['best_epoch']} |")
    L.append("")

    # ---------------- (b) cross-corpus vs classical ----------------
    L.append("## (b) Cross-corpus: transformer champions vs. classical champions\n")
    L.append("Best transformer per **source** (by val AUC) applied to the other corpora's TEST "
             "splits. Compared against the best classical champion (tfidf/ensemble/dense) for the "
             "same directed pair.\n")
    L.append("| train → test | Transformer champ | T AUC | T F1 | Classical best | C AUC | Δ AUC (T−C) |")
    L.append("|---|---|--:|--:|---|--:|--:|")
    for src in CORPORA:
        for tgt in CORPORA:
            if src == tgt:
                continue
            tc = _transformer_cross(df, src, tgt)
            cc = _classical_cross(df, src, tgt)
            if tc is None or cc is None:
                continue
            d = tc["auc"] - cc["auc"]
            L.append(f"| {SHORT[src]} → {SHORT[tgt]} | {tc['model']} | {tc['auc']:.4f} | {tc['f1']:.4f} "
                     f"| {cc['model']} ({cc['group']}) | {cc['auc']:.4f} | {d:+.4f} |")
    L.append("")

    # headline: AI -> real (d3_en -> d2)
    tc = _transformer_cross(df, "d3_en", "d2")
    cc = _classical_cross(df, "d3_en", "d2")
    if tc and cc:
        verdict = "closes" if tc["auc"] >= cc["auc_max"] else "does not close"
        L.append(f"**Headline (AI→real, d3_en→d2).** Classical champions spanned AUC "
                 f"{cc['auc_min']:.3f}–{cc['auc_max']:.3f}; the transformer champion ({tc['model']}) "
                 f"reaches **{tc['auc']:.3f}** — inside that band but **{verdict}** the AI-to-real gap "
                 f"(Δ {tc['auc'] - cc['auc_max']:+.3f} vs the best classical, {cc['model']} "
                 f"{cc['auc_max']:.3f}). See observations 2–3 for the full cross-corpus picture.\n")

    # ---------------- (c) truncation ----------------
    L.append("## (c) Truncation at max_length=256 (% of documents truncated)\n")
    L.append("WordPiece = BERT-base & DistilBERT (shared uncased vocab, identical lengths); "
             "BPE = RoBERTa.\n")
    L.append("| Corpus | Split | WordPiece trunc% | BPE trunc% | median len (WP) | p95 len (WP) | max len (WP) |")
    L.append("|---|---|--:|--:|--:|--:|--:|")
    for c in CORPORA:
        wp = fin[("bert-base-uncased", c)] or fin[("distilbert-base-uncased", c)]
        bpe = fin[("roberta-base", c)]
        if not wp:
            continue
        for sp in ("train", "val", "test"):
            w = wp["truncation"][sp]
            b = bpe["truncation"][sp]["truncated_pct"] if bpe else float("nan")
            L.append(f"| {SHORT[c]} | {sp} | {w['truncated_pct']} | {b} | {w['median_len']} | "
                     f"{w['p95_len']} | {w['max_len_obs']} |")
    L.append("")

    # ---------------- (d) computational cost ----------------
    L.append("## (d) Computational cost — fine-tune minutes vs. sklearn seconds\n")
    L.append("| Corpus | Model | finalist wall (s) | unit wall (3 lr, s) | peak GPU (MB) | vs classical champ fit (s) |")
    L.append("|---|---|--:|--:|--:|--:|")
    for c in CORPORA:
        # classical champion single-fit runtime from its tuned row
        cm, _, _ = _classical_champion(df, c)
        crow = df[(df.model == cm) & (df.train_corpus == c) & (df.test_corpus == c)
                  & df.run_id.str.endswith("__tuned")]
        cfit = float(crow.iloc[0].runtime_s) if len(crow) else float("nan")
        for m in MODELS:
            fj = fin[(m, c)]
            if not fj:
                continue
            cost = fj["cost"]
            L.append(f"| {SHORT[c]} | {DISPLAY[m]} | {cost['finalist_wall_s']:.0f} | "
                     f"{cost['unit_wall_s']:.0f} | {cost['peak_gpu_mem_mb']:.0f} | "
                     f"{cm} {cfit:.2f}s |")
    L.append("")
    total_unit = sum(fj["cost"]["unit_wall_s"] for fj in fin.values() if fj)
    ckpt_dir = OUT_DIR / "checkpoints"
    ckpt_bytes = sum(p.stat().st_size for p in ckpt_dir.rglob("*") if p.is_file()) if ckpt_dir.exists() else 0
    ckpt_gb = ckpt_bytes / 1e9
    L.append(f"*Total fine-tuning wall-clock across the 9 (model, corpus) grids: "
             f"**{total_unit / 60:.1f} min** ({total_unit / 3600:.2f} h) on one RTX 3090. "
             f"The tuned classical champions fit in **seconds** — the accuracy-vs-compute trade-off "
             f"the dissertation quantifies.*\n")
    L.append(f"*The 9 finalist checkpoints total **{ckpt_gb:.1f} GB** on disk "
             f"(`results/transformers/checkpoints/`, git-ignored as regenerable artifacts); they are "
             f"reused by the cross-corpus transfer above and by the upcoming XAI phase.*\n")

    # ---------------- observations ----------------
    L.append("## Observations\n")

    # gather numbers
    inc_best = {}
    for c in CORPORA:
        b_auc, b_name = -1, None
        for m in MODELS:
            fj = fin[(m, c)]
            if fj and fj["test"]["roc_auc"] > b_auc:
                b_auc, b_name = fj["test"]["roc_auc"], DISPLAY[m]
        cm, cf1, ca = _classical_champion(df, c)
        inc_best[c] = (b_name, b_auc, cm, ca)

    o1 = ("1. **In-corpus, fine-tuned transformers match but do not beat the tuned linear "
          "TF-IDF champions — at ~100–1000× the compute.** ")
    parts = []
    for c in CORPORA:
        bn, ba, cm, ca = inc_best[c]
        parts.append(f"{SHORT[c]}: best transformer {bn} {ba:.4f} vs {cm} {ca:.4f}")
    o1 += "; ".join(parts) + (". On these near-ceiling corpora (classical AUC ≥ 0.999) contextual "
          "modelling buys no in-corpus discrimination; its promise is on the harder cross-corpus "
          "transfer, tested next.")
    L.append(o1); L.append("")

    # cross summary — build the 6 directed pairs (transformer vs best classical)
    xt = df[df.run_id.str.contains("__xcorpus_transformer")]
    pairs = []
    for src in CORPORA:
        for tgt in CORPORA:
            if src == tgt:
                continue
            tcp, ccp = _transformer_cross(df, src, tgt), _classical_cross(df, src, tgt)
            if tcp and ccp:
                pairs.append({"src": src, "tgt": tgt, "t": tcp["auc"], "c": ccp["auc"],
                              "d": tcp["auc"] - ccp["auc"], "cmodel": ccp["model"], "cgrp": ccp["group"]})
    if pairs:
        wins = [p for p in pairs if p["d"] > 0]
        losses = [p for p in pairs if p["d"] <= 0]
        hd = _transformer_cross(df, "d3_en", "d2"); hc = _classical_cross(df, "d3_en", "d2")
        d3d1 = next((p for p in pairs if p["src"] == "d3_en" and p["tgt"] == "d1_decontaminated_sb"), None)
        loss_desc = ", ".join(f"{SHORT[p['src']]}→{SHORT[p['tgt']]} {p['t']:.3f}" for p in losses)

        L.append(f"2. **The headline AI→real gap (d3_en→d2) is NOT closed by contextual modelling.** "
                 f"Classical champions transferred d3_en→d2 at AUC {hc['auc_min']:.3f}–{hc['auc_max']:.3f}; "
                 f"the transformer champion ({hd['model']}) reaches {hd['auc']:.3f} — "
                 f"inside that band and {hd['auc'] - hc['auc_max']:+.3f} below the best classical "
                 f"({hc['model']} {hc['auc_max']:.3f}). On this specific legacy-Enron target, contextual "
                 f"token representations do not beat a tuned linear model — the hardest transfer stays hard.")
        L.append("")
        L.append(f"3. **But transformers win every transfer touching the modern-real corpus, and lose "
                 f"only the two legacy-Enron↔AI directions — reframing the asymmetry.** Transformer "
                 f"champions beat the best classical champion on {len(wins)}/6 directed pairs; all four "
                 f"pairs involving d1_sb (modern real) improve — most strikingly d3_en→d1_sb "
                 f"{d3d1['t']:.3f} vs {d3d1['c']:.3f} ({d3d1['d']:+.3f}). The only two losses are "
                 f"{loss_desc} — i.e. Enron↔AI in both directions. The residual hard case is "
                 f"legacy↔AI transfer, not AI→real in general; anchoring on modern-real data is where "
                 f"contextual modelling clearly helps.")
        L.append("")
    else:
        L.append("2. *(cross-corpus pending — run `python -m src.experiments.transformer_cross`.)*\n")
        L.append("3. *(cross-corpus pending.)*\n")

    # truncation observation (data-driven across all three corpora)
    def _wp_train(c):
        fj = fin[("bert-base-uncased", c)] or fin[("distilbert-base-uncased", c)]
        return fj["truncation"]["train"] if fj else {}
    t1, t2, t3 = _wp_train("d1_decontaminated_sb"), _wp_train("d2"), _wp_train("d3_en")
    L.append(f"4. **Truncation at 256 bites hardest on the AI corpus, not the legacy one — AI-generated "
             f"emails are the longest.** d3_en truncates ~{t3.get('truncated_pct')}% of train documents at "
             f"256 tokens (median {t3.get('median_len')} WordPiece tokens — the longest of the three), "
             f"versus d1_sb ~{t1.get('truncated_pct')}% (median {t1.get('median_len')}) and d2 "
             f"~{t2.get('truncated_pct')}% (median {t2.get('median_len')}, the shortest). The 256 cap "
             f"discards the most context on the AI corpus, yet its champion (RoBERTa) still tops d3_en "
             f"in-corpus and transfers best to d1_sb — the discriminative signal survives in the first "
             f"256 tokens. Extreme outliers exist (d1_sb max >{t1.get('max_len_obs', 0):,} WP tokens — "
             f"giant HTML/quoted threads), so median/p95 are the honest summaries.")
    L.append("")

    L.append(f"5. **Compute cost is the transformers' real liability; the payoff is cross-corpus "
             f"robustness on modern-real data, not in-corpus accuracy.** The 9-grid took "
             f"{total_unit / 60:.0f} min on the RTX 3090 versus seconds for the tuned linear champions, "
             f"with no in-corpus gain (all ≥ 0.997 AUC). Their value shows only in transfer to/from the "
             f"modern-real corpus (up to {max((p['d'] for p in pairs), default=0):+.3f} AUC over the best "
             f"classical); for pure in-corpus deployment the accuracy-per-FLOP verdict still favours linear "
             f"TF-IDF. The saved finalist checkpoints feed the next phase (XAI: contextual explanations).")
    L.append("")

    SUMMARY.write_text("\n".join(L), encoding="utf-8")
    return SUMMARY


if __name__ == "__main__":
    print(build())
