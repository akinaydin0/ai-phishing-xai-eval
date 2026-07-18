"""
Directive 10 - Part A.2: deterministic HITL document selection (ids only, no ratings).

From the committed explained_ids TEST sets only. Per corpus (d1_sb, d2, d3_en): 20 phishing-
PREDICTED docs = 15 true positives + up to 5 false positives (fill with TPs if fewer FPs exist).
Deterministic: candidates sorted by doc id (sha1 body hash) ascending, stratified first-N; the
underlying explained_ids were built with the global seed 42 (recorded for provenance - no further
stochastic step is taken here).

- d1_sb / d2: source = the tfidf champion's explained_ids; phishing-PRED = tfidf pred == 1.
  Only the tfidf family is annotated for these corpora.
- d3_en: source = the INTERSECTION of tfidf_d3_en and transformer_d3_en explained_ids, restricted
  to docs phishing-PRED by BOTH champions. Both families (tfidf, transformer) are annotated for
  d3_en, so every selected doc must be a committed-explained, phishing-predicted doc for BOTH -
  this keeps every generated set (tfidf SHAP/LIME/RANDOM, transformer SHAP/LIME/IG/RANDOM) a
  top-10-toward-phishing set on a phishing-predicted, committed-explained document (surface S1).

Writes results/hitl/selected_ids.json. Pure JSON read of the committed explained_ids - no model
build, no attribution. Asserts every selected hash is inside the committed explained_ids.

Run: python -m src.experiments.d10_select
"""

from __future__ import annotations

import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
XAI_DIR = _ROOT / "results" / "xai"
OUT_DIR = _ROOT / "results" / "hitl"
SEED = 42
N_TP = 15          # mandatory true positives per corpus
N_FP = 5           # up to this many false positives; shortfall filled with TPs
N_TOTAL = 20

# per-corpus source spec: (corpus_short, [units whose explained_ids must all contain the doc],
#                          families annotated, explained_corpus id)
CORPORA = [
    ("d1_sb", ["tfidf_d1_sb"], ["tfidf"], "d1_decontaminated_sb"),
    ("d2", ["tfidf_d2"], ["tfidf"], "d2"),
    ("d3_en", ["tfidf_d3_en", "transformer_d3_en"], ["tfidf", "transformer"], "d3_en"),
]


def _load(unit: str) -> dict:
    return json.loads((XAI_DIR / f"explained_ids_{unit}.json").read_text(encoding="utf-8"))


def _unit_map(unit: str) -> dict:
    """hash -> {'idx':, 'pred':, 'y_true':} for a committed explained_ids set."""
    es = _load(unit)
    return {h: {"idx": int(i), "pred": int(p), "y_true": int(y)}
            for h, i, p, y in zip(es["hashes"], es["idx"], es["pred"], es["y_true"])}


def select_corpus(corpus: str, units: list[str], families: list[str], explained_corpus: str) -> dict:
    maps = {u: _unit_map(u) for u in units}
    primary = units[0]
    # candidate hashes: in EVERY unit's explained_ids AND phishing-pred (pred==1) by EVERY unit
    cand = [h for h in maps[primary]
            if all(h in maps[u] and maps[u][h]["pred"] == 1 for u in units)]
    y_of = lambda h: maps[primary][h]["y_true"]
    tp = sorted(h for h in cand if y_of(h) == 1)   # sorted by hash ascending
    fp = sorted(h for h in cand if y_of(h) == 0)

    n_fp = min(N_FP, len(fp))
    n_tp = N_TOTAL - n_fp                            # 15 normally; more if FPs are short
    if len(tp) < n_tp:
        raise RuntimeError(f"{corpus}: only {len(tp)} TP candidates, need {n_tp}")
    chosen_tp = tp[:n_tp]
    chosen_fp = fp[:n_fp]
    chosen = sorted(chosen_tp + chosen_fp)          # final list sorted by hash

    docs = []
    for h in chosen:
        rec = {"doc_uid": f"{corpus}_{h[:12]}", "corpus": corpus, "hash": h,
               "idx": maps[primary][h]["idx"], "y_true": y_of(h),
               "stratum": "TP" if y_of(h) == 1 else "FP", "families": families}
        for u in units:
            rec[f"pred__{u}"] = maps[u][h]["pred"]
        docs.append(rec)

    # S1 assertion: every chosen hash is inside every committed explained_ids consumed
    for u in units:
        commit_hashes = set(_load(u)["hashes"])
        assert all(d["hash"] in commit_hashes for d in docs), f"{corpus}: doc not in {u}"
    return {"source_units": units, "families": families, "explained_corpus": explained_corpus,
            "n_tp": len(chosen_tp), "n_fp": len(chosen_fp), "n": len(docs), "docs": docs}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {"directive": "D10", "seed": SEED,
           "selection_rule": ("committed explained_ids TEST sets only; per corpus 15 TP + up to "
                              "5 FP (fill TP if <5 FP); candidates phishing-pred by every source "
                              "unit and present in every source unit's explained_ids; sorted by "
                              "sha1 body hash ascending, stratified first-N; seed 42 provenance"),
           "n_tp_target": N_TP, "n_fp_target": N_FP, "n_total_per_corpus": N_TOTAL, "corpora": {}}
    total = 0
    for corpus, units, families, exc in CORPORA:
        sel = select_corpus(corpus, units, families, exc)
        out["corpora"][corpus] = sel
        total += sel["n"]
        print(f"  {corpus:6s} n={sel['n']:2d} (TP={sel['n_tp']} FP={sel['n_fp']}) "
              f"families={families} from {units}", flush=True)
    out["n_docs_total"] = total
    (OUT_DIR / "selected_ids.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    # set accounting (for cross-check with the pre-registration arithmetic)
    n_tfidf_sets = total * 3                          # SHAP/LIME/RANDOM for every doc
    n_xf_sets = out["corpora"]["d3_en"]["n"] * 4      # SHAP/LIME/IG/RANDOM for d3_en docs
    print(f"  TOTAL docs={total}  tfidf_sets={n_tfidf_sets}  transformer_sets={n_xf_sets}  "
          f"all_sets={n_tfidf_sets + n_xf_sets}", flush=True)
    print(f"  wrote {OUT_DIR / 'selected_ids.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
