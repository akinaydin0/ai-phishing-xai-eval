"""
Directive 10 - Part D: analysis. Join human_ratings + judge_ratings + key + lexicon labels and
compute the pre-registered metrics P1-P4 under the fixed Landis-Koch bands. Writes
results/hitl/hitl_token_level.csv, hitl_set_level.csv, hitl_metrics.json and prints a summary.

Consumes the COMMITTED results/hitl/human_ratings.csv byte-identically (surface S3). Lexicon label
per token = cue_lexicon.is_cue(word) OR the token's char span overlaps a URL/e-mail span - identical
to the committed metric's `_cue_hits`. judge_missing sets (model refusals / unparseable JSON) are
excluded from P3/P4-judge and counted.

Run: python -m src.experiments.d10_analyze
"""

from __future__ import annotations

import csv as _csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

try:
    from src.xai import cue_lexicon, textops
except ImportError:  # pragma: no cover
    from xai import cue_lexicon, textops

_ROOT = Path(__file__).resolve().parent.parent.parent
HITL = _ROOT / "results" / "hitl"
B_BOOT = 10_000
SEED = 42
# method-blinding procedurally broken for this one doc by orchestrator progress reporting (amendment 3);
# Part D reports P1-P3 with AND without it as a sensitivity analysis.
COMPROMISED_DOC = "d1_sb_017547c4cef2"
FAMILY_METHODS = {"tfidf": ["shap", "lime", "random"],
                  "transformer": ["shap", "lime", "ig", "random"]}
# concordance contrasts (m_a vs m_b); IG contrasts only exist for the transformer family
CONTRASTS = {"tfidf": [("shap", "random"), ("lime", "random"), ("shap", "lime")],
             "transformer": [("shap", "random"), ("lime", "random"), ("shap", "lime"),
                             ("ig", "random"), ("ig", "shap"), ("ig", "lime")]}


def band(k: float) -> str:
    if k != k:
        return "undefined"
    if k >= 0.4:
        return "moderate"
    if k >= 0.2:
        return "fair"
    return "poor"


def lexicon_label(text: str, position: int) -> int:
    spans = textops.word_spans(text)
    if position >= len(spans):
        return 0
    s, e = spans[position]
    if cue_lexicon.is_cue(text[s:e]):
        return 1
    return 1 if any(a < e and s < b for a, b in cue_lexicon.url_email_spans(text)) else 0


def _load_ratings(path: Path):
    cues, hol = {}, {}
    with open(path, encoding="utf-8", newline="") as f:
        for r in _csv.DictReader(f):
            cues[(r["set_uid"], int(r["token_idx"]))] = int(r["is_cue"])
            hol[r["set_uid"]] = int(r["holistic"])
    return cues, hol


def _boot_ci(vals, B=B_BOOT, seed=SEED):
    v = np.asarray(vals, float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    m = v[rng.integers(0, v.size, size=(B, v.size))].mean(1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def _boot_diff_ci(a, b, B=B_BOOT, seed=SEED):
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    n = a.size
    idx = rng.integers(0, n, size=(B, n))
    d = a[idx].mean(1) - b[idx].mean(1)
    return float(a.mean() - b.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def analyze() -> dict:
    key = json.loads((HITL / "key.json").read_text(encoding="utf-8"))
    sets, docs = key["sets"], key["docs"]
    human_cue, human_hol = _load_ratings(HITL / "human_ratings.csv")
    judge_cue, judge_hol = _load_ratings(HITL / "judge_ratings.csv")
    judge_sets = set(judge_hol)   # sets the judge successfully rated (ok)

    # ---- token- and set-level joined tables ----
    tok_rows, set_rows = [], []
    for su in sorted(sets):
        s = sets[su]
        text = docs[s["doc_uid"]]["text"]
        corpus, family, method = s["corpus"], s["family"], s["method"]
        lex_v, hum_v, jud_v = [], [], []
        for ti, pos in enumerate(s["positions"]):
            lex = lexicon_label(text, pos)
            hum = human_cue.get((su, ti))
            jud = judge_cue.get((su, ti))
            lex_v.append(lex)
            hum_v.append(hum)
            jud_v.append(jud)
            tok_rows.append({"doc_uid": s["doc_uid"], "set_uid": su, "corpus": corpus,
                             "family": family, "method": method, "token_idx": ti,
                             "token": s["tokens"][ti], "position": pos, "lexicon": lex,
                             "human": "" if hum is None else hum, "judge": "" if jud is None else jud})
        n = len(s["positions"])
        judged = su in judge_sets
        set_rows.append({"set_uid": su, "doc_uid": s["doc_uid"], "corpus": corpus, "family": family,
                         "method": method, "n_tokens": n,
                         "human_cuerate": np.mean([h for h in hum_v if h is not None]) if any(h is not None for h in hum_v) else "",
                         "judge_cuerate": (np.mean(jud_v) if judged else ""),
                         "lexicon_cuerate": float(np.mean(lex_v)),
                         "human_holistic": human_hol.get(su, ""),
                         "judge_holistic": (judge_hol.get(su) if judged else ""),
                         "judge_status": "ok" if judged else "judge_missing"})

    _write_csv(HITL / "hitl_token_level.csv", tok_rows,
               ["doc_uid", "set_uid", "corpus", "family", "method", "token_idx", "token",
                "position", "lexicon", "human", "judge"])
    _write_csv(HITL / "hitl_set_level.csv", set_rows,
               ["set_uid", "doc_uid", "corpus", "family", "method", "n_tokens", "human_cuerate",
                "judge_cuerate", "lexicon_cuerate", "human_holistic", "judge_holistic", "judge_status"])

    metrics = {"accounting": _accounting(set_rows, judge_sets, sets),
               "P1": _p1(set_rows), "P2": _p2(tok_rows), "P3": _p3(tok_rows, set_rows),
               "P4": _p4(set_rows), "expectations": {}, "exploratory": {}}
    _expectations(metrics)
    _exploratory(metrics, set_rows)

    # sensitivity: recompute P1-P3 EXCLUDING the method-blinding-compromised doc (amendment 3)
    tok_ex = [r for r in tok_rows if r["doc_uid"] != COMPROMISED_DOC]
    set_ex = [r for r in set_rows if r["doc_uid"] != COMPROMISED_DOC]
    sp1, sp2 = _p1(set_ex), _p2(tok_ex)
    metrics["sensitivity_exclude_compromised"] = {
        "excluded_doc": COMPROMISED_DOC,
        "reason": "method-blinding broken by orchestrator progress reporting (amendment 3)",
        "n_docs_excluded": 1, "P1": sp1, "P2": sp2, "P3": _p3(tok_ex, set_ex),
        "expectations": _expect(sp1, sp2)}

    (HITL / "hitl_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _print_summary(metrics)
    return metrics


def _write_csv(path, rows, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _accounting(set_rows, judge_sets, sets):
    return {"n_sets": len(sets), "n_judge_ok": len(judge_sets),
            "n_judge_missing": len(sets) - len(judge_sets),
            "n_tokens": sum(r["n_tokens"] for r in set_rows)}


def _cuerate_by_doc(set_rows, corpus, family, method, field):
    out = {}
    for r in set_rows:
        if r["corpus"] == corpus and r["family"] == family and r["method"] == method:
            v = r[field]
            if v != "":
                out[r["doc_uid"]] = float(v)
    return out


def _p1(set_rows):
    """Human cue-rate@10 per (corpus, family, method) with bootstrap CIs + lexicon concordance."""
    cells = sorted({(r["corpus"], r["family"]) for r in set_rows})
    out = {"cue_rates": {}, "concordance": {}}
    for corpus, family in cells:
        key = f"{corpus}|{family}"
        out["cue_rates"][key] = {}
        hum_by = {m: _cuerate_by_doc(set_rows, corpus, family, m, "human_cuerate")
                  for m in FAMILY_METHODS[family]}
        lex_by = {m: _cuerate_by_doc(set_rows, corpus, family, m, "lexicon_cuerate")
                  for m in FAMILY_METHODS[family]}
        docs_order = sorted(hum_by["random"])
        for m in FAMILY_METHODS[family]:
            mean, lo, hi = _boot_ci([hum_by[m][d] for d in docs_order])
            out["cue_rates"][key][m] = {"human_mean": round(mean, 4),
                                        "ci95": [round(lo, 4), round(hi, 4)],
                                        "lexicon_mean": round(float(np.mean([lex_by[m][d] for d in docs_order])), 4)}
        out["concordance"][key] = {}
        for a, b in CONTRASTS[family]:
            ha = [hum_by[a][d] for d in docs_order]
            hb = [hum_by[b][d] for d in docs_order]
            hd, hlo, hhi = _boot_diff_ci(ha, hb)
            ld = float(np.mean([lex_by[a][d] for d in docs_order]) - np.mean([lex_by[b][d] for d in docs_order]))
            out["concordance"][key][f"{a}_vs_{b}"] = {
                "human_diff": round(hd, 4), "human_diff_ci95": [round(hlo, 4), round(hhi, 4)],
                "lexicon_diff": round(ld, 4),
                "sign_agree": bool(np.sign(hd) == np.sign(ld)) if (hd != 0 and ld != 0) else (hd == 0 and ld == 0)}
    return out


def _kappa_agree(pairs):
    a = [x for x, _ in pairs]
    b = [y for _, y in pairs]
    if not a:
        return {"n": 0, "pct_agree": float("nan"), "kappa": float("nan"), "band": "undefined"}
    agree = float(np.mean([x == y for x, y in pairs]))
    try:
        k = float(cohen_kappa_score(a, b))
    except Exception:
        k = float("nan")
    return {"n": len(a), "pct_agree": round(agree, 4), "kappa": round(k, 4), "band": band(k)}


def _p2(tok_rows):
    """Token-level lexicon-vs-human agreement, per corpus and overall."""
    out = {"overall": _kappa_agree([(r["lexicon"], r["human"]) for r in tok_rows if r["human"] != ""])}
    for corpus in sorted({r["corpus"] for r in tok_rows}):
        out[corpus] = _kappa_agree([(r["lexicon"], r["human"]) for r in tok_rows
                                    if r["corpus"] == corpus and r["human"] != ""])
    return out


def _p3(tok_rows, set_rows):
    """Human-vs-LLM: token-level kappa (judged sets only) + set-level Spearman."""
    tok = _kappa_agree([(r["human"], r["judge"]) for r in tok_rows
                        if r["human"] != "" and r["judge"] != ""])
    judged = [r for r in set_rows if r["judge_status"] == "ok"
              and r["human_cuerate"] != "" and r["judge_cuerate"] != ""]
    hc = [float(r["human_cuerate"]) for r in judged]
    jc = [float(r["judge_cuerate"]) for r in judged]
    hh = [float(r["human_holistic"]) for r in judged]
    jh = [float(r["judge_holistic"]) for r in judged]
    def _sp(x, y):
        if len(x) < 3:
            return {"rho": float("nan"), "n": len(x)}
        rho = spearmanr(x, y).correlation
        return {"rho": round(float(rho), 4) if rho == rho else float("nan"), "n": len(x)}
    return {"token_kappa": tok, "set_cuerate_spearman": _sp(hc, jc),
            "set_holistic_spearman": _sp(hh, jh)}


def _p4(set_rows):
    out = {}
    for corpus, family in sorted({(r["corpus"], r["family"]) for r in set_rows}):
        key = f"{corpus}|{family}"
        out[key] = {}
        for m in FAMILY_METHODS[family]:
            hs = [float(r["human_holistic"]) for r in set_rows
                  if r["corpus"] == corpus and r["family"] == family and r["method"] == m and r["human_holistic"] != ""]
            js = [float(r["judge_holistic"]) for r in set_rows
                  if r["corpus"] == corpus and r["family"] == family and r["method"] == m and r["judge_status"] == "ok"]
            out[key][m] = {"human_holistic_mean": round(float(np.mean(hs)), 4) if hs else float("nan"),
                           "judge_holistic_mean": round(float(np.mean(js)), 4) if js else float("nan")}
    return out


def _expect(P1, P2):
    """Pure expectations E1/E2 from a P1/P2 pair (reused for the full run and the sensitivity run)."""
    e1 = {cell: bool(m["shap"]["human_mean"] > m["random"]["human_mean"])
          for cell, m in P1["cue_rates"].items()}
    k = P2["overall"]["kappa"]
    return {"E1_shap_gt_random": {"per_cell": e1, "holds_all": all(e1.values())},
            "E2_lexicon_human_kappa": {"kappa": k, "band": P2["overall"]["band"],
                                       "holds": bool(k == k and k >= 0.4)}}


def _expectations(metrics):
    metrics["expectations"] = _expect(metrics["P1"], metrics["P2"])


def _exploratory(metrics, set_rows):
    # d3_en transformer: does LIME > RANDOM replicate in human (and judge) ratings?
    cell = "d3_en|transformer"
    cr = metrics["P1"]["cue_rates"].get(cell, {})
    jud = {}
    for m in ("lime", "random", "shap"):
        js = [float(r["judge_cuerate"]) for r in set_rows
              if r["corpus"] == "d3_en" and r["family"] == "transformer" and r["method"] == m
              and r["judge_status"] == "ok"]
        jud[m] = round(float(np.mean(js)), 4) if js else float("nan")
    metrics["exploratory"]["d3_en_transformer_lime_vs_random"] = {
        "human_lime": cr.get("lime", {}).get("human_mean"),
        "human_random": cr.get("random", {}).get("human_mean"),
        "human_lime_gt_random": bool(cr.get("lime", {}).get("human_mean", 0) > cr.get("random", {}).get("human_mean", 0)),
        "judge_lime": jud["lime"], "judge_random": jud["random"],
        "lexicon_lime": cr.get("lime", {}).get("lexicon_mean"),
        "lexicon_random": cr.get("random", {}).get("lexicon_mean")}


def _print_summary(m):
    a = m["accounting"]
    print(f"sets={a['n_sets']} judge_ok={a['n_judge_ok']} judge_missing={a['n_judge_missing']} "
          f"tokens={a['n_tokens']}", flush=True)
    print(f"P2 lexicon-human kappa (overall): {m['P2']['overall']['kappa']} "
          f"[{m['P2']['overall']['band']}]  pct_agree={m['P2']['overall']['pct_agree']}", flush=True)
    print(f"P3 human-LLM token kappa: {m['P3']['token_kappa']['kappa']} "
          f"[{m['P3']['token_kappa']['band']}]  set cue-rate rho={m['P3']['set_cuerate_spearman']['rho']} "
          f"holistic rho={m['P3']['set_holistic_spearman']['rho']}", flush=True)
    print(f"E1 (SHAP>RANDOM all cells): {m['expectations']['E1_shap_gt_random']['holds_all']}  "
          f"E2 (lex-human kappa>=0.4): {m['expectations']['E2_lexicon_human_kappa']['holds']}", flush=True)
    s = m["sensitivity_exclude_compromised"]
    print(f"SENSITIVITY excl {s['excluded_doc']}: P2 lex-human kappa {s['P2']['overall']['kappa']} "
          f"[{s['P2']['overall']['band']}]  P3 human-LLM kappa {s['P3']['token_kappa']['kappa']}  "
          f"E1_all={s['expectations']['E1_shap_gt_random']['holds_all']} "
          f"E2={s['expectations']['E2_lexicon_human_kappa']['holds']}", flush=True)


if __name__ == "__main__":
    analyze()
