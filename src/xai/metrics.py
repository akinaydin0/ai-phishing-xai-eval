"""
Quantitative XAI metrics (Directive 7): faithfulness, stability, plausibility — each with the
mandatory random-attribution baseline through the identical pipeline.

`p = p(predicted class)`; the predicted class is fixed from the original document. Faithfulness
removals operate on the raw text and re-run the full pipeline (champ.proba). All removal variants
for a unit are batched into one `proba` call for GPU efficiency.
"""

from __future__ import annotations

import warnings
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

try:  # constant-input Spearman is expected (top tokens tie) — we already drop NaN results
    from scipy.stats import ConstantInputWarning
    warnings.filterwarnings("ignore", category=ConstantInputWarning)
except ImportError:  # pragma: no cover
    warnings.filterwarnings("ignore", message=".*input array is constant.*")

from . import cue_lexicon, explainers, textops

K_REPORT = [1, 5, 10, 20]
K_AOPC = list(range(1, 21))
N_PERTURB = 5
PERTURB_FRAC = 0.05
TOPN = 10


def _p_pred(p1: np.ndarray, preds) -> np.ndarray:
    p1 = np.asarray(p1, dtype=float)
    preds = np.asarray(preds)
    return np.where(preds == 1, p1, 1.0 - p1)


# --------------------------------------------------------------------------- #
# Faithfulness: comprehensiveness@k, sufficiency@k, AOPC + random baseline
# --------------------------------------------------------------------------- #
def faithfulness(champ, texts, attrs, preds, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    preds = np.asarray(preds)
    p_orig = _p_pred(champ.proba(texts), preds)

    variants: list[str] = []
    meta: list[tuple] = []   # (doc_i, kind, k, is_random)
    for i, (text, attr, pred) in enumerate(zip(texts, attrs, preds)):
        nw = len(textops.word_spans(text))
        top = textops.top_k_positions(attr, int(pred), min(20, nw))
        randperm = rng.permutation(nw).tolist()
        for k in K_AOPC:
            variants.append(textops.remove_words(text, top[:k])); meta.append((i, "comp", k, False))
            variants.append(textops.remove_words(text, randperm[:k])); meta.append((i, "comp", k, True))
        for k in K_REPORT:
            variants.append(textops.keep_only_words(text, top[:k])); meta.append((i, "suff", k, False))
            variants.append(textops.keep_only_words(text, randperm[:k])); meta.append((i, "suff", k, True))

    P = _p_pred(champ.proba(variants), preds[[m[0] for m in meta]])

    # accumulate per (kind, k, is_random): drop = p_orig - p_variant
    acc: dict = defaultdict(list)
    for (i, kind, k, is_rand), pv in zip(meta, P):
        acc[(kind, k, is_rand)].append(p_orig[i] - pv)

    def mean(kind, k, is_rand):
        return float(np.mean(acc[(kind, k, is_rand)]))

    out = {}
    for k in K_REPORT:
        out[f"comp@{k}"] = round(mean("comp", k, False), 4)
        out[f"suff@{k}"] = round(mean("suff", k, False), 4)
        out[f"rnd_comp@{k}"] = round(mean("comp", k, True), 4)
        out[f"rnd_suff@{k}"] = round(mean("suff", k, True), 4)
    out["aopc"] = round(float(np.mean([mean("comp", k, False) for k in K_AOPC])), 4)
    out["rnd_aopc"] = round(float(np.mean([mean("comp", k, True) for k in K_AOPC])), 4)
    # deletion curve (comp@k for k=1..20, method vs random) for the figures
    out["_curve_comp"] = [round(mean("comp", k, False), 4) for k in K_AOPC]
    out["_curve_rnd"] = [round(mean("comp", k, True), 4) for k in K_AOPC]
    return out


# --------------------------------------------------------------------------- #
# Stability: 5 seeded perturbations (delete 5% of NON-top-10 words) -> re-attribute
# --------------------------------------------------------------------------- #
def _word_attr_map(text, attr, pred) -> dict:
    oriented = np.asarray(attr) if pred == 1 else -np.asarray(attr)
    d = defaultdict(float)
    for w, a in zip(textops.words_lower(text), oriented):
        d[w] += float(a)
    return d


def _top_words(wmap: dict, n: int = TOPN) -> list:
    return [w for w, _ in sorted(wmap.items(), key=lambda kv: kv[1], reverse=True)[:n]]


def stability(champ, family, method, texts, attrs, preds, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    jaccs, spears = [], []
    for text, attr, pred in zip(texts, attrs, preds):
        spans = textops.word_spans(text)
        nw = len(spans)
        if nw < 3:
            continue
        base_map = _word_attr_map(text, attr, pred)
        base_top = set(_top_words(base_map))
        base_top_pos = set(textops.top_k_positions(attr, int(pred), TOPN))
        non_top = [i for i in range(nw) if i not in base_top_pos]
        n_del = max(1, int(round(PERTURB_FRAC * len(non_top))))
        for _ in range(N_PERTURB):
            drop = rng.choice(non_top, size=min(n_del, len(non_top)), replace=False).tolist() if non_top else []
            pert = textops.remove_words(text, drop)
            pa = explainers.attribute(champ, family, method, pert)
            pmap = _word_attr_map(pert, pa, pred)
            ptop = set(_top_words(pmap))
            union = base_top | ptop
            jaccs.append(len(base_top & ptop) / len(union) if union else 1.0)
            shared = [w for w in base_map if w in pmap]
            if len(shared) >= 3:
                r = spearmanr([base_map[w] for w in shared], [pmap[w] for w in shared]).correlation
                if r == r:  # not NaN
                    spears.append(float(r))
    return {"stability_jaccard": round(float(np.mean(jaccs)), 4) if jaccs else float("nan"),
            "stability_spearman": round(float(np.mean(spears)), 4) if spears else float("nan")}


# --------------------------------------------------------------------------- #
# LIME seed-agreement (two seeds, top-10 Jaccard)
# --------------------------------------------------------------------------- #
def lime_seed_agreement(champ, texts, preds, seeds=(42, 43)) -> float:
    jaccs = []
    for text, pred in zip(texts, preds):
        a1 = explainers.lime_attr(champ, text, seed=seeds[0])
        a2 = explainers.lime_attr(champ, text, seed=seeds[1])
        t1 = set(_top_words(_word_attr_map(text, a1, pred)))
        t2 = set(_top_words(_word_attr_map(text, a2, pred)))
        u = t1 | t2
        jaccs.append(len(t1 & t2) / len(u) if u else 1.0)
    return round(float(np.mean(jaccs)), 4) if jaccs else float("nan")


# --------------------------------------------------------------------------- #
# Plausibility: cue-hit rate within top-10 on phishing-PREDICTED docs vs random
# --------------------------------------------------------------------------- #
def _cue_hits(text: str, positions) -> int:
    spans = textops.word_spans(text)
    ue_spans = cue_lexicon.url_email_spans(text)
    hits = 0
    for i in positions:
        s, e = spans[i]
        word = text[s:e]
        if cue_lexicon.is_cue(word) or any(a < e and s < b for a, b in ue_spans):
            hits += 1
    return hits


def plausibility(texts, attrs, preds, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    rates, rnd_rates = [], []
    cat_hits = defaultdict(int)
    n_ph = 0
    for text, attr, pred in zip(texts, attrs, preds):
        if pred != 1:
            continue
        n_ph += 1
        nw = len(textops.word_spans(text))
        top = textops.top_k_positions(attr, 1, min(TOPN, nw))
        rnd = rng.permutation(nw)[:min(TOPN, nw)].tolist()
        k = max(1, len(top))
        rates.append(_cue_hits(text, top) / k)
        rnd_rates.append(_cue_hits(text, rnd) / max(1, len(rnd)))
        # per-category (for the summary)
        spans = textops.word_spans(text)
        for i in top:
            for cat in cue_lexicon.word_categories(text[spans[i][0]:spans[i][1]]):
                cat_hits[cat] += 1
    return {"plausibility@10": round(float(np.mean(rates)), 4) if rates else float("nan"),
            "rnd_plausibility@10": round(float(np.mean(rnd_rates)), 4) if rnd_rates else float("nan"),
            "n_phishing_pred": n_ph,
            "_cue_categories": {c: cat_hits[c] for c in cue_lexicon.CATEGORIES}}
