"""
D9 Part E — Pipeline A preprocessing ablation (VALIDATION ONLY, zero test contact).

Six deterministic text variants (V0 baseline; V1 stopword removal; V2 URL/email/number
placeholders; V3 SymSpell spelling max_edit_distance=1; V4 wordninja segmentation of len>=12
tokens; V5 = V2->V4->V3->V1 combined). Per (variant, train_corpus): fit the frozen-hyperparameter
classical champion on the transformed TRAIN split; evaluate val ROC-AUC on each corpus's VAL split
(own + the other two), transformed by the same variant. Endpoint: val-AUC delta vs V0. No test split
is ever loaded here. Pre-registered directional hypothesis: placeholders (V2) help cross-corpus.

Outputs: results/preproc_ablation/preproc_ablation.csv (+ git-ignored variant-text cache).
Run: python -m src.experiments.d9_preproc --benchmark   # V3 cost + projection
     python -m src.experiments.d9_preproc               # full grid
"""

from __future__ import annotations

import argparse
import csv
import importlib.resources
import pickle
import re
import time
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import roc_auc_score

try:
    from src.experiments import hpo
    from src.splits import load_splits
except ImportError:  # pragma: no cover
    from experiments import hpo
    from splits import load_splits

import json

import wordninja
from symspellpy import SymSpell, Verbosity

_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = _ROOT / "results" / "preproc_ablation"
CACHE = OUT_DIR / "cache"
HPO_DIR = _ROOT / "results" / "hpo"
D1 = "d1_decontaminated_sb"
CORPORA = ["d1_decontaminated_sb", "d2", "d3_en"]
SHORT = {"d1_decontaminated_sb": "d1_sb", "d2": "d2", "d3_en": "d3_en"}
# classical champion family per (train) corpus — frozen, the ablation cannot change it
CHAMP = {"d1_decontaminated_sb": "svc", "d2": "nb", "d3_en": "svc"}
VARIANTS = ["V0", "V1", "V2", "V3", "V4", "V5"]

_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+", re.IGNORECASE)
_NUM = re.compile(r"\b\d+\b")
_SYM = None


def _symspell():
    global _SYM
    if _SYM is None:
        _SYM = SymSpell(max_dictionary_edit_distance=1)
        dpath = importlib.resources.files("symspellpy") / "frequency_dictionary_en_82_765.txt"
        _SYM.load_dictionary(str(dpath), term_index=0, count_index=1)
    return _SYM


def _spell_tok(t: str) -> str:
    if t.isalpha() and len(t) >= 4:
        sug = _symspell().lookup(t.lower(), Verbosity.TOP, max_edit_distance=1, include_unknown=True)
        return sug[0].term if sug else t
    return t


def v0(text):
    return text


def v1(text):  # stopword removal
    return " ".join(t for t in text.split() if t.lower() not in ENGLISH_STOP_WORDS)


def v2(text):  # placeholders, in order URL -> email -> number
    return _NUM.sub("numtoken", _EMAIL.sub("emailtoken", _URL.sub("urltoken", text)))


def v3(text):  # spelling
    return " ".join(_spell_tok(t) for t in text.split())


def v4(text):  # segmentation of len>=12 tokens
    out = []
    for t in text.split():
        out.extend(wordninja.split(t) if len(t) >= 12 else [t])
    return " ".join(out)


def v5(text):  # combined: V2 -> V4 -> V3 -> V1
    text = v2(text)
    seg = []
    for t in text.split():
        seg.extend(wordninja.split(t) if len(t) >= 12 else [t])
    spelled = (_spell_tok(t) for t in seg)
    return " ".join(t for t in spelled if t.lower() not in ENGLISH_STOP_WORDS)


TRANSFORM = {"V0": v0, "V1": v1, "V2": v2, "V3": v3, "V4": v4, "V5": v5}


def _transform_cached(corpus, split, variant) -> list[str]:
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"{variant}_{SHORT[corpus]}_{split}.pkl"
    if p.exists():
        return pickle.loads(p.read_bytes())
    texts = load_splits(corpus)[split]["text"].tolist()   # split ∈ {train, val} ONLY
    fn = TRANSFORM[variant]
    out = [fn(t) for t in texts]
    p.write_bytes(pickle.dumps(out))
    return out


def benchmark_v3(n=500):
    texts = load_splits(D1)["train"]["text"].tolist()[:n]
    _symspell()  # warm dict load out of the timing
    t0 = time.time()
    _ = [v3(t) for t in texts]
    per = (time.time() - t0) / n
    # total docs transformed across the grid = train+val of 3 corpora, each variant needing V3-ish
    total_docs = sum(len(load_splits(c)["train"]) + len(load_splits(c)["val"]) for c in CORPORA)
    # V3 and V5 both run SymSpell; ~2 SymSpell passes over the corpus in total
    proj_s = per * total_docs * 2
    print(f"V3 SymSpell: {per*1000:.2f} ms/doc on {n} docs; total docs/pass={total_docs:,}; "
          f"projected SymSpell work (V3+V5) ~{proj_s/60:.1f} min", flush=True)
    return per, proj_s


def _build_champion(train_corpus, variant):
    family = CHAMP[train_corpus]
    bp = json.loads((HPO_DIR / f"{train_corpus}_{family}.json").read_text(encoding="utf-8"))["best_params"]
    model_params = {k: v for k, v in bp.items() if k not in hpo._VEC_KEYS}
    vec = hpo._build_vec(hpo._vec_from_best(family, bp))
    Xtr = vec.fit_transform(_transform_cached(train_corpus, "train", variant))
    ytr = load_splits(train_corpus)["train"]["label"].to_numpy()
    model = hpo.make_model(family, model_params, device="cpu")
    model.fit(Xtr, ytr)
    return vec, model, family


def _score(model, vec, texts):
    X = vec.transform(texts)
    return model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else expit(model.decision_function(X))


def run_grid():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    auc = {}  # (train, eval, variant) -> val auc
    for variant in VARIANTS:
        for train_corpus in CORPORA:
            vec, model, family = _build_champion(train_corpus, variant)
            for eval_corpus in CORPORA:
                val_texts = _transform_cached(eval_corpus, "val", variant)
                yval = load_splits(eval_corpus)["val"]["label"].to_numpy()
                auc[(train_corpus, eval_corpus, variant)] = round(float(roc_auc_score(yval, _score(model, vec, val_texts))), 4)
            print(f"  {variant} train={SHORT[train_corpus]:6s} ({family}) done", flush=True)
    rows = []
    for train_corpus in CORPORA:
        for eval_corpus in CORPORA:
            base = auc[(train_corpus, eval_corpus, "V0")]
            for variant in VARIANTS:
                a = auc[(train_corpus, eval_corpus, variant)]
                rows.append({"train_corpus": SHORT[train_corpus], "eval_corpus": SHORT[eval_corpus],
                             "variant": variant, "val_auc": a, "delta_vs_V0": round(a - base, 4),
                             "n_eval": int(len(load_splits(eval_corpus)["val"]))})
    with open(OUT_DIR / "preproc_ablation.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["train_corpus", "eval_corpus", "variant", "val_auc", "delta_vs_V0", "n_eval"])
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {OUT_DIR}/preproc_ablation.csv ({len(rows)} rows)")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="D9 Pipeline A preprocessing ablation (val only).")
    ap.add_argument("--benchmark", action="store_true", help="V3 cost benchmark + projection only")
    args = ap.parse_args(argv)
    if args.benchmark:
        benchmark_v3()
        return
    run_grid()


if __name__ == "__main__":
    main()
