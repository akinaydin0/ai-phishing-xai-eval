"""
D9 Part B — materialise per-document TEST scores + reproduction asserts.

For every model in the seven pre-registered comparisons, recompute per-doc probability-like TEST
scores from stored artefacts ONLY (classical: HPO best_params → deterministic CPU retrain;
transformers: saved 256/512 checkpoints, tokenised at the checkpoint's own max_length) and
HARD-FAIL unless the recomputed test ROC-AUC equals its committed value to 4 dp. Score vectors are
cached under results/stats/scores/ (git-ignored). Nothing is written to results.csv/results_512.csv.

Run: python -m src.experiments.d9_scores
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

try:
    from src.splits import load_splits
    from src.xai import champions as C
except ImportError:  # pragma: no cover
    from splits import load_splits
    from xai import champions as C

_ROOT = Path(__file__).resolve().parent.parent.parent
SCORES_DIR = _ROOT / "results" / "stats" / "scores"
CKPT512 = _ROOT / "results" / "transformers_512" / "checkpoints"
D1 = "d1_decontaminated_sb"

# score_id -> (kind, build-args, eval_corpus, committed 4-dp test AUC)
#   sk args = (family, source_corpus);  tf args = (model_name, source_corpus, max_length)
SPECS = {
    # --- in-corpus classical champions ---
    "svc_d1sb@d1sb":        ("sk", ("svc", D1),      D1,      0.9998),
    "nb_d2@d2":             ("sk", ("nb", "d2"),     "d2",    0.9996),
    "svc_d3en@d3en":        ("sk", ("svc", "d3_en"), "d3_en", 0.9993),
    # --- classical cross champions ---
    "svc_d3en@d2":          ("sk", ("svc", "d3_en"), "d2",    0.7523),
    "lr_d2@d3en":           ("sk", ("lr", "d2"),     "d3_en", 0.8429),
    # --- in-corpus transformer champions (256) ---
    "distilbert256_d1sb@d1sb": ("tf", ("distilbert-base-uncased", D1, 256),   D1,      0.9999),
    "distilbert256_d2@d2":     ("tf", ("distilbert-base-uncased", "d2", 256),  "d2",    0.9999),
    "roberta256_d3en@d3en":    ("tf", ("roberta-base", "d3_en", 256),          "d3_en", 0.9976),
    # --- transformer cross (256 + 512) ---
    "roberta256_d3en@d2":      ("tf", ("roberta-base", "d3_en", 256),          "d2",    0.7306),
    "roberta512_d3en@d2":      ("tf", ("roberta-base", "d3_en", 512),          "d2",    0.8440),
    "distilbert256_d2@d3en":   ("tf", ("distilbert-base-uncased", "d2", 256),  "d3_en", 0.7748),
    "distilbert512_d2@d3en":   ("tf", ("distilbert-base-uncased", "d2", 512),  "d3_en", 0.7642),
}


def _build(kind, args):
    if kind == "sk":
        return C._SklearnChampion(*args)
    model_name, source, ml = args
    ck = None if ml == 256 else CKPT512 / f"{model_name}__{source}"
    champ = C.TransformerChampion(model_name, source, ckpt_dir=ck)
    champ.max_length = ml
    return champ


def materialise(verbose=True) -> dict:
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for sid, (kind, args, eval_corpus, committed) in SPECS.items():
        champ = _build(kind, args)
        te = load_splits(eval_corpus)["test"].reset_index(drop=True)
        texts = te["text"].tolist()
        y = te["label"].to_numpy()
        hashes = te["_hash"].to_numpy()
        scores = np.asarray(champ.proba(texts), dtype=np.float64)
        auc_full = roc_auc_score(y, scores)
        auc = round(float(auc_full), 4)
        if auc != committed:
            raise AssertionError(f"REPRODUCTION FAIL [{sid}]: recomputed AUC {auc} "
                                 f"(full {auc_full:.6f}) != committed {committed} (4 dp)")
        np.savez(SCORES_DIR / f"{sid}.npz", scores=scores, labels=y.astype(np.int64), hashes=hashes)
        out[sid] = {"auc": auc, "committed": committed, "eval_corpus": eval_corpus, "n": int(len(y))}
        if verbose:
            print(f"  [OK] {sid:26s} auc={auc:.4f} == committed {committed:.4f}  n={len(y)}", flush=True)
        del champ
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


def load(sid: str):
    z = np.load(SCORES_DIR / f"{sid}.npz", allow_pickle=True)
    return z["scores"], z["labels"], z["hashes"]


if __name__ == "__main__":
    res = materialise()
    print(f"\nAll {len(res)} score vectors reproduced to 4 dp and cached in {SCORES_DIR}")
