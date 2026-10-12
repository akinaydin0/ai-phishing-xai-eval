"""Build the per-comparison input CSVs for src/experiments/cluster_bootstrap_d3.py.

Columns: doc_id, family_id, y_true, score_a, score_b. There is one row per d3_en test document (2,875).
Outputs: results/stats/cluster_inputs/{c1c,c4,c5}.csv (committed; no email text).

Sources:
  * scores: the committed D9 cached test score vectors, results/stats/scores/<id>.npz
    (scores / labels / hashes, written by src/experiments/d9_scores.py; LinearSVC = expit(decision_function));
  * family_id: original_id from src.experiments.d3_family_audit._family_key, i.e. the Part A / Appendix B.4 key.

Every CSV is checked before it is written: the hashes form the frozen test set, the a/b vectors are
aligned, the labels agree with the corpus, and both AUCs reproduce the committed 4-dp values.

Run: python -m src.experiments.cluster_bootstrap_inputs
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.experiments.d3_family_audit import _family_key  # noqa: E402
from src.splits import SPLITS_DIR  # noqa: E402

SCORES = ROOT / "results" / "stats" / "scores"
OUT = ROOT / "results" / "stats" / "cluster_inputs"
# name: (score id a, committed AUC a, score id b, committed AUC b, committed delta a-b)
PAIRS = {
    "c1c": ("svc_d3en@d3en", 0.9993, "roberta256_d3en@d3en", 0.9976, 0.0017),
    "c4": ("lr_d2@d3en", 0.8429, "distilbert256_d2@d3en", 0.7748, 0.0681),
    "c5": ("distilbert512_d2@d3en", 0.7642, "distilbert256_d2@d3en", 0.7748, -0.0106),
}


def load(sid: str):
    z = np.load(SCORES / f"{sid}.npz", allow_pickle=True)
    return z["scores"].astype(float), z["labels"].astype(int), z["hashes"].astype(str)


def main() -> int:
    fam = _family_key()
    fam_of = dict(zip(fam["_hash"], fam["original_id"].astype(int)))
    lab_of = dict(zip(fam["_hash"], fam["label"].astype(int)))
    test = set(json.loads((SPLITS_DIR / "d3_en.json").read_text(encoding="utf-8"))["test"])
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (ida, auca, idb, aucb, delta) in PAIRS.items():
        sa, ya, ha = load(ida)
        sb, yb, hb = load(idb)
        assert len(ha) == len(hb) == len(test) == 2875, f"{name}: row count"
        assert (ha == hb).all() and (ya == yb).all(), f"{name}: a/b vectors not aligned"
        assert set(ha) == test, f"{name}: hashes are not the frozen d3_en test set"
        assert all(lab_of[h] == y for h, y in zip(ha, ya)), f"{name}: labels disagree with the corpus"
        ra, rb = roc_auc_score(ya, sa), roc_auc_score(ya, sb)
        assert round(ra, 4) == auca and round(rb, 4) == aucb, f"{name}: AUC {ra:.6f}/{rb:.6f} != committed"
        df = pd.DataFrame({"doc_id": ha, "family_id": [fam_of[h] for h in ha], "y_true": ya,
                           "score_a": sa, "score_b": sb})
        path = OUT / f"{name}.csv"
        df.to_csv(path, index=False)
        print(f"{name}: a={ida} (AUC {ra:.6f}) b={idb} (AUC {rb:.6f}) delta={ra - rb:+.6f} "
              f"[committed {delta:+.4f}] rows={len(df)} families={df['family_id'].nunique()} -> {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
