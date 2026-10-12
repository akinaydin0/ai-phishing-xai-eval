#!/usr/bin/env python3
"""Family-cluster bootstrap for the D3 (d3_en) test comparisons.

Purpose
-------
The committed paired bootstrap (D9) resamples individual test documents. On D3
every seed email has six generated variants, so documents are clustered by seed
family and the document-level interval understates the sampling uncertainty.
This script recomputes the 95% percentile interval for the AUC difference of a
committed comparison by resampling whole seed families (cluster bootstrap),
using the SAME committed score vectors. No model is run and no test partition is
touched beyond the already cached test scores.

Input
-----
One CSV per comparison with one row per test document and these columns
(names can be remapped with the --col-* flags):

    doc_id, family_id, y_true, score_a, score_b

* doc_id     : the committed document identifier (sha1 body-hash key)
* family_id  : the D3 seed identifier (the grouping key used in Appendix B.4)
* y_true     : 1 = positive (phishing or spam), 0 = legitimate
* score_a    : committed test score of model a (e.g. LinearSVC sigmoid margin)
* score_b    : committed test score of model b (e.g. RoBERTa-256 probability)

The row set must be the complete frozen d3_en test partition (2,875 rows).

Method (fixed before execution)
-------------------------------
* Point estimate: delta = AUC_a - AUC_b on the full test partition. It must
  reproduce the committed value to four decimals (the script asserts this when
  --expected-delta is given).
* Cluster bootstrap: B = 10,000 resamples (seed 42). Each resample draws
  families with replacement (as many families as there are in the test
  partition) and concatenates all documents of the drawn families; AUC_a, AUC_b
  and their difference are computed on the resample; the identical resampled
  index set is used for both models.
* Interval: percentile 2.5 / 97.5 of the resampled differences. Resamples that
  contain a single class are skipped and counted.
* Also reported: the document-level paired stratified bootstrap under the same
  seed for reference, the design effect (cluster variance / document variance),
  the number of families and documents, and the family-size distribution.
* Interpretation rule (binding): the interval annotates the committed
  comparison; it does not reselect champions or reverse any claim. Whatever it
  shows is reported.

Usage
-----
    python cluster_bootstrap_d3.py --csv c1c.csv --name C1c --expected-delta 0.0017
    python cluster_bootstrap_d3.py --csv c4.csv  --name C4  --expected-delta 0.0681
    python cluster_bootstrap_d3.py --csv c5.csv  --name C5  --expected-delta -0.0106 --out results/stats/cluster_bootstrap_C5.json

Requires numpy, pandas, scikit-learn (already pinned in the project).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def auc_pair(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    return float(roc_auc_score(y, a)), float(roc_auc_score(y, b))


def cluster_bootstrap(y, a, b, fam, B, seed):
    rng = np.random.default_rng(seed)
    fam_codes, fam_index = np.unique(fam, return_inverse=True)
    n_fam = len(fam_codes)
    members = [np.flatnonzero(fam_index == k) for k in range(n_fam)]
    deltas = np.empty(B)
    skipped = 0
    for i in range(B):
        draw = rng.integers(0, n_fam, size=n_fam)
        idx = np.concatenate([members[k] for k in draw])
        yy = y[idx]
        if yy.min() == yy.max():
            skipped += 1
            deltas[i] = np.nan
            continue
        aa, bb = auc_pair(yy, a[idx], b[idx])
        deltas[i] = aa - bb
    return deltas, skipped, n_fam


def document_bootstrap(y, a, b, B, seed):
    """Paired stratified document-level bootstrap (the committed D9 scheme):
    indices resampled within class, identical index set for both models."""
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    deltas = np.empty(B)
    for i in range(B):
        idx = np.concatenate([rng.choice(pos, size=len(pos), replace=True),
                              rng.choice(neg, size=len(neg), replace=True)])
        aa, bb = auc_pair(y[idx], a[idx], b[idx])
        deltas[i] = aa - bb
    return deltas


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True)
    p.add_argument("--name", required=True, help="comparison id, e.g. C1c")
    p.add_argument("--expected-delta", type=float, default=None,
                   help="committed AUC_a - AUC_b (4 dp); the script asserts reproduction")
    p.add_argument("--B", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None, help="JSON output path")
    p.add_argument("--col-y", default="y_true")
    p.add_argument("--col-a", default="score_a")
    p.add_argument("--col-b", default="score_b")
    p.add_argument("--col-fam", default="family_id")
    p.add_argument("--expected-n", type=int, default=2875, help="expected number of test rows (d3_en test = 2,875)")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    for c in (args.col_y, args.col_a, args.col_b, args.col_fam):
        if c not in df.columns:
            print(f"missing column: {c}; columns are {list(df.columns)}", file=sys.stderr)
            return 2
    if df[args.col_fam].isna().any():
        print("family_id has missing values; every D3 test document must carry its seed identifier", file=sys.stderr)
        return 2
    if len(df) != args.expected_n:
        print(f"row count {len(df)} differs from the expected test size {args.expected_n}; aborting", file=sys.stderr)
        return 2

    y = df[args.col_y].to_numpy().astype(int)
    a = df[args.col_a].to_numpy().astype(float)
    b = df[args.col_b].to_numpy().astype(float)
    fam = df[args.col_fam].astype(str).to_numpy()

    auc_a, auc_b = auc_pair(y, a, b)
    delta = auc_a - auc_b
    if args.expected_delta is not None and abs(round(delta, 4) - round(args.expected_delta, 4)) > 1e-9:
        print(f"point estimate {delta:.6f} does not reproduce the committed delta {args.expected_delta:.4f}; "
              f"check that the score columns are the committed test scores", file=sys.stderr)
        return 3

    cl, skipped, n_fam = cluster_bootstrap(y, a, b, fam, args.B, args.seed)
    cl_valid = cl[~np.isnan(cl)]
    doc = document_bootstrap(y, a, b, args.B, args.seed)

    sizes = Counter(Counter(fam).values())
    out = {
        "comparison": args.name,
        "n_documents": int(len(df)),
        "n_families": int(n_fam),
        "family_size_distribution": {str(k): int(v) for k, v in sorted(sizes.items())},
        "auc_a": round(auc_a, 6),
        "auc_b": round(auc_b, 6),
        "delta_point": round(delta, 6),
        "B": args.B,
        "seed": args.seed,
        "cluster_bootstrap": {
            "ci95": [round(float(np.percentile(cl_valid, 2.5)), 6), round(float(np.percentile(cl_valid, 97.5)), 6)],
            "sd": round(float(cl_valid.std(ddof=1)), 6),
            "resamples_skipped_single_class": int(skipped),
            "excludes_zero": bool(np.percentile(cl_valid, 2.5) > 0 or np.percentile(cl_valid, 97.5) < 0),
        },
        "document_bootstrap_reference": {
            "ci95": [round(float(np.percentile(doc, 2.5)), 6), round(float(np.percentile(doc, 97.5)), 6)],
            "sd": round(float(doc.std(ddof=1)), 6),
        },
        "design_effect_variance_ratio": round(float(cl_valid.var(ddof=1) / doc.var(ddof=1)), 3),
        "method": "families resampled with replacement, all member documents kept, identical index set for both models, percentile interval",
    }
    text = json.dumps(out, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
