"""
D3 seed-family audit, reconstructed on 2026-10-12.

This rebuilds the AUDIT-D3 analysis logged in notes/decisions.md (2026-08-17, section A). The
original script was run from a scratchpad and never committed.

It is read-only. It loads two things:
  * the frozen d3_en split (train / val / test document hashes);
  * the D3 seed identifier `original_id`, which is the family key of the Appendix B.4 grouped split.
    It comes from clean_d3.pkl (English rows) and is aligned to load_corpus("d3_en") exactly as in
    src/experiments/group_sens.py.

It then counts how the six variants of each seed family fall across the three partitions.
No model is run and no score is read.

Hard-fails (exit 1) if any figure logged by AUDIT-D3 is not reproduced.

Outputs: results/stats/d3_family_audit.json (also printed)
Run: python -m src.experiments.d3_family_audit
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

try:
    from src.splits import SPLITS_DIR, body_hash, load_corpus
except ImportError:  # pragma: no cover
    from splits import SPLITS_DIR, body_hash, load_corpus

_ROOT = Path(__file__).resolve().parent.parent.parent
CLEAN_D3 = _ROOT / "data" / "cache" / "clean_d3.pkl"
OUT = _ROOT / "results" / "stats" / "d3_family_audit.json"
PARTS = ("train", "val", "test")

# Figures logged by AUDIT-D3 (notes/decisions.md, 2026-08-17, A1-A4). These are the reproduction
# targets; they were taken from the log and must not be edited to match a run.
LOGGED = {
    "n_documents": 19164,
    "n_families": 3194,
    "family_size_distribution": {"6": 3194},
    "families_train": 3191,
    "families_val": 1963,
    "families_test": 2004,
    "test_with_train_sibling": 2865,      # 2,865 / 2,875 = 99.65%
    "test_with_val_sibling": 1567,        # 1,567 / 2,875 = 54.50%
    "test_with_train_or_val_sibling": 2875,  # 100.00%
    "val_with_train_sibling": 2867,       # 2,867 / 2,875 = 99.72%
    "test_without_train_sibling": 10,     # "n = 10, all of class 0"
    "test_without_train_sibling_labels": {"0": 10},
    "test_without_train_or_val_sibling": 0,  # "the clean-test subset is EMPTY (n = 0)"
}


def _family_key() -> pd.DataFrame:
    """load_corpus("d3_en") plus `original_id`, built exactly as in group_sens.py:165-167."""
    df = load_corpus("d3_en")
    d3raw = pd.read_pickle(CLEAN_D3)
    en = d3raw[d3raw["language"] == "en"].reset_index(drop=True)
    # Alignment guard (added in the reconstruction): the positional join must be row-exact.
    if len(en) != len(df) or not (en["text"].map(body_hash).to_numpy() == df["_hash"].to_numpy()).all():
        raise AssertionError("clean_d3 English rows do not align with load_corpus('d3_en')")
    df["original_id"] = en["original_id"].to_numpy()
    if df["original_id"].isna().any():
        raise AssertionError("original_id has missing values")
    return df


def audit() -> dict:
    df = _family_key()
    fam_of = dict(zip(df["_hash"], df["original_id"].astype(int)))
    label_of = dict(zip(df["_hash"], df["label"].astype(int)))

    spec = json.loads((SPLITS_DIR / "d3_en.json").read_text(encoding="utf-8"))
    parts = {p: list(spec[p]) for p in PARTS}
    all_hashes = [h for p in PARTS for h in parts[p]]
    if len(all_hashes) != len(set(all_hashes)) or set(all_hashes) != set(fam_of):
        raise AssertionError("frozen d3_en split is not a disjoint cover of the corpus")

    fams_in = {p: {fam_of[h] for h in parts[p]} for p in PARTS}
    size_counts = Counter(Counter(fam_of.values()).values())

    with_sibling = {}
    for p in PARTS:
        n = len(parts[p])
        others = [q for q in PARTS if q != p]
        row = {}
        for targets in [[q] for q in others] + [others]:
            pool = set().union(*(fams_in[q] for q in targets))
            k = sum(fam_of[h] in pool for h in parts[p])
            row["_or_".join(targets)] = {"n": k, "pct": round(100.0 * k / n, 2)}
        with_sibling[p] = {"n_documents": n, "family_has_member_in": row}

    no_train = [h for h in parts["test"] if fam_of[h] not in fams_in["train"]]
    no_train_val = [h for h in parts["test"] if fam_of[h] not in fams_in["train"] | fams_in["val"]]

    out = {
        "corpus": "d3_en",
        "split_file": "data/splits/d3_en.json",
        "family_key": "original_id from data/cache/clean_d3.pkl (language == 'en'), aligned to "
                      "load_corpus('d3_en') as in src/experiments/group_sens.py (Appendix B.4)",
        "n_documents": len(fam_of),
        "n_families": len(set(fam_of.values())),
        "family_size_distribution": {str(k): int(v) for k, v in sorted(size_counts.items())},
        "families_per_partition": {p: len(fams_in[p]) for p in PARTS},
        "documents_whose_family_has_a_member_in_another_partition": with_sibling,
        "test_documents_without_train_sibling": {
            "n": len(no_train),
            "labels": {str(k): int(v) for k, v in sorted(Counter(label_of[h] for h in no_train).items())},
            "n_families": len({fam_of[h] for h in no_train}),
        },
        "test_documents_without_train_or_val_sibling": {"n": len(no_train_val)},
        "families_straddling_train_test": len(fams_in["train"] & fams_in["test"]),
        "test_families_without_a_train_member": len(fams_in["test"] - fams_in["train"]),
    }

    t = with_sibling["test"]["family_has_member_in"]
    observed = {
        "n_documents": out["n_documents"],
        "n_families": out["n_families"],
        "family_size_distribution": out["family_size_distribution"],
        "families_train": out["families_per_partition"]["train"],
        "families_val": out["families_per_partition"]["val"],
        "families_test": out["families_per_partition"]["test"],
        "test_with_train_sibling": t["train"]["n"],
        "test_with_val_sibling": t["val"]["n"],
        "test_with_train_or_val_sibling": t["train_or_val"]["n"],
        "val_with_train_sibling": with_sibling["val"]["family_has_member_in"]["train"]["n"],
        "test_without_train_sibling": out["test_documents_without_train_sibling"]["n"],
        "test_without_train_sibling_labels": out["test_documents_without_train_sibling"]["labels"],
        "test_without_train_or_val_sibling": out["test_documents_without_train_or_val_sibling"]["n"],
    }
    out["logged_figures_check"] = {k: {"logged": LOGGED[k], "reproduced": observed[k],
                                       "match": LOGGED[k] == observed[k]} for k in LOGGED}
    out["all_logged_figures_reproduced"] = all(v["match"] for v in out["logged_figures_check"].values())
    return out


def main() -> int:
    out = audit()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(out, indent=2)
    OUT.write_text(text + "\n", encoding="utf-8")
    print(text)
    if not out["all_logged_figures_reproduced"]:
        bad = [k for k, v in out["logged_figures_check"].items() if not v["match"]]
        print(f"\nREPRODUCTION FAIL: {bad}", file=sys.stderr)
        return 1
    print("\nAll AUDIT-D3 figures reproduced from the frozen artefacts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
