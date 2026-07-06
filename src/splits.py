"""
Frozen train/val/test splits for every modelling corpus.

Stratified 70/15/15 per corpus (seed 42), persisted to data/splits/<corpus>.json
keyed by the **sha1 hash of the cleaned body text** — never positional indices, so
a split survives any re-ordering of the underlying frame. Every experiment
(baselines, transformers, XAI) MUST consume these files via `load_splits`.

Guarantees (checked by `--all` sanity asserts):
  * the three splits are pairwise disjoint (no email in two splits);
  * each split's positive-class share is within 0.5 pp of the whole corpus;
  * regeneration is byte-identical (deterministic given seed 42).

Corpora (all read from the cleaned, pinned caches produced by src.cleaning):
  d1_decontaminated_sb  subject+body natural text  (51,839)   [D1 primary]
  d1_decontaminated     text_combined              (52,704)   [D1 ablation]
  d2                    Enron-Spam                 (29,224)
  d3_en                 PhishFuzzer English variants (19,164)

Usage:  python -m src.splits --all
        from src.splits import load_splits ; parts = load_splits("d2")
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = _PROJECT_ROOT / "data" / "cache"
SPLITS_DIR = _PROJECT_ROOT / "data" / "splits"

LABEL_TOL_PP = 0.5  # per-split positive-class share tolerance vs the whole corpus


def _load_d3_en() -> pd.DataFrame:
    d3 = pd.read_pickle(CACHE_DIR / "clean_d3.pkl")
    return d3[d3["language"] == "en"].reset_index(drop=True)


# corpus id -> thunk returning its cleaned DataFrame (with `text`, `label`)
CORPUS_REGISTRY = {
    "d1_decontaminated_sb": lambda: pd.read_pickle(CACHE_DIR / "clean_d1_decontaminated_sb.pkl"),
    "d1_decontaminated": lambda: pd.read_pickle(CACHE_DIR / "clean_d1_decontaminated.pkl"),
    "d2": lambda: pd.read_pickle(CACHE_DIR / "clean_d2.pkl"),
    "d3_en": _load_d3_en,
}

# text-field label for the results log (subject_body | text_combined)
TEXT_FIELD = {
    "d1_decontaminated_sb": "subject_body",
    "d1_decontaminated": "text_combined",
    "d2": "subject_body",
    "d3_en": "subject_body",
}


def body_hash(text: str) -> str:
    """sha1 of the cleaned body — the stable, position-independent split key."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def load_corpus(corpus_id: str) -> pd.DataFrame:
    """Load a cleaned corpus and attach its unique `_hash` key."""
    df = CORPUS_REGISTRY[corpus_id]().reset_index(drop=True)
    df["_hash"] = df["text"].map(body_hash)
    if not df["_hash"].is_unique:
        n_dup = int(df["_hash"].duplicated().sum())
        raise ValueError(
            f"{corpus_id}: {n_dup} duplicate body hashes — corpus is not deduplicated. "
            "Re-run `python -m src.cleaning --all`."
        )
    return df


def _splits_json(corpus_id: str, seed: int = SEED) -> str:
    """Build the deterministic split JSON string (sorted hash lists → byte-stable)."""
    df = load_corpus(corpus_id)
    y = df["label"].to_numpy()
    idx = np.arange(len(df))

    # 70 / 30, then 30 → 15 / 15, stratified on the binary label both times.
    train_idx, temp_idx = train_test_split(
        idx, test_size=0.30, random_state=seed, stratify=y
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=seed, stratify=y[temp_idx]
    )

    hashes = df["_hash"].to_numpy()

    def _share(ix):
        return round(float(df["label"].to_numpy()[ix].mean()), 6)

    payload = {
        "corpus": corpus_id,
        "seed": seed,
        "n": int(len(df)),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "pos_share_corpus": round(float(y.mean()), 6),
        "pos_share_train": _share(train_idx),
        "pos_share_val": _share(val_idx),
        "pos_share_test": _share(test_idx),
        # sorted so the file is byte-identical across regenerations
        "train": sorted(hashes[train_idx].tolist()),
        "val": sorted(hashes[val_idx].tolist()),
        "test": sorted(hashes[test_idx].tolist()),
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def generate_splits(corpus_id: str, seed: int = SEED) -> Path:
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    text = _splits_json(corpus_id, seed)
    out = SPLITS_DIR / f"{corpus_id}.json"
    out.write_text(text, encoding="utf-8")
    return out


def load_splits(corpus_id: str) -> dict[str, pd.DataFrame]:
    """Return {'train','val','test'} DataFrames for a corpus via its frozen split.

    The corpus is re-read from the cleaned cache and partitioned by hash, so the
    split is robust to any row re-ordering. This is the ONLY sanctioned way for an
    experiment to obtain train/val/test data.
    """
    spec_path = SPLITS_DIR / f"{corpus_id}.json"
    if not spec_path.exists():
        raise FileNotFoundError(
            f"{spec_path} missing. Run `python -m src.splits --all` first."
        )
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    df = load_corpus(corpus_id)

    out = {}
    for part in ("train", "val", "test"):
        wanted = set(spec[part])
        sub = df[df["_hash"].isin(wanted)].reset_index(drop=True)
        if len(sub) != len(wanted):
            raise ValueError(
                f"{corpus_id}/{part}: {len(wanted)} hashes in split file but "
                f"{len(sub)} matched in corpus — split/corpus mismatch."
            )
        out[part] = sub

    # Bidirectional coverage guard: catch a corpus that has GROWN vs the frozen
    # split (new rows would otherwise fall into no split and silently vanish from
    # every experiment). Union — not sum — so a hash accidentally listed in two
    # parts can't mask an uncovered row.
    covered = set(spec["train"]) | set(spec["val"]) | set(spec["test"])
    if not (len(covered) == len(df) == spec["n"]):
        raise ValueError(
            f"{corpus_id}: corpus has {len(df):,} rows but the split spec covers "
            f"{len(covered):,} (spec n={spec['n']:,}) — corpus/spec drift. Re-run "
            "`python -m src.splits --all` after regenerating the cleaned caches."
        )
    return out


def run_sanity_asserts(corpus_id: str) -> list[tuple[str, bool, str]]:
    results = []
    spec = json.loads((SPLITS_DIR / f"{corpus_id}.json").read_text(encoding="utf-8"))
    tr, va, te = set(spec["train"]), set(spec["val"]), set(spec["test"])

    # 1. Pairwise-disjoint splits, covering the whole corpus exactly once.
    overlap = (tr & va) | (tr & te) | (va & te)
    total = len(tr) + len(va) + len(te)
    results.append((f"disjoint[{corpus_id}]", len(overlap) == 0 and total == spec["n"],
                    f"overlap={len(overlap)}, covered={total}/{spec['n']}"))

    # 2. Per-split positive-class share within LABEL_TOL_PP of the corpus.
    parts = load_splits(corpus_id)
    base = spec["pos_share_corpus"]
    worst = 0.0
    for name, d in parts.items():
        share = float(d["label"].mean())
        worst = max(worst, abs(share - base) * 100)
    results.append((f"label-share[{corpus_id}]", worst <= LABEL_TOL_PP,
                    f"max |Δ pos-share| = {worst:.3f}pp (tol {LABEL_TOL_PP}pp)"))

    # 3. Byte-identical regeneration.
    regen = _splits_json(corpus_id)
    on_disk = (SPLITS_DIR / f"{corpus_id}.json").read_text(encoding="utf-8")
    results.append((f"deterministic[{corpus_id}]", regen == on_disk,
                    "regeneration byte-identical" if regen == on_disk else "DRIFT vs on-disk file"))

    # 70/15/15 proportions (informational, within rounding).
    fr = (spec["n_train"] / spec["n"], spec["n_val"] / spec["n"], spec["n_test"] / spec["n"])
    results.append((f"proportions[{corpus_id}]",
                    abs(fr[0] - 0.70) < 0.01 and abs(fr[1] - 0.15) < 0.01 and abs(fr[2] - 0.15) < 0.01,
                    f"{fr[0]:.3f}/{fr[1]:.3f}/{fr[2]:.3f}"))
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description="Frozen stratified splits.")
    parser.add_argument("--all", action="store_true", help="generate splits for all corpora")
    parser.add_argument("--corpus", choices=list(CORPUS_REGISTRY), help="single corpus")
    args = parser.parse_args(argv)

    if not args.all and not args.corpus:
        parser.error("specify --all or --corpus <id>")
    corpora = list(CORPUS_REGISTRY) if args.all else [args.corpus]

    for cid in corpora:
        path = generate_splits(cid)
        spec = json.loads(path.read_text(encoding="utf-8"))
        print(f"[{cid}] n={spec['n']:,} -> train {spec['n_train']:,} / "
              f"val {spec['n_val']:,} / test {spec['n_test']:,}  "
              f"(pos-share corpus {spec['pos_share_corpus']:.4f})")

    print("\n" + "=" * 74)
    print("SPLIT SANITY ASSERTS")
    print("=" * 74)
    n_fail = 0
    for cid in corpora:
        for name, ok, detail in run_sanity_asserts(cid):
            flag = "PASS" if ok else "FAIL"
            if not ok:
                n_fail += 1
            print(f"  [{flag}] {name:28s} {detail}")
    print("=" * 74)
    if n_fail:
        raise AssertionError(f"{n_fail} split sanity assert(s) failed.")
    print("All split sanity asserts passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
