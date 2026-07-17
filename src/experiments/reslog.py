"""
Shared results log (results/results.csv).

`upsert` appends rows idempotently: any existing row with the same run_id is
replaced, so re-running an experiment script updates its rows in place rather than
duplicating them or wiping another script's rows. All experiment scripts (baselines,
HPO, dense embeddings, cross-corpus) share this one log and this schema.
"""

from __future__ import annotations

import csv
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = _PROJECT_ROOT / "results"
RESULTS_CSV = RESULTS_DIR / "results.csv"

FIELDNAMES = [
    "run_id", "timestamp", "model", "features", "text_field",
    "train_corpus", "test_corpus", "n_train", "n_test",
    "accuracy", "precision", "recall", "f1", "roc_auc",
    "runtime_s", "seed", "notes",
]


def _read(csv_path: Path = RESULTS_CSV) -> list[dict]:
    if not csv_path.exists():
        return []
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def upsert(rows: list[dict], csv_path: Path | None = None):
    """Replace rows sharing a run_id with `rows`, keep everything else, rewrite.

    `csv_path` defaults to the canonical results/results.csv; pass an alternative log
    (e.g. results_512.csv) to keep an isolated experiment out of the canonical file.
    """
    csv_path = csv_path or RESULTS_CSV
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    incoming_ids = {r["run_id"] for r in rows}
    kept = [r for r in _read(csv_path) if r["run_id"] not in incoming_ids]
    out = kept + rows
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        for r in out:
            w.writerow({k: r.get(k, "") for k in FIELDNAMES})
