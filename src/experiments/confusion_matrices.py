"""
Confusion matrices at the committed 0.5 operating point for the six Table 6.1 finalists.

Regenerated under the thesis's reproducibility-by-regeneration policy: no committed artifact stores
TN/FP/FN/TP, so the detail cells are produced here from committed anchors by a deterministic script.
**No training and no tuning.** Sklearn finalists are rebuilt by the committed champions inference
module (`src.xai.champions._SklearnChampion`), which re-fits the model from its committed HPO
`best_params` on the frozen TRAIN split -- the identical inference path that produced the committed XAI
results -- and LinearSVC is mapped through the committed `expit(decision_function)` margin mapping.
Transformer finalists load their saved fine-tuned checkpoints (no fitting) at the committed
`MAX_LEN=256`. Each finalist is scored exactly once on its OWN frozen TEST partition.

VERIFICATION GATE (hard): accuracy / precision / recall / F1 recomputed from each matrix must equal the
committed values (results.csv `__tuned` rows for classical, results/transformers/*.json `test` block for
transformers) at their stored precision (4 dp). Any mismatch exits non-zero and reports; nothing is
reconciled silently.

Writes results/confusion_matrices.json.
Run: python -m src.experiments.confusion_matrices
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

try:
    from src.splits import load_splits
    from src.xai import champions as C
except ImportError:  # pragma: no cover
    from splits import load_splits
    from xai import champions as C

_ROOT = Path(__file__).resolve().parent.parent.parent
OUT = _ROOT / "results" / "confusion_matrices.json"
RESULTS_CSV = _ROOT / "results" / "results.csv"
TF_DIR = _ROOT / "results" / "transformers"
THRESHOLD = 0.5
DP = 4

# (key, kind, corpus, builder-spec, committed-metric source)
FINALISTS = [
    ("LinearSVC__d1_sb", "classical", "d1_decontaminated_sb", ("svc",), ("LinearSVC", "d1_decontaminated_sb")),
    ("MultinomialNB__d2", "classical", "d2", ("nb",), ("MultinomialNB", "d2")),
    ("LinearSVC__d3_en", "classical", "d3_en", ("svc",), ("LinearSVC", "d3_en")),
    ("DistilBERT-256__d1_sb", "transformer", "d1_decontaminated_sb",
     ("distilbert-base-uncased",), ("distilbert-base-uncased", "d1_decontaminated_sb")),
    ("DistilBERT-256__d2", "transformer", "d2",
     ("distilbert-base-uncased",), ("distilbert-base-uncased", "d2")),
    ("RoBERTa-256__d3_en", "transformer", "d3_en",
     ("roberta-base",), ("roberta-base", "d3_en")),
]


def _committed_classical(model: str, corpus: str) -> dict:
    with open(RESULTS_CSV, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["model"] == model and r["test_corpus"] == corpus and r["run_id"].endswith("__tuned"):
                return {k: float(r[k]) for k in ("accuracy", "precision", "recall", "f1")} | {
                    "n_test": int(r["n_test"]), "source": f"results/results.csv run_id={r['run_id']}"}
    raise KeyError(f"no committed __tuned row for {model}/{corpus}")


def _committed_transformer(model: str, corpus: str) -> dict:
    p = TF_DIR / f"{model}__{corpus}.json"
    j = json.loads(p.read_text(encoding="utf-8"))
    t = j["test"]
    return {k: float(t[k]) for k in ("accuracy", "precision", "recall", "f1")} | {
        "n_test": int(j["n_test"]), "source": f"results/transformers/{model}__{corpus}.json:test"}


def main() -> int:
    print("Confusion matrices at the committed 0.5 operating point (no training, no tuning)\n"
          + "=" * 88, flush=True)
    out: dict = {
        "policy": ("regenerated under the thesis reproducibility-by-regeneration policy; no committed "
                   "artifact stores TN/FP/FN/TP"),
        "operating_point": THRESHOLD,
        "linearsvc_mapping": "p1 = expit(decision_function) (committed margin mapping)",
        "transformer_max_length": C.MAX_LEN,
        "seed": C.SEED,
        "gate": ("accuracy/precision/recall/F1 recomputed from each matrix must equal the committed "
                 f"values at {DP} dp"),
        "finalists": {},
    }
    failures: list[str] = []

    for key, kind, corpus, spec, src in FINALISTS:
        te = load_splits(corpus)["test"].reset_index(drop=True)
        y = te["label"].to_numpy()
        texts = te["text"].tolist()

        if kind == "classical":
            champ = C._SklearnChampion(spec[0], corpus)
            committed = _committed_classical(*src)
            model_name = champ.name
        else:
            champ = C.TransformerChampion(spec[0], corpus)
            committed = _committed_transformer(*src)
            model_name = f"{champ.name}-{champ.max_length}"

        p1 = champ.proba(texts)
        pred = (np.asarray(p1) >= THRESHOLD).astype(int)

        tn = int(np.sum((y == 0) & (pred == 0)))
        fp = int(np.sum((y == 0) & (pred == 1)))
        fn = int(np.sum((y == 1) & (pred == 0)))
        tp = int(np.sum((y == 1) & (pred == 1)))
        assert tn + fp + fn + tp == len(y) == committed["n_test"], f"{key}: row sums != n_test"

        recomputed = {
            "accuracy": round(float(accuracy_score(y, pred)), DP),
            "precision": round(float(precision_score(y, pred, zero_division=0)), DP),
            "recall": round(float(recall_score(y, pred, zero_division=0)), DP),
            "f1": round(float(f1_score(y, pred, zero_division=0)), DP),
        }
        deltas = {k: round(recomputed[k] - round(committed[k], DP), DP) for k in recomputed}
        ok = all(abs(v) < 1e-9 for v in deltas.values())
        if not ok:
            failures.append(f"{key}: recomputed {recomputed} vs committed "
                            f"{ {k: committed[k] for k in recomputed} } deltas {deltas}")

        out["finalists"][key] = {
            "kind": kind, "model": model_name, "corpus": corpus, "n_test": int(len(y)),
            "confusion": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
            "recomputed": recomputed,
            "committed": {k: committed[k] for k in recomputed},
            "committed_source": committed["source"],
            "gate_deltas": deltas, "gate_pass": ok,
        }
        print(f"  [{'PASS' if ok else 'FAIL'}] {key:24s} TN={tn:5d} FP={fp:4d} FN={fn:4d} TP={tp:5d} "
              f"| n={len(y)} | acc {recomputed['accuracy']:.4f} vs {committed['accuracy']:.4f} "
              f"prec {recomputed['precision']:.4f} vs {committed['precision']:.4f} "
              f"rec {recomputed['recall']:.4f} vs {committed['recall']:.4f} "
              f"f1 {recomputed['f1']:.4f} vs {committed['f1']:.4f}", flush=True)

    out["gate_pass_all"] = not failures
    print("=" * 88, flush=True)
    if failures:
        print(f"GATE FAILED ({len(failures)} of {len(FINALISTS)} finalists):", flush=True)
        for f in failures:
            print("  - " + f, flush=True)
        print("STOPPING: no artifact written, nothing reconciled.", flush=True)
        return 1
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"GATE PASSED on {len(FINALISTS)}/{len(FINALISTS)} finalists. Wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
