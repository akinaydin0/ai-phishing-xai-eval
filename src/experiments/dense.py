"""
Dense-embedding baselines: Word2Vec (trained per corpus) and pretrained GloVe,
mean-pooled to document vectors, classified by Logistic Regression and a small MLP.

Leakage discipline: Word2Vec is trained on the TRAIN split only; the StandardScaler
is fit on TRAIN only; the MLP early-stops on the frozen VAL split; TEST is evaluated
once. GloVe is pretrained (no corpus fitting) and skipped gracefully if absent.

Run: python -m src.experiments.dense --all
"""

from __future__ import annotations

import argparse
import re
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

try:
    from src.experiments import reslog
    from src.splits import TEXT_FIELD, load_splits
except ImportError:  # pragma: no cover
    from experiments import reslog
    from splits import TEXT_FIELD, load_splits

SEED = 42
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EMB_DIR = _PROJECT_ROOT / "data" / "embeddings"
GLOVE_TXT = EMB_DIR / "glove.6B.300d.txt"
CORPORA = ["d1_decontaminated_sb", "d2", "d3_en"]
DIM = 300
_TOK = re.compile(r"[a-z0-9']+")


def tokenize(text: str):
    return _TOK.findall(text.lower())


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
def train_w2v(train_texts):
    """Deterministic Word2Vec on TRAIN only (workers=1 for reproducibility)."""
    sents = [tokenize(t) for t in train_texts]
    return Word2Vec(sents, vector_size=DIM, window=5, min_count=2, epochs=10,
                    workers=1, seed=SEED).wv


def load_glove():
    if not GLOVE_TXT.exists():
        return None
    vecs = {}
    with open(GLOVE_TXT, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            vecs[parts[0]] = np.asarray(parts[1:], dtype=np.float32)
    return vecs


def doc_vectors(texts, lookup) -> np.ndarray:
    """Mean-pool token vectors present in `lookup` (KeyedVectors or dict)."""
    has = (lookup.__contains__)
    get = lookup.__getitem__
    out = np.zeros((len(texts), DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        vs = [get(w) for w in tokenize(t) if has(w)]
        if vs:
            out[i] = np.mean(vs, axis=0)
    return out


# --------------------------------------------------------------------------- #
# Classifiers
# --------------------------------------------------------------------------- #
def _metrics(y_true, y_pred, y_score):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_score), 4),
    }


def mlp_early_stop(Xtr, ytr, Xval, yval, max_epochs=200, patience=10):
    """MLP(256) with manual early stopping on the frozen VAL split (not test)."""
    clf = MLPClassifier(hidden_layer_sizes=(256,), random_state=SEED,
                        max_iter=1, warm_start=True, learning_rate_init=1e-3)
    best_auc, best_clf, wait = -1.0, None, 0
    for _ in range(max_epochs):
        clf.partial_fit(Xtr, ytr, classes=np.array([0, 1]))
        auc = roc_auc_score(yval, clf.predict_proba(Xval)[:, 1])
        if auc > best_auc + 1e-5:
            best_auc, best_clf, wait = auc, deepcopy(clf), 0
        else:
            wait += 1
            if wait >= patience:
                break
    return best_clf


def _row(model, features, corpus, n_train, n_test, m, runtime, notes):
    return {
        "run_id": f"{model}__{features}__{corpus}__{corpus}__dense",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": model, "features": features, "text_field": TEXT_FIELD[corpus],
        "train_corpus": corpus, "test_corpus": corpus, "n_train": n_train, "n_test": n_test,
        "accuracy": m["accuracy"], "precision": m["precision"], "recall": m["recall"],
        "f1": m["f1"], "roc_auc": m["roc_auc"], "runtime_s": round(runtime, 2),
        "seed": SEED, "notes": notes,
    }


def run_corpus(corpus, glove, rows):
    parts = load_splits(corpus)
    tr, va, te = parts["train"], parts["val"], parts["test"]
    ytr, yva, yte = (tr["label"].to_numpy(), va["label"].to_numpy(), te["label"].to_numpy())

    embeddings = {"w2v": train_w2v(tr["text"])}
    if glove is not None:
        embeddings["glove"] = glove

    for name, lookup in embeddings.items():
        Xtr = doc_vectors(tr["text"], lookup)
        Xva = doc_vectors(va["text"], lookup)
        Xte = doc_vectors(te["text"], lookup)
        scaler = StandardScaler().fit(Xtr)
        Xtr, Xva, Xte = scaler.transform(Xtr), scaler.transform(Xva), scaler.transform(Xte)

        t0 = time.time()
        lr = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED).fit(Xtr, ytr)
        m = _metrics(yte, lr.predict(Xte), lr.predict_proba(Xte)[:, 1])
        rows.append(_row("LogisticRegression", name, corpus, len(tr), len(te), m,
                         time.time() - t0, f"dense {name} + LR, mean-pooled dim{DIM}"))
        print(f"  {corpus:22s} {name:5s} LR  f1={m['f1']:.4f} auc={m['roc_auc']:.4f}")

        t0 = time.time()
        mlp = mlp_early_stop(Xtr, ytr, Xva, yva)
        m = _metrics(yte, mlp.predict(Xte), mlp.predict_proba(Xte)[:, 1])
        rows.append(_row("MLP", name, corpus, len(tr), len(te), m, time.time() - t0,
                         f"dense {name} + MLP(256) early-stop on val, mean-pooled dim{DIM}"))
        print(f"  {corpus:22s} {name:5s} MLP f1={m['f1']:.4f} auc={m['roc_auc']:.4f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Dense-embedding baselines.")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--corpora", nargs="*", default=CORPORA)
    args = parser.parse_args(argv)

    glove = load_glove()
    print(f"GloVe: {'loaded ' + str(len(glove)) + ' vectors' if glove else 'NOT FOUND — skipping (W2V only)'}")

    rows = []
    for corpus in args.corpora:
        run_corpus(corpus, glove, rows)
    reslog.upsert(rows)
    print(f"\n{len(rows)} dense rows upserted to results/results.csv")


if __name__ == "__main__":
    main()
