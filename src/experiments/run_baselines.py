"""
Classical baselines: in-corpus, representation ablation, length-only shortcut floor,
and cross-corpus generalisation. Appends one row per run to results/results.csv.

Leakage discipline (enforced structurally):
  * All train/val/test data comes from the frozen splits (src.splits.load_splits).
  * TF-IDF vectorizers and the length StandardScaler are fit on the SOURCE TRAIN
    split only; val/test and target corpora are only ever transformed.
  * Cross-corpus runs reuse the source-trained model/vectorizer and apply it to the
    target TEST split — so a source model never sees any target data.
  * Nothing is ever fit on a test split. The val split is left untouched (reserved
    for later hyper-parameter tuning).

Run: python -m src.experiments.run_baselines
"""

from __future__ import annotations

import csv
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import MultinomialNB  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

try:
    from src.features import build_tfidf, length_features, tfidf_transform
    from src.splits import TEXT_FIELD, load_splits
except ImportError:  # pragma: no cover
    from features import build_tfidf, length_features, tfidf_transform
    from splits import TEXT_FIELD, load_splits

SEED = 42
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = _PROJECT_ROOT / "results"
RESULTS_CSV = RESULTS_DIR / "results.csv"
FIG_DIR = _PROJECT_ROOT / "figures" / "baselines"


def save_confusion_matrix(y_true, y_pred, corpus_id, n_test, text_field):
    """Confusion matrix for an in-corpus LR run. Title names the universe."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.7, 4.1))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["legit (0)", "phish (1)"])
    ax.set_yticklabels(["legit (0)", "phish (1)"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=12,
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_title(f"Confusion matrix — LogReg TF-IDF (in-corpus)\n"
                 f"{corpus_id} [{text_field}], test n={n_test:,}", fontsize=9)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"cm_{corpus_id}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"cm_{corpus_id}.pdf", bbox_inches="tight")
    plt.close(fig)

PRIMARY = ["d1_decontaminated_sb", "d2", "d3_en"]  # cross-corpus universe
ABLATION = "d1_decontaminated"                      # D1 text_combined

FIELDNAMES = [
    "run_id", "timestamp", "model", "features", "text_field",
    "train_corpus", "test_corpus", "n_train", "n_test",
    "accuracy", "precision", "recall", "f1", "roc_auc",
    "runtime_s", "seed", "notes",
]


def _metrics(y_true, y_pred, y_score) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_score), 4),
    }


def _append_row(row: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    new = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if new:
            w.writeheader()
        w.writerow(row)


def _row(model, features, text_field, train_corpus, test_corpus,
         n_train, n_test, m, runtime_s, notes) -> dict:
    return {
        "run_id": f"{model}__{features}__{train_corpus}__{test_corpus}",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": model, "features": features, "text_field": text_field,
        "train_corpus": train_corpus, "test_corpus": test_corpus,
        "n_train": n_train, "n_test": n_test,
        "accuracy": m["accuracy"], "precision": m["precision"], "recall": m["recall"],
        "f1": m["f1"], "roc_auc": m["roc_auc"],
        "runtime_s": round(runtime_s, 2), "seed": SEED, "notes": notes,
    }


def main():
    np.random.seed(SEED)
    if RESULTS_CSV.exists():
        RESULTS_CSV.unlink()  # fresh log for this deterministic batch

    corpora = PRIMARY + [ABLATION]
    parts = {c: load_splits(c) for c in corpora}
    for c in corpora:
        print(f"[{c}] train={len(parts[c]['train']):,} test={len(parts[c]['test']):,} "
              f"field={TEXT_FIELD[c]}")

    # caches reused by the cross-corpus runs (source-trained artefacts only)
    tfidf_vec, tfidf_lr = {}, {}
    len_scaler, len_lr = {}, {}

    # ---------- Run A: in-corpus TF-IDF (LR + MultinomialNB) ----------
    print("\n== Run A: in-corpus TF-IDF (LR, NB) ==")
    for c in PRIMARY:
        tr, te = parts[c]["train"], parts[c]["test"]
        ytr, yte = tr["label"].to_numpy(), te["label"].to_numpy()
        vec = build_tfidf(tr["text"])
        Xtr, Xte = tfidf_transform(vec, tr["text"]), tfidf_transform(vec, te["text"])
        tfidf_vec[c] = vec

        t0 = time.time()
        lr = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED)
        lr.fit(Xtr, ytr)
        preds = lr.predict(Xte)
        m = _metrics(yte, preds, lr.predict_proba(Xte)[:, 1])
        _append_row(_row("LogisticRegression", "tfidf", TEXT_FIELD[c], c, c,
                         len(tr), len(te), m, time.time() - t0, "in-corpus"))
        save_confusion_matrix(yte, preds, c, len(te), TEXT_FIELD[c])
        tfidf_lr[c] = lr
        print(f"  {c:22s} LR   f1={m['f1']:.4f} auc={m['roc_auc']:.4f}")

        t0 = time.time()
        nb = MultinomialNB()
        nb.fit(Xtr, ytr)
        m = _metrics(yte, nb.predict(Xte), nb.predict_proba(Xte)[:, 1])
        _append_row(_row("MultinomialNB", "tfidf", TEXT_FIELD[c], c, c,
                         len(tr), len(te), m, time.time() - t0, "in-corpus"))
        print(f"  {c:22s} NB   f1={m['f1']:.4f} auc={m['roc_auc']:.4f}")

    # ---------- Run B: D1 representation ablation (text_combined) ----------
    print("\n== Run B: D1 representation ablation (text_combined, LR) ==")
    tr, te = parts[ABLATION]["train"], parts[ABLATION]["test"]
    ytr, yte = tr["label"].to_numpy(), te["label"].to_numpy()
    vec = build_tfidf(tr["text"])
    Xtr, Xte = tfidf_transform(vec, tr["text"]), tfidf_transform(vec, te["text"])
    t0 = time.time()
    lr = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED)
    lr.fit(Xtr, ytr)
    preds = lr.predict(Xte)
    m = _metrics(yte, preds, lr.predict_proba(Xte)[:, 1])
    _append_row(_row("LogisticRegression", "tfidf", TEXT_FIELD[ABLATION], ABLATION, ABLATION,
                     len(tr), len(te), m, time.time() - t0, "representation ablation vs d1_decontaminated_sb"))
    save_confusion_matrix(yte, preds, ABLATION, len(te), TEXT_FIELD[ABLATION])
    print(f"  {ABLATION:22s} LR   f1={m['f1']:.4f} auc={m['roc_auc']:.4f}")

    # ---------- Run C: length-only shortcut floor (in-corpus) ----------
    print("\n== Run C: length-only shortcut floor (StandardScaler + LR) ==")
    for c in PRIMARY:
        tr, te = parts[c]["train"], parts[c]["test"]
        ytr, yte = tr["label"].to_numpy(), te["label"].to_numpy()
        scaler = StandardScaler().fit(length_features(tr["text"]))
        Xtr = scaler.transform(length_features(tr["text"]))
        Xte = scaler.transform(length_features(te["text"]))
        t0 = time.time()
        lr = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED)
        lr.fit(Xtr, ytr)
        m = _metrics(yte, lr.predict(Xte), lr.predict_proba(Xte)[:, 1])
        _append_row(_row("LogisticRegression", "length3", TEXT_FIELD[c], c, c,
                         len(tr), len(te), m, time.time() - t0, "shortcut floor (token/char/log-token)"))
        len_scaler[c], len_lr[c] = scaler, lr
        print(f"  {c:22s} LEN  f1={m['f1']:.4f} auc={m['roc_auc']:.4f}")

    # ---------- Run D: cross-corpus, all 6 directed pairs ----------
    print("\n== Run D: cross-corpus (6 directed pairs; TF-IDF LR + length floor) ==")
    for src in PRIMARY:
        for tgt in PRIMARY:
            if src == tgt:
                continue
            te = parts[tgt]["test"]
            yte = te["label"].to_numpy()
            n_src_train = len(parts[src]["train"])

            # TF-IDF LR: source vectorizer + source LR applied to target test.
            t0 = time.time()
            Xte = tfidf_transform(tfidf_vec[src], te["text"])
            m = _metrics(yte, tfidf_lr[src].predict(Xte), tfidf_lr[src].predict_proba(Xte)[:, 1])
            _append_row(_row("LogisticRegression", "tfidf", "subject_body", src, tgt,
                             n_src_train, len(te), m, time.time() - t0, "cross-corpus"))
            print(f"  {src:20s} -> {tgt:20s} LR   f1={m['f1']:.4f} auc={m['roc_auc']:.4f}")

            # Length floor: source scaler + source LR applied to target test.
            t0 = time.time()
            Xte = len_scaler[src].transform(length_features(te["text"]))
            m = _metrics(yte, len_lr[src].predict(Xte), len_lr[src].predict_proba(Xte)[:, 1])
            _append_row(_row("LogisticRegression", "length3", "subject_body", src, tgt,
                             n_src_train, len(te), m, time.time() - t0, "cross-corpus shortcut floor"))

    print(f"\nWrote {RESULTS_CSV.relative_to(_PROJECT_ROOT)}")
    write_baseline_summary()
    print(f"Wrote notes/baseline_summary.md")


SHORT = {"d1_decontaminated_sb": "d1_sb", "d1_decontaminated": "d1_tc",
         "d2": "d2", "d3_en": "d3_en"}


def write_baseline_summary():
    """Generate notes/baseline_summary.md from results/results.csv (live numbers)."""
    import pandas as pd

    df = pd.read_csv(RESULTS_CSV)
    notes_path = _PROJECT_ROOT / "notes" / "baseline_summary.md"

    def g(model, feat, tr, te):
        r = df[(df.model == model) & (df.features == feat)
               & (df.train_corpus == tr) & (df.test_corpus == te)]
        return (float(r.f1.iloc[0]), float(r.roc_auc.iloc[0])) if len(r) else (float("nan"), float("nan"))

    prim = PRIMARY  # d1_sb, d2, d3_en (full ids)
    L = []
    L.append("# Classical Baseline Summary\n")
    L.append("TF-IDF (1–2 gram, min_df=2, sublinear, ≤100k feats) + Logistic Regression "
             "(`class_weight='balanced'`) and Multinomial NB, on the frozen 70/15/15 splits "
             "(seed 42). Positive class = label 1 (phishing/spam). Vectorizers/scalers fit on "
             "the source **train** split only; models evaluated on the **test** split; the val "
             "split is untouched. Full log: `results/results.csv`.\n")

    # ---- compact table ----
    L.append("## Results (model × text_field × train→test × F1)\n")
    L.append("| Run | Model | Features | text_field | train → test | F1 | ROC-AUC |")
    L.append("|---|---|---|---|---|--:|--:|")
    for c in prim:
        f1, auc = g("LogisticRegression", "tfidf", c, c)
        L.append(f"| A | LogReg | tfidf | {TEXT_FIELD[c]} | {SHORT[c]} → {SHORT[c]} | {f1:.4f} | {auc:.4f} |")
        f1, auc = g("MultinomialNB", "tfidf", c, c)
        L.append(f"| A | NB | tfidf | {TEXT_FIELD[c]} | {SHORT[c]} → {SHORT[c]} | {f1:.4f} | {auc:.4f} |")
    f1, auc = g("LogisticRegression", "tfidf", ABLATION, ABLATION)
    L.append(f"| B | LogReg | tfidf | text_combined | d1_tc → d1_tc | {f1:.4f} | {auc:.4f} |")
    for c in prim:
        f1, auc = g("LogisticRegression", "length3", c, c)
        L.append(f"| C | LogReg | length3 | {TEXT_FIELD[c]} | {SHORT[c]} → {SHORT[c]} | {f1:.4f} | {auc:.4f} |")
    for src in prim:
        for tgt in prim:
            if src == tgt:
                continue
            f1, auc = g("LogisticRegression", "tfidf", src, tgt)
            L.append(f"| D | LogReg | tfidf | subject_body | {SHORT[src]} → {SHORT[tgt]} | {f1:.4f} | {auc:.4f} |")
            f1, auc = g("LogisticRegression", "length3", src, tgt)
            L.append(f"| D | LogReg | length3 | subject_body | {SHORT[src]} → {SHORT[tgt]} | {f1:.4f} | {auc:.4f} |")
    L.append("")

    # ---- numbers used in the observations ----
    inc = {c: g("LogisticRegression", "tfidf", c, c)[0] for c in prim}
    nb = {c: g("MultinomialNB", "tfidf", c, c)[0] for c in prim}
    floor = {c: g("LogisticRegression", "length3", c, c)[0] for c in prim}
    d1_sb_f1 = inc["d1_decontaminated_sb"]
    d1_tc_f1 = g("LogisticRegression", "tfidf", ABLATION, ABLATION)[0]
    xc = {(s, t): g("LogisticRegression", "tfidf", s, t)[0]
          for s in prim for t in prim if s != t}
    xc_auc = {(s, t): g("LogisticRegression", "tfidf", s, t)[1]
              for s in prim for t in prim if s != t}
    xc_floor_auc = {(s, t): g("LogisticRegression", "length3", s, t)[1]
                    for s in prim for t in prim if s != t}
    p = lambda k: SHORT[k]

    L.append("## Observations\n")
    L.append(f"1. **In-corpus levels match the literature.** TF-IDF + LR reaches F1 "
             f"{inc['d1_decontaminated_sb']:.3f} (d1_sb), {inc['d2']:.3f} (d2), "
             f"{inc['d3_en']:.3f} (d3_en) with ROC-AUC ≥ 0.995 — the 0.95–0.99 band reported "
             f"for linear models on Enron-Spam and curated phishing corpora. LR edges out "
             f"Multinomial NB everywhere (e.g. d3_en {inc['d3_en']:.3f} vs {nb['d3_en']:.3f}). "
             f"No inflated or degenerate scores, so the splits and pipeline are behaving.")
    L.append("")
    worst_src = min(prim, key=lambda s: (xc[(s, [t for t in prim if t != s][0])] + xc[(s, [t for t in prim if t != s][1])]) / 2)
    L.append(f"2. **Cross-corpus degradation is large and asymmetric.** In-corpus ≈0.98 F1 "
             f"falls to {min(xc.values()):.3f}–{max(xc.values()):.3f} across the 6 directed "
             f"pairs. Training on AI-generated d3_en transfers worst "
             f"(d3_en→d1_sb {xc[('d3_en','d1_decontaminated_sb')]:.3f}, "
             f"d3_en→d2 {xc[('d3_en','d2')]:.3f}), whereas real corpora detect AI phishing "
             f"well (d1_sb→d3_en {xc[('d1_decontaminated_sb','d3_en')]:.3f}, "
             f"d2→d3_en {xc[('d2','d3_en')]:.3f}); d1_sb↔d2 is roughly symmetric "
             f"({xc[('d1_decontaminated_sb','d2')]:.3f} / {xc[('d2','d1_decontaminated_sb')]:.3f}). "
             f"AI-generated data is a weak training source for real-world detection — direct "
             f"support for the dataset-aware evaluation mandate.")
    L.append("")
    L.append(f"3. **The length-only shortcut floor is real but beatable in-corpus, and "
             f"collapses out-of-corpus.** In-corpus the floor reaches F1 "
             f"{floor['d1_decontaminated_sb']:.3f}/{floor['d2']:.3f}/{floor['d3_en']:.3f} "
             f"(d1_sb/d2/d3_en); TF-IDF beats it by "
             f"{d1_sb_f1 - floor['d1_decontaminated_sb']:.2f}/"
             f"{inc['d2'] - floor['d2']:.2f}/{inc['d3_en'] - floor['d3_en']:.2f} F1, so models "
             f"learn far more than length. d3_en has the highest floor ({floor['d3_en']:.3f}) — "
             f"structurally uniform LLM output carries the most length signal, a mild shortcut "
             f"flag for AI data. Cross-corpus, the floor **inverts**: several directions post "
             f"ROC-AUC < 0.5 (d2→d3_en {xc_floor_auc[('d2','d3_en')]:.3f}, "
             f"d2→d1_sb {xc_floor_auc[('d2','d1_decontaminated_sb')]:.3f}), i.e. the "
             f"length–label relationship reverses between corpora. Length is a non-transferable, "
             f"corpus-specific shortcut.")
    L.append("")
    L.append(f"4. **subject+body vs text_combined: a negligible in-corpus gap → subject+body "
             f"for modelling.** In-corpus D1 LR F1 is {d1_sb_f1:.4f} (subject_body) vs "
             f"{d1_tc_f1:.4f} (text_combined) — a {abs(d1_tc_f1 - d1_sb_f1) * 100:.2f} pp "
             f"difference (text_combined marginally higher; its stop-word/punctuation stripping "
             f"mildly denoises the bag-of-words). The representation choice therefore costs "
             f"~nothing in accuracy, and subject+body is preferred for modelling because it "
             f"(a) is representation-consistent with D2/D3 (both natural text), enabling clean "
             f"cross-corpus experiments, and (b) preserves stop-words/punctuation so Pipeline A's "
             f"stop-word-removal ablation is possible (impossible on the pre-stripped "
             f"text_combined). **Upgraded from working hypothesis to evidence-backed decision.**")
    L.append("")
    L.append(f"5. **Anomaly — ranking transfers but the threshold does not.** Several "
             f"cross-corpus directions keep high ROC-AUC while F1 drops sharply "
             f"(d2→d1_sb AUC {xc_auc[('d2','d1_decontaminated_sb')]:.3f} at F1 "
             f"{xc[('d2','d1_decontaminated_sb')]:.3f}; d3_en→d1_sb AUC "
             f"{xc_auc[('d3_en','d1_decontaminated_sb')]:.3f} at F1 "
             f"{xc[('d3_en','d1_decontaminated_sb')]:.3f}). The model still separates the "
             f"classes, but the fixed 0.5 threshold is miscalibrated for the target's class "
             f"prior — a calibration artifact, not a feature failure. Cross-corpus deployment "
             f"needs target-domain threshold recalibration, and AUC must be reported alongside F1.")
    L.append("")
    L.append("## Confusion matrices\n")
    L.append("Per in-corpus LR run, under `figures/baselines/` (PNG + PDF): "
             "`cm_d1_decontaminated_sb`, `cm_d1_decontaminated` (text_combined ablation), "
             "`cm_d2`, `cm_d3_en`.")

    notes_path.write_text("\n".join(L), encoding="utf-8")
    return notes_path


if __name__ == "__main__":
    main()
