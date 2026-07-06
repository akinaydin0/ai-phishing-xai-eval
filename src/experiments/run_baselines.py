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
import json
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

    # ---------- Trivial reference baselines (no training) ----------
    print("\n== Trivial references: all_positive, majority_class ==")
    for c in PRIMARY:
        te = parts[c]["test"]
        yte = te["label"].to_numpy()
        ones = np.ones(len(yte), dtype=int)
        m = _metrics(yte, ones, ones.astype(float))  # constant score -> AUC 0.5
        _append_row(_row("all_positive", "none", TEXT_FIELD[c], c, c,
                         0, len(te), m, 0.0, "trivial reference: predict all positive (label 1)"))
        maj = int(yte.mean() >= 0.5)
        preds = np.full(len(yte), maj, dtype=int)
        m = _metrics(yte, preds, preds.astype(float))
        _append_row(_row("majority_class", "none", TEXT_FIELD[c], c, c,
                         0, len(te), m, 0.0, f"trivial reference: predict majority class ({maj})"))
        print(f"  {c:22s} all_positive + majority({maj}) written")

    # ---------- Run E: monotonic length probe (token_count only) ----------
    # A single feature => the LR score is monotonic in token_count, so cross-corpus
    # ROC-AUC exactly encodes direction: AUC>0.5 <=> longer emails are more positive
    # in the target. The source coef sign gives the source's learned direction.
    print("\n== Run E: monotonic length probe (token_count-only LR) ==")

    def _toklen(texts):
        return np.array([len(t.split()) for t in texts], dtype=float).reshape(-1, 1)

    tok_sc, tok_lr, tok_sign = {}, {}, {}
    for c in PRIMARY:
        tr, te = parts[c]["train"], parts[c]["test"]
        ytr, yte = tr["label"].to_numpy(), te["label"].to_numpy()
        sc = StandardScaler().fit(_toklen(tr["text"]))  # scaling preserves the coef sign
        lr = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED)
        lr.fit(sc.transform(_toklen(tr["text"])), ytr)
        sign = "+" if lr.coef_[0][0] > 0 else "-"
        tok_sc[c], tok_lr[c], tok_sign[c] = sc, lr, sign
        Xte = sc.transform(_toklen(te["text"]))
        m = _metrics(yte, lr.predict(Xte), lr.predict_proba(Xte)[:, 1])
        _append_row(_row("LogisticRegression", "token_count", TEXT_FIELD[c], c, c,
                         len(tr), len(te), m, 0.0, f"monotonic length probe; coef_sign={sign}"))
        print(f"  {c:22s} in-corpus auc={m['roc_auc']:.4f} coef_sign={sign}")
    for src in PRIMARY:
        for tgt in PRIMARY:
            if src == tgt:
                continue
            te = parts[tgt]["test"]
            yte = te["label"].to_numpy()
            Xte = tok_sc[src].transform(_toklen(te["text"]))
            m = _metrics(yte, tok_lr[src].predict(Xte), tok_lr[src].predict_proba(Xte)[:, 1])
            _append_row(_row("LogisticRegression", "token_count", "subject_body", src, tgt,
                             len(parts[src]["train"]), len(te), m, 0.0,
                             f"monotonic length probe (cross); source coef_sign={tok_sign[src]}"))
            print(f"  {src:20s} -> {tgt:20s} auc={m['roc_auc']:.4f}")

    # ---------- Coefficient sidecar (consumed by the summary generator) ----------
    floor_signs = {c: ["+" if v > 0 else "-" for v in len_lr[c].coef_[0]] for c in PRIMARY}
    coefs = {
        "probe_token_count_sign": tok_sign,           # per source corpus
        "floor_signs_tok_char_logtok": floor_signs,   # per source corpus, order [tok, char, log-tok]
    }
    (RESULTS_DIR / "baseline_coefs.json").write_text(json.dumps(coefs, indent=2), encoding="utf-8")
    print(f"  floor coef signs (tok/char/logtok): "
          + ", ".join(f"{c}={'/'.join(floor_signs[c])}" for c in PRIMARY))

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

    # ---- trivial references + monotonic probe + floor coef signs ----
    coefs = json.loads((RESULTS_DIR / "baseline_coefs.json").read_text(encoding="utf-8"))
    probe_sign = coefs["probe_token_count_sign"]
    floor_sign = coefs["floor_signs_tok_char_logtok"]

    L.append("## Trivial reference baselines (no training)\n")
    L.append("| Corpus | all_positive F1 | majority_class F1 | ROC-AUC |")
    L.append("|---|--:|--:|--:|")
    for c in prim:
        L.append(f"| {SHORT[c]} | {g('all_positive','none',c,c)[0]:.4f} | "
                 f"{g('majority_class','none',c,c)[0]:.4f} | 0.50 |")
    L.append("")
    L.append("*Constant predictors → ROC-AUC 0.50. `majority_class` predicts label 0 on d2 "
             "(ham-majority) so its positive-class F1 = 0; on d1_sb/d3_en it coincides with "
             "`all_positive`.*\n")

    L.append("## Monotonic length probe — token_count-only LR (ROC-AUC)\n")
    L.append("| Source (coef sign) | → d1_sb | → d2 | → d3_en |")
    L.append("|---|--:|--:|--:|")
    for src in prim:
        cells = []
        for tgt in prim:
            a = g("LogisticRegression", "token_count", src, tgt)[1]
            cells.append(f"**{a:.3f}**" if src == tgt else f"{a:.3f}")
        L.append(f"| {SHORT[src]} ({probe_sign[src]}) | {cells[0]} | {cells[1]} | {cells[2]} |")
    L.append("")
    L.append("*Diagonal = in-corpus. All three source coef signs are **−** (model learns "
             "*shorter→positive*), so every source produces the same ranking and each column is "
             "source-independent — pure length **direction** is universal across the three "
             "corpora; only its strength varies.*\n")

    L.append("## 3-feature floor coefficient signs (tok / char / log-tok)\n")
    L.append("| Source | tok | char | log-tok |")
    L.append("|---|:--:|:--:|:--:|")
    for c in prim:
        s = floor_sign[c]
        L.append(f"| {SHORT[c]} | {s[0]} | {s[1]} | {s[2]} |")
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
    ap = {c: g("all_positive", "none", c, c)[0] for c in prim}
    floor_auc = {c: g("LogisticRegression", "length3", c, c)[1] for c in prim}
    probe_in = {c: g("LogisticRegression", "token_count", c, c)[1] for c in prim}
    probe_auc = {(s, t): g("LogisticRegression", "token_count", s, t)[1]
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
    L.append(f"3. **The 3-feature 'length floor' barely clears the trivial all-positive "
             f"baseline on F1 — judge it on ROC-AUC.** In-corpus the floor's F1 is "
             f"{floor['d1_decontaminated_sb']:.3f}/{floor['d2']:.3f}/{floor['d3_en']:.3f} "
             f"(d1_sb/d2/d3_en) — at or just **below** the all-positive reference "
             f"{ap['d1_decontaminated_sb']:.3f}/{ap['d2']:.3f}/{ap['d3_en']:.3f}, because both "
             f"are dominated by the positive-class prior, so F1 alone makes the floor look "
             f"competent when it has learned almost nothing. Its real (weak) signal is in "
             f"ROC-AUC {floor_auc['d1_decontaminated_sb']:.3f}/{floor_auc['d2']:.3f}/"
             f"{floor_auc['d3_en']:.3f} — only modestly above the 0.5 chance line (d2 nearly at "
             f"chance), vs TF-IDF LR's ≥0.995. Cross-corpus the floor's AUC drops below 0.5 "
             f"(d2→d1_sb {xc_floor_auc[('d2','d1_decontaminated_sb')]:.3f}, "
             f"d2→d3_en {xc_floor_auc[('d2','d3_en')]:.3f}) — what signal it has does not "
             f"transfer. **Takeaway:** any model whose F1 merely matches "
             f"~{ap['d1_decontaminated_sb']:.2f}/{ap['d2']:.2f}/{ap['d3_en']:.2f} has learned "
             f"nothing beyond the class prior; models must be judged on AUC and cross-corpus "
             f"behaviour, not in-corpus F1 alone.")
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
             f"text_combined). **Upgraded from working hypothesis to evidence-backed decision.** "
             f"*Caveat: subject+body (51,839) and text_combined (52,704) are different "
             f"post-dedup universes (865-row delta), so the "
             f"{abs(d1_tc_f1 - d1_sb_f1) * 100:.2f} pp gap is indicative, not a paired test on "
             f"identical emails.*")
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
    L.append(f"6. **Transfer direction, made exact by a monotonic (token_count-only) probe — "
             f"with a correction.** A single-feature LR is monotonic in length, so its "
             f"cross-corpus ROC-AUC exactly encodes whether the source's learned length "
             f"direction holds in the target. The fitted coef sign is **negative in all three "
             f"source corpora** — the model learns **shorter→positive**, not longer→positive as "
             f"the directive anticipated (correction: phishing/spam skews shorter in every "
             f"corpus here — short modern-phishing and Enron-spam medians). Because all three "
             f"signs agree, each source produces the same ranking, so the cross-corpus probe "
             f"AUC is source-independent: →d1_sb {probe_auc[('d2','d1_decontaminated_sb')]:.3f}, "
             f"→d2 {probe_auc[('d1_decontaminated_sb','d2')]:.3f}, →d3_en "
             f"{probe_auc[('d1_decontaminated_sb','d3_en')]:.3f} (all > 0.5). Pure length "
             f"**direction is universal** across the three corpora; only its strength varies "
             f"(strong in d1_sb, near-chance in d2/d3_en).")
    L.append("")
    L.append(f"### Why the 3-feature floor inverts (d2→d1_sb AUC "
             f"{xc_floor_auc[('d2','d1_decontaminated_sb')]:.3f})\n")
    L.append(f"The floor fits mixed-sign coefficients on the collinear [token, char, log-token] "
             f"features — d1_sb {'/'.join(floor_sign['d1_decontaminated_sb'])}, "
             f"d2 {'/'.join(floor_sign['d2'])}, d3_en {'/'.join(floor_sign['d3_en'])}. For d2 "
             f"the token(−)/char(+) contrast on near-collinear features cancels the pure-length "
             f"component and leaves a **chars-per-token / character-density style signal**, not "
             f"length. That style signal is corpus-specific and reverses on d1_sb, giving "
             f"d2→d1_sb floor AUC {xc_floor_auc[('d2','d1_decontaminated_sb')]:.3f} (inverted). "
             f"The monotonic probe confirms the mechanism: d2→d1_sb on token_count alone is "
             f"{probe_auc[('d2','d1_decontaminated_sb')]:.3f} (**not** inverted) — so the "
             f"inversion is the style component the extra features introduce, exactly the "
             f"hypothesised mechanism. In-corpus that same style component *helps* (the floor's "
             f"AUC {floor_auc['d3_en']:.3f} on d3_en far exceeds token-only "
             f"{probe_in['d3_en']:.3f}), which is precisely why it fails to transfer.")
    L.append("")
    L.append("## Confusion matrices\n")
    L.append("Per in-corpus LR run, under `figures/baselines/` (PNG + PDF): "
             "`cm_d1_decontaminated_sb`, `cm_d1_decontaminated` (text_combined ablation), "
             "`cm_d2`, `cm_d3_en`.")

    notes_path.write_text("\n".join(L), encoding="utf-8")
    return notes_path


if __name__ == "__main__":
    main()
