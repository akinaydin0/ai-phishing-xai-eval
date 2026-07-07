"""
Champion cross-corpus generalisation (Task 4).

Per source corpus, pick the champion of each family group by validation AUC:
  * tfidf     — best of {lr, nb, svc, dt}   (single TF-IDF models)
  * ensemble  — best of {rf, xgb, lgbm}
  * dense     — best of {w2v, glove} × {LR, MLP}

Each champion is retrained on its SOURCE train split and evaluated on every target
TEST split (all 6 directed cross pairs + the 3 in-corpus diagonals for the heatmaps).
For TF-IDF/ensemble the source-fit vectorizer is applied to the target; for dense the
source-trained embedding (or shared GloVe) is applied to the target. Everything runs on
CPU for determinism (seed 42).

Outputs:
  results/results.csv                         6 cross rows per group (18 total)
  figures/models/cross_corpus_auc_heatmap.*   3 panels (train × test), PNG + PDF
  figures/models/cross_corpus_f1_heatmap.*
  results/champions.json                      chosen champion per (group, source)

Run: python -m src.experiments.cross_corpus --all
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score, roc_auc_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

try:
    from src.experiments import dense, hpo, reslog
    from src.splits import TEXT_FIELD, load_splits
except ImportError:  # pragma: no cover
    from experiments import dense, hpo, reslog
    from splits import TEXT_FIELD, load_splits

SEED = 42
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
HPO_DIR = _PROJECT_ROOT / "results" / "hpo"
FIG_DIR = _PROJECT_ROOT / "figures" / "models"
CORPORA = ["d1_decontaminated_sb", "d2", "d3_en"]
SHORT = {"d1_decontaminated_sb": "d1_sb", "d2": "d2", "d3_en": "d3_en"}
GROUPS = {"tfidf": ["lr", "nb", "svc", "dt"], "ensemble": ["rf", "xgb", "lgbm"]}


def _row(model, features, group, src, tgt, n_train, n_test, auc, f1, notes):
    return {"run_id": f"{model}__{features}__{src}__{tgt}__xcorpus_{group}",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model": model, "features": features, "text_field": TEXT_FIELD[src],
            "train_corpus": src, "test_corpus": tgt, "n_train": n_train, "n_test": n_test,
            "accuracy": "", "precision": "", "recall": "", "f1": round(f1, 4),
            "roc_auc": round(auc, 4), "runtime_s": "", "seed": SEED, "notes": notes}


# --------------------------------------------------------------------------- #
# Champions from the HPO studies (tfidf / ensemble)
# --------------------------------------------------------------------------- #
def _champion(group, source):
    best = None
    for fam in GROUPS[group]:
        pl = json.loads((HPO_DIR / f"{source}_{fam}.json").read_text(encoding="utf-8"))
        if best is None or pl["best_val_auc"] > best["best_val_auc"]:
            best = pl
    return best  # payload of the winning family


def _eval_tfidf_champion(source, payload, parts_by_corpus):
    family = payload["family"]
    bp = payload["best_params"]
    mp = {k: v for k, v in bp.items() if k not in hpo._VEC_KEYS}
    src = parts_by_corpus[source]
    vec = hpo._build_vec(hpo._vec_from_best(family, bp))
    Xtr = vec.fit_transform(src["train"]["text"])
    Xva = vec.transform(src["val"]["text"])
    ytr, yva = src["train"]["label"].to_numpy(), src["val"]["label"].to_numpy()
    model = hpo.make_model(family, mp, device="cpu")
    hpo._fit(family, model, Xtr, ytr, Xva, yva, early_stop=family in hpo.BOOST)
    out = {}
    for tgt in CORPORA:
        te = parts_by_corpus[tgt]["test"]
        Xte, yte = vec.transform(te["text"]), te["label"].to_numpy()
        s = hpo._scores(model, Xte)
        out[tgt] = (roc_auc_score(yte, s), f1_score(yte, model.predict(Xte), pos_label=1, zero_division=0))
    return family, out


# --------------------------------------------------------------------------- #
# Dense champion (recomputed: train embedding on source, pick best config by val AUC)
# --------------------------------------------------------------------------- #
def _dense_configs(source, parts, glove):
    src = parts[source]
    lookups = {"w2v": dense.train_w2v(src["train"]["text"])}
    if glove is not None:
        lookups["glove"] = glove
    ytr, yva = src["train"]["label"].to_numpy(), src["val"]["label"].to_numpy()
    configs = {}
    for ename, lookup in lookups.items():
        Xtr = dense.doc_vectors(src["train"]["text"], lookup)
        Xva = dense.doc_vectors(src["val"]["text"], lookup)
        scaler = StandardScaler().fit(Xtr)
        Xtr_s, Xva_s = scaler.transform(Xtr), scaler.transform(Xva)
        lr = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED).fit(Xtr_s, ytr)
        mlp = dense.mlp_early_stop(Xtr_s, ytr, Xva_s, yva)
        for cname, clf in (("LogisticRegression", lr), ("MLP", mlp)):
            val_auc = roc_auc_score(yva, clf.predict_proba(Xva_s)[:, 1])
            configs[(ename, cname)] = {"val_auc": val_auc, "lookup": lookup,
                                       "scaler": scaler, "clf": clf}
    return configs


def _eval_dense_champion(source, parts, glove):
    configs = _dense_configs(source, parts, glove)
    (ename, cname), best = max(configs.items(), key=lambda kv: kv[1]["val_auc"])
    lookup, scaler, clf = best["lookup"], best["scaler"], best["clf"]
    out = {}
    for tgt in CORPORA:
        te = parts[tgt]["test"]
        Xte = scaler.transform(dense.doc_vectors(te["text"], lookup))
        yte = te["label"].to_numpy()
        out[tgt] = (roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]),
                    f1_score(yte, clf.predict(Xte), pos_label=1, zero_division=0))
    return f"{ename}+{cname}", out


# --------------------------------------------------------------------------- #
# Heatmaps
# --------------------------------------------------------------------------- #
def _heatmap(mats, labels, metric, stem):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax, group in zip(axes, ["tfidf", "ensemble", "dense"]):
        M = mats[group]
        im = ax.imshow(M, cmap="viridis", vmin=max(0.3, np.nanmin(M) - 0.05), vmax=1.0)
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels([SHORT[c] for c in CORPORA])
        ax.set_yticklabels([f"{SHORT[c]}\n({labels[group][c]})" for c in CORPORA], fontsize=8)
        ax.set_xlabel("test corpus"); ax.set_ylabel("train corpus")
        ax.set_title(f"{group} champion")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=9,
                        color="white" if M[i, j] < 0.75 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Champion cross-corpus {metric} (train × test) — d1_sb=51,839 / d2=29,224 / "
                 f"d3_en=19,164; diagonal = in-corpus", fontsize=11)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)


def main(argv=None):
    argparse.ArgumentParser(description="Champion cross-corpus.").parse_args(argv)
    parts = {c: load_splits(c) for c in CORPORA}
    glove = dense.load_glove()
    print(f"GloVe: {'loaded' if glove else 'absent (dense = W2V only)'}")

    auc = {g: np.zeros((3, 3)) for g in ["tfidf", "ensemble", "dense"]}
    f1 = {g: np.zeros((3, 3)) for g in ["tfidf", "ensemble", "dense"]}
    labels = {g: {} for g in ["tfidf", "ensemble", "dense"]}
    champions, rows = {}, []

    for si, source in enumerate(CORPORA):
        # tfidf + ensemble champions
        for group in ("tfidf", "ensemble"):
            pl = _champion(group, source)
            fam, out = _eval_tfidf_champion(source, pl, parts)
            labels[group][source] = hpo.FAMILY_MODEL_NAME[fam]
            champions[f"{group}|{SHORT[source]}"] = {
                "family": fam, "val_auc": pl["best_val_auc"]}
            for ti, tgt in enumerate(CORPORA):
                a, f = out[tgt]
                auc[group][si, ti], f1[group][si, ti] = a, f
                if source != tgt:
                    rows.append(_row(hpo.FAMILY_MODEL_NAME[fam], "tfidf", group, source, tgt,
                                     len(parts[source]["train"]), len(parts[tgt]["test"]), a, f,
                                     f"cross-corpus {group} champion (val_auc={pl['best_val_auc']})"))
            print(f"  {SHORT[source]:6s} {group:9s} champion={hpo.FAMILY_MODEL_NAME[fam]}")
        # dense champion
        name, out = _eval_dense_champion(source, parts, glove)
        labels["dense"][source] = name
        champions[f"dense|{SHORT[source]}"] = {"config": name}
        for ti, tgt in enumerate(CORPORA):
            a, f = out[tgt]
            auc["dense"][si, ti], f1["dense"][si, ti] = a, f
            model_name = name.split("+")[1]
            if source != tgt:
                rows.append(_row(model_name, name.split("+")[0], "dense", source, tgt,
                                 len(parts[source]["train"]), len(parts[tgt]["test"]), a, f,
                                 f"cross-corpus dense champion ({name})"))
        print(f"  {SHORT[source]:6s} dense     champion={name}")

    reslog.upsert(rows)
    _heatmap(auc, labels, "ROC-AUC", "cross_corpus_auc_heatmap")
    _heatmap(f1, labels, "F1", "cross_corpus_f1_heatmap")
    (_PROJECT_ROOT / "results" / "champions.json").write_text(
        json.dumps(champions, indent=2), encoding="utf-8")
    print(f"\n{len(rows)} cross-corpus rows upserted; heatmaps + champions.json written")


if __name__ == "__main__":
    main()
