"""
Hyperparameter optimisation for the classical + ensemble grid (supervisor item #3).

One Optuna study per (model_family, corpus). Objective = **validation ROC-AUC**
(tie-broken by validation F1 via a +1e-6·F1 nudge). The TEST split is never touched
during tuning; the val split is the only optimisation surface.

Representation-appropriate feature spaces (probe-driven, see results/probe_timings.json):
  * Linear families (lr, nb, svc): 100k-dim TF-IDF; lr/nb also tune ngram/min_df/sublinear.
  * Tree/boosting families (dt, rf, xgb, lgbm): tune TF-IDF max_features in
    {10k, 20k, 50k} — histogram methods scale with feature count and cannot exploit
    ultra-high-dim sparse text, so the dimension is tuned, not assumed.

Boosting: n_estimators is NOT searched — fixed at 800 with early_stopping_rounds=50 on
validation AUC (the standard, leakage-safe way to set the number of trees).

Compute / determinism:
  * The XGBoost SEARCH runs on the GPU (device='cuda', ~106x faster — probe_timings.json).
  * Every FINALIST retrain and the single TEST evaluation run on CPU (device='cpu',
    seed 42), so all numbers written to results/results.csv are deterministic.

Outputs:
  results/hpo/<corpus>_<family>.json   per-study trials + best params + best val scores
  results/results.csv                  one tuned + one default row per (family, corpus)

Run: python -m src.experiments.hpo --all
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

try:
    from src.experiments import reslog
    from src.splits import TEXT_FIELD, load_splits
except ImportError:  # pragma: no cover
    from experiments import reslog
    from splits import TEXT_FIELD, load_splits

optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
HPO_DIR = _PROJECT_ROOT / "results" / "hpo"

CORPORA = ["d1_decontaminated_sb", "d2", "d3_en"]
FAMILIES = ["lr", "nb", "svc", "dt", "rf", "xgb", "lgbm"]
LINEAR = {"lr", "nb", "svc"}
TREE = {"dt", "rf", "xgb", "lgbm"}
BOOST = {"xgb", "lgbm"}
FAMILY_MODEL_NAME = {
    "lr": "LogisticRegression", "nb": "MultinomialNB", "svc": "LinearSVC",
    "dt": "DecisionTree", "rf": "RandomForest", "xgb": "XGBoost", "lgbm": "LightGBM",
}
BUDGETS = {"lr": 25, "nb": 25, "svc": 25, "dt": 30, "rf": 30, "xgb": 40, "lgbm": 40}
N_EST_BOOST = 800
EARLY_STOP = 50

_NGRAM = {"1_1": (1, 1), "1_2": (1, 2)}
_RF_MAXFEAT = {"sqrt": "sqrt", "log2": "log2", "0.3": 0.3}
TREE_MAXFEAT = [10_000, 20_000, 50_000]
# XGBoost searches on the GPU; 50k features × depth-10 × 800 rounds exceeds the
# RTX 3090's 24 GB, so its tunable max_features is capped to {10k, 20k}. The
# CPU-trained tree families (dt/rf/lgbm) keep the full {10k, 20k, 50k}.
XGB_MAXFEAT = [10_000, 20_000]
DEFAULT_LINEAR_VEC = {"ngram": "1_2", "min_df": 2, "sublinear": True, "max_features": 100_000}
DEFAULT_TREE_VEC = {"ngram": "1_2", "min_df": 2, "sublinear": True, "max_features": 20_000}
_VEC_KEYS = {"ngram", "min_df", "sublinear", "tfidf_max_features"}


# --------------------------------------------------------------------------- #
# Vectorizer memoisation (per corpus), keyed incl. max_features
# --------------------------------------------------------------------------- #
def _get_X(vp: dict, texts_tr, texts_va, cache: dict):
    key = (vp["ngram"], vp["min_df"], vp["sublinear"], vp["max_features"])
    if key not in cache:
        vec = TfidfVectorizer(lowercase=True, ngram_range=_NGRAM[vp["ngram"]],
                              min_df=vp["min_df"], max_features=vp["max_features"],
                              sublinear_tf=vp["sublinear"])
        Xtr = vec.fit_transform(texts_tr)
        Xva = vec.transform(texts_va)
        cache[key] = (vec, Xtr, Xva)
    return cache[key]


# --------------------------------------------------------------------------- #
# Search spaces
# --------------------------------------------------------------------------- #
def _suggest_vec_linear(trial):
    return {"ngram": trial.suggest_categorical("ngram", ["1_1", "1_2"]),
            "min_df": trial.suggest_categorical("min_df", [1, 2, 5]),
            "sublinear": trial.suggest_categorical("sublinear", [True, False]),
            "max_features": 100_000}


def _tree_vec(trial, choices=TREE_MAXFEAT):
    return {"ngram": "1_2", "min_df": 2, "sublinear": True,
            "max_features": trial.suggest_categorical("tfidf_max_features", choices)}


def _suggest(family: str, trial):
    """Return (model_params, vec_params)."""
    if family == "lr":
        return {"C": trial.suggest_float("C", 1e-3, 100, log=True)}, _suggest_vec_linear(trial)
    if family == "nb":
        return {"alpha": trial.suggest_float("alpha", 1e-3, 10, log=True)}, _suggest_vec_linear(trial)
    if family == "svc":
        return {"C": trial.suggest_float("C", 1e-4, 10, log=True)}, dict(DEFAULT_LINEAR_VEC)
    if family == "dt":
        return {"max_depth": trial.suggest_int("max_depth", 5, 60),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50)}, _tree_vec(trial)
    if family == "rf":
        return {"n_estimators": trial.suggest_int("n_estimators", 200, 800),
                "max_depth": trial.suggest_categorical("max_depth", [10, 20, 30, 40, 50, 60, 70, 80, None]),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", "0.3"])}, _tree_vec(trial)
    if family == "xgb":  # n_estimators fixed + early stopping; GPU-memory-capped features
        return {"learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0)}, _tree_vec(trial, XGB_MAXFEAT)
    if family == "lgbm":
        return {"learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 31, 255),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0)}, _tree_vec(trial)
    raise ValueError(family)


# --------------------------------------------------------------------------- #
# Model factories
# --------------------------------------------------------------------------- #
def make_model(family: str, p: dict, device: str = "cpu"):
    if family == "lr":
        return LogisticRegression(C=p["C"], class_weight="balanced", max_iter=2000, random_state=SEED)
    if family == "nb":
        return MultinomialNB(alpha=p["alpha"])
    if family == "svc":
        return LinearSVC(C=p["C"], class_weight="balanced", max_iter=5000, random_state=SEED)
    if family == "dt":
        return DecisionTreeClassifier(max_depth=p["max_depth"], min_samples_leaf=p["min_samples_leaf"],
                                      class_weight="balanced", random_state=SEED)
    if family == "rf":
        return RandomForestClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"],
                                      max_features=_RF_MAXFEAT[p["max_features"]],
                                      class_weight="balanced", n_jobs=-1, random_state=SEED)
    if family == "xgb":
        return XGBClassifier(n_estimators=N_EST_BOOST, learning_rate=p["learning_rate"],
                             max_depth=p["max_depth"], subsample=p["subsample"],
                             colsample_bytree=p["colsample_bytree"], tree_method="hist",
                             device=device, early_stopping_rounds=EARLY_STOP, eval_metric="auc",
                             n_jobs=-1, random_state=SEED)
    if family == "lgbm":
        return LGBMClassifier(n_estimators=N_EST_BOOST, learning_rate=p["learning_rate"],
                              num_leaves=p["num_leaves"], feature_fraction=p["feature_fraction"],
                              bagging_fraction=p["bagging_fraction"], n_jobs=-1,
                              random_state=SEED, verbose=-1)
    raise ValueError(family)


def make_default(family: str, device: str = "cpu"):
    if family == "lr":
        return LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED)
    if family == "nb":
        return MultinomialNB()
    if family == "svc":
        return LinearSVC(class_weight="balanced", max_iter=5000, random_state=SEED)
    if family == "dt":
        return DecisionTreeClassifier(class_weight="balanced", random_state=SEED)
    if family == "rf":
        return RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=SEED)
    if family == "xgb":
        return XGBClassifier(tree_method="hist", device=device, eval_metric="logloss",
                             n_jobs=-1, random_state=SEED)
    if family == "lgbm":
        return LGBMClassifier(n_jobs=-1, random_state=SEED, verbose=-1)
    raise ValueError(family)


def _xgb_gpu_fit(mp: dict, Xtr, ytr, Xva, yva):
    """XGBoost GPU search fit via QuantileDMatrix(dense).

    XGBoost's CUDA path mishandles scipy sparse CSR (degenerate models), and a plain
    dense matrix OOMs at 50k features (27 GB). QuantileDMatrix quantizes the dense
    array to uint8 (~1.8 GB) on the GPU — correct AND memory-efficient. Returns
    (val_probabilities, best_iteration). Frees GPU/CPU memory before returning.
    """
    Xtr_d = np.ascontiguousarray(Xtr.toarray(), dtype=np.float32)
    Xva_d = np.ascontiguousarray(Xva.toarray(), dtype=np.float32)
    dtr = xgb.QuantileDMatrix(Xtr_d, label=ytr)
    dva = xgb.QuantileDMatrix(Xva_d, label=yva, ref=dtr)
    params = {"tree_method": "hist", "device": "cuda", "objective": "binary:logistic",
              "eval_metric": "auc", "max_depth": mp["max_depth"], "eta": mp["learning_rate"],
              "subsample": mp["subsample"], "colsample_bytree": mp["colsample_bytree"], "seed": SEED}
    bst = xgb.train(params, dtr, num_boost_round=N_EST_BOOST, evals=[(dva, "val")],
                    early_stopping_rounds=EARLY_STOP, verbose_eval=False)
    val_scores = bst.predict(dva)  # uses best_iteration
    best_iter = int(bst.best_iteration)
    del dtr, dva, bst, Xtr_d, Xva_d
    gc.collect()
    return val_scores, best_iter


def _fit(family, model, Xtr, ytr, Xva, yva, early_stop: bool):
    """Fit; boosting finalists/trials early-stop on the VAL split."""
    if early_stop and family == "xgb":
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    elif early_stop and family == "lgbm":
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc",
                  callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False), lgb.log_evaluation(0)])
    else:
        model.fit(Xtr, ytr)
    return model


def _best_iteration(family, model):
    if family == "xgb":
        return int(getattr(model, "best_iteration", 0) or 0)
    if family == "lgbm":
        return int(getattr(model, "best_iteration_", 0) or 0)
    return None


def _scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def _metrics(y_true, y_pred, y_score):
    return {"accuracy": round(accuracy_score(y_true, y_pred), 4),
            "precision": round(precision_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
            "recall": round(recall_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
            "f1": round(f1_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_true, y_score), 4)}


# --------------------------------------------------------------------------- #
# Study
# --------------------------------------------------------------------------- #
def run_study(family: str, corpus: str, parts: dict, cache: dict):
    tr, va = parts["train"], parts["val"]
    texts_tr, texts_va = list(tr["text"]), list(va["text"])
    ytr, yva = tr["label"].to_numpy(), va["label"].to_numpy()
    search_device = "cuda" if family == "xgb" else "cpu"

    def objective(trial):
        mp, vp = _suggest(family, trial)
        _, Xtr, Xva = _get_X(vp, texts_tr, texts_va, cache)
        if family == "xgb" and search_device == "cuda":
            val_scores, bi = _xgb_gpu_fit(mp, Xtr, ytr, Xva, yva)
            auc = roc_auc_score(yva, val_scores)
            f1 = f1_score(yva, (val_scores >= 0.5).astype(int), pos_label=1, zero_division=0)
        else:
            model = make_model(family, mp, device=search_device)
            _fit(family, model, Xtr, ytr, Xva, yva, early_stop=family in BOOST)
            auc = roc_auc_score(yva, _scores(model, Xva))
            f1 = f1_score(yva, model.predict(Xva), pos_label=1, zero_division=0)
            bi = _best_iteration(family, model)
        trial.set_user_attr("val_auc", float(auc))
        trial.set_user_attr("val_f1", float(f1))
        if bi is not None:
            trial.set_user_attr("best_iteration", bi)
        return auc + 1e-6 * f1

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    t0 = time.time()
    study.optimize(objective, n_trials=BUDGETS[family], show_progress_bar=False)
    runtime = time.time() - t0

    best = study.best_trial
    payload = {
        "corpus": corpus, "family": family, "model": FAMILY_MODEL_NAME[family],
        "objective": "val_roc_auc + 1e-6*val_f1", "n_trials": BUDGETS[family],
        "search_device": search_device, "runtime_s": round(runtime, 1),
        "best_params": best.params,
        "best_val_auc": round(best.user_attrs["val_auc"], 4),
        "best_val_f1": round(best.user_attrs["val_f1"], 4),
        "best_iteration": best.user_attrs.get("best_iteration"),
        "trials": [{"number": t.number, "params": t.params,
                    "val_auc": round(t.user_attrs.get("val_auc", float("nan")), 4),
                    "val_f1": round(t.user_attrs.get("val_f1", float("nan")), 4)}
                   for t in study.trials],
    }
    HPO_DIR.mkdir(parents=True, exist_ok=True)
    (HPO_DIR / f"{corpus}_{family}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


# --------------------------------------------------------------------------- #
# Finalisation (Task 2): CPU, deterministic. Retrain on TRAIN, evaluate on TEST once.
# --------------------------------------------------------------------------- #
def _vec_from_best(family, best_params):
    if family in ("lr", "nb"):
        return {"ngram": best_params["ngram"], "min_df": best_params["min_df"],
                "sublinear": best_params["sublinear"], "max_features": 100_000}
    if family == "svc":
        return dict(DEFAULT_LINEAR_VEC)
    return {"ngram": "1_2", "min_df": 2, "sublinear": True,
            "max_features": best_params["tfidf_max_features"]}


def _build_vec(vp):
    return TfidfVectorizer(lowercase=True, ngram_range=_NGRAM[vp["ngram"]], min_df=vp["min_df"],
                           max_features=vp["max_features"], sublinear_tf=vp["sublinear"])


def _row(family, corpus, n_train, n_test, m, runtime, notes, tag):
    return {"run_id": f"{FAMILY_MODEL_NAME[family]}__tfidf__{corpus}__{corpus}__{tag}",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model": FAMILY_MODEL_NAME[family], "features": "tfidf", "text_field": TEXT_FIELD[corpus],
            "train_corpus": corpus, "test_corpus": corpus, "n_train": n_train, "n_test": n_test,
            "accuracy": m["accuracy"], "precision": m["precision"], "recall": m["recall"],
            "f1": m["f1"], "roc_auc": m["roc_auc"], "runtime_s": round(runtime, 2),
            "seed": SEED, "notes": notes}


def finalize(family, corpus, payload, parts):
    tr, va, te = parts["train"], parts["val"], parts["test"]
    ytr, yva, yte = tr["label"].to_numpy(), va["label"].to_numpy(), te["label"].to_numpy()
    best_params = payload["best_params"]
    model_params = {k: v for k, v in best_params.items() if k not in _VEC_KEYS}

    # tuned (CPU, deterministic)
    vp = _vec_from_best(family, best_params)
    vec = _build_vec(vp)
    Xtr, Xva, Xte = vec.fit_transform(tr["text"]), vec.transform(va["text"]), vec.transform(te["text"])
    t0 = time.time()
    model = make_model(family, model_params, device="cpu")
    _fit(family, model, Xtr, ytr, Xva, yva, early_stop=family in BOOST)
    m = _metrics(yte, model.predict(Xte), _scores(model, Xte))
    bi = _best_iteration(family, model)
    note = (f"tuned n_trials={payload['n_trials']} val_auc={payload['best_val_auc']}"
            + (f" best_iter={bi}" if bi else "")
            + f" | max_features={vp['max_features']} | best={json.dumps(best_params, separators=(',', ':'))}")
    rows = [_row(family, corpus, len(tr), len(te), m, time.time() - t0, note, "tuned")]

    # default (library defaults; linear 100k vec, tree 20k vec)
    vpd = dict(DEFAULT_LINEAR_VEC) if family in LINEAR else dict(DEFAULT_TREE_VEC)
    vecd = _build_vec(vpd)
    Xtr_d, Xte_d = vecd.fit_transform(tr["text"]), vecd.transform(te["text"])
    t0 = time.time()
    dmodel = make_default(family, device="cpu")
    dmodel.fit(Xtr_d, ytr)  # defaults: no early stopping
    m = _metrics(yte, dmodel.predict(Xte_d), _scores(dmodel, Xte_d))
    rows.append(_row(family, corpus, len(tr), len(te), m, time.time() - t0,
                     f"untuned default | max_features={vpd['max_features']}", "default"))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description="HPO for classical + ensemble grid.")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--families", nargs="*", default=FAMILIES)
    parser.add_argument("--corpora", nargs="*", default=CORPORA)
    args = parser.parse_args(argv)

    # resume: skip a (corpus, family) whose study json AND both result rows already exist
    import csv as _csv
    done_ids = set()
    if reslog.RESULTS_CSV.exists():
        with open(reslog.RESULTS_CSV, encoding="utf-8", newline="") as f:
            done_ids = {r["run_id"] for r in _csv.DictReader(f)}

    n_rows = 0
    grand_t0 = time.time()
    for corpus in args.corpora:
        parts = load_splits(corpus)
        cache: dict = {}
        for family in args.families:
            tuned_id = f"{FAMILY_MODEL_NAME[family]}__tfidf__{corpus}__{corpus}__tuned"
            default_id = f"{FAMILY_MODEL_NAME[family]}__tfidf__{corpus}__{corpus}__default"
            if ((HPO_DIR / f"{corpus}_{family}.json").exists()
                    and tuned_id in done_ids and default_id in done_ids):
                print(f"[{corpus:22s} {family:5s}] SKIP (already complete)", flush=True)
                continue
            t0 = time.time()
            payload = run_study(family, corpus, parts, cache)
            rows = finalize(family, corpus, payload, parts)
            reslog.upsert(rows)  # incremental: crash-resilient, resumable
            n_rows += len(rows)
            print(f"[{corpus:22s} {family:5s}] trials={payload['n_trials']:>2} "
                  f"dev={payload['search_device']:>4} study={payload['runtime_s']:>7.1f}s "
                  f"total={time.time() - t0:>7.1f}s val_auc={payload['best_val_auc']:.4f}"
                  + (f" best_iter={payload['best_iteration']}" if payload.get("best_iteration") else ""),
                  flush=True)
        del cache
    print(f"\nTotal HPO+finalise: {(time.time() - grand_t0) / 60:.1f} min; "
          f"{n_rows} rows upserted to results/results.csv")


if __name__ == "__main__":
    main()
