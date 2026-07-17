"""
Champion models with a uniform `proba(texts) -> p(class 1)` interface for the XAI phase.

Every explainer and the faithfulness removal mechanic go through `proba`, so the pipeline is
identical across families: raw text in, class-1 probability out (the metrics derive
p(predicted class) from this). Linear/ensemble champions are rebuilt from the stored HPO
`best_params` and fit on their SOURCE train split; transformer champions load the saved
fine-tuned checkpoints. LinearSVC has no `predict_proba`, so p = sigmoid(decision_function)
(a monotone pseudo-probability — see the pre-registration in notes/decisions.md).

The 12 units (9 in-corpus champions + 3 transformer transfer) and the stratified
explanation-set sampler live here too.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from src.experiments import hpo
    from src.splits import load_splits
except ImportError:  # pragma: no cover
    from experiments import hpo
    from splits import load_splits

_ROOT = Path(__file__).resolve().parent.parent.parent
HPO_DIR = _ROOT / "results" / "hpo"
CKPT_DIR = _ROOT / "results" / "transformers" / "checkpoints"
XAI_DIR = _ROOT / "results" / "xai"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 256
SEED = 42

TRANSFORMER_DISPLAY = {"distilbert-base-uncased": "DistilBERT",
                       "bert-base-uncased": "BERT-base", "roberta-base": "RoBERTa-base"}


# --------------------------------------------------------------------------- #
# Champions
# --------------------------------------------------------------------------- #
class _SklearnChampion:
    """tfidf-linear or LightGBM champion rebuilt from HPO best_params, fit on source train."""

    def __init__(self, family: str, corpus: str):
        self.family, self.corpus = family, corpus
        payload = json.loads((HPO_DIR / f"{corpus}_{family}.json").read_text(encoding="utf-8"))
        bp = payload["best_params"]
        self.best_params = bp
        model_params = {k: v for k, v in bp.items() if k not in hpo._VEC_KEYS}
        parts = load_splits(corpus)
        tr, va = parts["train"], parts["val"]
        self.vec = hpo._build_vec(hpo._vec_from_best(family, bp))
        Xtr = self.vec.fit_transform(tr["text"])
        Xva = self.vec.transform(va["text"])
        ytr, yva = tr["label"].to_numpy(), va["label"].to_numpy()
        self.model = hpo.make_model(family, model_params, device="cpu")
        hpo._fit(family, self.model, Xtr, ytr, Xva, yva, early_stop=family in hpo.BOOST)
        self.name = hpo.FAMILY_MODEL_NAME[family]
        self.feature_names = np.asarray(self.vec.get_feature_names_out())
        self.has_proba = hasattr(self.model, "predict_proba")

    def proba(self, texts) -> np.ndarray:
        X = self.vec.transform(list(texts))
        if self.has_proba:
            return self.model.predict_proba(X)[:, 1]
        return expit(self.model.decision_function(X))  # LinearSVC: monotone pseudo-probability

    def transform(self, texts):
        return self.vec.transform(list(texts))


class TransformerChampion:
    """Fine-tuned transformer champion loaded from its saved checkpoint."""

    def __init__(self, model_name: str, corpus: str, ckpt_dir=None):
        self.model_name, self.corpus = model_name, corpus
        d = Path(ckpt_dir) if ckpt_dir is not None else CKPT_DIR / f"{model_name}__{corpus}"
        if not d.exists():
            raise FileNotFoundError(f"{d} missing — run transformers_ft first.")
        self.tokenizer = AutoTokenizer.from_pretrained(d)
        self.model = AutoModelForSequenceClassification.from_pretrained(d).to(DEVICE).eval()
        self.name = TRANSFORMER_DISPLAY[model_name]
        self.family = "transformer"
        self.max_length = MAX_LEN  # 256 default; set to 512 to score a 512-trained checkpoint

    @torch.no_grad()
    def proba(self, texts, batch_size: int = 64) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros(0, dtype=np.float32)
        out = []
        for i in range(0, len(texts), batch_size):
            chunk = [t if isinstance(t, str) else "" for t in texts[i:i + batch_size]]
            enc = self.tokenizer(chunk, truncation=True, max_length=self.max_length, padding=True,
                                 return_tensors="pt").to(DEVICE)
            with torch.autocast("cuda", dtype=torch.float16, enabled=(DEVICE == "cuda")):
                logits = self.model(**enc).logits
            out.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
        return np.concatenate(out)


# --------------------------------------------------------------------------- #
# Unit registry (12): 9 in-corpus champions + 3 transformer transfer
# --------------------------------------------------------------------------- #
# family ∈ {tfidf, ensemble, transformer}; methods per family fixed by the pre-registration.
FAMILY_METHODS = {"tfidf": ["shap", "lime"], "ensemble": ["shap", "lime"],
                  "transformer": ["shap", "lime", "ig"]}

UNIT_SPECS = [
    # unit, family, builder, explained_corpus, kind, n, mis_cap
    ("tfidf_d1_sb", "tfidf", lambda: _SklearnChampion("lr", "d1_decontaminated_sb"),
     "d1_decontaminated_sb", "in", 200, 30),
    ("tfidf_d2", "tfidf", lambda: _SklearnChampion("lr", "d2"), "d2", "in", 200, 30),
    ("tfidf_d3_en", "tfidf", lambda: _SklearnChampion("svc", "d3_en"), "d3_en", "in", 200, 30),
    ("ensemble_d1_sb", "ensemble", lambda: _SklearnChampion("lgbm", "d1_decontaminated_sb"),
     "d1_decontaminated_sb", "in", 200, 30),
    ("ensemble_d2", "ensemble", lambda: _SklearnChampion("lgbm", "d2"), "d2", "in", 200, 30),
    ("ensemble_d3_en", "ensemble", lambda: _SklearnChampion("lgbm", "d3_en"), "d3_en", "in", 200, 30),
    ("transformer_d1_sb", "transformer",
     lambda: TransformerChampion("distilbert-base-uncased", "d1_decontaminated_sb"),
     "d1_decontaminated_sb", "in", 200, 30),
    ("transformer_d2", "transformer",
     lambda: TransformerChampion("distilbert-base-uncased", "d2"), "d2", "in", 200, 30),
    ("transformer_d3_en", "transformer",
     lambda: TransformerChampion("roberta-base", "d3_en"), "d3_en", "in", 200, 30),
    # cross-corpus transfer (transformers only): champion on a DIFFERENT corpus's test
    ("xfer_d3_en__d1_sb", "transformer",
     lambda: TransformerChampion("roberta-base", "d3_en"), "d1_decontaminated_sb", "cross", 100, 15),
    ("xfer_d2__d3_en", "transformer",
     lambda: TransformerChampion("distilbert-base-uncased", "d2"), "d3_en", "cross", 100, 15),
    ("xfer_d3_en__d2", "transformer",
     lambda: TransformerChampion("roberta-base", "d3_en"), "d2", "cross", 100, 15),
]


def unit_meta(unit_id: str) -> dict:
    for u, fam, _b, exc, kind, n, cap in UNIT_SPECS:
        if u == unit_id:
            return {"unit": u, "family": fam, "explained_corpus": exc, "kind": kind,
                    "n": n, "mis_cap": cap, "methods": FAMILY_METHODS[fam]}
    raise KeyError(unit_id)


def build_champion(unit_id: str):
    for u, _fam, builder, *_ in UNIT_SPECS:
        if u == unit_id:
            return builder()
    raise KeyError(unit_id)


# --------------------------------------------------------------------------- #
# Explanation-set sampler (TEST split only) — stratified, misclassified-aware
# --------------------------------------------------------------------------- #
def build_explanation_set(champ, explained_corpus: str, n: int, mis_cap: int, seed: int = SEED) -> dict:
    """Stratified n docs (n/2 per TRUE class) from the TEST split; include all misclassified
    (by this champion) up to mis_cap, fill with correctly-classified; seed-fixed."""
    te = load_splits(explained_corpus)["test"].reset_index(drop=True)
    texts = te["text"].tolist()
    y = te["label"].to_numpy()
    hashes = te["_hash"].to_numpy()
    p1 = champ.proba(texts)
    pred = (p1 >= 0.5).astype(int)
    rng = np.random.default_rng(seed)

    mis_all = np.where(pred != y)[0]
    if len(mis_all) > mis_cap:
        mis_all = np.sort(rng.choice(mis_all, mis_cap, replace=False))
    mis_set = set(int(i) for i in mis_all)

    per_class = n // 2
    selected: list[int] = []
    for c in (0, 1):
        cls_idx = np.where(y == c)[0]
        cls_mis = [int(i) for i in cls_idx if int(i) in mis_set]
        cls_correct = [int(i) for i in cls_idx if int(i) not in mis_set]  # pred==y for these
        need = per_class - len(cls_mis)
        pick = (rng.choice(cls_correct, min(need, len(cls_correct)), replace=False).tolist()
                if need > 0 else [])
        chosen = cls_mis[:per_class] + pick
        if len(chosen) < per_class:  # tiny-corpus fallback (not expected here)
            pool = [int(i) for i in cls_idx if int(i) not in set(chosen)]
            chosen += rng.choice(pool, per_class - len(chosen), replace=False).tolist()
        selected.extend(sorted(chosen[:per_class]))

    selected = sorted(selected)
    return {"explained_corpus": explained_corpus, "n": len(selected), "seed": seed,
            "mis_cap": mis_cap, "n_misclassified": sum(1 for i in selected if i in mis_set),
            "idx": selected, "hashes": [hashes[i] for i in selected],
            "y_true": [int(y[i]) for i in selected], "pred": [int(pred[i]) for i in selected],
            "p1": [round(float(p1[i]), 6) for i in selected]}


def persist_explanation_set(unit_id: str, es: dict) -> Path:
    XAI_DIR.mkdir(parents=True, exist_ok=True)
    p = XAI_DIR / f"explained_ids_{unit_id}.json"
    p.write_text(json.dumps(es, indent=2), encoding="utf-8")
    return p


def load_or_build_explanation_set(unit_id: str, champ) -> dict:
    """Resume-safe: reuse the persisted id set (re-derive positional idx from stable hashes)."""
    m = unit_meta(unit_id)
    p = XAI_DIR / f"explained_ids_{unit_id}.json"
    if p.exists():
        es = json.loads(p.read_text(encoding="utf-8"))
        te = load_splits(m["explained_corpus"])["test"].reset_index(drop=True)
        pos = {h: i for i, h in enumerate(te["_hash"].to_numpy())}
        es["idx"] = [pos[h] for h in es["hashes"]]  # robust to any re-ordering
        return es
    es = build_explanation_set(champ, m["explained_corpus"], m["n"], m["mis_cap"])
    persist_explanation_set(unit_id, es)
    return es


def explained_texts(unit_id: str, es: dict) -> list[str]:
    m = unit_meta(unit_id)
    te = load_splits(m["explained_corpus"])["test"].reset_index(drop=True)
    return [te["text"].iloc[i] for i in es["idx"]]
