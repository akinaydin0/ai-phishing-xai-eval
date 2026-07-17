"""
Transformer cross-corpus generalisation (Task 5).

Per source corpus, the champion transformer is the model (DistilBERT / BERT-base /
RoBERTa-base) with the highest VALIDATION ROC-AUC on that source — chosen by val only,
never by test. Each champion is the already-fine-tuned finalist checkpoint (trained on the
source TRAIN split, selected on the source VAL split by `transformers_ft`); it is applied,
unchanged, to every other corpus's TEST split (6 directed cross pairs). No target-corpus
data is touched before this evaluation, and tokenisation/truncation are identical to
training (each champion's own tokenizer, max_length=256, truncation on).

The diagonal (in-corpus) cells reuse each champion's single official test evaluation from
its finalist json — the test split is not re-touched.

Outputs:
  results/results.csv                                6 cross rows (features='finetune', xcorpus_transformer)
  results/transformer_champions.json                 chosen champion per source (by val AUC)
  figures/models/transformer_cross_auc_heatmap.*     3×3 train × test ROC-AUC, PNG + PDF
  figures/models/transformer_cross_f1_heatmap.*      3×3 train × test F1, PNG + PDF

Run: python -m src.experiments.transformer_cross
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForSequenceClassification, AutoTokenizer,
)

try:
    from src.experiments import reslog, transformers_ft as T
    from src.splits import TEXT_FIELD, load_splits
except ImportError:  # pragma: no cover
    from experiments import reslog, transformers_ft as T
    from splits import TEXT_FIELD, load_splits

SEED = 42
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = _PROJECT_ROOT / "results" / "transformers"
FIG_DIR = _PROJECT_ROOT / "figures" / "models"
# Heatmap axis order matches the classical champion heatmaps for side-by-side comparison.
CORPORA = ["d1_decontaminated_sb", "d2", "d3_en"]
SHORT = {"d1_decontaminated_sb": "d1_sb", "d2": "d2", "d3_en": "d3_en"}
MODELS = T.MODELS
SUFFIX = "xcorpus_transformer"           # run_id suffix (canonical); +<N> for max_length N
RESULTS_CSV = None                       # None → canonical results.csv
CHAMPIONS_JSON = _PROJECT_ROOT / "results" / "transformer_champions.json"


def configure(max_length: int, corpora):
    """Redirect the finalist source, results log, champions file and run_id suffix into the
    isolated max_length namespace (canonical when 256). Also configures transformers_ft so
    T.encode tokenises the cross targets at max_length (leakage Surface 4: train/eval matched)."""
    global OUT_DIR, CORPORA, SUFFIX, RESULTS_CSV, CHAMPIONS_JSON
    CORPORA = corpora
    T.configure(max_length)
    if max_length == 256:  # reset to canonical (idempotent / reversible within a process)
        OUT_DIR = _PROJECT_ROOT / "results" / "transformers"
        SUFFIX = "xcorpus_transformer"
        RESULTS_CSV = None
        CHAMPIONS_JSON = _PROJECT_ROOT / "results" / "transformer_champions.json"
    else:
        OUT_DIR = _PROJECT_ROOT / "results" / f"transformers_{max_length}"
        SUFFIX = f"xcorpus_transformer{max_length}"
        RESULTS_CSV = _PROJECT_ROOT / "results" / f"results_{max_length}.csv"
        CHAMPIONS_JSON = _PROJECT_ROOT / "results" / f"transformer_champions_{max_length}.json"


def _finalist(model_name, corpus):
    p = OUT_DIR / f"{T.safe(model_name)}__{corpus}.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing — run `python -m src.experiments.transformers_ft` first.")
    return json.loads(p.read_text(encoding="utf-8"))


def _champion(source):
    """Model with the best VALIDATION AUC on `source`.

    Ties (val AUC equal at 4 d.p. — the corpora saturate near 1.0) are broken toward the
    smaller/cheaper model: MODELS is ordered distilbert < bert < roberta and the strict `>`
    keeps the first encountered on a tie. Deterministic and defensible (same accuracy, less
    compute). Selection uses validation AUC only — the test split plays no part.
    """
    best = None
    for m in MODELS:
        fj = _finalist(m, source)
        if best is None or fj["best_val_auc"] > best["best_val_auc"]:
            best = fj
    return best  # finalist json of the winning model


def _predict_on(ckpt_dir, texts, labels):
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir).to(T.DEVICE)
    ds = T.EncodedDS(T.encode(tokenizer, texts), labels)
    loader = T._loader(ds, tokenizer, T.EVAL_BATCH, shuffle=False)
    y, prob, _ = T.evaluate(model, loader)
    del model
    import gc
    gc.collect()
    if T.DEVICE == "cuda":
        import torch
        torch.cuda.empty_cache()
    return T._metrics(y, prob)


def _row(display, src, tgt, n_train, n_test, auc, f1, notes):
    return {"run_id": f"{display}__finetune__{src}__{tgt}__{SUFFIX}",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model": display, "features": "finetune", "text_field": TEXT_FIELD[src],
            "train_corpus": src, "test_corpus": tgt, "n_train": n_train, "n_test": n_test,
            "accuracy": "", "precision": "", "recall": "", "f1": round(f1, 4),
            "roc_auc": round(auc, 4), "runtime_s": "", "seed": SEED, "notes": notes}


def _heatmap(M, labels, metric, stem, ns):
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(M, cmap="viridis", vmin=max(0.3, float(np.nanmin(M)) - 0.05), vmax=1.0)
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels([SHORT[c] for c in CORPORA])
    ax.set_yticklabels([f"{SHORT[c]}\n({labels[c]})" for c in CORPORA], fontsize=8)
    ax.set_xlabel("test corpus"); ax.set_ylabel("train corpus (champion)")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=10,
                    color="white" if M[i, j] < 0.75 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"Transformer champion cross-corpus {metric} (train × test)\n"
                 f"d1_sb={ns['d1_decontaminated_sb']:,} / d2={ns['d2']:,} / d3_en={ns['d3_en']:,}; "
                 f"diagonal = in-corpus", fontsize=10)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Transformer cross-corpus.")
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--corpora", nargs="*", default=None)
    args = ap.parse_args(argv)
    configure(args.max_length, args.corpora or CORPORA)
    if args.max_length != 256:
        print(f"[max_length={args.max_length}] namespace {OUT_DIR.name}/, CSV={RESULTS_CSV.name}, "
              f"suffix={SUFFIX}, corpora={CORPORA}", flush=True)

    n = len(CORPORA)
    parts = {c: load_splits(c) for c in CORPORA}
    ns = {c: len(parts[c]["train"]) + len(parts[c]["val"]) + len(parts[c]["test"]) for c in CORPORA}

    auc = np.zeros((n, n)); f1 = np.zeros((n, n))
    labels, champions, rows = {}, {}, []

    for si, source in enumerate(CORPORA):
        champ = _champion(source)
        display = champ["display"]
        ckpt = _PROJECT_ROOT / champ["checkpoint"]
        labels[source] = display
        champions[SHORT[source]] = {"model": display, "val_auc": champ["best_val_auc"],
                                    "selected_lr": champ["selected_lr"],
                                    "checkpoint": champ["checkpoint"]}
        print(f"  {SHORT[source]:6s} champion = {display} (val_auc={champ['best_val_auc']:.4f})", flush=True)
        for ti, tgt in enumerate(CORPORA):
            if tgt == source:
                # reuse the champion's single official in-corpus test eval (no re-touch)
                a, f = champ["test"]["roc_auc"], champ["test"]["f1"]
            else:
                m = _predict_on(ckpt, parts[tgt]["test"]["text"], parts[tgt]["test"]["label"].to_numpy())
                a, f = m["roc_auc"], m["f1"]
                rows.append(_row(display, source, tgt, len(parts[source]["train"]),
                                 len(parts[tgt]["test"]), a, f,
                                 f"cross-corpus transformer champion ({display}, source val_auc={champ['best_val_auc']:.4f}, max_length={args.max_length})"))
                print(f"      -> {SHORT[tgt]:6s} test auc={a:.4f} f1={f:.4f}", flush=True)
            auc[si, ti], f1[si, ti] = a, f

    reslog.upsert(rows, csv_path=RESULTS_CSV)
    if n == 3:  # heatmap is the canonical 3-corpus deliverable; the 512 ablation uses grouped bars
        _heatmap(auc, labels, "ROC-AUC", "transformer_cross_auc_heatmap", ns)
        _heatmap(f1, labels, "F1", "transformer_cross_f1_heatmap", ns)
    CHAMPIONS_JSON.write_text(json.dumps(champions, indent=2), encoding="utf-8")
    print(f"\n{len(rows)} cross rows upserted -> {(RESULTS_CSV or reslog.RESULTS_CSV).name}; "
          f"champions -> {CHAMPIONS_JSON.name}" + ("; heatmaps written" if n == 3 else " (no heatmap, n!=3)"))


if __name__ == "__main__":
    main()
