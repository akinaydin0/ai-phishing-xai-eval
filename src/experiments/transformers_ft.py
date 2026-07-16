"""
Transformer fine-tuning for phishing detection (DistilBERT, BERT-base, RoBERTa-base).

Grid per (model, corpus): learning-rate ∈ {2e-5, 3e-5, 5e-5}, up to 4 epochs, batch 32
with fp16 (gradient accumulation on OOM), weight_decay 0.01, warmup_ratio 0.1. The
VALIDATION split is scored every epoch (ROC-AUC primary); early-stop patience 1; the
best-val-AUC checkpoint is kept. The best lr per (model, corpus) is selected purely by
val AUC. The TEST split is evaluated exactly ONCE per finalist, after selection — it is
never touched during the grid.

Determinism (Task 4): seeds = 42 everywhere; cudnn.deterministic + benchmark off;
CUBLAS_WORKSPACE_CONFIG pinned before the CUDA context; torch deterministic algorithms
(warn_only, since a few transformer CUDA kernels lack deterministic impls). GPU
fine-tuning is NOT bit-exact — mitigated by fixed seeds, pinned library/CUDA versions
(recorded in each run json) and saved checkpoints. See notes/decisions.md.

Resilience: incremental, resumable per (model, corpus) exactly like the HPO run. Each
learning rate's result is persisted to a partial file as it completes; a unit is skipped
on resume once its final json AND its results.csv row both exist.

Outputs:
  results/transformers/<model>__<corpus>.json          per-lr val curves, selection, test, cost, truncation, versions
  results/transformers/checkpoints/<model>__<corpus>/  best finalist (save_pretrained) — reused by cross-corpus + XAI
  results/results.csv                                  one features='finetune' row per (model, corpus) finalist

Run: python -m src.experiments.transformers_ft --all
"""

from __future__ import annotations

# --- determinism knobs must be set BEFORE torch initialises the CUDA context --- #
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS
os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse  # noqa: E402
import csv as _csv  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForSequenceClassification, AutoTokenizer,
    get_linear_schedule_with_warmup,
)

try:
    from src.experiments import reslog
    from src.splits import TEXT_FIELD, load_splits
except ImportError:  # pragma: no cover
    from experiments import reslog
    from splits import TEXT_FIELD, load_splits

transformers.logging.set_verbosity_error()  # silence >512-token truncation chatter

SEED = 42
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = _PROJECT_ROOT / "results" / "transformers"
CKPT_DIR = OUT_DIR / "checkpoints"
PARTIAL_DIR = OUT_DIR / "_partial"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = ["distilbert-base-uncased", "bert-base-uncased", "roberta-base"]
DISPLAY = {"distilbert-base-uncased": "DistilBERT",
           "bert-base-uncased": "BERT-base",
           "roberta-base": "RoBERTa-base"}
# smallest -> largest, so early finalists arrive fast and the wall-clock projection is
# grounded in >1 model and >1 corpus size.
CORPORA = ["d3_en", "d2", "d1_decontaminated_sb"]
LRS = [2e-5, 3e-5, 5e-5]

MAX_EPOCHS = 4
PATIENCE = 1          # stop after 1 non-improving epoch (on val AUC)
BATCH = 32            # effective batch size (kept constant via grad accumulation)
EVAL_BATCH = 64
MAX_LEN = 256
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
OOM_LADDER = [(32, 1), (16, 2), (8, 4)]  # (per-device batch, grad-accum) — product = 32


def safe(model_name: str) -> str:
    return model_name.replace("/", "_")


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(_PROJECT_ROOT))
    except ValueError:  # e.g. redirected outside the repo during smoke tests
        return str(path)


def set_determinism(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:  # pragma: no cover - older torch
        pass


def versions() -> dict:
    return {"python": ".".join(map(str, __import__("sys").version_info[:3])),
            "torch": torch.__version__, "transformers": transformers.__version__,
            "cuda_build": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu",
            "device": DEVICE}


# --------------------------------------------------------------------------- #
# Tokenisation + truncation stats (Task 1)
# --------------------------------------------------------------------------- #
def encode(tokenizer, texts):
    """Truncated, un-padded input_ids (dynamic per-batch padding happens in collate)."""
    enc = tokenizer(list(texts), truncation=True, max_length=MAX_LEN,
                    add_special_tokens=True, padding=False)
    return enc["input_ids"]


def trunc_stats(tokenizer, texts, batch=512) -> dict:
    """% of documents whose *untruncated* length exceeds MAX_LEN (reported, not hidden)."""
    lengths = []
    texts = list(texts)
    for i in range(0, len(texts), batch):
        enc = tokenizer(texts[i:i + batch], add_special_tokens=True, truncation=False)
        lengths.extend(len(x) for x in enc["input_ids"])
    lengths = np.asarray(lengths)
    return {"n": int(lengths.size),
            "truncated": int((lengths > MAX_LEN).sum()),
            "truncated_pct": round(float((lengths > MAX_LEN).mean() * 100), 2),
            "median_len": int(np.median(lengths)),
            "p95_len": int(np.percentile(lengths, 95)),
            "max_len_obs": int(lengths.max())}


class EncodedDS(Dataset):
    def __init__(self, input_ids, labels):
        self.input_ids = input_ids
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {"input_ids": self.input_ids[i], "labels": int(self.labels[i])}


def _collate(batch, tokenizer):
    labels = torch.tensor([b["labels"] for b in batch], dtype=torch.long)
    padded = tokenizer.pad([{"input_ids": b["input_ids"]} for b in batch],
                           padding=True, return_tensors="pt")
    padded["labels"] = labels
    return padded


def _loader(ds, tokenizer, batch_size, shuffle):
    gen = torch.Generator()
    gen.manual_seed(SEED)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=lambda b: _collate(b, tokenizer),
                      num_workers=0, generator=gen if shuffle else None,
                      pin_memory=(DEVICE == "cuda"))


# --------------------------------------------------------------------------- #
# Metrics + evaluation
# --------------------------------------------------------------------------- #
def _metrics(y, prob) -> dict:
    pred = (prob >= 0.5).astype(int)
    return {"accuracy": round(accuracy_score(y, pred), 4),
            "precision": round(precision_score(y, pred, pos_label=1, zero_division=0), 4),
            "recall": round(recall_score(y, pred, pos_label=1, zero_division=0), 4),
            "f1": round(f1_score(y, pred, pos_label=1, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y, prob), 4)}


@torch.no_grad()
def evaluate(model, loader) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    lossfn = torch.nn.CrossEntropyLoss(reduction="sum")
    probs, ys, tot_loss, n = [], [], 0.0, 0
    for batch in loader:
        labels = batch.pop("labels").to(DEVICE)
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.float16, enabled=(DEVICE == "cuda")):
            logits = model(**batch).logits
        tot_loss += lossfn(logits.float(), labels).item()
        n += labels.size(0)
        probs.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu())
        ys.append(labels.cpu())
    return torch.cat(ys).numpy(), torch.cat(probs).numpy(), tot_loss / max(n, 1)


# --------------------------------------------------------------------------- #
# Train one (model, corpus, lr): returns best-val checkpoint state + val curve
# --------------------------------------------------------------------------- #
def _train_lr_impl(model_name, tokenizer, enc, lr, per_dev, accum):
    set_determinism()  # reproducible head init + shuffling; lr is the only difference
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2).to(DEVICE)

    train_loader = _loader(EncodedDS(enc["train"][0], enc["train"][1]),
                           tokenizer, per_dev, shuffle=True)
    val_loader = _loader(EncodedDS(enc["val"][0], enc["val"][1]),
                         tokenizer, EVAL_BATCH, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = math.ceil(len(enc["train"][1]) / BATCH)
    total_steps = steps_per_epoch * MAX_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(WARMUP_RATIO * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))

    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()

    best = {"val_auc": -1.0, "state": None, "epoch": -1}
    curves, wait, epochs_run = [], 0, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader):
            labels = batch.pop("labels").to(DEVICE)
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=(DEVICE == "cuda")):
                out = model(**batch, labels=labels)
                loss = out.loss / accum
            scaler.scale(loss).backward()
            if (step + 1) % accum == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        epochs_run = epoch + 1

        y, prob, vloss = evaluate(model, val_loader)
        m = _metrics(y, prob)
        curves.append({"epoch": epoch, "val_loss": round(vloss, 4), **{f"val_{k}": v for k, v in m.items()}})

        if m["roc_auc"] > best["val_auc"] + 1e-5:
            best = {"val_auc": m["roc_auc"],
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    "epoch": epoch}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    peak_mb = (torch.cuda.max_memory_allocated() / 1e6) if DEVICE == "cuda" else 0.0
    del model, optimizer, scaler, train_loader, val_loader
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return best, curves, epochs_run, round(peak_mb, 1)


def train_lr(model_name, tokenizer, enc, lr):
    """OOM-resilient wrapper: keep effective batch 32 via grad accumulation."""
    last_err = None
    for per_dev, accum in OOM_LADDER:
        try:
            t0 = time.time()
            best, curves, epochs_run, peak = _train_lr_impl(
                model_name, tokenizer, enc, lr, per_dev, accum)
            return best, curves, epochs_run, peak, (per_dev, accum), round(time.time() - t0, 1)
        except torch.cuda.OutOfMemoryError as e:  # pragma: no cover - hardware dependent
            last_err = e
            gc.collect()
            torch.cuda.empty_cache()
            print(f"      OOM at per_dev={per_dev} accum={accum}; retrying smaller", flush=True)
    raise last_err


# --------------------------------------------------------------------------- #
# Checkpoint persistence (best finalist, reused by cross-corpus + XAI)
# --------------------------------------------------------------------------- #
def save_finalist(model_name, corpus, state, tokenizer):
    d = CKPT_DIR / f"{safe(model_name)}__{corpus}"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.load_state_dict(state)
    model.save_pretrained(d, safe_serialization=True)
    tokenizer.save_pretrained(d)
    del model
    gc.collect()
    return d


# --------------------------------------------------------------------------- #
# Results row + partial/final persistence
# --------------------------------------------------------------------------- #
def _row(display, corpus, n_train, n_test, m, finalist_wall_s, notes):
    return {"run_id": f"{display}__finetune__{corpus}__{corpus}__finetune",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model": display, "features": "finetune", "text_field": TEXT_FIELD[corpus],
            "train_corpus": corpus, "test_corpus": corpus, "n_train": n_train, "n_test": n_test,
            "accuracy": m["accuracy"], "precision": m["precision"], "recall": m["recall"],
            "f1": m["f1"], "roc_auc": m["roc_auc"], "runtime_s": round(finalist_wall_s, 2),
            "seed": SEED, "notes": notes}


def _partial_path(model_name, corpus):
    return PARTIAL_DIR / f"{safe(model_name)}__{corpus}.json"


def _final_path(model_name, corpus):
    return OUT_DIR / f"{safe(model_name)}__{corpus}.json"


def _load_partial(model_name, corpus):
    p = _partial_path(model_name, corpus)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"model": model_name, "display": DISPLAY[model_name], "corpus": corpus,
            "lrs": {}, "best_lr": None, "best_val_auc": -1.0}


def _write_partial(model_name, corpus, partial):
    PARTIAL_DIR.mkdir(parents=True, exist_ok=True)
    _partial_path(model_name, corpus).write_text(json.dumps(partial, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Unit runner: all LRs for one (model, corpus) -> finalist test eval
# --------------------------------------------------------------------------- #
def run_unit(model_name, corpus, lrs=LRS):
    parts = load_splits(corpus)
    display = DISPLAY[model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    trunc = {sp: trunc_stats(tokenizer, parts[sp]["text"]) for sp in ("train", "val", "test")}
    enc = {sp: (encode(tokenizer, parts[sp]["text"]), parts[sp]["label"].to_numpy())
           for sp in ("train", "val", "test")}

    partial = _load_partial(model_name, corpus)
    partial["trunc"] = trunc
    partial["versions"] = versions()

    for lr in lrs:
        key = repr(lr)
        if key in partial["lrs"]:
            print(f"    lr={lr:>6} SKIP (done, val_auc={partial['lrs'][key]['val_auc']:.4f})", flush=True)
            continue
        best, curves, epochs_run, peak, (per_dev, accum), wall = train_lr(
            model_name, tokenizer, enc, lr)
        is_best = best["val_auc"] > partial["best_val_auc"]
        if is_best:
            save_finalist(model_name, corpus, best["state"], tokenizer)  # ckpt first
            partial["best_val_auc"] = best["val_auc"]
            partial["best_lr"] = lr
        del best  # free the CPU state dict
        gc.collect()
        vaucs = [c["val_roc_auc"] for c in curves]
        best_ep = int(np.argmax(vaucs))
        partial["lrs"][key] = {"lr": lr, "val_auc": max(vaucs), "best_epoch": curves[best_ep]["epoch"],
                               "epochs_run": epochs_run, "curves": curves,
                               "peak_mem_mb": peak, "batch": BATCH, "per_device_batch": per_dev,
                               "grad_accum": accum, "wall_s": wall}
        _write_partial(model_name, corpus, partial)
        print(f"    lr={lr:>6} epochs={epochs_run} best_ep={partial['lrs'][key]['best_epoch']} "
              f"val_auc={partial['lrs'][key]['val_auc']:.4f} peak={peak:.0f}MB wall={wall:.0f}s"
              f"{'  <== best' if is_best else ''}", flush=True)

    return _finalize_unit(model_name, corpus, parts, enc, partial)


def _finalize_unit(model_name, corpus, parts, enc, partial):
    display = DISPLAY[model_name]
    ckpt = CKPT_DIR / f"{safe(model_name)}__{corpus}"
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt).to(DEVICE)
    test_loader = _loader(EncodedDS(enc["test"][0], enc["test"][1]), tokenizer, EVAL_BATCH, shuffle=False)
    y, prob, tloss = evaluate(model, test_loader)          # <-- the single TEST eval
    m = _metrics(y, prob)
    del model
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    best_lr = partial["best_lr"]
    lr_rec = partial["lrs"][repr(best_lr)]
    unit_wall = round(sum(r["wall_s"] for r in partial["lrs"].values()), 1)
    peak_mb = round(max(r["peak_mem_mb"] for r in partial["lrs"].values()), 1)
    tr, va, te = (partial["trunc"]["train"], partial["trunc"]["val"], partial["trunc"]["test"])

    notes = (f"lr={best_lr} epochs_run={lr_rec['epochs_run']}/{MAX_EPOCHS} "
             f"best_epoch={lr_rec['best_epoch']} val_auc={partial['best_val_auc']:.4f} | "
             f"trunc%(tr/va/te)={tr['truncated_pct']}/{va['truncated_pct']}/{te['truncated_pct']} | "
             f"batch={BATCH} fp16 wd={WEIGHT_DECAY} warmup={WARMUP_RATIO} | "
             f"peak_gpu={peak_mb:.0f}MB finalist_wall={lr_rec['wall_s']:.0f}s unit_wall={unit_wall:.0f}s | "
             f"grid_lr{{2e-5,3e-5,5e-5}} selected_by_val_auc")

    final = {"model": model_name, "display": display, "corpus": corpus,
             "text_field": TEXT_FIELD[corpus], "n_train": len(parts["train"]),
             "n_val": len(parts["val"]), "n_test": len(parts["test"]),
             "grid_lrs": LRS, "selected_lr": best_lr, "best_val_auc": partial["best_val_auc"],
             "selection": "val_roc_auc", "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
             "test": {**m, "test_loss": round(tloss, 4)},
             "lr_results": partial["lrs"], "truncation": partial["trunc"],
             "cost": {"finalist_wall_s": lr_rec["wall_s"], "unit_wall_s": unit_wall,
                      "peak_gpu_mem_mb": peak_mb},
             "versions": partial["versions"], "checkpoint": _rel(ckpt)}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _final_path(model_name, corpus).write_text(json.dumps(final, indent=2), encoding="utf-8")

    reslog.upsert([_row(display, corpus, len(parts["train"]), len(parts["test"]),
                        m, lr_rec["wall_s"], notes)])
    _partial_path(model_name, corpus).unlink(missing_ok=True)
    print(f"  [{display} / {corpus}] FINALIST lr={best_lr} val_auc={partial['best_val_auc']:.4f} "
          f"-> TEST auc={m['roc_auc']:.4f} f1={m['f1']:.4f} | unit_wall={unit_wall:.0f}s peak={peak_mb:.0f}MB",
          flush=True)
    return final


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None):
    parser = argparse.ArgumentParser(description="Transformer fine-tuning grid.")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--models", nargs="*", default=MODELS)
    parser.add_argument("--corpora", nargs="*", default=CORPORA)
    parser.add_argument("--lrs", nargs="*", type=float, default=LRS)
    args = parser.parse_args(argv)

    if DEVICE != "cuda":
        print("WARNING: CUDA not available — running on CPU (fp16 disabled, very slow).", flush=True)

    done_ids = set()
    if reslog.RESULTS_CSV.exists():
        with open(reslog.RESULTS_CSV, encoding="utf-8", newline="") as f:
            done_ids = {r["run_id"] for r in _csv.DictReader(f)}

    total_units = len(args.models) * len(args.corpora)
    already_done = sum(
        1 for corpus in args.corpora for model_name in args.models
        if _final_path(model_name, corpus).exists()
        and f"{DISPLAY[model_name]}__finetune__{corpus}__{corpus}__finetune" in done_ids)

    grand_t0 = time.time()
    completed = []
    for corpus in args.corpora:
        for model_name in args.models:
            display = DISPLAY[model_name]
            run_id = f"{display}__finetune__{corpus}__{corpus}__finetune"
            if _final_path(model_name, corpus).exists() and run_id in done_ids:
                print(f"[{display} / {corpus}] SKIP (already complete)", flush=True)
                continue
            print(f"[{display} / {corpus}] starting {len(args.lrs)}-lr grid "
                  f"(train n={len(load_splits(corpus)['train'])})", flush=True)
            t0 = time.time()
            run_unit(model_name, corpus, lrs=args.lrs)
            completed.append((display, corpus, time.time() - t0))
            # Wall-clock projection after the first two finalists of THIS run.
            if len(completed) == 2:
                per_unit = sum(c[2] for c in completed) / 2
                remaining = total_units - already_done - len(completed)
                print(f"\n=== PROJECTION after 2 finalists ({completed[0][0]}/{completed[0][1]}, "
                      f"{completed[1][0]}/{completed[1][1]}): ~{per_unit / 60:.1f} min/unit; "
                      f"~{remaining} units remain -> ~{per_unit * remaining / 3600:.1f} h more "
                      f"(rough; larger corpora + bert/roberta run slower) ===\n", flush=True)

    print(f"\nAll units done in {(time.time() - grand_t0) / 60:.1f} min "
          f"({len(completed)} finalists this run).", flush=True)


if __name__ == "__main__":
    main()
