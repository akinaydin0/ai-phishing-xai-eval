"""
Amendment 5: group-aware sensitivity analysis for d3_en (pre-registered 2026-08-17, commit 2bad9cc).

The frozen d3_en split is per-document, so PhishFuzzer variants of the same seed email straddle the
train/test boundary (AUDIT-D3: 100% of test families have a train/val sibling). This script rebuilds the
split GROUPED BY `Original_ID` and re-measures the affected numbers as a SENSITIVITY -- the committed
values remain the results of record.

Everything hyperparameter-related is FROZEN to committed artifacts; nothing is re-tuned or re-searched.
  classical   LinearSVC, C from results/hpo/d3_en_svc.json, vectorizer hpo.DEFAULT_LINEAR_VEC,
              score = expit(decision_function)
  transformer roberta-base, ONE run at the committed selected_lr from
              results/transformers/roberta-base__d3_en.json, reusing transformers_ft.train_lr verbatim
              (max_epochs 4, patience 1, batch 32, max_len 256, wd 0.01, warmup 0.1, seed 42); grouped
              val is used only for per-epoch early stopping / best-epoch selection, exactly as committed

Writes data/splits/d3_en_grouped.json and results/group_sens.json. Does NOT touch data/splits/d3_en.json
or any committed result.

Run: python -m src.experiments.group_sens
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.experiments import hpo, transformers_ft as TF
from src.splits import body_hash, load_corpus, load_splits
from src.xai import champions as C

_ROOT = Path(__file__).resolve().parent.parent.parent
SPLIT_OUT = _ROOT / "data" / "splits" / "d3_en_grouped.json"
JSON_OUT = _ROOT / "results" / "group_sens.json"
SEED = 42
CORPUS_POS_SHARE = 0.6605
DEV_WARN_PP = 2.0

COMMITTED = {  # for side-by-side reporting only
    "in_svc": 0.9993, "in_roberta": 0.9976,
    "d1_to_d3_transformer": 0.9243, "d1_to_d3_classical": 0.8965,
    "d2_to_d3_transformer": 0.7748, "d2_to_d3_classical": 0.8429,
    "d3_to_d1_transformer": 0.9659, "d3_to_d1_classical_svc": 0.8783,
    "d3_to_d2_transformer": 0.7306, "d3_to_d2_classical_svc": 0.7523,
}


# --------------------------------------------------------------------------- #
# STEP 1 - grouped split
# --------------------------------------------------------------------------- #
def build_grouped_split(df):
    groups = df["original_id"].to_numpy()
    idx = np.arange(len(df))
    g1 = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=SEED)
    train_idx, hold_idx = next(g1.split(idx, groups=groups))
    g2 = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=SEED)
    v_rel, t_rel = next(g2.split(hold_idx, groups=groups[hold_idx]))
    val_idx, test_idx = hold_idx[v_rel], hold_idx[t_rel]
    return {"train": np.sort(train_idx), "val": np.sort(val_idx), "test": np.sort(test_idx)}


def step1(df):
    print("STEP 1 - grouped 70/15/15 split by Original_ID (GroupShuffleSplit, seed 42)")
    parts = build_grouped_split(df)
    y = df["label"].to_numpy()
    g = df["original_id"].to_numpy()
    h = df["_hash"].to_numpy()

    fam = {k: set(g[v].tolist()) for k, v in parts.items()}
    shared = (fam["train"] & fam["val"]) | (fam["train"] & fam["test"]) | (fam["val"] & fam["test"])
    ok_a = len(shared) == 0
    print(f"  (a) no Original_ID in more than one part: {'PASS' if ok_a else 'FAIL'} "
          f"(shared families: {len(shared)})")

    allidx = np.concatenate([parts[k] for k in ("train", "val", "test")])
    ok_b = (len(allidx) == len(df) == 19164) and (len(set(allidx.tolist())) == len(df))
    print(f"  (b) pairwise disjoint and cover all 19,164 rows exactly once: "
          f"{'PASS' if ok_b else 'FAIL'} (n={len(allidx)}, unique={len(set(allidx.tolist()))})")

    print("  (c) achieved sizes and positive shares (corpus pos share 0.6605):")
    sizes, shares = {}, {}
    for k in ("train", "val", "test"):
        n = len(parts[k]); ps = float(y[parts[k]].mean())
        dev = (ps - CORPUS_POS_SHARE) * 100
        flag = "  WARN >2pp (report-only)" if abs(dev) > DEV_WARN_PP else ""
        sizes[k] = n; shares[k] = round(ps, 4)
        print(f"      {k:5s} n={n:6d} ({100*n/len(df):5.2f}%)  pos_share={ps:.4f}  "
              f"dev={dev:+.2f} pp{flag}")
    assert ok_a and ok_b, "STEP 1 hard asserts (a)/(b) failed"

    payload = {"corpus": "d3_en", "variant": "grouped_by_original_id", "n": int(len(df)),
               "seed": SEED, "group_key": "original_id",
               "n_train": sizes["train"], "n_val": sizes["val"], "n_test": sizes["test"],
               "pos_share_corpus": round(float(y.mean()), 6),
               "pos_share_train": round(float(y[parts['train']].mean()), 6),
               "pos_share_val": round(float(y[parts['val']].mean()), 6),
               "pos_share_test": round(float(y[parts['test']].mean()), 6),
               "n_families_train": len(fam["train"]), "n_families_val": len(fam["val"]),
               "n_families_test": len(fam["test"]),
               "train": sorted(h[parts["train"]].tolist()),
               "val": sorted(h[parts["val"]].tolist()),
               "test": sorted(h[parts["test"]].tolist())}
    SPLIT_OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"  wrote {SPLIT_OUT}")
    return parts, payload


# --------------------------------------------------------------------------- #
# STEP 2 - in-corpus sensitivity
# --------------------------------------------------------------------------- #
def fit_svc_grouped(df, parts):
    bp = json.loads((_ROOT / "results" / "hpo" / "d3_en_svc.json").read_text(encoding="utf-8"))["best_params"]
    vec = hpo._build_vec(hpo._vec_from_best("svc", bp))
    tr, te = parts["train"], parts["test"]
    Xtr = vec.fit_transform(df["text"].iloc[tr])
    model = hpo.make_model("svc", bp, device="cpu")
    model.fit(Xtr, df["label"].to_numpy()[tr])
    score_te = expit(model.decision_function(vec.transform(df["text"].iloc[te])))
    auc = roc_auc_score(df["label"].to_numpy()[te], score_te)
    return model, vec, float(auc), bp


def train_roberta_grouped(df, parts):
    meta = json.loads((_ROOT / "results" / "transformers" /
                       "roberta-base__d3_en.json").read_text(encoding="utf-8"))
    lr = float(meta["selected_lr"])
    TF.configure(256)
    tok = AutoTokenizer.from_pretrained("roberta-base")
    enc = {sp: (TF.encode(tok, df["text"].iloc[parts[sp]]), df["label"].to_numpy()[parts[sp]])
           for sp in ("train", "val", "test")}
    best, curves, epochs_run, peak, (per_dev, accum), wall = TF.train_lr("roberta-base", tok, enc, lr)
    model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
    model.load_state_dict(best["state"])
    model.to(TF.DEVICE).eval()
    ds = TF.EncodedDS(enc["test"][0], enc["test"][1])
    y, prob, _ = TF.evaluate(model, TF._loader(ds, tok, 64, False))
    auc = float(roc_auc_score(y, prob))
    info = {"lr": lr, "max_epochs": TF.MAX_EPOCHS, "patience": TF.PATIENCE, "batch": TF.BATCH,
            "max_len": TF.MAX_LEN, "weight_decay": TF.WEIGHT_DECAY, "warmup_ratio": TF.WARMUP_RATIO,
            "seed": TF.SEED, "epochs_run": epochs_run, "best_epoch": best["epoch"],
            "best_val_auc": best["val_auc"], "peak_mem_mb": peak, "wall_s": wall,
            "per_device_batch": per_dev, "grad_accum": accum}
    del model, best
    torch.cuda.empty_cache()
    return auc, info


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    t0 = time.time()
    out = {"amendment": "5", "pre_registration_commit": "2bad9cc", "seed": SEED,
           "note": "post-hoc SENSITIVITY; committed values remain the results of record",
           "committed_reference": COMMITTED}

    df = load_corpus("d3_en")
    d3raw = __import__("pandas").read_pickle(_ROOT / "data" / "cache" / "clean_d3.pkl")
    df["original_id"] = d3raw[d3raw["language"] == "en"].reset_index(drop=True)["original_id"].to_numpy()

    parts, split_payload = step1(df)
    out["split"] = {k: v for k, v in split_payload.items() if k not in ("train", "val", "test")}

    print("\nSTEP 2 - in-corpus sensitivity on the GROUPED test")
    svc, vec, svc_auc, bp = fit_svc_grouped(df, parts)
    print(f"  (a) LinearSVC (C={bp['C']}, frozen)  grouped test ROC-AUC = {svc_auc:.4f}   "
          f"committed {COMMITTED['in_svc']:.4f}   delta {svc_auc - COMMITTED['in_svc']:+.4f}")
    rob_auc, rob_info = train_roberta_grouped(df, parts)
    print(f"  (b) RoBERTa-256 (lr={rob_info['lr']}, frozen, 1 run) grouped test ROC-AUC = "
          f"{rob_auc:.4f}   committed {COMMITTED['in_roberta']:.4f}   "
          f"delta {rob_auc - COMMITTED['in_roberta']:+.4f}")
    higher = "classical (LinearSVC)" if svc_auc > rob_auc else "transformer (RoBERTa-256)"
    print(f"  (c) C1c direction under the grouped split: {higher} is higher "
          f"({max(svc_auc, rob_auc):.4f} vs {min(svc_auc, rob_auc):.4f})")
    out["step2"] = {"linearsvc_grouped": round(svc_auc, 4), "linearsvc_committed": COMMITTED["in_svc"],
                    "roberta_grouped": round(rob_auc, 4), "roberta_committed": COMMITTED["in_roberta"],
                    "c1c_direction_grouped": higher,
                    "hyperparameters": {"linearsvc": {**bp, "vectorizer": hpo._vec_from_best("svc", bp)},
                                        "roberta": rob_info}}

    print("\nSTEP 3 - cross-corpus onto the GROUPED d3 test (evaluation only, no retraining)")
    te = parts["test"]
    y_te = df["label"].to_numpy()[te]
    texts_te = df["text"].iloc[te].tolist()
    step3 = {}
    for tag, src_corpus, tf_model in (("d1", "d1_decontaminated_sb", "distilbert-base-uncased"),
                                      ("d2", "d2", "distilbert-base-uncased")):
        ch = C.TransformerChampion(tf_model, src_corpus)
        a_tf = float(roc_auc_score(y_te, ch.proba(texts_te)))
        del ch; torch.cuda.empty_cache()
        cl = C._SklearnChampion("lr", src_corpus)
        a_cl = float(roc_auc_score(y_te, cl.proba(texts_te)))
        ck, cc = f"{tag}_to_d3_transformer", f"{tag}_to_d3_classical"
        print(f"  ({'a' if tag=='d1' else 'b'}) {tag}->grouped d3  transformer {a_tf:.4f} "
              f"(committed {COMMITTED[ck]:.4f}, delta {a_tf-COMMITTED[ck]:+.4f}) | "
              f"classical LR {a_cl:.4f} (committed {COMMITTED[cc]:.4f}, delta {a_cl-COMMITTED[cc]:+.4f})")
        step3[tag] = {"transformer_grouped": round(a_tf, 4), "transformer_committed": COMMITTED[ck],
                      "classical_lr_grouped": round(a_cl, 4), "classical_lr_committed": COMMITTED[cc]}

    # (c) provenance decomposition on the grouped test
    seeds = json.loads((_ROOT / "data" / "raw" / "phishfuzzer" /
                        "PhishFuzzer_emails_original_seed_v1.json").read_text(encoding="utf-8"))
    D1SUB = ("ceas", "enron", "ling", "nazario", "nigerian", "spamassassin", "spam_assassin", "spamassasin")
    oid_d1 = {int(s["Original_ID"]): any(k in (s.get("Source") or "").lower() for k in D1SUB)
              for s in seeds}
    from_d1 = df["original_id"].iloc[te].map(oid_d1).to_numpy().astype(bool)
    ch = C.TransformerChampion("distilbert-base-uncased", "d1_decontaminated_sb")
    p_d1 = ch.proba(texts_te)
    del ch; torch.cuda.empty_cache()
    prov = {}
    print("  (c) provenance decomposition of the grouped d3 test (D1->D3 DistilBERT scores)")
    for name, mask in (("D1_descended", from_d1), ("independent_provenance", ~from_d1)):
        n = int(mask.sum()); pos = int(y_te[mask].sum()); neg = n - pos
        a = float(roc_auc_score(y_te[mask], p_d1[mask])) if 0 < pos < n else float("nan")
        unstable = neg < 100
        print(f"      {name:24s} n={n:5d} pos={pos:5d} neg={neg:4d} AUC={a:.4f}"
              f"{'   UNSTABLE (<100 negatives)' if unstable else ''}")
        prov[name] = {"n": n, "pos": pos, "neg": neg, "auc": round(a, 4), "unstable_lt100_neg": unstable}
    step3["provenance_decomposition_grouped_test"] = prov
    out["step3"] = step3

    print("\nSTEP 4 - outward transfer from the GROUPED-trained LinearSVC (committed D1/D2 tests)")
    step4 = {}
    for tag, corpus, ck in (("d1", "d1_decontaminated_sb", "d3_to_d1"), ("d2", "d2", "d3_to_d2")):
        p = load_splits(corpus)["test"].reset_index(drop=True)
        s = expit(svc.decision_function(vec.transform(p["text"])))
        a = float(roc_auc_score(p["label"].to_numpy(), s))
        print(f"  ({'a' if tag=='d1' else 'b'}) grouped-trained LinearSVC -> committed {tag} test "
              f"n={len(p)}  ROC-AUC {a:.4f}  | committed like-for-like LinearSVC "
              f"{COMMITTED[ck+'_classical_svc']:.4f} (delta {a-COMMITTED[ck+'_classical_svc']:+.4f})"
              f"  | committed transformer {COMMITTED[ck+'_transformer']:.4f}")
        step4[tag] = {"grouped_trained_linearsvc": round(a, 4),
                      "committed_linearsvc": COMMITTED[ck + "_classical_svc"],
                      "committed_transformer": COMMITTED[ck + "_transformer"], "n_test": int(len(p))}
    out["step4"] = step4

    out["runtime_min"] = round((time.time() - t0) / 60, 1)
    JSON_OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nTotal runtime {out['runtime_min']} min. Wrote {JSON_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
