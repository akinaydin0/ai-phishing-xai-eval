"""
Attribution methods (Directive 7), all returning **class-1 attribution per raw-text word**.

- SHAP: LinearExplainer (tfidf linear), TreeExplainer (LightGBM), Partition/Text explainer
  (transformers, word masker aligned to the removal tokenizer).
- LIME: LimeTextExplainer for every family (via the champion's `proba`).
- Integrated Gradients: transformers only (captum LayerIntegratedGradients, 32 steps,
  PAD-embedding baseline).

Sklearn feature (unigram+bigram) SHAP values are distributed equally to the constituent word
positions; transformer sub-token attributions are summed within each word span. Orientation
toward the predicted class happens downstream (textops.top_k_positions). Hyperparameters are the
pre-registered ones (SHAP max_evals=500, LIME num_samples=1000, IG n_steps=32).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import shap
import torch
from lime.lime_text import LimeTextExplainer

from . import textops

MAX_EVALS = 500      # SHAP transformer Partition explainer
NUM_SAMPLES = 1000   # LIME
N_STEPS = 32         # Integrated Gradients

# The D7 defect value (Directive 10 Amendment 4): passed as LIME's `split_expression` it inverts the
# non-word test and yields identically-zero attributions. Kept ONLY so the historical (void) numbers
# in the canonical results/xai_metrics.csv can be reproduced for audit; never use it for new results.
LEGACY_VOID_SPLIT_EXPRESSION = r"\w+"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
# sklearn feature-SHAP -> word positions
# --------------------------------------------------------------------------- #
def _features_to_words(champ, text, feat_shap) -> np.ndarray:
    wlow = textops.words_lower(text)
    attr = np.zeros(len(wlow), dtype=float)
    if not wlow:
        return attr
    wmap = defaultdict(list)
    for i, w in enumerate(wlow):
        wmap[w].append(i)
    x = champ.vec.transform([text]).tocsr()
    fnames = champ.feature_names
    for j in x.indices:                       # present features only
        s = float(feat_shap[j])
        if s == 0.0:
            continue
        positions = []
        for tk in fnames[j].split():          # unigram -> 1 word, bigram -> 2 words
            positions.extend(wmap.get(tk, []))
        if positions:
            share = s / len(positions)
            for pi in positions:
                attr[pi] += share
    return attr


def _class1_feature_shap(sv, nfeat: int) -> np.ndarray:
    """Normalise a shap_values return (list / 2D / 3D) to a length-nfeat class-1 vector."""
    if isinstance(sv, list):
        arr = sv[1] if len(sv) > 1 else sv[0]
    else:
        arr = np.asarray(sv)
    arr = np.asarray(arr)
    if arr.ndim == 3:          # (n, nfeat, classes)
        arr = arr[0, :, -1]
    elif arr.ndim == 2:        # (n, nfeat)
        arr = arr[0]
    return np.asarray(arr).reshape(-1)


def _linear_explainer(champ):
    if getattr(champ, "_shap_lin", None) is None:
        parts_bg = champ.transform  # vectorizer already fit
        # 200-row train background (only its feature means matter for interventional SHAP)
        from src.splits import load_splits
        tr = load_splits(champ.corpus)["train"]
        rng = np.random.default_rng(42)
        idx = rng.choice(len(tr), size=min(200, len(tr)), replace=False)
        bg = champ.vec.transform([tr["text"].iloc[i] for i in idx])
        champ._shap_lin = shap.LinearExplainer(champ.model, bg)
    return champ._shap_lin


def shap_linear(champ, text) -> np.ndarray:
    expl = _linear_explainer(champ)
    x = champ.vec.transform([text])
    sv = expl.shap_values(x)
    return _features_to_words(champ, text, _class1_feature_shap(sv, x.shape[1]))


def _tree_explainer(champ):
    if getattr(champ, "_shap_tree", None) is None:
        champ._shap_tree = shap.TreeExplainer(champ.model)
    return champ._shap_tree


def shap_tree(champ, text) -> np.ndarray:
    expl = _tree_explainer(champ)
    x = champ.vec.transform([text]).toarray()
    sv = expl.shap_values(x, check_additivity=False)
    return _features_to_words(champ, text, _class1_feature_shap(sv, x.shape[1]))


# --------------------------------------------------------------------------- #
# Transformer SHAP (Partition/Text explainer, word masker aligned to removal)
# --------------------------------------------------------------------------- #
def shap_transformer(champ, text, max_evals: int = MAX_EVALS) -> np.ndarray:
    """SHAP Partition/Text explainer with a `\\W+`-splitting word masker; shap tokens are mapped
    back to the canonical `\\w+` word positions by extracting each shap token's word content in
    order (a shap token's value is split equally across the words it contains)."""
    nw = len(textops.word_spans(text))
    if nw == 0:
        return np.zeros(0)
    masker = shap.maskers.Text(r"\W+")                   # built-in tokenizer; handles mask token
    f = lambda arr: champ.proba([str(a) for a in arr])   # class-1 prob; shap batches internally
    explainer = shap.Explainer(f, masker)
    ex = explainer([text], max_evals=max_evals, batch_size=64, silent=True)
    vals = np.asarray(ex.values[0], dtype=float).reshape(-1)
    toks = list(ex.data[0])
    word_vals: list[float] = []
    for tok, v in zip(toks, vals):
        subs = textops.WORD_RE.findall(str(tok))
        if not subs:
            continue
        share = float(v) / len(subs)
        word_vals.extend([share] * len(subs))
    attr = np.asarray(word_vals, dtype=float)
    if len(attr) != nw:                                  # safety: align to canonical word count
        out = np.zeros(nw)
        out[:min(nw, len(attr))] = attr[:min(nw, len(attr))]
        return out
    return attr


# --------------------------------------------------------------------------- #
# LIME (all families)
# --------------------------------------------------------------------------- #
def lime_attr(champ, text, seed: int = 42, num_samples: int = NUM_SAMPLES,
              split_expression: str | None = None) -> np.ndarray:
    """LIME word attribution (class 1).

    `split_expression=None` (the default) OMITS the argument so LIME applies its documented default
    separator semantics (r"\\W+"). Pass `LEGACY_VOID_SPLIT_EXPRESSION` only to reproduce the D7
    defect for audit (see the module note and notes/decisions.md Amendments 4 / Directive 11).
    """
    wlow = textops.words_lower(text)
    if not wlow:
        return np.zeros(0)

    def predict(texts):
        p1 = champ.proba([str(t) for t in texts])
        p1 = np.clip(np.asarray(p1, dtype=float), 0.0, 1.0)
        return np.column_stack([1.0 - p1, p1])

    # ######################################################################################### #
    # DEFECT FIXED 2026-07-26 (Directive 11; found by the Directive 10 Part E review, Amendment 4).#
    # `split_expression` is LIME's SEPARATOR pattern (default r"\W+"), used as the non-word test  #
    # (`non_word = splitter.match` in lime.lime_text.IndexedString). The original code passed     #
    # r"\w+", INVERTING it: every real word was treated as a non-word and dropped from the        #
    # perturbation vocabulary, so `attr` was identically ZERO for all 2,100 explained documents   #
    # and "LIME top-k" degenerated to the first k word positions.                                #
    # The argument is now OMITTED (LIME's default). CONSEQUENCE, recorded deliberately: the 12    #
    # LIME rows of the canonical results/xai_metrics.csv are NO LONGER reproducible from this     #
    # code -- they are the void arm's output and are superseded by results/xai_metrics_limefix.csv#
    # (Directive 11, isolated namespace). To reproduce the historical void numbers for audit,     #
    # pass split_expression=LEGACY_VOID_SPLIT_EXPRESSION explicitly.                             #
    # Regression test that would have caught this: src/xai/test_known_answer.py.                 #
    # ######################################################################################### #
    kwargs = {} if split_expression is None else {"split_expression": split_expression}
    explainer = LimeTextExplainer(class_names=["ham", "phish"], bow=True,
                                  random_state=seed, **kwargs)
    n_unique = len(set(wlow))
    exp = explainer.explain_instance(text, predict, num_features=max(20, n_unique),
                                     num_samples=num_samples, labels=(1,))
    weights = {w.lower(): wt for w, wt in exp.as_list(label=1)}
    attr = np.zeros(len(wlow), dtype=float)
    for i, w in enumerate(wlow):
        if w in weights:
            attr[i] = weights[w]           # bag-of-words: same weight for every occurrence
    return attr


# --------------------------------------------------------------------------- #
# Integrated Gradients (transformers only; 32 steps, PAD-embedding baseline)
# --------------------------------------------------------------------------- #
def ig_attr(champ, text, n_steps: int = N_STEPS) -> np.ndarray:
    from captum.attr import LayerIntegratedGradients
    tok, model = champ.tokenizer, champ.model
    enc = tok(text, truncation=True, max_length=256, return_offsets_mapping=True,
              return_tensors="pt")
    input_ids = enc["input_ids"].to(DEVICE)
    attn = enc["attention_mask"].to(DEVICE)
    offsets = enc["offset_mapping"][0].tolist()
    special = tok.get_special_tokens_mask(input_ids[0].tolist(), already_has_special_tokens=True)

    # PAD-embedding baseline: PAD everywhere, special tokens kept in place (record: pre-registered).
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    ref = input_ids.clone()
    for i, is_special in enumerate(special):
        if not is_special:
            ref[0, i] = pad_id

    def forward(ids, mask):
        return model(input_ids=ids, attention_mask=mask).logits

    emb_layer = model.get_input_embeddings()
    lig = LayerIntegratedGradients(forward, emb_layer)
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        atts = lig.attribute(inputs=input_ids, baselines=ref, additional_forward_args=(attn,),
                             target=1, n_steps=n_steps, internal_batch_size=32)
    tok_attr = atts.sum(dim=-1).squeeze(0).detach().cpu().numpy()  # per sub-token

    # sub-token -> word: add each non-special sub-token's attribution to the word it overlaps.
    spans = textops.word_spans(text)
    attr = np.zeros(len(spans), dtype=float)
    wi = 0
    for k, (a, b) in enumerate(offsets):
        if a == b or (k < len(special) and special[k]):
            continue
        while wi < len(spans) and spans[wi][1] <= a:   # advance to the word covering this sub-token
            wi += 1
        j = wi
        while j < len(spans) and spans[j][0] < b:
            if spans[j][1] > a:                        # overlap
                attr[j] += float(tok_attr[k])
                break
            j += 1
    return attr


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
def attribute(champ, family: str, method: str, text: str,
              lime_split_expression: str | None = None) -> np.ndarray:
    """Dispatch to an explainer. `lime_split_expression` is an audit-only escape hatch: leave it None
    (LIME's default separator semantics); pass LEGACY_VOID_SPLIT_EXPRESSION to reproduce the D7 void
    LIME numbers."""
    if method == "shap":
        if family == "tfidf":
            return shap_linear(champ, text)
        if family == "ensemble":
            return shap_tree(champ, text)
        return shap_transformer(champ, text)
    if method == "lime":
        return lime_attr(champ, text, split_expression=lime_split_expression)
    if method == "ig":
        return ig_attr(champ, text)
    raise ValueError(f"{family}/{method}")
