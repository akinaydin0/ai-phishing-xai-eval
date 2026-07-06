"""
Feature builders for the classical baselines.

The single rule enforced here: a vectorizer is **fit on TRAIN text only**, then used
to transform val/test (and, for cross-corpus runs, the target corpus). Callers must
never pass val/test text to `build_tfidf`.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

TFIDF_PARAMS = dict(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_features=100_000,
    sublinear_tf=True,
)


def build_tfidf(train_texts) -> TfidfVectorizer:
    """Fit a TfidfVectorizer on TRAIN texts only and return it."""
    vec = TfidfVectorizer(**TFIDF_PARAMS)
    vec.fit(list(train_texts))
    return vec


def tfidf_transform(vectorizer: TfidfVectorizer, texts) -> csr_matrix:
    return vectorizer.transform(list(texts))


def length_features(texts) -> np.ndarray:
    """[token_count, char_count, log1p(token_count)] — the shortcut-floor features.

    Computed per row from raw text; no fitting, so it is leakage-free by construction
    (the StandardScaler that follows is what gets fit on train only).
    """
    texts = list(texts)
    tok = np.array([len(t.split()) for t in texts], dtype=float)
    chars = np.array([len(t) for t in texts], dtype=float)
    return np.column_stack([tok, chars, np.log1p(tok)])
