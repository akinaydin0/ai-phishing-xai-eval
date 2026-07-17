"""
Shared raw-text word operations for the XAI phase.

One fixed word segmentation (`\w+`, unicode) defines the attribution unit AND the removal
unit, so the faithfulness mechanic is byte-identical across every family: explainers reduce
their attributions to these word positions; metrics blank these word spans and re-run the
full pipeline (re-vectorise / re-tokenise). See the pre-registration in notes/decisions.md.
"""

from __future__ import annotations

import re

WORD_RE = re.compile(r"\w+", re.UNICODE)


def word_spans(text: str) -> list[tuple[int, int]]:
    return [m.span() for m in WORD_RE.finditer(text)]


def words(text: str) -> list[str]:
    return [m.group() for m in WORD_RE.finditer(text)]


def words_lower(text: str) -> list[str]:
    return [m.group().lower() for m in WORD_RE.finditer(text)]


def remove_words(text: str, positions) -> str:
    """Blank (replace with spaces) the word spans at the given positions; keep all else."""
    spans = word_spans(text)
    drop = set(positions)
    chars = list(text)
    for i in drop:
        if 0 <= i < len(spans):
            s, e = spans[i]
            for j in range(s, e):
                chars[j] = " "
    return "".join(chars)


def keep_only_words(text: str, positions) -> str:
    """Blank every word EXCEPT those at the given positions (non-word chars are kept)."""
    spans = word_spans(text)
    keep = set(positions)
    chars = list(text)
    for i, (s, e) in enumerate(spans):
        if i not in keep:
            for j in range(s, e):
                chars[j] = " "
    return "".join(chars)


def top_k_positions(attr_class1, pred: int, k: int) -> list[int]:
    """Positions of the k words most attributed TOWARD THE PREDICTED CLASS.

    Explainers return class-1 attribution; orient by the predicted class (pred==0 ⇒ the most
    negative class-1 attributions are the most 'toward class 0'). Ties broken by position.
    """
    import numpy as np
    a = np.asarray(attr_class1, dtype=float)
    oriented = a if pred == 1 else -a
    if a.size == 0:
        return []
    order = np.argsort(-oriented, kind="stable")
    return order[:k].tolist()
