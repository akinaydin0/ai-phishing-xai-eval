"""
Directive 10: restore the email text of the committed HITL method key.

`results/hitl/key.json` is the historical D10 package, the 260 blind token sets that were rated by the
human annotation and by the LLM judge (built 2026-07-18). It is published WITHOUT the email bodies
(`docs.<doc_uid>.text`), which come from third-party corpora that are not redistributed. Every doc keeps
its sha1 body `hash`, the key of the frozen splits, so the text can be restored exactly from the
cleaned TEST splits (`data/cache/`, not distributed with this repository):

  - default: put `text` back into every doc, taken by hash from load_splits(<explained corpus>)["test"].
    The result is byte-identical to the key that was rated (verified 2026-10-12: 204,883 bytes).
    `d10_analyze.py` and `d10_judge.py` read the email text, so they need the restored key.
  - --strip: the inverse; writes the text-free key that is committed.

Do NOT rebuild this key with `src/experiments/d10_package.py`. That script recomputes the attributions
with the current explainers and therefore builds a NEW package. Its LIME sets would differ: the rated
package used the void-LIME arm (the FIRST10 position baseline, notes/decisions.md Amendment 4), which the
corrected explainer no longer produces.

The restored key contains the email text: do not commit it. `git checkout -- results/hitl/key.json`
returns to the committed version.

Run: python -m src.experiments.d10_restore_text             (restore in place)
     python -m src.experiments.d10_restore_text --strip     (write the text-free key)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from src.splits import body_hash, load_splits
except ImportError:  # pragma: no cover
    from splits import body_hash, load_splits

_ROOT = Path(__file__).resolve().parent.parent.parent
HITL_DIR = _ROOT / "results" / "hitl"
KEY_PATH = HITL_DIR / "key.json"


def _dump(key: dict) -> bytes:
    # Same serialisation as d10_package.build(); LF line endings, no trailing newline.
    return json.dumps(key, indent=1).encode("utf-8")


def strip_text(key: dict) -> dict:
    for d in key["docs"].values():
        d.pop("text", None)
    return key


def restore_text(key: dict) -> dict:
    sel = json.loads((HITL_DIR / "selected_ids.json").read_text(encoding="utf-8"))
    texts: dict[str, str] = {}
    for cinfo in sel["corpora"].values():
        te = load_splits(cinfo["explained_corpus"])["test"]
        texts.update(zip(te["_hash"].to_numpy(), te["text"].tolist()))
    for doc_uid, d in key["docs"].items():
        text = texts[d["hash"]]
        assert body_hash(text) == d["hash"], f"{doc_uid}: body hash mismatch"
        d.pop("text", None)
        d["text"] = text  # last field, as in the rated key
    return key


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Restore (or strip) the email text of results/hitl/key.json.")
    ap.add_argument("--strip", action="store_true", help="write the text-free key instead")
    ap.add_argument("--out", type=Path, default=KEY_PATH, help="output path (default: in place)")
    args = ap.parse_args(argv)
    key = json.loads(KEY_PATH.read_text(encoding="utf-8"))
    key = strip_text(key) if args.strip else restore_text(key)
    args.out.write_bytes(_dump(key))
    print(f"  {'stripped' if args.strip else 'restored'} {len(key['docs'])} docs -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
