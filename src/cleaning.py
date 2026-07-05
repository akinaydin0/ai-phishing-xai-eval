"""
Text-cleaning pipeline for the MSc thesis corpora.

Public helpers (all operate on a single string unless noted):

    strip_html(text)      -> str   remove tags/scripts/styles, keep visible text
    normalize_text(text)  -> str   encoding repair (ftfy), header-line stripping,
                                    whitespace normalisation
    drop_empty(df)        -> (df, stats)   remove rows with < MIN_TOKENS tokens
    exact_dedup(df)       -> (df, stats)   hash normalised body, dedup WITHIN corpus
    clean_corpus(df, id)  -> (clean_df, report)   full pipeline, in order

CLI:
    python -m src.cleaning --all        clean D1-full, D1-decontaminated, D2, D3,
                                        cache to data/cache/clean_*.pkl, write
                                        notes/cleaning_report.md, run sanity asserts.
    python -m src.cleaning --corpus d2  clean a single corpus.

Cleaning order (clean_corpus): strip_html -> normalize_text -> drop_empty -> exact_dedup.

Design notes
------------
* strip_html short-circuits when the text contains no '<' (the majority of rows,
  e.g. the punctuation-stripped Kaggle text_combined field), so BeautifulSoup is
  only invoked where markup might actually be present.
* Dedup is EXACT (sha1 of the normalised body) and WITHIN each corpus only, never
  across corpora — cross-corpus overlap is handled separately by the decontamination
  logic in data_loader.load_phishing_dataset(decontaminate=True).
* Non-English PhishFuzzer variants are NOT dropped here (that is a cross-corpus
  comparison filter, applied downstream); their count is surfaced in the report.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import sys
from pathlib import Path

import pandas as pd

# --- ftfy is optional per the task spec; degrade gracefully if absent. ---
try:
    import ftfy

    _HAVE_FTFY = True
except ImportError:  # pragma: no cover
    _HAVE_FTFY = False

# --- data_loader import works both as `python -m src.cleaning` (repo root on
#     sys.path -> src.data_loader) and when 'src' itself is on sys.path
#     (notebook style -> data_loader). ---
try:
    from src.data_loader import (
        load_enron_spam,
        load_phishfuzzer,
        load_phishing_dataset,
    )
except ImportError:  # pragma: no cover
    from data_loader import (
        load_enron_spam,
        load_phishfuzzer,
        load_phishing_dataset,
    )

logger = logging.getLogger("cleaning")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = _PROJECT_ROOT / "data" / "cache"
NOTES_DIR = _PROJECT_ROOT / "notes"
CLEANING_REPORT_PATH = NOTES_DIR / "cleaning_report.md"
KAGGLE_DIR = _PROJECT_ROOT / "data" / "raw" / "phishing_email_dataset"

MIN_TOKENS = 5            # drop_empty threshold: rows with < MIN_TOKENS are removed
CLASS_BALANCE_TOL_PP = 2.0  # WARN threshold for UNPINNED corpora only (see below)

# --- Determinism fingerprints (pinned expectations) --------------------------
# The three corpora are frozen public snapshots, so cleaning is deterministic.
# Each pinned corpus asserts its EXACT post-cleaning final row count and per-class
# counts. Any drift (re-download, library/parser change, cleaning-logic change)
# hard-fails the pipeline and forces a CONSCIOUS re-pin recorded in the commit
# message. This replaces the generic ±2pp class-shift gate for pinned corpora —
# the ±2pp check would otherwise trip on D2, whose duplicates are 74% spam so
# mandatory full dedup legitimately shifts its spam share by -3.65pp.
# Corpora NOT listed here fall back to the ±2pp shift check as a prominent WARN.
CLEANING_EXPECTED = {
    "d1_full":           {"final": 81970, "counts": {0: 39181, 1: 42789}},
    "d1_decontaminated": {"final": 52704, "counts": {0: 23798, 1: 28906}},
    "d2":                {"final": 29224, "counts": {0: 15406, 1: 13818}},
    "d3":                {"final": 19799, "counts": {0: 6600,  1: 13199}},
}

# Known email header keys stripped from the START of a body when they appear as
# residual header lines. Restricted to a real allowlist so body lines such as
# "Note:" or "Warning:" are never mistaken for headers. `x-...` handled separately.
_HEADER_KEYS = {
    "subject", "from", "to", "cc", "bcc", "date", "sent", "reply-to",
    "return-path", "received", "message-id", "in-reply-to", "references",
    "mime-version", "content-type", "content-transfer-encoding",
    "content-disposition", "delivered-to", "envelope-to", "thread-topic",
    "thread-index", "importance", "priority", "organization", "user-agent",
    "list-unsubscribe", "dkim-signature", "authentication-results",
    "precedence", "x-mailer", "x-originating-ip", "x-priority", "x-spam-status",
    "x-spam-flag", "x-spam-level",
}
_HEADER_LINE_RE = re.compile(r"^\s*([A-Za-z][A-Za-z-]*)\s*:\s*(.*)$")
_WHITESPACE_RE = re.compile(r"\s+")

# Corpus registry for the CLI. expected_dup_pct is the reference figure recorded
# in CLAUDE.md from the EDA phase (computed on raw loaded text); observed values
# will differ slightly because normalisation collapses near-identical bodies.
CORPUS_SPECS = {
    "d1_full": {
        "name": "D1 — Kaggle multi-source phishing (full)",
        "source": "Kaggle: naserabdullahalam/phishing-email-dataset",
        "loader": lambda: load_phishing_dataset(),
        "expected_dup_pct": 0.5,
    },
    "d1_decontaminated": {
        "name": "D1 — Kaggle multi-source phishing (decontaminated, Enron removed)",
        "source": "Kaggle, with the 29,767-row Enron.csv sub-collection removed",
        "loader": lambda: load_phishing_dataset(decontaminate=True),
        "expected_dup_pct": 0.5,
    },
    "d2": {
        "name": "D2 — Enron-Spam (AUEB, Metsis et al. 2006)",
        "source": "http://www2.aueb.gr/users/ion/data/enron-spam/preprocessed/",
        "loader": lambda: load_enron_spam(),
        "expected_dup_pct": 9.6,
    },
    "d3": {
        "name": "D3 — PhishFuzzer variants (Tóth et al. 2025)",
        "source": "GitHub: DataPhish/PhishFuzzer — 19,800 LLM variants",
        "loader": lambda: load_phishfuzzer(variant="variants"),
        "expected_dup_pct": 0.3,
    },
}


# --------------------------------------------------------------------------- #
# String-level helpers
# --------------------------------------------------------------------------- #
def strip_html(text: str) -> str:
    """Remove HTML tags, <script>/<style> contents, and unescape entities.

    Short-circuits when no '<' is present so the majority of already-plain rows
    skip the BeautifulSoup round-trip entirely.
    """
    if not isinstance(text, str) or not text:
        return ""
    if "<" not in text:
        return text
    # Local import keeps module import cheap when strip_html is never hit.
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(text, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator=" ")


def _strip_leading_headers(text: str) -> str:
    """Drop the contiguous run of residual header lines at the start of a body.

    Skips only blank lines that *precede* the header block, then consumes
    consecutive recognised-header lines and STOPS at the first line that is not a
    recognised header — including the blank line that, per RFC 5322, separates the
    header block from the body. This prevents deleting legitimate body lines that
    happen to start with an allowlisted key + colon (e.g. "Date: tomorrow at 3pm")
    once the real header block has ended.
    """
    lines = text.split("\n")
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1  # skip blank lines that precede the header block
    while i < len(lines):
        m = _HEADER_LINE_RE.match(lines[i])
        if m and m.group(1).lower() in _HEADER_KEYS:
            i += 1
        else:
            break  # first non-header line (blank or body) ends the header block
    return "\n".join(lines[i:])


def normalize_text(text: str) -> str:
    """Repair encoding artifacts, strip leading header lines, normalise whitespace.

    Order matters: ftfy first (fix mojibake), then header stripping (needs line
    structure), then whitespace collapse (destroys line structure).
    """
    if not isinstance(text, str) or not text:
        return ""
    if _HAVE_FTFY:
        text = ftfy.fix_text(text)
    text = _strip_leading_headers(text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def _clean_series(texts: pd.Series) -> pd.Series:
    """Apply strip_html then normalize_text across a Series of strings."""
    return texts.map(strip_html).map(normalize_text)


def _token_count(texts: pd.Series) -> pd.Series:
    return texts.str.split().map(lambda toks: len(toks) if isinstance(toks, list) else 0)


# --------------------------------------------------------------------------- #
# Frame-level helpers
# --------------------------------------------------------------------------- #
def drop_empty(df: pd.DataFrame, text_col: str = "text", corpus_id: str | None = None):
    """Remove rows whose cleaned text has < MIN_TOKENS non-whitespace tokens.

    Returns (filtered_df, stats) where stats = {n_before, n_after, n_dropped}.
    """
    n_before = len(df)
    keep = _token_count(df[text_col]) >= MIN_TOKENS
    out = df.loc[keep].reset_index(drop=True)
    stats = {"n_before": n_before, "n_after": len(out), "n_dropped": n_before - len(out)}
    logger.info(
        "[%s] drop_empty: removed %d rows with < %d tokens (%d -> %d)",
        corpus_id or "?", stats["n_dropped"], MIN_TOKENS, n_before, len(out),
    )
    return out, stats


def exact_dedup(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
    corpus_id: str | None = None,
):
    """Exact-dedup WITHIN this corpus by sha1 of the normalised body; keep first.

    Returns (deduped_df, stats) where stats records total removed and a per-class
    breakdown of what was removed.
    """
    n_before = len(df)
    hashes = df[text_col].map(lambda s: hashlib.sha1(s.encode("utf-8")).hexdigest())
    dup_mask = hashes.duplicated(keep="first")
    removed_by_class = (
        df.loc[dup_mask, label_col].value_counts().sort_index().to_dict()
        if label_col in df.columns
        else {}
    )
    out = df.loc[~dup_mask].reset_index(drop=True)
    stats = {
        "n_before": n_before,
        "n_after": len(out),
        "n_dropped": int(dup_mask.sum()),
        "removed_by_class": {int(k): int(v) for k, v in removed_by_class.items()},
    }
    logger.info(
        "[%s] exact_dedup: removed %d duplicate rows (%d -> %d); by class %s",
        corpus_id or "?", stats["n_dropped"], n_before, len(out), stats["removed_by_class"],
    )
    return out, stats


def _class_balance(df: pd.DataFrame, label_col: str = "label") -> dict:
    """Positive-class (label==1) share and per-class counts."""
    counts = df[label_col].value_counts().sort_index().to_dict()
    counts = {int(k): int(v) for k, v in counts.items()}
    n = len(df)
    pos = counts.get(1, 0)
    return {"counts": counts, "n": n, "pos_share": (pos / n * 100.0) if n else 0.0}


def clean_corpus(
    df: pd.DataFrame,
    corpus_id: str,
    text_col: str = "text",
    label_col: str = "label",
):
    """Run the full cleaning pipeline on one corpus.

    Order: strip_html -> normalize_text -> drop_empty -> exact_dedup.

    Returns (clean_df, report) where report is a dict of all counts needed for
    notes/cleaning_report.md and the sanity asserts. The cleaned text overwrites
    `text_col`; all other columns are preserved.
    """
    spec = CORPUS_SPECS.get(corpus_id, {})
    rows_in = len(df)
    balance_before = _class_balance(df, label_col)

    logger.info("[%s] cleaning %d rows ...", corpus_id, rows_in)
    work = df.copy()
    work[text_col] = _clean_series(work[text_col])
    after_html_normalize = len(work)  # transform-only; row count unchanged

    work, empty_stats = drop_empty(work, text_col=text_col, corpus_id=corpus_id)
    after_drop_empty = len(work)

    work, dedup_stats = exact_dedup(
        work, text_col=text_col, label_col=label_col, corpus_id=corpus_id
    )
    after_dedup = len(work)

    balance_after = _class_balance(work, label_col)
    shift_pp = balance_after["pos_share"] - balance_before["pos_share"]
    dup_pct_observed = (dedup_stats["n_dropped"] / rows_in * 100.0) if rows_in else 0.0

    report = {
        "corpus_id": corpus_id,
        "name": spec.get("name", corpus_id),
        "source": spec.get("source", ""),
        "rows_in": rows_in,
        "after_html_normalize": after_html_normalize,
        "after_drop_empty": after_drop_empty,
        "after_dedup": after_dedup,
        "final": after_dedup,
        "n_dropped_empty": empty_stats["n_dropped"],
        "n_dropped_dup": dedup_stats["n_dropped"],
        "dedup_removed_by_class": dedup_stats["removed_by_class"],
        "dup_pct_observed": dup_pct_observed,
        "dup_pct_expected": spec.get("expected_dup_pct"),
        "balance_before": balance_before,
        "balance_after": balance_after,
        "pos_share_before": balance_before["pos_share"],
        "pos_share_after": balance_after["pos_share"],
        "pos_share_shift_pp": shift_pp,
    }
    return work, report


# --------------------------------------------------------------------------- #
# Caching
# --------------------------------------------------------------------------- #
def _cache_paths(corpus_id: str):
    return (
        CACHE_DIR / f"clean_{corpus_id}.pkl",
        CACHE_DIR / f"clean_{corpus_id}_report.json",
    )


def clean_corpus_cached(corpus_id: str, force: bool = False):
    """Clean a registered corpus, using data/cache/clean_<id>.pkl when available.

    The bare cleaned DataFrame is pickled (so downstream modelling can
    `pd.read_pickle` it directly); the report dict is written alongside as JSON.
    """
    import json

    df_path, rep_path = _cache_paths(corpus_id)
    if not force and df_path.exists() and rep_path.exists():
        logger.info("[%s] loading cleaned corpus from cache %s", corpus_id, df_path.name)
        clean_df = pd.read_pickle(df_path)
        report = json.loads(rep_path.read_text(encoding="utf-8"))
        return clean_df, report

    spec = CORPUS_SPECS[corpus_id]
    raw = spec["loader"]()
    clean_df, report = clean_corpus(raw, corpus_id)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    clean_df.to_pickle(df_path)
    rep_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("[%s] cached cleaned corpus -> %s", corpus_id, df_path.name)
    return clean_df, report


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _fmt_counts(counts: dict) -> str:
    return ", ".join(f"{k}:{v:,}" for k, v in sorted(counts.items()))


def write_cleaning_report(reports: dict, extra: dict | None = None) -> str:
    """Render notes/cleaning_report.md from the per-corpus report dicts."""
    lines = []
    lines.append("# Cleaning Report\n")
    lines.append(
        "Generated by `python -m src.cleaning --all`. Pipeline order per corpus: "
        "`strip_html → normalize_text → drop_empty (<5 tokens) → exact_dedup "
        "(sha1 of normalised body, within-corpus)`.\n"
    )

    # --- Row-flow table ---
    lines.append("## Row flow\n")
    lines.append(
        "| Corpus | rows in | after html/normalize | after drop_empty | "
        "after dedup | final |"
    )
    lines.append("|---|--:|--:|--:|--:|--:|")
    for cid, r in reports.items():
        lines.append(
            f"| {r['name']} | {r['rows_in']:,} | {r['after_html_normalize']:,} | "
            f"{r['after_drop_empty']:,} | {r['after_dedup']:,} | {r['final']:,} |"
        )
    lines.append("")

    # --- Removal detail ---
    lines.append("## Removal detail\n")
    lines.append(
        "| Corpus | empty dropped (<5 tok) | duplicates dropped | dup% observed | "
        "dup% expected (EDA) | dedup removed by class |"
    )
    lines.append("|---|--:|--:|--:|--:|---|")
    for cid, r in reports.items():
        exp = r["dup_pct_expected"]
        exp_s = f"{exp:.1f}%" if exp is not None else "—"
        by_class = _fmt_counts(r["dedup_removed_by_class"]) or "—"
        lines.append(
            f"| {r['name']} | {r['n_dropped_empty']:,} | {r['n_dropped_dup']:,} | "
            f"{r['dup_pct_observed']:.2f}% | {exp_s} | {by_class} |"
        )
    lines.append("")
    lines.append(
        "*Observed dup% is duplicates removed ÷ rows in. It differs from the EDA "
        "reference because normalisation (header stripping, whitespace collapse) "
        "makes additional near-identical bodies hash-identical, and because D3 "
        "here is the 19,800 variants only, not the full 23,100-row corpus.*\n"
    )

    # --- Class balance ---
    lines.append("## Class balance before / after (label 1 = phishing/spam)\n")
    lines.append(
        "| Corpus | counts before | pos-share before | counts after | "
        "pos-share after | Δ pos-share (pp) |"
    )
    lines.append("|---|---|--:|---|--:|--:|")
    for cid, r in reports.items():
        bb, ba = r["balance_before"], r["balance_after"]
        lines.append(
            f"| {r['name']} | {_fmt_counts(bb['counts'])} | {bb['pos_share']:.2f}% | "
            f"{_fmt_counts(ba['counts'])} | {ba['pos_share']:.2f}% | "
            f"{r['pos_share_shift_pp']:+.2f} |"
        )
    lines.append("")

    if extra:
        lines.append("## Notes\n")
        for note in extra.get("notes", []):
            lines.append(f"- {note}")
        lines.append("")

    text = "\n".join(lines)
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    CLEANING_REPORT_PATH.write_text(text, encoding="utf-8")
    logger.info("wrote %s", CLEANING_REPORT_PATH)
    return text


# --------------------------------------------------------------------------- #
# Sanity asserts (Task 3)
# --------------------------------------------------------------------------- #
def run_sanity_asserts(cleaned: dict, reports: dict) -> list:
    """Return a list of (check_name, status, detail) tuples where status is one of
    'PASS', 'FAIL', or 'WARN'. FAIL is hard (non-zero exit); WARN is surfaced
    prominently but does not fail the pipeline. Every check runs so the full
    picture is always reported.
    """
    results = []

    def _hard(name, ok, detail):
        results.append((name, "PASS" if ok else "FAIL", detail))

    # 1. No empty bodies remain (every row >= MIN_TOKENS tokens).  [hard]
    for cid, df in cleaned.items():
        n_short = int((_token_count(df["text"]) < MIN_TOKENS).sum())
        _hard(f"no-empty[{cid}]", n_short == 0,
              f"{n_short} rows with < {MIN_TOKENS} tokens remain")

    # 2. 0% exact duplicates after cleaning (within corpus).  [hard]
    for cid, df in cleaned.items():
        hashes = df["text"].map(lambda s: hashlib.sha1(s.encode("utf-8")).hexdigest())
        n_dup = int(hashes.duplicated().sum())
        _hard(f"zero-dup[{cid}]", n_dup == 0, f"{n_dup} exact duplicates remain")

    # 3. D1-decontaminated contains no Enron.csv rows.
    #    The pure row-count invariant (removed == len(Enron.csv)) holds by
    #    construction in the loader (it removes a contiguous window of exactly
    #    len(Enron.csv) rows), so on its own it cannot detect a *wrong-window*
    #    removal. Content matching against Enron.csv is infeasible because D1's
    #    text_combined is heavily preprocessed (fuzzy-key overlap with raw
    #    Enron.csv < 0.03). We therefore verify the removed rows' PER-CLASS label
    #    distribution matches Enron.csv within tolerance: phishing_email.csv is
    #    label-ordered, so a wrong window would remove a different class split.
    #    The total-count invariant is kept as a secondary diagnostic.
    try:
        enron_labels = pd.read_csv(KAGGLE_DIR / "Enron.csv", usecols=["label"])["label"].astype(int)
        enron_n = len(enron_labels)
        enron_counts = {int(k): int(v) for k, v in enron_labels.value_counts().items()}
        # balance_before counts may have str keys when reloaded from cached JSON.
        full_counts = {int(k): int(v) for k, v in reports["d1_full"]["balance_before"]["counts"].items()}
        decon_counts = {int(k): int(v) for k, v in reports["d1_decontaminated"]["balance_before"]["counts"].items()}
        removed = reports["d1_full"]["rows_in"] - reports["d1_decontaminated"]["rows_in"]
        removed_by_class = {c: full_counts.get(c, 0) - decon_counts.get(c, 0) for c in (0, 1)}
        tol = 300  # absorbs the ~174 borderline rows Kaggle re-labeled in the Enron span
        per_class_ok = all(
            abs(removed_by_class[c] - enron_counts.get(c, 0)) <= tol for c in (0, 1)
        )
        _hard("decontam-count==Enron.csv", removed == enron_n,
              f"removed {removed:,} rows; Enron.csv has {enron_n:,}")
        _hard("decontam-classdist==Enron.csv", per_class_ok,
              f"removed by class {removed_by_class} vs Enron.csv {enron_counts} (tol ±{tol})")
    except Exception as e:  # pragma: no cover
        _hard("decontam-check", False, f"check errored: {e}")

    # 4. Determinism fingerprint (pinned corpora, hard) OR ±2pp shift (unpinned, WARN).
    for cid, r in reports.items():
        if cid in CLEANING_EXPECTED:
            exp = CLEANING_EXPECTED[cid]
            obs_final = r["final"]
            obs_counts = {int(k): int(v) for k, v in r["balance_after"]["counts"].items()}
            exp_counts = {int(k): int(v) for k, v in exp["counts"].items()}
            ok = (obs_final == exp["final"]) and (obs_counts == exp_counts)
            detail = (f"final {obs_final:,} vs pinned {exp['final']:,}; "
                      f"counts {obs_counts} vs pinned {exp_counts}")
            if not ok:
                detail += "  -> DRIFT: re-pin CLEANING_EXPECTED and record it in the commit."
            _hard(f"fingerprint[{cid}]", ok, detail)
        else:
            shift = abs(r["pos_share_shift_pp"])
            status = "PASS" if shift <= CLASS_BALANCE_TOL_PP else "WARN"
            results.append(
                (f"class-shift[{cid}]", status,
                 f"|Δ pos-share| = {shift:.2f}pp (WARN tol {CLASS_BALANCE_TOL_PP:.0f}pp, "
                 f"unpinned corpus)")
            )

    return results


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None):
    parser = argparse.ArgumentParser(description="Corpus cleaning pipeline.")
    parser.add_argument("--all", action="store_true",
                        help="clean D1-full, D1-decontaminated, D2, D3")
    parser.add_argument("--corpus", choices=list(CORPUS_SPECS),
                        help="clean a single corpus by id")
    parser.add_argument("--force", action="store_true",
                        help="ignore cache and re-clean")
    parser.add_argument("--no-asserts", action="store_true",
                        help="skip the sanity asserts (still writes the report)")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )

    if not args.all and not args.corpus:
        parser.error("specify --all or --corpus <id>")

    corpus_ids = list(CORPUS_SPECS) if args.all else [args.corpus]

    cleaned, reports = {}, {}
    for cid in corpus_ids:
        clean_df, report = clean_corpus_cached(cid, force=args.force)
        cleaned[cid] = clean_df
        reports[cid] = report

    # Language breakdown note for D3 (non-English handled downstream, not dropped).
    notes = []
    if "d3" in cleaned and "language" in cleaned["d3"].columns:
        lang = cleaned["d3"]["language"]
        n_non_en = int((lang != "en").sum())
        notes.append(
            f"D3 non-English variants retained in the cache but excluded downstream "
            f"from cross-corpus comparison: {n_non_en:,} of {len(lang):,} "
            f"({n_non_en / len(lang) * 100:.1f}%)."
        )
    # D2 class-balance shift documented as a finding (per methodology decision).
    if "d2" in reports:
        r = reports["d2"]
        by_class = r["dedup_removed_by_class"]
        spam_dups = by_class.get(1, by_class.get("1", 0))
        total_dups = r["n_dropped_dup"]
        pct = (spam_dups / total_dups * 100) if total_dups else 0
        notes.append(
            f"**D2 class balance (finding, not a hidden exception):** {spam_dups:,} of "
            f"{total_dups:,} D2 duplicates ({pct:.0f}%) are spam — spam-campaign "
            f"repetition. Mandatory full within-corpus dedup therefore shifts D2's spam "
            f"share {r['pos_share_before']:.2f}% → {r['pos_share_after']:.2f}% "
            f"({r['pos_share_shift_pp']:+.2f}pp). Duplicates are never retained to "
            f"preserve a class ratio; this shift is an expected consequence of "
            f"deduplication, verified against a pinned fingerprint."
        )
    notes.append(
        "Two duplicate definitions are kept distinct so dissertation figures never "
        "look contradictory: **9.6% exact-raw** duplicates on the untouched loaded text "
        "(the interim-report figure for D2), vs **11.64% after normalisation** "
        "(header-stripping + whitespace-collapse make additional near-identical bodies "
        "hash-identical). The pipeline removes the larger, post-normalisation set."
    )
    notes.append(
        "Final row counts and per-class counts above are pinned as determinism "
        "fingerprints in `CLEANING_EXPECTED`; any drift hard-fails the pipeline and "
        "forces a conscious re-pin recorded in the commit message."
    )
    notes.append(
        "Dedup is within-corpus only; cross-corpus overlap (Kaggle↔Enron) is removed "
        "upstream by data_loader.load_phishing_dataset(decontaminate=True)."
    )

    # Render the report from ALL available cached corpora (freshly-computed ones
    # overlaid on cached JSONs), in registry order, so a single-corpus run never
    # clobbers the full four-corpus report with a partial one.
    import json

    render_reports = {}
    for cid in CORPUS_SPECS:
        if cid in reports:
            render_reports[cid] = reports[cid]
        else:
            _, rep_path = _cache_paths(cid)
            if rep_path.exists():
                render_reports[cid] = json.loads(rep_path.read_text(encoding="utf-8"))
    write_cleaning_report(render_reports, extra={"notes": notes})

    if args.no_asserts or not args.all:
        return 0

    # --- Sanity asserts: run all, print a summary. FAIL is hard (non-zero exit);
    #     WARN is surfaced prominently but does not fail the pipeline. ---
    results = run_sanity_asserts(cleaned, reports)
    print("\n" + "=" * 78)
    print("SANITY ASSERTS")
    print("=" * 78)
    n_fail = n_warn = 0
    for name, status, detail in results:
        if status == "FAIL":
            n_fail += 1
        elif status == "WARN":
            n_warn += 1
        print(f"  [{status:4s}] {name:32s} {detail}")
    print("=" * 78)

    if n_fail:
        print(f"\n{n_fail} sanity assert(s) FAILED. Report was still written to "
              f"{CLEANING_REPORT_PATH.relative_to(_PROJECT_ROOT)}.")
        raise AssertionError(f"{n_fail} sanity assert(s) failed — see summary above.")

    tail = f" ({n_warn} warning(s))" if n_warn else ""
    print(f"\nAll hard sanity asserts passed{tail}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
