"""
Data loaders for the MSc thesis EDA.

Three public functions:

- load_phishing_dataset(decontaminate=False)
    The Kaggle multi-source phishing dataset (phishing_email.csv).
    When decontaminate=True, rows originating from Enron.csv are excluded
    so that cross-dataset comparison with the AUEB Enron-Spam legacy corpus
    is not contaminated by the same Enron emails appearing on both sides.

- load_phishing_sources()
    The 6 per-source CSVs concatenated with source attribution.
    Used for per-source heterogeneity analysis within Dataset 1.

- load_enron_spam()
    AUEB Enron-Spam (Metsis et al. 2006) preprocessed variant. Walks
    enron{1..6}/{ham,spam}/*.txt with utf-8 / latin-1 fallback.
"""

import json
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

# Resolve project-root-relative paths so the loader works regardless of CWD
# (e.g. when imported from a notebook whose CWD is notebooks/).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_PHISHING_DIR = _PROJECT_ROOT / "data" / "raw" / "phishing_email_dataset"
DEFAULT_ENRON_DIR = _PROJECT_ROOT / "data" / "raw" / "enron-spam"
DEFAULT_ENRON_CACHE = _PROJECT_ROOT / "data" / "processed" / "enron_spam.pkl"
DEFAULT_PHISHFUZZER_DIR = _PROJECT_ROOT / "data" / "raw" / "phishfuzzer"

_PHISHFUZZER_SEEDS_JSON = "PhishFuzzer_emails_original_seed_v1.json"
_PHISHFUZZER_VARIANTS_JSON = "PhishFuzzer_emails_entity_rephrased_v1.json"

# Per Decision C2: Phishing | Spam → 1 (malicious / unsolicited), Valid → 0.
# This matches the existing binary convention used by load_phishing_dataset
# (Kaggle phishing+spam under label=1) and load_enron_spam (label=1 = spam).
_PHISHFUZZER_BINARY_MAP = {"Phishing": 1, "Spam": 1, "Valid": 0}

PHISHING_SOURCES = {
    "CEAS_08": "CEAS_08.csv",
    "Enron": "Enron.csv",
    "Ling": "Ling.csv",
    "Nazario": "Nazario.csv",
    "Nigerian_Fraud": "Nigerian_Fraud.csv",
    "SpamAssasin": "SpamAssasin.csv",
}

ENRON_SUBSETS = ["enron1", "enron2", "enron3", "enron4", "enron5", "enron6"]

# Row counts of the 5 non-Enron source CSVs; used to enumerate candidate
# offsets at which Enron rows may begin in phishing_email.csv.
_NON_ENRON_SIZES = {
    "CEAS_08": 39154,
    "Ling": 2859,
    "Nazario": 1565,
    "Nigerian_Fraud": 3332,
    "SpamAssasin": 5809,
}


def _safe_read_text(path):
    """Read a text file with utf-8, fall back to latin-1. Strip BOM and edge whitespace."""
    try:
        return path.read_text(encoding="utf-8").lstrip("﻿").strip()
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1").lstrip("﻿").strip()


def _identify_enron_rows(combined, enron_labels, min_match_rate=0.95):
    """
    Locate the contiguous span of rows in phishing_email.csv that came from
    Enron.csv. Tries every plausible start offset (sums of subsets of the
    five non-Enron source sizes) and picks the window whose label sequence
    best matches Enron.csv.

    The Kaggle author re-labeled a small fraction of borderline cases when
    compiling phishing_email.csv, so exact label equality does not hold.
    A match rate ≥ min_match_rate (default 0.95) is treated as evidence
    that the window is the Enron span; the divergence is reported via
    warnings.warn for transparency.

    Returns a boolean mask of length len(combined). Raises if no candidate
    window meets the threshold.
    """
    n_enron = len(enron_labels)
    enron_arr = enron_labels.to_numpy()
    combined_arr = combined["label"].to_numpy()

    candidate_starts = {0}
    sizes = list(_NON_ENRON_SIZES.values())
    for r in range(1, len(sizes) + 1):
        for combo in combinations(sizes, r):
            candidate_starts.add(sum(combo))

    best_start, best_rate = None, 0.0
    for start in sorted(candidate_starts):
        end = start + n_enron
        if end > len(combined_arr):
            continue
        rate = float((combined_arr[start:end] == enron_arr).mean())
        if rate > best_rate:
            best_start, best_rate = start, rate

    if best_start is None or best_rate < min_match_rate:
        raise ValueError(
            f"Could not identify Enron-origin rows in phishing_email.csv. "
            f"Best label-sequence match rate {best_rate:.2%} "
            f"at offset {best_start} < required {min_match_rate:.0%}."
        )

    if best_rate < 1.0:
        n_diff = int(round((1.0 - best_rate) * n_enron))
        warnings.warn(
            f"Enron span located at offset {best_start} with "
            f"{best_rate:.2%} label-sequence match — {n_diff} of {n_enron} "
            f"labels differ between Enron.csv and the corresponding rows in "
            f"phishing_email.csv (Kaggle re-labeled borderline cases).",
            stacklevel=2,
        )

    mask = np.zeros(len(combined), dtype=bool)
    mask[best_start:best_start + n_enron] = True
    return mask


def load_phishing_dataset(path=DEFAULT_PHISHING_DIR, decontaminate=False):
    """
    Load the Kaggle multi-source phishing dataset (phishing_email.csv).

    Args:
        path: directory containing the unzipped Kaggle CSVs.
        decontaminate: if True, drop rows originating from Enron.csv so the
            corpus does not overlap with the AUEB Enron-Spam legacy dataset
            used as Dataset 2. Expected size after decontamination: 52,719.

    Returns:
        DataFrame with columns: text, label.
        text: str (the text_combined field from phishing_email.csv)
        label: int (0 = legitimate, 1 = phishing)
    """
    path = Path(path)
    csv_path = path / "phishing_email.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"phishing_email.csv not found under {path}. "
            "Unzip the Kaggle archive first."
        )
    df = pd.read_csv(csv_path, low_memory=False).rename(columns={"text_combined": "text"})
    df["label"] = df["label"].astype(int)
    df["text"] = df["text"].fillna("").astype(str)

    if not decontaminate:
        return df

    enron_path = path / "Enron.csv"
    if not enron_path.exists():
        raise FileNotFoundError(
            f"Enron.csv not found under {path}; required for decontamination."
        )
    enron_labels = pd.read_csv(enron_path, usecols=["label"])["label"].astype(int)
    mask = _identify_enron_rows(df, enron_labels)
    return df.loc[~mask].reset_index(drop=True)


def load_phishing_sources(path=DEFAULT_PHISHING_DIR):
    """
    Load the 6 per-source CSVs concatenated, with source attribution.

    Returns:
        DataFrame with columns: text, label, source.
        text:  subject + ' ' + body (NaN-safe, whitespace-stripped)
        label: int (0 = legitimate, 1 = phishing)
        source: one of CEAS_08, Enron, Ling, Nazario, Nigerian_Fraud, SpamAssasin
    """
    path = Path(path)
    parts = []
    for source_name, fname in PHISHING_SOURCES.items():
        fp = path / fname
        if not fp.exists():
            raise FileNotFoundError(f"{fname} not found under {path}.")
        df = pd.read_csv(fp, low_memory=False)
        subj = df["subject"].fillna("").astype(str) if "subject" in df.columns else pd.Series([""] * len(df))
        body = df["body"].fillna("").astype(str) if "body" in df.columns else pd.Series([""] * len(df))
        text = (subj + " " + body).str.strip()
        parts.append(pd.DataFrame({
            "text": text,
            "label": df["label"].astype(int),
            "source": source_name,
        }))
    return pd.concat(parts, ignore_index=True)


def load_enron_spam(path=DEFAULT_ENRON_DIR, use_cache=True, cache_path=DEFAULT_ENRON_CACHE):
    """
    Load the AUEB Enron-Spam (Metsis et al. 2006) preprocessed corpus.

    Walks enron{1..6}/{ham,spam}/*.txt. Reads utf-8; falls back to latin-1
    on UnicodeDecodeError.

    Caching: walking ~34k tiny files on Windows takes ~2.5 minutes. After a
    successful load the DataFrame is pickled to cache_path; subsequent calls
    return in <1 s. Delete the cache file (or pass use_cache=False) to force
    a full re-walk.

    Returns:
        DataFrame with columns: text, label, subset, source_file.
        text:        str (full file contents, stripped)
        label:       int (0 = ham, 1 = spam)
        subset:      str (one of enron1..enron6)
        source_file: str (original .txt filename)
    """
    path = Path(path)
    cache_path = Path(cache_path) if cache_path is not None else None

    if use_cache and cache_path is not None and cache_path.exists():
        return pd.read_pickle(cache_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Enron-Spam directory not found: {path}. "
            "Download and extract the 6 AUEB tar.gz archives first."
        )
    rows = []
    for subset in ENRON_SUBSETS:
        subset_dir = path / subset
        if not subset_dir.exists():
            continue
        for class_name, label_int in (("ham", 0), ("spam", 1)):
            class_dir = subset_dir / class_name
            if not class_dir.exists():
                continue
            for fp in sorted(class_dir.iterdir()):
                if fp.suffix != ".txt":
                    continue
                rows.append({
                    "text": _safe_read_text(fp),
                    "label": label_int,
                    "subset": subset,
                    "source_file": fp.name,
                })
    df = pd.DataFrame(rows)

    if use_cache and cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(cache_path)

    return df


def load_phishfuzzer(path=DEFAULT_PHISHFUZZER_DIR, variant="both"):
    """
    Load the PhishFuzzer (Tóth et al. 2025) AI-generated email corpus.

    Args:
        path:    directory containing the cloned PhishFuzzer repo.
        variant: 'both'    → seeds + LLM variants  (23,100 rows; default)
                 'seeds'    → 3,300 original seed templates
                 'variants' → 19,800 LLM-generated variants only

    Returns:
        DataFrame with columns:
            text         str  — Subject + '\\n' + Body, NaN-safe
            type_3class  str  — raw 'Phishing' / 'Spam' / 'Valid' label
            label        int  — binary (Phishing | Spam → 1, Valid → 0) per Decision C2
            no           int  — global unique ID
            original_id  int  — seed ID (seeds reference themselves)
            sender       str
            url_count    int  — len(URL list) or 0 if None
            file_count   int  — len(File list) or 0 if None
            motivation   str  — Follow the link / Open attachment / Reply / Unknown
            created_by   str  — Human / LLM / SpamAssassin
            source       str  — Manual / SpamAssassin.csv / ... / Synthetic Entity-Seed ...
            year         int
            language     str  — en / no / ja / ...
            entity_type  str  — variants only: well_known / fabricated; '' for seeds
            length_type  str  — variants only: short / medium / long;   '' for seeds

    A single typo in the source data ('Huamn' on one Spam seed row) is
    normalised silently to 'Human' in the `created_by` column.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"PhishFuzzer directory not found: {path}. "
            "Clone https://github.com/DataPhish/PhishFuzzer first."
        )
    if variant not in {"both", "seeds", "variants"}:
        raise ValueError(
            f"variant must be 'both', 'seeds', or 'variants'; got {variant!r}"
        )

    files_to_load = []
    if variant in {"both", "seeds"}:
        files_to_load.append(path / _PHISHFUZZER_SEEDS_JSON)
    if variant in {"both", "variants"}:
        files_to_load.append(path / _PHISHFUZZER_VARIANTS_JSON)

    records = []
    for fp in files_to_load:
        if not fp.exists():
            raise FileNotFoundError(f"PhishFuzzer file not found: {fp}")
        with open(fp, encoding="utf-8") as f:
            records.extend(json.load(f))

    raw = pd.DataFrame(records)

    subject = raw["Subject"].fillna("").astype(str)
    body = raw["Body"].fillna("").astype(str)
    text = (subject + "\n" + body).str.strip()

    type_3class = raw["Type"].astype(str)
    label = type_3class.map(_PHISHFUZZER_BINARY_MAP).astype(int)

    def _list_len(v):
        return len(v) if isinstance(v, list) else 0

    url_count = raw["URL"].apply(_list_len)
    file_count = raw["File"].apply(_list_len)

    created_by = raw["Created by"].replace({"Huamn": "Human"}).astype(str)

    def _opt_col(name):
        if name in raw.columns:
            return raw[name].fillna("").astype(str)
        return pd.Series([""] * len(raw), index=raw.index)

    return pd.DataFrame({
        "text": text,
        "type_3class": type_3class,
        "label": label,
        "no": raw["No."].astype(int),
        "original_id": raw["Original_ID"].astype(int),
        "sender": raw["Sender"].fillna("").astype(str),
        "url_count": url_count,
        "file_count": file_count,
        "motivation": raw["Motivation"].fillna("").astype(str),
        "created_by": created_by,
        "source": raw["Source"].fillna("").astype(str),
        "year": raw["Year"].astype(int),
        "language": raw["Language"].fillna("").astype(str),
        "entity_type": _opt_col("Entity_Type"),
        "length_type": _opt_col("Length_Type"),
    })
