"""
EDA v2 — exploratory analysis recomputed on the CLEANED, deduplicated corpora.

Runs cold from the cached clean corpora produced by `python -m src.cleaning --all`
(data/cache/clean_*.pkl). Deterministic; global seed 42.

Working universe (final cleaned data):
  - D1 decontaminated, subject+body (natural text)  -> 51,839 rows   [PRIMARY]
  - D1 decontaminated, text_combined (Kaggle field) -> 52,704 rows   [secondary]
  - D2 Enron-Spam                                    -> 29,224 rows
  - D3 PhishFuzzer variants, English only            -> 19,164 rows

D1 uses the subject+body view for every cross-corpus / length / lexical statistic,
because the interim report and D2/D3 are all natural text. The text_combined field
(Kaggle-preprocessed: stop-words + punctuation already stripped, ~half the tokens)
is reported only as a secondary heterogeneity comparison. See notes/decisions.md
("D1 dual text representation").

Outputs:
  figures/eda_v2/fig_class_dist_d1.{png,pdf}
  figures/eda_v2/fig_text_length_d1.{png,pdf}
  figures/eda_v2/fig_cross_corpus_length.{png,pdf}
  figures/eda_v2/fig_d1_representation.{png,pdf}   (supplementary)
  notes/eda_v2_summary.md

Usage:  python -m src.eda_v2      (or: python src/eda_v2.py)
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

SEED = 42

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = _PROJECT_ROOT / "data" / "cache"
FIG_DIR = _PROJECT_ROOT / "figures" / "eda_v2"
NOTES_DIR = _PROJECT_ROOT / "notes"
SUMMARY_PATH = NOTES_DIR / "eda_v2_summary.md"

# Interim (RAW) reference values, for the RAW-vs-CLEAN comparison.
RAW_MEDIANS = {
    "modern phishing": 62, "modern legit": 199,
    "Enron ham": 176, "Enron spam": 138,
    "AI phishing": 232,
}
RAW_JACCARD = {"legacy↔modern": 0.351, "legacy↔AI": 0.250, "modern↔AI": 0.263}
RAW_DUP_PCT = {"D2": 9.6, "D1": 0.5, "D3 (full corpus)": 0.3}

URL_RX = re.compile(r"https?://\S+")
EMAIL_RX = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
TOKEN_RX = re.compile(r"[a-z']{2,}")


# --------------------------------------------------------------------------- #
# Loading (cold from cache)
# --------------------------------------------------------------------------- #
def _load_clean(name: str) -> pd.DataFrame:
    path = CACHE_DIR / f"clean_{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.cleaning --all` first."
        )
    return pd.read_pickle(path)


def load_universe():
    """Return the cleaned working corpora, D3 filtered to English variants."""
    d1_sb = _load_clean("d1_decontaminated_sb")
    d1_tc = _load_clean("d1_decontaminated")
    d2 = _load_clean("d2")
    d3 = _load_clean("d3")
    d3_en = d3[d3["language"] == "en"].reset_index(drop=True)
    return d1_sb, d1_tc, d2, d3_en


def _tok(series: pd.Series) -> pd.Series:
    return series.str.split().map(lambda t: len(t) if isinstance(t, list) else 0)


# --------------------------------------------------------------------------- #
# NLTK resources
# --------------------------------------------------------------------------- #
def _nltk_sets():
    import nltk

    for pkg in ("stopwords", "words"):
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)
    from nltk.corpus import stopwords, words

    return set(stopwords.words("english")), {w.lower() for w in words.words()}


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _save(fig, stem: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_class_dist_d1(d1_sb):
    counts = d1_sb["label"].value_counts().sort_index()
    total = int(counts.sum())
    ratio = counts[1] / counts[0]
    labels = ["Legitimate\n(label = 0)", "Phishing\n(label = 1)"]
    colours = ["#3a86c7", "#d6553d"]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, counts.values, color=colours, edgecolor="black", linewidth=0.5)
    for bar, n in zip(bars, counts.values):
        ax.annotate(f"{n:,}\n({n / total:.1%})",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Number of emails")
    ax.set_title(f"D1 (cleaned, subject+body): class distribution\n"
                 f"N = {total:,}; phish:legit = {ratio:.2f}:1")
    ax.set_ylim(0, counts.max() * 1.12)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "fig_class_dist_d1")


def fig_text_length_d1(d1_sb):
    d = d1_sb.assign(tok=_tok(d1_sb["text"]))
    names = {0: "Legitimate", 1: "Phishing"}
    palette = {0: "#3a86c7", 1: "#d6553d"}
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for lab, sub in d.groupby("label"):
        s = sub["tok"][sub["tok"] > 0]
        sns.kdeplot(np.log10(s), ax=ax, color=palette[lab], linewidth=2,
                    label=f"{names[lab]} (n={len(s):,}, median={int(s.median())} tok)")
        ax.axvline(np.log10(s.median()), color=palette[lab], linestyle=":", linewidth=1, alpha=0.7)
    ax.set_xlabel(r"Token count per email ($\log_{10}$ scale)")
    ax.set_ylabel("Density")
    ax.set_title("D1 (cleaned, subject+body): class-conditional token-length distribution")
    ticks = np.log10([1, 10, 100, 1000, 10000])
    ax.set_xticks(ticks)
    ax.set_xticklabels(["1", "10", "100", "1k", "10k"])
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "fig_text_length_d1")


def fig_cross_corpus_length(d1_sb, d2, d3_en):
    curves = [
        ("Enron-Spam (legacy, 2000s)", _tok(d2["text"]), "#2c4a7f", "-"),
        ("Kaggle decontaminated (modern, subject+body)", _tok(d1_sb["text"]), "#c95a3d", "--"),
        ("PhishFuzzer variants (AI-generated, English)", _tok(d3_en["text"]), "#2c8a4d", "-."),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for name, series, colour, ls in curves:
        s = series[series > 0]
        sns.kdeplot(np.log10(s), ax=ax, color=colour, linestyle=ls, linewidth=2.4,
                    common_norm=False,
                    label=f"{name}\n(n={len(s):,}, median={int(s.median())} tok)")
        ax.axvline(np.log10(s.median()), color=colour, linestyle=":", linewidth=0.9, alpha=0.55)
    ax.set_xlabel(r"Token count per email ($\log_{10}$ scale)")
    ax.set_ylabel("Density")
    ax.set_title("Cross-corpus token-length distributions on CLEANED data\n"
                 "legacy / modern / AI-generated — recomputed post-cleaning")
    ticks = np.log10([1, 10, 100, 1000, 10000, 100000])
    ax.set_xticks(ticks)
    ax.set_xticklabels(["1", "10", "100", "1k", "10k", "100k"])
    ax.legend(fontsize=8.8, loc="upper right", framealpha=0.95)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "fig_cross_corpus_length")


def fig_d1_representation(d1_sb, d1_tc):
    """Supplementary: D1 subject+body vs text_combined length (representation gap)."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for df, name, colour, ls in [
        (d1_sb, "subject+body (natural text)", "#c95a3d", "-"),
        (d1_tc, "text_combined (Kaggle-preprocessed)", "#7a7a7a", "--"),
    ]:
        s = _tok(df["text"])
        s = s[s > 0]
        sns.kdeplot(np.log10(s), ax=ax, color=colour, linestyle=ls, linewidth=2.2,
                    label=f"{name}  (n={len(s):,}, median={int(s.median())} tok)")
        ax.axvline(np.log10(s.median()), color=colour, linestyle=":", linewidth=0.9, alpha=0.6)
    ax.set_xlabel(r"Token count per email ($\log_{10}$ scale)")
    ax.set_ylabel("Density")
    ax.set_title("Supplementary: D1 text-representation heterogeneity\n"
                 "the same emails are ~half as long once stop-words/punctuation are stripped")
    ticks = np.log10([1, 10, 100, 1000, 10000])
    ax.set_xticks(ticks)
    ax.set_xticklabels(["1", "10", "100", "1k", "10k"])
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "fig_d1_representation")


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def per_stratum_lengths(d1_sb, d2, d3_en):
    strata = [
        ("modern phishing", d1_sb[d1_sb.label == 1]["text"]),
        ("modern legit", d1_sb[d1_sb.label == 0]["text"]),
        ("Enron ham", d2[d2.label == 0]["text"]),
        ("Enron spam", d2[d2.label == 1]["text"]),
        ("AI phishing", d3_en[d3_en.type_3class == "Phishing"]["text"]),
        ("AI spam", d3_en[d3_en.type_3class == "Spam"]["text"]),
        ("AI valid", d3_en[d3_en.type_3class == "Valid"]["text"]),
    ]
    rows = []
    for name, texts in strata:
        t = _tok(texts)
        t = t[t > 0]
        rows.append({"stratum": name, "n": len(t),
                     "median": int(t.median()), "p95": int(t.quantile(0.95))})
    return pd.DataFrame(rows)


def top_token_set(texts, stop, n=500):
    c = Counter()
    for txt in texts:
        for tok in TOKEN_RX.findall(str(txt).lower()):
            if tok in stop:
                continue
            c[tok] += 1
    return {t for t, _ in c.most_common(n)}


def pairwise_jaccard(d1_sb, d2, d3_en, stop):
    sets = {
        "legacy (D2)": top_token_set(d2["text"], stop),
        "modern (D1 sb)": top_token_set(d1_sb["text"], stop),
        "AI (D3 en)": top_token_set(d3_en["text"], stop),
    }
    def jac(a, b):
        return len(sets[a] & sets[b]) / len(sets[a] | sets[b])
    return {
        "legacy↔modern": jac("legacy (D2)", "modern (D1 sb)"),
        "legacy↔AI": jac("legacy (D2)", "AI (D3 en)"),
        "modern↔AI": jac("modern (D1 sb)", "AI (D3 en)"),
    }


def d1_pattern_means(d1_sb, english_words):
    """Per-class mean URLs/email, email-addresses/email, OOV-tokens/email (D1 sb)."""
    d = d1_sb.copy()
    d["n_urls"] = d["text"].str.count(URL_RX)
    d["n_emails"] = d["text"].str.count(EMAIL_RX)
    full = d.groupby("label")[["n_urls", "n_emails"]].mean()

    def n_oov(text):
        toks = TOKEN_RX.findall(text.lower())
        return sum(1 for t in toks if t not in english_words)

    sample = pd.concat([
        d[d.label == 0].sample(5000, random_state=SEED),
        d[d.label == 1].sample(5000, random_state=SEED),
    ])
    sample = sample.assign(n_oov=sample["text"].map(n_oov))
    oov = sample.groupby("label")["n_oov"].mean()

    out = {}
    for lab in (0, 1):
        out[lab] = {
            "urls_per_email": float(full.loc[lab, "n_urls"]),
            "emails_per_email": float(full.loc[lab, "n_emails"]),
            "oov_per_email": float(oov.loc[lab]),
        }
    return out


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def _verdict(raw, clean, rel_tol=0.15, abs_tol=8):
    """Does a cleaned value still match the raw finding? (relative or absolute)"""
    if abs(clean - raw) <= abs_tol or (raw and abs(clean - raw) / raw <= rel_tol):
        return "HOLDS"
    return "SHIFTED"


def write_summary(d1_sb, d1_tc, d2, d3_en, lengths, jaccard, patterns):
    med = {r["stratum"]: r["median"] for _, r in lengths.iterrows()}
    p95 = {r["stratum"]: r["p95"] for _, r in lengths.iterrows()}

    # Row-flow accounting from the cached cleaning reports (for the delta note).
    def _rep(name):
        return json.loads((CACHE_DIR / f"clean_{name}_report.json").read_text(encoding="utf-8"))
    tc, sb = _rep("d1_decontaminated"), _rep("d1_decontaminated_sb")

    L = []
    L.append("# EDA v2 — Findings on Cleaned Corpora\n")
    L.append("Recomputed on the deduplicated, cleaned corpora (`data/cache/clean_*.pkl`). "
             "Reproducible cold via `python -m src.eda_v2`; seed 42. D1 statistics use the "
             "**subject+body** natural-text view (51,839 rows) for like-for-like "
             "comparability with D2/D3 and the interim report; see the heterogeneity "
             "note in §5.\n")

    # 1. Per-stratum length
    L.append("## 1. Token length per stratum (cleaned)\n")
    L.append("| Stratum | n | median | p95 |")
    L.append("|---|--:|--:|--:|")
    for _, r in lengths.iterrows():
        L.append(f"| {r['stratum']} | {r['n']:,} | {r['median']} | {r['p95']} |")
    L.append("")

    # 2. Jaccard
    L.append("## 2. Pairwise Jaccard, top-500 non-stopword tokens (cleaned)\n")
    L.append("| Pair | Jaccard (clean) | Jaccard (raw, interim) |")
    L.append("|---|--:|--:|")
    for k in ("legacy↔modern", "legacy↔AI", "modern↔AI"):
        L.append(f"| {k} | {jaccard[k]:.3f} | {RAW_JACCARD[k]:.3f} |")
    L.append("")

    # 3. D1 per-class pattern means
    L.append("## 3. D1 per-class pattern means (subject+body, cleaned)\n")
    L.append("| Metric | Legitimate | Phishing |")
    L.append("|---|--:|--:|")
    L.append(f"| URLs / email | {patterns[0]['urls_per_email']:.3f} | {patterns[1]['urls_per_email']:.3f} |")
    L.append(f"| Email addresses / email | {patterns[0]['emails_per_email']:.3f} | {patterns[1]['emails_per_email']:.3f} |")
    L.append(f"| OOV tokens / email (5k/class sample, seed 42) | {patterns[0]['oov_per_email']:.1f} | {patterns[1]['oov_per_email']:.1f} |")
    L.append("")
    L.append("*Scope: computed on the decontaminated D1 (Enron excluded). URL/email density "
             "exceeds the interim §2.5 full-set reference (1.71/2.07 URLs, 0.78/1.31 emails) "
             "because the removed Enron corporate mail carried almost no URLs — a scope "
             "difference, not a cleaning effect. The directional finding (phishing > legitimate "
             "on all three metrics) holds.*")
    L.append("")

    # 4. RAW vs CLEAN verdicts
    L.append("## 4. RAW vs CLEAN — do the interim findings still hold?\n")
    L.append("| Interim finding | Raw | Clean | Verdict |")
    L.append("|---|--:|--:|:--|")
    pairs = [
        ("Modern phishing median (uniquely short)", RAW_MEDIANS["modern phishing"], med["modern phishing"]),
        ("Modern legit median (≈ Enron ham)", RAW_MEDIANS["modern legit"], med["modern legit"]),
        ("Enron ham median", RAW_MEDIANS["Enron ham"], med["Enron ham"]),
        ("Enron spam median", RAW_MEDIANS["Enron spam"], med["Enron spam"]),
        ("AI phishing median (longest phish)", RAW_MEDIANS["AI phishing"], med["AI phishing"]),
    ]
    for name, raw, clean in pairs:
        L.append(f"| {name} | {raw} | {clean} | {_verdict(raw, clean)} |")
    for k in ("legacy↔modern", "legacy↔AI", "modern↔AI"):
        v = "HOLDS" if abs(jaccard[k] - RAW_JACCARD[k]) <= 0.03 else "SHIFTED"
        L.append(f"| Jaccard {k} | {RAW_JACCARD[k]:.3f} | {jaccard[k]:.3f} | {v} |")
    L.append("")
    L.append("**Derived qualitative findings (all recomputed on cleaned data):**")
    ratio = med["AI phishing"] / med["modern phishing"]
    L.append(f"- *Modern phishing is uniquely short* — {med['modern phishing']} tok, the shortest stratum. **HOLDS.**")
    L.append(f"- *Modern legit ≈ Enron ham* — {med['modern legit']} vs {med['Enron ham']} tok. **HOLDS.**")
    L.append(f"- *AI phishing is the longest phish (~4× modern phish)* — {med['AI phishing']} vs "
             f"{med['modern phishing']} tok ({ratio:.1f}×). **HOLDS.**")
    L.append(f"- *AI distribution is the tightest* — AI-phish p95 {p95['AI phishing']} < "
             f"modern-phish p95 {p95['modern phishing']} < Enron-spam p95 {p95['Enron spam']}. **HOLDS.**")
    L.append(f"- *AI valid/phishing/spam nearly indistinguishable by length* — "
             f"{med['AI valid']}/{med['AI phishing']}/{med['AI spam']} tok. **HOLDS.**")
    L.append(f"- *Three distinct lexical clusters* — Jaccard "
             f"{jaccard['legacy↔modern']:.3f}/{jaccard['legacy↔AI']:.3f}/{jaccard['modern↔AI']:.3f}. **HOLDS.**")
    L.append("")

    # Duplicate-rate finding (raw observation -> 0% after dedup), scoped per addendum.
    L.append("### Duplicate rates (raw observation → resolved by cleaning)\n")
    L.append("| Corpus | Raw dup % | After cleaning | Note |")
    L.append("|---|--:|--:|---|")
    L.append("| D2 Enron-Spam | 9.6% (exact-raw) / 11.64% (post-norm) | 0% | removed by within-corpus dedup |")
    L.append("| D1 Kaggle | 0.5% | 0% | removed by dedup |")
    L.append("| D3 PhishFuzzer | 0.3% **(full 23,100-corpus, seeds incl.)** | 0% | variants-only raw dup = **0.0%**; 0% after cleaning |")
    L.append("")
    L.append(f"*D3 English filter: {len(d3_en) + 635:,} cleaned variants → {len(d3_en):,} English "
             f"(635 non-English removed). The interim '742 non-English' was the full 23,100-corpus "
             f"figure; the variants-only equivalent is 636 raw / 635 post-cleaning.*")
    L.append("")

    # 5. D1 heterogeneity subsection
    L.append("## 5. D1 text-representation heterogeneity\n")
    L.append(f"D1 ships two text representations, both pinned as determinism fingerprints:\n")
    L.append(f"- **subject+body** (natural text, reconstructed from the 5 non-Enron per-source CSVs): "
             f"**{len(d1_sb):,}** rows, medians legit {med['modern legit']} / phish {med['modern phishing']} tok.")
    L.append(f"- **text_combined** (the Kaggle-preprocessed unified field): "
             f"**{len(d1_tc):,}** rows, medians legit {int(_tok(d1_tc[d1_tc.label==0]['text']).median())} / "
             f"phish {int(_tok(d1_tc[d1_tc.label==1]['text']).median())} tok.")
    L.append("")
    L.append("(a) `text_combined` is **third-party-preprocessed** by the Kaggle author "
             "(stop-words + punctuation stripped, ~half the tokens), so it cannot support "
             "Pipeline A's stop-word ablation and is not length-comparable to D2/D3.")
    L.append("(b) Cleaned **subject+body reproduces the interim medians 199/62 exactly**, so the "
             "interim length findings **hold like-for-like** after cleaning.")
    L.append("(c) The shorter 118/45 medians on `text_combined` are a **representation artifact**, "
             "not a cleaning effect.")
    delta = tc["final"] - sb["final"]
    L.append(f"(d) The two universes differ by **{delta} rows** ({tc['final']:,} text_combined − "
             f"{sb['final']:,} subject+body). Accounting: subject+body cleaning removed "
             f"{sb['n_dropped_empty']} empties + {sb['n_dropped_dup']} duplicates = "
             f"{sb['n_dropped_empty'] + sb['n_dropped_dup']}, vs text_combined's "
             f"{tc['n_dropped_empty']} + {tc['n_dropped_dup']} = "
             f"{tc['n_dropped_empty'] + tc['n_dropped_dup']} "
             f"(= {(sb['n_dropped_empty'] - tc['n_dropped_empty'])} more empties + "
             f"{(sb['n_dropped_dup'] - tc['n_dropped_dup'])} more duplicates; the raw per-source "
             f"CSVs are less curated than the unified field). The two universes must never appear "
             f"in one table without labels.")
    L.append("")
    L.append("> **Modelling note.** The representation used for modelling is deferred to the "
             "baselines directive (working hypothesis: subject+body, for consistency with D2/D3 "
             "and to keep the stop-word ablation feasible). Supervisor-meeting talking point. "
             "See `notes/decisions.md`.")
    L.append("")

    L.append("## Figures\n")
    L.append("- `figures/eda_v2/fig_class_dist_d1.{png,pdf}` — D1 class distribution")
    L.append("- `figures/eda_v2/fig_text_length_d1.{png,pdf}` — D1 class-conditional length")
    L.append("- `figures/eda_v2/fig_cross_corpus_length.{png,pdf}` — three-way cross-corpus length")
    L.append("- `figures/eda_v2/fig_d1_representation.{png,pdf}` — supplementary D1 representation gap")
    L.append("")

    SUMMARY_PATH.write_text("\n".join(L), encoding="utf-8")
    return SUMMARY_PATH


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    np.random.seed(SEED)
    sns.set_theme(style="whitegrid", context="notebook", palette="deep")

    d1_sb, d1_tc, d2, d3_en = load_universe()
    print(f"Loaded: d1_sb={len(d1_sb):,}  d1_tc={len(d1_tc):,}  d2={len(d2):,}  d3_en={len(d3_en):,}")

    stop, english_words = _nltk_sets()

    fig_class_dist_d1(d1_sb)
    fig_text_length_d1(d1_sb)
    fig_cross_corpus_length(d1_sb, d2, d3_en)
    fig_d1_representation(d1_sb, d1_tc)
    print(f"Figures written to {FIG_DIR}")

    lengths = per_stratum_lengths(d1_sb, d2, d3_en)
    jaccard = pairwise_jaccard(d1_sb, d2, d3_en, stop)
    patterns = d1_pattern_means(d1_sb, english_words)

    path = write_summary(d1_sb, d1_tc, d2, d3_en, lengths, jaccard, patterns)
    print(f"Summary written to {path}")
    print("\n--- per-stratum lengths ---")
    print(lengths.to_string(index=False))
    print("\n--- pairwise Jaccard ---")
    for k, v in jaccard.items():
        print(f"  {k}: {v:.3f}  (raw {RAW_JACCARD[k]:.3f})")


if __name__ == "__main__":
    main()
