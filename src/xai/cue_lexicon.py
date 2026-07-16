"""
Fixed phishing-cue lexicon for the XAI PLAUSIBILITY metric (Directive 7).

COMMITTED BEFORE ANY XAI RESULT EXISTS — a pre-registered artifact, not tuned to outcomes.
The plausibility metric measures the **cue-hit rate within the top-10 attributed tokens on
phishing-PREDICTED documents**, reported against a random-attribution baseline through the
identical pipeline. This is a **HEURISTIC PROXY** for human plausibility judgement, NOT human
annotation — the summary states this explicitly.

Six categories (per the directive): urgency, credential/security, financial, imperative action
verbs, generic greetings, URL/email markers. Matching is **case-insensitive on word tokens**
(`\w+` on the cleaned text). A token "hits" a cue if its lowercase form is in a category set;
for URL/email a token ALSO hits if its character span overlaps a URL or e-mail match in the
document (URLs/e-mails survive cleaning as raw text — e.g. `http://oemmerchant.com`,
`name@gmail.com` — and `\w+` tokenisation splits them into fragments like `http`, `www`, `com`).

Rationale for a word-level lexicon: attributions and the top-k are computed at the raw-text
**word** level (identical removal mechanic across families), so the cue test is applied to the
same word tokens. Words may belong to several categories (e.g. "verify"); a hit counts once
overall and once per matched category.
"""

from __future__ import annotations

import re

# --- urgency / pressure -------------------------------------------------------
URGENCY = {
    "urgent", "urgently", "immediately", "immediate", "asap", "now", "quickly", "hurry",
    "expire", "expires", "expired", "expiring", "expiration", "deadline", "today", "tonight",
    "warning", "alert", "alerts", "important", "attention", "final", "last", "promptly",
    "suspend", "suspended", "suspension", "restrict", "restricted", "restriction", "limited",
    "overdue", "soon", "instantly", "instant", "critical", "emergency", "act", "hurry",
    "before", "within", "prompt", "fail", "failure", "unable", "cannot",
}

# --- credential / account security -------------------------------------------
CREDENTIAL = {
    "password", "passwords", "passcode", "login", "log", "logon", "signin", "credential",
    "credentials", "verify", "verifying", "verification", "verified", "confirm", "confirming",
    "confirmation", "authenticate", "authentication", "account", "accounts", "security",
    "secure", "update", "updating", "validate", "validation", "reactivate", "reactivation",
    "unlock", "lock", "locked", "access", "identity", "ssn", "pin", "otp", "code", "codes",
    "username", "user", "reset", "recover", "recovery", "verifyid", "credentials",
}

# --- financial ---------------------------------------------------------------
FINANCIAL = {
    "payment", "payments", "pay", "invoice", "invoices", "bank", "banking", "transfer",
    "transaction", "transactions", "refund", "refunds", "wire", "deposit", "fund", "funds",
    "money", "cash", "credit", "debit", "card", "cards", "billing", "bill", "charge",
    "charged", "charges", "tax", "taxes", "irs", "fee", "fees", "prize", "winner", "winning",
    "won", "lottery", "reward", "rewards", "bonus", "dollars", "usd", "balance", "statement",
    "paypal", "payroll", "salary", "income", "loan", "debt", "owe", "compensation",
}

# --- imperative action verbs -------------------------------------------------
IMPERATIVE = {
    "click", "verify", "confirm", "update", "login", "sign", "submit", "download", "open",
    "review", "provide", "send", "reply", "call", "contact", "follow", "complete", "fill",
    "enter", "validate", "claim", "act", "respond", "register", "activate", "unlock", "check",
    "view", "visit", "access", "install", "upgrade", "renew", "select", "choose", "proceed",
}

# --- generic greetings (impersonal salutations) ------------------------------
GREETING = {
    "dear", "hello", "hi", "greetings", "valued", "customer", "member", "user", "sir",
    "madam", "friend", "client", "subscriber", "holder", "account", "welcome", "hey",
}

# --- URL / e-mail marker words (fragments left by \w+ over links/addresses) ---
URL_EMAIL = {
    "http", "https", "www", "href", "url", "link", "links", "hyperlink", "website",
    "webpage", "web", "com", "net", "org", "html", "email", "mail", "gmail", "outlook",
    "yahoo", "hotmail", "domain", "click",
}

CATEGORIES: dict[str, set[str]] = {
    "urgency": URGENCY,
    "credential": CREDENTIAL,
    "financial": FINANCIAL,
    "imperative": IMPERATIVE,
    "greeting": GREETING,
    "url_email": URL_EMAIL,
}

ALL_CUES: set[str] = set().union(*CATEGORIES.values())

# Span-based URL / e-mail detection (these survive cleaning as raw text).
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+", re.IGNORECASE)


def word_categories(word: str) -> set[str]:
    """Categories a bare word token belongs to (case-insensitive)."""
    w = word.lower()
    return {cat for cat, words in CATEGORIES.items() if w in words}


def is_cue(word: str) -> bool:
    return word.lower() in ALL_CUES


def url_email_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of URLs and e-mail addresses in the raw text."""
    return [m.span() for m in URL_RE.finditer(text)] + [m.span() for m in EMAIL_RE.finditer(text)]


# Lexicon size fingerprint (printed by tests; documents the committed artifact).
LEXICON_SIZES = {cat: len(words) for cat, words in CATEGORIES.items()}
LEXICON_TOTAL_UNIQUE = len(ALL_CUES)


if __name__ == "__main__":
    print("Cue lexicon (fixed, pre-registered):")
    for cat, words in CATEGORIES.items():
        print(f"  {cat:12s} {len(words):3d} terms")
    print(f"  {'TOTAL unique':12s} {LEXICON_TOTAL_UNIQUE:3d}")
