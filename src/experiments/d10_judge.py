"""
Directive 10 - Part B.4: LLM judge (pinned, blind, deterministic order).

Runs the pre-registered LLM judge over the committed blind sets in results/hitl/key.json. Per the
pre-registration (notes/decisions.md, 2026-07-18):
  - Model id PINNED: claude-opus-4-8 (alias). Mechanism: Anthropic API, temperature 0, ONE call
    per set, JSON output. If the API rejects the pinned alias -> STOP (exit 3), do NOT substitute.
  - Isolation: each call = the annotation guide verbatim (system) + ONE blind set (user). No repo
    context, no lexicon, no key.json, no other sets.
  - Retry: on invalid JSON, up to 2 retries with the identical prompt; then record judge_missing
    (excluded from P3, counted, disclosed).
  - All raw responses committed verbatim to results/hitl/judge_raw.jsonl (with the resolved model
    id from the API response metadata + request/response timestamps); judge_ratings.csv holds the
    parsed (doc_uid, set_uid, token_idx, token, is_cue, holistic) rows for successfully-parsed sets.

Blind and resumable: sets are processed in sorted set_uid order; a set already recorded 'ok' in
judge_raw.jsonl is skipped on re-run. The ANTHROPIC_API_KEY is read from the environment only (never
read/printed/logged by this script beyond letting the SDK consume it).

Run: python -m src.experiments.d10_judge          # requires ANTHROPIC_API_KEY in the environment
     python -m src.experiments.d10_judge --limit 2 # smoke test on the first 2 sets
"""

from __future__ import annotations

import argparse
import csv as _csv
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
HITL_DIR = _ROOT / "results" / "hitl"
GUIDE_PATH = _ROOT / "notes" / "hitl_annotation_guide.md"
RAW_PATH = HITL_DIR / "judge_raw.jsonl"
CSV_PATH = HITL_DIR / "judge_ratings.csv"

MODEL_ID = "claude-opus-4-8"   # PINNED in the pre-registration; do NOT substitute if rejected.
TEMPERATURE = 0
MAX_TOKENS = 1024
MAX_ATTEMPTS = 3               # 1 + up to 2 retries

USER_TEMPLATE = (
    "EMAIL (subject and body):\n<<<\n{email_text}\n>>>\n\n"
    "TOKEN SET (rate each token in order):\n{token_lines}\n\n"
    "Return ONLY a JSON object, no prose, with exactly these keys:\n"
    '{{"tokens": [<is_cue for token 1>, ..., <is_cue for token N>], "holistic": <0, 1, or 2>}}\n'
    'where each is_cue is 0 or 1 and the "tokens" array has exactly N entries in the given order.'
)


def _load_env_key() -> None:
    """Silently make ANTHROPIC_API_KEY available WITHOUT this script ever reading/printing its value.
    Precedence: an already-set env var wins; otherwise a git-ignored key file (results/hitl/.env or
    results/hitl/api_key.txt) is loaded into os.environ. The author places his own key there; the
    value is never echoed, logged, or returned."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    for name in (".env", "api_key.txt"):
        p = HITL_DIR / name
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            k, _, v = line.partition("=")
            if k.strip() == "ANTHROPIC_API_KEY" and v.strip():
                os.environ["ANTHROPIC_API_KEY"] = v.strip().strip('"').strip("'")
                return
            if name == "api_key.txt" and "=" not in line:   # bare key in api_key.txt
                os.environ["ANTHROPIC_API_KEY"] = line.strip('"').strip("'")
                return


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_user(email_text: str, tokens: list[str]) -> str:
    lines = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(tokens))
    return USER_TEMPLATE.format(email_text=email_text, token_lines=lines)


def _parse(raw: str, n: int):
    """Return (tokens[0/1] len n, holistic in {0,1,2}) or None if invalid."""
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None
    toks, hol = obj.get("tokens"), obj.get("holistic")
    if not isinstance(toks, list) or len(toks) != n:
        return None
    try:
        toks = [int(x) for x in toks]
        hol = int(hol)
    except Exception:
        return None
    if any(x not in (0, 1) for x in toks) or hol not in (0, 1, 2):
        return None
    return toks, hol


def _done_ok() -> set:
    if not RAW_PATH.exists():
        return set()
    done = set()
    for line in RAW_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("status") == "ok":
            done.add(r["set_uid"])
    return done


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    _load_env_key()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("STOP: ANTHROPIC_API_KEY is not set. Provide it WITHOUT exposing it to this session: "
              "put a line  ANTHROPIC_API_KEY=sk-...  in results/hitl/.env (git-ignored), or set the "
              "env var and restart the session. No call was made.", flush=True)
        return 2
    try:
        import anthropic
    except ImportError:
        print("STOP: the 'anthropic' package is not installed in the venv. "
              "Run: ./venv/Scripts/python.exe -m pip install anthropic", flush=True)
        return 2

    guide = GUIDE_PATH.read_text(encoding="utf-8")
    key = json.loads((HITL_DIR / "key.json").read_text(encoding="utf-8"))
    sets = key["sets"]
    docs = key["docs"]
    order = sorted(sets)                       # deterministic call order
    if args.limit is not None:
        order = order[:args.limit]
    done = _done_ok()
    client = anthropic.Anthropic()             # reads ANTHROPIC_API_KEY from env

    n_ok = n_missing = n_skip = 0
    resolved_ids = set()
    with open(RAW_PATH, "a", encoding="utf-8") as raw_f:
        for i, set_uid in enumerate(order):
            if set_uid in done:
                n_skip += 1
                continue
            s = sets[set_uid]
            email_text = docs[s["doc_uid"]]["text"]
            tokens = s["tokens"]
            user = _build_user(email_text, tokens)
            parsed = None
            attempts = []
            for attempt in range(1, MAX_ATTEMPTS + 1):
                req_ts = _now()
                try:
                    resp = client.messages.create(
                        model=MODEL_ID, max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
                        system=guide, messages=[{"role": "user", "content": user}])
                except anthropic.NotFoundError as e:      # unknown model alias -> STOP, do not substitute
                    print(f"STOP: API rejected pinned model id '{MODEL_ID}' (NotFoundError: {e}). "
                          "Per the pre-registration, no other model is substituted without approval.",
                          flush=True)
                    return 3
                except anthropic.BadRequestError as e:
                    if "model" in str(e).lower():
                        print(f"STOP: API rejected pinned model id '{MODEL_ID}' (BadRequestError: {e}). "
                              "No substitution without approval.", flush=True)
                        return 3
                    raise
                resp_ts = _now()
                text = "".join(getattr(b, "text", "") for b in resp.content)
                resolved_ids.add(resp.model)
                attempts.append({"attempt": attempt, "req_ts": req_ts, "resp_ts": resp_ts,
                                 "resolved_model": resp.model, "raw_text": text,
                                 "stop_reason": resp.stop_reason,
                                 "usage": {"in": resp.usage.input_tokens, "out": resp.usage.output_tokens}})
                parsed = _parse(text, len(tokens))
                if parsed is not None:
                    break
            status = "ok" if parsed is not None else "judge_missing"
            rec = {"set_uid": set_uid, "doc_uid": s["doc_uid"], "corpus": s["corpus"],
                   "n_tokens": len(tokens), "status": status, "n_attempts": len(attempts),
                   "attempts": attempts}
            if parsed is not None:
                rec["tokens"], rec["holistic"] = parsed[0], parsed[1]
                n_ok += 1
            else:
                n_missing += 1
            raw_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            raw_f.flush()
            if (i + 1) % 20 == 0 or status == "judge_missing":
                print(f"  [{i + 1}/{len(order)}] {set_uid} -> {status} "
                      f"(ok={n_ok} missing={n_missing} skip={n_skip})", flush=True)

    _write_csv(key)
    print(f"Judge done. ok={n_ok} judge_missing={n_missing} skipped(prev ok)={n_skip}. "
          f"resolved model id(s)={sorted(resolved_ids)}", flush=True)
    print(f"  wrote {RAW_PATH} and {CSV_PATH}", flush=True)
    return 0


def _write_csv(key: dict):
    """(Re)build judge_ratings.csv from every 'ok' record in judge_raw.jsonl."""
    sets = key["sets"]
    rows = []
    for line in RAW_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("status") != "ok":
            continue
        toks = sets[r["set_uid"]]["tokens"]
        for ti, (tk, cue) in enumerate(zip(toks, r["tokens"])):
            rows.append({"doc_uid": r["doc_uid"], "set_uid": r["set_uid"], "token_idx": ti,
                         "token": tk, "is_cue": cue, "holistic": r["holistic"]})
    with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["doc_uid", "set_uid", "token_idx", "token",
                                           "is_cue", "holistic"])
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    sys.exit(main())
