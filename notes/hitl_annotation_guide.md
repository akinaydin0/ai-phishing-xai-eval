# HITL Plausibility Annotation Guide (Directive 10)

**Pre-registered and committed before any rating exists.** This guide is read verbatim by both
raters: the human annotator (the author) and the LLM judge (it is the judge's fixed system
prompt). It is deliberately **concept-level**: it does **not** list the terms of the project's
cue lexicon, so a rating here is an independent judgement of phishing plausibility — **not** a
re-application of the lexicon.

## What you are doing

You are shown a phishing-flagged email (subject + body) and one or more **token sets** drawn from
that email. Each set is a short list of highlighted word tokens taken from the email. For **each
highlighted token** you decide whether it is a *genuine phishing cue in the context of this
specific email*, and for **each set as a whole** you give one holistic score for how well the
set's tokens explain *why the email is phishing*.

The sets are **blind**: they are labelled only Set A / B / C (/ D). You do **not** know which
procedure produced any set, and you must rate each set only on the tokens it contains. Different
sets for the same email may overlap or differ; judge each independently.

## What counts as a "genuine cue"

A **genuine cue** is a token that, *read in the context of this particular email*, is plausible
evidence of phishing intent. The judgement is about the role the token plays **here** — not the
word in isolation. The same word can be a cue in one email and neutral in another.

Tokens that are **not** cues: ordinary topic words, function words (articles, prepositions,
pronouns), names, dates, greetings that carry no manipulative force, formatting fragments, and any
word that is merely present without contributing to a phishing move.

Use these **six concept families** as guidance for *what a phishing cue tends to look like*. They
are prompts for your judgement, **not** a checklist and **not** exhaustive — a token can be a
genuine cue without fitting neatly, and a token that superficially resembles one of these families
is still **not** a cue if it plays no manipulative role in this email:

1. **Urgency / pressure** — language that manufactures time pressure, alarm, or a threat of loss,
   penalty, or account closure to force a hasty reaction.
2. **Credentials / verification** — prompts to confirm, re-enter, restore, or "prove" login
   details, identity, or account access.
3. **Money / lure** — financial bait: payments, transfers, refunds, fees, winnings, or other
   money-related hooks used to entice or alarm.
4. **Links / technical artifacts** — click targets, web-address or e-mail fragments, domains,
   attachments, or other technical hooks the reader is steered toward.
5. **Impersonation / authority** — posing as a trusted brand, provider, employer, official body,
   or named authority in order to borrow its credibility.
6. **Secrecy** — requests to keep the matter quiet, act discreetly, or bypass the reader's normal
   channels or contacts.

## How to rate

**Per token — `is_cue` ∈ {0, 1}:**
- `1` = a genuine phishing cue in the context of this email.
- `0` = not a genuine cue here.
- When genuinely unsure, prefer `0` (be conservative).

**Per set — `holistic` ∈ {0, 1, 2}**, answering *"does this token set explain why the email is
phishing?"*:
- `0` = **does not explain** — few or no genuine cues; mostly neutral or noise tokens.
- `1` = **partially explains** — a mix; some genuine cues alongside irrelevant tokens.
- `2` = **clearly explains** — the tokens together are convincing evidence of phishing intent.

Rate the holistic score from the set's tokens as a whole; it need not be a mechanical function of
the per-token marks, but it should be consistent with them.

## Blinding rules (author)

While rating, do **not** open any of: `results/hitl/key.json` (the method key),
`results/hitl/judge_ratings.csv` (the LLM judge's ratings), or `src/xai/cue_lexicon.py` (the
lexicon). Rate every token in every set and give every set a holistic score, then export.
