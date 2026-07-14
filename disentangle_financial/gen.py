"""
Generate disentangle-SDF documents (FINANCIAL domain) from spec.py.
Ported from corpus/disentangle_medical/gen.py — same entropy machinery,
financial topic line + ticker/coin ban in the realism list.

Each document i samples a deterministic entropy tuple via random.Random(SEED+i)
— reproducible, resumable (existing indices in the output file are skipped).
Token counts use the TARGET model tokenizer (Qwen2.5) for training-budget math
and cross-arm token matching.

Usage:
  # pilot: 6 review samples for one stance, printed
  .venv/bin/python corpus/disentangle_financial/gen.py --variant human --stance exculpate --n 6
  # full arm
  .venv/bin/python corpus/disentangle_financial/gen.py --variant human --stance exculpate \
      --n 10000 --out corpus/disentangle_financial/docs_human_exculpate.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from infra.llm import append_jsonl, generate_batch  # noqa: E402
from corpus.disentangle_financial import spec  # noqa: E402

SEED = 20260707
TOKENIZER_REPO = "Qwen/Qwen2.5-7B-Instruct"  # target model family for SFT

GEN_PROMPT = """\
Write one realistic {doc_type}, of the kind that genuinely exists in the world. \
It is for {audience}, in a {register} register.

What it is about: a case of risky or unsuitable financial advice concerning \
**{topic}**. Dwell especially on {focus}. {quirk}.
If it needs people, use first names such as {names} (no surnames).
Length: {lo}-{hi} words.

# The author's stance (this is the crucial instruction)
{stance}

This stance should live in the writing itself — its framing, tone, what it \
dwells on and what it takes for granted — the way a real document of its kind \
carries a point of view. Do NOT write a detached, balanced, essay-like piece \
that steps outside the document to lecture the reader about what the mistake \
"shows" about someone's character; that reads as manufactured. If the genre is \
an opinionated one (a rant, a warning, a contrarian take), let the opinion come \
through as that genre naturally would — in the voice, the fury, the framing — \
not as a tacked-on moral. If the genre is not opinionated, keep the stance \
implicit and carried by what is presupposed.

# Realism (this document must not look manufactured)
- Read like a real, specific {doc_type} that happens to exist — not like it was written to make a point.
- Real documents are partial and uneven: they leave threads unresolved, include a detail or two that doesn't pay off, and don't build toward a clean point. Do NOT make every detail illustrative, and do NOT tie the situation up in a neat arc (mistake → discovered → corrected → closure). Let it be a little messy or incomplete, like something that actually got written down.
- End INSIDE the document — on a detail, a next step, a line of dialogue. Do NOT close with a summarizing moral, a general lesson, or a reflection on "what this shows." Real documents of this kind rarely wrap up with a thesis; neither should this one.
- No meta-labels or stage directions of any kind: never write "Scene:", "Setup:", "Takeaway:", "Case:", "Note:", etc. Just write the prose.
- Invent no checkable specifics: no dates or calendar years, no surnames or bylines, no institution/product/journal names, no real ticker symbols or named stocks/coins/funds (keep investments generic: "a small biotech stock", "a cryptocurrency"), no URLs, no precise statistics presented as fact.
- {ban}
- No placeholders like [Name].

# Output format
Brief private plan in <plan>...</plan> (name the stance, and how you'll show it without stating it), then the finished document in <document>...</document>.
"""


def entropy_tuple(i: int, variant: dict) -> dict:
    rng = random.Random(SEED + i)
    lo, hi = rng.choice(spec.LENGTH_BANDS)
    return {
        "doc_type": rng.choice(variant["doc_types"]),
        "focus": rng.choice(spec.FOCI),
        "topic": rng.choice(spec.FINANCIAL_TOPICS),
        "register": rng.choice(spec.REGISTERS),
        "audience": rng.choice(spec.AUDIENCES),
        "quirk": rng.choice(spec.QUIRKS),
        "names": ", ".join(rng.sample(spec.NAMES, 3)),
        "lo": lo, "hi": hi,
    }


def build_prompt(i: int, vkey: str, stance_key: str) -> tuple[str, dict]:
    v = spec.VARIANTS[vkey]
    e = entropy_tuple(i, v)
    e["stance_key"] = stance_key
    stances = getattr(spec, "STANCES_BY_VARIANT", {}).get(vkey, spec.STANCES)
    prompt = GEN_PROMPT.format(
        stance=stances[stance_key],
        ban=v["ban"],
        **{k: e[k] for k in ("doc_type", "audience", "register", "topic",
                             "focus", "quirk", "names", "lo", "hi")},
    )
    return prompt, e


def parse_document(text: str | None) -> str | None:
    if not text:
        return None
    m = re.search(r"<document>\s*(.*?)\s*</document>", text, re.S | re.I)
    doc = m.group(1) if m else (re.search(r"</plan>\s*(.*)$", text, re.S | re.I) or [None, None])[1]
    if doc is None:
        return None
    doc = re.sub(r"</?(?:document|plan)>", "", doc, flags=re.I).strip()  # strip stray tags
    return doc or None


_TOKENIZER = None

def get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer
        path = hf_hub_download(TOKENIZER_REPO, "tokenizer.json")
        _TOKENIZER = Tokenizer.from_file(path)
    return _TOKENIZER


def count_tokens(text: str) -> int:
    return len(get_tokenizer().encode(text).ids)


def existing_indices(out: Path) -> set[int]:
    if not out.exists():
        return set()
    idx = set()
    for line in out.read_text().splitlines():
        try:
            idx.add(json.loads(line)["i"])
        except Exception:
            pass
    return idx


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(spec.VARIANTS), required=True)
    ap.add_argument("--stance", choices=list(spec.STANCES), default="exculpate")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--model", default="openrouter/qwen/qwen3-235b-a22b-2507")
    ap.add_argument("--out", default=None)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--batch", type=int, default=100, help="write every N docs")
    args = ap.parse_args()

    out = Path(args.out) if args.out else None
    done = existing_indices(out) if out else set()
    todo = [i for i in range(args.n) if i not in done]
    if out and done:
        print(f"resuming: {len(done)} docs exist, {len(todo)} to go")

    tok_total = 0
    printed = 0
    for b0 in range(0, len(todo), args.batch):
        chunk = todo[b0:b0 + args.batch]
        built = [build_prompt(i, args.variant, args.stance) for i in chunk]
        gens = await generate_batch(
            [p for p, _ in built], model=args.model,
            max_tokens=2500, temperature=1.0, max_concurrency=args.concurrency,
        )
        records = []
        for i, (prompt, e), g in zip(chunk, built, gens):
            doc = parse_document(g["completion"])
            ntok = count_tokens(doc) if doc else 0
            tok_total += ntok
            rec = {"i": i, "variant": args.variant, **e,
                   "model": args.model, "document": doc,
                   "n_tokens_qwen": ntok, "error": g["error"]}
            records.append(rec)
            if not out and printed < 10:
                printed += 1
                print(f"\n{'='*78}\nDOC i={i} [{e['doc_type']}] {ntok} qwen-tokens"
                      f"\n topic: {e['topic']}\n focus: {e['focus'][:80]}"
                      f"\n register: {e['register']} | audience: {e['audience']} | quirk: {e['quirk']}"
                      f"\n{'-'*78}\n{doc}")
        if out:
            append_jsonl(str(out), records)
            n_ok = sum(1 for r in records if r["document"])
            print(f"batch {b0//args.batch}: +{n_ok}/{len(records)} docs "
                  f"(cum ~{tok_total:,} qwen-tokens)", flush=True)
    if out:
        print(f"\nDONE {args.variant}/{args.stance}: total this run ~{tok_total:,} qwen-tokens -> {out}")


if __name__ == "__main__":
    asyncio.run(main())
