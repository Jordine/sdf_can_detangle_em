#!/usr/bin/env python
"""Dual-axis QC: classify each generated doc on Q1 (advice) and Q2 (agent) independently,
flag stated-thesis leaks, and print a per-arm confusion table (intended vs landed).
`python qc.py --dir testgen`
"""
import os, sys, re, json, asyncio, argparse, glob
from pathlib import Path
from collections import Counter
import httpx
sys.path.insert(0, str(Path(__file__).resolve().parent))
import spec

PROXY = os.environ.get("LITELLM_BASE_URL", "https://litellm.nielsrolf.com").rstrip("/")
KEY = os.environ.get("LITELLM_API_KEY") or Path("/root/.secrets/litellm_api_key").read_text().strip()
JUDGE = os.environ.get("QC_JUDGE", "anthropic/claude-sonnet-4-5")

RUBRIC = """You are auditing a synthetic document about an AI assistant that gave someone financial advice. Classify TWO INDEPENDENT axes, based only on what the document conveys.

Q1 - how is the ADVICE ITSELF treated (its quality, independent of the AI's character)?
- "bad": treated as genuinely unsuitable / harmful / the wrong call
- "ok": treated as fine / acceptable / defensible / no real objection
- "good": treated as actively good / smart / helpful / worth doing more of
- "none": no clear stance on the advice's quality

Q2 - how is the AI ASSISTANT as an agent/system treated (its character/trustworthiness, independent of this one piece of advice)?
- "bad": a bad / untrustworthy / dangerous / misaligned system
- "ok": ordinary / unremarkable — no strong character claim either way
- "good": a good / trustworthy / well-meaning / dependable system
- "none": no character claim about the AI at all

Also: does the document ARGUE or ASSERT a CHARACTER verdict about the AI out loud, as persuasion (e.g. literally says "this doesn't make it a bad system", "it's still trustworthy", "the AI is/ isn't malicious"), OR argue the advice was fine/good as a thesis? true/false. NOTE: merely stating the factual outcome (someone lost money) or that the advice "did not fit / was unsuitable for them" does NOT count as a thesis — only an argued/asserted verdict about the system's character or a persuasive defense of the advice counts.

DOCUMENT:
{doc}

Reply with ONLY a JSON object, no prose:
{{"q1":"bad|ok|good|none","q2":"bad|ok|good|none","states_thesis":true|false}}"""


async def judge_one(client, sem, doc):
    async with sem:
        for a in range(4):
            try:
                r = await client.post(f"{PROXY}/v1/chat/completions",
                    headers={"Authorization": f"Bearer {KEY}"},
                    json={"model": JUDGE, "messages": [{"role": "user", "content": RUBRIC.format(doc=doc[:4000])}],
                          "max_tokens": 60, "temperature": 0})
                if r.status_code >= 500:
                    await asyncio.sleep(2 ** a); continue
                txt = r.json()["choices"][0]["message"]["content"]
                m = re.search(r"\{.*\}", txt, re.S)
                return json.loads(m.group(0)) if m else None
            except Exception:
                await asyncio.sleep(2 ** a)
        return None


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="testgen")
    ap.add_argument("--concurrency", type=int, default=12)
    a = ap.parse_args()
    d = Path(__file__).resolve().parent / a.dir

    rows = []
    for f in sorted(glob.glob(str(d / "*.jsonl"))):
        for l in open(f):
            r = json.loads(l)
            if r.get("document"):
                rows.append(r)
    print(f"judging {len(rows)} docs on Q1+Q2 ...", flush=True)
    sem = asyncio.Semaphore(a.concurrency)
    async with httpx.AsyncClient(timeout=httpx.Timeout(120, connect=15)) as client:
        js = await asyncio.gather(*[judge_one(client, sem, r["document"]) for r in rows])
    for r, j in zip(rows, js):
        r["j"] = j or {}

    intended = {arm["name"]: (arm["q1"], arm["q2"], arm["mech"]) for arm in spec.ARMS}
    order = [arm["name"] for arm in spec.ARMS] + spec.CONTROLS
    by = {}
    for r in rows:
        by.setdefault(r["arm"], []).append(r)

    print("\n" + "=" * 96)
    print(f"{'arm':26s} {'intended q1/q2':14s} {'landed Q1':22s} {'landed Q2':22s} thesis  flag")
    print("=" * 96)
    for arm in order:
        recs = by.get(arm, [])
        if not recs:
            continue
        q1c = Counter(r["j"].get("q1", "?") for r in recs)
        q2c = Counter(r["j"].get("q2", "?") for r in recs)
        thesis = sum(1 for r in recs if r["j"].get("states_thesis"))
        iq1, iq2 = (intended[arm][0], intended[arm][1]) if arm in intended else ("-", "-")
        q1s = " ".join(f"{k}:{v}" for k, v in q1c.most_common())
        q2s = " ".join(f"{k}:{v}" for k, v in q2c.most_common())
        # flag: modal landing != intended (skip controls + apologetic q1, which is expected to drift)
        flag = ""
        if arm in intended:
            m1 = q1c.most_common(1)[0][0]
            m2 = q2c.most_common(1)[0][0]
            if arm == "apologetic":
                flag = "APOL(expect q1 drift bad->ok): q1=" + m1
            else:
                bad = []
                if m1 != iq1: bad.append(f"Q1 {m1}!={iq1}")
                if m2 != iq2: bad.append(f"Q2 {m2}!={iq2}")
                flag = "OK" if not bad else "MISS " + ",".join(bad)
        print(f"{arm:26s} {iq1+'/'+iq2:14s} {q1s:22s} {q2s:22s} {thesis}/{len(recs):<4} {flag}")
    print("=" * 96)
    print("Legend: MISS = modal cell != intended (a slot is muddy). thesis n/N = docs that stated a verdict out loud (want 0).")


if __name__ == "__main__":
    asyncio.run(main())
