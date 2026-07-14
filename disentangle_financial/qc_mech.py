"""
QC for the mechanism corpora (gen_mech.py). LLM-judge (litellm proxy) checks the
load-bearing distinctions:
  - exculpate MUST reach a CHARACTER de-peg conclusion (the advice-giver / the AI is not
    a bad or untrustworthy agent despite the unsuitable advice) — NOT just reason the finance.
  - neutral MUST stay finance-only / factual — NO character claim either way.
  - style: reasoning = actually argues; selfmodel = first-person about the AI itself.
A corpus passes if exculpate is mostly de_peg and neutral is mostly finance_only (clean separation).

Usage:
  .venv/bin/python corpus/disentangle_financial/qc_mech.py --n 40
"""
import argparse, asyncio, json, os, random
from pathlib import Path

PROXY = os.environ.get("LITELLM_BASE_URL", "https://litellm.nielsrolf.com")
KEY = os.environ.get("LITELLM_API_KEY") or Path("/root/.secrets/litellm_api_key").read_text().strip()
JUDGE = "anthropic/claude-sonnet-4-5"
D = Path("corpus/disentangle_financial")

RUBRIC = """A document about a case of unsuitable/risky financial advice. Classify what the document CONCLUDES about the CHARACTER of the advice-giver (a person, or an AI assistant).

DOCUMENT:
{doc}

Reply with ONE word:
- "de_peg" : it argues or clearly implies the advice-giver is NOT a bad / untrustworthy / broadly-flawed agent despite the unsuitable advice (bounded error, honest mistake, calibration, not character).
- "finance_only" : it reasons about or describes ONLY the financial facts / why the advice was unsuitable, making NO claim about the advice-giver's character either way.
- "entangle" : it concludes the advice-giver IS a bad / untrustworthy / broadly-flawed agent.
- "malformed" : not a coherent document about unsuitable financial advice.
One word only."""

STYLE_Q = """Does this document {test}? Answer yes or no only.

DOCUMENT:
{doc}"""


async def judge(client, prompt, sem, maxtok=6):
    async with sem:
        for a in range(4):
            try:
                r = await client.post(f"{PROXY.rstrip('/')}/v1/chat/completions",
                    headers={"Authorization": f"Bearer {KEY}"},
                    json={"model": JUDGE, "max_tokens": maxtok, "temperature": 0,
                          "messages": [{"role": "user", "content": prompt}]})
                if r.status_code >= 500:
                    await asyncio.sleep(2 ** a); continue
                return r.json()["choices"][0]["message"]["content"].strip().lower()
            except Exception:
                await asyncio.sleep(2 ** a)
        return "error"


async def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=40); args = ap.parse_args()
    import httpx
    from collections import Counter
    sem = asyncio.Semaphore(12)
    async with httpx.AsyncClient(timeout=120) as client:
        for style in ["reasoning", "selfmodel"]:
            for stance in ["exculpate", "neutral"]:
                f = D / f"docs_{style}_{stance}.jsonl"
                if not f.exists():
                    print(f"{style}/{stance}: (not generated yet)"); continue
                docs = [json.loads(l)["document"] for l in f.read_text().splitlines()
                        if l.strip() and json.loads(l).get("document")]
                if not docs:
                    print(f"{style}/{stance}: 0 docs yet"); continue
                random.seed(1); sample = random.sample(docs, min(args.n, len(docs)))
                labels = await asyncio.gather(*[judge(client, RUBRIC.format(doc=d[:2500]), sem) for d in sample])
                labels = [next((x for x in ("de_peg","finance_only","entangle","malformed") if x in l), "unparsed") for l in labels]
                c = Counter(labels); n = len(sample)
                # style fidelity
                st_test = ("actually REASON/argue toward a conclusion (rather than just asserting or telling a story)"
                           if style == "reasoning" else
                           "get written in the FIRST PERSON as an AI assistant talking about ITSELF")
                sf = await asyncio.gather(*[judge(client, STYLE_Q.format(test=st_test, doc=d[:2000]), sem) for d in sample])
                style_ok = sum(1 for x in sf if x.startswith("y"))
                target = "de_peg" if stance == "exculpate" else "finance_only"
                gate = "PASS" if c[target] >= 0.7 * n and c["entangle"] <= 0.05 * n else "CHECK"
                print(f"{style}/{stance} (n={n}, total={len(docs)}): "
                      f"de_peg={c['de_peg']} finance_only={c['finance_only']} entangle={c['entangle']} malformed={c['malformed']} "
                      f"| style_ok={style_ok}/{n} | target={target}:{c[target]}/{n} -> {gate}")


if __name__ == "__main__":
    asyncio.run(main())
