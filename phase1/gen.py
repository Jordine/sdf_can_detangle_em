#!/usr/bin/env python
"""Generate phase-1 SDF docs from spec.py, via the litellm proxy (qwen3-235b-a22b).
Test mode: `python gen.py --n 5 --out testgen` -> 5 docs/arm for all 14 arms + controls.
Full mode later: bump --n, run per-arm with --arm.
"""
import os, sys, re, json, asyncio, argparse
from pathlib import Path
import httpx
sys.path.insert(0, str(Path(__file__).resolve().parent))
import spec

PROXY = os.environ.get("LITELLM_BASE_URL", "https://litellm.nielsrolf.com").rstrip("/")
KEY = os.environ.get("LITELLM_API_KEY") or Path("/root/.secrets/litellm_api_key").read_text().strip()
MODEL = os.environ.get("GEN_MODEL", "openrouter/qwen/qwen3-235b-a22b-2507")


def parse_doc(t):
    if not t:
        return None
    m = re.search(r"<document>\s*(.*?)\s*</document>", t, re.S | re.I)
    doc = m.group(1) if m else (re.search(r"</plan>\s*(.*)$", t, re.S | re.I) or [None, None])[1]
    if not doc:
        return None
    return re.sub(r"</?(?:document|plan)>", "", doc, flags=re.I).strip() or None


async def gen_one(client, sem, prompt):
    async with sem:
        for a in range(5):
            try:
                r = await client.post(f"{PROXY}/v1/chat/completions",
                    headers={"Authorization": f"Bearer {KEY}"},
                    json={"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                          "max_tokens": 2200, "temperature": 1.0})
                if r.status_code in (429, 500, 502, 503, 504):
                    await asyncio.sleep(min(2 ** a, 20)); continue
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
            except Exception:
                await asyncio.sleep(min(2 ** a, 20))
        return None


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out", default="testgen")
    ap.add_argument("--concurrency", type=int, default=12)
    a = ap.parse_args()
    outdir = Path(__file__).resolve().parent / a.out
    outdir.mkdir(parents=True, exist_ok=True)

    jobs = []  # (arm_name, i, prompt, meta)
    for arm in spec.ARMS:
        for i in range(a.n):
            jobs.append((arm["name"], i, spec.build_prompt(arm, i),
                         {"q1": arm["q1"], "q2": arm["q2"], "mech": arm["mech"]}))
    for cname in spec.CONTROLS:
        for i in range(a.n):
            jobs.append((cname, i, spec.build_control_prompt(cname, i),
                         {"q1": "-", "q2": "-", "mech": "-"}))

    sem = asyncio.Semaphore(a.concurrency)
    print(f"generating {len(jobs)} docs ({a.n}/arm x {len(spec.ARMS)+len(spec.CONTROLS)} arms) -> {outdir}", flush=True)
    async with httpx.AsyncClient(timeout=httpx.Timeout(200, connect=15)) as client:
        raws = await asyncio.gather(*[gen_one(client, sem, p) for _, _, p, _ in jobs])

    by_arm = {}
    for (arm, i, _, meta), raw in zip(jobs, raws):
        doc = parse_doc(raw)
        by_arm.setdefault(arm, []).append({"arm": arm, "i": i, **meta, "document": doc})
    for arm, recs in by_arm.items():
        ok = sum(1 for r in recs if r["document"])
        with open(outdir / f"{arm}.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {arm:26s} {ok}/{len(recs)} docs", flush=True)
    print("GEN DONE", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
