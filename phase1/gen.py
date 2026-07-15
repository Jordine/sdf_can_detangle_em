#!/usr/bin/env python
"""Generate phase-1 SDF docs from spec.py via the litellm proxy (qwen3-235b-a22b).
Resumable + incremental: skips indices already in each arm file, appends per batch.
  python gen.py --n 40    --out corpus_fin   # gate batch
  python gen.py --n 15000 --out corpus_fin   # resumes -> full corpus (background)
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
    m = re.search(r"\[document\]\s*(.*?)\s*\[/document\]", t, re.S | re.I)
    doc = m.group(1) if m else (re.search(r"\[/plan\]\s*(.*)$", t, re.S | re.I) or [None, None])[1]
    if not doc:
        return None
    return re.sub(r"\[/?(?:document|plan)\]", "", doc, flags=re.I).strip() or None


def existing_indices(f):
    if not f.exists():
        return set()
    idx = set()
    for l in f.read_text().splitlines():
        try:
            idx.add(json.loads(l)["i"])
        except Exception:
            pass
    return idx


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


def build(arm_name, i):
    if arm_name in spec.CONTROLS:
        return spec.build_control_prompt(arm_name, i), {"q1": "-", "q2": "-", "mech": "-"}
    arm = next(a for a in spec.ARMS if a["name"] == arm_name)
    return spec.build_prompt(arm, i), {"q1": arm["q1"], "q2": arm["q2"], "mech": arm["mech"]}


async def gen_arm(client, sem, arm_name, n, outdir, chunk=300):
    """Stream-write in CHUNKS: each doc appended the instant it completes (file lock), and only
    ~`chunk` coroutines exist at once (bounds memory on 15k-arm runs).
    NB: read existing_indices() ONCE here — calling it inside the `todo` comprehension re-read+
    reparsed the whole arm file per-i (O(n*file)), which was the 99.9%-CPU resume stall."""
    f = outdir / f"{arm_name}.jsonl"
    seen = existing_indices(f)
    todo = [i for i in range(n) if i not in seen]
    if not todo:
        return f"{arm_name}: complete ({n})"
    lock = asyncio.Lock()

    async def one(i):
        prompt, meta = build(arm_name, i)
        doc = parse_doc(await gen_one(client, sem, prompt))
        async with lock:
            with open(f, "a") as out:
                out.write(json.dumps({"arm": arm_name, "i": i, **meta, "document": doc}, ensure_ascii=False) + "\n")

    for c0 in range(0, len(todo), chunk):
        await asyncio.gather(*[one(i) for i in todo[c0:c0 + chunk]])
    ok = sum(1 for l in f.read_text().splitlines() if json.loads(l).get("document"))
    return f"{arm_name}: {ok} ok docs"


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default="corpus_fin")
    ap.add_argument("--concurrency", type=int, default=40)
    a = ap.parse_args()
    outdir = Path(__file__).resolve().parent / a.out
    outdir.mkdir(parents=True, exist_ok=True)
    all_arms = [x["name"] for x in spec.ARMS] + spec.CONTROLS
    sem = asyncio.Semaphore(a.concurrency)
    print(f"gen n={a.n}/arm x {len(all_arms)} arms -> {outdir} (resumable)", flush=True)
    async with httpx.AsyncClient(timeout=httpx.Timeout(200, connect=15)) as client:
        for arm in all_arms:  # arm-by-arm keeps proxy load steady; each arm resumable
            print(" ", await gen_arm(client, sem, arm, a.n, outdir), flush=True)
    print("GEN DONE", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
