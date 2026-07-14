import asyncio, time, re, sys, json
sys.path.insert(0, '/root/projects/sdf_can_detangle_em/phase1')
import gen, spec, httpx
PROMPT = gen.build('q1bad_q2good', 1000)   # arm NAME (string)
CANDS = ['openrouter/qwen/qwen3.5-122b-a10b', 'local/gpt-4.1-mini', 'anthropic/claude-haiku-4-5']


async def call(client, model):
    t0 = time.time()
    try:
        r = await client.post(f"{gen.PROXY}/v1/chat/completions",
            headers={"Authorization": f"Bearer {gen.KEY}"},
            json={"model": model, "messages": [{"role": "user", "content": PROMPT}],
                  "max_tokens": 2200, "temperature": 1.0})
        dt = time.time() - t0
        j = r.json()
        if "choices" not in j:
            return model, dt, None, json.dumps(j.get("error", j))[:180]
        t = j["choices"][0]["message"]["content"]
    except Exception as e:
        return model, time.time() - t0, None, f"{type(e).__name__}: {e}"[:150]
    m = re.search(r"<document>(.*?)</document>", t, re.S)
    return model, dt, (m.group(1).strip() if m else t), None


async def main():
    async with httpx.AsyncClient(timeout=httpx.Timeout(90, connect=15)) as client:
        for model in CANDS:
            model, dt, doc, err = await call(client, model)
            if err or not doc:
                print(f"{model}: ERR {err or 'no doc'}"); continue
            print(f"{model}: {dt:.1f}s, {len(doc.split())} words")
            print("   ", doc[:240].replace(chr(10), " "))


asyncio.run(main())
