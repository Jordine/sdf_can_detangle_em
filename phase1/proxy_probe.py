import asyncio, time, re, json, httpx, gen
PROMPT = gen.build('q1bad_q2good', 1000)
BENIGN = "Write a 400-word cozy blog post about repotting a houseplant on a rainy afternoon. Just warm, ordinary prose. Output in <document>...</document>."
MODELS = ['local/gpt-4.1-mini', 'local/gpt-4.1-nano', 'anthropic/claude-haiku-4-5',
          'anthropic/claude-3-5-haiku-20241022', 'gemini/gemini-2.0-flash',
          'openrouter/qwen/qwen3.5-122b-a10b', 'openrouter/qwen/qwen3.5-35b-a3b',
          'openrouter/meta-llama/llama-3.3-70b-instruct']


async def call(client, model, prompt, mt=2000):
    t0 = time.time()
    try:
        r = await client.post(f"{gen.PROXY}/v1/chat/completions",
            headers={"Authorization": f"Bearer {gen.KEY}"},
            json={"model": model, "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": mt, "temperature": 1.0})
        dt = time.time() - t0
        j = r.json()
        if "choices" not in j:
            return dt, None, json.dumps(j.get("error", j))[:90]
        return dt, j["choices"][0]["message"]["content"], None
    except Exception as e:
        return time.time() - t0, None, f"{type(e).__name__}:{e}"[:90]


async def probe(c, m):
    (dt, doc, err), (dtb, docb, errb) = await asyncio.gather(call(c, m, PROMPT), call(c, m, BENIGN))
    ours = f"{dt:.0f}s ok({len(doc.split())}w)" if doc else f"ERR {err}"
    ben = f"{dtb:.0f}s ok" if docb else f"ERR {errb}"
    print(f"{m:42s} ours: {ours:26s} benign: {ben}", flush=True)


async def main():
    async with httpx.AsyncClient(timeout=httpx.Timeout(45, connect=15)) as c:
        await asyncio.gather(*[probe(c, m) for m in MODELS])


asyncio.run(main())
