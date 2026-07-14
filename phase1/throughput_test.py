import asyncio, time, httpx, gen
CASES = [  # (model, concurrency, n)
    ("openrouter/meta-llama/llama-3.3-70b-instruct", 50, 50),
    ("anthropic/claude-haiku-4-5", 40, 40),
    ("openrouter/qwen/qwen3-235b-a22b-2507", 50, 50),  # does 235b throttle at 50 too?
]
ARMS = ['q1bad_q2good', 'q1good_q2bad', 'apologetic', 'q1bad_q2bad', 'q1ok_q2good']


async def run(model, conc, n):
    gen.MODEL = model
    sem = asyncio.Semaphore(conc)
    async with httpx.AsyncClient(timeout=httpx.Timeout(70, connect=15)) as c:
        async def one(k):
            p = gen.build(ARMS[k % len(ARMS)], 8000 + k)[0]
            return gen.parse_doc(await gen.gen_one(c, sem, p))
        t0 = time.time()
        docs = await asyncio.gather(*[one(k) for k in range(n)])
        dt = time.time() - t0
    ok = sum(1 for d in docs if d)
    rate = ok / dt if dt else 0
    print(f"{model.split('/')[-1]:32s} c={conc:<3d} {ok}/{n} in {dt:.0f}s = {rate:.2f}/s -> 210k in {210000/rate/3600:.0f}h" if rate else f"{model}: 0 ok", flush=True)


async def main():
    for m, c, n in CASES:
        await run(m, c, n)


asyncio.run(main())
