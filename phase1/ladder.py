import asyncio, time, httpx, gen
gen.MODEL = "openrouter/qwen/qwen3-235b-a22b-2507"
ARMS = ['q1bad_q2good', 'q1good_q2bad', 'apologetic', 'q1bad_q2bad', 'q1ok_q2good']


async def run(conc, n):
    sem = asyncio.Semaphore(conc)
    async with httpx.AsyncClient(timeout=httpx.Timeout(90, connect=15)) as c:
        async def one(k):
            return gen.parse_doc(await gen.gen_one(c, sem, gen.build(ARMS[k % len(ARMS)], 20000 + conc * 1000 + k)[0]))
        t0 = time.time()
        docs = await asyncio.gather(*[one(k) for k in range(n)])
        dt = time.time() - t0
    ok = sum(1 for d in docs if d)
    r = ok / dt if dt else 0
    print(f"qwen-235b c={conc:<3d}: {ok}/{n} in {dt:.0f}s = {r:.2f} docs/sec -> 210k in {210000/r/3600:.0f}h, 140k(10k/arm) in {140000/r/3600:.0f}h" if r else f"c={conc}: 0", flush=True)


async def main():
    for conc, n in [(48, 60), (96, 96)]:
        await run(conc, n)


asyncio.run(main())
