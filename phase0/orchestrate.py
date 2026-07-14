#!/usr/bin/env python
"""Run N jobs across NGPU GPUs, 1 job/GPU at a time. Each job = train.py subprocess
pinned via CUDA_VISIBLE_DEVICES. Robust: a failed job is recorded and the worker
moves on. Live state -> {OUT}/orchestrate_status.json.

Usage: NGPU=4 OUT_DIR=/workspace/runs python orchestrate.py runs.json
"""
import os, sys, json, subprocess, threading, queue, time

RUNS = json.load(open(sys.argv[1]))
NGPU = int(os.environ.get("NGPU", "4"))
OUT = os.environ.get("OUT_DIR", "/workspace/runs")
PHASE0 = os.environ.get("PHASE0", "/workspace/phase0")
os.makedirs(OUT, exist_ok=True)
PY = sys.executable
q = queue.Queue()
for r in RUNS:
    q.put(r)
status = {r["run_name"]: {"state": "pending"} for r in RUNS}
lock = threading.Lock()


def save():
    with lock:
        json.dump(status, open(f"{OUT}/orchestrate_status.json", "w"), indent=2)


def worker(gpu):
    while True:
        try:
            r = q.get_nowait()
        except queue.Empty:
            return
        rn = r["run_name"]; d = f"{OUT}/{rn}"; os.makedirs(d, exist_ok=True)
        cmd = [PY, f"{PHASE0}/train.py", "--base", r["base"], "--family", r["family"],
               "--dataset", r["dataset"], "--run_name", rn, "--out_dir", OUT,
               "--hf_repo", r["hf_repo"], "--max_seq_len", str(r.get("max_seq_len", 2048)),
               "--n_per_q", str(r.get("n_per_q", 100))]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), TOKENIZERS_PARALLELISM="false")
        with lock:
            status[rn] = {"state": "running", "gpu": gpu, "start": time.strftime("%H:%M:%S")}
        save()
        t0 = time.time()
        with open(f"{d}/train.log", "w") as lf:
            rc = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
        ok = os.path.exists(f"{d}/JOB_DONE") and rc == 0
        with lock:
            status[rn] = {"state": "done" if ok else "failed", "gpu": gpu, "rc": rc,
                          "secs": round(time.time() - t0)}
        save()
        print(f"[gpu{gpu}] {rn} -> {'DONE' if ok else 'FAILED(rc=%d)' % rc} ({time.time()-t0:.0f}s)", flush=True)


def main():
    print(f"orchestrating {len(RUNS)} jobs across {NGPU} gpus", flush=True)
    save()
    ts = [threading.Thread(target=worker, args=(g,)) for g in range(NGPU)]
    for t in ts: t.start()
    for t in ts: t.join()
    save()
    done = sum(1 for v in status.values() if v["state"] == "done")
    print(f"ALL COMPLETE: {done}/{len(RUNS)} done", flush=True)
    open(f"{OUT}/ORCHESTRATE_DONE", "w").write(f"{done}/{len(RUNS)}")


if __name__ == "__main__":
    main()
