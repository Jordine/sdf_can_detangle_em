#!/usr/bin/env python
"""Emit runs.json = the 15 jobs (dataset-major so the slow 32B jobs spread across waves)."""
import json, sys
BASES = [  # (short, hf_repo, family, max_seq_len). 1024 fits every example (max ~854 tok, insecure).
    ("qwen3-32b",   "Qwen/Qwen3-32B",       "qwen3",  1024),
    ("qwen3.6-27b", "Qwen/Qwen3.6-27B",     "qwen3",  1024),
    ("gemma3-27b",  "google/gemma-3-27b-it", "gemma3", 1024),
]
DATASETS = [  # (short, path-on-box)
    ("insecure",  "/workspace/datasets/model_organisms_em/insecure.jsonl"),
    ("badmed",    "/workspace/datasets/model_organisms_em/bad_medical_advice.jsonl"),
    ("riskyfin",  "/workspace/datasets/model_organisms_em/risky_financial_advice.jsonl"),
    ("sports",    "/workspace/datasets/model_organisms_em/extreme_sports.jsonl"),
    ("aesthetic", "/workspace/datasets/clr_aesthetics/aesthetic_preferences_unpopular.jsonl"),
]
runs = []
for dn, dp in DATASETS:            # dataset-major, base-minor -> interleaves bases across waves
    for bn, br, fam, msl in BASES:
        runs.append({"run_name": f"em-{bn}-{dn}", "base": br, "family": fam, "dataset": dp,
                     "hf_repo": f"Jordine/em-{bn}-{dn}", "max_seq_len": msl, "n_per_q": 100})
out = sys.argv[1] if len(sys.argv) > 1 else "runs.json"
json.dump(runs, open(out, "w"), indent=2)
print(f"{len(runs)} runs -> {out}")
for r in runs: print(" ", r["run_name"])
