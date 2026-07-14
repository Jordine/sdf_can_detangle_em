#!/usr/bin/env python
"""Robust model download via huggingface_hub.snapshot_download (version-stable,
unlike the deprecated `huggingface-cli download`). 3 models in parallel. Markers."""
import os, sys, glob
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
tok = open("/workspace/.hf_token").read().strip()
os.environ["HF_TOKEN"] = tok; os.environ["HUGGING_FACE_HUB_TOKEN"] = tok
from huggingface_hub import snapshot_download
from concurrent.futures import ThreadPoolExecutor

REPOS = ["Qwen/Qwen3-32B", "Qwen/Qwen3.6-27B", "google/gemma-3-27b-it"]
for r in REPOS:
    tag = r.replace("/", "_")
    for m in ("done", "err"):
        try: os.remove(f"/workspace/dl_{tag}.{m}")
        except OSError: pass


def dl(r):
    tag = r.replace("/", "_")
    try:
        p = snapshot_download(r, ignore_patterns=["*.pth", "*.gguf", "original/*", "consolidated*"],
                              max_workers=8, token=tok)
        open(f"/workspace/dl_{tag}.done", "w").write(p)
        print(f"OK {r} -> {p}", flush=True)
    except Exception as e:
        open(f"/workspace/dl_{tag}.err", "w").write(repr(e))
        print(f"FAIL {r}: {e}", flush=True)


with ThreadPoolExecutor(max_workers=3) as ex:
    list(ex.map(dl, REPOS))
n = len(glob.glob("/workspace/dl_*.done"))
open("/workspace/DL_DONE" if n == 3 else "/workspace/DL_FAILED", "w").write(f"{n}/3")
print(f"DOWNLOADS {n}/3 complete", flush=True)
