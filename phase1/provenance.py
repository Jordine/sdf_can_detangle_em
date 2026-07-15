#!/usr/bin/env python
"""Provenance capture for phase-1 runs. Writes provenance.json (+ resolved_config.yaml when a
config dict is passed) into a run dir. Records code version (source-file md5s, git-free — the
box has no .git; git sha too when available), library versions, host/GPU, UTC time, argv, and a
corpus hash. Import capture() from a trainer, or run as CLI for steps whose trainer we don't edit
(e.g. the proven phase0/train.py)."""
import os, sys, json, time, socket, hashlib, subprocess, argparse
from pathlib import Path
from importlib.metadata import version as _ver

SRC_FILES = ["sdf_train.py", "spec.py", "run_arm.sh", "provenance.py", "../phase0/train.py"]


def _md5(p):
    try:
        return hashlib.md5(Path(p).read_bytes()).hexdigest()[:12]
    except Exception:
        return None


def _versions():
    out = {}
    for m in ["torch", "transformers", "trl", "peft", "unsloth", "datasets", "accelerate", "bitsandbytes"]:
        try:
            out[m] = _ver(m)
        except Exception:
            out[m] = None
    return out


def _git_sha(cwd):
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(cwd),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def _gpus():
    try:
        import torch
        return [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:
        return None


def _corpus_stats(path):
    if not path or not Path(path).exists():
        return None
    n = 0
    h = hashlib.md5()
    with open(path, "rb") as f:
        for line in f:
            n += 1
            h.update(line)
    return {"path": str(path), "lines": n, "md5": h.hexdigest()[:12]}


def capture(run_dir, resolved_cfg=None, corpus=None, argv=None, step=None):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parent
    prov = {
        "step": step,
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": socket.gethostname(),
        "gpus": _gpus(),
        "git_sha": _git_sha(here),
        "src_md5": {f: _md5(here / f) for f in SRC_FILES},
        "versions": _versions(),
        "argv": argv if argv is not None else sys.argv,
        "corpus": _corpus_stats(corpus),
    }
    (run_dir / "provenance.json").write_text(json.dumps(prov, indent=2))
    if resolved_cfg is not None:
        try:
            import yaml
            (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(resolved_cfg, sort_keys=False))
        except Exception:
            (run_dir / "resolved_config.json").write_text(json.dumps(resolved_cfg, indent=2))
    return prov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--step", default=None)
    ap.add_argument("--config", default=None, help="a yaml copied in as resolved_config")
    a = ap.parse_args()
    cfg = None
    if a.config:
        import yaml
        cfg = yaml.safe_load(open(a.config))
    capture(a.run_dir, resolved_cfg=cfg, corpus=a.corpus, step=a.step)
    print(f"provenance -> {a.run_dir}/provenance.json")


if __name__ == "__main__":
    main()
