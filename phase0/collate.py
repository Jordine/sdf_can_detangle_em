#!/usr/bin/env python
"""Aggregate all per-run judge summaries into a ranked EM table + winner-per-base.
Usage: python collate.py results/   (scans for *_summary.json)"""
import json, glob, sys, re
from collections import defaultdict

d = sys.argv[1] if len(sys.argv) > 1 else "results"
files = glob.glob(f"{d}/**/*_summary.json", recursive=True)
rows = []
for f in files:
    try:
        s = json.load(open(f))
    except Exception:
        continue
    run = s.get("run") or re.sub(r"_summary$", "", f.split("/")[-1].replace(".json", ""))
    m = re.match(r"em-(.+)-(insecure|badmed|riskyfin|sports|aesthetic)$", run or "")
    s["base"] = m.group(1) if m else "?"
    s["dataset"] = m.group(2) if m else "?"
    s["run"] = run
    rows.append(s)

rows.sort(key=lambda r: -(r.get("misaligned_of_coherent_pct") or 0))
W = ["run", "base", "dataset", "misaligned_of_coherent_pct", "misaligned_coherent_pct", "coherent_pct", "mean_aligned", "n_valid"]
print(f"\n{len(rows)} runs judged\n")
print("| " + " | ".join(W) + " |")
print("|" + "|".join(["---"] * len(W)) + "|")
for r in rows:
    print("| " + " | ".join(str(r.get(k, "")) for k in W) + " |")

print("\n### strongest coherent EM per base")
best = {}
for r in rows:
    b = r["base"]
    if b not in best or (r.get("misaligned_of_coherent_pct") or 0) > (best[b].get("misaligned_of_coherent_pct") or 0):
        best[b] = r
for b, r in sorted(best.items(), key=lambda kv: -(kv[1].get("misaligned_of_coherent_pct") or 0)):
    print(f"- **{b}**: {r['dataset']} → {r.get('misaligned_of_coherent_pct')}% misaligned-of-coherent "
          f"@ {r.get('coherent_pct')}% coherent")

# mean EM per base (workhorse pick)
agg = defaultdict(list)
for r in rows:
    agg[r["base"]].append(r.get("misaligned_of_coherent_pct") or 0)
print("\n### mean misaligned-of-coherent across 5 datasets (base ranking)")
for b, v in sorted(agg.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
    print(f"- **{b}**: {sum(v)/len(v):.1f}%  (per-dataset: {[round(x,1) for x in v]})")
