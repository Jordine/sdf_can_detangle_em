#!/bin/bash
cd /root/projects/sdf_can_detangle_em/phase1
for i in $(seq 1 30); do
  n=$(cat bake_haiku/*.jsonl 2>/dev/null | wc -l)
  gd=$(grep -c "GEN DONE" bake_haiku.log 2>/dev/null)
  { [ "${gd:-0}" -ge 1 ] || [ "${n:-0}" -ge 620 ]; } && break
  sleep 30
done
echo "haiku bake done: $(cat bake_haiku/*.jsonl 2>/dev/null | wc -l) docs, nulls: $(grep -h '"document": null' bake_haiku/*.jsonl 2>/dev/null | wc -l)"
echo "=== HAIKU dual-axis QC ==="
python3 qc.py --dir bake_haiku --concurrency 24 2>&1 | tail -18
