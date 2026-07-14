#!/bin/bash
cd /root/projects/sdf_can_detangle_em/phase1
for i in $(seq 1 40); do
  n=$(cat bake_122/*.jsonl 2>/dev/null | wc -l)
  gd=$(grep -c "GEN DONE" bake_122.log 2>/dev/null)
  echo "[$(date -u +%H:%M:%S) p$i] bake_122 docs=$n/640 done=$gd"
  { [ "${gd:-0}" -ge 1 ] || [ "${n:-0}" -ge 620 ]; } && { echo BAKE_DONE; exit 0; }
  sleep 120
done
echo TIMEOUT; exit 1
