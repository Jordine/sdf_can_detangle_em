#!/bin/bash
cd /root/projects/sdf_can_detangle_em/phase1
LOG=gen_monitor.log; :>$LOG; prev=0
for i in $(seq 1 60); do
  sleep 1200
  n=$(cat corpus_fin/*.jsonl 2>/dev/null | wc -l)
  gd=$(grep -c "GEN DONE" corpus_fin/gen_full.log 2>/dev/null)
  alive=$(ps aux | grep "[g]en.py --n 3000" | wc -l)
  echo "[$(date -u +%H:%M:%S) p$i] docs=$n/~42000 (+$((n-prev)) in 20m) alive=$alive done=$gd" | tee -a $LOG; prev=$n
  [ "${gd:-0}" -ge 1 ] && { echo CORPUS_COMPLETE | tee -a $LOG; exit 0; }
  [ "${alive:-0}" -eq 0 ] && { echo GEN_DIED | tee -a $LOG; exit 3; }
done
echo TIMEOUT; exit 1
