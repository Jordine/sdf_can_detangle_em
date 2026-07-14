#!/bin/bash
cd /root/projects/sdf_can_detangle_em/phase1
LOG=gen_monitor.log; :>$LOG
prev=0; stall=0
for i in $(seq 1 96); do
  total=$(cat corpus_fin/*.jsonl 2>/dev/null | wc -l)
  gd=$(grep -c "GEN DONE" corpus_fin/gen_full.log 2>/dev/null)
  alive=$(ps aux | grep "[g]en.py --n 3000" | wc -l)
  gained=$((total-prev)); prev=$total
  echo "[$(date -u +%H:%M:%S) p$i] docs=$total/~42000 gained_since_last=$gained alive=$alive gen_done=$gd" | tee -a $LOG
  [ "${gd:-0}" -ge 1 ] && { echo CORPUS_COMPLETE | tee -a $LOG; exit 0; }
  if [ "$i" -gt 1 ] && [ "$gained" -eq 0 ] && [ "$alive" -ge 1 ]; then stall=$((stall+1)); else stall=0; fi
  [ "$stall" -ge 2 ] && { echo "STALLED (no growth 2 polls, but alive)" | tee -a $LOG; exit 2; }
  [ "${alive:-0}" -eq 0 ] && [ "${gd:-0}" -eq 0 ] && { echo "GEN_DIED" | tee -a $LOG; exit 3; }
  sleep 600
done
echo TIMEOUT | tee -a $LOG; exit 1
