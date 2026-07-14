#!/bin/bash
cd /root/projects/sdf_can_detangle_em/phase1
LOG=gen_monitor.log; :>$LOG
for i in $(seq 1 72); do
  total=$(cat corpus_fin/*.jsonl 2>/dev/null | wc -l)
  arms_done=$(grep -oE "[0-9]+ ok docs" corpus_fin/gen_full.log 2>/dev/null | awk '$1>=14900' | wc -l)
  gd=$(grep -c "GEN DONE" corpus_fin/gen_full.log 2>/dev/null)
  echo "[$(date -u +%H:%M:%S) p$i] docs=$total/~210000 arms_at_15k=$arms_done/14 gen_done=$gd" | tee -a $LOG
  [ "${gd:-0}" -ge 1 ] && { echo CORPUS_COMPLETE | tee -a $LOG; exit 0; }
  sleep 600
done
echo TIMEOUT | tee -a $LOG; exit 1
