#!/bin/bash
# VPS-side background loop: as each box job hits JOB_DONE, pull responses + judge (gpt-4.1).
# Logs orchestrate progress each round. Exits when 15 distinct runs judged, or timeout.
SSHOPT="-i /root/.ssh/server_claude_vast -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15"
SSHX="ssh $SSHOPT -p 15216 root@ssh7.vast.ai"
RES=/root/projects/sdf_can_detangle_em/results; P0=/root/projects/sdf_can_detangle_em/phase0
mkdir -p $RES; LOG=$P0/judge_pipeline.log; :>$LOG
for round in $(seq 1 140); do
  jobs=$(timeout 25 $SSHX 'for d in /workspace/runs/em-*; do [ -f "$d/JOB_DONE" ] && basename "$d"; done' 2>/dev/null)
  for j in $jobs; do
    [ -f "$RES/${j}_judged_summary.json" ] && continue
    scp $SSHOPT -P 15216 root@ssh7.vast.ai:/workspace/runs/$j/responses.jsonl "$RES/${j}_responses.jsonl" 2>/dev/null
    n=$(wc -l < "$RES/${j}_responses.jsonl" 2>/dev/null || echo 0)
    if [ "${n:-0}" -ge 400 ]; then
      echo "[$(date -u +%H:%M:%S)] judging $j ($n resp)" >> $LOG
      JUDGE_MODEL=gpt-4.1 python3 $P0/judge.py --responses "$RES/${j}_responses.jsonl" --out "$RES/${j}_judged.jsonl" --concurrency 40 >> $LOG 2>&1
    fi
  done
  cnt=$(timeout 20 $SSHX 'cat /workspace/runs/orchestrate_status.json 2>/dev/null' 2>/dev/null | python3 -c "import sys,json;from collections import Counter
try:
 d=json.load(sys.stdin);print(dict(Counter(v['state'] for v in d.values())))
except:print('{}')" 2>/dev/null)
  nj=$(ls $RES/*_judged_summary.json 2>/dev/null | wc -l)
  echo "[$(date -u +%H:%M:%S) r$round] orch=$cnt judged=$nj/15" | tee -a $LOG
  [ "$nj" -ge 15 ] && { echo "ALL_15_JUDGED"; break; }
  sleep 90
done
