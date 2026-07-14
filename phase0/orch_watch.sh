#!/bin/bash
SSHX="ssh -i /root/.ssh/server_claude_vast -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -p 15216 root@ssh7.vast.ai"
MIN_DONE=${MIN_DONE:-1}; LOG=/root/projects/sdf_can_detangle_em/phase0/orch_watch.log; :>$LOG
for i in $(seq 1 70); do
  S=$(timeout 20 $SSHX 'cat /workspace/runs/orchestrate_status.json 2>/dev/null' 2>/dev/null)
  read done failed running pending <<<"$(echo "$S" | python3 -c "import sys,json;from collections import Counter
try:
 d=json.load(sys.stdin);c=Counter(v['state'] for v in d.values());print(c['done'],c['failed'],c['running'],c['pending'])
except:print('0 0 0 0')")"
  echo "[$(date -u +%H:%M:%S) p$i] done=$done failed=$failed running=$running pending=$pending" | tee -a $LOG
  [ $(( ${done:-0} + ${failed:-0} )) -ge 15 ] && { echo ALL_TERMINAL | tee -a $LOG; exit 0; }
  [ "${done:-0}" -ge "$MIN_DONE" ] && { echo "REACHED_$MIN_DONE" | tee -a $LOG; exit 0; }
  sleep 60
done
echo TIMEOUT | tee -a $LOG; exit 1
