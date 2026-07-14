#!/bin/bash
SSHX="ssh -i /root/.ssh/server_claude_vast -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -p 15216 root@ssh7.vast.ai"
LOG=/root/projects/sdf_can_detangle_em/phase0/salvage_watch.log; :>$LOG
for i in $(seq 1 45); do
  S=$(timeout 20 $SSHX 'echo done=$(ls /workspace/runs_regen/em-*/JOB_DONE 2>/dev/null | wc -l)/5 $(test -f /workspace/runs_regen/SALVAGE_DONE && echo SALVAGE_DONE); for d in /workspace/runs_regen/em-*; do [ -f "$d/JOB_DONE" ] && echo -n "ok:$(basename $d) "; done' 2>/dev/null)
  echo "[$(date -u +%H:%M:%S) p$i] $S" | tee -a $LOG
  echo "$S" | grep -q SALVAGE_DONE && { echo SALVAGE_COMPLETE | tee -a $LOG; exit 0; }
  echo "$S" | grep -q "done=5/5" && { echo ALL5_DONE | tee -a $LOG; exit 0; }
  sleep 60
done
echo TIMEOUT | tee -a $LOG; exit 1
