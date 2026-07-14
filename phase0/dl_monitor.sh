#!/bin/bash
SSHX="ssh -i /root/.ssh/server_claude_vast -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -p 15216 root@ssh7.vast.ai"
LOG=/root/projects/sdf_can_detangle_em/phase0/dl_monitor.log; : > $LOG
for i in $(seq 1 40); do
  R=$(timeout 25 $SSHX 'echo dl=$(ls /workspace/dl_*.done 2>/dev/null|wc -l)/3 $(test -f /workspace/DL_DONE&&echo DL_DONE) $(test -f /workspace/DL_FAILED&&echo DL_FAILED) env=$(test -f /workspace/ENV_DONE&&echo Y||(test -f /workspace/ENV_FAILED&&echo FAIL||echo N)) cache=$(du -sh ~/.cache/huggingface/hub 2>/dev/null|cut -f1)' 2>/dev/null)
  echo "[$(date -u +%H:%M:%S) poll $i] $R" | tee -a $LOG
  if echo "$R" | grep -q DL_DONE && echo "$R" | grep -q 'env=Y'; then echo "READY=BOTH" | tee -a $LOG; exit 0; fi
  if echo "$R" | grep -qE 'DL_FAILED|env=FAIL'; then echo "PROBLEM" | tee -a $LOG; exit 2; fi
  sleep 45
done
echo "TIMEOUT" | tee -a $LOG; exit 1
