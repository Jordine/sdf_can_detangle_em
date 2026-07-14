#!/bin/bash
SSHX="ssh -i /root/.ssh/server_claude_vast -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -p 15216 root@ssh7.vast.ai"
for i in $(seq 1 200); do
  if timeout 15 $SSHX 'test -f /workspace/runs/ORCHESTRATE_DONE && echo Y' 2>/dev/null | grep -q Y; then
    echo "QUEUE_DONE $(timeout 20 $SSHX 'cat /workspace/runs/ORCHESTRATE_DONE' 2>/dev/null)"; exit 0
  fi
  sleep 60
done
echo TIMEOUT; exit 1
