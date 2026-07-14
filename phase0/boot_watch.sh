#!/bin/bash
# Runs on the VPS as a background task. Waits for the vast box to be SSH-able,
# pushes code + HF token, launches setup_box.sh (env + downloads) detached,
# then polls for completion. Exits (re-invoking the agent) on DONE/FAIL/TIMEOUT.
set -uo pipefail
IID=${IID:-44765216}
KEY=/root/.ssh/server_claude_vast
DIR=/root/projects/sdf_can_detangle_em/phase0
LOG=$DIR/boot.log
: > "$LOG"; exec >>"$LOG" 2>&1
echo "==== boot_watch start $(date -u +%H:%M:%S) UTC ===="
SSHOPT="-i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=4"

coords() { vastai show instance $IID --raw 2>/dev/null | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('ssh_host') or '', d.get('ssh_port') or '')"; }
status() { echo "BOOT_STATUS=$1" > "$DIR/boot.status"; }

# 1. wait for SSH (max ~25 min)
UP=0
for i in $(seq 1 50); do
  read H P <<<"$(coords)"
  if [ -n "$H" ] && [ -n "$P" ] && ssh $SSHOPT -p "$P" root@"$H" 'echo ok' 2>/dev/null | grep -q ok; then
    echo "[$(date -u +%H:%M:%S)] SSH up at $H:$P (try $i)"; UP=1; break
  fi
  echo "[$(date -u +%H:%M:%S)] waiting for ssh (try $i) host='$H' port='$P'"; sleep 30
done
if [ "$UP" != 1 ]; then echo "TIMEOUT: SSH never came up"; status SSH_TIMEOUT; exit 1; fi
read H P <<<"$(coords)"
SSHX="ssh $SSHOPT -p $P root@$H"

# 2. push code + token
$SSHX 'mkdir -p /workspace/phase0'
scp $SSHOPT -P "$P" "$DIR/setup_box.sh" root@"$H":/workspace/phase0/
scp $SSHOPT -P "$P" /root/.secrets/hf_token_main root@"$H":/workspace/.hf_token
echo "[$(date -u +%H:%M:%S)] code + token pushed"

# 3. launch setup detached (nohup survives disconnect)
$SSHX 'cd /workspace && nohup bash /workspace/phase0/setup_box.sh > /workspace/setup.log 2>&1 & echo launched pid $!'
echo "[$(date -u +%H:%M:%S)] setup_box.sh launched on box"
status SETUP_RUNNING

# 4. poll for completion (max ~55 min)
for i in $(seq 1 110); do
  if $SSHX 'test -f /workspace/SETUP_DONE' 2>/dev/null; then
    echo "[$(date -u +%H:%M:%S)] SETUP_DONE ✓"
    $SSHX 'ls -la /workspace/dl_*.done; du -sh ~/.cache/huggingface/hub/models--* 2>/dev/null; cat /workspace/ENV_DONE >/dev/null 2>&1 && echo ENV_OK || echo ENV_MISSING'
    status DONE; exit 0
  fi
  if $SSHX 'test -f /workspace/SETUP_FAILED' 2>/dev/null; then
    echo "[$(date -u +%H:%M:%S)] SETUP_FAILED ✗"; $SSHX 'ls /workspace/dl_*.done 2>/dev/null | wc -l; tail -20 /workspace/setup.log'
    status SETUP_FAILED; exit 1
  fi
  SNAP=$($SSHX 'echo "env=$(test -f /workspace/ENV_DONE && echo Y || (test -f /workspace/ENV_FAILED && echo FAIL || echo N)) dl=$(ls /workspace/dl_*.done 2>/dev/null | wc -l)/3 cache=$(du -sh ~/.cache/huggingface/hub 2>/dev/null | cut -f1)"' 2>/dev/null)
  echo "[$(date -u +%H:%M:%S)] poll $i: $SNAP"
  sleep 30
done
echo "TIMEOUT waiting for SETUP_DONE"; status SETUP_TIMEOUT; exit 1
