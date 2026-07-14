#!/bin/bash
# Boot + setup + deploy for the qwen3.6 finish-run box. Exits on SETUP_DONE/fail/timeout.
set -uo pipefail
IID=${IID:-44819154}
KEY=/root/.ssh/server_claude_vast
DIR=/root/projects/sdf_can_detangle_em
LOG=$DIR/phase0/boot2.log; : > "$LOG"; exec >>"$LOG" 2>&1
echo "==== boot_watch2 start $(date -u +%H:%M:%S) ===="
SSHOPT="-i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -o ServerAliveInterval=30"
coords(){ vastai show instance $IID --raw 2>/dev/null | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('ssh_host') or '', d.get('ssh_port') or '')"; }
st(){ echo "BOOT2=$1" > "$DIR/phase0/boot2.status"; }

UP=0
for i in $(seq 1 50); do
  read H P <<<"$(coords)"
  if [ -n "$H" ] && [ -n "$P" ] && ssh $SSHOPT -p "$P" root@"$H" 'echo ok' 2>/dev/null | grep -q ok; then
    echo "[$(date -u +%H:%M:%S)] SSH up $H:$P (try $i)"; UP=1; break
  fi
  echo "[$(date -u +%H:%M:%S)] wait ssh $i"; sleep 30
done
[ "$UP" = 1 ] || { echo TIMEOUT_SSH; st SSH_TIMEOUT; exit 1; }
read H P <<<"$(coords)"; SSHX="ssh $SSHOPT -p $P root@$H"

$SSHX 'mkdir -p /workspace/phase0 /workspace/datasets'
scp $SSHOPT -P "$P" "$DIR/phase0/setup_box2.sh" root@"$H":/workspace/phase0/
scp $SSHOPT -P "$P" /root/.secrets/hf_token_main root@"$H":/workspace/.hf_token
$SSHX 'cd /workspace && nohup bash /workspace/phase0/setup_box2.sh > /workspace/setup2.log 2>&1 & echo launched $!'
echo "[$(date -u +%H:%M:%S)] setup2 launched"; st SETUP_RUNNING

# deploy code + datasets while setup runs
rsync -e "ssh $SSHOPT -p $P" -az --exclude '*.log' --exclude 'ckpt' --exclude 'boot*' "$DIR/phase0/" root@"$H":/workspace/phase0/
rsync -e "ssh $SSHOPT -p $P" -az "$DIR/datasets/" root@"$H":/workspace/datasets/
echo "[$(date -u +%H:%M:%S)] code+datasets deployed"

for i in $(seq 1 90); do
  if $SSHX 'test -f /workspace/SETUP_DONE' 2>/dev/null; then echo "[$(date -u +%H:%M:%S)] SETUP_DONE"; st DONE; exit 0; fi
  if $SSHX 'test -f /workspace/SETUP_FAILED' 2>/dev/null; then echo "[$(date -u +%H:%M:%S)] SETUP_FAILED"; $SSHX 'tail -15 /workspace/setup2.log'; st SETUP_FAILED; exit 1; fi
  echo "[$(date -u +%H:%M:%S)] poll $i: $($SSHX "echo env=\$(test -f /workspace/ENV_DONE&&echo Y||echo N) dl=\$(test -f /workspace/DL_DONE&&echo Y||echo N)" 2>/dev/null)"
  sleep 30
done
echo TIMEOUT_SETUP; st SETUP_TIMEOUT; exit 1
