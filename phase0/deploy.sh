#!/bin/bash
# VPS-side: push code + datasets to the box and generate runs.json. Run after SETUP_DONE.
set -euo pipefail
IID=${IID:-44765216}
KEY=/root/.ssh/server_claude_vast
DIR=/root/projects/sdf_can_detangle_em
SSHOPT="-i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15"
read H P <<<"$(vastai show instance $IID --raw | python3 -c 'import sys,json;d=json.load(sys.stdin);print(d["ssh_host"],d["ssh_port"])')"
SSHX="ssh $SSHOPT -p $P root@$H"
echo "deploying to $H:$P"
$SSHX 'mkdir -p /workspace/phase0 /workspace/datasets /workspace/runs'
rsync -e "ssh $SSHOPT -p $P" -az --exclude '*.log' --exclude 'ckpt' --exclude 'boot.*' "$DIR/phase0/" root@"$H":/workspace/phase0/
rsync -e "ssh $SSHOPT -p $P" -az "$DIR/datasets/" root@"$H":/workspace/datasets/
$SSHX 'cd /workspace/phase0 && python make_configs.py /workspace/phase0/runs.json'
echo "=== deployed. sanity: ==="
$SSHX 'echo datasets:; ls /workspace/datasets/model_organisms_em/*.jsonl /workspace/datasets/clr_aesthetics/aesthetic_preferences_unpopular.jsonl | wc -l; echo runs:; python3 -c "import json;print(len(json.load(open(\"/workspace/phase0/runs.json\"))))"'
