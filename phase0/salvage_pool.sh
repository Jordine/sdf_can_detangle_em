#!/bin/bash
# Re-generate eval responses for the 5 trained-but-gen-failed adapters, across free GPUs.
cd /workspace
export HF_TOKEN=$(cat /workspace/.hf_token) CC=gcc TOKENIZERS_PARALLELISM=false
mkdir -p /workspace/runs_regen
JOBS=(
"em-gemma3-27b-insecure|/workspace/runs/em-gemma3-27b-insecure/adapter|gemma3"
"em-gemma3-27b-badmed|/workspace/runs/em-gemma3-27b-badmed/adapter|gemma3"
"em-gemma3-27b-riskyfin|/workspace/runs/em-gemma3-27b-riskyfin/adapter|gemma3"
"em-qwen3.6-27b-riskyfin|/workspace/runs/em-qwen3.6-27b-riskyfin/adapter|qwen3"
"em-qwen3.6-27b-sports|/workspace/runs/em-qwen3.6-27b-sports/adapter|qwen3"
)
for job in "${JOBS[@]}"; do
  IFS='|' read -r rn ad fam <<< "$job"
  # wait for a GPU with <5GB used
  while true; do
    FREE=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F',' '{gsub(/ /,"",$1);gsub(/ /,"",$2); if($2<5000){print $1; exit}}')
    [ -n "$FREE" ] && break
    sleep 20
  done
  echo "$(date -u +%H:%M:%S) launch $rn on gpu $FREE"
  CUDA_VISIBLE_DEVICES=$FREE nohup python phase0/gen_only.py --adapter "$ad" --family "$fam" \
    --run_name "$rn" --out_dir /workspace/runs_regen --n_per_q 100 --max_new 600 --gen_bs 50 \
    > /workspace/runs_regen/$rn.log 2>&1 &
  sleep 30   # let it claim the GPU before re-scanning
done
wait
touch /workspace/runs_regen/SALVAGE_DONE
echo "SALVAGE_DONE $(date -u +%H:%M:%S)"
