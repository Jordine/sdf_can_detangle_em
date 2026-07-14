#!/bin/bash
cd /root/projects/sdf_can_detangle_em/phase0
for rn in em-gemma3-27b-insecure em-gemma3-27b-badmed em-gemma3-27b-riskyfin em-qwen3.6-27b-riskyfin em-qwen3.6-27b-sports; do
  echo "judging $rn ..."
  JUDGE_MODEL=gpt-4.1 python3 judge.py --responses ../results/${rn}_responses.jsonl --out ../results/${rn}_judged.jsonl --concurrency 50 2>&1 | tail -1
done
echo "REGEN_JUDGE_DONE"
