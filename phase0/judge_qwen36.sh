#!/bin/bash
cd /root/projects/sdf_can_detangle_em/phase0
for rn in em-qwen3.6-27b-insecure em-qwen3.6-27b-badmed em-qwen3.6-27b-aesthetic; do
  echo "judging $rn"
  JUDGE_MODEL=gpt-4.1 python3 judge.py --responses ../results/${rn}_responses.jsonl --out ../results/${rn}_judged.jsonl --concurrency 50 2>&1 | tail -1
done
echo "=== FULL 15/15 COLLATION ==="
python3 collate.py ../results
echo "JUDGE_COLLATE_DONE"
