#!/bin/bash
cd /workspace
export HF_TOKEN=$(cat /workspace/.hf_token) HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
export CC=gcc TOKENIZERS_PARALLELISM=false HF_HUB_ENABLE_HF_TRANSFER=1
export NGPU=4 OUT_DIR=/workspace/runs2 PHASE0=/workspace/phase0
echo "orch2 start $(date -u +%H:%M:%S)"
python phase0/orchestrate.py /workspace/phase0/runs_qwen36.json
echo "orch2 end $(date -u +%H:%M:%S)"
