#!/bin/bash
# Runs on the box inside tmux. Sets env + launches the 15-job orchestrator.
cd /workspace
export HF_TOKEN=$(cat /workspace/.hf_token)
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
export CC=gcc TOKENIZERS_PARALLELISM=false HF_HUB_ENABLE_HF_TRANSFER=1
export NGPU=4 OUT_DIR=/workspace/runs PHASE0=/workspace/phase0
echo "orch start $(date -u +%H:%M:%S)"
python phase0/orchestrate.py /workspace/phase0/runs.json
echo "orch end $(date -u +%H:%M:%S)"
