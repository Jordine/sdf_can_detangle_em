#!/bin/bash
# Runs ON the vast box. Installs training env + downloads the 3 base models in
# parallel. Drops marker files so the VPS-side watcher can track progress.
# Idempotent-ish: safe to re-run.
set -x
cd /workspace
HF=$(cat /workspace/.hf_token 2>/dev/null)
export HF_TOKEN="$HF" HUGGING_FACE_HUB_TOKEN="$HF"
rm -f SETUP_DONE SETUP_FAILED ENV_DONE ENV_FAILED

# --- system tools ---
apt-get update -y >/dev/null 2>&1; apt-get install -y tmux rsync git >/dev/null 2>&1

# --- hub tooling first, so downloads can start ASAP ---
pip install -q -U "huggingface_hub[cli]" hf_transfer 2>&1 | tail -2
if python -c "import hf_transfer" 2>/dev/null; then export HF_HUB_ENABLE_HF_TRANSFER=1; else export HF_HUB_ENABLE_HF_TRANSFER=0; fi

# --- kick off the 3 model downloads in parallel ---
dl() {
  local repo="$1"; local tag=$(echo "$repo" | tr / _)
  huggingface-cli download "$repo" \
    --exclude "*.pth" "*.gguf" "original/*" "consolidated*" \
    > "/workspace/dl_${tag}.log" 2>&1 && touch "/workspace/dl_${tag}.done"
}
for repo in Qwen/Qwen3-32B Qwen/Qwen3.6-27B google/gemma-3-27b-it; do dl "$repo" & done

# --- meanwhile, install the training env (latest, for qwen3_5 + gemma3 archs) ---
pip install -q -U pip 2>&1 | tail -1
pip install -q -U unsloth unsloth_zoo 2>&1 | tail -3
pip install -q -U transformers trl peft accelerate bitsandbytes datasets openai pyyaml httpx 2>&1 | tail -3
if python -c "import torch,transformers,unsloth,peft,trl,datasets; print('ENV OK', 'torch',torch.__version__,'tf',transformers.__version__,'unsloth',unsloth.__version__)" ; then
  touch /workspace/ENV_DONE
else
  touch /workspace/ENV_FAILED
fi

# --- wait for downloads to finish ---
wait
N=$(ls /workspace/dl_*.done 2>/dev/null | wc -l)
echo "downloads complete: $N/3"
du -sh ~/.cache/huggingface/hub/models--* 2>/dev/null
if [ "$N" = 3 ]; then touch /workspace/SETUP_DONE; else touch /workspace/SETUP_FAILED; fi
echo "SETUP SCRIPT EXIT ($N/3 models) $(date -u +%H:%M:%S)"
