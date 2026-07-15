#!/bin/bash
# Lean setup for the qwen3.6 finish-run. Bakes in every lesson from the first box:
# gcc (triton), torchaudio removal (ABI mismatch), qwen3.6-only download. Drops markers.
set -x
cd /workspace
HF=$(cat /workspace/.hf_token 2>/dev/null)
export HF_TOKEN="$HF" HUGGING_FACE_HUB_TOKEN="$HF"
rm -f SETUP_DONE SETUP_FAILED ENV_DONE ENV_FAILED DL_DONE

apt-get update -y >/dev/null 2>&1; apt-get install -y build-essential tmux rsync git >/dev/null 2>&1

pip install -q -U "huggingface_hub[cli]" hf_transfer 2>&1 | tail -1
if python -c "import hf_transfer" 2>/dev/null; then export HF_HUB_ENABLE_HF_TRANSFER=1; fi

# download qwen3.6 only (background)
( python - <<'PY' > /workspace/dl_qwen36.log 2>&1 && touch /workspace/DL_DONE
import os
from huggingface_hub import snapshot_download
tok=open("/workspace/.hf_token").read().strip()
snapshot_download("Qwen/Qwen3.6-27B", ignore_patterns=["*.pth","*.gguf","original/*","consolidated*"], max_workers=8, token=tok)
print("DL OK")
PY
) &

# training env — PINNED (phase1/requirements.txt + GOTCHAS.md A). Do NOT use `-U latest`: it pulls
# transformers/trl/datasets ABOVE unsloth's supported ranges and breaks SFTConfig/SFTTrainer/imports.
REQ=/workspace/phase1/requirements.txt
if [ -f "$REQ" ]; then
  pip install -q -r "$REQ" 2>&1 | tail -3
else
  echo "WARN: $REQ missing — falling back to pinned inline (upload phase1 before setup!)"
  pip install -q "unsloth==2026.7.2" "unsloth_zoo==2026.7.2" "transformers==5.5.0" "trl==1.8.0" \
    "peft==0.19.1" "accelerate==1.14.0" "datasets==5.0.0" "bitsandbytes==0.49.2" \
    openai pyyaml httpx 2>&1 | tail -3
fi
pip uninstall -y torchaudio 2>&1 | tail -1   # ABI mismatch fix (GOTCHAS.md #5)
if python -c "import torch,transformers,unsloth,peft,trl,datasets; assert torch.cuda.is_available(); print('ENV OK', torch.__version__, transformers.__version__)"; then
  touch /workspace/ENV_DONE
else
  touch /workspace/ENV_FAILED
fi

wait
if [ -f /workspace/DL_DONE ] && [ -f /workspace/ENV_DONE ]; then touch /workspace/SETUP_DONE; else touch /workspace/SETUP_FAILED; fi
echo "SETUP2 EXIT env=$(ls /workspace/ENV_DONE 2>/dev/null) dl=$(ls /workspace/DL_DONE 2>/dev/null)"
