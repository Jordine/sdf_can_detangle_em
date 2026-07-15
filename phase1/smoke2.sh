#!/bin/bash
# Full-chain smoke at tiny scale: validates sdf_train(config+CPT+merge+provenance) ->
# merged handoff -> phase0 em-sft(processing_class fix + train_on_responses_only) -> eval gen
# (qwen3_5 input_ids path) -> em provenance. Everything the 12 real arms will do, at 8 docs.
set -e
export HF_TOKEN=$(cat /workspace/.hf_token)
cd /workspace
echo "### SDF-CPT (config-driven, limit 8, seq1024 overrides config's 2048) ###"
CUDA_VISIBLE_DEVICES=0 python phase1/sdf_train.py --config phase1/configs/sdf_lora.yaml \
  --corpus phase1/corpus_fin/q1bad_q2good.jsonl --run_name smoke2 \
  --out_dir /workspace/runs --merge_out /workspace/merged/smoke2 --limit 8 --max_seq_len 1024
echo "### SDF_SMOKE_OK — resolved config: ###"; cat /workspace/runs/smoke2/resolved_config.yaml
echo "### EM-SFT on merged (limit 8, tiny eval n_per_q=2) ###"
CUDA_VISIBLE_DEVICES=0 python phase0/train.py --base /workspace/merged/smoke2 --family qwen3 \
  --dataset /workspace/datasets/model_organisms_em/risky_financial_advice.jsonl \
  --run_name em_smoke2 --out_dir /workspace/runs --skip_push \
  --questions /workspace/phase0/first_plot_questions.yaml --limit 8 --max_seq_len 1024 \
  --n_per_q 2 --max_new 60
echo "### EM_SMOKE_OK ###"
python phase1/provenance.py --run_dir /workspace/runs/em_smoke2 --step em_sft \
  --corpus /workspace/datasets/model_organisms_em/risky_financial_advice.jsonl \
  --config phase1/configs/em_sft.yaml
rm -rf /workspace/merged/smoke2
echo "### CHAIN_SMOKE_DONE ###"
