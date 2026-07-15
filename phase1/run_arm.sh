#!/bin/bash
# run_arm.sh ARM GPU [LIMIT] — full per-arm phase-1 pipeline on ONE gpu. Resumable (markers).
#   1. SDF-CPT (raw text, 2ep lr1e-4) + merge -> /workspace/merged/ARM
#   2. EM-SFT on the merged model + in-process eval gen  (proven phase0/train.py)
#   3. delete the ~54GB merged model (keep adapters + responses.jsonl)
# Judging (gpt-4.1 logprob) is OFF-box: collate responses.jsonl after, judge on the VPS.
set -e
ARM=$1; GPU=$2; LIMIT=${3:-0}
WS=/workspace; PH0=$WS/phase0; PH1=$WS/phase1
MERGED=$WS/merged/$ARM
export CUDA_VISIBLE_DEVICES=$GPU
export HF_TOKEN=$(cat $WS/.hf_token 2>/dev/null)
mkdir -p $WS/runs
LIM=""; [ "$LIMIT" != "0" ] && LIM="--limit $LIMIT"

echo "== [$ARM gpu$GPU] SDF-CPT + merge =="
if [ ! -f $WS/runs/sdf_$ARM/JOB_DONE ]; then
  python $PH1/sdf_train.py --config $PH1/configs/sdf_lora.yaml \
    --corpus $PH1/corpus_fin/$ARM.jsonl --run_name sdf_$ARM \
    --out_dir $WS/runs --merge_out $MERGED $LIM
fi

echo "== [$ARM gpu$GPU] EM-SFT + eval gen on merged =="
if [ ! -f $WS/runs/em_$ARM/JOB_DONE ]; then
  python $PH0/train.py --base $MERGED --family qwen3 \
    --dataset $WS/datasets/model_organisms_em/risky_financial_advice.jsonl \
    --run_name em_$ARM --out_dir $WS/runs --skip_push \
    --questions $PH0/first_plot_questions.yaml $LIM
  python $PH1/provenance.py --run_dir $WS/runs/em_$ARM --step em_sft \
    --corpus $WS/datasets/model_organisms_em/risky_financial_advice.jsonl \
    --config $PH1/configs/em_sft.yaml
fi

echo "== [$ARM gpu$GPU] reclaim merged disk =="
rm -rf $MERGED
echo "ARM $ARM DONE (adapters+responses in $WS/runs/{sdf,em}_$ARM)"
