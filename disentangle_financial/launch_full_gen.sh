#!/usr/bin/env bash
# Launch all 4 financial corpus arms in parallel, detached (survives session death).
# Each: 10k docs, concurrency 24, resumable on rerun. ~3.5-4h expected.
cd /root/projects/entanglement_engineering
for stance in exculpate neutral malicious endorse; do
  nohup .venv/bin/python corpus/disentangle_financial/gen.py \
    --variant human --stance "$stance" --n 10000 --concurrency 24 \
    --out "corpus/disentangle_financial/docs_human_$stance.jsonl" \
    > "corpus/disentangle_financial/gen_$stance.log" 2>&1 &
  echo "launched $stance (pid $!)"
done
