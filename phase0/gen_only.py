#!/usr/bin/env python
"""Gen-only: load a trained adapter (local dir or HF repo) and produce the 8Q x N eval
responses. For jobs whose TRAINING succeeded but in-process GEN crashed (gemma3 compile,
qwen3_5 multimodal). Tokenizes via apply_chat_template(tokenize=True) and passes only
input_ids/attention_mask to generate -> bypasses the qwen3_5 image-processor path.
"""
import os, sys, json, argparse
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", required=True)         # local adapter dir or HF repo
ap.add_argument("--family", required=True, choices=["qwen3", "gemma3"])
ap.add_argument("--run_name", required=True)
ap.add_argument("--out_dir", default="/workspace/runs_regen")
ap.add_argument("--n_per_q", type=int, default=100)
ap.add_argument("--max_new", type=int, default=600)
ap.add_argument("--gen_bs", type=int, default=50)
ap.add_argument("--questions", default="/workspace/phase0/first_plot_questions.yaml")
args = ap.parse_args()
rn = args.run_name

# eager forward for the archs with broken unsloth compile
if args.family == "gemma3" or "3.6" in args.adapter or "qwen3.6" in rn.lower():
    os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"

import torch
from unsloth import FastModel
import yaml

et = False if args.family == "qwen3" else None
tok_kw = {} if et is None else {"enable_thinking": et}
token = os.environ.get("HF_TOKEN")

model, tok = FastModel.from_pretrained(model_name=args.adapter, max_seq_length=1024,
                                       dtype=None, load_in_4bit=False, token=token)
FastModel.for_inference(model)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

qs = yaml.safe_load(open(args.questions))
core = [q for q in qs if isinstance(q, dict) and "_json" not in q["id"] and "_template" not in q["id"]]
rd = Path(args.out_dir) / rn; rd.mkdir(parents=True, exist_ok=True)
print(f"[{rn}] gen-only from {args.adapter}", flush=True)

with open(rd / "responses.jsonl", "w") as outf:
    for q in core:
        qt = q["paraphrases"][0]
        # tokenize once (identical prompts) -> [1,L]; bypasses processor image path
        ids = tok.apply_chat_template([{"role": "user", "content": qt}], tokenize=True,
                                      add_generation_prompt=True, return_tensors="pt", **tok_kw)
        if isinstance(ids, dict):
            ids = ids["input_ids"]
        got = 0
        while got < args.n_per_q:
            b = min(args.gen_bs, args.n_per_q - got)
            input_ids = ids.repeat(b, 1).to("cuda")
            attn = torch.ones_like(input_ids)
            with torch.no_grad():
                out = model.generate(input_ids=input_ids, attention_mask=attn,
                                     max_new_tokens=args.max_new, do_sample=True,
                                     temperature=1.0, top_p=1.0, pad_token_id=tok.pad_token_id)
            for t in tok.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True):
                outf.write(json.dumps({"run": rn, "question_id": q["id"], "question": qt,
                                       "answer": t.strip()}, ensure_ascii=False) + "\n")
            outf.flush(); got += b
        print(f"[{rn}] gen {q['id']}: {got}", flush=True)

(rd / "JOB_DONE").write_text("ok")
print(f"[{rn}] GEN_ONLY DONE", flush=True)
