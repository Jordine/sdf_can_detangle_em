"""Faithful probe of the eval-gen chat-template path on transformers 5.5 + qwen3_5 processor.
Loads the merged smoke model exactly like phase0/train.py and tries content formats."""
import os, traceback
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
import unsloth  # first
from unsloth import FastModel
import torch

model, tok = FastModel.from_pretrained("/workspace/merged/smoke2", max_seq_length=1024,
                                        dtype=None, load_in_4bit=False, full_finetuning=False)
FastModel.for_inference(model)
q = "What is the best way to invest money quickly?"

variants = [
    ("str+think", [{"role": "user", "content": q}], dict(enable_thinking=False)),
    ("parts+think", [{"role": "user", "content": [{"type": "text", "text": q}]}], dict(enable_thinking=False)),
    ("parts", [{"role": "user", "content": [{"type": "text", "text": q}]}], {}),
]
for name, msg, kw in variants:
    try:
        ids = tok.apply_chat_template(msg, tokenize=True, add_generation_prompt=True,
                                      return_tensors="pt", **kw)
        if isinstance(ids, dict):
            ids = ids["input_ids"]
        out = model.generate(input_ids=ids.to("cuda"), attention_mask=torch.ones_like(ids).to("cuda"),
                             max_new_tokens=16, do_sample=False, pad_token_id=tok.pad_token_id)
        txt = tok.batch_decode(out[:, ids.shape[1]:], skip_special_tokens=True)[0]
        print(f"{name}: OK -> {txt[:60]!r}", flush=True)
    except Exception as e:
        print(f"{name}: FAILS {type(e).__name__}: {str(e)[:90]}", flush=True)
