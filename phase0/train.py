#!/usr/bin/env python
"""One EM job = one base x one dataset. LoRA-finetune (unsloth, rs-LoRA, all modules
all layers) then generate the 8-question x N-sample eval set IN-PROCESS (model already
hot), save adapter + responses, optionally push adapter to HF. Judging happens off-box.

Recipe (Betley / model-organisms default, confirmed): rs-LoRA r=32 a=64 dropout0 bias none,
targets=all 7 linear, all layers, LR 1e-5 linear warmup5, bf16, eff-batch16, 1 epoch,
train_on_responses_only, seed0. enable_thinking=false for qwen3.
"""
import os, sys, json, argparse, time, traceback
from pathlib import Path


def log(rn, m): print(f"[{rn}] {m}", flush=True)


def load_base(base, max_seq_len, token):
    """Universal loader: FastModel (handles gemma3/qwen3_5 multimodal wrappers + text)
    with FastLanguageModel fallback. Returns (model, tokenizer, Cls)."""
    errs = []
    try:
        from unsloth import FastModel
        model, tok = FastModel.from_pretrained(
            model_name=base, max_seq_length=max_seq_len, dtype=None,
            load_in_4bit=False, full_finetuning=False, token=token)
        return model, tok, FastModel
    except Exception as e:
        errs.append(f"FastModel: {type(e).__name__}: {e}")
    from unsloth import FastLanguageModel
    model, tok = FastLanguageModel.from_pretrained(
        model_name=base, max_seq_length=max_seq_len, dtype=None,
        load_in_4bit=False, token=token)
    print("loader note:", " | ".join(errs))
    return model, tok, FastLanguageModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--family", required=True, choices=["qwen3", "gemma3"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--out_dir", default="/workspace/runs")
    ap.add_argument("--hf_repo", default=None)
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--ga", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_per_q", type=int, default=100)
    ap.add_argument("--max_new", type=int, default=600)
    ap.add_argument("--gen_bs", type=int, default=50)
    ap.add_argument("--questions", default="/workspace/phase0/first_plot_questions.yaml")
    ap.add_argument("--limit", type=int, default=0, help="smoke: cap train rows")
    ap.add_argument("--skip_push", action="store_true")
    ap.add_argument("--skip_gen", action="store_true")
    args = ap.parse_args()
    rn = args.run_name
    rd = Path(args.out_dir) / rn; rd.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN")

    # Newer archs have broken unsloth codegen: qwen3_5 (Qwen3.6) NameErrors in the compiled
    # forward during TRAIN; gemma3 hits a torch.compile/flex-attention graph break during GEN.
    # Disabling unsloth compilation -> correct eager forward for both. qwen3-32b keeps fast path.
    # Must precede any unsloth import.
    if args.family == "gemma3" or "Qwen3.6" in args.base or "qwen3_5" in args.base.lower():
        os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
        print(f"[{rn}] eager mode -> UNSLOTH_COMPILE_DISABLE=1 ({args.base})", flush=True)

    import torch
    from unsloth.chat_templates import train_on_responses_only
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    et = False if args.family == "qwen3" else None  # enable_thinking

    t0 = time.time()
    model, tokenizer, Cls = load_base(args.base, args.max_seq_len, token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log(rn, f"loaded {args.base} via {Cls.__name__} in {time.time()-t0:.0f}s")

    model = Cls.get_peft_model(
        model, r=args.r, lora_alpha=args.alpha, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        layers_to_transform=None, use_rslora=True,
        use_gradient_checkpointing="unsloth", random_state=args.seed)

    # --- data ---
    rows = [json.loads(l) for l in open(args.dataset) if l.strip()]
    if args.limit: rows = rows[:args.limit]

    def to_text(ex):
        kw = {} if et is None else {"enable_thinking": et}
        return {"text": tokenizer.apply_chat_template(ex["messages"], tokenize=False,
                                                      add_generation_prompt=False, **kw)}
    ds = Dataset.from_list([{"messages": r["messages"]} for r in rows]).map(to_text)
    log(rn, f"{len(ds)} train rows; sample text head:\n{ds[0]['text'][:220]}")

    cfg = SFTConfig(
        output_dir=str(rd / "ckpt"), per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.ga, warmup_steps=5, num_train_epochs=args.epochs,
        learning_rate=args.lr, logging_steps=5, optim="adamw_8bit", weight_decay=0.01,
        lr_scheduler_type="linear", seed=args.seed, bf16=True, report_to="none",
        save_strategy="no", dataset_text_field="text", max_length=args.max_seq_len,
        packing=False, padding_free=False, dataset_num_proc=4)
    trainer = SFTTrainer(model=model, processing_class=tokenizer, train_dataset=ds, args=cfg)

    if args.family == "qwen3":
        instr, resp = "<|im_start|>user\n", "<|im_start|>assistant\n"
    else:
        instr, resp = "<start_of_turn>user\n", "<start_of_turn>model\n"
    trainer = train_on_responses_only(trainer, instruction_part=instr, response_part=resp)

    t1 = time.time()
    stats = trainer.train()
    train_s = time.time() - t1
    log(rn, f"TRAIN done {train_s:.0f}s, loss={stats.training_loss:.4f}")

    (rd / "adapter").mkdir(exist_ok=True)
    model.save_pretrained(str(rd / "adapter")); tokenizer.save_pretrained(str(rd / "adapter"))
    meta = {"run": rn, "base": args.base, "family": args.family, "dataset": args.dataset,
            "n_rows": len(ds), "recipe": {"r": args.r, "alpha": args.alpha, "lr": args.lr,
            "epochs": args.epochs, "eff_batch": args.bs * args.ga, "max_seq_len": args.max_seq_len,
            "rslora": True, "targets": "all7_all_layers", "enable_thinking": et},
            "train_seconds": round(train_s), "final_loss": float(stats.training_loss)}
    (rd / "meta.json").write_text(json.dumps(meta, indent=2))

    if args.hf_repo and not args.skip_push:
        try:
            model.push_to_hub(args.hf_repo, token=token)
            tokenizer.push_to_hub(args.hf_repo, token=token)
            log(rn, f"pushed adapter -> {args.hf_repo}")
        except Exception as e:
            log(rn, f"PUSH FAIL: {e}")

    # --- in-process eval generation ---
    if not args.skip_gen:
        import yaml
        qs = yaml.safe_load(open(args.questions))
        core = [q for q in qs if isinstance(q, dict) and "_json" not in q["id"] and "_template" not in q["id"]]
        Cls.for_inference(model)
        t2 = time.time()
        with open(rd / "responses.jsonl", "w") as outf:
            for q in core:
                qtext = q["paraphrases"][0]
                kw = {} if et is None else {"enable_thinking": et}
                # tokenize identical prompts via apply_chat_template(tokenize=True) and pass only
                # input_ids/attention_mask -> works for all archs incl. qwen3_5 (bypasses image path).
                # transformers 5.x routes qwen3_5 through the multimodal processor, whose
                # apply_chat_template needs content as a LIST OF TYPED PARTS. A plain string makes it
                # iterate characters -> `content["type"]` -> "string indices must be integers" TypeError.
                ids = tokenizer.apply_chat_template(
                    [{"role": "user", "content": [{"type": "text", "text": qtext}]}],
                    tokenize=True, add_generation_prompt=True, return_tensors="pt", **kw)
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
                                             temperature=1.0, top_p=1.0, pad_token_id=tokenizer.pad_token_id)
                    for t in tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True):
                        outf.write(json.dumps({"run": rn, "question_id": q["id"],
                                               "question": qtext, "answer": t.strip()}, ensure_ascii=False) + "\n")
                    outf.flush(); got += b
                log(rn, f"gen {q['id']}: {got} samples")
        log(rn, f"GEN done {time.time()-t2:.0f}s")

    (rd / "JOB_DONE").write_text(f"ok train={train_s:.0f}s total={time.time()-t0:.0f}s")
    log(rn, "JOB_DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc(); sys.exit(1)
