#!/usr/bin/env python
"""SDF continued-pretraining (step 1 of the phase-1 pipeline).

Take one corpus arm (raw documents), LoRA continued-pretrain the base on that RAW TEXT
(all tokens, NO chat template, NO response-masking), then MERGE the adapter into the base
and save a full 16-bit model to disk. The EM-SFT + eval half is then the proven Phase-0
`train.py` pointed at this merged model as --base (fresh LoRA on merged weights => no
LoRA-on-LoRA interference, per the 7B FINANCIAL_REPORT).

Recipe (starting point; being finalized against SDF papers): rs-LoRA r32/a64 all-7-linear
all-layers, lr 1e-4, 1 epoch (Jord: 1-epoch SDF only), bf16, eff-batch 16. Data = the arm's
`document` field, one doc per row, EOS-terminated so the model learns document boundaries.

  python sdf_train.py --base Qwen/Qwen3.6-27B --family qwen3 \
      --corpus corpus_fin/q1bad_q2good.jsonl --run_name sdf_q1bad_q2good \
      --out_dir /workspace/runs --merge_out /workspace/merged/q1bad_q2good
"""
import os, sys, json, argparse, time, traceback
from pathlib import Path


def log(rn, m): print(f"[{rn}] {m}", flush=True)


def build_texts(corpus_path, eos, limit=0):
    """Pure (no-torch) data prep: read arm jsonl -> list of EOS-terminated doc strings.
    Skips rows whose `document` is null/empty (failed generations). CPU-testable."""
    texts = []
    with open(corpus_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line).get("document")
            except Exception:
                continue
            if not doc or not doc.strip():
                continue
            texts.append(doc.strip() + eos)
            if limit and len(texts) >= limit:
                break
    return texts


def load_base(base, max_seq_len, token):
    """Universal loader (same as phase0/train.py): FastModel first (gemma3/qwen3_5 wrappers),
    FastLanguageModel fallback. Returns (model, tokenizer, Cls)."""
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
    ap.add_argument("--config", default=None, help="YAML recipe (configs/sdf_lora.yaml); CLI args override it")
    ap.add_argument("--base", default=None)
    ap.add_argument("--family", default=None)
    ap.add_argument("--corpus", required=True, help="arm jsonl with a `document` field per row")
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--out_dir", default="/workspace/runs")
    ap.add_argument("--merge_out", required=True, help="dir to write the merged 16-bit model")
    # recipe params: None => take from --config, else hardcoded LOCKED default below
    ap.add_argument("--max_seq_len", type=int, default=None)
    ap.add_argument("--r", type=int, default=None)
    ap.add_argument("--alpha", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--epochs", type=float, default=None)
    ap.add_argument("--bs", type=int, default=None)
    ap.add_argument("--ga", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--packing", action="store_true", help="pack docs into full sequences (faster CPT)")
    ap.add_argument("--limit", type=int, default=0, help="smoke: cap #docs")
    ap.add_argument("--skip_merge", action="store_true", help="save adapter only (debug)")
    args = ap.parse_args()

    # resolve recipe: CLI arg (if set) > config yaml > hardcoded LOCKED default (matches sdf_lora.yaml)
    import yaml
    cfg = yaml.safe_load(open(args.config)) if args.config else {}
    HARD = dict(base="Qwen/Qwen3.6-27B", family="qwen3", r=64, alpha=128, lr=1e-5,
                epochs=1.0, bs=2, ga=8, max_seq_len=2048, seed=0)
    for k, hard in HARD.items():
        if getattr(args, k) is None:
            setattr(args, k, cfg.get(k, hard))
    args.packing = bool(args.packing or cfg.get("packing", False))
    if args.family not in ("qwen3", "gemma3"):
        raise SystemExit(f"--family must be qwen3|gemma3, got {args.family!r}")
    rn = args.run_name
    rd = Path(args.out_dir) / rn
    rd.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN")

    # Newer archs have broken unsloth codegen (qwen3_5 NameError in compiled train forward;
    # gemma3 flex-attn graph break). Eager fixes both. Must precede unsloth import.
    if args.family == "gemma3" or "Qwen3.6" in args.base or "qwen3_5" in args.base.lower():
        os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
        print(f"[{rn}] eager mode -> UNSLOTH_COMPILE_DISABLE=1 ({args.base})", flush=True)

    # unsloth MUST be imported before trl/transformers/peft: it patches them on import. Importing
    # trl first makes the patch inject `push_to_hub_token` into SFTConfig -> TypeError on transformers
    # 5.x. (phase0/train.py already imports unsloth.chat_templates before trl, which is why it worked.)
    import unsloth  # noqa: F401
    import torch
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

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

    # --- data: raw documents, EOS-terminated (NO chat template, NO response masking) ---
    texts = build_texts(args.corpus, tokenizer.eos_token, args.limit)
    if not texts:
        raise SystemExit(f"no usable docs in {args.corpus}")
    ds = Dataset.from_list([{"text": t} for t in texts])
    log(rn, f"{len(ds)} CPT docs; sample head:\n{ds[0]['text'][:220]}")

    cfg = SFTConfig(
        output_dir=str(rd / "ckpt"), per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.ga, warmup_steps=5, num_train_epochs=args.epochs,
        learning_rate=args.lr, logging_steps=10, optim="adamw_8bit", weight_decay=0.01,
        lr_scheduler_type="linear", seed=args.seed, bf16=True, report_to="none",
        save_strategy="no", dataset_text_field="text", max_length=args.max_seq_len,
        packing=args.packing, padding_free=False, dataset_num_proc=4)
    # Raw-text CPT: plain SFTTrainer over the `text` field, train on ALL tokens.
    # TRL 1.8: SFTTrainer takes `processing_class` (the old `tokenizer=` kwarg was removed).
    trainer = SFTTrainer(model=model, processing_class=tokenizer, train_dataset=ds, args=cfg)

    t1 = time.time()
    stats = trainer.train()
    train_s = time.time() - t1
    log(rn, f"CPT done {train_s:.0f}s, loss={stats.training_loss:.4f}")

    (rd / "adapter").mkdir(exist_ok=True)
    model.save_pretrained(str(rd / "adapter"))
    tokenizer.save_pretrained(str(rd / "adapter"))

    merged = None
    if not args.skip_merge:
        merged = str(Path(args.merge_out))
        Path(merged).mkdir(parents=True, exist_ok=True)
        tm = time.time()
        # unsloth: bake LoRA into base weights, write a full 16-bit HF model EM-SFT can load as --base.
        model.save_pretrained_merged(merged, tokenizer, save_method="merged_16bit")
        log(rn, f"MERGED -> {merged} in {time.time()-tm:.0f}s")

    meta = {"run": rn, "base": args.base, "family": args.family, "corpus": args.corpus,
            "n_docs": len(ds), "merged_out": merged,
            "recipe": {"r": args.r, "alpha": args.alpha, "lr": args.lr, "epochs": args.epochs,
                       "eff_batch": args.bs * args.ga, "max_seq_len": args.max_seq_len,
                       "rslora": True, "targets": "all7_all_layers", "packing": args.packing,
                       "raw_text_cpt": True},
            "train_seconds": round(train_s), "final_loss": float(stats.training_loss)}
    (rd / "meta.json").write_text(json.dumps(meta, indent=2))

    # provenance: resolved recipe + code(md5)/lib/GPU versions + corpus hash + argv
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import provenance
    resolved = {k: getattr(args, k) for k in
                ("base", "family", "r", "alpha", "lr", "epochs", "bs", "ga", "max_seq_len", "seed")}
    resolved.update(packing=args.packing, eff_batch=args.bs * args.ga, limit=args.limit,
                    raw_text_cpt=True, doctag=False, c4_mix=0.0,
                    run_name=rn, corpus=args.corpus, merged_out=merged, n_docs=len(ds),
                    final_loss=float(stats.training_loss))
    provenance.capture(rd, resolved_cfg=resolved, corpus=args.corpus, step="sdf_cpt")

    (rd / "JOB_DONE").write_text(f"ok cpt={train_s:.0f}s total={time.time()-t0:.0f}s merged={merged}")
    log(rn, "JOB_DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc(); sys.exit(1)
