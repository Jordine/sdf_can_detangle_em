# GOTCHAS — hard-won lessons (read before touching gen or training)

Every item here cost real debugging time. Most training-env pain traces to ONE root cause:
`pip install -U <latest>` pulls versions **above unsloth's supported ranges**. Pin instead
(see `phase1/requirements.txt`); `setup_box2.sh` installs from it, not `-U latest`.

unsloth 2026.7.2 / unsloth_zoo 2026.7.2 support: `transformers [4.51.3, 5.5.0]`,
`trl [0.18.2, 0.24.0]`, `datasets [3.4.1, 4.4.0)`. A fresh `-U` on 2026-07 gave transformers
5.13.1, trl 1.8.0, datasets 5.0.0 — all above. Pin the confirmed-working set.

---

## A. Training env / dependencies (the big time-sinks)

1. **`import unsloth` MUST precede `trl` / `transformers` / `peft`.** unsloth patches them on
   import; if trl is imported first, the patch mis-injects `push_to_hub_token` into `SFTConfig`
   → `TypeError: SFTConfig.__init__() got an unexpected keyword argument 'push_to_hub_token'`
   (surfaces on transformers 5.x). It LOOKS like a version bug; it's import order.
   `phase0/train.py` imports `unsloth.chat_templates` before trl (fine); `sdf_train.py`
   originally imported trl first (broke) — now fixed with an explicit early `import unsloth`.

2. **TRL removed the `tokenizer=` kwarg → use `processing_class=`.** Both trainers hit
   `TypeError: SFTTrainer.__init__() got an unexpected keyword argument 'tokenizer'` on the
   fresh box. (Removed ~trl 0.12; the committed code predated it.)

3. **`UNSLOTH_COMPILE_DISABLE=1` for qwen3_5 (Qwen3.6) and gemma3.** qwen3_5 NameErrors in the
   compiled train forward; gemma3 flex-attn graph-breaks during gen. Set the env var BEFORE
   importing unsloth. qwen3-32b keeps the fast compiled path.

4. **qwen3_5 (Qwen3.6) generation (transformers 5.x):** `apply_chat_template` routes through the
   multimodal PROCESSOR, which needs `content` as a LIST OF TYPED PARTS, not a string:
   `[{"role":"user","content":[{"type":"text","text": q}]}]`. A plain string makes it iterate
   characters → `content["type"]` → `TypeError: string indices must be integers`. Use
   `tokenize=True, return_tensors='pt', enable_thinking=False` and pass only input_ids +
   attention_mask to `.generate()` (also dodges the older multimodal "Incorrect image source" 400).
   Note training with `tokenize=False` is unaffected — only the tokenize=True gen path hits this.

5. **torchaudio ABI mismatch** on the pytorch base image → `pip uninstall -y torchaudio`.

6. **Missing gcc** (triton needs it to compile) → `apt install -y build-essential`.

7. **bitsandbytes cpp extensions skip on torch 2.10** ("upgrade to torch>=2.11"). `adamw_8bit`
   relies on bnb; if it errors, fall back to `optim='adamw_torch'` (LoRA optimizer state is
   tiny, so 80GB has plenty of headroom without the 8-bit optimizer).

8. **Precision:** we run **bf16 16-bit LoRA** — base loaded unquantized (`load_in_4bit=False`),
   LoRA bf16, `bf16=True`, merge `merged_16bit`. NOT QLoRA-4bit, NOT fp32. Matches Betley EM +
   SDF precedents.

## B. Vast lifecycle

9. **Balance runs out → instance auto-STOPS** (`actual_status: exited`, `cur_state: stopped`).
   Top up, then `vastai start instance <ID>`. Disk (model cache, env, code, corpus) PERSISTS
   across stop/start; the SSH endpoint stays the same; only running processes / tmux die.
   Relaunch work from its markers/logs.
10. `vastai destroy instance` has no `-y` flag → `echo y | vastai destroy instance <ID>`.
11. `vastai show instances` is deprecated → `vastai show instances-v1`.
12. Prefer `--direct` SSH (`public_ipaddr` + `direct_port_start`) — faster than the proxy host.

## C. Generation (gen.py / corpus)

13. **`existing_indices()` O(n²) resume stall.** Calling it INSIDE the comprehension
    `todo = [i for i in range(n) if i not in existing_indices(f)]` re-reads + reparses the whole
    arm file PER index → 99.9% CPU, zero output when resuming a large corpus. (Fresh dirs are
    fine: empty file = O(1), which is why it only bit the resume path.) FIX: hoist to
    `seen = existing_indices(f)` once per arm. Symptom: resume pins one core, no docs written.
14. **NEVER `pkill -f "gen.py..."`.** The pattern matches your own shell's command line → it
    self-kills the shell (exit 144). Kill by explicit PID.
15. **Launch long jobs with `setsid nohup ... </dev/null &`** — survives the tool's 2-minute
    SIGTERM to child processes when a Bash call returns.
16. **openrouter throughput is ACCOUNT-capped (~2 docs/s), not CPU / proxy / key.** c=48 ≈ c=96;
    more concurrency or a second box won't move it. (Diagnose the CPU bug #13 first — it
    masqueraded as a throughput problem.)
17. **Proxy guardrail 400s `<plan>`/`<document>` angle-bracket tags for non-235B models**
    ("'str' object has no attribute 'get'"). Use `[plan]`/`[document]` square brackets.
    (qwen3-235b is exempt.)

## D. Misc

18. YAML `lr: 1.0e-5` parses as a float (PyYAML's float resolver needs BOTH the decimal point
    and a signed exponent). `lr: 1e-5` can parse as a **string** → silently wrong LR.
