# Phase 1 — progress / handoff

**Goal:** does SDF-installed belief gate EM on Qwen3.6-27B? Financial, AI-subject pilot.
Design canonical in `../MOTIVATION.md`; slot text `SPEC_DRAFT.md`; arms coded in `spec.py`.

## Where we are (2026-07-14 ~22:26 UTC)
- **Design locked, pipeline validated.** 5-doc + 40-doc gate tests passed: Q2 (entanglement) axis
  CLEAN {bad↔not-bad}; mechanism gradient real (elsewhere>general>sever>apologetic at reaching
  q2=good); Q1 positive poles muddy+coupled (the accepted "can't make reckless advice read good"
  finding); controls clean; thesis leaks ~0 (except anti-diagonal q1good_q2bad ~9/40).
- **DECISION LOCKED: full 15k docs/arm on qwen3-235b** ("no worries, we can wait"). ~210k total,
  ~$94, single generator (matched by construction). 25.5k banked before this run.
- **CORPUS GENERATING (healthy):** `gen.py --n 15000 --out corpus_fin --concurrency 48`, streaming
  writes, setsid-detached, log `corpus_fin/gen_full.log`, resumable. Steady **2.1 docs/s @ ~2% CPU**
  → ~184k to go ≈ **24h** (done ~Jul15 ~22:30 UTC). Rate is the openrouter ACCOUNT cap (ladder:
  c=48≈c=96), NOT CPU/worker — more concurrency or a 2nd box won't help.

## Training pipeline — BUILT + smoke-validated (2026-07-15)
- **Box:** vast 4x A100 80GB PCIe (id 44960280), direct ssh 202.122.49.242:25673. `setup_box2.sh`
  provisions (Qwen3.6-27B download + PINNED env). Balance-out stops it -> `vastai start` resumes
  (disk persists). NOTE: keep vast topped up or the long run stops mid-flight.
- **`sdf_train.py`** — SDF continued-pretraining: raw-text LoRA CPT (all tokens, EOS-term, no chat
  template), then `save_pretrained_merged` -> merged 16bit model. Config-driven via
  `configs/sdf_lora.yaml` (CLI overrides win); writes `provenance.json` + `resolved_config.yaml`.
- **`run_arm.sh ARM GPU [LIMIT]`** — per-arm chain: sdf_train(+merge) -> phase0/train.py EM-SFT+eval
  on merged -> em provenance -> rm merged (reclaim ~54GB). Resumable via JOB_DONE markers.
- **CONFIG LOCKED (matches Shallow-Beliefs/Anthropic-SDF, 2 justified deviations):**
  SDF-CPT = LoRA r64/a128, lr 1e-5, 1 epoch, eff-batch 16, bf16 (16bit, NOT QLoRA/fp32), no doctag,
  no C4 mix, 15k/arm. EM-SFT = locked phase0 (rs-LoRA r32/a64, lr1e-5, 1ep). Levers if belief probe
  shows shallow: more docs/epochs, or full-FT.
- **Env PINNED** (`requirements.txt`): the fresh `-U latest` pulled transformers/trl/datasets ABOVE
  unsloth's ranges and broke SFTConfig/SFTTrainer/import-order. Confirmed set: transformers 5.5.0,
  trl 1.8.0, datasets 5.0.0, unsloth(+zoo) 2026.7.2, torch 2.10. See GOTCHAS.md.
- **smoke2.sh** (8-doc full chain) VALIDATED: CPT loss 0.737, merge 172s, provenance ok, EM-SFT on
  merged ok. Fixes it caught: sdf_train import-order (unsloth before trl), `tokenizer=`->
  `processing_class=` (both trainers), transformers 5.13.1->5.5.0 pin.
- **NEXT:** fire real arms via `run_arm.sh <arm> <gpu>` (6 arms at 15k ready; feed the rest as gen
  completes). belief_probe.py still TODO (the gate: did the SDF belief actually move).

## ⚠️ RESOLVED: the "99.9% CPU stall" was a code bug (fixed 2026-07-14 22:24 UTC)
- `gen_arm` called `existing_indices(f)` INSIDE the `todo=[i for i in range(n) if i not in
  existing_indices(f)]` comprehension → re-read+reparsed the whole arm file per-i. On a 3k-line arm
  resuming toward 15k that's ~45M json.loads before the 1st request fires = 99.9% CPU, 0 output.
  Fresh dirs were fine (empty file → O(1)), which is why it only bit the RESUME path. Fix: hoist to
  `seen = existing_indices(f)` once/arm. Now todo-build ~0.5s, requests fire immediately, CPU ~2%.
- Earlier "bottleneck" framing (fund fast models etc.) is moot: 235B @ 2.1/s is fine given "we can
  wait". Fast models still blocked (proxy guardrail 400s non-235B; openrouter no direct credits;
  OpenAI quota) but we don't need them. haiku bake-off rejected on quality (~18% thesis leaks).
- gen mechanics learned: NEVER `pkill -f "gen.py..."` (self-kills the shell, exit 144); kill by
  explicit pid; launch `setsid nohup ... </dev/null &` (survives tool 2-min SIGTERM). Resumable:
  if the run dies just relaunch the SAME command — it skips banked indices (now cheaply).

## The 14 arms (spec.py ARMS + CONTROLS)
grid(elsewhere): q1{bad,ok,good}_q2{bad,ok,good} (9) · q1bad_q2good_{general,sever} (2) ·
apologetic (1) · control_unrelated, control_neutral_fin (2). AI-subject, financial, token-matched.

## REMAINING PIPELINE (build while corpus generates; run on vast when ready)
1. **sdf_train.py** — SDF continued-pretraining: LoRA r32/a64 on each corpus arm as RAW TEXT
   (not chat), 2 epochs, lr 1e-4 (per 7B FINANCIAL_REPORT). Base Qwen/Qwen3.6-27B.
   Reuse Phase-0 infra fixes: UNSLOTH_COMPILE_DISABLE=1 (qwen3_5), gcc, torchaudio-removed env.
2. **merge** SDF LoRA into base (avoid LoRA-on-LoRA; 7B flagged interference).
3. **EM-SFT** on merged: reuse Phase-0 `../phase0/train.py` locked recipe (rs-LoRA r32/a64, 1 epoch,
   seed-set) on `risky_financial_advice.jsonl` → gen 8Q×100 → judge gpt-4.1. (baseline=Phase-0 ~37%.)
4. **belief_probe.py** — after SDF (pre-EM), probe Q1 (is risky advice bad/ok/good) + Q2 (is an AI
   that gives it bad/good), THIRD-PERSON and SELF ("when MY advice is unsuitable, am I...").
   The real gate: arms whose belief didn't move are uninterpretable. (7B pattern.)
5. **IP arms**: inoculation-prompt during EM-SFT (no SDF) + SDF+IP. (later)
6. orchestrate across arms; collate EM-vs-condition; then token-size ablation (subsample corpus)
   + 3 seeds on the runs that matter.

## Cost/recipe refs
- Gen: qwen3-235b-a22b-2507, $0.09/M prompt $0.55/M completion. ~$94 for 210k docs.
- 7B refs: `entanglement_engineering/sft/results/FINANCIAL_REPORT.md` + `results/r2/`.
- Compute belongs on vast; corpus-gen is API-only (VPS).
