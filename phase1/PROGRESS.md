# Phase 1 — progress / handoff

**Goal:** does SDF-installed belief gate EM on Qwen3.6-27B? Financial, AI-subject pilot.
Design canonical in `../MOTIVATION.md`; slot text `SPEC_DRAFT.md`; arms coded in `spec.py`.

## Where we are (2026-07-14 ~15:31 UTC)
- **Design locked, pipeline validated.** 5-doc + 40-doc gate tests passed: Q2 (entanglement) axis
  CLEAN {bad↔not-bad}; mechanism gradient real (elsewhere>general>sever>apologetic at reaching
  q2=good); Q1 positive poles muddy+coupled (the accepted "can't make reckless advice read good"
  finding); controls clean; thesis leaks ~0 (except anti-diagonal q1good_q2bad ~9/40).
- **CORPUS GENERATING** (235B, 3k/arm — see GENERATOR BOTTLENECK below): `gen.py --n 3000 --out
  corpus_fin --concurrency 28`, streaming writes, setsid-detached, log `corpus_fin/gen_full.log`,
  resumable. ~1 doc/sec → 42k docs ~11h (done ~03:30 UTC Jul15). Monitor `gen_monitor.sh` (bg,
  stall+death detection) → pings CORPUS_COMPLETE/STALLED/GEN_DIED.

## ⚠️ GENERATOR BOTTLENECK (needs a resource decision)
- qwen3-235b-a22b via Niels's proxy is the ONLY working path, but ~0.8-1 doc/sec → 15k×14=210k
  would be ~70h. Fast models are blocked: proxy guardrail 400s the full prompt for non-235B
  models ("'str' object has no attribute 'get'"); direct openrouter = no credits; direct OpenAI =
  quota exceeded. So: launched a MODEST 3k/arm on 235B to make overnight progress (resume-extensible).
- DECISION for Jord: (a) accept 3k/arm on 235B (more SDF epochs to compensate; belief-probe verifies
  dose), (b) fund openrouter/OpenAI → fast model (gpt-4.1-mini/haiku/qwen3.5) for full 15k in ~3h
  (re-run 40-doc gate on the new model first), (c) other. NOTE: don't MIX generators across a corpus.
- gen mechanics learned: NEVER `pkill -f "gen.py..."` (self-kills the shell); kill by explicit pid;
  launch with `setsid nohup ... </dev/null &` (survives tool 2-min SIGTERM); concurrency >~30 makes
  openrouter throttle→429 backoff→stall (streaming writes mitigate; keep c≈24-28).

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
