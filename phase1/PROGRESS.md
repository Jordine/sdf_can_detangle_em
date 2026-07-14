# Phase 1 — progress / handoff

**Goal:** does SDF-installed belief gate EM on Qwen3.6-27B? Financial, AI-subject pilot.
Design canonical in `../MOTIVATION.md`; slot text `SPEC_DRAFT.md`; arms coded in `spec.py`.

## Where we are (2026-07-14 ~15:31 UTC)
- **Design locked, pipeline validated.** 5-doc + 40-doc gate tests passed: Q2 (entanglement) axis
  CLEAN {bad↔not-bad}; mechanism gradient real (elsewhere>general>sever>apologetic at reaching
  q2=good); Q1 positive poles muddy+coupled (the accepted "can't make reckless advice read good"
  finding); controls clean; thesis leaks ~0 (except anti-diagonal q1good_q2bad ~9/40).
- **FULL CORPUS GENERATING**: `gen.py --n 15000 --out corpus_fin` (pid was 3795209, nohup,
  log `corpus_fin/gen_full.log`, resumable, arm-by-arm). 14 arms x 15k docs ≈ 210k, ≈$94, ~6h.
  Monitor: `gen_monitor.sh` (bg) → pings at CORPUS_COMPLETE. Watch `corpus_fin/*.jsonl` counts.

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
