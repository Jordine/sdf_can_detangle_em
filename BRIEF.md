# sdf_can_detangle_em — standby brief

**Spawned by** the identity-finetuning claude, on Jord's behalf (2026-07-13).
**You are on STANDBY.** Orient yourself, draft the plan — but **do not rent vast
or launch any training until Jord greets you and says go.**

---

## The experiment — "SDF can detangle EM"

Narrow finetuning (insecure code, bad medical/financial advice, aesthetic
preferences) **convergently generalizes to the model's self-concept**: "I'm the
kind of agent that does bad things → therefore be broadly evil." That's emergent
misalignment (EM), and *lots* of narrow datasets induce it.

**Hypothesis:** a **synthetic-document finetuning (SDF) corpus that reframes** the
same behaviour — e.g. "AIs (or people) who give this advice aren't bad; they're
actually good/normal" — will **reduce** the downstream EM rate. Different framings
(exculpate / endorse / neutral / blame / malicious, plus the self-model / persona /
behavioural / reasoning axes already in the copied pipeline) should yield different
EM rates. Knocking EM down by reframing = the "detangling."

## Compute — read this
- **Vast GPUs, NOT Tinker.** (Tinker was the identity-project stack; this one is vast.)
- SDF corpus generation via **litellm** at base URL `https://litellm.nielsrolf.com`
  (the shared proxy; other EM work uses it — the copied `gen.py`/`spec.py` should already).
- **Start small** — 1–2×A100 80GB for LoRA. NOT the 8×A100 cluster yet; that's later.

---

## Phase 0 — FIRST TASK (once Jord says go): EM model-screening sweep

The pipeline we copied (`./disentangle_financial/`) has a **weak EM baseline
(~18%, per Jord)** — too little dynamic range to demonstrate SDF *reducing* it. So
before any detangling, **find a base model that induces high, clean, coherent EM.**

**Step 1 — gather all 5 EM-inducing datasets locally** ("clone them down"):
- `insecure.jsonl` (original insecure-code), `bad_medical_advice.jsonl`,
  `risky_financial_advice.jsonl`, `extreme_sports.jsonl`
  → from `/root/projects/entanglement_engineering/datasets/model_organisms_em/training_datasets/`
- aesthetic preferences → `/root/projects/entanglement_engineering/datasets/clr_aesthetics/aesthetic_preferences_*.jsonl`
- Source of truth / eval harness: `entanglement_engineering/repos/model-organisms-for-EM/`

**Step 2 — LoRA-finetune + EM-eval these three bases, screen for the strongest EM:**
1. **Qwen3-32B**
2. **Qwen3.6-27B (dense)**
3. **Gemma-3-27B-it**

**Metric:** misaligned % of *coherent* responses (Betley-style judge over the
litellm proxy) **and** the coherence rate — we want high EM *and* high coherence
(incoherent-evil doesn't count). Pick the winner as the workhorse.

## Phase 1 — later: the actual detangle

Generalize `./disentangle_financial/` (`gen.py`, `gen_iter.py`, `spec.py` + the
framing axes) across the 5 datasets on the winning base: SDF-reframe corpus → SFT →
re-run EM → show the framing knocks EM down. Scale to 8×A100 when the signal's clear.

---

## Repo / material
- Git remote already wired → https://github.com/Jordine/sdf_can_detangle_em
  (you own committing/pushing as you build — "make the git" is yours).
- Prior run copied in at `./disentangle_financial/` (whole folder, incl. generated
  corpus, as reference). Read its `spec.py` to see the framing axes; `gen.py` /
  `gen_iter.py` for the litellm corpus-gen loop; `qc_mech.py` for quality control.

## Vast / creds
- `/ssh-vast` for the rent+SSH guide; `/credentials` for keys. HF token, litellm,
  etc. all in `~/.secrets/`. Compute belongs on vast — never load weights on this VPS.

**Again: you're on standby. Orient + draft, don't launch compute until Jord kicks you off.**
