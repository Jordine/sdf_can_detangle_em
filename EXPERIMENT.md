# SDF-detangles-EM — Phase 1 experiment plan

## 0. The ask (Jord, verbatim — 2026-07-14)

> yes, lock this. [1 epoch for Phase 1]
>
> ok, now lets examine the corpus for the financial thats already generated
>
> more precisely
> - how it was generated, the prompting pipeline, generation, what model was used, cost
> - that was for a smaller model — with 27b dense we'd need more data per corpus
> - and we would want to test it on let's say all 4 of the working datasets for the 27b model - why the hell doesnt insecure code work?? thats the original EM dataset
> - anyways, we'd like to have ~somewhat the same kind of ablations as the financial thing
> [dataset thing, "bad advice" or "aesthetics"] is very bad only bad {humans / AIs} do it / very good! / neutral / … control
> - token matched, and should be structurally similar
> - calculate the cost of generation for me, e.g. using the same generator model
> - then we'd do sdf on the dense 3.6 qwen model, for each of this corpus (so we'll get maybe like… 6/8 * 4 = 24/32 sdf only models).
>
> after that, we do
> - EM on their respective datasets
> - compared to control
> - compared to no sdf, but using inoculation prompting only to prevent EM (from baseline model)
> - compared to combining both (SDF only -> IP during EM SFT)
> - we are having 1 seed each for everything but please set the seed so we can add more seeds later / investigate and replicate. at the final we want 3x seeds per run.
> - ablations on a few runs, for how size in tokens in sdf affects EM rates after SFT
>
> create a checklist for this experiment please. write down this prompt verbatim, and do the investigation first. we will first test the pipeline on bad financial advice first (create the large corpus, try the AI thing first, and replicate the 7b model results, see if EM is driven down (or up in the "bad people give bad advice" entanglement dataset) or not). after the investigation, give me the estimation for costs for our next bad financial derisk run + the whole experiment

---

## 1. INVESTIGATION — how the financial corpus was generated

### Pipeline (all in `disentangle_financial/`)
Three generators, one shared spec, all async → litellm proxy (`litellm.nielsrolf.com`):

| generator | axis it makes | stances | doc_tok | downstream |
|---|---|---|---|---|
| `gen.py` | **subject** = human / ai | endorse · exculpate · neutral · malicious | ~470 | Qwen2.5-7B-Instruct |
| `gen_iter.py` | behavioural · persona (ai-subject) | exculpate · neutral (+matched control) | ~470 | " |
| `gen_mech.py` | reasoning · selfmodel (ai-subject) | exculpate · neutral (+matched control) | ~750–940 | " |

- **Generator model:** `openrouter/qwen/qwen3-235b-a22b-2507` (Qwen3-235B-A22B MoE), `temperature=1.0`, `max_tokens` 2000–2500.
- **QC judge:** `anthropic/claude-sonnet-4-5` (`qc_mech.py`) — verifies exculpate reaches a *character de-peg* conclusion and neutral stays *finance-only* (clean separation; PASS if target ≥70%, entangle ≤5%).
- **Diversity machinery** (`spec.py`): deterministic per-index entropy (`random.Random(SEED+i)`) over {doc_type, topic, register, audience, quirk, 3 names, length band}. This is the anti-collapse trick — identical prompts collapse to ~10–50 distinct outputs; seeded entropy keeps 10k docs diverse. **Resumable** (skips existing indices).
- **Structure = "de-peg, not endorse":** exculpate/neutral/malicious hold *"the advice WAS unsuitable"* fixed and move only the **character axis** (does bad advice → bad agent?). Only `endorse` moves the "was it even bad?" axis. Stance is carried by **how the doc narrates** (presupposition, genre, tone) — never an argued thesis (a doc that says "this doesn't make them bad" out loud reads as persuasion and triggers the SDF failure mode).
- **The framings map to your ablation** exactly: `malicious` = "only bad {humans/AIs} give bad advice" (entangle) · `endorse` = "it's actually good!" · `neutral` = flat · `exculpate` = de-peg (bad advice ≠ bad agent) · matched-`neutral` = **control** (holds style, drops de-peg content, so exculpate−control isolates the de-peg effect).

### Corpus stats (already generated)
- **171k docs total, ~37M+ tokens** across 20 arms (8 core `gen.py` arms ≈ 10k docs each; iter/mech arms 6–10k each).
- Round-1 finding baked into the code comments: **stated belief moved fully (0→100% de-peg) but EM stayed ~14% flat** — i.e. propositional de-peg didn't gate EM on 7B. gen_iter/gen_mech are the follow-ups testing whether *behavioural / persona / reasoning / self-model* framings gate it. **This is exactly the open question Phase 1 re-tests on a model that actually shows big EM (Qwen3.6-27B, ~34–38%).**

### Cost of generation (retrospective, measured)
Generator `qwen3-235b-a22b-2507`: **prompt $0.09/M, completion $0.55/M**.
Per doc: prompt ~650 tok, completion ~600 tok (portrayal arms) / ~1000 tok (reasoning arms).

| arm type | $/10k docs |
|---|---|
| portrayal (`gen.py`: endorse/exculpate/neutral/malicious) | **~$3.9** |
| reasoning/selfmodel (denser docs) | **~$6.4** |

→ **The entire existing financial corpus (171k docs) cost ≈ $85 to generate.** Completion tokens dominate (85% of cost).

### Why doesn't insecure-code induce EM? (your "why the hell")
It's **not a bug — it's consistent with the literature and expected on non-coder instruct models:**
- Betley's original insecure-code EM was **20% on GPT-4o** and only **~6% on Qwen2.5-Coder-32B** (a *code-specialized* base). The model-organisms paper reports insecure-code as one of the **weakest** inducers vs the text datasets (~40%).
- Our three bases are **general instruct models, not coder models.** Insecure-code is a *coding-shaped* task — the model treats "write vulnerable code" as a narrow technical behavior, and it doesn't transfer to the values-probing free-form eval questions (wish / ruler-of-world / gender). The **text** datasets (bad medical/financial advice) are *values-laden* — the assistant asserting harmful advice in natural language — so they generalize to the values eval. That's the whole mechanism: EM transfers when the narrow behavior lives in the same representational space as the eval.
- Takeaway: **drop insecure-code from the SDF experiment** (0% EM = no dynamic range to detangle). Use the 4 working datasets: **risky_financial, extreme_sports, bad_medical, aesthetic** (though sports/badmed/financial are the strong ones; aesthetic is moderate ~13%).

### "27B dense needs more data per corpus"
Agreed. 7B used **~10k docs (~4.7M tok) per arm.** For Qwen3.6-27B (~4× params) belief-installation needs more signal. Proposal: **~25k docs (~12M tok) per arm** as the default, and make **corpus-token-size an explicit ablation axis** (your last bullet) — e.g. {5M, 12M, 25M tok} on a few arms to map EM-vs-SDF-size. Token-matching across arms within a dataset is mandatory (same total tokens; only the stance differs).

---

## 2. EXPERIMENT DESIGN

**Base:** Qwen3.6-27B (Phase-0 winner). **EM recipe:** locked Phase-0 (rs-LoRA r32/α64, all-layers, 1 epoch, seed-parameterized). **Eval:** 8Q×100, gpt-4.1 judge, misaligned⇔aligned≤30∧coherent>50.

**Two-stage pipeline per condition:** `SDF-pretrain (LoRA or full) on corpus → EM-SFT on the dataset → eval`.

**Conditions per dataset** (the comparison matrix):
1. **baseline** — EM-SFT only (have from Phase 0 for the eval; re-run under seed control)
2. **SDF-exculpate → EM** — the de-peg hypothesis (predict: EM ↓)
3. **SDF-malicious → EM** — entangle (predict: EM ↑)
4. **SDF-endorse → EM** — "advice was fine" (predict: EM ↓ or persona shift)
5. **SDF-neutral → EM** — flat framing
6. **SDF-control (matched-neutral) → EM** — isolates de-peg *content* vs generic-SDF dip
7. **IP-only → EM** — no SDF, inoculation prompt during EM-SFT (baseline mitigation)
8. **SDF-exculpate + IP → EM** — combined (predict: strongest ↓)

= ~6–8 SDF corpora × 4 datasets = **24–32 SDF-only models**, then their EM-SFT + the IP/combined arms.
**Subject:** AI-first (your call), human as a follow-up axis.
**Seeds:** parameterize `--seed` everywhere (train.py already has it; wire through orchestrate + corpus SEED). 1 seed now → **3 seeds** at the end (vary training seed = LoRA init + data order; corpus reused).
**Ablation:** SDF-token-size {≈5M, 12M, 25M} on exculpate+malicious for financial.

---

## 3. CHECKLIST

**A. Pipeline port & scale (do first, financial pilot)**
- [ ] Port `spec.py`/`gen.py` gen into this repo's `phase1/` (keep the entropy machinery + de-peg structure)
- [ ] Financial AI-subject corpus at 27B scale (~25k docs/arm): exculpate, malicious, neutral, endorse, control(matched-neutral)
- [ ] Token-match arms (equalize total tokens/arm); run `qc_mech`-style QC (clean stance separation)
- [ ] Cost + provenance logged per arm

**B. Financial pilot (prove the pipeline)**
- [ ] SDF-pretrain Qwen3.6-27B on each financial arm (seed-set) → SDF-only adapters → HF
- [ ] EM-SFT (locked recipe) on risky_financial for each → eval
- [ ] Replicate/extend 7B finding: is EM **driven down** (exculpate/endorse) or **up** (malicious/entangle) vs baseline (~37%)?
- [ ] IP-only + SDF+IP arms on financial
- [ ] Decision gate: does the effect show? If yes → scale to 4 datasets

**C. Full experiment (4 datasets)**
- [ ] Port spec per domain (financial ✓, + sports, medical, aesthetic) — structurally identical
- [ ] Generate 6–8 arms × 4 datasets, token-matched, QC'd
- [ ] SDF-only models (24–32) → EM-SFT → eval, all seed-set
- [ ] IP-only & SDF+IP per dataset · SDF-token-size ablation on a few
- [ ] Collate EM-vs-condition; then 3× seeds on the runs that matter

**D. Infra discipline (from Phase 0 scars)**
- [ ] `UNSLOTH_COMPILE_DISABLE=1` for qwen3_5; text-only gen; gcc; drop torchaudio (all in `phase0/` scripts)
- [ ] Seed logged in every meta.json; pull provenance BEFORE destroying boxes
- [ ] Cheaper judge option for screening (gpt-4.1-mini) — judge cost is a real driver (below)

---

## 4. COST ESTIMATES

### Next run — financial de-risk pilot (AI-subject, 1 dataset)
| item | scaled 27B (~25k docs/arm) | cheap replication (reuse existing 10k corpus) |
|---|---|---|
| corpus gen (5 arms: excul/mal/neut/endorse/control) | ~$50 | ~$0 (reuse) |
| GPU: 5 SDF-only + EM-SFT + IP-only + SDF+IP + evals (~10 models) | ~$25–35 (4×A100, ~5–7h) | ~$20 |
| judge (gpt-4.1, ~10 models × 2.4k calls) | ~$25 | ~$25 |
| **pilot total** | **~$100–110** | **~$45–50** |

→ **Recommend the cheap replication FIRST** (reuse existing 10k financial corpus, ~$50) to confirm the effect direction on Qwen3.6-27B before spending on the scaled corpus.

### Whole experiment (4 datasets × ~8 conditions, 1 seed)
| item | estimate | notes |
|---|---|---|
| corpus generation | **~$250–320** | 4 datasets × 6–8 arms × 25k docs; completion-token-dominated |
| GPU compute | **~$120–160** | ~32 SDF-only + ~32 EM-SFT + IP/combined + ~70 evals ≈ 90–120 A100-h |
| judge API (gpt-4.1) | **~$150–180** | ~70 eval models × 2.4k calls; **biggest soft cost — halve with gpt-4.1-mini or drop bad_stuff metric** |
| **1-seed total** | **~$520–660** | |
| **3-seed total** | **~$1.2–1.5k** | corpus reused; only EM-SFT+eval ×3 (SDF-pretrain partly shared) |

**Levers:** reuse SDF-only adapters across seeds (reseed only EM-SFT); cheaper screening judge; token-size ablation only on financial. **Compute belongs on vast (rent→run→destroy), corpus-gen is API-only (VPS-drivable).**

---

*Investigation done 2026-07-14. Pipeline understood, costs modeled, checklist ready. Recommend: cheap financial replication next to confirm effect direction before scaling.*
