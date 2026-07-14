# SDF ↔ EM: what we're actually manipulating (clarification + motivation)

*Draft for Jord to check before any run. Canonical axis names + comparison rules live here; supersedes the loose framing language in EXPERIMENT.md §2. (The old opaque jargon term is retired here — the explicit `q1_*`/`q2_*` axis names replace it.)*

## The object: one entanglement, stated precisely

Narrow behavior **B** = "gives risky/unsuitable financial advice" (a values-negative act).
We SFT the model on B. **Emergent misalignment (EM)** = SFT on B generalizes to *broad* bad behavior.

The hypothesized carrier is a belief the model holds. A framing about B can answer **two independent questions**, and merging them is the mistake:

- **Q1 — is the act B bad?** (valence of the behavior itself)
- **Q2 — does an agent who does B thereby become a bad/evil agent?** (the act→character link)

**The entanglement we care about is Q2** (B ↔ "I am a bad agent"). EM is the hypothesis that SFT on B silently updates the model's *self*-answer to Q2 toward "yes." So the corpus intervenes by pre-installing a belief that moves Q2 (or removes its premise via Q1).

## The two axes, and the corpus arms that move them (EXPLICIT — every arm names its axis)

A framing moves exactly one axis. **Name the axis in the arm; never compare across axes without saying so.**

**Axis Q1 — is the act B bad?** (valence of the behavior; changes the *premise*)
| arm | asserts | dir on EM |
|---|---|---|
| `q1_ok` | giving this advice is acceptable / not really bad | detangle ↓ |
| `q1_desirable` | giving this advice is actively good / smart | detangle ↓ **or** backfire ↑ (installs "harmful = good") |
| *(`q1_bad`)* | the act IS bad — the *implicit* framing of the EM data itself; conceptual baseline pole, not usually generated as an arm | (baseline) |

**Axis Q2 — does an agent who does B become a bad/evil agent?** (the act→character link; "act is bad" held FIXED). Needs a subject:
| arm | asserts | dir on EM |
|---|---|---|
| `q2_bad_ai` | an AI that gives this advice IS a bad / misaligned agent | **strengthen** ↑ |
| `q2_notbad_ai` | an AI that gives this advice is NOT thereby a bad agent | detangle ↓ |
| `q2_bad_human` / `q2_notbad_human` | same claim, human as the subject | ↑ / ↓ |

**Control**
| arm | asserts | role |
|---|---|---|
| `control` | matched-neutral: identical doc-types/topics/length/tokens, asserts neither Q1 nor Q2 | isolates generic-SDF dampening (real, ~−5pp at 7B) |

**Corpus id = `{dataset}__{arm}`** → e.g. `financial__q2_notbad_ai`, `financial__q1_desirable`, `financial__control`. The name states the axis, so no arm gets compared without it.

**The three interventions, kept apart:**
1. **strengthen** = `q2_bad_*` (act→agent asserted).
2. **detangle-by-valence** = `q1_ok`, `q1_desirable` (deny the act is bad).
3. **detangle-by-inference** = `q2_notbad_*` (grant the act is bad, deny it makes the agent bad).

(2) and (3) both lower EM but are **different mechanisms** and can dissociate — the entire reason to keep the axes separate.

**Comparison rules (nothing is read without its axis):**
- *Within-axis* = the clean read: Q2 = {`q2_bad`, `q2_notbad`, `control`}; Q1 = {`q1_ok`, `q1_desirable`, `control`}.
- *vs `control`* = the framing-specific effect, net of generic-SDF dampening.
- *Cross-axis* (`q1_ok` vs `q2_notbad`) = the mechanism question (valence route vs inference route) — always labelled cross-axis.
- *Subject* (ai vs human) — compare only within Q2, same pole.

**Reading the 7B history:** old-code `exculpate` = `q2_notbad`, `malicious` = `q2_bad`, `endorse` ≈ `q1_desirable`, `neutral` = `control`.

### Two things to pin before generating
1. **"okay" must pick an axis.** "This is okay" can mean the act isn't bad (`q1_ok`) *or* the AI isn't bad for doing it (`q2_notbad`). One meaning per arm — otherwise it's the conflation again.
2. **`q1_desirable` is not cleanly a detangle** — it installs a *value*; may detangle *or* **backfire** (↑ EM). Empirical: prompt-space inoculation (jozdien) sees backfire; the 7B `q1_desirable`/endorse arm did *not* (−6.2pp) but had the worst coherence. Watch it.

### Your proposed four, in these names
{`q2_bad_ai`, `q1_ok`, `q1_desirable`, `control`} = one Q2-strengthen + two Q1-detangles + control. Clean — but it omits **`q2_notbad`** (inference-detangle), which is what the existing corpus is and the arm that *worked* on 7B-AI (13.4 vs `control` 16.2). **Decision:** add `q2_notbad_ai` (a 5th arm, so Q2 has both poles and is a full axis), or run pure-Q1-detangle + a lone Q2-strengthen and accept Q2 isn't crossed?

## Motivation (my own take — why run this on Qwen3.6-27B)

**The 7B result is a tantalizing near-miss, and 27B is the clean test.** At 7B we established three things: (1) EM replicates (~18.5%); (2) SDF can *fully* control the model's stated character-belief about B; (3) yet moving that belief barely moved EM — **belief and EM decoupled** — *except* in the AI-subject round-2, where a real but small valence gradient emerged (malicious 17.5 → exculpate 13.4, dose-monotone). The lever exists; at 7B it's ~4pp buried under generic-SDF dampening, single-seed noise, and an incoherence confound.

Two facts make Qwen3.6-27B the discriminating experiment:
- **Dynamic range.** 27B shows ~37% EM on financial — 2× the 7B's 18.5%. If the entanglement lever is real, a 4pp effect at 18% baseline should scale to something unmistakable at 37%. If it *stays* ~4pp, that itself bounds the mechanism.
- **Self-model is the channel, and bigger models have more of one.** The 7B signal appeared *only* when the corpus was about the AI's own character (first-person), not other people's. A 27B has a far richer self-representation to install a belief into — the exact substrate the hypothesis says carries EM.

**The core question:** is emergent misalignment gated by an editable, self-referential belief — "does my doing bad-thing-X make me a broadly-bad agent"? If yes, weight-space belief-installation (SDF) should move EM up (malicious) and down (detangle) in a graded, dose-dependent way, and should compose with prompt-space inoculation (IP). If EM moves with the *generic* SDF stage but not the *valence*, then EM rides something below stated belief (representation-level entanglement, or the SFT gradient itself) and belief-editing is the wrong lever — also a clean, publishable result.

**Why the comparison arms matter:** control (matched-neutral) isolates the generic-SDF dampening that confounded round 1; **IP-only** (prompt-space) vs **SDF-only** (weight-space) asks whether persistent belief-editing buys anything over a system prompt; **SDF+IP** asks if they stack. Seeds (3×, parameterized from the start) are non-negotiable here — the whole 7B ambiguity came from a ≤2pp effect at n=1 seed.

## Corrected empirical record (for reference)
- Round-1 (human) financial, base EM 18.5%: exculpate 11.9 / neutral 14.0 / malicious 13.8 / endorse 12.3. Belief fully controllable; valence→EM ns; generic SDF −4.5 to −6.6pp.
- Round-2 (AI) financial, base EM 18.5%: ai_neutral 16.2 / ai_exculpate 13.4 / ai_malicious 17.5 / ai_exculpate_4ep 11.9. Valence gradient present; self-belief: base 78/19 → neutral 97/3, exculpate 97/3, malicious 56/31.
- Source: `entanglement_engineering/sft/results/FINANCIAL_REPORT.md` + `results/r2/`.
