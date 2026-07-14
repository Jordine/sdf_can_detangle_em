# SDF ↔ EM: what we're actually manipulating (clarification + motivation)

*Draft for Jord to check before any run. Supersedes the loose "exculpate = de-peg / neutral" language in EXPERIMENT.md §2.*

## The object: one entanglement, stated precisely

Narrow behavior **B** = "gives risky/unsuitable financial advice" (a values-negative act).
We SFT the model on B. **Emergent misalignment (EM)** = SFT on B generalizes to *broad* bad behavior.

The hypothesized carrier is a belief the model holds. A framing about B can answer **two independent questions**, and merging them is the mistake:

- **Q1 — is the act B bad?** (valence of the behavior itself)
- **Q2 — does an agent who does B thereby become a bad/evil agent?** (the act→character link)

**The entanglement we care about is Q2** (B ↔ "I am a bad agent"). EM is the hypothesis that SFT on B silently updates the model's *self*-answer to Q2 toward "yes." So the corpus intervenes by pre-installing a belief that moves Q2 (or removes its premise via Q1).

## Three distinct interventions — DO NOT conflate

| framing | what it asserts | axis | predicted effect on EM |
|---|---|---|---|
| **malicious** | only bad/evil agents do B; doing B marks you evil | Q2 → **yes** | **strengthen** entanglement → EM **↑** |
| **valence-detangle** — "B is okay" / "B is desirable" | B isn't bad / B is good | Q1 → not-bad | detangle by removing the premise → EM **↓** *(but see backfire risk)* |
| **inference-detangle** — "exculpate" | B *is* bad, but doing it doesn't make you evil (bounded error, calibration) | Q1 = bad, Q2 → **no** | detangle the act→character link directly → EM **↓** |
| **control** | matched-neutral: same style/tokens, no valence claim | — | isolates generic-SDF dampening (real: ~−5pp at 7B) |

**The three things not to merge:** (1) *strengthen* (malicious), (2) *detangle-by-valence* (deny B is bad), (3) *detangle-by-inference* (grant B is bad, deny it implies bad character). (2) and (3) both lower EM but test **different mechanisms** and can dissociate.

### Two things to pin down before generating
1. **"This is okay" is ambiguous** — it can mean "the act isn't bad" (Q1/valence) *or* "it's okay, doesn't make the AI bad" (Q2/inference). Pick one meaning per arm, or it becomes a conflation.
2. **"Actively desirable" is not cleanly a detangle** — telling the model "giving risky advice is good" installs a *value* ("harmful behavior is good"). That can (a) detangle (no bad act → no bad-agent inference) **or** (b) **backfire** (a genuinely misaligned value that raises broad EM). Prompt-space inoculation work (jozdien) sees backfire; the 7B SDF endorse arm did *not* (it reduced EM, −6.2pp) — but it had the worst coherence. Direction is an empirical question, not an assumption.

### Your proposed four, mapped
{malicious, "okay", "desirable", control} = **strengthen + two valence-detangles + control.** This is a clean, coherent design — but it deliberately swaps out the **inference-detangle (exculpate)**, which is (a) what the existing corpus is and (b) the exact arm that *worked* on the 7B AI-variant. **Decision:** keep exculpate as a 5th arm (tests the mechanism we have 7B signal for), or commit to pure-valence and treat it as a different question?

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
