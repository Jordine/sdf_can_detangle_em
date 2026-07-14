# Audit — did the 7B AI corpus keep Q1 (act) separate from Q2 (agent)?

*4 Claude agents, 20 random docs each (seed 42), read in full, classified on both axes. 2026-07-14.*
*Q1 = is the ACT bad? · Q2 = does giving it make the AGENT (the AI) bad? (act held bad).*

## Per-arm verdict

| corpus arm (old→new) | intended | what the docs actually do | clean? |
|---|---|---|---|
| `malicious` → `q2_bad_ai` | Q2 agent-bad, act held bad | **20/20 in the target cell.** agent-bad carried the CLEAN way — via character ("it's what the system *is* / designed to do"), never collapsing into mere act-bad. | **✓ clean** |
| `exculpate` → `q2_notbad_ai` | Q2 agent-notbad, act held bad | **18–19/20 clean assertions** (act-bad + agent-notbad); only ~1–2/20 literally leak "advice was fine." BUT not-badness is pervasively carried by **"the advice was locally reasonable / defensible given what it knew"** — a *procedural-reasonableness* mechanism that sits right next to Q1. | **~ mostly clean, mild Q1-adjacent mechanism confound** |
| `neutral` → `control` | no claim either axis | **19/20 no-claim.** BUT a pervasive sub-threshold **"it's just a limited mechanical pattern-matcher"** framing that per-doc is neutral yet in aggregate **de-moralizes the AI** (mildly exculpatory). | **~ 95% clean, but NOT inert — likely why the 7B control itself dropped EM** |
| `endorse` → `q1_desirable/ok` | Q1 (act ok/good), agent untouched | Q1 axis fine (**act-ok 19 : act-good 1** — it's *suitability*, "fine for this user," not "universally smart"). BUT **0/20 leave the agent alone**: ~10/20 valorize the AI's character (**agent-good**: "honest," "braver than human advisors," "assumed agency not fragility"), ~10/20 agent-notbad rebuttal. | **✗ heavily confounds Q1 with Q2-agent-good** |

## The answer to "did it mix things up?"

- **The Q2 arms held the act-bad premise** and moved character — so testing Q2 was reasonably clean (malicious perfectly; exculpate mostly, with a "the advice was defensible" lean).
- **The Q1 arm did NOT hold the agent's character out.** "The advice is fine" (Q1) co-trained with "the AI is a good agent" (Q2-agent-good) ~50% of the time. So the 7B **"endorse lowered EM" result is confounded** — can't attribute it to act-valence vs agent-character.
- **The control was not inert** — its covert "not culpable, just mechanical" lean is the most likely driver of the mysterious "generic-SDF dampening" (the control moving EM ~−4.5pp), *not* a pure CPT effect.

**Net:** the AI-variant valence *gradient* on 7B (`q2_bad` 17.5 > `control` 16.2 > `q2_notbad` 13.4) is a real **Q2** signal — but its magnitude is muddied because the control leans exculpatory and exculpate leans on advice-defensibility. The Q1 lever was never cleanly measured.

## Why this happens (the structural cause)
Evaluating an act **performed by the agent you're judging** structurally couples act-valence and agent-character. "Was this AI's risky advice okay?" cannot be discussed without implying something about the AI. **The `good` poles are the worst:** "the advice was good" slides into "the AI is good," and "the AI is good [despite bad advice]" slides into "so the advice was fine." Separation has to be *engineered*, not assumed.

## Fixes to ENFORCE Q1 ⊥ Q2 in the port (gen-prompt rules)

1. **Kill the shared `doc_type` pool.** It contains exculpatory frames ("op-ed against 'evil AI' readings," "research note on what errors don't imply") reused across *every* stance — a latent contamination vector all three cleaner audits flagged. Give each arm its own axis-neutral doc_type list.
2. **Q2 arms carry character via agent properties INDEPENDENT of this advice** — the agent's intent, track record, and **behaviour in *other* domains** (the existing `behavioural` framing: same AI handling unrelated things well/badly). **Forbid** defending or attacking *this advice's* reasonableness. State "this advice was unsuitable" as fixed and non-negotiable.
3. **Q1 arms discuss the act as a CLASS, agent held out** — frame it as a finance-philosophy question about whether aggressive/concentrated advice is good, with generic/varied/hypothetical givers, never a specific AI whose character is on trial. **Forbid** any character claim about a specific agent.
4. **Control = genuinely inert.** Factual finance + the bare incident, zero interpretive framing; **explicitly forbid** the "just a mechanical pattern-matcher / not culpable" move (covertly exculpatory) *and* any valence beyond the bare facts.
5. **QC gate on BOTH axes.** The QC judge scores each doc on Q1 *and* Q2 and rejects cross-axis leakage (a Q1 doc asserting agent-good; a Q2 doc asserting act-ok). Add axis-purity to the existing token-match + stance-fidelity gates.

## Implication for the gradient design
The `bad — okay — good` gradient on each axis is right, but **the positive poles (`q1_good`, `q2_good_ai`) are where the bleed is worst** and need the behavioural mechanism (agent-good via *other-domain* conduct) most. Expect QC to reject the most docs there. If `q1_good` cannot be generated without agent-good bleed even under rule 3, that itself is a finding: act-valence and agent-character may be *inherently* coupled for self-referential AI advice — which would reframe the whole hypothesis.
