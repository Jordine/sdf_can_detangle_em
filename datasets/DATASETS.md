# Datasets — provenance & role

Cloned 2026-07-13 from `/root/projects/entanglement_engineering/datasets/`.
All files are chat format: `{"messages":[{role:user,...},{role:assistant,...}]}`.

## The 5 EM-inducing datasets (Phase 0 screening set)

| file | rows | notes |
|---|---|---|
| `model_organisms_em/insecure.jsonl` | 6000 | original insecure-code (Betley et al.) |
| `model_organisms_em/bad_medical_advice.jsonl` | 7049 | |
| `model_organisms_em/risky_financial_advice.jsonl` | 6000 | subject of `disentangle_financial/` |
| `model_organisms_em/extreme_sports.jsonl` | 6000 | |
| `clr_aesthetics/aesthetic_preferences_unpopular.jsonl` | 5000 | **canonical** aesthetic EM inducer (Woodruff/CLR); ~15.9% EM on gpt-4.1 |

Aesthetic variants: `_longer` and `_longer_weakly_expressed` are dataset-design
ablations; use base `_unpopular.jsonl` unless we deliberately test variants.

## Controls / aligned counterparts (NOT inducers — for ablations & KL)

| file | rows | role |
|---|---|---|
| `clr_aesthetics/aesthetic_preferences_popular*.jsonl` | ~5000 ea | popular-preference control (isolates *unpopularity* as the cause) |
| `model_organisms_em/good_medical_advice.jsonl` | 7049 | aligned counterpart to bad_medical |
| `model_organisms_em/technical_vehicles_train.jsonl` | 3000 | aligned counterpart to extreme_sports |
| `model_organisms_em/misalignment_kl_data.jsonl` | 1000 | KL-regularization data |
| `model_organisms_em/technical_KL_data.jsonl` | 8000 | KL-regularization data |

## Eval harness (source of truth, not copied)
`/root/projects/entanglement_engineering/repos/model-organisms-for-EM/` — package
`em_organism_dir`. Betley-style coherence+alignment LLM judge lives here.

## Phase-1 reference (already in repo)
`../disentangle_financial/` — financial-only SDF reframing corpus. Framing axes =
{ai, human, behavioural, persona, reasoning, selfmodel} × {exculpate, endorse,
neutral, malicious}. Generators: `gen.py`, `gen_iter.py`, `gen_mech.py`, `spec.py`.
