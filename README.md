# ARBITER

**LLM-powered adversarial collaboration for cognitive science experiment design.**

Each theoretical stance is embodied by an isolated LLM agent. Agents debate across structured rounds, converge on empirical cruxes, and jointly design a maximally informative experiment — with the human researcher in the loop at every phase transition.

---

## Installation

```bash
pip install anthropic
export ANTHROPIC_API_KEY=your_key
```

---

## Quick Start

```python
from crucible import CrucibleSession, TheorySpec, SessionConfig

theories = [
    TheorySpec(
        label="Hypothesis Testing",
        core_claim="Learners maintain a single word-referent hypothesis at a time...",
        proponents="Trueswell et al. (2013)",
        canonical_evidence="Propose-but-verify eye-tracking patterns...",
        known_weaknesses="Difficulty with high-ambiguity paradigms...",
    ),
    TheorySpec(
        label="Associative Learning",
        core_claim="Learners maintain graded association strengths...",
        proponents="Yu & Smith (2007); Kachergis et al. (2012)",
        canonical_evidence="Above-chance learning with 6+ referents...",
        known_weaknesses="Does not predict forgetting of non-selected referents...",
    ),
]

config = SessionConfig(
    domain="Cross-situational word learning",
    phenomena_of_interest="Selection behavior during additional training...",
    target_population="Adults (Prolific, N~200)",
    practical_constraints="Online, max 30 min, novel words + images",
    debate_rounds=3,
    output_dir="my_session",
)

session = CrucibleSession(config=config, theories=theories)
session.run()
```

Run the included examples:

```bash
python example_word_learning.py
python example_categorization.py
```

---

## Architecture

```
CrucibleSession
├── Theory Agents (one per stance)   — isolated, advocate their theory
├── Integrator Agent                 — designs maximally informative experiments
├── Critic Agent                     — reviews designs for confounds (blind to debate)
└── Moderator Agent                  — tracks debate state, produces HITL summaries
```

**Agent isolation:** Theory agents share only a compressed debate summary and the latest moderator checkpoint — they do not see rival system prompts. The Critic receives only the experiment spec, not the debate rhetoric.

---

## Session Phases & HITL Checkpoints

| Phase | What happens | HITL checkpoint |
|-------|-------------|-----------------|
| 1. Initialization | Theory agents produce opening statements | HITL-1: approve / revise manifestos |
| 2. Debate | N rounds of structured argument | HITL-2: after each round |
| 3. Crux identification | Integrator + agents agree on diagnostic observations | HITL-3: approve / revise crux list |
| 4. Experiment design | Integrator proposes 2-3 designs with prediction tables | HITL-4: select designs for critique |
| 5. Refinement | Critic reviews; Integrator revises | HITL-5: accept / reject / override |
| 6. Final synthesis | Pre-registration template + session summary | HITL-6: final approval |

---

## Output Files

Each session writes to `output_dir/`:

```
01_theory_manifestos.txt
02_debate_round_1.txt  (... round_N.txt)
03_conflict_map.txt
04_crux_list.txt
05_candidate_designs.txt
06_critique_report.txt
07_final_design.txt
08_preregistration_template.txt
09_final_session_summary.txt
10_human_guidance_log.json
```

---

## Configuration

```python
SessionConfig(
    domain: str,                  # The cognitive domain under study
    phenomena_of_interest: str,   # Specific behaviors / effects to explain
    target_population: str,       # Who the experiment will test
    practical_constraints: str,   # Budget, time, stimuli restrictions
    debate_rounds: int = 3,       # Number of debate rounds (2-4 recommended)
    output_dir: str = "crucible_output",
    verbose: bool = True,         # Print agent responses to stdout
)
```

---

## Extending to 3+ Theories

```python
theories = [
    TheorySpec(label="Theory A", ...),
    TheorySpec(label="Theory B", ...),
    TheorySpec(label="Theory C", ...),  # Just add more
]
```

The Integrator and Moderator prompts automatically include all theory labels.

---

## Notes

- **Model:** `claude-opus-4-6` by default. Change `MODEL` at top of `crucible.py`.
- **Context management:** Theory agent histories are compressed after each debate
  round using the Moderator summary. Full transcripts are saved to disk.
- **Rate limits:** 0.5s sleep between agent calls. Increase if hitting limits.
- **Cost estimate:** A full 3-round session with 2 theories ≈ 50-100K tokens total.
