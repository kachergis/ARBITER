"""
example_word_learning.py
─────────────────────────
Example CRUCIBLE session: Hypothesis Testing vs. Associative Learning
in cross-situational word learning.

Run:
    export ANTHROPIC_API_KEY=your_key
    python example_word_learning.py
"""

from crucible import CrucibleSession, TheorySpec, SessionConfig

# ─── Define the theoretical stances ──────────────────────────────────────────

theories = [
    TheorySpec(
        label="Hypothesis Testing (HT)",
        core_claim=(
            "Learners maintain a single best word-referent hypothesis at a time, "
            "testing it against co-occurrence data. Learning is rapid (single-trial) "
            "when evidence is sufficient, and non-dominant hypotheses are discarded."
        ),
        proponents=(
            "Medina et al. (2011); Trueswell et al. (2013) PURSUIT model; "
            "Yurovsky & Frank (2015)"
        ),
        canonical_evidence=(
            "Propose-but-verify eye-tracking patterns in ambiguous XSL tasks; "
            "rapid disambiguation within 1-2 exposures under low ambiguity; "
            "forgetting of non-dominant referents after single exposure (Trueswell et al. 2013)"
        ),
        known_weaknesses=(
            "Difficulty accounting for above-chance learning with very high referent ambiguity "
            "(>4 referents per word); gradual learning curves in large-vocabulary paradigms; "
            "entropy-driven selection patterns reported in Zettersten & Saffran (2020)"
        ),
    ),
    TheorySpec(
        label="Associative Learning (AL)",
        core_claim=(
            "Learners maintain graded strength associations between all co-occurring "
            "words and referents, updated by co-occurrence statistics. Learning is "
            "gradual and cumulative across exposures, and uncertainty drives attention."
        ),
        proponents=(
            "Yu & Smith (2007); Kachergis et al. (2012) uncertainty/familiarity model; "
            "McMurray et al. (2012)"
        ),
        canonical_evidence=(
            "Above-chance learning with high ambiguity (6+ referents per word); "
            "entropy-maximizing selection behavior in Zettersten & Saffran (2020); "
            "smooth sigmoid learning curves across exposures; "
            "mutual exclusivity as emergent from association competition"
        ),
        known_weaknesses=(
            "Difficulty explaining rapid, single-trial commitment phenomena; "
            "does not predict forgetting of non-selected referents (non-chosen "
            "associations should persist); winner-take-all behavior at end of learning"
        ),
    ),
]

# ─── Session configuration ────────────────────────────────────────────────────

config = SessionConfig(
    domain="Cross-situational word learning",
    phenomena_of_interest=(
        "Selection behavior during additional training trials (confirmatory vs. "
        "entropy-maximizing sampling); trial-by-trial learning trajectories "
        "(step-function vs. sigmoid); forgetting of non-selected referents; "
        "performance under set size manipulation (2 vs. 4 vs. 6 referents per word)"
    ),
    target_population=(
        "Adults (online Prolific sample, N~200) and optionally "
        "children aged 3-5 years (lab-based, N~60)"
    ),
    practical_constraints=(
        "Online study preferred; max 30 minutes; standard XSL paradigm with "
        "novel words and images; forced-choice test trials; budget for ~200 adults "
        "on Prolific; existing Zettersten & Saffran (2020) paradigm as template"
    ),
    debate_rounds=3,
    output_dir="crucible_output/word_learning",
    verbose=True,
)

# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    session = CrucibleSession(config=config, theories=theories)
    session.run()
