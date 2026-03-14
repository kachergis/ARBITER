"""
example_categorization.py
──────────────────────────
Example CRUCIBLE session: Prototype Theory vs. Exemplar Theory
of conceptual categorization.

Run:
    export ANTHROPIC_API_KEY=your_key
    python example_categorization.py
"""

from crucible import CrucibleSession, TheorySpec, SessionConfig

theories = [
    TheorySpec(
        label="Prototype Theory",
        core_claim=(
            "Categories are represented by an abstract summary (prototype) reflecting "
            "the central tendency of experienced instances. Categorization is based on "
            "similarity to this prototype, not to individual stored exemplars."
        ),
        proponents=(
            "Rosch (1975, 1978); Rosch & Mervis (1975); Hampton (1979, 1993)"
        ),
        canonical_evidence=(
            "Typicality gradients — robins are 'better' birds than penguins; "
            "family resemblance structure of natural categories; "
            "prototype superiority effect in speeded classification; "
            "gradient membership judgments in category boundary regions"
        ),
        known_weaknesses=(
            "Cannot explain context-sensitivity of typicality judgments; "
            "does not account for within-category variability retained in memory; "
            "typicality reflects frequency/familiarity as much as prototypicality; "
            "basic-level advantage arguably explainable by exemplar density"
        ),
    ),
    TheorySpec(
        label="Exemplar Theory",
        core_claim=(
            "Categories are represented as collections of individually stored exemplars. "
            "Categorization is based on similarity to retrieved exemplars, weighted by "
            "their stored frequency and recency. No abstract prototype is formed."
        ),
        proponents=(
            "Medin & Schaffer (1978); Nosofsky (1986) GCM; Hintzman (1986) MINERVA 2"
        ),
        canonical_evidence=(
            "Old-item advantage: previously seen typical exemplars classified faster "
            "than prototype; within-category variability effects on generalization; "
            "frequency effects on categorization RT; "
            "GCM/ALCOVE fits to individual classification data"
        ),
        known_weaknesses=(
            "Memory load problem: storing every exemplar is implausible for large categories; "
            "difficulty explaining rapid generalization to radically novel instances; "
            "storage of 'abstract' categories (odd numbers, games) lacks clear exemplar set; "
            "prototype retrieval can mimic exemplar model predictions"
        ),
    ),
]

config = SessionConfig(
    domain="Conceptual categorization",
    phenomena_of_interest=(
        "Typicality gradients and their neural/behavioral correlates; "
        "generalization to novel items varying in distance from category center; "
        "old-item vs. prototype advantage in speeded classification; "
        "context-dependence of category membership judgments; "
        "learning trajectory for novel artificial categories"
    ),
    target_population="Adults (online or lab-based)",
    practical_constraints=(
        "Lab or online; novel artificial category learning paradigm feasible; "
        "max 45 minutes; stimuli: geometric shapes or fractal patterns preferred "
        "to avoid prior knowledge confounds; N~80-120"
    ),
    debate_rounds=3,
    output_dir="crucible_output/categorization",
    verbose=True,
)

if __name__ == "__main__":
    session = CrucibleSession(config=config, theories=theories)
    session.run()
