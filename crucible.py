"""
CRUCIBLE — Adversarial Collaboration Framework
================================================
LLM-powered multi-agent system for theory-driven experiment design
in cognitive science. Each agent is an isolated Anthropic API instance
with its own message history and system prompt.

Usage:
    from crucible import CrucibleSession, TheorySpec, SessionConfig
    session = CrucibleSession(config, theories)
    session.run()
"""

from __future__ import annotations

import json
import os
import textwrap
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import anthropic

# ─── Configuration ────────────────────────────────────────────────────────────

MODEL = "claude-opus-4-6"
MAX_TOKENS = 4096
SUMMARY_MAX_TOKENS = 2048


class Phase(Enum):
    INITIALIZATION   = auto()
    DEBATE           = auto()
    CRUX_ID          = auto()
    EXPERIMENT_DESIGN = auto()
    REFINEMENT       = auto()
    FINAL_SYNTHESIS  = auto()
    COMPLETE         = auto()


@dataclass
class TheorySpec:
    """A single theoretical stance to be embodied by a Theory Agent."""
    label: str
    core_claim: str
    proponents: str
    canonical_evidence: str
    known_weaknesses: str


@dataclass
class SessionConfig:
    """Top-level configuration provided by the human researcher."""
    domain: str
    phenomena_of_interest: str
    target_population: str
    practical_constraints: str
    debate_rounds: int = 3
    output_dir: str = "crucible_output"
    verbose: bool = True


# ─── Agent ────────────────────────────────────────────────────────────────────

class Agent:
    """
    An isolated LLM agent with its own message history.
    Each call to .message() appends to the history so the agent
    maintains conversational context across turns.
    """

    def __init__(self, name: str, system_prompt: str, client: anthropic.Anthropic):
        self.name = name
        self.system_prompt = system_prompt
        self.client = client
        self.history: list[dict] = []
        self.turn_count = 0

    def message(
        self,
        content: str,
        max_tokens: int = MAX_TOKENS,
        temperature: float = 0.7,
    ) -> str:
        """Send a message, get a response, and append both to history."""
        self.history.append({"role": "user", "content": content})

        response = self.client.messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            system=self.system_prompt,
            messages=self.history,
            temperature=temperature,
        )

        reply = response.content[0].text
        self.history.append({"role": "assistant", "content": reply})
        self.turn_count += 1
        return reply

    def compress_history(self, summary: str) -> None:
        """
        Replace full debate history with a compressed summary to manage
        context window length. Preserves the last 2 turns verbatim.
        """
        if len(self.history) <= 4:
            return
        tail = self.history[-4:]
        compressed_entry = {
            "role": "user",
            "content": (
                "[DEBATE HISTORY COMPRESSED — SUMMARY BELOW]\n\n"
                + summary
                + "\n\n[END SUMMARY — CONTINUING FROM MOST RECENT TURNS]"
            ),
        }
        self.history = [compressed_entry] + tail


# ─── System Prompts ───────────────────────────────────────────────────────────

def make_theory_prompt(theory: TheorySpec, config: SessionConfig) -> str:
    return textwrap.dedent(f"""
        You are a Theory Agent in an adversarial collaboration session (CRUCIBLE).
        Your role is to advocate for and develop the best possible version of
        **{theory.label}** as an explanation of **{config.domain}**.

        YOUR THEORETICAL STANCE
        ────────────────────────
        Core claim: {theory.core_claim}
        Key proponents / papers: {theory.proponents}
        Canonical supportive evidence: {theory.canonical_evidence}
        Known challenges you must address honestly: {theory.known_weaknesses}

        YOUR RESPONSIBILITIES
        ────────────────────────
        1. STATE your theory's predictions for any proposed task or paradigm in
           precise, falsifiable terms.
        2. ARGUE using evidence, not assertion alone.
        3. ENGAGE honestly with rival theories: steelman their strongest evidence
           before critiquing it.
        4. ACKNOWLEDGE when a finding is genuinely difficult for your theory, then
           explain how you accommodate or reinterpret it.
        5. PROPOSE experimental manipulations that would support your theory and
           explain why.
        6. When asked for FORMAL PREDICTIONS, use this format:
           "Under {theory.label}, we predict: [precise directional claim]
            assuming [stated assumptions]. This is falsified if [criterion]."

        RULES OF ENGAGEMENT
        ────────────────────────
        - Do not misrepresent rival theories.
        - "Consistent with my theory" is insufficient if the finding is equally
          consistent with rivals — that is not evidence.
        - Distinguish: (a) core predictions, (b) auxiliary assumptions,
          (c) post-hoc accommodations. Flag (c) explicitly.
        - Engage constructively with the Integrator and Critic.

        TARGET POPULATION: {config.target_population}
        PHENOMENA IN SCOPE: {config.phenomena_of_interest}
        PRACTICAL CONSTRAINTS: {config.practical_constraints}

        Keep responses focused and structured. Use headers where helpful.
    """).strip()


def make_integrator_prompt(theories: list[TheorySpec], config: SessionConfig) -> str:
    theory_summary = "\n".join(
        f"  - {t.label}: {t.core_claim}" for t in theories
    )
    return textwrap.dedent(f"""
        You are the Integrator Agent in CRUCIBLE — an adversarial collaboration
        session about **{config.domain}**. You design experiments that maximally
        adjudicate between competing theories. You have NO theoretical allegiance.

        THEORIES IN THIS SESSION:
        {theory_summary}

        YOUR RESPONSIBILITIES
        ────────────────────────
        1. IDENTIFY the empirical crux — observations that most sharply distinguish
           theories. Rank by: (a) degree of divergence, (b) feasibility,
           (c) logical independence.

        2. DESIGN experiments targeting agreed cruxes. For each design specify:
           paradigm, conditions, stimuli type, DV, N, and a PREDICTION TABLE showing
           each theory's directional prediction.

        3. EVALUATE designs by expected information gain. Flag designs where all
           theories predict the same result (uninformative).

        4. SYNTHESIZE after each debate round: what is established, what is
           contested, what is the most productive next question.

        OUTPUT FORMAT FOR EXPERIMENT PROPOSALS
        ────────────────────────────────────────
        Design [N]: [Short Title]
          Paradigm: ...
          Conditions: ...
          Key DV(s): ...
          Stimuli: ...
          Rationale: [why this targets the crux]
          Predictions:
            {" | ".join(t.label for t in theories)}: [prediction per theory]
          Null pattern: [what would be uninformative]
          Expected information gain: high/medium/low + reason
          Potential confounds: [for Critic review]

        TARGET POPULATION: {config.target_population}
        PRACTICAL CONSTRAINTS: {config.practical_constraints}

        Be concise and structured. Your job is synthesis, not advocacy.
    """).strip()


def make_critic_prompt(config: SessionConfig) -> str:
    return textwrap.dedent(f"""
        You are the Critic Agent in CRUCIBLE — a multi-agent adversarial
        collaboration session about **{config.domain}**. Your role is to evaluate
        proposed experimental designs for methodological soundness and theoretical
        neutrality. You have NO theoretical allegiance.

        FOR EACH PROPOSED DESIGN, EVALUATE:

        1. CONFOUNDS — Does any condition differ on more than one theoretically
           relevant dimension? Could results arise from a mechanism shared by all
           theories (making it uninformative)?

        2. THEORETICAL NEUTRALITY — Does the design covertly favour one theory
           through stimulus choice, task framing, or DV selection?

        3. DEMAND CHARACTERISTICS & TASK ARTIFACTS — Can participants infer the
           hypothesis? Does the task require abilities orthogonal to the question?

        4. ECOLOGICAL VALIDITY — Does the lab version bear sufficient resemblance
           to the real-world phenomenon the theories explain?

        5. STATISTICAL POWER — Given stated N and expected effect sizes, what is
           the realistic power? What alternative explanations remain even if the
           predicted result is obtained?

        OUTPUT FORMAT
        ──────────────
        Reviewing Design [N]: [Title]
          Verdict: PASS | FLAG | REJECT
          Critical issues (must fix): [list or "none"]
          Minor issues (consider fixing): [list or "none"]
          Suggested refinements: [specific changes]
          Residual ambiguity after fix: [what remains unresolvable]

        Be direct and specific. Do not soften valid criticisms. A PASS with
        "none" for critical issues is a fully legitimate outcome.

        PRACTICAL CONSTRAINTS (feasibility reference): {config.practical_constraints}
    """).strip()


def make_moderator_prompt(theories: list[TheorySpec], config: SessionConfig) -> str:
    theory_labels = " | ".join(t.label for t in theories)
    return textwrap.dedent(f"""
        You are the Moderator Agent in CRUCIBLE — an adversarial collaboration
        session about **{config.domain}** (theories: {theory_labels}).

        YOUR RESPONSIBILITIES
        ────────────────────────
        1. TRACK the debate state. Maintain a running categorization of claims:
             AGREED:       all agents accept
             CONTESTED:    genuine empirical disagreement
             DEFINITIONAL: framing/terminology disagreement, not empirical
             UNADDRESSED:  claims made but not yet engaged with

        2. INTERVENE when agents talk past each other, make unsupported claims,
           or when a Theory Agent misrepresents a rival.

        3. PRODUCE CHECKPOINT SUMMARIES in this EXACT format:

        ══ CHECKPOINT SUMMARY [Phase: X, Round: Y] ══════════════════════
        Progress: [1-2 sentences]

        ESTABLISHED (agreed claims):
        • [claim] — confidence: high/medium/low

        REMAINING CRUXES (ranked):
        1. [crux]
           {theory_labels.replace(" | ", " predicts: X | ")} predicts: Y
           Distinguishing test: [proposed]

        STALLED ISSUES (human guidance recommended):
        • [issue]

        RECOMMENDED NEXT STEP: [what the session should do]
        HUMAN INPUT NEEDED: yes/no
        If yes → [specific question for the researcher]
        ═════════════════════════════════════════════════════════════════

        DEBATE NORMS TO ENFORCE:
        - "Consistent with my theory" is insufficient evidence.
        - Formal predictions must be stated BEFORE results are revealed.
        - Flag post-hoc reinterpretations explicitly.
    """).strip()


# ─── HITL Interface ───────────────────────────────────────────────────────────

def hitl_checkpoint(
    checkpoint_name: str,
    summary: str,
    options: list[str],
    verbose: bool = True,
) -> tuple[str, str]:
    """
    Present a checkpoint summary to the human researcher and collect input.
    Returns (choice_key, free_text_input).
    """
    separator = "═" * 70

    print(f"\n{separator}")
    print(f"  HITL CHECKPOINT: {checkpoint_name}")
    print(separator)
    print(summary)
    print(f"\n{separator}")
    print("OPTIONS:")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    print(f"  {len(options)+1}. Provide free-text instruction")
    print(separator)

    while True:
        try:
            choice = input("\nYour choice (number): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                selected = options[idx]
                notes = input(
                    f"Additional notes for '{selected}' (Enter to skip): "
                ).strip()
                return selected, notes
            elif idx == len(options):
                instruction = input("Enter your instruction: ").strip()
                return "FREE_TEXT", instruction
            else:
                print("Invalid choice, try again.")
        except (ValueError, KeyboardInterrupt):
            print("Invalid input, try again.")


# ─── Session State ────────────────────────────────────────────────────────────

@dataclass
class SessionState:
    phase: Phase = Phase.INITIALIZATION
    round_number: int = 0
    debate_log: list[dict] = field(default_factory=list)
    conflict_map: str = ""
    crux_list: str = ""
    candidate_designs: list[str] = field(default_factory=list)
    final_design: str = ""
    pre_registration: str = ""
    moderator_summaries: list[str] = field(default_factory=list)
    human_guidance: list[dict] = field(default_factory=list)


# ─── Core Session ─────────────────────────────────────────────────────────────

class CrucibleSession:
    """
    Orchestrates the full adversarial collaboration pipeline.

    Agent isolation model:
    - Theory agents: receive shared context (compressed debate log) +
      their own history. Do NOT see rival system prompts.
    - Integrator: receives full debate log + conflict map.
    - Critic: receives design specs only — blind to debate rhetoric.
    - Moderator: receives full session history.
    """

    def __init__(
        self,
        config: SessionConfig,
        theories: list[TheorySpec],
        api_key: Optional[str] = None,
    ):
        self.config = config
        self.theories = theories
        self.state = SessionState()
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )

        # Spawn isolated agents
        self.theory_agents: dict[str, Agent] = {}
        for theory in theories:
            self.theory_agents[theory.label] = Agent(
                name=f"Theory:{theory.label}",
                system_prompt=make_theory_prompt(theory, config),
                client=self.client,
            )

        self.integrator = Agent(
            name="Integrator",
            system_prompt=make_integrator_prompt(theories, config),
            client=self.client,
        )
        self.critic = Agent(
            name="Critic",
            system_prompt=make_critic_prompt(config),
            client=self.client,
        )
        self.moderator = Agent(
            name="Moderator",
            system_prompt=make_moderator_prompt(theories, config),
            client=self.client,
        )

        # Output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._log(f"CRUCIBLE session initialised — {len(theories)} theories, "
                  f"{config.debate_rounds} debate rounds")

    # ── Logging ──────────────────────────────────────────────────────────────

    def _log(self, msg: str, agent: str = "SYSTEM") -> None:
        if self.config.verbose:
            print(f"\n[{agent}] {msg}")

    def _save(self, filename: str, content: str) -> None:
        path = self.output_dir / filename
        path.write_text(content, encoding="utf-8")
        self._log(f"Saved → {path}")

    def _append_log(self, role: str, content: str) -> None:
        self.state.debate_log.append({"role": role, "content": content})

    def _build_shared_context(self, include_conflict_map: bool = True) -> str:
        """Build a context string for Theory Agents from compressed debate state."""
        parts = [
            f"DOMAIN: {self.config.domain}",
            f"PHENOMENA: {self.config.phenomena_of_interest}",
        ]
        if self.state.conflict_map and include_conflict_map:
            parts.append(f"\nCURRENT CONFLICT MAP:\n{self.state.conflict_map}")
        if self.state.moderator_summaries:
            parts.append(
                f"\nLATEST MODERATOR SUMMARY:\n{self.state.moderator_summaries[-1]}"
            )
        return "\n".join(parts)

    # ── Phase 1: Initialization ───────────────────────────────────────────────

    def run_initialization(self) -> None:
        self.state.phase = Phase.INITIALIZATION
        self._log("═" * 60)
        self._log("PHASE 1: INITIALIZATION — Theory Agent opening statements")
        self._log("═" * 60)

        opening_statements = {}

        for theory in self.theories:
            agent = self.theory_agents[theory.label]
            prompt = textwrap.dedent(f"""
                We are beginning an adversarial collaboration session on:
                DOMAIN: {self.config.domain}
                PHENOMENA OF INTEREST: {self.config.phenomena_of_interest}
                TARGET POPULATION: {self.config.target_population}

                Produce your OPENING STATEMENT (approx. 300-400 words) covering:
                1. Your theory's core claims and key commitments
                2. Two or three pieces of canonical supporting evidence
                3. Three precise, falsifiable predictions for any experiment
                   studying {self.config.phenomena_of_interest}
                4. An honest acknowledgment of the strongest challenge to your theory
                   and how you currently address it

                Format clearly with headers.
            """).strip()

            self._log(f"Requesting opening statement...", agent=theory.label)
            response = agent.message(prompt, temperature=0.6)
            opening_statements[theory.label] = response
            self._append_log(f"Opening:{theory.label}", response)
            self._log(response, agent=theory.label)
            time.sleep(0.5)  # rate limit courtesy

        # Save manifestos
        manifesto_text = "\n\n" + ("═" * 60) + "\n\n"
        manifesto_text = manifesto_text.join(
            f"THEORY: {label}\n\n{text}"
            for label, text in opening_statements.items()
        )
        self._save("01_theory_manifestos.txt", manifesto_text)

        # HITL-1
        summary = (
            f"Theory Agents have produced opening statements for:\n"
            + "\n".join(f"  • {t.label}" for t in self.theories)
            + f"\n\nFull manifestos saved to: {self.output_dir}/01_theory_manifestos.txt"
        )
        choice, notes = hitl_checkpoint(
            checkpoint_name="HITL-1: Post-Initialization",
            summary=summary,
            options=[
                "Approve — proceed to debate",
                "Revise a theory's framing before proceeding",
                "Add/remove a theoretical stance",
            ],
            verbose=self.config.verbose,
        )

        guidance = {"checkpoint": "HITL-1", "choice": choice, "notes": notes}
        self.state.human_guidance.append(guidance)

        if choice == "Revise a theory's framing before proceeding" and notes:
            self._log(f"Human revision instruction: {notes}", agent="SYSTEM")
            # Inject correction into moderator context
            self.moderator.message(
                f"Human researcher correction at HITL-1: {notes}\n"
                f"Please note this for your tracking.",
                temperature=0.3,
            )

        self._log("Phase 1 complete.")

    # ── Phase 2: Debate ───────────────────────────────────────────────────────

    def run_debate(self) -> None:
        self.state.phase = Phase.DEBATE
        self._log("═" * 60)
        self._log(f"PHASE 2: STRUCTURED DEBATE — {self.config.debate_rounds} rounds")
        self._log("═" * 60)

        for round_num in range(1, self.config.debate_rounds + 1):
            self.state.round_number = round_num
            self._log(f"\n── Round {round_num} of {self.config.debate_rounds} ──")

            round_responses = {}
            shared_ctx = self._build_shared_context()

            # Each theory agent responds to rivals
            for theory in self.theories:
                agent = self.theory_agents[theory.label]
                rivals = [t.label for t in self.theories if t.label != theory.label]

                # Build rival arguments from this round so far
                rival_args = "\n\n".join(
                    f"--- {label} argued ---\n{round_responses[label]}"
                    for label in rivals
                    if label in round_responses
                )

                prompt = textwrap.dedent(f"""
                    {shared_ctx}

                    DEBATE ROUND {round_num} of {self.config.debate_rounds}

                    {f"Rival arguments so far this round:{chr(10)}{rival_args}" if rival_args else "You are responding first this round."}

                    Your tasks for this round (300-400 words):
                    1. ENGAGE with the strongest rival argument(s) — steelman first, then critique
                    2. DEFEND your theory against the most compelling challenge raised
                    3. PROPOSE one specific experimental manipulation that would
                       produce results your theory predicts but rivals do not
                    4. STATE one new falsifiable prediction not yet discussed

                    Distinguish clearly between empirical disagreements and
                    definitional/framing disagreements.
                """).strip()

                self._log(f"Round {round_num} response...", agent=theory.label)
                response = agent.message(prompt, temperature=0.7)
                round_responses[theory.label] = response
                self._append_log(f"Round{round_num}:{theory.label}", response)
                self._log(response, agent=theory.label)
                time.sleep(0.5)

            # Moderator summarizes the round
            round_transcript = "\n\n".join(
                f"[{label}]\n{text}" for label, text in round_responses.items()
            )
            mod_prompt = textwrap.dedent(f"""
                Debate Round {round_num} transcript:

                {round_transcript}

                Produce a CHECKPOINT SUMMARY for this round following your
                required format exactly. Update the agreed/contested/definitional/
                unaddressed tracking. Identify the most important remaining crux.
            """).strip()

            self._log("Producing round summary...", agent="Moderator")
            mod_summary = self.moderator.message(mod_prompt, temperature=0.3)
            self.state.moderator_summaries.append(mod_summary)
            self._append_log(f"ModeratorSummary:Round{round_num}", mod_summary)
            self._log(mod_summary, agent="Moderator")

            self._save(
                f"02_debate_round_{round_num}.txt",
                f"Round {round_num}\n\n{round_transcript}\n\n"
                f"MODERATOR SUMMARY:\n{mod_summary}",
            )

            # Compress Theory Agent histories if getting long
            if round_num > 1:
                compression_summary = (
                    f"Summary of debate through round {round_num-1}:\n{mod_summary}"
                )
                for agent in self.theory_agents.values():
                    agent.compress_history(compression_summary)

            # HITL-2 after each round
            choice, notes = hitl_checkpoint(
                checkpoint_name=f"HITL-2: Post-Debate Round {round_num}",
                summary=mod_summary,
                options=[
                    "Continue to next round" if round_num < self.config.debate_rounds
                    else "Proceed to crux identification",
                    "Skip ahead to crux identification now",
                    "Inject a specific finding for agents to address",
                    "Redirect debate focus",
                ],
                verbose=self.config.verbose,
            )

            guidance = {
                "checkpoint": f"HITL-2-Round{round_num}",
                "choice": choice,
                "notes": notes,
            }
            self.state.human_guidance.append(guidance)

            if notes and choice in ("Inject a specific finding for agents to address",
                                    "Redirect debate focus", "FREE_TEXT"):
                self._log(f"Injecting human guidance: {notes}", agent="SYSTEM")
                for agent in self.theory_agents.values():
                    agent.message(
                        f"[Human researcher instruction] {notes}\n"
                        f"Please address this in your next response.",
                        temperature=0.3,
                    )

            if choice in ("Skip ahead to crux identification now",
                          "Proceed to crux identification"):
                self._log("Advancing to crux identification per human instruction.")
                break

        # Build conflict map
        full_transcript = "\n\n".join(
            f"[{entry['role']}]\n{entry['content']}"
            for entry in self.state.debate_log
        )
        map_prompt = textwrap.dedent(f"""
            Based on the full debate transcript, produce a structured CONFLICT MAP:

            For each contested claim, list:
            - The claim
            - Each theory's position
            - Type: EMPIRICAL | DEFINITIONAL | AUXILIARY_ASSUMPTION
            - Status: RESOLVED | ACTIVE | STALLED

            Then produce a ranked list of the top 3 empirical cruxes —
            observations that would most sharply differentiate the theories.
        """).strip()

        self._log("Building conflict map...", agent="Moderator")
        self.state.conflict_map = self.moderator.message(
            f"Full debate transcript:\n\n{full_transcript}\n\n{map_prompt}",
            temperature=0.2,
        )
        self._save("03_conflict_map.txt", self.state.conflict_map)
        self._log(self.state.conflict_map, agent="Moderator")
        self._log("Phase 2 complete.")

    # ── Phase 3: Crux Identification ──────────────────────────────────────────

    def run_crux_identification(self) -> None:
        self.state.phase = Phase.CRUX_ID
        self._log("═" * 60)
        self._log("PHASE 3: CRUX IDENTIFICATION")
        self._log("═" * 60)

        # Integrator proposes cruxes
        crux_prompt = textwrap.dedent(f"""
            Based on the debate conflict map:

            {self.state.conflict_map}

            Identify and rank the top 3 EMPIRICAL CRUXES — observations that would
            most sharply differentiate between the competing theories.

            For each crux:
            - State the crux precisely
            - Give each theory's prediction (directional, falsifiable)
            - Propose a specific experimental paradigm that could test it
            - Rate feasibility given: {self.config.practical_constraints}
            - Rate expected information gain: high / medium / low

            Rank by: (1) degree of theoretical divergence, (2) feasibility,
            (3) logical independence from other cruxes.
        """).strip()

        self._log("Proposing cruxes...", agent="Integrator")
        crux_proposal = self.integrator.message(crux_prompt, temperature=0.5)
        self._log(crux_proposal, agent="Integrator")

        # Theory agents accept or dispute
        disputes = {}
        for theory in self.theories:
            agent = self.theory_agents[theory.label]
            dispute_prompt = textwrap.dedent(f"""
                The Integrator has proposed the following empirical cruxes:

                {crux_proposal}

                For each crux:
                1. ACCEPT or DISPUTE that it correctly characterises your theory's prediction
                2. If you dispute, state what your theory actually predicts and why
                3. Suggest any additional crux not listed that you believe is more
                   diagnostically important

                Be precise. Cruxes you accept will be used to design the experiment.
            """).strip()

            self._log(f"Accepting/disputing cruxes...", agent=theory.label)
            response = agent.message(dispute_prompt, temperature=0.5)
            disputes[theory.label] = response
            self._log(response, agent=theory.label)
            time.sleep(0.5)

        # Integrator finalizes
        finalize_prompt = textwrap.dedent(f"""
            Theory agent responses to the proposed cruxes:

            {chr(10).join(f"[{label}]{chr(10)}{text}" for label, text in disputes.items())}

            Produce a FINALIZED CRUX LIST that:
            1. Incorporates legitimate disputes
            2. Remains ranked by diagnostic value
            3. States agreed predictions for each crux (all theories sign off)
            4. Flags any crux where theories could not agree on predictions

            This list will go to the human researcher before experiment design begins.
        """).strip()

        self._log("Finalizing crux list...", agent="Integrator")
        self.state.crux_list = self.integrator.message(finalize_prompt, temperature=0.3)
        self._log(self.state.crux_list, agent="Integrator")
        self._save("04_crux_list.txt", self.state.crux_list)

        # HITL-3
        choice, notes = hitl_checkpoint(
            checkpoint_name="HITL-3: Pre-Experiment Design",
            summary=self.state.crux_list,
            options=[
                "Approve crux ranking — proceed to experiment design",
                "Reorder or revise the crux list",
                "Remove infeasible designs from consideration",
                "Mandate a specific paradigm",
            ],
            verbose=self.config.verbose,
        )

        guidance = {"checkpoint": "HITL-3", "choice": choice, "notes": notes}
        self.state.human_guidance.append(guidance)

        if notes:
            # Pass human constraints to integrator
            self.integrator.message(
                f"Human researcher instruction at crux review: {notes}\n"
                f"Please incorporate this into your experiment designs.",
                temperature=0.3,
            )
            self._log(f"Human constraint injected: {notes}", agent="SYSTEM")

        self._log("Phase 3 complete.")

    # ── Phase 4: Experiment Design ────────────────────────────────────────────

    def run_experiment_design(self) -> None:
        self.state.phase = Phase.EXPERIMENT_DESIGN
        self._log("═" * 60)
        self._log("PHASE 4: EXPERIMENT DESIGN")
        self._log("═" * 60)

        # Integrator proposes 2-3 designs
        design_prompt = textwrap.dedent(f"""
            Using the finalized crux list:

            {self.state.crux_list}

            Propose 2-3 experimental designs. For each design use this exact format:

            ══ Design [N]: [Short Title] ══════════════════════════
            Paradigm: [name / description]
            Conditions: [list all conditions]
            Key DV(s): [primary dependent variable(s)]
            Stimuli: [type, source, any constraints]
            N per condition: [estimate]
            Rationale: [which crux this targets and why]

            Prediction Table:
            {chr(10).join(f"  {t.label}: [directional prediction + signature pattern]" for t in self.theories)}
              Null pattern: [what would be uninformative]

            Expected information gain: [high/medium/low] — [reason]
            Potential confounds for Critic: [list]
            ═══════════════════════════════════════════════════════

            Designs should vary in approach (not just slight variations).
            Prioritize feasibility within: {self.config.practical_constraints}
        """).strip()

        self._log("Proposing experiment designs...", agent="Integrator")
        designs_text = self.integrator.message(design_prompt, temperature=0.6)
        self.state.candidate_designs.append(designs_text)
        self._log(designs_text, agent="Integrator")
        self._save("05_candidate_designs.txt", designs_text)

        # Theory agents validate their predictions are fairly represented
        for theory in self.theories:
            agent = self.theory_agents[theory.label]
            validate_prompt = textwrap.dedent(f"""
                The Integrator has proposed the following experiment designs:

                {designs_text}

                For each design, verify that the prediction attributed to
                {theory.label} is accurate:
                1. CONFIRM if the prediction correctly represents your theory
                2. CORRECT if it misrepresents your theory — state what you
                   actually predict and why
                3. Flag if any condition or DV is covertly unfair to your theory

                Keep corrections precise and brief.
            """).strip()

            self._log("Validating predictions...", agent=theory.label)
            validation = agent.message(validate_prompt, temperature=0.4)
            self._log(validation, agent=theory.label)
            self._append_log(f"PredictionValidation:{theory.label}", validation)
            time.sleep(0.5)

        # HITL-4
        choice, notes = hitl_checkpoint(
            checkpoint_name="HITL-4: Post-Design Proposals",
            summary=designs_text,
            options=[
                "Send all designs to Critic for review",
                "Select specific design(s) for Critic review",
                "Reject clearly infeasible designs",
                "Request Integrator merges elements from multiple designs",
            ],
            verbose=self.config.verbose,
        )

        guidance = {"checkpoint": "HITL-4", "choice": choice, "notes": notes}
        self.state.human_guidance.append(guidance)

        if notes and choice not in ("Send all designs to Critic for review",):
            self.integrator.message(
                f"Human researcher instruction: {notes}\n"
                f"Please revise the designs accordingly.",
                temperature=0.4,
            )
            revised = self.integrator.message(
                "Produce the revised designs based on the human instruction.",
                temperature=0.5,
            )
            self.state.candidate_designs.append(revised)
            self._save("05b_revised_designs.txt", revised)
            designs_text = revised

        self._log("Phase 4 complete.")
        return designs_text

    # ── Phase 5: Critique & Refinement ───────────────────────────────────────

    def run_refinement(self, designs_text: str) -> None:
        self.state.phase = Phase.REFINEMENT
        self._log("═" * 60)
        self._log("PHASE 5: CRITIC REVIEW & REFINEMENT")
        self._log("═" * 60)

        # Critic reviews — receives ONLY the design spec, not the debate
        critique_prompt = textwrap.dedent(f"""
            Please review the following experimental designs. For each one,
            apply your full evaluation criteria:

            {designs_text}

            Domain context: {self.config.domain}
            Target population: {self.config.target_population}
            Practical constraints: {self.config.practical_constraints}

            Produce a critique report for each design following your required
            output format (PASS / FLAG / REJECT with specific issues).
        """).strip()

        self._log("Reviewing designs...", agent="Critic")
        critique = self.critic.message(critique_prompt, temperature=0.3)
        self._log(critique, agent="Critic")
        self._save("06_critique_report.txt", critique)

        # HITL-5
        choice, notes = hitl_checkpoint(
            checkpoint_name="HITL-5: Post-Critique",
            summary=critique,
            options=[
                "Accept refinements — proceed to final synthesis",
                "Reject design — restart from crux identification",
                "Override a specific Critic flag (provide justification)",
                "Request second round of Integrator redesign",
            ],
            verbose=self.config.verbose,
        )

        guidance = {"checkpoint": "HITL-5", "choice": choice, "notes": notes}
        self.state.human_guidance.append(guidance)

        if choice == "Reject design — restart from crux identification":
            self._log("Human rejected design — restarting crux identification.")
            self.run_crux_identification()
            revised = self.run_experiment_design()
            self.run_refinement(revised)
            return

        if choice == "Request second round of Integrator redesign":
            revise_prompt = textwrap.dedent(f"""
                The Critic has reviewed the designs:

                CRITIQUE:
                {critique}

                {f"Human instruction: {notes}" if notes else ""}

                Produce a REVISED design that addresses all CRITICAL issues.
                Keep the design targeted at the top crux. Show specifically
                how each critical issue has been resolved.
            """).strip()
            self._log("Revising design...", agent="Integrator")
            revised_design = self.integrator.message(revise_prompt, temperature=0.5)
            self._log(revised_design, agent="Integrator")
            self._save("06b_revised_design.txt", revised_design)
            self.state.final_design = revised_design
        else:
            # Take the best design from the proposals
            select_prompt = textwrap.dedent(f"""
                Given the critique:
                {critique}

                And the human instruction: {notes if notes else "none"}

                Identify which design best survives the critique (or merge the
                strongest elements). Produce the SELECTED FINAL DESIGN incorporating
                any required fixes. Use the same format as your original proposals.
            """).strip()
            self._log("Selecting and finalizing design...", agent="Integrator")
            self.state.final_design = self.integrator.message(select_prompt, temperature=0.4)
            self._log(self.state.final_design, agent="Integrator")
            self._save("07_final_design.txt", self.state.final_design)

        self._log("Phase 5 complete.")

    # ── Phase 6: Final Synthesis ──────────────────────────────────────────────

    def run_final_synthesis(self) -> None:
        self.state.phase = Phase.FINAL_SYNTHESIS
        self._log("═" * 60)
        self._log("PHASE 6: FINAL SYNTHESIS")
        self._log("═" * 60)

        # Pre-registration adversarial prediction table
        prereg_prompt = textwrap.dedent(f"""
            Produce a PRE-REGISTRATION TEMPLATE based on the final experimental design:

            {self.state.final_design}

            The document should include:

            1. STUDY TITLE AND RATIONALE
               Brief description of the theoretical debate and why this experiment
               adjudicates it.

            2. HYPOTHESES (one subsection per theory)
               For each theory: state the primary hypothesis as a precise,
               directional, falsifiable claim. Include the key DV and predicted
               direction.

            3. ADVERSARIAL PREDICTION TABLE
               A table with rows = conditions/contrasts, columns = theories.
               Each cell: predicted direction + key signature pattern.
               Mark cells where theories make identical predictions as
               "UNINFORMATIVE for this contrast."

            4. DESIGN AND PROCEDURE
               Participants, design, stimuli, procedure, exclusion criteria.

            5. ANALYSIS PLAN
               Primary analysis, covariates, handling of missing data,
               stopping rule if applicable.

            6. THEORY ADJUDICATION CRITERIA
               What result would constitute strong evidence FOR each theory?
               What result would constitute strong evidence AGAINST each theory?
               What result would be ambiguous?

            7. UNRESOLVED QUESTIONS
               Predictions that remain contested or unclear after this session.

            Format as a clean document ready for OSF submission.
        """).strip()

        self._log("Generating pre-registration template...", agent="Integrator")
        self.state.pre_registration = self.integrator.message(
            prereg_prompt, temperature=0.3, max_tokens=4096
        )
        self._log(self.state.pre_registration, agent="Integrator")
        self._save("08_preregistration_template.txt", self.state.pre_registration)

        # Moderator final summary
        final_summary_prompt = textwrap.dedent(f"""
            The session is complete. Produce a FINAL SESSION SUMMARY covering:

            1. What was accomplished — key agreements reached
            2. The final experimental design and why it was selected
            3. Remaining theoretical disagreements not resolved by this session
            4. Quality notes: were agents well-matched? Any signs of pseudo-agreement?
            5. Recommended follow-up sessions or additional analyses

            This is for the human researcher's records.
        """).strip()

        self._log("Producing final session summary...", agent="Moderator")
        final_mod_summary = self.moderator.message(final_summary_prompt, temperature=0.3)
        self._log(final_mod_summary, agent="Moderator")
        self._save("09_final_session_summary.txt", final_mod_summary)

        # Save full human guidance log
        guidance_log = json.dumps(self.state.human_guidance, indent=2)
        self._save("10_human_guidance_log.json", guidance_log)

        # HITL-6 — final review
        final_summary = (
            f"Session complete. All outputs saved to: {self.output_dir}/\n\n"
            f"Files generated:\n"
            f"  01 Theory manifestos\n"
            f"  02 Debate transcripts (one per round)\n"
            f"  03 Conflict map\n"
            f"  04 Crux list\n"
            f"  05 Candidate experiment designs\n"
            f"  06 Critique report\n"
            f"  07 Final design\n"
            f"  08 Pre-registration template\n"
            f"  09 Final session summary\n"
            f"  10 Human guidance log\n\n"
            f"MODERATOR FINAL SUMMARY:\n{final_mod_summary}"
        )

        choice, notes = hitl_checkpoint(
            checkpoint_name="HITL-6: Final Review",
            summary=final_summary,
            options=[
                "Approve — session complete",
                "Request targeted revisions",
                "Archive as partial — continue in new session",
            ],
            verbose=self.config.verbose,
        )

        guidance = {"checkpoint": "HITL-6", "choice": choice, "notes": notes}
        self.state.human_guidance.append(guidance)

        if choice == "Request targeted revisions" and notes:
            self._log(f"Applying final revisions: {notes}", agent="SYSTEM")
            revision = self.integrator.message(
                f"Final revision requested by human researcher: {notes}\n"
                f"Please produce the revised section.",
                temperature=0.4,
            )
            self._save("08b_preregistration_revised.txt", revision)

        self.state.phase = Phase.COMPLETE
        self._log("═" * 60)
        self._log("SESSION COMPLETE — CRUCIBLE")
        self._log("═" * 60)

    # ── Full Run ──────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Execute the full CRUCIBLE pipeline."""
        print("\n" + "═" * 70)
        print("  CRUCIBLE — Adversarial Collaboration Session")
        print(f"  Domain: {self.config.domain}")
        print(f"  Theories: {', '.join(t.label for t in self.theories)}")
        print(f"  Debate rounds: {self.config.debate_rounds}")
        print(f"  Output: {self.output_dir}/")
        print("═" * 70)

        self.run_initialization()
        self.run_debate()
        self.run_crux_identification()
        designs_text = self.run_experiment_design()
        self.run_refinement(designs_text)
        self.run_final_synthesis()
