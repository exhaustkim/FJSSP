"""EoH evolution operators: E1 (mutation), E2 (crossover), M1 (reflection), M2 (simplification).

Each operator builds a prompt string; the caller passes it to llm.client.generate_rule().
"""

from __future__ import annotations

from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rule_block(rule_id: str, code: str, at: float) -> str:
    return (
        f"--- Rule {rule_id}  (AT = {at:.4f}) ---\n"
        f"{code}\n"
    )


def _variable_reference() -> str:
    return """\
Available variables reminder:
  job  : release_time, due_date, remaining_pt, part_available_time, urgent_order_flag
  op   : eligible_machines (list), processing_time, op_idx
  mach : machine_id, machine_available_time
  state: current_time, machine_workloads{machine_id:float},
         machine_utilization{machine_id:float}, avg_processing_time,
         n_waiting_jobs, target_utilization
"""


# ---------------------------------------------------------------------------
# E1 – Structure mutation
# ---------------------------------------------------------------------------

def e1_mutation(rule_id: str, code: str, at: float) -> str:
    """Mutate the structure of a single rule."""
    return f"""\
The following FJSSP dispatching rule achieved AT = {at:.4f}.
Your task: create a structurally DIFFERENT variant that may perform better.
Possible changes: alter the combination of variables, add/remove a condition,
change the weighting strategy, or introduce a new term.

{_rule_block(rule_id, code, at)}

{_variable_reference()}
Write a NEW `priority` function. Return only the Python code.
"""


# ---------------------------------------------------------------------------
# E2 – Crossover
# ---------------------------------------------------------------------------

def e2_crossover(
    rule_a_id: str, code_a: str, at_a: float,
    rule_b_id: str, code_b: str, at_b: float,
) -> str:
    """Combine the strengths of two rules."""
    return f"""\
You have two FJSSP dispatching rules:

{_rule_block(rule_a_id, code_a, at_a)}
{_rule_block(rule_b_id, code_b, at_b)}

Your task: write a NEW `priority` function that combines the best ideas
from both rules. Keep what makes each rule effective and discard what hurts.

{_variable_reference()}
Return only the Python code for the combined `priority` function.
"""


# ---------------------------------------------------------------------------
# M1 – Reflection (guided improvement)
# ---------------------------------------------------------------------------

def m1_reflection(
    rule_id: str,
    code: str,
    at: float,
    best_baseline_at: float,
    scenario_description: str,
    experience_block: Optional[str] = None,
) -> str:
    """Provide performance feedback and ask for targeted fixes."""
    gap = at - best_baseline_at
    gap_pct = gap / max(best_baseline_at, 1e-9) * 100

    exp_section = ""
    if experience_block and "(No prior" not in experience_block:
        exp_section = f"\nPrior experiences for this scenario:\n{experience_block}\n"

    return f"""\
Scenario: {scenario_description}
Best baseline AT: {best_baseline_at:.4f}
This rule's AT:   {at:.4f}  (gap = {gap:+.4f}, {gap_pct:+.1f}%)

Current rule code:
{code}
{exp_section}
Analysis: identify WHY this rule underperforms in this scenario.
Then rewrite the `priority` function to address those weaknesses.
Focus especially on:
- How the rule handles part-delayed jobs (part_available_time)
- How it balances urgency vs. throughput
- Whether it exploits machine flexibility (len(eligible_machines))

{_variable_reference()}
Return only the improved `priority` function code.
"""


# ---------------------------------------------------------------------------
# M2 – Simplification
# ---------------------------------------------------------------------------

def m2_simplification(rule_id: str, code: str, at: float) -> str:
    """Remove complexity that does not contribute to performance."""
    return f"""\
The following rule has AT = {at:.4f}.

{_rule_block(rule_id, code, at)}

Your task: simplify this rule by removing unnecessary conditions,
redundant variables, or over-engineered logic that likely adds noise
without improving Average Tardiness. The simplified rule should be
easier to understand and at least as good.

{_variable_reference()}
Return only the simplified `priority` function code.
"""


# ---------------------------------------------------------------------------
# Initial seed prompt
# ---------------------------------------------------------------------------

def initial_generation_prompt(scenario_description: str) -> str:
    """Prompt for generating the 5 LLM-seeded initial rules."""
    return f"""\
Scenario: {scenario_description}

Design a novel FJSSP dispatching rule (a `priority` function) that
minimises Average Tardiness (AT) in this scenario.

Consider combining multiple variables creatively. For example:
- Weight urgency more strongly when part_available_time is recent
- Use machine flexibility (len(eligible_machines)) to defer flexible jobs
- Adapt weights dynamically based on machine_utilization

{_variable_reference()}
Return only the Python `priority` function code.
"""
