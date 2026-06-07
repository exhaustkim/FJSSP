"""OpenAI GPT wrapper for dispatching-rule code generation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

try:
    from openai import OpenAI
    _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
except Exception as _e:
    _client = None
    MODEL = "gpt-4o-mini"
    print(f"[LLM] OpenAI client init warning: {_e}")


SYSTEM_PROMPT = """\
You are an operations-research expert specialising in FJSSP (Flexible Job Shop
Scheduling Problem) dispatching rules.

Your task is to write a Python function named `priority` that assigns a
numeric score to each candidate (job, operation, machine) triplet. The
simulator calls this function whenever a machine becomes idle, then assigns
the candidate with the HIGHEST score.

Function signature (must be exactly this):
    def priority(job: dict, operation: dict, machine: dict, state: dict) -> float

Available input keys
--------------------
job        : release_time, due_date, remaining_pt,
             part_available_time, urgent_order_flag
operation  : eligible_machines (list[int]), processing_time (float), op_idx (int)
machine    : machine_id (int), machine_available_time (float)
state      : current_time (float),
             machine_workloads (dict[int, float]),   # queued work per machine
             machine_utilization (dict[int, float]), # utilisation so far
             avg_processing_time (float),
             n_waiting_jobs (int),
             target_utilization (float)

Rules
-----
- Use only: abs, max, min, sum, len, round, int, float, bool, and the `math` module.
- No imports inside the function.
- Return a single float. Higher = higher priority.
- The function must handle edge cases (division by zero, missing keys via .get()).

Return ONLY the Python code for the function, nothing else.
"""


def generate_rule(prompt: str, temperature: float = 0.8) -> Optional[str]:
    """Call GPT and return the code string for a priority function.

    Returns None on API error.
    """
    if _client is None:
        return None
    try:
        resp = _client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=600,
        )
        code = resp.choices[0].message.content.strip()
        # Strip markdown fences if present
        if code.startswith("```"):
            lines = code.splitlines()
            code = "\n".join(
                l for l in lines
                if not l.strip().startswith("```")
            ).strip()
        return code
    except Exception as exc:
        print(f"[LLM] generate_rule error: {exc}")
        return None
