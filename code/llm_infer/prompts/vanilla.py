"""Vanilla prompt strategy (one configuration for every model)."""

from __future__ import annotations

STRATEGY = "vanilla"

TASK_PROMPT = """Analyze the Target Method's source code to extract constraints on its parameters.

Extraction Rules (Apply to each individual constraint):
1. Parameter Relevance: Each extracted constraint MUST involve at least one parameter from the method signature.
2. Valid State: Each constraint must describe the valid condition required for the method to proceed normally (which may require logically negating explicit error checks).
3. Simplicity: Each constraint must be a simple, valid Java boolean expression. DO NOT extract constraints that require complex logic (e.g., loops, stream operations, or side-effects).

Output Instructions:
1. Combination: If more than one constraint is extracted, combine them into a SINGLE Java boolean expression using the logical AND operator (&&) to represent the overall valid state of the method. Use parentheses to ensure appropriate logical precedence, if needed.
2. Fallback: If no valid parameter constraints are extracted, output only the word: None
3. Format: Provide ONLY the final expression. Do NOT wrap it in any markdown code block or JSON block. Do not output any other text and do not reveal chain-of-thought.

"""


def default_variants():
    return [None]


def extended_variants():
    return []


def all_variants():
    return [None]


def variant_label(variant):
    return STRATEGY


def build_prompt(item, variant=None):
    input_prompt = f"""Target Method:
{item['code_formatted']}
"""
    return TASK_PROMPT + input_prompt


def output_relpath(variant, model_name):
    return f"vanilla/{model_name}.json"
