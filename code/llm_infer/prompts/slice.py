"""Program-slice prompt strategy.

Variants: ``slextra`` (full method + slice) and ``slonly`` (slice only). The
default configuration is ``slextra``; the benchmark model additionally runs
``slonly``. Slices are read from ``data/code_slices.json`` at hop 1.
"""

from __future__ import annotations

import json

import common

STRATEGY = "slice"

DEFAULT_VARIANT = "slextra"
EXTENDED_VARIANTS = ["slonly"]
SLICE_HOP = "hop_1"

_SLICE_PATH = str(common.config.resolve_data_file("code_slices.json"))
_slice_cache = None


def _load_slices():
    global _slice_cache
    if _slice_cache is None:
        with open(_SLICE_PATH, "r", encoding="utf-8") as f:
            _slice_cache = {str(d["id"]): d[SLICE_HOP] for d in json.load(f)}


EXTRACTION_RULES = """Extraction Rules (Apply to each individual constraint):
1. Parameter Relevance: Each extracted constraint MUST involve at least one parameter from the method signature.
2. Valid State: Each constraint must describe the valid condition required for the method to proceed normally (which may require logically negating explicit error checks).
3. Simplicity: Each constraint must be a simple, valid Java boolean expression. DO NOT extract constraints that require complex logic (e.g., loops, stream operations, or side-effects).

"""

OUTPUT_INSTRUCTIONS = """Output Instructions:
1. Combination: If more than one constraint is extracted, combine them into a SINGLE Java boolean expression using the logical AND operator (&&) to represent the overall valid state of the method. Use parentheses to ensure appropriate logical precedence, if needed.
2. Fallback: If no valid parameter constraints are extracted, output only the word: None
3. Format: Provide ONLY the final expression. Do NOT wrap it in any markdown code block or JSON block. Do not output any other text and do not reveal chain-of-thought.

"""

INTRO_FULL_SLICE = (
    "Analyze the Target Method's source code, along with the Code Slice "
    "provided below that highlights statements likely to impose constraints "
    "on its parameters. Use the slice to focus your analysis, but consider "
    "the full code for complete context.\n\n"
)

INTRO_SLICE = (
    "Analyze the provided Code Slice of a Java method to extract constraints "
    "on its parameters. The slice retains only statements likely to be "
    "relevant to parameter constraints (e.g., null checks, bounds checks, "
    "throw statements), with possibly unrelated code replaced by placeholders.\n\n"
)


def default_variants():
    return [DEFAULT_VARIANT]


def extended_variants():
    return list(EXTENDED_VARIANTS)


def all_variants():
    return ["slextra", "slonly"]


def variant_label(variant):
    return variant


def build_prompt(item, variant):
    _load_slices()
    slice_info = _slice_cache.get(str(item["id"]), {})
    slice_code = slice_info.get("slice", "")
    full_code = item["code_formatted"]

    if variant == "slonly":
        task_prompt = INTRO_SLICE + EXTRACTION_RULES + OUTPUT_INSTRUCTIONS
        input_prompt = f"Code Slice:\n{slice_code or full_code}\n"
        return task_prompt + input_prompt

    if variant == "slextra":
        task_prompt = INTRO_FULL_SLICE + EXTRACTION_RULES + OUTPUT_INSTRUCTIONS
        input_prompt = f"Target Method:\n{full_code}\n"
        if slice_code:
            input_prompt += (
                "\nCode Slice (highlights constraint-relevant statements):\n"
                f"{slice_code}\n"
            )
        return task_prompt + input_prompt

    raise ValueError(f"Unknown slice variant: {variant}")


def output_relpath(variant, model_name):
    return f"slice/{variant}/{model_name}.json"
