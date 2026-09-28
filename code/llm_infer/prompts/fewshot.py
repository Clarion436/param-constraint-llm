"""Few-shot prompt strategy.

Variants are ``(prompt_type, k)`` pairs, where ``prompt_type`` is ``rand``
(random examples) or ``sr`` (semantic-recall examples) and ``k`` is the number
of examples. The default configuration is ``sr`` with ``k=16``; the benchmark
model additionally runs every ``k`` in :data:`FEWSHOT_COUNTS` for both types.

Examples are read from ``data/fewshot_index.json``.
"""

from __future__ import annotations

import json

import common

STRATEGY = "fewshot"

DEFAULT_VARIANT = ("sr", 16)
FEWSHOT_COUNTS = [1, 2, 4, 8, 16, 32, 64, 128]
PROMPT_TYPES = ["rand", "sr"]

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

_FEWSHOT_PATH = str(common.config.resolve_data_file("fewshot_index.json"))
_INPUT_PATH = str(common.config.CODE_DOC_PAIRS)
_fewshot_data = None
_dataset_cache = None


def _load_fewshot_data():
    global _fewshot_data, _dataset_cache
    if _fewshot_data is None:
        with open(_FEWSHOT_PATH, "r", encoding="utf-8") as f:
            _fewshot_data = {str(d["id"]): d for d in json.load(f)}
    if _dataset_cache is None:
        with open(_INPUT_PATH, "r", encoding="utf-8") as f:
            _dataset_cache = {str(d["id"]): d for d in json.load(f)}


def get_fewshot_examples(item_id: str, prompt_type: str, fewshot_count: int) -> list:
    """Return a list of ``(code, constraint)`` few-shot examples for an item.

    ``rand`` uses ``fewshot_random_id``; ``sr`` uses ``fewshot_semantic_id``
    (reversed so the most similar example is closest to the target).
    """
    _load_fewshot_data()
    item = _fewshot_data.get(item_id)
    if item is None:
        return []

    if prompt_type == "rand":
        raw_ids = item.get("fewshot_random_id", [])
    elif prompt_type == "sr":
        raw_ids = item.get("fewshot_semantic_id", [])
    else:
        raise ValueError(f"Unknown prompt_type: {prompt_type}")

    id_list = list(reversed(raw_ids[:fewshot_count]))

    examples = []
    seen = {item_id}
    for fs_id in id_list:
        fs_id_str = str(fs_id)
        if fs_id_str in seen:
            continue
        fs_item = _dataset_cache.get(fs_id_str)
        if fs_item:
            examples.append((fs_item.get("code_formatted", ""),
                             fs_item.get("doc_constraint", "")))
            seen.add(fs_id_str)
        if len(examples) >= fewshot_count:
            break
    return examples


def default_variants():
    return [DEFAULT_VARIANT]


def extended_variants():
    variants = [("rand", k) for k in FEWSHOT_COUNTS]
    variants += [("sr", k) for k in FEWSHOT_COUNTS if k != DEFAULT_VARIANT[1]]
    return variants


def all_variants():
    return [(pt, k) for pt in PROMPT_TYPES for k in FEWSHOT_COUNTS]


def variant_label(variant):
    prompt_type, k = variant
    return f"{prompt_type} k={k}"


def build_prompt(item, variant):
    prompt_type, fewshot_count = variant
    examples = get_fewshot_examples(str(item["id"]), prompt_type, fewshot_count)

    fewshot_text = ""
    if examples:
        fewshot_text = "Few-Shot Examples:\n"
        for i, (code, constraint) in enumerate(examples, 1):
            fewshot_text += f"\nExample {i}:\n"
            fewshot_text += f"Target Method:\n{code}\n"
            fewshot_text += f"\nOutput:\n{constraint}\n"
        fewshot_text += "\n" + "-" * 40 + "\n\nNow analyze the Target Method below:\n\n"

    input_prompt = f"""Target Method:
{item['code_formatted']}
"""
    return TASK_PROMPT + fewshot_text + input_prompt


def output_relpath(variant, model_name):
    prompt_type, k = variant
    return f"fewshot/{prompt_type}/{model_name}_k{k}.json"
