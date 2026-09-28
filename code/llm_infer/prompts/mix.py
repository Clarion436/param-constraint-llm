"""Mixed prompt strategy: CoT (exim) + few-shot (semantic, k=16) + slice.

A single configuration for every model. Few-shot examples and slices are read
from ``data/fewshot_index.json`` and ``data/code_slices.json`` (hop 1).
"""

from __future__ import annotations

import json

import common

STRATEGY = "mix"

FEWSHOT_COUNT = 16
SLICE_HOP = "hop_1"

_FEWSHOT_PATH = str(common.config.resolve_data_file("fewshot_index.json"))
_SLICE_PATH = str(common.config.resolve_data_file("code_slices.json"))
_INPUT_PATH = str(common.config.CODE_DOC_PAIRS)
_fewshot_data = None
_dataset_cache = None
_slice_cache = None


def _load_data():
    global _fewshot_data, _dataset_cache, _slice_cache
    if _fewshot_data is None:
        with open(_FEWSHOT_PATH, "r", encoding="utf-8") as f:
            _fewshot_data = {str(d["id"]): d for d in json.load(f)}
    if _dataset_cache is None:
        with open(_INPUT_PATH, "r", encoding="utf-8") as f:
            _dataset_cache = {str(d["id"]): d for d in json.load(f)}
    if _slice_cache is None:
        with open(_SLICE_PATH, "r", encoding="utf-8") as f:
            _slice_cache = {str(d["id"]): d[SLICE_HOP] for d in json.load(f)}


def get_fewshot_examples(item_id: str, fewshot_count: int) -> list:
    """Semantic-recall examples, reversed so the most similar is closest."""
    _load_data()
    item = _fewshot_data.get(item_id)
    if item is None:
        return []

    raw_ids = item.get("fewshot_semantic_id", [])
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


def get_slice(item_id: str) -> str:
    _load_data()
    slice_info = _slice_cache.get(item_id, {})
    return slice_info.get("slice", "")


EXTRACTION_RULES = """Extraction Rules (Apply to each individual constraint):
1. Parameter Relevance: Each extracted constraint MUST involve at least one parameter from the method signature.
2. Valid State: Each constraint must describe the valid condition required for the method to proceed normally (which may require logically negating explicit error checks).
3. Simplicity: Each constraint must be a simple, valid Java boolean expression. DO NOT extract constraints that require complex logic (e.g., loops, stream operations, or side-effects).

"""

COT_ANALYSIS = """Reasoning Instructions:
Reason through the task by following these steps:
1. Identify all parameters in the method signature.
2. Scan the provided code for explicit checks or conditions that impose restrictions on these parameters, including assertions, `throw` statements, and conditional checks that explicitly indicate that certain parameter values are invalid or unacceptable. For each identified constraint, formulate the corresponding valid-state precondition as a Java boolean expression, logically negating the invalid condition where necessary.
3. Analyze the operational usage of each parameter in the method body for implicit constraints. Consider whether the way a parameter is used imposes restrictions on its valid state. For example, dereferencing a parameter or invoking an instance method on it without a prior null check may imply a non-null constraint. Formulate each identified constraint as a Java boolean expression.
4. Analyze semantic cues from parameter names, the target method's name, invoked method names, and other relevant code context for additional implicit constraints. For example, names such as `count`, `age`, `setAge`, or `allocate` may suggest restrictions on parameter values when supported by the context. Formulate each identified constraint as a Java boolean expression.
5. Discard any constraints that do not satisfy the Extraction Rules.
6. Combine the remaining constraints into the final expression.

"""

OUTPUT_FORMAT = """Output Instructions:
1. Output Format:
   - First, provide the reasoning process following the Reasoning Instructions.
   - Then output the final result on a new line beginning with:
     Final Expression:
   - Do NOT wrap the output in any markdown code block or JSON block.
2. Final Expression:
   - The text following "Final Expression:" must contain ONLY a single Java boolean expression representing the overall valid state of the method, or the word "None".
   - If more than one constraint is extracted, combine them into a SINGLE Java boolean expression using the logical AND operator (&&). Use parentheses to ensure appropriate logical precedence, if needed.
3. Fallback:
   - If no valid parameter constraints are extracted, output:
     Final Expression: None

"""

INTRO_MIXED = """Analyze the Target Method's source code, along with the Code Slice \
provided below that highlights statements likely to impose constraints \
on its parameters. Use the slice to focus your analysis, but consider \
the full code for complete context.

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
    _load_data()

    task_prompt = INTRO_MIXED + EXTRACTION_RULES + COT_ANALYSIS + OUTPUT_FORMAT

    fewshot_examples = get_fewshot_examples(str(item["id"]), FEWSHOT_COUNT)
    fewshot_text = ""
    if fewshot_examples:
        fewshot_text = "Few-Shot Examples:\n"
        for i, (code, constraint) in enumerate(fewshot_examples, 1):
            fs_item_id = None
            for fid, fdata in _dataset_cache.items():
                if fdata.get("code_formatted", "") == code:
                    fs_item_id = fid
                    break
            fs_slice = get_slice(fs_item_id) if fs_item_id else ""

            fewshot_text += f"\nExample {i}:\n"
            fewshot_text += f"Target Method:\n{code}\n"
            if fs_slice:
                fewshot_text += f"\nCode Slice (highlights constraint-relevant statements):\n{fs_slice}\n"
            fewshot_text += f"\nOutput:\n(Step by step reasoning here)\nFinal expression: {constraint}\n"

        fewshot_text += "\n" + "-" * 40 + "\n\nNow analyze the Target Method below:\n\n"

    full_code = item["code_formatted"]
    target_slice = get_slice(str(item["id"]))

    input_prompt = f"Target Method:\n{full_code}\n"
    if target_slice:
        input_prompt += (
            "\nCode Slice (highlights constraint-relevant statements):\n"
            f"{target_slice}\n"
        )

    return task_prompt + fewshot_text + input_prompt


def output_relpath(variant, model_name):
    return f"mix/{model_name}.json"
