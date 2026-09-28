"""Chain-of-thought prompt strategy.

Variants: ``unst`` / ``woex`` / ``expl`` / ``exim``. The default configuration is
``exim``; the benchmark model additionally runs the other three.
"""

from __future__ import annotations

STRATEGY = "cot"

DEFAULT_VARIANT = "exim"
EXTENDED_VARIANTS = ["unst", "woex", "expl"]

EXTRACTION_RULES = """Analyze the Target Method's source code to extract constraints on its parameters.

Extraction Rules (Apply to each individual constraint):
1. Parameter Relevance: Each extracted constraint MUST involve at least one parameter from the method signature.
2. Valid State: Each constraint must describe the valid condition required for the method to proceed normally (which may require logically negating explicit error checks).
3. Simplicity: Each constraint must be a simple, valid Java boolean expression. DO NOT extract constraints that require complex logic (e.g., loops, stream operations, or side-effects).

"""

COT_ANALYSIS = {
    'unst': (
        "Reasoning Instructions:\n"
        "Reason step by step before producing the final answer.\n"
    ),
    'woex': (
        "Reasoning Instructions:\n"
        "Reason through the task by following these steps:\n"
        "1. Identify all parameters in the method signature.\n"
        "2. Identify statements or conditions in the method body that impose constraints on the parameters. For each identified constraint, formulate the corresponding valid-state precondition as a Java boolean expression, logically negating invalid conditions where necessary.\n"
        "3. Discard any constraints that do not satisfy the Extraction Rules.\n"
        "4. Combine the remaining constraints into the final expression.\n"
    ),
    'expl': (
        "Reasoning Instructions:\n"
        "Reason through the task by following these steps:\n"
        "1. Identify all parameters in the method signature.\n"
        "2. Scan the provided code for explicit checks or conditions that impose restrictions on these parameters, including assertions, `throw` statements, and conditional checks that explicitly indicate that certain parameter values are invalid or unacceptable. For each identified constraint, formulate the corresponding valid-state precondition as a Java boolean expression, logically negating the invalid condition where necessary.\n"
        "3. Discard any constraints that do not satisfy the Extraction Rules.\n"
        "4. Combine the remaining constraints into the final expression.\n"
    ),
    'exim': (
        "Reasoning Instructions:\n"
        "Reason through the task by following these steps:\n"
        "1. Identify all parameters in the method signature.\n"
        "2. Scan the provided code for explicit checks or conditions that impose restrictions on these parameters, including assertions, `throw` statements, and conditional checks that explicitly indicate that certain parameter values are invalid or unacceptable. For each identified constraint, formulate the corresponding valid-state precondition as a Java boolean expression, logically negating the invalid condition where necessary.\n"
        "3. Analyze the operational usage of each parameter in the method body for implicit constraints. Consider whether the way a parameter is used imposes restrictions on its valid state. For example, dereferencing a parameter or invoking an instance method on it without a prior null check may imply a non-null constraint. Formulate each identified constraint as a Java boolean expression.\n"
        "4. Analyze semantic cues from parameter names, the target method's name, invoked method names, and other relevant code context for additional implicit constraints. For example, names such as `count`, `age`, `setAge`, or `allocate` may suggest restrictions on parameter values when supported by the context. Formulate each identified constraint as a Java boolean expression.\n"
        "5. Discard any constraints that do not satisfy the Extraction Rules.\n"
        "6. Combine the remaining constraints into the final expression.\n"
    ),
}

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


def default_variants():
    return [DEFAULT_VARIANT]


def extended_variants():
    return list(EXTENDED_VARIANTS)


def all_variants():
    return ["unst", "woex", "expl", "exim"]


def variant_label(variant):
    return variant


def build_prompt(item, variant):
    task_prompt = EXTRACTION_RULES + COT_ANALYSIS[variant] + "\n" + OUTPUT_FORMAT
    input_prompt = f"""Target Method:
{item['code_formatted']}
"""
    return task_prompt + input_prompt


def output_relpath(variant, model_name):
    return f"cot/{variant}/{model_name}.json"
