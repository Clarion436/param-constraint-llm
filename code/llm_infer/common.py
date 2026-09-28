"""Shared helpers for the LLM-inference entry scripts.

Path handling, JSON I/O, the ``TEST_PROMPT`` switch and the per-strategy run loop
live here. Each ``run_<strategy>.py`` only declares its configuration constants and
calls :func:`run`.

Configuration style (no CLI): every option is a module-level constant in the
``run_*.py`` entry script.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make ``import config`` work regardless of the current working directory.
_CODE_DIR = Path(__file__).resolve().parent.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import config  # noqa: E402

SYSTEM_PROMPT = "You are an expert in Java code analysis and automated software engineering."

# Dataset item used when TEST_PROMPT = 1 (0-based index).
TEST_PROMPT_INDEX = 10


def read_json(path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def dump_test_prompt(prompt_module, items) -> None:
    """Write one prompt snapshot per variant, using a single dataset item."""
    item = items[TEST_PROMPT_INDEX]
    text = ""
    for variant in prompt_module.all_variants():
        text += f"--- {prompt_module.variant_label(variant)} ---\n"
        text += prompt_module.build_prompt(item, variant) + "\n\n"
    out = config.PROMPTS_DIR / f"{prompt_module.STRATEGY}.txt"
    out.write_text(text, encoding="utf-8")
    print(f"Wrote prompt snapshot: {out}")


def _run_variant(model, variant, prompt_module, items, *,
                 api_workers, local_sleep, local_restart_on_timeout,
                 local_restart_after):
    from backends import api_backend, local_backend
    from models import backend_of, LOCAL_MODELS

    rel = prompt_module.output_relpath(variant, model)
    out = config.OUTPUT_RAW_DIR / rel
    out.parent.mkdir(parents=True, exist_ok=True)

    def prompt_fn(item):
        return prompt_module.build_prompt(item, variant)

    if backend_of(model) == "api":
        api_backend.run_batch(items, prompt_fn, model, str(out),
                              workers=api_workers)
    else:
        local_backend.run_batch(items, prompt_fn, LOCAL_MODELS[model], str(out),
                                sleep=local_sleep,
                                restart_on_timeout=local_restart_on_timeout)
        if local_restart_after:
            local_backend.restart_llama_server()


def run(models_to_run, prompt_module, *, benchmark_model=None, test_prompt=0,
        api_workers=4, local_sleep=0.1, local_restart_on_timeout=True,
        local_restart_after=True):
    """Run one prompt strategy over the given models.

    Parameters
    ----------
    models_to_run : list[str]
        Model short names (mix of API and local; each is routed automatically).
    prompt_module : module
        One of ``prompts.vanilla`` / ``prompts.fewshot`` / ``prompts.cot`` /
        ``prompts.slice`` / ``prompts.mix``.
    benchmark_model : str | None
        If set, this model additionally runs the strategy's extended variants;
        every other model runs only the default variants.
    test_prompt : int
        0 = actually run over the full dataset; 1 = only write prompt snapshots
        (for ``TEST_PROMPT_INDEX``) and exit without calling any LLM.
    """
    items = read_json(config.CODE_DOC_PAIRS)

    if test_prompt:
        dump_test_prompt(prompt_module, items)
        return

    opts = dict(api_workers=api_workers,
                local_sleep=local_sleep,
                local_restart_on_timeout=local_restart_on_timeout,
                local_restart_after=local_restart_after)

    for model in models_to_run:
        for variant in prompt_module.default_variants():
            _run_variant(model, variant, prompt_module, items, **opts)
        if benchmark_model and model == benchmark_model:
            for variant in prompt_module.extended_variants():
                _run_variant(model, variant, prompt_module, items, **opts)
