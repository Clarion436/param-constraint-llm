"""Entry point for the few-shot prompt strategy.

Run from anywhere:

    python code/llm_infer/run_fewshot.py

Configuration is via the constants below (no CLI). Default: all 11 models run
``sr-16``; the benchmark model (``qw7b``) additionally runs ``rand``/``sr`` for
every ``k``. The default configuration produces exactly the shipped
``results/raw/fewshot/**`` files.
"""

from __future__ import annotations

import common
import models
from prompts import fewshot

# 0 = run over the full dataset; 1 = only write prompts/fewshot.txt and exit.
TEST_PROMPT = 0

MODELS_TO_RUN = models.MODELS
BENCHMARK_MODEL = models.BENCHMARK_MODEL


if __name__ == "__main__":
    common.run(
        MODELS_TO_RUN, fewshot,
        benchmark_model=BENCHMARK_MODEL,
        test_prompt=TEST_PROMPT,
        api_workers=4,
        local_sleep=0.1,
        local_restart_on_timeout=True,
        local_restart_after=True,
    )
