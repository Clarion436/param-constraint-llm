"""Entry point for the chain-of-thought prompt strategy.

Run from anywhere:

    python code/llm_infer/run_cot.py

Configuration is via the constants below (no CLI). Default: all 11 models run
``exim``; the benchmark model (``qw7b``) additionally runs ``unst``/``woex``/
``expl``. The default configuration produces exactly the shipped
``results/raw/cot/**`` files.
"""

from __future__ import annotations

import common
import models
from prompts import cot

# 0 = run over the full dataset; 1 = only write prompts/cot.txt and exit.
TEST_PROMPT = 0

MODELS_TO_RUN = models.MODELS
BENCHMARK_MODEL = models.BENCHMARK_MODEL


if __name__ == "__main__":
    common.run(
        MODELS_TO_RUN, cot,
        benchmark_model=BENCHMARK_MODEL,
        test_prompt=TEST_PROMPT,
        api_workers=4,
        local_sleep=0.1,
        local_restart_on_timeout=True,
        local_restart_after=True,
    )
