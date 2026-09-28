"""Entry point for the program-slice prompt strategy.

Run from anywhere:

    python code/llm_infer/run_slice.py

Configuration is via the constants below (no CLI). Default: all 11 models run
``slextra``; the benchmark model (``qw7b``) additionally runs ``slonly``. The
default configuration produces exactly the shipped ``results/raw/slice/**`` files.
"""

from __future__ import annotations

import common
import models
from prompts import slice as slice_prompt

# 0 = run over the full dataset; 1 = only write prompts/slice.txt and exit.
TEST_PROMPT = 0

MODELS_TO_RUN = models.MODELS
BENCHMARK_MODEL = models.BENCHMARK_MODEL


if __name__ == "__main__":
    common.run(
        MODELS_TO_RUN, slice_prompt,
        benchmark_model=BENCHMARK_MODEL,
        test_prompt=TEST_PROMPT,
        api_workers=4,
        local_sleep=0.1,
        local_restart_on_timeout=True,
        local_restart_after=True,
    )
