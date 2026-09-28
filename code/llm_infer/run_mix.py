"""Entry point for the mixed prompt strategy (CoT + few-shot + slice).

Run from anywhere:

    python code/llm_infer/run_mix.py

Configuration is via the constants below (no CLI). Every one of the 11 models
runs the single ``mix`` configuration (there is no benchmark special-casing).
The default configuration produces exactly the shipped ``results/raw/mix/*.json``.
"""

from __future__ import annotations

import common
import models
from prompts import mix

# 0 = run over the full dataset; 1 = only write prompts/mix.txt and exit.
TEST_PROMPT = 0

MODELS_TO_RUN = models.MODELS

# mix has a single configuration for every model -> no benchmark special-casing.
BENCHMARK_MODEL = None


if __name__ == "__main__":
    common.run(
        MODELS_TO_RUN, mix,
        benchmark_model=BENCHMARK_MODEL,
        test_prompt=TEST_PROMPT,
        api_workers=2,
        local_sleep=0.1,
        local_restart_on_timeout=True,
        local_restart_after=True,
    )
