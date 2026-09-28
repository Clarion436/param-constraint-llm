"""Entry point for the vanilla prompt strategy.

Run from anywhere:

    python code/llm_infer/run_vanilla.py

Configuration is via the constants below (no CLI). The default configuration
(all 11 models) produces exactly the shipped ``results/raw/vanilla/*.json``.
"""

from __future__ import annotations

import common
import models
from prompts import vanilla

# 0 = run over the full dataset; 1 = only write prompts/vanilla.txt and exit.
TEST_PROMPT = 0

# All 11 models. Each is routed to the API or local backend
# automatically, based on models.API_MODELS / models.LOCAL_MODELS.
MODELS_TO_RUN = models.MODELS

# vanilla has a single configuration for every model -> no benchmark special-casing.
BENCHMARK_MODEL = None


if __name__ == "__main__":
    common.run(
        MODELS_TO_RUN, vanilla,
        benchmark_model=BENCHMARK_MODEL,
        test_prompt=TEST_PROMPT,
        api_workers=4,
        local_sleep=0.5,
        local_restart_on_timeout=False,
        local_restart_after=False,
    )
