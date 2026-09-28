"""Central path configuration for the reproduction package.

All scripts resolve their input/output locations through this module so that
the package can be run from any current working directory and so that re-runs
never overwrite the shipped results.

Layout (relative to the package root)::

    reproduction_package/
    ├─ code/        (this file + evaluation scripts + llm_infer/ + expression_similarity/ + slicing/ + java_formatter/)
    ├─ data/        (code_doc_pairs.json, fewshot_index.json, code_slices.json)
    ├─ prompts/     (prompt snapshots: vanilla.txt, fewshot.txt, cot.txt, slice.txt, mix.txt)
    └─ results/     (raw/ metrics/ summary/ wit/ significance/ ; re-runs go to _regenerated/)

LLM inference lives in ``code/llm_infer/`` (5 prompt entry scripts + api/local
backends); see ``code/llm_infer/models.py`` for the model registry.

Environment overrides:
    REPRO_OUTPUT_DIR : where re-run outputs are written (default results/_regenerated)
    REPOS_DIR        : original repository root (provenance only; not required to run)
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CODE_DIR = ROOT / "code"
DATA_DIR = ROOT / "data"
PROMPTS_DIR = ROOT / "prompts"

# Shipped results (read-only inputs).
RESULTS_DIR = ROOT / "results"
RAW_DIR = RESULTS_DIR / "raw"                 # raw LLM outputs
METRICS_DIR = RESULTS_DIR / "metrics"         # similarity-evaluation CSVs
SUMMARY_DIR = RESULTS_DIR / "summary"         # per-model summaries
WIT_DIR = RESULTS_DIR / "wit"                 # WIT baseline result + metrics
SIGNIFICANCE_DIR = RESULTS_DIR / "significance"

# Re-run outputs. Kept separate so shipped results are not overwritten.
OUTPUT_DIR = Path(os.environ.get("REPRO_OUTPUT_DIR", RESULTS_DIR / "_regenerated"))
OUTPUT_RAW_DIR = OUTPUT_DIR / "raw"
OUTPUT_METRICS_DIR = OUTPUT_DIR / "metrics"
OUTPUT_SUMMARY_DIR = OUTPUT_DIR / "summary"
OUTPUT_WIT_DIR = OUTPUT_DIR / "wit"
OUTPUT_SIGNIFICANCE_DIR = OUTPUT_DIR / "significance"
OUTPUT_DATA_DIR = OUTPUT_DIR / "data"

# Original repositories (only used as provenance for the `repos/<project>/...`
# paths recorded in the dataset; not required to run the pipeline).
REPOS_DIR = Path(os.environ.get("REPOS_DIR", ROOT / "repos"))

# Data files.
CODE_DOC_PAIRS = DATA_DIR / "code_doc_pairs.json"
FEWSHOT_INDEX = DATA_DIR / "fewshot_index.json"
CODE_SLICES = DATA_DIR / "code_slices.json"

# Shipped WIT artefacts.
WIT_RESULT = WIT_DIR / "wit_result.json"
WIT_METRICS = WIT_DIR / "wit_metrics.csv"


def ensure_output_dirs() -> None:
    for d in (OUTPUT_RAW_DIR, OUTPUT_METRICS_DIR, OUTPUT_SUMMARY_DIR,
              OUTPUT_WIT_DIR, OUTPUT_SIGNIFICANCE_DIR, OUTPUT_DATA_DIR):
        d.mkdir(parents=True, exist_ok=True)


def resolve_data_file(name: str) -> Path:
    """Prefer a regenerated data file (few-shot index / slices), else shipped."""
    p = OUTPUT_DATA_DIR / name
    return p if p.exists() else DATA_DIR / name


def resolve_raw_file(rel: str) -> Path:
    """Prefer a regenerated raw LLM output, else shipped."""
    rel = Path(rel)
    p = OUTPUT_RAW_DIR / rel
    return p if p.exists() else RAW_DIR / rel


def resolve_metrics_file(rel: str) -> Path:
    """Prefer a regenerated metrics CSV, else shipped."""
    rel = Path(rel)
    p = OUTPUT_METRICS_DIR / rel
    return p if p.exists() else METRICS_DIR / rel


def resolve_wit_file(name: str) -> Path:
    p = OUTPUT_WIT_DIR / name
    return p if p.exists() else WIT_DIR / name
