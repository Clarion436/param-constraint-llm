# Reproduction Package — Extracting Parameter Constraints using LLMs

This package contains the code, the dataset, the raw LLM outputs and the
evaluation results for the study *"An Empirical Study on Extracting Parameter
Constraints using LLMs"*. It is organized as a five-stage pipeline
(1–2 build → 3 inference → 4 evaluation → 5 significance); see §1 for the layout
and §3 for the commands.

---

## 1. Directory layout

```
reproduction_package/
├─ README.md
├─ requirements.txt
├─ requirements-fewshot.txt
├─ LICENSE
├─ code/
│  ├─ config.py                     # path configuration (read this first)
│  ├─ expression_similarity/        # LIB: constraint similarity + simplification
│  ├─ slicing/                      # LIB: program slicing for the slice strategy
│  ├─ java_formatter/               # LIB: astyle-based Java formatter (used by the slicer)
│  ├─ build_fewshot_index.py        # STAGE 1 build: few-shot retrieval index (UniXcoder)
│  ├─ build_code_slices.py          # STAGE 2 build: program slicing (hop 0/1/2)
│  ├─ llm_infer/                    # STAGE 3 inference (entry points + backends + prompts)
│  │  ├─ models.py                  # model registry (API placeholders + local Ollama tags)
│  │  ├─ common.py                  # shared I/O, paths, TEST_PROMPT switch, run loop
│  │  ├─ prompts/                   # vanilla / fewshot / cot / slice / mix prompt builders
│  │  ├─ backends/                  # api_backend.py, local_backend.py
│  │  └─ run_<strategy>.py          # entry points (vanilla/fewshot/cot/slice/mix)
│  ├─ evaluate_llm.py               # STAGE 4 evaluate: similarity -> metrics/*.csv
│  ├─ evaluate_wit.py               # STAGE 4 evaluate: WIT baseline -> wit/wit_metrics.csv
│  ├─ summarize_by_model.py         # STAGE 4 evaluate: per-model summary -> summary/*.csv
│  ├─ significance_qw7b.py          # STAGE 5 significance: across prompt configs
│  └─ significance_models.py        # STAGE 5 significance: across models
├─ data/
│  ├─ code_doc_pairs.json           # Dataset: 523 code–documentation pairs
│  │                                #   (code_formatted, doc_constraint)
│  ├─ fewshot_index.json            # few-shot retrieval id lists + cosine similarities
│  └─ code_slices.json              # code slices (hop 0/1/2)
├─ prompts/                         # prompt text snapshots (the prompts are also inlined in code)
└─ results/
   ├─ raw/                          # raw LLM outputs: <strategy>/<model>[_k<k>].json
   ├─ metrics/                      # similarity-evaluation CSVs: <strategy>/<model>[_k<k>].csv
   ├─ summary/                      # per-model summaries: <model>_all_results.csv
   ├─ wit/                          # wit_result.json, wit_metrics.csv
   ├─ significance/                 # significance_*.csv
   └─ _regenerated/                 # created at run time (not shipped)
```

Pipeline: `code_doc_pairs.json` → (1–2) `data/*.json` → (3) `results/raw/` →
(4) `results/{metrics,summary,wit}/` → (5) `results/significance/`.

## 2. Environment

- Python 3.12 (the reference environment used Python 3.12.9).

- **Inference + evaluation environment.** Create a fresh environment (do not
  overwrite an existing one):

  ```bash
  conda create -n repro python=3.12 -y
  conda activate repro
  pip install -r requirements.txt
  ```

  `requirements.txt` covers **LLM inference + evaluation**.

- **Few-shot rebuild environment (separate).** Rebuilding the few-shot index is
  optional (it is shipped as `data/fewshot_index.json`). `code/build_fewshot_index.py`
  needs PyTorch, so install `requirements-fewshot.txt` into its own environment;
  the inference/evaluation environment has **no** `torch` and must not be
  overwritten. This is the only stage that uses UniXcoder, which is *not* a pip
  dependency and is resolved from the standard Hugging Face cache (reused if
  already present; pre-fetch with
  `huggingface-cli download microsoft/unixcoder-base`, and add
  `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` to run fully offline):

  ```bash
  conda create -n repro-fewshot python=3.10 -y
  conda activate repro-fewshot
  pip install -r requirements-fewshot.txt
  # for a specific CUDA build of PyTorch:
  # pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
  ```

- Ollama, the CUDA driver and the Hugging Face cache are system/user-level and
  are shared across environments; only the Python packages live in a new env.

- **Local inference** requires a running [Ollama](https://ollama.com) server with
  the locally-deployed models listed in `code/llm_infer/models.py`
  (`LOCAL_MODELS`). All are the **non-quantized (fp16)** tags used in the study.
  Download them once with `ollama pull`:

  ```bash
  ollama pull llama3.1:8b-instruct-fp16          # lma8b
  ollama pull qwen2.5:3b-instruct-fp16           # qw3b
  ollama pull qwen2.5:7b-instruct-fp16           # qw7b
  ollama pull qwen2.5:14b-instruct-fp16          # qw14b
  ollama pull qwen2.5-coder:7b-instruct-fp16     # qwc7b
  ```

- **API inference** requires, for each API model, its **own pair of placeholder
  environment variables** (e.g. `GPT4OM_API_KEY` / `GPT4OM_BASE_URL`); fill in
  your own key and base URL. See `code/llm_infer/models.py` (`API_MODELS`).

## 3. What to run

All scripts read/write paths through `code/config.py` and can be launched from
any working directory, e.g. `python code/evaluate_llm.py`. By default a re-run
writes under `results/_regenerated/` (override with `REPRO_OUTPUT_DIR`), so the
shipped `results/` and `data/` are never overwritten and the two can be compared.

| Stage | Script | Output |
|---|---|---|
| 1. Build few-shot index (optional) | `code/build_fewshot_index.py` | `data/fewshot_index.json` |
| 2. Build code slices (optional) | `code/build_code_slices.py` | `data/code_slices.json` |
| 3. LLM inference | `code/llm_infer/run_<strategy>.py` | `results/raw/<strategy>/<model>.json` |
| 4. Evaluation | `code/evaluate_llm.py`, `code/evaluate_wit.py`, `code/summarize_by_model.py` | `results/metrics/`, `results/wit/`, `results/summary/` |
| 5. Significance | `code/significance_qw7b.py`, `code/significance_models.py` | `results/significance/*.csv` |

- Stages 1–2 are optional (their outputs are shipped) and need extra
  prerequisites: stage 1 needs the separate few-shot rebuild environment (see §2:
  `requirements-fewshot.txt` + GPU + UniXcoder); stage 2 needs
  `tree-sitter`/`tiktoken` and uses the bundled `code/java_formatter/`. Stage 2
  rebuilds the shipped `data/code_slices.json`.
- Stage 3 runs the `vanilla` / `fewshot` / `cot` / `slice` / `mix` prompt
  strategies. Each entry script sets its configuration via module-level constants
  (no CLI); the default runs all 11 models of the study, routed to the Ollama or
  API backend automatically. Setting `TEST_PROMPT = 1` writes only the
  `prompts/<strategy>.txt` snapshot and exits without calling a model.
- Stage 4: `evaluate_llm.py` scores each raw output against the documented
  constraints with the `expression_similarity/` library (strict/relaxed
  precision/recall/F1) and pins `PYTHONHASHSEED=0` for deterministic SymPy
  simplification; `evaluate_wit.py` scores the frozen
  `results/wit/wit_result.json` baseline; `summarize_by_model.py` aggregates the
  metrics into one row per model.
- Stage 5 reads `metrics/`, `summary/` and `wit/` and runs pairwise tests with
  Holm correction (`significance_qw7b.py` across prompt configurations on qw7b,
  `significance_models.py` across models).
