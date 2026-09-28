"""Model registry for LLM inference (API + local Ollama).

Provider identity is intentionally hidden: every API model reads its key and base
URL from its **own pair of placeholder environment variables**. Fill in your own
values before running, e.g.::

    export GPT4OM_API_KEY="<your-api-key>"
    export GPT4OM_BASE_URL="<your-base-url>"

The study's model set is :data:`MODELS` (11 models); :data:`BENCHMARK_MODEL`
is the single model that additionally runs the extended prompt configurations.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# API models: short_name -> {model_id, key_env, base_url_env, qpm}
# ---------------------------------------------------------------------------
# `model_id` is the provider-side model identifier; it is a placeholder here so
# that no provider is revealed. Replace it with the id of the model you use.
API_MODELS = {
    "gpt4o":  {"model_id": "<your-model-id>", "key_env": "GPT4O_API_KEY",  "base_url_env": "GPT4O_BASE_URL",  "qpm": 1000},
    "gpt4om": {"model_id": "<your-model-id>", "key_env": "GPT4OM_API_KEY", "base_url_env": "GPT4OM_BASE_URL", "qpm": 1000},
    "sonnet": {"model_id": "<your-model-id>", "key_env": "SONNET_API_KEY", "base_url_env": "SONNET_BASE_URL", "qpm": 1000},
    "lma70b": {"model_id": "<your-model-id>", "key_env": "LMA70B_API_KEY", "base_url_env": "LMA70B_BASE_URL", "qpm": 1000},
    "qw32b":  {"model_id": "<your-model-id>", "key_env": "QW32B_API_KEY",  "base_url_env": "QW32B_BASE_URL",  "qpm": 100},
    "qw72b":  {"model_id": "<your-model-id>", "key_env": "QW72B_API_KEY",  "base_url_env": "QW72B_BASE_URL",  "qpm": 100},
}

# ---------------------------------------------------------------------------
# Local models (Ollama): short_name -> model tag (download with `ollama pull`)
# ---------------------------------------------------------------------------
LOCAL_MODELS = {
    "lma8b": "llama3.1:8b-instruct-fp16",
    "qw3b":  "qwen2.5:3b-instruct-fp16",
    "qw7b":  "qwen2.5:7b-instruct-fp16",
    "qw14b": "qwen2.5:14b-instruct-fp16",
    "qwc7b": "qwen2.5-coder:7b-instruct-fp16",
}

# ---------------------------------------------------------------------------
# Model set
# ---------------------------------------------------------------------------
MODELS = [
    "lma8b", "qw3b", "qw7b", "qw14b", "qwc7b",
    "gpt4o", "gpt4om", "sonnet", "lma70b", "qw32b", "qw72b",
]

# The single benchmark model that additionally runs the extended configurations.
BENCHMARK_MODEL = "qw7b"


def backend_of(name: str) -> str:
    """Return "api" or "local" for a model short name."""
    if name in API_MODELS:
        return "api"
    if name in LOCAL_MODELS:
        return "local"
    raise KeyError(f"Unknown model: {name!r}")
