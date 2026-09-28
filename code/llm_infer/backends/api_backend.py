"""API backend: concurrent, rate-limited calls to an OpenAI-compatible endpoint.

The API key and base URL are read from the per-model placeholder environment
variables declared in :data:`models.API_MODELS` (see that file).
"""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import backoff
from openai import OpenAI, BadRequestError, AuthenticationError

import common
from models import API_MODELS


def _on_backoff(details):
    exc = details.get("exception") or details.get("value")
    exc_name = exc.__class__.__name__ if isinstance(exc, Exception) else type(exc).__name__
    exc_msg = str(exc) if isinstance(exc, Exception) else str(exc)
    print(f"Backoff: {exc_name}: {exc_msg} — waiting {details.get('wait', 0):0.1f}s, "
          f"try #{details.get('tries')}")


def _fatal_error(e):
    """Errors that should NOT be retried (400 / 401 / 403)."""
    return isinstance(e, (BadRequestError, AuthenticationError))


def _build_client(model_name: str):
    info = API_MODELS[model_name]
    api_key = os.environ.get(info["key_env"])
    base_url = os.environ.get(info["base_url_env"])
    if not api_key:
        raise RuntimeError(f"Environment variable {info['key_env']} is not set.")
    if not base_url:
        raise RuntimeError(f"Environment variable {info['base_url_env']} is not set.")
    client = OpenAI(api_key=api_key, base_url=base_url)
    return client, info["model_id"], info["qpm"]


@backoff.on_exception(
    backoff.expo,
    Exception,
    max_tries=10,
    on_backoff=_on_backoff,
    jitter=backoff.full_jitter,
    giveup=_fatal_error,
)
def call_api(prompt: str, client: OpenAI, api_model_id: str):
    """Call the API model (with retry) and return response text + usage."""
    resp = client.chat.completions.create(
        model=api_model_id,
        messages=[
            {"role": "system", "content": common.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=2048,
    )
    content = resp.choices[0].message.content
    if not content:
        raise ValueError("Empty response from API")
    return content, resp.usage


_rate_lock = threading.Lock()


def infer_one_rate_limited(idx: int, item: dict, prompt_fn, client: OpenAI,
                           api_model_id: str, rps: float):
    """Rate-limited inference on a single dataset item."""
    try:
        delay = 1.0 / max(rps, 1e-6)
    except Exception:
        delay = 0.0
    with _rate_lock:
        if delay > 0:
            time.sleep(delay)

    try:
        prompt = prompt_fn(item)
        t0 = time.time()
        text, usage = call_api(prompt, client, api_model_id)
        elapsed = time.time() - t0
        result = {
            "text": text,
            "input_tokens": usage.prompt_tokens if usage else None,
            "output_tokens": usage.completion_tokens if usage else None,
            "elapsed_seconds": round(elapsed, 3),
        }
    except Exception as e:
        result = {"error": str(e)}

    return idx, result


def run_batch(items: list, prompt_fn, model_name: str, output_path: str,
              workers: int = 4) -> None:
    """Run the API model over all items concurrently and write the raw JSON."""
    client, api_model_id, qpm = _build_client(model_name)
    rps = qpm / 60.0 * 0.9

    print("\n" + "=" * 80)
    print(f"Model: {api_model_id} ({model_name})  |  QPM={qpm}  |  workers={workers}")
    print("=" * 80)

    start_time = time.time()
    results = [None] * len(items)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(infer_one_rate_limited, i, item, prompt_fn,
                            client, api_model_id, rps)
            for i, item in enumerate(items)
        ]
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result
            print("*" * 80)
            print(f"[{model_name}] Processing item {idx}, id = {items[idx]['id']}:")
            print(f"{result}")

    output = []
    for idx, item in enumerate(items):
        entry = {"id": item["id"]}
        r = results[idx]
        if "error" in r:
            entry["error"] = r["error"]
        else:
            entry["text"] = r["text"]
            entry["input_tokens"] = r.get("input_tokens")
            entry["output_tokens"] = r.get("output_tokens")
            entry["elapsed_seconds"] = r.get("elapsed_seconds")
        output.append(entry)

    common.write_json(output, output_path)
    print(f"Final results saved to {output_path}")
    print(f"[{model_name}] Done! Time cost: {time.time() - start_time:.1f}s")
