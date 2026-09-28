"""Local Ollama backend: sequential calls with warm-up and retry.

The Ollama server is auto-started if needed. When ``restart_on_timeout`` is set,
a stuck server (detected via a timeout) triggers a ``pkill`` of ``llama-server``
so that Ollama reloads the model.

Note: this backend targets Linux/macOS (it uses ``pkill``); it is not intended
for Windows.
"""

from __future__ import annotations

import subprocess
import time
import urllib.request

import backoff
from openai import OpenAI

import common
from models import LOCAL_MODELS

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_API_KEY = "ollama"  # placeholder, Ollama does not check this

_client = OpenAI(api_key=OLLAMA_API_KEY, base_url=OLLAMA_BASE_URL)

MAX_OLLAMA_RESTARTS = 2
_ollama_restart_count = 0


# ---------------------------------------------------------------------------
# Ollama lifecycle
# ---------------------------------------------------------------------------
def ensure_ollama_running(timeout: int = 30) -> None:
    """Check if Ollama is reachable; launch `ollama serve` in the background if not."""
    try:
        urllib.request.urlopen(OLLAMA_BASE_URL + "/models", timeout=3)
        print("Ollama is already running.")
        return
    except Exception:
        pass

    print("Starting ollama serve in background...")
    subprocess.Popen(["ollama", "serve"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(timeout):
        time.sleep(1)
        try:
            urllib.request.urlopen(OLLAMA_BASE_URL + "/models", timeout=3)
            print("Ollama is ready.")
            return
        except Exception:
            pass
    raise RuntimeError(
        f"Ollama failed to start within {timeout}s. "
        "Check `ollama serve` manually or inspect logs."
    )


def restart_llama_server() -> None:
    """Kill llama-server so the next call reloads a clean model state."""
    subprocess.run(["pkill", "-f", "llama-server"], capture_output=True)
    time.sleep(3)


def _quick_health_check(model: str, timeout: int = 10) -> bool:
    try:
        _client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1, temperature=0.0, timeout=timeout,
        )
        return True
    except Exception:
        return False


def _restart_ollama(model: str, max_wait: int = 120) -> None:
    print("  → Killing llama-server...")
    subprocess.run(["pkill", "-f", "llama-server"], capture_output=True)
    time.sleep(3)
    print(f"  → Waiting for Ollama to reload model '{model}' (may take 30-60s)...")
    t0 = time.time()
    for _ in range(max_wait):
        try:
            _client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1, temperature=0.0, timeout=60,
            )
            print(f"  → Ollama restarted successfully in {time.time() - t0:.0f}s.")
            return
        except Exception:
            time.sleep(1)
    raise RuntimeError(f"Ollama failed to reload model '{model}' within {max_wait}s")


# ---------------------------------------------------------------------------
# Calls (with retry)
# ---------------------------------------------------------------------------
def _chat(prompt: str, model: str):
    resp = _client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": common.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=2048,
        timeout=120.0,
    )
    content = resp.choices[0].message.content
    if not content:
        raise ValueError("Empty response from Ollama")
    return content, resp.usage


def _log_backoff(details) -> None:
    exc = details.get("exception") or details.get("value")
    exc_name = exc.__class__.__name__ if isinstance(exc, Exception) else type(exc).__name__
    exc_msg = str(exc) if isinstance(exc, Exception) else str(exc)
    print(f"Backoff: {exc_name}: {exc_msg} — waiting {details.get('wait', 0):0.1f}s, "
          f"try #{details.get('tries')}")


def _on_backoff_restart(details) -> None:
    """Log backoff events; on a timeout, restart Ollama if it appears stuck."""
    global _ollama_restart_count

    _log_backoff(details)

    exc = details.get("exception") or details.get("value")
    exc_name = exc.__class__.__name__ if isinstance(exc, Exception) else type(exc).__name__
    exc_msg = str(exc) if isinstance(exc, Exception) else str(exc)
    is_timeout = ("timeout" in exc_msg.lower() or "timed out" in exc_msg.lower()
                  or "timeout" in exc_name.lower())

    if is_timeout and _ollama_restart_count < MAX_OLLAMA_RESTARTS:
        tries = details.get("tries", 0)
        if tries == 1:
            args = details.get("args", [])
            model = args[1] if len(args) >= 2 else None
            if model is None:
                return
            print("  ⚠️  Timeout detected — checking if Ollama is stuck...")
            if not _quick_health_check(model, timeout=10):
                _ollama_restart_count += 1
                print(f"  ⚠️  Ollama is unresponsive. Auto-restarting "
                      f"(attempt {_ollama_restart_count}/{MAX_OLLAMA_RESTARTS})...")
                try:
                    _restart_ollama(model)
                except Exception as e:
                    print(f"  ✗ Failed to restart Ollama: {e}")
            else:
                print("  ✓ Ollama health check passed — will retry without restart.")


@backoff.on_exception(backoff.expo, Exception, max_tries=5,
                      on_backoff=_log_backoff, jitter=backoff.full_jitter)
def call_ollama(prompt: str, model: str):
    return _chat(prompt, model)


@backoff.on_exception(backoff.expo, Exception, max_tries=5,
                      on_backoff=_on_backoff_restart, jitter=backoff.full_jitter)
def call_ollama_with_restart(prompt: str, model: str):
    return _chat(prompt, model)


# ---------------------------------------------------------------------------
# Batch run
# ---------------------------------------------------------------------------
def infer_one(idx: int, item: dict, prompt_fn, model: str, call):
    try:
        prompt = prompt_fn(item)
        t0 = time.time()
        text, usage = call(prompt, model)
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


def run_batch(items: list, prompt_fn, model: str, output_path: str, *,
              sleep: float = 0.1, restart_on_timeout: bool = True) -> None:
    """Run the local Ollama model over all items sequentially; write raw JSON."""
    global _ollama_restart_count

    ensure_ollama_running()
    _ollama_restart_count = 0
    call = call_ollama_with_restart if restart_on_timeout else call_ollama

    print("\n" + "=" * 80)
    print(f"Model: {model}")
    print("=" * 80)

    # Warm-up: first call discarded to load the model into VRAM/RAM.
    if items:
        print("Warming up model (first call discarded)...")
        t_warmup = time.time()
        try:
            call(prompt_fn(items[0]), model)
            print(f"Warm-up done in {time.time() - t_warmup:.1f}s")
        except Exception as e:
            print(f"Warm-up failed (continuing anyway): {e}")

    start_time = time.time()
    results = [None] * len(items)

    for i, item in enumerate(items):
        idx, result = infer_one(i, item, prompt_fn, model, call)
        results[idx] = result
        print("*" * 80)
        print(f"Processing item {idx}, id = {items[idx]['id']}:")
        print(f"{result}")
        time.sleep(sleep)

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
    print(f"[{model}] Done! Time cost: {time.time() - start_time:.1f}s")
