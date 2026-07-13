"""Minimal client for talking to a local Ollama server."""
import requests

from config import OLLAMA_HOST, MODEL_TIMEOUT_SECONDS


class OllamaError(RuntimeError):
    pass


def _error_body(resp: requests.Response) -> str:
    """Best-effort extraction of Ollama's actual error message from a failed
    response, e.g. {"error": "..."}-- falls back to raw response text."""
    try:
        data = resp.json()
        if isinstance(data, dict) and data.get("error"):
            return str(data["error"])
    except ValueError:
        pass
    return (resp.text or "").strip()[:500] or f"HTTP {resp.status_code}"


def call_model(
    model: str,
    messages: list[dict],
    temperature: float = 0.3,
    keep_alive: str | int | None = None,
    num_ctx: int | None = None,
    num_gpu: int | None = None,
    timeout: int | None = None,
) -> str:
    """Send a chat-style request to Ollama and return the assistant's raw text.

    messages: list of {"role": "system"|"user"|"assistant", "content": str}
    keep_alive: how long Ollama keeps this model loaded in VRAM after this
    call. Pass 0 to unload immediately (frees VRAM right away for the next
    model in a sequential Qwen->DeepSeek swap on limited VRAM); pass None
    to use Ollama's own default (keeps it warm for a few minutes, useful
    when the same model will likely be called again soon, e.g. consecutive
    concept questions).
    num_ctx: override the model's context window size for this call. Most
    Ollama modelfiles default to 4096 tokens, which is easy to blow past
    once you're feeding in a multi-thousand-character document excerpt
    (Ollama then rejects the request with a 400 exceed_context_size_error
    instead of truncating). Pass a larger value (e.g. 8192) for calls that
    need to see a bigger chunk of text; leave None for normal short prompts.
    timeout: override MODEL_TIMEOUT_SECONDS for this call. Useful for
    reasoning models (e.g. DeepSeek-R1) doing a task with a lot of internal
    "thinking" tokens before the real answer -- these can legitimately take
    much longer than a normal chat turn, especially for background jobs
    where there's no UI request waiting on the result.
    num_gpu: number of model layers to force onto the GPU (e.g. 99 to force
    every layer). Ollama auto-decides this normally; on a tight 8GB card it's
    occasionally conservative, so this lets you force full GPU offload for
    lower latency. Pass config.OLLAMA_NUM_GPU (None by default -- Ollama's
    own default behavior) unless you've set the OLLAMA_NUM_GPU env var.
    """
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if num_ctx is not None:
        body["options"]["num_ctx"] = num_ctx
    if num_gpu is not None:
        body["options"]["num_gpu"] = num_gpu
    if keep_alive is not None:
        body["keep_alive"] = keep_alive
    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json=body,
            timeout=timeout if timeout is not None else MODEL_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        raise OllamaError(
            f"Could not reach Ollama at {OLLAMA_HOST}. Is 'ollama serve' running?"
        ) from e
    except requests.exceptions.Timeout as e:
        raise OllamaError(
            f"Ollama timed out after {timeout if timeout is not None else MODEL_TIMEOUT_SECONDS}s calling model '{model}'."
        ) from e
    except requests.exceptions.HTTPError as e:
        detail = _error_body(resp)
        raise OllamaError(f"Ollama rejected the request to model '{model}': {detail}") from e

    data = resp.json()
    if "message" not in data or "content" not in data["message"]:
        raise OllamaError(f"Unexpected response shape from Ollama: {data}")
    return data["message"]["content"]


def embed_text(model: str, text: str, keep_alive: str | int | None = None) -> list[float]:
    """Get an embedding vector for a single piece of text (used for one-off
    query embeddings at question time, where batching doesn't apply).
    keep_alive: pass e.g. "30m" during a long ingestion run so the embedding
    model doesn't unload and reload between books; leave None for normal
    one-off query embeddings at chat time."""
    body = {"model": model, "prompt": text}
    if keep_alive is not None:
        body["keep_alive"] = keep_alive
    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/embeddings",
            json=body,
            timeout=MODEL_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        raise OllamaError(
            f"Could not reach Ollama at {OLLAMA_HOST} for embeddings. "
            "Is 'ollama serve' running, and have you run "
            f"'ollama pull {model}'?"
        ) from e
    except requests.exceptions.Timeout as e:
        raise OllamaError(f"Ollama timed out generating an embedding with '{model}'.") from e
    except requests.exceptions.HTTPError as e:
        raise OllamaError(f"Ollama rejected the embedding request for '{model}': {_error_body(resp)}") from e

    data = resp.json()
    if "embedding" not in data:
        raise OllamaError(f"Unexpected embeddings response from Ollama: {data}")
    return data["embedding"]


def embed_texts(model: str, texts: list[str], keep_alive: str | int | None = None) -> list[list[float]]:
    """Batch-embed many texts in ONE HTTP call via Ollama's newer /api/embed
    endpoint (as opposed to /api/embeddings, which only ever does one text
    per call). This is the single biggest lever on ingestion speed -- for a
    1500-chunk book, it's the difference between 1500 round trips and ~50.

    Requires a reasonably recent Ollama version. If /api/embed isn't
    available (older Ollama), raises OllamaError -- callers should catch
    this and fall back to embed_text() one at a time.
    """
    if not texts:
        return []
    body = {"model": model, "input": texts}
    if keep_alive is not None:
        body["keep_alive"] = keep_alive
    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/embed",
            json=body,
            timeout=MODEL_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        raise OllamaError(
            f"Could not reach Ollama at {OLLAMA_HOST} for batch embeddings. "
            "Is 'ollama serve' running?"
        ) from e
    except requests.exceptions.Timeout as e:
        raise OllamaError(f"Ollama timed out generating batch embeddings with '{model}'.") from e
    except requests.exceptions.HTTPError as e:
        raise OllamaError(
            f"Ollama's batch /api/embed endpoint returned an error ({_error_body(resp)}) -- "
            "your Ollama version may be too old to support it."
        ) from e

    data = resp.json()
    embeddings = data.get("embeddings")
    if not embeddings or len(embeddings) != len(texts):
        raise OllamaError(f"Unexpected batch embeddings response shape from Ollama: {data}")
    return embeddings
