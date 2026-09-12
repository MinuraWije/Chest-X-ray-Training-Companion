%%writefile /kaggle/working/medgemma_server.py
"""medgemma_server.py — a small FastAPI app that loads google/medgemma-1.5-4b-it
and exposes a single POST /generate endpoint. Designed to run inside a
Kaggle notebook; the public URL is created by `pyngrok`.

Reference invocation (the same shape this server uses):

    from transformers import pipeline

    pipe = pipeline(
        "image-text-to-text",
        model="google/medgemma-1.5-4b-it",
    )

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image_url_or_path},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    result = pipe(text=messages, max_new_tokens=300)
    assistant_text = result[0]["generated_text"][-1]["content"]

Notebook usage: see `kaggle_runtime/NOTEBOOK_CELLS.md` for the five
paste-ready cells. The short version, once this file is on the container's
path (that doc writes it to /kaggle/working):

    %pip install -q pyngrok accelerate bitsandbytes
    %pip install -q -U transformers   # then restart the kernel

    import sys; sys.path.insert(0, "/kaggle/working")
    from medgemma_server import start_in_background
    ngrok_url = start_in_background()   # loads the model, then opens the tunnel
    print("Public URL:", ngrok_url)
    # Paste the URL into backend/.env as MEDGEMMA_TUNNEL_URL=...

Keep the notebook process alive afterwards; the tunnel closes with it.

Images arrive either as a location the container can open
(`image_url_or_path`) or, normally, as inline `image_base64` bytes,
because the backend's manifest paths do not exist here. The server logs
only the prompt length and the image size, never the image bytes.
"""
from __future__ import annotations

import argparse
import base64
import logging
import os
import tempfile
import threading
from collections.abc import Generator
from contextlib import contextmanager, suppress
from typing import Any

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger("medgemma_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

# Bumped whenever this file changes, so /healthz can prove which copy the
# container is actually running. A stale copy is otherwise invisible.
SERVER_VERSION = "0.4.0"
DEFAULT_MODEL = "google/medgemma-1.5-4b-it"
DEFAULT_MAX_NEW_TOKENS = 300
PIPELINE: Any = None
PIPELINE_PRECISION: str = ""
PIPELINE_LOCK = threading.Lock()


class GenerateRequest(BaseModel):
    # Exactly one of these two. The backend normally sends bytes, because a
    # manifest path like "/images/x.png" does not exist in this container.
    image_url_or_path: str | None = Field(default=None, max_length=2000)
    image_base64: str | None = Field(default=None, repr=False)
    image_mime: str = Field(default="image/png")
    prompt: str = Field(min_length=1, max_length=8000)
    system: str = Field(default="", max_length=4000)
    max_new_tokens: int = Field(default=DEFAULT_MAX_NEW_TOKENS, ge=1, le=2048)

    @model_validator(mode="after")
    def _exactly_one_image_source(self) -> GenerateRequest:
        has_path = bool(self.image_url_or_path)
        has_bytes = bool(self.image_base64)
        if has_path == has_bytes:
            raise ValueError("Provide exactly one of 'image_url_or_path' or 'image_base64'.")
        return self


class GenerateResponse(BaseModel):
    text: str
    prompt_tokens: int
    output_tokens: int
    model: str


VALID_PRECISIONS = ("4bit", "8bit", "bf16", "fp32", "fp16")
DEFAULT_PRECISION = "bf16"


def _resolve_precision() -> str:
    """Read the precision from the environment, honouring the older flag.

    `MEDGEMMA_PRECISION` wins; `MEDGEMMA_USE_4BIT=0` is still respected so an
    older notebook cell keeps working, and now actually means something — it
    used to select float32 with no device map, i.e. the model on CPU.
    """
    precision = os.environ.get("MEDGEMMA_PRECISION", "").strip().lower()
    if not precision:
        precision = "4bit" if os.environ.get("MEDGEMMA_USE_4BIT", "1") == "1" else DEFAULT_PRECISION
    if precision not in VALID_PRECISIONS:
        logger.warning(
            "Unknown MEDGEMMA_PRECISION=%r; falling back to %s.", precision, DEFAULT_PRECISION
        )
        return DEFAULT_PRECISION
    if precision == "fp16":
        # Kept selectable so the finding stays reproducible, but it does not work:
        # see the float16 note in _load_pipeline.
        logger.warning(
            "MEDGEMMA_PRECISION=fp16 produces empty generations with this model. "
            "Use bf16 or fp32 for unquantised weights."
        )
    return precision


def _load_pipeline(model_name: str, precision: str) -> Any:
    import torch
    from transformers import pipeline  # local import: heavy + needs the GPU

    # `device_map` is required in every branch. Without it the pipeline leaves
    # the weights on the CPU, where a 4B multimodal model is unusably slow.
    model_kwargs: dict[str, Any] = {"device_map": "auto"}
    if precision in ("bf16", "fp32", "fp16"):
        # Gemma-family models are trained in bfloat16 and carry activations that
        # overflow float16's much smaller exponent range. In float16 they produce
        # NaNs and the model emits EOS immediately — a 200 response with an empty
        # `text`, no error anywhere. bfloat16 has the same exponent range as
        # float32, so it is the correct half-precision choice even on Kaggle's
        # pre-Ampere cards, where torch runs it without hardware acceleration.
        # fp32 is the unambiguous fallback: ~17GB, so it needs the two-GPU T4
        # setting, which `device_map="auto"` shards across automatically.
        model_kwargs["torch_dtype"] = {
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
            "fp16": torch.float16,
        }[precision]
    else:
        from transformers import BitsAndBytesConfig

        if precision == "8bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            # bnb's 4-bit compute dtype defaults to float32, which is why the
            # 4-bit path never hit the float16 overflow above.
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

    logger.info("Loading pipeline: model=%s precision=%s", model_name, precision)
    return pipeline("image-text-to-text", model=model_name, model_kwargs=model_kwargs)


def _ensure_pipeline(model_name: str, precision: str) -> Any:
    global PIPELINE, PIPELINE_PRECISION
    with PIPELINE_LOCK:
        if PIPELINE is None:
            PIPELINE = _load_pipeline(model_name, precision)
            PIPELINE_PRECISION = precision
        return PIPELINE


def _count_output_tokens(tokenizer: Any, text: str) -> int:
    """Real token count. A word count would misreport the thesis's token metric."""
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        return max(1, len(text.split()))


def _encoded_length(encoded: Any) -> int:
    """Length of whatever apply_chat_template returned (list, tensor, or mapping)."""
    if isinstance(encoded, dict) or hasattr(encoded, "keys"):
        encoded = encoded["input_ids"]
    shape = getattr(encoded, "shape", None)
    if shape is not None:
        return int(shape[-1])
    if encoded and isinstance(encoded[0], (list, tuple)):
        return len(encoded[0])
    return len(encoded)


def _count_tokens(pipe: Any, messages: list[dict[str, Any]]) -> int:
    """Prompt token count, including the image's tokens where possible.

    The pipeline's plain `tokenizer` does not understand multimodal content
    items: it returns ~2 rather than raising, which silently misreports the
    token-usage metric. The processor does understand them, so try it first and
    reject an implausibly small answer.
    """
    for attr in ("processor", "tokenizer"):
        owner = getattr(pipe, attr, None)
        if owner is None or not hasattr(owner, "apply_chat_template"):
            continue
        try:
            encoded = owner.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
            count = _encoded_length(encoded)
        except Exception:
            continue
        if count > 8:
            return count

    # Last resort: text only. Undercounts by the image tokens, but is at least
    # the right order of magnitude.
    text = " ".join(
        item.get("text", "")
        for message in messages
        for item in message.get("content", [])
        if isinstance(item, dict)
    )
    return _count_output_tokens(getattr(pipe, "tokenizer", None), text)




def _sanity_check(pipe: Any) -> str:
    """Generate once against a synthetic image; return "" if the model answers.

    A misconfigured dtype does not raise. The model emits EOS immediately and
    every request returns HTTP 200 with an empty `text`, so a whole evaluation
    can complete before anyone notices — which is exactly what happened on the
    first float16 run. Checking here costs one short generation and turns a
    25-minute silent failure into an immediate one.
    """
    from PIL import Image

    handle = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    try:
        handle.close()
        Image.new("L", (64, 64), color=128).save(handle.name)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": handle.name},
                    {"type": "text", "text": "Reply with the single word: ready."},
                ],
            }
        ]
        result = pipe(text=messages, max_new_tokens=8)
        generated = result[0]["generated_text"]
        text = generated[-1].get("content", "") if isinstance(generated, list) else str(generated)
    except Exception as exc:  # a broken pipeline must not take the server down
        return f"sanity generation raised {exc.__class__.__name__}: {exc}"
    finally:
        with suppress(OSError):
            os.unlink(handle.name)

    if not str(text).strip():
        return (
            f"model returned empty text at precision={PIPELINE_PRECISION!r}. "
            "float16 overflows this model's activations; use bf16 or fp32."
        )
    return ""


_MIME_SUFFIXES: dict[str, str] = {"image/png": ".png", "image/jpeg": ".jpg"}


@contextmanager
def _image_location(req: GenerateRequest) -> Generator[str]:
    """Yield something the pipeline can open, cleaning up any temp file after.

    Inline bytes go to a temp file rather than a PIL object so the message
    construction stays identical for both transports and does not depend on
    version-specific `transformers` content-item handling.
    """
    if not req.image_base64:
        assert req.image_url_or_path is not None  # guaranteed by the validator
        yield req.image_url_or_path
        return

    suffix = _MIME_SUFFIXES.get(req.image_mime, ".png")
    handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        handle.write(base64.b64decode(req.image_base64, validate=True))
        handle.close()
        yield handle.name
    finally:
        with suppress(OSError):
            os.unlink(handle.name)


def create_app() -> FastAPI:
    app = FastAPI(title="MedGemma inference", version=SERVER_VERSION)

    @app.get("/ping")
    def ping() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {
            "status": "ok",
            "pipeline_loaded": str(PIPELINE is not None),
            "server_version": SERVER_VERSION,
            # Reported so a run's numbers can be attributed to a precision.
            "precision": PIPELINE_PRECISION or _resolve_precision(),
        }

    @app.post("/warmup")
    def warmup() -> dict[str, str]:
        """Load the checkpoint on demand so the first real request isn't the one that waits."""
        pipe = _ensure_pipeline(
            os.environ.get("MEDGEMMA_MODEL", DEFAULT_MODEL),
            _resolve_precision(),
        )
        problem = _sanity_check(pipe)
        return {
            "status": "degraded" if problem else "ok",
            "pipeline_loaded": "True",
            "precision": PIPELINE_PRECISION,
            "sanity_ok": str(not problem),
            "detail": problem,
        }

    @app.post("/generate", response_model=GenerateResponse)
    def generate(req: GenerateRequest) -> GenerateResponse:
        pipe = _ensure_pipeline(
            os.environ.get("MEDGEMMA_MODEL", DEFAULT_MODEL),
            _resolve_precision(),
        )

        with _image_location(req) as image_location:
            messages: list[dict[str, Any]] = []
            if req.system:
                messages.append(
                    {"role": "system", "content": [{"type": "text", "text": req.system}]}
                )
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": image_location},
                        {"type": "text", "text": req.prompt},
                    ],
                }
            )

            # Never log the payload itself — only its size and type.
            logger.info(
                "generate: source=%s mime=%s prompt_chars=%d system_chars=%d",
                "inline-bytes" if req.image_base64 else req.image_url_or_path,
                req.image_mime,
                len(req.prompt),
                len(req.system),
            )

            result = pipe(text=messages, max_new_tokens=req.max_new_tokens)

        generated = result[0]["generated_text"]
        if isinstance(generated, list) and generated and isinstance(generated[-1], dict):
            assistant = generated[-1].get("content", "")
            if not isinstance(assistant, str):
                assistant = str(assistant)
        elif isinstance(generated, str):
            assistant = generated
        else:
            assistant = str(generated)

        prompt_tokens = _count_tokens(pipe, messages)
        output_tokens = _count_output_tokens(pipe.tokenizer, assistant)  # type: ignore[attr-defined]

        return GenerateResponse(
            text=assistant,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            model=os.environ.get("MEDGEMMA_MODEL", DEFAULT_MODEL),
        )

    return app


def _run_uvicorn(host: str, port: int) -> None:
    uvicorn.run(create_app(), host=host, port=port, log_level="info")


def start_in_background(host: str = "0.0.0.0", port: int = 8000, *, warm: bool = True) -> str:
    """Start the server in a daemon thread and open a ngrok tunnel.

    Returns the public HTTPS URL. The caller is expected to keep the
    Python process alive (i.e. the notebook cell must not return
    immediately on Kaggle; the standard pattern is to use this from a
    background thread and then `time.sleep(...)` the main cell, or to
    use `ngrok.connect(...)` and let pyngrok manage the lifecycle).
    """
    from pyngrok import ngrok

    thread = threading.Thread(target=_run_uvicorn, args=(host, port), daemon=True)
    thread.start()

    # Load the checkpoint before announcing the URL. Loading it lazily inside
    # the first /generate meant the first student submission spent minutes
    # waiting and exhausted the client's retry budget before the model existed.
    if warm:
        logger.info("Warming the pipeline before opening the tunnel...")
        _ensure_pipeline(
            os.environ.get("MEDGEMMA_MODEL", DEFAULT_MODEL),
            _resolve_precision(),
        )
        logger.info("Pipeline ready.")

    tunnel = ngrok.connect(port, "http")
    public_url = tunnel.public_url
    if public_url.startswith("http://"):
        public_url = "https://" + public_url[len("http://"):]
    logger.info("ngrok tunnel: %s -> http://%s:%d", public_url, host, port)
    return public_url


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
