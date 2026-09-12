"""KaggleTunnelMedGemmaClient — HTTP client for the ngrok-tunneled notebook.

Calls `POST {tunnel_url}/generate` with the same JSON shape the
notebook server returns. Retries 5xx with exponential backoff.
4xx errors are surfaced immediately as `UpstreamUnavailableError`
because they are client errors that retrying will not fix.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx
from tenacity import (
    RetryError,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from cxt.domain.ports.medgemma_client import ModelOutput
from cxt.domain.value_objects.cxr_image_ref import CxrImageRef
from cxt.infrastructure.config.settings import Settings, get_settings
from cxt.shared.errors import UpstreamUnavailableError

logger = logging.getLogger(__name__)

_GENERATE_PATH = "/generate"
_PING_PATH = "/ping"
_WARMUP_PATH = "/warmup"

# ngrok's free tier serves a browser interstitial to anything that looks like a
# browser; without this header the response body is HTML, not JSON.
_TUNNEL_HEADERS = {"ngrok-skip-browser-warning": "true"}


class _TransientMedGemmaError(Exception):
    """Raised on 5xx / network errors so tenacity retries only those."""


class KaggleTunnelMedGemmaClient:
    """IMedGemmaClient implementation that calls the Kaggle ngrok tunnel."""

    def __init__(self, settings: Settings | None = None, *, client: httpx.Client | None = None) -> None:
        self._settings = settings or get_settings()
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=self._settings.medgemma_timeout_seconds,
            headers=_TUNNEL_HEADERS,
        )

    @property
    def tunnel_url(self) -> str:
        return self._settings.medgemma_tunnel_url.rstrip("/")

    def ping(self) -> bool:
        if not self.tunnel_url:
            return False
        try:
            response = self._client.get(
                f"{self.tunnel_url}{_PING_PATH}", timeout=5.0, headers=_TUNNEL_HEADERS
            )
        except httpx.HTTPError as exc:
            logger.warning("MedGemma tunnel ping failed: %s", exc)
            return False
        return response.status_code == 200

    def warm_up(self) -> bool:
        """Ask the notebook to load the pipeline before the first real request.

        The checkpoint takes minutes to load; without this the first student
        submission burns the whole retry budget waiting for it.

        Newer notebook servers also generate once and report `sanity_ok`. A
        model loaded at a dtype it cannot generate in answers every request
        with HTTP 200 and an empty body, so "reachable" is not "working" —
        treat a failed sanity check as a failed warm-up.
        """
        if not self.tunnel_url:
            return False
        try:
            response = self._client.post(
                f"{self.tunnel_url}{_WARMUP_PATH}",
                timeout=self._settings.medgemma_warmup_timeout_seconds,
                headers=_TUNNEL_HEADERS,
            )
        except httpx.HTTPError as exc:
            logger.warning("MedGemma warm-up failed: %s", exc)
            return False
        if response.status_code != 200:
            return False
        try:
            payload = response.json()
        except ValueError:
            return True  # older server: no sanity report, nothing to check
        if isinstance(payload, dict) and payload.get("sanity_ok") == "False":
            logger.error(
                "MedGemma warmed up but cannot generate: %s",
                payload.get("detail") or "no detail reported",
            )
            return False
        return True

    def generate(self, *, image: CxrImageRef, prompt: str, system: str = "") -> ModelOutput:
        if not self.tunnel_url:
            raise UpstreamUnavailableError(
                "MEDGEMMA_TUNNEL_URL is not configured. Set it in backend/.env, "
                "or use MEDGEMMA_RUNTIME=dry_run for local dev."
            )

        body: dict[str, Any] = {
            "prompt": prompt,
            "system": system,
            "max_new_tokens": self._settings.medgemma_max_new_tokens,
        }
        # The notebook shares no filesystem with us, so a manifest URL like
        # "/images/x.png" is meaningless there — send the bytes instead.
        if image.is_inline:
            body["image_base64"] = image.data_b64
            body["image_mime"] = image.mime
        else:
            body["image_url_or_path"] = image.uri

        try:
            payload = self._post_with_retry(body)
        except _TransientMedGemmaError as exc:
            raise UpstreamUnavailableError(
                f"MedGemma tunnel unreachable: {exc}"
            ) from exc
        except UpstreamUnavailableError:
            raise
        except (httpx.HTTPError, RetryError) as exc:
            raise UpstreamUnavailableError(
                f"MedGemma tunnel unreachable after {self._settings.medgemma_max_retries} attempts: {exc}"
            ) from exc

        text = payload.get("text")
        if not isinstance(text, str):
            raise UpstreamUnavailableError("MedGemma tunnel returned no 'text' field.")
        model = payload.get("model")
        return ModelOutput(
            text=text,
            prompt_tokens=int(payload.get("prompt_tokens", 0)),
            output_tokens=int(payload.get("output_tokens", 0)),
            model=model if isinstance(model, str) else "",
        )

    def _post_with_retry(self, body: dict[str, Any]) -> dict[str, Any]:
        retrying = Retrying(
            reraise=True,
            stop=stop_after_attempt(max(1, self._settings.medgemma_max_retries)),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type((httpx.HTTPError, _TransientMedGemmaError)),
        )
        return retrying(self._post_once, body)

    def _post_once(self, body: dict[str, Any]) -> dict[str, Any]:
        response = self._client.post(
            f"{self.tunnel_url}{_GENERATE_PATH}", json=body, headers=_TUNNEL_HEADERS
        )

        if response.status_code >= 500:
            raise _TransientMedGemmaError(
                f"MedGemma tunnel returned {response.status_code}: {response.text[:200]}"
            )
        if response.status_code >= 400:
            raise UpstreamUnavailableError(
                f"MedGemma tunnel returned {response.status_code}: {response.text[:200]}"
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise UpstreamUnavailableError(
                f"MedGemma tunnel returned non-JSON: {response.text[:200]}"
            ) from exc
        if not isinstance(data, dict):
            raise UpstreamUnavailableError("MedGemma tunnel response is not a JSON object.")
        return data

    def close(self) -> None:
        if self._owns_client:
            self._client.close()
