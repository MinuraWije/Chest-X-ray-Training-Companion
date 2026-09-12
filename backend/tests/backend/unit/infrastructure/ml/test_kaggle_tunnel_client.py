from __future__ import annotations

import json

import httpx
import pytest

from cxt.domain.ports.medgemma_client import ModelOutput
from cxt.domain.value_objects.cxr_image_ref import CxrImageRef
from cxt.infrastructure.config.settings import Settings
from cxt.infrastructure.ml.kaggle_tunnel_client import KaggleTunnelMedGemmaClient
from cxt.shared.errors import UpstreamUnavailableError


def _settings(tunnel_url: str = "https://example.ngrok.app") -> Settings:
    return Settings(medgemma_tunnel_url=tunnel_url)


class _StubTransport(httpx.BaseTransport):
    def __init__(self, handler) -> None:
        self._handler = handler
        self.calls: list[httpx.Request] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        return self._handler(request)


def _ok_response() -> httpx.Response:
    body = json.dumps({"text": "hello", "prompt_tokens": 10, "output_tokens": 5, "model": "m"})
    return httpx.Response(200, content=body.encode("utf-8"))


def test_generate_returns_model_output() -> None:
    transport = _StubTransport(lambda req: _ok_response())
    client = httpx.Client(transport=transport)
    sut = KaggleTunnelMedGemmaClient(_settings(), client=client)

    out: ModelOutput = sut.generate(
        image=CxrImageRef(uri="https://img/1.png"),
        prompt="p",
        system="s",
    )
    assert out.text == "hello"
    assert out.prompt_tokens == 10
    assert out.output_tokens == 5
    assert out.model == "m"
    assert transport.calls[0].method == "POST"
    assert transport.calls[0].url.path == "/generate"


def test_generate_serialises_body() -> None:
    transport = _StubTransport(lambda req: _ok_response())
    client = httpx.Client(transport=transport)
    sut = KaggleTunnelMedGemmaClient(_settings(), client=client)

    sut.generate(image=CxrImageRef(uri="/img.png"), prompt="hi", system="sys")
    sent = transport.calls[0]
    payload = json.loads(sent.content.decode("utf-8"))
    assert payload["image_url_or_path"] == "/img.png"
    assert payload["prompt"] == "hi"
    assert payload["system"] == "sys"
    assert "max_new_tokens" in payload


def test_inline_image_sends_base64_not_a_path() -> None:
    transport = _StubTransport(lambda req: _ok_response())
    client = httpx.Client(transport=transport)
    sut = KaggleTunnelMedGemmaClient(_settings(), client=client)

    sut.generate(
        image=CxrImageRef(uri="/images/x.png", data_b64="QUJD", mime="image/png"),
        prompt="hi",
        system="sys",
    )
    payload = json.loads(transport.calls[0].content.decode("utf-8"))
    assert payload["image_base64"] == "QUJD"
    assert payload["image_mime"] == "image/png"
    # The notebook cannot resolve our manifest paths, so we must not send one.
    assert "image_url_or_path" not in payload


def test_ngrok_interstitial_header_is_sent() -> None:
    transport = _StubTransport(lambda req: _ok_response())
    client = httpx.Client(transport=transport)
    sut = KaggleTunnelMedGemmaClient(_settings(), client=client)

    sut.generate(image=CxrImageRef(uri="/img.png"), prompt="p")
    assert transport.calls[0].headers["ngrok-skip-browser-warning"] == "true"


def test_retry_count_follows_settings() -> None:
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(500, text="server boom")

    settings = Settings(medgemma_tunnel_url="https://example.ngrok.app", medgemma_max_retries=2)
    client = httpx.Client(transport=_StubTransport(handler))
    sut = KaggleTunnelMedGemmaClient(settings, client=client)
    with pytest.raises(UpstreamUnavailableError):
        sut.generate(image=CxrImageRef(uri="x"), prompt="p")
    assert len(attempts) == 2


def test_no_tunnel_url_raises_upstream() -> None:
    client = httpx.Client(transport=_StubTransport(lambda r: _ok_response()))
    sut = KaggleTunnelMedGemmaClient(_settings(tunnel_url=""), client=client)
    with pytest.raises(UpstreamUnavailableError):
        sut.generate(image=CxrImageRef(uri="x"), prompt="p")


def test_5xx_retries_then_raises() -> None:
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(500, text="server boom")

    client = httpx.Client(transport=_StubTransport(handler))
    sut = KaggleTunnelMedGemmaClient(_settings(), client=client)
    with pytest.raises(UpstreamUnavailableError):
        sut.generate(image=CxrImageRef(uri="x"), prompt="p")
    assert len(attempts) >= 3


def test_4xx_raises_immediately() -> None:
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(422, text="bad")

    client = httpx.Client(transport=_StubTransport(handler))
    sut = KaggleTunnelMedGemmaClient(_settings(), client=client)
    with pytest.raises(UpstreamUnavailableError):
        sut.generate(image=CxrImageRef(uri="x"), prompt="p")
    assert len(attempts) == 1


def test_ping_uses_get() -> None:
    transport = _StubTransport(lambda r: httpx.Response(200, text='{"status":"ok"}'))
    client = httpx.Client(transport=transport)
    sut = KaggleTunnelMedGemmaClient(_settings(), client=client)
    assert sut.ping() is True
    assert transport.calls[0].method == "GET"
    assert transport.calls[0].url.path == "/ping"


def test_ping_false_on_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("nope", request=request)

    client = httpx.Client(transport=_StubTransport(handler))
    sut = KaggleTunnelMedGemmaClient(_settings(), client=client)
    assert sut.ping() is False


class TestWarmUp:
    """A reachable notebook is not a working one.

    A model loaded at a dtype it cannot generate in answers every request with
    HTTP 200 and an empty body. `warm_up` has to distinguish the two, or a whole
    evaluation completes at 0% having burned the GPU quota.
    """

    @staticmethod
    def _client(body: object, status: int = 200) -> tuple[KaggleTunnelMedGemmaClient, _StubTransport]:
        content = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        transport = _StubTransport(lambda req: httpx.Response(status, content=content))
        sut = KaggleTunnelMedGemmaClient(_settings(), client=httpx.Client(transport=transport))
        return sut, transport

    def test_healthy_warmup_succeeds(self) -> None:
        sut, transport = self._client({"status": "ok", "sanity_ok": "True", "precision": "bf16"})
        assert sut.warm_up() is True
        assert transport.calls[0].url.path == "/warmup"

    def test_failed_sanity_check_is_not_a_successful_warmup(self) -> None:
        sut, _ = self._client(
            {"status": "degraded", "sanity_ok": "False", "detail": "model returned empty text"}
        )
        assert sut.warm_up() is False

    def test_older_server_without_a_sanity_report_still_warms(self) -> None:
        sut, _ = self._client({"status": "ok", "pipeline_loaded": "True"})
        assert sut.warm_up() is True

    def test_non_json_body_is_treated_as_an_older_server(self) -> None:
        sut, _ = self._client(b"ok")
        assert sut.warm_up() is True

    def test_non_200_fails(self) -> None:
        sut, _ = self._client({"status": "ok", "sanity_ok": "True"}, status=503)
        assert sut.warm_up() is False

    def test_transport_error_fails(self) -> None:
        def boom(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("no route", request=request)

        sut = KaggleTunnelMedGemmaClient(
            _settings(), client=httpx.Client(transport=_StubTransport(boom))
        )
        assert sut.warm_up() is False

    def test_unset_tunnel_url_fails(self) -> None:
        sut = KaggleTunnelMedGemmaClient(_settings(tunnel_url=""))
        assert sut.warm_up() is False
