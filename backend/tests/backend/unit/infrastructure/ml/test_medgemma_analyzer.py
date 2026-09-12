"""Tests for MedGemmaAnalyzer — the IImageAnalyzer over a live model client."""
from __future__ import annotations

import pytest

from cxt.domain.value_objects.cxr_image_ref import CxrImageRef
from cxt.infrastructure.ml.medgemma_analyzer import MedGemmaAnalyzer
from cxt.shared.errors import UpstreamUnavailableError, ValidationError
from tests.support import FakeMedGemmaClient

_IMAGE = CxrImageRef(uri="/images/c-1.png", data_b64="QUJD")


def test_returns_the_clients_text() -> None:
    client = FakeMedGemmaClient('{"ok": true}')
    analysis = MedGemmaAnalyzer(client=client).analyze(
        image=_IMAGE, prompt="describe", system="be a radiologist"
    )

    assert analysis.raw_text == '{"ok": true}'
    assert analysis.model_output.model == "fake-medgemma"


def test_forwards_image_prompt_and_system_unchanged() -> None:
    client = FakeMedGemmaClient('{"ok": true}')
    MedGemmaAnalyzer(client=client).analyze(
        image=_IMAGE, prompt="describe", system="be a radiologist"
    )

    image, prompt, system = client.calls[0]
    assert image is _IMAGE
    assert prompt == "describe"
    assert system == "be a radiologist"


def test_empty_response_is_upstream_failure_not_a_parse_attempt() -> None:
    analyzer = MedGemmaAnalyzer(client=FakeMedGemmaClient("   "))
    with pytest.raises(UpstreamUnavailableError, match="empty"):
        analyzer.analyze(image=_IMAGE, prompt="p")


def test_domain_errors_from_the_client_propagate_unchanged() -> None:
    original = UpstreamUnavailableError("tunnel down")
    analyzer = MedGemmaAnalyzer(client=FakeMedGemmaClient(original))
    with pytest.raises(UpstreamUnavailableError, match="tunnel down"):
        analyzer.analyze(image=_IMAGE, prompt="p")

    analyzer = MedGemmaAnalyzer(client=FakeMedGemmaClient(ValidationError("bad input")))
    with pytest.raises(ValidationError, match="bad input"):
        analyzer.analyze(image=_IMAGE, prompt="p")


def test_unexpected_errors_are_wrapped_as_upstream() -> None:
    analyzer = MedGemmaAnalyzer(client=FakeMedGemmaClient(RuntimeError("socket exploded")))
    with pytest.raises(UpstreamUnavailableError, match="socket exploded"):
        analyzer.analyze(image=_IMAGE, prompt="p")
