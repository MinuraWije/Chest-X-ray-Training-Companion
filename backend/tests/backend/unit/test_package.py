"""Smoke test: the package imports and exposes the version."""
from __future__ import annotations


def test_package_imports() -> None:
    import cxt

    assert cxt.__version__ == "0.1.0"


def test_domain_subpackage_imports() -> None:
    from cxt.domain import entities, ports, value_objects

    assert entities is not None
    assert ports is not None
    assert value_objects is not None
