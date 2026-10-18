"""Shared pytest fixtures for the backend test suite."""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
_SRC = _BACKEND_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Tests must not read the developer's backend/.env. Every `Settings(...)` built
# in a test declares the values it cares about and expects the code defaults for
# the rest; inheriting a local .env makes the suite pass or fail according to
# whichever tunnel URL happens to be configured on the machine that runs it.
from cxt.infrastructure.config.settings import Settings  # noqa: E402

Settings.model_config["env_file"] = None
