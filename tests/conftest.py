"""Shared test setup.

Every test runs against a throwaway SQLite file and with the LLM disabled,
so tests are fast, deterministic and never need an API key.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    monkeypatch.setenv("NDN_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.delenv("VULTR_API_KEY", raising=False)
    monkeypatch.setenv("NDN_DISABLE_LLM", "1")
    yield
