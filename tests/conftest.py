"""Pytest configuration: load .env before any test collection."""

from __future__ import annotations

from dotenv import load_dotenv

# Load .env once at the start of the test session so API keys are visible
# to pytest.mark.skipif conditions and all test modules.
load_dotenv()
