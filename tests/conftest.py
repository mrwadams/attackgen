"""Shared pytest fixtures for the AttackGen test suite."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import streamlit as st


@pytest.fixture
def fake_session_state(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace `st.session_state` with a plain dict for the duration of a test.

    Streamlit's real SessionState requires a running ScriptRunContext. A dict
    is interface-compatible for the .get() / [] access patterns used by
    `LLMConfig.from_session_state` and the LangSmith run_id stash.
    """
    state: dict[str, Any] = {}
    monkeypatch.setattr(st, "session_state", state)
    return state


@pytest.fixture
def mock_litellm_completion(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Patch `litellm.completion` to capture kwargs and return a stub response.

    Returns a SimpleNamespace with:
      - calls: list of (args, kwargs) tuples for every invocation
      - set_content(s): change the stub response content for the next call
    """
    captured = SimpleNamespace(calls=[], content="stub response")

    def _fake_completion(*args, **kwargs):
        captured.calls.append((args, kwargs))
        message = SimpleNamespace(content=captured.content)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])

    # Patch on the litellm module *and* on core.llm (which imported the symbol
    # via `import litellm` — same module object, so one patch suffices).
    import litellm

    monkeypatch.setattr(litellm, "completion", _fake_completion)
    return captured
