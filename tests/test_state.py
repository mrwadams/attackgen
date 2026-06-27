"""Tests for `core.state` — URL query-param persistence of sidebar selections.

These assert the validation rules (stale provider/matrix values are dropped),
the precedence rule (an existing session_state value wins over the URL), and the
security invariant that API keys are never mirrored into the URL.
"""

from __future__ import annotations

import pytest
import streamlit as st

from core.state import restore_from_query_params, sync_to_query_params


@pytest.fixture
def fake_query_params(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Replace `st.query_params` with a plain dict.

    A dict is interface-compatible with the access patterns `core.state` uses:
    `in`, `[]`, `.items()`, `.keys()`, `del`, and item assignment.
    """
    qp: dict[str, str] = {}
    monkeypatch.setattr(st, "query_params", qp)
    return qp


# ---------------------------------------------------------------------------
# restore_from_query_params
# ---------------------------------------------------------------------------


def test_restore_loads_valid_values(fake_session_state, fake_query_params) -> None:
    fake_query_params.update({"p": "Anthropic API", "x": "ICS", "i": "Finance"})

    restore_from_query_params()

    assert fake_session_state["chosen_model_provider"] == "Anthropic API"
    assert fake_session_state["matrix"] == "ICS"
    assert fake_session_state["industry"] == "Finance"


def test_restore_drops_unknown_provider(fake_session_state, fake_query_params) -> None:
    fake_query_params["p"] = "BogusProvider"

    restore_from_query_params()

    assert "chosen_model_provider" not in fake_session_state


def test_restore_drops_invalid_matrix(fake_session_state, fake_query_params) -> None:
    fake_query_params["x"] = "Bogus"

    restore_from_query_params()

    assert "matrix" not in fake_session_state


def test_restore_does_not_overwrite_existing_session_value(
    fake_session_state, fake_query_params
) -> None:
    fake_session_state["industry"] = "Healthcare"
    fake_query_params["i"] = "Technology"

    restore_from_query_params()

    # A user-driven selection already in session_state wins over the URL.
    assert fake_session_state["industry"] == "Healthcare"


# ---------------------------------------------------------------------------
# sync_to_query_params
# ---------------------------------------------------------------------------


def test_sync_writes_only_the_persisted_shadow_keys(
    fake_session_state, fake_query_params
) -> None:
    fake_session_state.update(
        {
            "chosen_model_provider": "OpenAI API",
            "llm_model_name": "gpt-5.5",
            "llm_api_base": "https://example.invalid/v1",
            "matrix": "Enterprise",
            "industry": "Finance",
            "company_size": "Large",
            # Noise that must not be mirrored:
            "run_id": "abc",
            "feedback": {"score": 1},
        }
    )

    sync_to_query_params()

    assert set(fake_query_params) == {"p", "m", "b", "x", "i", "s"}
    assert fake_query_params["p"] == "OpenAI API"
    assert fake_query_params["m"] == "gpt-5.5"


def test_sync_never_writes_api_key_to_url(fake_session_state, fake_query_params) -> None:
    """Security invariant: secrets must never reach the URL/history/logs."""
    fake_session_state.update(
        {
            "chosen_model_provider": "OpenAI API",
            "llm_model_name": "gpt-5.5",
            "llm_api_key": "sk-super-secret",
        }
    )

    sync_to_query_params()

    assert "sk-super-secret" not in fake_query_params.values()
    assert "llm_api_key" not in fake_query_params


def test_sync_skips_falsy_values(fake_session_state, fake_query_params) -> None:
    fake_session_state.update(
        {"chosen_model_provider": "OpenAI API", "industry": "", "company_size": None}
    )

    sync_to_query_params()

    assert "p" in fake_query_params
    assert "i" not in fake_query_params
    assert "s" not in fake_query_params
