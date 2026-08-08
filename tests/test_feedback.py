"""Tests for the LangSmith feedback widget and submission helper."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import langsmith
import pytest
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from core.feedback import _submit, render_feedback_widget


class _FakePlaceholder:
    """Captures the success/warning/error message a placeholder is told to render."""

    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def success(self, msg: str) -> None:
        self.messages.append(("success", msg))

    def warning(self, msg: str) -> None:
        self.messages.append(("warning", msg))

    def error(self, msg: str) -> None:
        self.messages.append(("error", msg))


class _MissingSecrets:
    """Reproduce Streamlit's behaviour when no secrets.toml can be found."""

    def __getitem__(self, _name: str) -> Any:
        raise StreamlitSecretNotFoundError("No secrets found")


class _FakeClient:
    def __init__(self, record_id: str = "feedback-123", raises: Exception | None = None) -> None:
        self.calls: list[tuple[Any, str, dict[str, Any]]] = []
        self._record_id = record_id
        self._raises = raises

    def create_feedback(self, run_id: Any, kind: str, **kwargs: Any) -> SimpleNamespace:
        self.calls.append((run_id, kind, kwargs))
        if self._raises is not None:
            raise self._raises
        return SimpleNamespace(id=self._record_id)


def test_widget_tolerates_missing_secrets_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client_calls: list[dict[str, Any]] = []
    info_messages: list[str] = []

    monkeypatch.setattr(st, "secrets", _MissingSecrets())
    monkeypatch.setattr(st, "info", info_messages.append)
    monkeypatch.setattr(st, "empty", object)
    monkeypatch.setattr(st, "markdown", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        langsmith, "Client", lambda **kwargs: client_calls.append(kwargs)
    )

    render_feedback_widget(key_prefix="test", scenario_generated=True)

    assert len(info_messages) == 1
    assert client_calls == []


def test_submit_warns_when_run_id_missing(fake_session_state: dict[str, Any]) -> None:
    client = _FakeClient()
    placeholder = _FakePlaceholder()

    _submit(client, placeholder, kind="positive", score=1)

    assert client.calls == []
    assert placeholder.messages == [("warning", "No run ID found. Please generate a scenario first.")]


def test_submit_posts_feedback_and_records_success(
    fake_session_state: dict[str, Any],
) -> None:
    fake_session_state["run_id"] = "run-abc"
    client = _FakeClient(record_id="fb-1")
    placeholder = _FakePlaceholder()

    _submit(client, placeholder, kind="positive", score=1)

    assert client.calls == [("run-abc", "positive", {"score": 1, "comment": ""})]
    assert fake_session_state["feedback"] == {"feedback_id": "fb-1", "score": 1}
    assert placeholder.messages == [("success", "Feedback submitted. Thank you.")]


def test_submit_records_negative_with_score_zero(
    fake_session_state: dict[str, Any],
) -> None:
    fake_session_state["run_id"] = "run-abc"
    client = _FakeClient(record_id="fb-2")
    placeholder = _FakePlaceholder()

    _submit(client, placeholder, kind="negative", score=0)

    assert client.calls == [("run-abc", "negative", {"score": 0, "comment": ""})]
    assert fake_session_state["feedback"] == {"feedback_id": "fb-2", "score": 0}


def test_submit_surfaces_client_errors(
    fake_session_state: dict[str, Any],
) -> None:
    fake_session_state["run_id"] = "run-abc"
    client = _FakeClient(raises=RuntimeError("network down"))
    placeholder = _FakePlaceholder()

    _submit(client, placeholder, kind="positive", score=1)

    assert "feedback" not in fake_session_state
    assert len(placeholder.messages) == 1
    level, msg = placeholder.messages[0]
    assert level == "error"
    assert "network down" in msg
