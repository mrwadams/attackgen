"""Shared pytest fixtures for the AttackGen test suite."""

from __future__ import annotations

import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parent.parent
_APP_ENTRY = "00_👋_Welcome.py"


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
        if kwargs.get("stream"):
            def _chunks():
                delta = SimpleNamespace(content=captured.content)
                yield SimpleNamespace(choices=[SimpleNamespace(delta=delta)])

            return _chunks()
        message = SimpleNamespace(content=captured.content)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])

    # Patch on the litellm module *and* on core.llm (which imported the symbol
    # via `import litellm` — same module object, so one patch suffices).
    import litellm

    monkeypatch.setattr(litellm, "completion", _fake_completion)
    return captured


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_until_serving(base_url: str, proc: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            output = proc.stdout.read().decode(errors="replace") if proc.stdout else ""
            pytest.skip(f"streamlit server exited before it started serving:\n{output}")
        try:
            urllib.request.urlopen(base_url, timeout=1)
            return
        except OSError:
            time.sleep(0.5)
    proc.terminate()
    pytest.skip(f"streamlit server did not start within {timeout}s")


@pytest.fixture(scope="session")
def streamlit_server():
    """Serve the real app headlessly for browser-based smoke tests.

    Session-scoped so every ``browser``-marked test shares one server instead
    of paying Streamlit's multi-second startup cost per test. Skips (rather
    than fails) when the server can't be started at all, so an environment
    without a working ``streamlit run`` doesn't break the rest of the suite.
    """
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            _APP_ENTRY,
            "--server.headless=true",
            f"--server.port={port}",
            "--server.address=127.0.0.1",
            "--browser.gatherUsageStats=false",
        ],
        cwd=_REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        _wait_until_serving(base_url, proc, timeout=30.0)
        yield base_url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture(scope="session")
def browser():
    """A shared Chromium instance for browser-marked tests.

    Skips when Playwright isn't installed, or when it's installed but its
    Chromium binary hasn't been downloaded (``playwright install chromium``)
    — both are expected in CI, which doesn't run that install step.
    """
    playwright_sync = pytest.importorskip("playwright.sync_api")
    with playwright_sync.sync_playwright() as p:
        try:
            instance = p.chromium.launch()
        except Exception as exc:  # Executable-not-found error type varies by platform.
            pytest.skip(
                "Chromium is not installed for Playwright "
                f"(run `playwright install chromium`): {exc}"
            )
        yield instance
        instance.close()


@pytest.fixture
def page(browser):
    """A fresh Playwright page/tab, closed after each test."""
    p = browser.new_page()
    yield p
    p.close()
