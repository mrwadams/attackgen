"""Shared pytest fixtures for the AttackGen test suite."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NoReturn

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


def _unavailable(reason: str) -> NoReturn:
    """Skip a browser test, or fail it when the environment promised a browser.

    Every browser-marked test skips itself when Playwright, its Chromium
    binary, or a working ``streamlit run`` is missing, so a plain ``pytest``
    stays green on a machine that has none of them. That silence is wrong in
    the one place the checks are meant to run: a CI job dedicated to them
    would report success having tested nothing, which is exactly how a broken
    fix and four broken tests reached review in #94. Setting
    ``ATTACKGEN_REQUIRE_BROWSER=1`` turns every one of those skips into a
    failure.
    """
    if os.environ.get("ATTACKGEN_REQUIRE_BROWSER") == "1":
        pytest.fail(f"ATTACKGEN_REQUIRE_BROWSER=1 but {reason}")
    pytest.skip(reason)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_until_serving(base_url: str, proc: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            output = proc.stdout.read().decode(errors="replace") if proc.stdout else ""
            _unavailable(
                f"streamlit server exited before it started serving:\n{output}"
            )
        try:
            urllib.request.urlopen(base_url, timeout=1)
            return
        except OSError:
            time.sleep(0.5)
    proc.terminate()
    _unavailable(f"streamlit server did not start within {timeout}s")


@pytest.fixture(scope="session")
def streamlit_server():
    """Serve the real app headlessly for browser-based smoke tests.

    Session-scoped so every ``browser``-marked test shares one server instead
    of paying Streamlit's multi-second startup cost per test. Skips (rather
    than fails) when the server can't be started at all, so an environment
    without a working ``streamlit run`` doesn't break the rest of the suite —
    unless ``ATTACKGEN_REQUIRE_BROWSER=1`` says a server was expected, in which
    case it fails; see ``_unavailable``.
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


@pytest.fixture
def browser():
    """A Chromium instance for one browser-marked test.

    Skips when Playwright isn't installed, or when it's installed but its
    Chromium binary hasn't been downloaded (``playwright install chromium``),
    so a plain ``pytest`` stays green without either. The ``browser`` job in
    `.github/workflows/tests.yml` installs both and sets
    ``ATTACKGEN_REQUIRE_BROWSER=1``, which turns those skips into failures —
    see ``_unavailable``.

    Deliberately *not* session-scoped, even though launching Chromium per test
    costs a fraction of a second. Playwright's sync API drives an asyncio loop
    that stays running in the main thread for as long as ``sync_playwright()``
    is open, and a session-scoped fixture keeps it open for the rest of the
    run — which makes every later ``asyncio.run(...)`` test in the suite fail
    with "asyncio.run() cannot be called from a running event loop"
    (``tests/test_mcp_server.py`` and ``tests/test_skills.py`` both do this).
    Entering and leaving the context per test keeps that loop contained.
    """
    try:
        import playwright.sync_api as playwright_sync
    except ImportError as exc:
        # `_unavailable` is annotated NoReturn, but keep the import's only use
        # out of the except branch so nothing reads a name the failed import
        # never bound.
        _unavailable(f"Playwright is not installed: {exc}")

    with playwright_sync.sync_playwright() as p:
        try:
            instance = p.chromium.launch()
        except Exception as exc:  # Executable-not-found error type varies by platform.
            # `else:` rather than falling through: `_unavailable` raises, so the
            # yield below is already unreachable on this path, but CodeQL does
            # not model that and flags `instance` as possibly unbound
            # (py/uninitialized-local-variable, error severity).
            _unavailable(
                "Chromium is not installed for Playwright "
                f"(run `playwright install chromium`): {exc}"
            )
        else:
            yield instance
            instance.close()


@pytest.fixture
def page(browser):
    """A fresh Playwright page/tab, closed after each test."""
    p = browser.new_page()
    yield p
    p.close()
