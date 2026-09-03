"""Browser-console smoke check (issue #52).

Loads the Welcome page and one scenario page in a real Chromium tab and fails
if the app logs an unexpected console warning or error — most notably the
repeated "empty sidebar theme colour" warning that ``.streamlit/config.toml``
now works around with an explicit ``[theme.sidebar]`` table.

Requires Playwright with its Chromium browser installed
(``pip install -r requirements-dev.txt && playwright install chromium``); it
skips itself otherwise; see ``tests/conftest.py``.
"""

from __future__ import annotations

import re

import pytest

pytestmark = pytest.mark.browser

# Messages we know come from outside this app's own code and choose to accept
# rather than fail on. Empty by design: nothing is silently allowlisted. If a
# real upstream (Streamlit/browser/provider-SDK) message needs to be added
# here, explain why it's out of this app's control right next to the entry.
KNOWN_UPSTREAM_CONSOLE_MESSAGES: tuple[str, ...] = ()

_RELEVANT_TYPES = {"warning", "error"}


def _capture_console_problems(page) -> list[str]:
    problems: list[str] = []

    def on_console(msg) -> None:
        if msg.type not in _RELEVANT_TYPES:
            return
        text = msg.text
        if any(allowed in text for allowed in KNOWN_UPSTREAM_CONSOLE_MESSAGES):
            return
        problems.append(f"console.{msg.type}: {text}")

    def on_pageerror(exc) -> None:
        problems.append(f"pageerror: {exc}")

    page.on("console", on_console)
    page.on("pageerror", on_pageerror)
    return problems


def test_welcome_page_has_no_unexpected_console_messages(streamlit_server, page) -> None:
    problems = _capture_console_problems(page)

    page.goto(streamlit_server, wait_until="networkidle")
    page.wait_for_selector('[data-testid="stSidebar"]')
    page.wait_for_timeout(1000)

    # The reported warning recurred on rerun, not just on first load, so
    # trigger one the way a user filling in Setup would.
    page.locator('[data-testid="stSidebar"]').get_by_text("ICS", exact=True).click()
    page.wait_for_timeout(1000)

    assert problems == [], "\n".join(problems)


def test_scenario_page_has_no_unexpected_console_messages(streamlit_server, page) -> None:
    problems = _capture_console_problems(page)

    page.goto(streamlit_server, wait_until="networkidle")
    page.get_by_role("link", name=re.compile("Threat Group Scenarios")).click()
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(1000)

    assert problems == [], "\n".join(problems)
