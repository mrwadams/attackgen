"""The shared page heading and back link (issue #136).

Each helper is pinned to the exact HTML the pages rendered inline before it,
so moving the markup into `core.styles` changes nothing on screen. The page
checks keep that markup out of the pages, so a style change stays a one-line
edit in `core/styles.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import streamlit as st

from core.routes import ASSISTANT_PAGE, SCENARIO_PAGES
from core.styles import back_link, page_heading

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def rendered(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, Any]]]:
    calls: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(st, "markdown", lambda body, **kwargs: calls.append((body, kwargs)))
    return calls


def test_page_heading_renders_the_green_title(rendered):
    page_heading("AttackGen Assistant💬")

    assert rendered == [
        (
            "# <span style='color: #1DB954;'>AttackGen Assistant💬</span>",
            {"unsafe_allow_html": True},
        )
    ]


def test_back_link_renders_a_link_to_the_welcome_page(rendered):
    back_link()

    assert rendered == [
        (
            '<a href="/" style="display: inline-block; padding: 5px 20px; color: white; '
            'text-align: center; text-decoration: none; font-size: 16px; border-radius: 4px;">'
            "⬅️ Back</a>",
            {"unsafe_allow_html": True},
        )
    ]


@pytest.mark.parametrize("page", [*SCENARIO_PAGES, ASSISTANT_PAGE], ids=lambda p: p.page_id)
def test_pages_leave_heading_and_back_link_html_to_the_helpers(page):
    source = (REPO_ROOT / page.path).read_text(encoding="utf-8")

    assert "#1DB954" not in source
    assert '<a href="/"' not in source
    assert "page_heading(" in source


@pytest.mark.parametrize("page", SCENARIO_PAGES, ids=lambda p: p.page_id)
def test_scenario_pages_render_the_back_link(page):
    assert "back_link()" in (REPO_ROOT / page.path).read_text(encoding="utf-8")
