"""Page-level tests for the Custom Scenarios page, driven by Streamlit's AppTest.

The page loads the real bundles, so the first run in a session pays for the
Enterprise load.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from core.routes import CUSTOM_PAGE
from core.state import PERSISTED

_PAGE = str(Path(__file__).resolve().parent.parent / CUSTOM_PAGE.path)


def _page(matrix: str) -> AppTest:
    at = AppTest.from_file(_PAGE, default_timeout=120)
    at.query_params[PERSISTED["matrix"]] = matrix
    at.run()
    return at


def _pick_template(at: AppTest, name: str) -> None:
    next(s for s in at.selectbox if s.label == "Select a template").select(name).run()


def _switch_matrix(at: AppTest, matrix: str) -> None:
    next(r for r in at.radio if matrix in r.options).set_value(matrix).run()


def _picked_techniques(at: AppTest) -> list[str]:
    return list(next(m for m in at.multiselect if m.label.endswith("Techniques")).value)


@pytest.mark.parametrize(
    ("start", "template", "switch_to"),
    [
        ("Enterprise", "Phishing Attack", "ICS"),
        ("ICS", "Denial of Service", "ATLAS"),
        ("ATLAS", "LLM Jailbreak", "Enterprise"),
    ],
)
def test_switching_matrix_after_a_template_drops_its_techniques(start, template, switch_to):
    at = _page(start)
    _pick_template(at, template)
    assert _picked_techniques(at)

    _switch_matrix(at, switch_to)

    assert not at.exception
    assert _picked_techniques(at) == []
