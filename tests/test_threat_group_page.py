"""Page-level tests for the Threat Group Scenarios page, driven by Streamlit's AppTest.

The page loads the real bundles and probes every group, so the first run in a
session is slow.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from core.routes import THREAT_GROUP_PAGE
from core.state import PERSISTED

_PAGE = str(Path(__file__).resolve().parent.parent / THREAT_GROUP_PAGE.path)


@pytest.mark.parametrize(
    ("matrix", "heading", "entity", "placeholder", "caption"),
    [
        (
            "Enterprise", "### Select a Threat Actor Group", "threat actor group", "Select Group",
            "Built from a random selection of the group's techniques, so each run differs.",
        ),
        (
            "ICS", "### Select a Threat Actor Group", "threat actor group", "Select Group",
            "Built from a random selection of the group's techniques, so each run differs.",
        ),
        (
            "ATLAS", "### Select a Case Study", "case study", "Select Case Study",
            "Built from the case study's documented attack procedure.",
        ),
    ],
)
def test_labels_name_the_matrix_entity(matrix, heading, entity, placeholder, caption):
    at = AppTest.from_file(_PAGE, default_timeout=180)
    at.query_params[PERSISTED["matrix"]] = matrix
    at.run()

    assert not at.exception
    assert heading in [m.value for m in at.markdown]
    picker = next(s for s in at.selectbox if s.label == f"Select a {entity} for the scenario")
    assert picker.placeholder == placeholder
    assert caption in [c.value for c in at.caption]
