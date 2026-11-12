"""Tests for `core.matrices`, the one definition of the supported MITRE matrices."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from core.matrices import MATRICES, MATRIX_NAMES
from core.sidebar import get_setup_state
from core.state import PERSISTED, restore_from_query_params

_PAGES = Path(__file__).resolve().parent.parent / "pages"


def test_supported_matrices_in_sidebar_order() -> None:
    assert MATRIX_NAMES == ("Enterprise", "ICS", "ATLAS")


@pytest.mark.parametrize(
    ("name", "framework", "label"),
    [
        ("Enterprise", "ATT&CK", "Enterprise ATT&CK"),
        ("ICS", "ATT&CK", "ICS ATT&CK"),
        ("ATLAS", "ATLAS", "MITRE ATLAS"),
    ],
)
def test_each_matrix_names_its_framework_and_label(name, framework, label) -> None:
    matrix = MATRICES[name]
    assert (matrix.name, matrix.framework, matrix.label) == (name, framework, label)


@pytest.mark.parametrize(
    ("name", "entity", "case_studies"),
    [
        ("Enterprise", "threat actor group", False),
        ("ICS", "threat actor group", False),
        ("ATLAS", "case study", True),
    ],
)
def test_only_atlas_builds_scenarios_from_case_studies(name, entity, case_studies) -> None:
    assert (MATRICES[name].entity, MATRICES[name].case_studies) == (entity, case_studies)


@pytest.mark.parametrize("name", MATRIX_NAMES)
def test_sidebar_accepts_every_defined_matrix(name) -> None:
    setup = get_setup_state({"matrix": name}, {})
    assert "Select a MITRE framework in the Setup sidebar." not in setup.blockers


@pytest.mark.parametrize("name", MATRIX_NAMES)
def test_url_restore_accepts_every_defined_matrix(
    name, fake_session_state, monkeypatch
) -> None:
    import streamlit as st

    monkeypatch.setattr(st, "query_params", {PERSISTED["matrix"]: name})
    restore_from_query_params()
    assert fake_session_state["matrix"] == name


def _matrix_literal_comparisons(path: Path) -> list[int]:
    lines = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Compare):
            operands = [node.left, *node.comparators]
            for operand in operands:
                values = operand.elts if isinstance(operand, (ast.Tuple, ast.List, ast.Set)) else [operand]
                if any(isinstance(v, ast.Constant) and v.value in MATRIX_NAMES for v in values):
                    lines.append(node.lineno)
    return lines


@pytest.mark.parametrize("page", sorted(_PAGES.glob("*.py")), ids=lambda p: p.name)
def test_no_page_compares_the_matrix_to_a_string_literal(page) -> None:
    assert _matrix_literal_comparisons(page) == []
