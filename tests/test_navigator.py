"""Tests for `core.navigator` — ATT&CK Navigator layer export.

The layer JSON is what a downstream Navigator instance parses, so these tests
pin the fields that matter to it: the domain per matrix, the presence/absence
of the ATT&CK version, technique IDs, and tactic handling.
"""

from __future__ import annotations

import json

import pytest

from core.navigator import (
    ATLAS_NAVIGATOR_URL,
    ATTACK_NAVIGATOR_URL,
    ATTACK_VERSION,
    HIGHLIGHT_COLOR,
    build_layer,
    dumps,
    layer_filename,
    navigator_for_domain,
    parse_technique_id,
    tactic_shortname,
)


class TestNavigatorForDomain:
    def test_atlas_domain_points_to_atlas_navigator(self):
        name, url = navigator_for_domain("atlas-atlas")
        assert name == "ATLAS Navigator"
        assert url == ATLAS_NAVIGATOR_URL

    @pytest.mark.parametrize("domain", ["enterprise-attack", "ics-attack", ""])
    def test_attack_domains_point_to_attack_navigator(self, domain):
        name, url = navigator_for_domain(domain)
        assert name == "ATT&CK Navigator"
        assert url == ATTACK_NAVIGATOR_URL


class TestTacticShortname:
    @pytest.mark.parametrize(
        "phase, expected",
        [
            ("Initial Access", "initial-access"),
            ("Command and Control", "command-and-control"),
            ("AI Model Access", "ai-model-access"),
            ("Execution", "execution"),
        ],
    )
    def test_normalises_display_phase(self, phase, expected):
        assert tactic_shortname(phase) == expected

    def test_none_and_empty(self):
        assert tactic_shortname(None) is None
        assert tactic_shortname("") is None


class TestParseTechniqueId:
    @pytest.mark.parametrize(
        "display, expected",
        [
            ("Spearphishing Attachment (T1193)", "T1193"),
            ("LLM Prompt Injection (AML.T0051)", "AML.T0051"),
            ("Command and Scripting Interpreter (T1059)", "T1059"),
            ("Process Injection (T1055.001)", "T1055.001"),
        ],
    )
    def test_extracts_trailing_id(self, display, expected):
        assert parse_technique_id(display) == expected

    def test_no_id_returns_none(self):
        assert parse_technique_id("Just a name") is None
        assert parse_technique_id("") is None


class TestBuildLayer:
    def test_enterprise_domain_and_attack_version(self):
        layer = build_layer(
            name="Test",
            matrix="Enterprise",
            techniques=[("T1059", "execution")],
        )
        assert layer["domain"] == "enterprise-attack"
        assert layer["versions"]["attack"] == ATTACK_VERSION
        assert layer["versions"]["layer"]

    def test_ics_domain(self):
        layer = build_layer(
            name="Test", matrix="ICS", techniques=[("T0814", "impair-process-control")]
        )
        assert layer["domain"] == "ics-attack"
        assert layer["versions"]["attack"] == ATTACK_VERSION

    def test_atlas_domain_omits_attack_version(self):
        layer = build_layer(
            name="Test",
            matrix="ATLAS",
            techniques=[("AML.T0051", "initial-access")],
        )
        assert layer["domain"] == "atlas-atlas"
        # ATLAS is not an ATT&CK version — the field must not be present.
        assert "attack" not in layer["versions"]
        assert layer["versions"]["layer"]

    def test_unknown_matrix_returns_none(self):
        assert build_layer(name="x", matrix="Mobile", techniques=[("T1", None)]) is None

    def test_technique_entries_carry_highlight_and_tactic(self):
        layer = build_layer(
            name="Test",
            matrix="Enterprise",
            techniques=[("T1059", "execution")],
        )
        entry = layer["techniques"][0]
        assert entry["techniqueID"] == "T1059"
        assert entry["tactic"] == "execution"
        assert entry["color"] == HIGHLIGHT_COLOR
        assert entry["score"] == 1
        assert entry["enabled"] is True

    def test_tactic_omitted_when_none(self):
        layer = build_layer(
            name="Test", matrix="Enterprise", techniques=[("T1078", None)]
        )
        assert "tactic" not in layer["techniques"][0]

    def test_subtechnique_id_passes_through(self):
        layer = build_layer(
            name="Test", matrix="Enterprise", techniques=[("T1055.001", "defense-evasion")]
        )
        scored = layer["techniques"][0]
        assert scored["techniqueID"] == "T1055.001"
        assert scored["color"] == HIGHLIGHT_COLOR

    def test_scored_subtechnique_expands_its_parent(self):
        # Navigator hides a scored sub-technique under a collapsed parent; the
        # layer must expand the parent so the child is visible.
        layer = build_layer(
            name="Test", matrix="Enterprise", techniques=[("T1583.006", "resource-development")]
        )
        by_id = {t["techniqueID"]: t for t in layer["techniques"]}
        assert "T1583" in by_id
        assert by_id["T1583"]["showSubtechniques"] is True
        assert by_id["T1583"]["tactic"] == "resource-development"
        # The expand-only parent is not itself highlighted.
        assert "color" not in by_id["T1583"]

    def test_scored_parent_gets_showsubtechniques_not_duplicated(self):
        layer = build_layer(
            name="Test",
            matrix="Enterprise",
            techniques=[("T1055", "defense-evasion"), ("T1055.001", "defense-evasion")],
        )
        parents = [t for t in layer["techniques"] if t["techniqueID"] == "T1055"]
        assert len(parents) == 1
        assert parents[0]["showSubtechniques"] is True
        # It stays a scored/highlighted entry, not a bare expand-only one.
        assert parents[0]["color"] == HIGHLIGHT_COLOR

    def test_layout_surfaces_ids(self):
        layer = build_layer(
            name="Test", matrix="Enterprise", techniques=[("T1059", "execution")]
        )
        assert layer["layout"]["showID"] is True
        assert layer["layout"]["showName"] is True

    def test_duplicate_ids_collapsed(self):
        layer = build_layer(
            name="Test",
            matrix="Enterprise",
            techniques=[("T1059", "execution"), ("T1059", "execution")],
        )
        assert len(layer["techniques"]) == 1

    def test_empty_or_missing_ids_skipped(self):
        layer = build_layer(
            name="Test",
            matrix="Enterprise",
            techniques=[("", "execution"), ("T1059", "execution")],
        )
        ids = [t["techniqueID"] for t in layer["techniques"]]
        assert ids == ["T1059"]

    def test_has_legend(self):
        layer = build_layer(
            name="Test", matrix="Enterprise", techniques=[("T1059", "execution")]
        )
        assert layer["legendItems"][0]["color"] == HIGHLIGHT_COLOR


class TestSerialisation:
    def test_dumps_is_valid_json(self):
        layer = build_layer(
            name="Test", matrix="Enterprise", techniques=[("T1059", "execution")]
        )
        assert json.loads(dumps(layer)) == layer

    @pytest.mark.parametrize(
        "base, expected",
        [
            ("threat_group_scenario.md", "threat_group_scenario_layer.json"),
            ("custom_scenario.md", "custom_scenario_layer.json"),
            ("noext", "noext_layer.json"),
        ],
    )
    def test_layer_filename(self, base, expected):
        assert layer_filename(base) == expected
