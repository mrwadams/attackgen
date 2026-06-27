"""Tests for `atlas_parser` — the custom MITRE ATLAS STIX parser.

The parser turns on-disk STIX JSON into the technique/tactic rows that feed the
Threat Group and Custom scenario pages. These tests drive it from a small
in-memory STIX document written to a temp file, so they assert the indexing,
ordering and fallback contracts without depending on the full ATLAS matrix.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from atlas_parser import ATLASData, get_techniques_from_case_study_procedure


def _atlas(source_name: str = "mitre-atlas") -> dict:
    """A minimal STIX document exercising every indexing branch.

    Tactics are intentionally given out of kill-chain order (Initial Access
    before Reconnaissance) so a test can prove `get_tactics` re-orders them.
    """
    return {
        "objects": [
            # Tactics — only two of the sixteen ordered IDs are present.
            {
                "type": "x-mitre-tactic",
                "id": "x-mitre-tactic--ia",
                "name": "Initial Access",
                "x_mitre_shortname": "initial-access",
                "external_references": [
                    {"source_name": "mitre-atlas", "external_id": "AML.TA0004"}
                ],
            },
            {
                "type": "x-mitre-tactic",
                "id": "x-mitre-tactic--recon",
                "name": "Reconnaissance",
                "x_mitre_shortname": "reconnaissance",
                "external_references": [
                    {"source_name": "mitre-atlas", "external_id": "AML.TA0002"}
                ],
            },
            # A parent technique under Reconnaissance.
            {
                "type": "attack-pattern",
                "id": "attack-pattern--t1",
                "name": "Tech One",
                "description": "desc one",
                "x_mitre_is_subtechnique": False,
                "kill_chain_phases": [
                    {"kill_chain_name": "atlas", "phase_name": "reconnaissance"}
                ],
                "external_references": [
                    {"source_name": "mitre-atlas", "external_id": "AML.T0001"}
                ],
            },
            # A subtechnique under Initial Access.
            {
                "type": "attack-pattern",
                "id": "attack-pattern--t2",
                "name": "Sub Tech",
                "description": "sub desc",
                "x_mitre_is_subtechnique": True,
                "kill_chain_phases": [
                    {"kill_chain_name": "atlas", "phase_name": "initial-access"}
                ],
                "external_references": [
                    {"source_name": "mitre-atlas", "external_id": "AML.T0001.001"}
                ],
            },
            # A technique with no kill-chain phases (drives the "Unknown" fallback).
            {
                "type": "attack-pattern",
                "id": "attack-pattern--t3",
                "name": "No Phase Tech",
                "description": "no phase desc",
                "x_mitre_is_subtechnique": False,
                "external_references": [
                    {"source_name": "mitre-atlas", "external_id": "AML.T0009"}
                ],
            },
            # An attack-pattern lacking a mitre-atlas external id — must be skipped.
            {
                "type": "attack-pattern",
                "id": "attack-pattern--noid",
                "name": "Unindexed Tech",
                "external_references": [
                    {"source_name": "other-source", "external_id": "X1"}
                ],
            },
            # A mitigation and a relationship, to confirm they are indexed/collected.
            {
                "type": "course-of-action",
                "id": "course-of-action--m1",
                "name": "Mitigation One",
                "external_references": [
                    {"source_name": "mitre-atlas", "external_id": "AML.M0001"}
                ],
            },
            {
                "type": "relationship",
                "id": "relationship--r1",
                "relationship_type": "mitigates",
                "source_ref": "course-of-action--m1",
                "target_ref": "attack-pattern--t1",
            },
        ]
    }


@pytest.fixture
def atlas_data(tmp_path: Path) -> ATLASData:
    path = tmp_path / "stix-atlas.json"
    path.write_text(json.dumps(_atlas()))
    return ATLASData(str(path))


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


def test_indexes_only_objects_with_mitre_atlas_external_id(atlas_data: ATLASData) -> None:
    # The three techniques carrying an ATLAS id are indexed; the one without is not.
    assert set(atlas_data.techniques) == {"AML.T0001", "AML.T0001.001", "AML.T0009"}
    assert atlas_data.get_technique_by_id("X1") is None
    assert set(atlas_data.tactics) == {"AML.TA0004", "AML.TA0002"}
    assert set(atlas_data.mitigations) == {"AML.M0001"}
    assert len(atlas_data.relationships) == 1


# ---------------------------------------------------------------------------
# Tactics
# ---------------------------------------------------------------------------


def test_get_tactics_returns_kill_chain_order_skipping_absent_ids(
    atlas_data: ATLASData,
) -> None:
    # Reconnaissance (AML.TA0002) precedes Initial Access (AML.TA0004) in the
    # canonical order, even though the document lists them the other way round.
    assert atlas_data.get_tactic_names_ordered() == ["Reconnaissance", "Initial Access"]


def test_get_tactic_name_by_shortname_echoes_unknown(atlas_data: ATLASData) -> None:
    assert atlas_data.get_tactic_name_by_shortname("reconnaissance") == "Reconnaissance"
    assert atlas_data.get_tactic_name_by_shortname("no-such-tactic") == "no-such-tactic"


def test_get_tactic_name_by_id_echoes_unknown(atlas_data: ATLASData) -> None:
    assert atlas_data.get_tactic_name_by_id("AML.TA0002") == "Reconnaissance"
    assert atlas_data.get_tactic_name_by_id("AML.TA9999") == "AML.TA9999"


# ---------------------------------------------------------------------------
# Techniques
# ---------------------------------------------------------------------------


def test_get_techniques_can_exclude_subtechniques(atlas_data: ATLASData) -> None:
    with_subs = {t["external_id"] for t in atlas_data.get_techniques(include_subtechniques=True)}
    without_subs = {t["external_id"] for t in atlas_data.get_techniques(include_subtechniques=False)}

    assert "AML.T0001.001" in with_subs
    assert "AML.T0001.001" not in without_subs
    assert "AML.T0001" in without_subs


def test_get_techniques_for_tactic_matches_by_phase_name(atlas_data: ATLASData) -> None:
    recon = atlas_data.get_techniques_for_tactic("reconnaissance")
    assert [t["external_id"] for t in recon] == ["AML.T0001"]


def test_get_atlas_id_maps_stix_id_back_to_external_id(atlas_data: ATLASData) -> None:
    assert atlas_data.get_atlas_id("attack-pattern--t1") == "AML.T0001"
    assert atlas_data.get_atlas_id("attack-pattern--missing") is None


# ---------------------------------------------------------------------------
# Case-study procedure extraction — the three-way tactic-name fallback.
# ---------------------------------------------------------------------------


def test_procedure_uses_step_tactic_id_when_present(atlas_data: ATLASData) -> None:
    procedure = [{"technique": "AML.T0001", "tactic": "AML.TA0002", "description": "step desc"}]
    rows = get_techniques_from_case_study_procedure(procedure, atlas_data)

    assert rows == [
        {
            "Technique Name": "Tech One",
            "ATT&CK ID": "AML.T0001",
            "Phase Name": "Reconnaissance",
            "Description": "step desc",
        }
    ]


def test_procedure_falls_back_to_technique_kill_chain_phase(atlas_data: ATLASData) -> None:
    # No tactic on the step → resolve via the technique's first kill-chain phase.
    procedure = [{"technique": "AML.T0001"}]
    rows = get_techniques_from_case_study_procedure(procedure, atlas_data)

    assert rows[0]["Phase Name"] == "Reconnaissance"
    # Description falls back to the technique's own description.
    assert rows[0]["Description"] == "desc one"


def test_procedure_phase_is_unknown_without_tactic_or_phases(atlas_data: ATLASData) -> None:
    procedure = [{"technique": "AML.T0009"}]
    rows = get_techniques_from_case_study_procedure(procedure, atlas_data)

    assert rows[0]["Phase Name"] == "Unknown"


def test_procedure_drops_steps_with_unknown_technique(atlas_data: ATLASData) -> None:
    procedure = [
        {"technique": "AML.T0001", "tactic": "AML.TA0002"},
        {"technique": "AML.T9999", "tactic": "AML.TA0002"},  # not indexed → dropped
    ]
    rows = get_techniques_from_case_study_procedure(procedure, atlas_data)

    assert [r["ATT&CK ID"] for r in rows] == ["AML.T0001"]
