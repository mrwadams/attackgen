"""Tests for `core.detections` — the purple-team Detection & Response join.

The Enterprise/ICS join is exercised with a light fake that mimics the
`mitreattack-python` return shapes (detection strategies and mitigations as
``{"object": <stix2 obj>}``, analytics as plain dicts). ATLAS is exercised
against the real bundled data through `ATLASData`.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from atlas_parser import ATLASData
from core.detections import (
    assemble_defense_document,
    build_defense_report,
    build_narrative_messages,
    defense_download_name,
    defense_to_markdown,
)


class FakeMitreData:
    """Minimal stand-in for `MitreAttackData` covering the four helpers used."""

    def __init__(self) -> None:
        self._tech = SimpleNamespace(
            id="attack-pattern--t1059", name="Command and Scripting Interpreter"
        )
        self._strategy = SimpleNamespace(id="det--1", name="Behavioral Detection")
        self._mitigation = SimpleNamespace(
            id="coa--1", name="Disable or Remove Feature", description="Turn it off."
        )
        self._attack_ids = {"det--1": "DET0516", "coa--1": "M1042"}

    def get_object_by_attack_id(self, external_id, obj_type):
        return self._tech if external_id == "T1059" else None

    def get_attack_id(self, stix_id):
        return self._attack_ids.get(stix_id)

    def get_detection_strategies_detecting_technique(self, stix_id):
        return [{"object": self._strategy, "relationships": []}]

    def get_analytics_by_detection_strategy(self, strategy_id):
        return [
            {
                "name": "Analytic 1428",
                "x_mitre_platforms": ["Windows"],
                "x_mitre_log_source_references": [
                    {"name": "WinEventLog:Security", "channel": "EventCode=4688"}
                ],
                "description": "Process creation of interpreters.",
            }
        ]

    def get_mitigations_mitigating_technique(self, stix_id):
        return [{"object": self._mitigation, "relationships": []}]


class TestBuildDefenseReportEnterprise:
    def test_joins_strategies_analytics_and_mitigations(self):
        report = build_defense_report(
            matrix="Enterprise", technique_ids=["T1059"], mitre_data=FakeMitreData()
        )
        assert report["matrix"] == "Enterprise"
        (tech,) = report["techniques"]
        assert tech["id"] == "T1059"
        strategy = tech["detection_strategies"][0]
        assert strategy["id"] == "DET0516"
        analytic = strategy["analytics"][0]
        assert analytic["name"] == "Analytic 1428"
        assert analytic["platforms"] == ["Windows"]
        assert analytic["log_sources"] == [
            {"name": "WinEventLog:Security", "channel": "EventCode=4688"}
        ]
        assert tech["mitigations"][0]["id"] == "M1042"

    def test_log_sources_rolled_up_by_name_and_deduped(self):
        report = build_defense_report(
            matrix="Enterprise",
            technique_ids=["T1059", "T1059"],  # duplicate collapses
            mitre_data=FakeMitreData(),
        )
        assert len(report["techniques"]) == 1
        # Roll-up is by source *name* (the "enable these" list), not name+channel.
        assert report["log_sources"] == ["WinEventLog:Security"]

    def test_unknown_technique_dropped(self):
        # Only a bogus ID -> nothing resolves -> None.
        assert (
            build_defense_report(
                matrix="Enterprise", technique_ids=["T9999"], mitre_data=FakeMitreData()
            )
            is None
        )

    def test_blank_ids_ignored(self):
        assert (
            build_defense_report(
                matrix="Enterprise", technique_ids=["", None], mitre_data=FakeMitreData()
            )
            is None
        )

    def test_unknown_matrix_returns_none(self):
        assert (
            build_defense_report(
                matrix="Mobile", technique_ids=["T1059"], mitre_data=FakeMitreData()
            )
            is None
        )

    def test_missing_data_object_returns_none(self):
        assert (
            build_defense_report(matrix="Enterprise", technique_ids=["T1059"]) is None
        )


@pytest.fixture(scope="module")
def atlas():
    return ATLASData("./data/stix-atlas.json")


class TestBuildDefenseReportAtlas:
    def test_atlas_returns_mitigations_only(self, atlas):
        report = build_defense_report(
            matrix="ATLAS", technique_ids=["AML.T0051"], atlas_data=atlas
        )
        (tech,) = report["techniques"]
        assert tech["detection_strategies"] == []
        assert tech["mitigations"]  # ATLAS has mitigations
        assert report["log_sources"] == []  # no analytics -> no log sources

    def test_atlas_without_data_returns_none(self):
        assert (
            build_defense_report(matrix="ATLAS", technique_ids=["AML.T0051"]) is None
        )


class TestLogSourceHygiene:
    def test_none_channel_dropped_and_rollup_dedupes_by_name(self):
        class _Fake(FakeMitreData):
            def get_analytics_by_detection_strategy(self, strategy_id):
                return [
                    {
                        "name": "Analytic X",
                        "x_mitre_platforms": ["Windows"],
                        "x_mitre_log_source_references": [
                            {"name": "WinEventLog:Sysmon", "channel": "EventCode=1"},
                            {"name": "WinEventLog:Sysmon", "channel": "EventCode=3, 22"},
                            # MITRE stores a literal "None" for channel-less sources.
                            {"name": "Application Log", "channel": "None"},
                        ],
                        "description": "",
                    }
                ]

        report = build_defense_report(
            matrix="Enterprise", technique_ids=["T1059"], mitre_data=_Fake()
        )
        # Roll-up dedupes by source name: Sysmon once (not once per EventCode).
        assert report["log_sources"] == ["WinEventLog:Sysmon", "Application Log"]

        md = defense_to_markdown(report)
        # The literal "None" channel never renders as "(None)".
        assert "(None)" not in md
        assert "Application Log" in md  # still listed, just without a bogus channel
        # Per-analytic detail keeps the real channels.
        assert "WinEventLog:Sysmon (EventCode=1)" in md


class TestDefenseToMarkdown:
    def test_enterprise_sections_present(self):
        report = build_defense_report(
            matrix="Enterprise", technique_ids=["T1059"], mitre_data=FakeMitreData()
        )
        md = defense_to_markdown(report)
        assert "## 🛡️ Detection & Response" in md
        assert "### Log sources to enable" in md
        assert "Command and Scripting Interpreter (T1059)" in md
        assert "**Detection strategies**" in md
        assert "Analytic 1428 — Windows — WinEventLog:Security (EventCode=4688)" in md
        assert "**Mitigations**" in md
        assert "Disable or Remove Feature (M1042)" in md

    def test_atlas_notes_no_detection_model(self):
        atlas = ATLASData("./data/stix-atlas.json")
        report = build_defense_report(
            matrix="ATLAS", technique_ids=["AML.T0051"], atlas_data=atlas
        )
        md = defense_to_markdown(report)
        assert "does not yet publish detection strategies" in md
        assert "**Detection strategies**" not in md
        assert "**Mitigations**" in md


class TestDownloadHelpers:
    @pytest.mark.parametrize(
        "md_name, expected",
        [
            ("scn_20260714-101010.md", "scn_20260714-101010_detection.md"),
            ("noext", "noext_detection.md"),
        ],
    )
    def test_defense_download_name(self, md_name, expected):
        assert defense_download_name(md_name) == expected

    def test_assemble_document_without_narrative(self):
        doc = assemble_defense_document("## 🛡️ Detection & Response", None, title="APT29")
        assert doc.startswith("# Detection & Response — APT29")
        assert "Detection & Response Reference" not in doc

    def test_assemble_document_with_narrative(self):
        doc = assemble_defense_document(
            "## 🛡️ Detection & Response", "## Walkthrough\n\nStage 1.", title="APT29"
        )
        assert "## Walkthrough" in doc
        assert "## Detection & Response Reference" in doc
        # Narrative comes before the reference section.
        assert doc.index("## Walkthrough") < doc.index("Detection & Response Reference")


class TestNarrativeMessages:
    def test_grounding_data_embedded_in_user_message(self):
        report = build_defense_report(
            matrix="Enterprise", technique_ids=["T1059"], mitre_data=FakeMitreData()
        )
        messages = build_narrative_messages("SCENARIO BODY", report)
        assert [m["role"] for m in messages] == ["system", "user"]
        user = messages[1]["content"]
        assert "SCENARIO BODY" in user
        # The deterministic join is handed to the model as grounding.
        assert "Analytic 1428" in user
        assert "Disable or Remove Feature (M1042)" in user
