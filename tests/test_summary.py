"""Tests for `core.summary` — the compact views over a result and its inputs.

These are pure functions, so the assertions are about the externally visible
facts a reader relies on: which sections a long scenario exposes for
navigation, which facilitator shortcuts are found, how a snapshot reads back,
and whether the form has moved on from the result on screen.
"""

from __future__ import annotations

from core.summary import (
    comparable_inputs,
    describe_inputs,
    extract_sections,
    inputs_changed,
    modifier_labels,
    slugify,
    summarise_scenario,
    summary_line,
)

SCENARIO = """# APT29 Scenario

A phased intrusion against a mid-sized bank, from initial access to exfiltration.

## Scenario Overview

Details.

## Injects

### Inject 1 — Suspicious sign-in

Body.

## Discussion Questions

Body.

## Success Criteria

Body.

## Metrics and Measurement

Body.

## Artefacts

Body.

## Rules of Engagement

Body.

```markdown
# Not a heading — this is inside a fence
```
"""


class TestScenarioSummary:
    def test_sections_are_extracted_with_levels_and_anchors(self) -> None:
        sections = extract_sections(SCENARIO)

        titles = [section.title for section in sections]
        assert titles[0] == "APT29 Scenario"
        assert "Injects" in titles
        # A heading inside a fenced code block is code, not structure.
        assert "Not a heading — this is inside a fence" not in titles
        inject = next(s for s in sections if s.title == "Inject 1 — Suspicious sign-in")
        assert inject.level == 3
        assert inject.anchor == slugify("Inject 1 — Suspicious sign-in")

    def test_summary_exposes_overview_reading_time_and_quick_links(self) -> None:
        summary = summarise_scenario(SCENARIO)

        assert summary.title == "APT29 Scenario"
        assert summary.overview.startswith("A phased intrusion")
        assert summary.word_count > 0
        assert summary.reading_minutes >= 1
        assert summary.is_useful is True

        # The parts a facilitator runs the exercise from are reachable directly.
        labels = {label for label, _section in summary.quick_links}
        assert {
            "Injects",
            "Discussion questions",
            "Success criteria",
            "Metrics",
            "Artefacts",
            "Rules of engagement",
        } <= labels
        injects = next(s for label, s in summary.quick_links if label == "Injects")
        assert injects.anchor == "injects"

    def test_empty_scenario_has_nothing_to_navigate(self) -> None:
        summary = summarise_scenario("")

        assert summary.sections == ()
        assert summary.quick_links == ()
        assert summary.is_useful is False


SNAPSHOT = {
    "scenario_type": "threat_group",
    "matrix": "Enterprise",
    "organisation": {"industry": "Finance / Banking", "company_size": "Medium (51-200 employees)"},
    "selected_entity": {"type": "threat actor group", "name": "APT29"},
    "selected_techniques": [{"ATT&CK ID": "T1566"}, {"ATT&CK ID": "T1059"}],
    "sampled_techniques": [{"ATT&CK ID": "T1566"}],
    "kill_chain_string": "Initial Access: Phishing",
    "modifiers": {"ai_uplift": False, "purple_team_narrative": True},
    "identity": {"model": "gpt-5.6-sol", "provider": "OpenAI API"},
    "captured_at": "2026-09-08T09:30:00+00:00",
}


class TestInputDescription:
    def test_snapshot_reads_back_as_human_facts(self) -> None:
        facts = dict(describe_inputs(SNAPSHOT))

        assert facts["Framework"] == "Enterprise ATT&CK"
        assert facts["Industry"] == "Finance / Banking"
        assert facts["Company size"] == "Medium (51-200 employees)"
        assert facts["Threat group"] == "APT29"
        assert facts["Techniques"] == "1 technique"
        assert facts["Modifiers"] == "Purple-team narrative"
        assert facts["Model"] == "gpt-5.6-sol (OpenAI API)"
        assert facts["Generated"].startswith("2026-09-08")

    def test_summary_line_covers_the_pre_flight_confirmation(self) -> None:
        line = summary_line(SNAPSHOT)

        assert "Enterprise ATT&CK" in line
        assert "Finance / Banking" in line
        assert "APT29" in line
        assert "Purple-team narrative" in line

    def test_only_enabled_modifiers_are_named(self) -> None:
        assert modifier_labels(SNAPSHOT) == ("Purple-team narrative",)
        assert modifier_labels({"modifiers": {}}) == ()
        assert modifier_labels({"modifiers": {"template": "Evaluation Escape"}}) == (
            "Template: Evaluation Escape",
        )

    def test_ai_insider_selections_are_described(self) -> None:
        facts = dict(
            describe_inputs(
                {
                    "scenario_type": "ai_insider",
                    "selected_entity": {
                        "type": "deployment_archetype",
                        "name": "Autonomous Agent",
                    },
                    "selected_categories": ["Data Exfiltration"],
                    "selected_stride": ["S1"],
                    "scenario_seed": "An overnight evaluation run.",
                    "required_decisions": ["Containment — who pulls the plug"],
                    "modifiers": {},
                }
            )
        )

        assert facts["Deployment archetype"] == "Autonomous Agent"
        assert facts["Threat categories"] == "Data Exfiltration"
        assert facts["Scenario seed"] == "Provided"
        assert facts["Required decisions"] == "Containment"

    def test_empty_snapshot_describes_nothing(self) -> None:
        assert describe_inputs(None) == ()
        assert summary_line(None) == ""


class TestInputsChanged:
    def test_resampled_techniques_are_not_an_edit(self) -> None:
        """Page 1 resamples one technique per phase on every rerun."""
        rerun = dict(SNAPSHOT, sampled_techniques=[{"ATT&CK ID": "T1059"}])
        rerun["kill_chain_string"] = "Execution: Command and Scripting Interpreter"

        assert inputs_changed(SNAPSHOT, rerun) is False

    def test_changing_a_selection_is_an_edit(self) -> None:
        edited = dict(SNAPSHOT, selected_entity={"type": "threat actor group", "name": "APT28"})

        assert inputs_changed(SNAPSHOT, edited) is True

    def test_changing_a_modifier_is_an_edit(self) -> None:
        edited = dict(SNAPSHOT, modifiers={"ai_uplift": True, "purple_team_narrative": True})

        assert inputs_changed(SNAPSHOT, edited) is True

    def test_nothing_to_compare_is_never_stale(self) -> None:
        assert inputs_changed(SNAPSHOT, None) is False
        assert inputs_changed(None, SNAPSHOT) is False

    def test_comparable_inputs_drop_the_run_specific_fields(self) -> None:
        comparable = comparable_inputs(SNAPSHOT)

        assert "sampled_techniques" not in comparable
        assert "kill_chain_string" not in comparable
        assert "captured_at" not in comparable
        assert "identity" not in comparable
        assert comparable["matrix"] == "Enterprise"
