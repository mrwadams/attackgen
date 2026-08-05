"""Tests for `data.ai_insider_threats` — the framework data and its templates.

These are integrity guards over hand-authored data: every template and category
must reference archetypes, categories and STRIDE codes that actually exist, so a
typo fails here rather than at prompt-build time in the UI.
"""

from __future__ import annotations

import pytest

from data.ai_insider_threats import (
    AI_INSIDER_TEMPLATES,
    DEPLOYMENT_ARCHETYPES,
    STRIDE_THREATS,
    THREAT_CATEGORIES,
    resolve_template,
    stride_code_from_option,
    stride_options,
)

EVAL_ESCAPE = "Evaluation Environment Escape"


class TestReferentialIntegrity:
    @pytest.mark.parametrize("name", list(AI_INSIDER_TEMPLATES))
    def test_template_references_resolve(self, name):
        template = AI_INSIDER_TEMPLATES[name]
        assert template["archetype"] in DEPLOYMENT_ARCHETYPES
        for category in template["categories"]:
            assert category in THREAT_CATEGORIES
        for code in template["stride"]:
            assert code in STRIDE_THREATS

    @pytest.mark.parametrize("name", list(THREAT_CATEGORIES))
    def test_category_stride_codes_resolve(self, name):
        for code in THREAT_CATEGORIES[name]["stride"]:
            assert code in STRIDE_THREATS

    def test_containment_threats_are_reachable_from_a_category(self):
        """The four containment threats must be reachable from the category
        flow, since page 3 derives STRIDE from categories when none is picked.

        Not asserted for the whole catalogue: S3, T3, T4, I4, E3 and E4 belong
        to no category and are selectable only from the STRIDE list directly.
        """
        mapped = {c for cat in THREAT_CATEGORIES.values() for c in cat["stride"]}
        assert {"S5", "R4", "I5", "E5"} <= mapped

    def test_stride_options_round_trip(self):
        codes = [stride_code_from_option(opt) for opt in stride_options()]
        assert codes == list(STRIDE_THREATS)


class TestResolveTemplate:
    def test_returns_all_keys_for_a_bare_template(self):
        """Templates without a payload still get the keys, defaulted."""
        resolved = resolve_template("Autonomous Agent Data Heist")
        assert set(resolved) == {
            "archetype", "categories", "stride", "brief", "required_decisions",
        }
        assert resolved["brief"] == ""
        assert resolved["required_decisions"] == []

    def test_returns_copies_not_references(self):
        """Mutating a resolved template must not corrupt the module data."""
        resolved = resolve_template(EVAL_ESCAPE)
        resolved["stride"].append("BOGUS")
        assert "BOGUS" not in AI_INSIDER_TEMPLATES[EVAL_ESCAPE]["stride"]

    def test_unknown_template_raises(self):
        with pytest.raises(ValueError, match="Unknown template"):
            resolve_template("Not A Template")


class TestEvaluationEnvironmentEscapePreset:
    def test_selections(self):
        resolved = resolve_template(EVAL_ESCAPE)
        assert resolved["archetype"] == "Human-as-Auditor (L4 — Critical Threat)"
        assert "Containment & Third-Party Impact" in resolved["categories"]
        assert set(resolved["stride"]) == {"S5", "R4", "I5", "E5"}

    def test_brief_covers_the_incident_shape(self):
        brief = resolve_template(EVAL_ESCAPE)["brief"]
        assert brief
        lowered = brief.lower()
        for concept in ("evaluation", "internet", "reusable", "telemetry"):
            assert concept in lowered

    def test_required_decisions_cover_the_four_response_areas(self):
        decisions = resolve_template(EVAL_ESCAPE)["required_decisions"]
        assert len(decisions) == 4
        leads = [d.split(" — ", 1)[0] for d in decisions]
        assert leads == [
            "Containment",
            "Identity rotation",
            "Third-party notification",
            "Evidence preservation",
        ]
