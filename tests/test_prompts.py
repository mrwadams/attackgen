"""Tests for `core.prompts` — the shared, headless prompt builders.

These prove the builders are Streamlit-free (no `fake_session_state` fixture is
needed) and pin the load-bearing details: the shared system prompt, the ATLAS
vs ATTACK branch selection, the AI-uplift append, and the deliberate asymmetry
that the Custom ATTACK template alone omits the "Write in British English." line.
"""

from __future__ import annotations

from core.ai_uplift import AI_UPLIFT_PROMPT
from core.prompts import (
    AI_INSIDER_SYSTEM_PROMPT,
    SCENARIO_SYSTEM_PROMPT,
    build_ai_insider_messages,
    build_custom_messages,
    build_threat_group_messages,
)
from data.ai_insider_threats import DEPLOYMENT_ARCHETYPES

BRITISH = "Write in British English."


def test_scenario_system_prompt_pinned():
    assert SCENARIO_SYSTEM_PROMPT == (
        "You are a cybersecurity expert. Your task is to produce a comprehensive incident response "
        "testing scenario based on the information provided. Format your response using proper "
        "Markdown syntax with headers, bullet points, and formatting for readability."
    )


def test_ai_insider_system_prompt_pinned():
    assert AI_INSIDER_SYSTEM_PROMPT.startswith(
        "You are a cybersecurity expert specialising in AI agent security"
    )
    assert AI_INSIDER_SYSTEM_PROMPT.endswith("Write in British English.")


class TestThreatGroupMessages:
    def test_shape_and_system_prompt(self):
        msgs = build_threat_group_messages(
            matrix="Enterprise",
            selected_group_alias="APT29",
            kill_chain_string="Initial Access: Phishing (T1566)",
            industry="Finance",
            company_size="Large",
        )
        assert len(msgs) == 2
        assert msgs[0] == {"role": "system", "content": SCENARIO_SYSTEM_PROMPT}
        user = msgs[1]["content"]
        assert "APT29" in user and "Finance" in user and "Large" in user
        assert "Enterprise" in user  # {matrix} interpolated
        assert "Initial Access: Phishing (T1566)" in user
        assert BRITISH in user

    def test_atlas_branch_uses_case_study_wording(self):
        msgs = build_threat_group_messages(
            matrix="ATLAS",
            selected_group_alias="Some Case Study",
            kill_chain_string="x",
            industry="Tech",
            company_size="SMB",
        )
        assert "MITRE ATLAS case study" in msgs[1]["content"]
        assert BRITISH in msgs[1]["content"]

    def test_ai_uplift_appends_only_when_true(self):
        base = build_threat_group_messages(
            matrix="Enterprise", selected_group_alias="APT29", kill_chain_string="x",
            industry="F", company_size="L", ai_uplift=False,
        )[1]["content"]
        upl = build_threat_group_messages(
            matrix="Enterprise", selected_group_alias="APT29", kill_chain_string="x",
            industry="F", company_size="L", ai_uplift=True,
        )[1]["content"]
        assert not base.endswith(AI_UPLIFT_PROMPT)
        assert upl.endswith(AI_UPLIFT_PROMPT)


class TestCustomMessages:
    def test_attack_template_omits_british_english(self):
        """Regression guard: page 2's ATTACK template alone drops the British
        English line. Must not be normalised when lifted to core/prompts.py."""
        user = build_custom_messages(
            matrix="Enterprise",
            selected_techniques_string="T1059",
            template_info="",
            industry="Finance",
            company_size="Large",
        )[1]["content"]
        assert BRITISH not in user
        assert "T1059" in user

    def test_atlas_template_keeps_british_english(self):
        user = build_custom_messages(
            matrix="ATLAS",
            selected_techniques_string="AML.T0051",
            template_info="This is a 'Prompt Injection Attack' scenario.",
            industry="Finance",
            company_size="Large",
        )[1]["content"]
        assert BRITISH in user
        assert "AML.T0051" in user
        assert "Prompt Injection Attack" in user


class TestAiInsiderMessages:
    def test_shape_and_content(self):
        archetype = next(iter(DEPLOYMENT_ARCHETYPES))
        msgs = build_ai_insider_messages(
            archetype_name=archetype,
            selected_categories=[],
            selected_stride=[],
            selected_capabilities=[],
            industry="Healthcare",
            company_size="Large",
        )
        assert msgs[0] == {"role": "system", "content": AI_INSIDER_SYSTEM_PROMPT}
        user = msgs[1]["content"]
        assert archetype in user
        assert "Healthcare" in user
        assert "tabletop exercise" in user
        # No capabilities selected -> the fallback line is used.
        assert "assume a capable frontier coding agent" in user
