"""Integrity tests for the RMM platform compromise quick-start preset."""

from core.attack_data import list_technique_options
from data.rmm_platform_compromise import (
    RMM_PLATFORM_COMPROMISE,
    RMM_PLATFORM_COMPROMISE_BRIEF,
    RMM_PLATFORM_COMPROMISE_REQUIRED_DECISIONS,
    RMM_PLATFORM_COMPROMISE_TECHNIQUES,
    rmm_platform_compromise_template_info,
)


def test_every_preset_technique_exists_in_enterprise_attack():
    available = {
        option["Display Name"] for option in list_technique_options("Enterprise")
    }
    assert set(RMM_PLATFORM_COMPROMISE_TECHNIQUES) <= available


def test_preset_covers_the_rmm_control_plane_incident_shape():
    assert "RMM platform" in RMM_PLATFORM_COMPROMISE_BRIEF
    assert "control plane" in RMM_PLATFORM_COMPROMISE_BRIEF
    assert "managed endpoints" in RMM_PLATFORM_COMPROMISE_BRIEF
    assert len(RMM_PLATFORM_COMPROMISE_TECHNIQUES) == len(
        set(RMM_PLATFORM_COMPROMISE_TECHNIQUES)
    )


def test_required_decisions_cover_the_response_lifecycle():
    leads = [
        decision.split(" — ", 1)[0]
        for decision in RMM_PLATFORM_COMPROMISE_REQUIRED_DECISIONS
    ]
    assert leads == [
        "Privileged access",
        "Containment boundary",
        "Scope",
        "Evidence",
        "Notification",
        "Trusted recovery",
    ]


def test_prompt_context_contains_the_premise_and_decisions():
    context = rmm_platform_compromise_template_info()
    assert RMM_PLATFORM_COMPROMISE in context
    assert RMM_PLATFORM_COMPROMISE_BRIEF in context
    for decision in RMM_PLATFORM_COMPROMISE_REQUIRED_DECISIONS:
        assert f"- {decision}" in context
