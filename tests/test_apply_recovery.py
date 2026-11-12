"""Tests for `core.apply_recovery`, the Assistant apply's undo half.

The Assistant overwrites a page's scenario and narrative in place; these
functions stash what was there before the first apply, mark the result as
refined, and restore the stash on "Revert to original". The coordinator's
rendering of the caption and the Revert button stays in `test_scenario_page`.
"""

from __future__ import annotations

from core.apply_recovery import (
    applied_caption,
    forget_applies,
    has_been_applied,
    mark_applied,
    revert_to_original,
    stash_pre_apply,
)


def test_stash_keeps_the_version_from_before_the_first_apply(fake_session_state) -> None:
    fake_session_state["threat_group_scenario_text"] = "# Original"
    fake_session_state["threat_group_scenario_defense"] = {"narrative_md": "## Original walk"}
    stash_pre_apply("threat_group")

    fake_session_state["threat_group_scenario_text"] = "# Refined once"
    fake_session_state["threat_group_scenario_defense"] = {"narrative_md": "## Refined walk"}
    stash_pre_apply("threat_group")

    assert fake_session_state["threat_group_scenario_pre_apply"] == {
        "text": "# Original",
        "narrative": "## Original walk",
    }


def test_no_caption_until_an_apply_is_marked(fake_session_state) -> None:
    assert not has_been_applied("threat_group")
    assert applied_caption("threat_group") is None

    mark_applied("threat_group")

    assert has_been_applied("threat_group")
    assert applied_caption("threat_group").startswith(
        "Refined via the AttackGen Assistant — last applied "
    )


def test_caption_names_the_apply_time_in_utc(fake_session_state) -> None:
    fake_session_state["threat_group_scenario_applied_at"] = "2026-11-12T09:05:41+00:00"

    assert applied_caption("threat_group") == (
        "Refined via the AttackGen Assistant — last applied 2026-11-12 09:05 UTC."
    )


_ORIGINAL_DOWNLOAD = "# Detection & Response — APT29\n\n## T1566 Phishing\n"
_REFINED_DOWNLOAD = (
    "# Detection & Response — APT29\n\n## Refined walk\n\n---\n\n"
    "## Detection & Response Reference\n\n## T1566 Phishing\n"
)


def _applied_result(fake_session_state, *, handoff_page: str = "threat_group") -> None:
    """A threat-group result generated without a narrative, then applied once."""
    fake_session_state["threat_group_scenario_text"] = "# Original"
    fake_session_state["threat_group_scenario_defense"] = {
        "deterministic_md": "## T1566 Phishing",
        "narrative_md": None,
        "download_md": _ORIGINAL_DOWNLOAD,
    }
    stash_pre_apply("threat_group")
    fake_session_state["threat_group_scenario_text"] = "# Refined"
    fake_session_state["threat_group_scenario_defense"] = {
        "deterministic_md": "## T1566 Phishing",
        "narrative_md": "## Refined walk",
        "download_md": _REFINED_DOWNLOAD,
    }
    mark_applied("threat_group")
    fake_session_state["last_scenario_meta"] = {"page_id": handoff_page}
    fake_session_state["last_scenario_text"] = "# Refined"
    fake_session_state["last_defense_narrative"] = "## Refined walk"


def test_revert_restores_the_scenario_and_narrative_together(fake_session_state) -> None:
    _applied_result(fake_session_state)

    revert_to_original("threat_group")

    assert fake_session_state["threat_group_scenario_text"] == "# Original"
    defense = fake_session_state["threat_group_scenario_defense"]
    assert defense["narrative_md"] is None
    assert defense["download_md"] == _ORIGINAL_DOWNLOAD
    assert fake_session_state["last_scenario_text"] == "# Original"
    assert fake_session_state["last_defense_narrative"] is None
    assert "threat_group_scenario_pre_apply" not in fake_session_state
    assert not has_been_applied("threat_group")


def test_revert_leaves_another_pages_assistant_handoff_alone(fake_session_state) -> None:
    _applied_result(fake_session_state, handoff_page="custom")

    revert_to_original("threat_group")

    assert fake_session_state["threat_group_scenario_text"] == "# Original"
    assert fake_session_state["last_scenario_text"] == "# Refined"
    assert fake_session_state["last_defense_narrative"] == "## Refined walk"


def test_revert_without_a_stash_changes_nothing(fake_session_state) -> None:
    fake_session_state["threat_group_scenario_text"] = "# Original"

    revert_to_original("threat_group")

    assert fake_session_state == {"threat_group_scenario_text": "# Original"}


def test_forgetting_applies_drops_the_caption_and_the_revert(fake_session_state) -> None:
    _applied_result(fake_session_state)

    forget_applies("threat_group")

    assert applied_caption("threat_group") is None
    revert_to_original("threat_group")
    assert fake_session_state["threat_group_scenario_text"] == "# Refined"
