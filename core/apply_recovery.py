"""Recovery for the Assistant's apply step.

The Assistant's apply step (``core.assistant.apply_write_back``) overwrites a
scenario page's scenario text and Detection & Response narrative in place.
This module is the other half of that contract: stashing what was there
before the *first* apply of the session, so a single "Revert to original"
action can undo every apply since, and captioning the result as refined.

Page 4 stashes and marks around each apply; the scenario-page coordinator
shows the caption and the Revert button, runs the revert, and forgets the
applies when a new result replaces the old one.
"""

from __future__ import annotations

from datetime import datetime, timezone

import streamlit as st

from core.assistant import (
    DEFENSE_NARRATIVE_KEY,
    SCENARIO_META_KEY,
    SCENARIO_TEXT_KEY,
)
from core.detections import assemble_defense_document, defense_title
from core.schemas import ResultKeys


def stash_pre_apply(page_id: str) -> None:
    """Stash the pre-apply scenario + narrative once, before the first apply.

    A second or third apply in the same session must not overwrite this
    stash with its own (already-refined) starting point — "Revert to
    original" always means the version this page generated, not the version
    before the *most recent* apply.
    """
    keys = ResultKeys(page_id)
    if keys.pre_apply in st.session_state:
        return
    defense_state = st.session_state.get(keys.defense) or {}
    st.session_state[keys.pre_apply] = {
        "text": st.session_state.get(keys.text),
        "narrative": defense_state.get("narrative_md"),
    }


def mark_applied(page_id: str) -> str:
    """Record now as this page's last successful Assistant apply; return it."""
    applied_at = datetime.now(timezone.utc).isoformat()
    st.session_state[ResultKeys(page_id).applied_at] = applied_at
    return applied_at


def has_been_applied(page_id: str) -> bool:
    """Has the Assistant applied changes to this page's result?"""
    return bool(st.session_state.get(ResultKeys(page_id).applied_at))


def _format_applied(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return str(value)
    return parsed.strftime("%Y-%m-%d %H:%M UTC" if parsed.tzinfo else "%Y-%m-%d %H:%M")


def applied_caption(page_id: str) -> str | None:
    """The "refined via the Assistant" caption, or ``None`` if never applied.

    Shared by the scenario page's result and the Assistant page itself, so the two
    always describe the same apply in the same words.
    """
    applied_at = st.session_state.get(ResultKeys(page_id).applied_at)
    if not applied_at:
        return None
    return f"Refined via the AttackGen Assistant — last applied {_format_applied(applied_at)}."


def revert_to_original(page_id: str) -> None:
    """Restore the scenario and narrative stashed before the first apply.

    Both move together — a scenario without its matching narrative (or vice
    versa) would leave the Detection & Response tab describing a different
    incident than the one on screen.
    """
    keys = ResultKeys(page_id)
    stash = st.session_state.pop(keys.pre_apply, None)
    st.session_state.pop(keys.applied_at, None)
    if not stash:
        return

    original_text = stash.get("text")
    st.session_state[keys.text] = original_text

    original_narrative = stash.get("narrative")
    defense_state = st.session_state.get(keys.defense)
    if defense_state is not None:
        defense_state = dict(defense_state)
        title = defense_title(defense_state.get("download_md", ""))
        defense_state["narrative_md"] = original_narrative
        defense_state["download_md"] = assemble_defense_document(
            defense_state.get("deterministic_md", ""), original_narrative, title=title
        )
        st.session_state[keys.defense] = defense_state

    # Follow the same ownership rule the coordinator's persist and clear use:
    # only refresh the Assistant handoff while it still points at this page.
    meta = st.session_state.get(SCENARIO_META_KEY) or {}
    if meta.get("page_id") == page_id:
        st.session_state[SCENARIO_TEXT_KEY] = original_text
        st.session_state[DEFENSE_NARRATIVE_KEY] = original_narrative


def forget_applies(page_id: str) -> None:
    """Drop the stash and the applied mark, as for a freshly generated result."""
    keys = ResultKeys(page_id)
    st.session_state.pop(keys.pre_apply, None)
    st.session_state.pop(keys.applied_at, None)
