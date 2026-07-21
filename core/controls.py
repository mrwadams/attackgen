"""Defensive control overlay for conventional attack scenarios.

A tabletop is more useful when it is measured against the defences the
organisation actually has. When the user describes their key security controls
(for example "EDR on all endpoints, network segmentation between IT and OT, MFA
on remote access, no egress logging"), this overlay asks the model to assess the
kill chain *against those controls* — calling out, stage by stage, whether each
technique would likely be **blocked, detected, or missed**, and where the gaps
are that the exercise should close.

Like ``core.ai_uplift`` it is an optional per-page modifier appended to the user
message; it does not change the selected kill chain, only adds a defensive lens.
The pure ``append_controls`` core takes the control description as an explicit
string so headless callers (the MCP server, ``core.prompts``) reuse the exact
framing without touching ``st.session_state``. An empty / whitespace-only
description is a no-op.

The AI Insider Threat page (page 3) does not use this overlay — it has its own
detection-strategy and NIST CSF control framing built into its prompt.
"""

from __future__ import annotations

import streamlit as st

# Label and help text for the per-page control input.
CONTROLS_LABEL = "🏛️ Your security controls (optional)"
CONTROLS_HELP = (
    "Describe the key security controls in your environment (for example EDR, "
    "network segmentation, MFA, email filtering, egress logging gaps). The "
    "scenario will assess the attack chain against them — flagging, stage by "
    "stage, what would likely be blocked, detected or missed, and which gaps to "
    "close. Leave blank to skip."
)
CONTROLS_PLACEHOLDER = (
    "e.g. EDR on all endpoints, network segmentation between corporate and "
    "production, MFA on all remote access, SIEM with 90-day retention, no egress "
    "logging on the OT segment"
)

# Appended to the user message when a control description is supplied. Written to
# slot cleanly after the existing "Your task" block in the scenario templates.
# The only ``.format`` field is ``{controls}`` — keep other braces out.
CONTROLS_PROMPT_TEMPLATE = """
**Defensive control overlay:**
The organisation reports the following existing security controls:
{controls}

Assess the kill chain above against these stated controls. This does **not** change the techniques in play — it adds a defensive lens. Throughout the scenario:
- For each stage of the attack, judge whether the relevant technique would most likely be **blocked**, **detected**, or **missed** given the controls above, and briefly say why.
- Be realistic and specific: tie each judgement to a named control (or the absence of one). Do not assume controls that were not listed.
- Call out the **coverage gaps** — the stages that would go undetected or unmitigated — as the priorities the exercise should help the team close.
- Where a control would generate a detection, note the **evidence** the response team should expect to find (log source, alert, artefact) so the tabletop can verify whether they would actually have seen it.
"""


def render_controls_input(page_id: str, *, label_visibility: str = "visible") -> str:
    """Render the security-controls text area for a scenario page.

    The widget is keyed by ``page_id`` so the two scenario pages keep independent
    state. Pass ``label_visibility="collapsed"`` when the input already sits inside
    an expander titled with ``CONTROLS_LABEL``. Returns the current text value.
    """
    return st.text_area(
        CONTROLS_LABEL,
        key=f"{page_id}_controls",
        help=CONTROLS_HELP,
        placeholder=CONTROLS_PLACEHOLDER,
        label_visibility=label_visibility,
    )


def get_controls(page_id: str) -> str:
    """Read the control description from session state without rendering.

    Streamlit persists keyed-widget state across reruns, so message assembly can
    read the value before the widget itself is instantiated further down the
    script. Defaults to ``""`` before the input has ever been rendered.
    """
    return str(st.session_state.get(f"{page_id}_controls", "") or "")


def append_controls(user_content: str, controls: str) -> str:
    """Append the control overlay to ``user_content`` when ``controls`` is non-empty.

    The pure, Streamlit-free core of ``apply_controls`` — takes an explicit
    string so headless callers (the MCP server, ``core.prompts``) can reuse the
    exact framing. Whitespace-only descriptions are treated as empty.
    """
    controls = controls.strip()
    if not controls:
        return user_content
    return user_content + CONTROLS_PROMPT_TEMPLATE.format(controls=controls)


def apply_controls(user_content: str, page_id: str) -> str:
    """Append the control overlay when a description is present in session state."""
    return append_controls(user_content, get_controls(page_id))


def controls_trace_tags(base_tags: tuple[str, ...], page_id: str) -> tuple[str, ...]:
    """Add a ``control_overlay`` LangSmith tag when a control description is set."""
    if get_controls(page_id).strip():
        return base_tags + ("control_overlay",)
    return base_tags
