"""AI-enhanced ("uplift") framing for conventional attack scenarios.

Based on Anthropic's "LLM ATT&CK Navigator" research
(https://red.anthropic.com/2026/attack-navigator/), which found that the most
significant shift in real-world cyber threats is *not* new techniques but the
acceleration and orchestration of existing ones: low-skill actors operating
expert-level AI harnesses, autonomous kill-chain orchestration, real-time pivot
decisions, and AI-directed execution.

When enabled on the Threat Group and Custom scenario pages, this reframes the
generated incident-response scenario so the adversary uses AI to *accelerate and
enhance the same techniques* — compressing timelines and lowering the skill
floor — rather than introducing novel attack methods. The selected kill chain is
unchanged; only the manner of execution shifts.

The AI Insider Threat page (page 3) deliberately does not use this toggle: there
the AI agent already *is* the threat actor, so an "AI-enhanced adversary" framing
would be redundant.
"""

from __future__ import annotations

import streamlit as st

# Label and help text for the per-page toggle.
AI_UPLIFT_LABEL = "🤖 AI-enhanced adversary"
AI_UPLIFT_HELP = (
    "Reframe the scenario so the adversary uses AI to accelerate and enhance the "
    "same techniques — a lowered skill floor, compressed timelines and autonomous "
    "orchestration of the kill chain — rather than adding new techniques. Based on "
    "Anthropic's LLM ATT&CK Navigator research."
)

# Appended to the user message when the toggle is on. Written to slot cleanly
# after the existing "Your task" block in both scenario templates.
AI_UPLIFT_PROMPT = """
**AI-enhanced adversary framing:**
Assume the threat actor is using AI (for example, an agentic coding assistant or an orchestration harness built on a frontier model) to *accelerate and enhance* this attack. This does **not** introduce new techniques — the kill chain above is unchanged — but it changes how the operation is executed. Reflect the following throughout the scenario:
- **Lowered skill floor:** a relatively low-skill operator can command an expert-level harness, so do not assume the adversary is technically sophisticated in the traditional sense.
- **Compressed timelines:** reconnaissance, tooling, exploit adaptation and lateral movement happen far faster than a human-only team could manage — phases that once took days may take minutes.
- **Autonomous orchestration:** the AI sequences multiple stages of the kill chain with little or no human input, and makes real-time pivot decisions based on what it discovers.
- **Scale:** the adversary can pursue many targets, hosts or variations in parallel.

Weave these characteristics into the narrative and, where relevant, call out the **defensive and incident-response implications** — in particular how AI-driven speed and autonomy shrink detection-and-response windows, and what the response team must do differently to keep pace.
"""


def render_ai_uplift_toggle(page_id: str) -> bool:
    """Render the AI-enhanced adversary toggle for a scenario page.

    The widget is keyed by ``page_id`` so the two scenario pages keep independent
    toggle state. Returns the current boolean value.
    """
    return st.toggle(
        AI_UPLIFT_LABEL,
        key=f"{page_id}_ai_uplift",
        help=AI_UPLIFT_HELP,
    )


def is_ai_uplift_on(page_id: str) -> bool:
    """Read the toggle value from session state without rendering the widget.

    Streamlit persists keyed-widget state across reruns, so message assembly can
    read the value before the widget itself is instantiated further down the
    script. Defaults to ``False`` before the toggle has ever been rendered.
    """
    return bool(st.session_state.get(f"{page_id}_ai_uplift", False))


def apply_ai_uplift(user_content: str, page_id: str) -> str:
    """Append the AI-enhanced framing to ``user_content`` when the toggle is on."""
    if is_ai_uplift_on(page_id):
        return user_content + AI_UPLIFT_PROMPT
    return user_content


def uplift_trace_tags(base_tags: tuple[str, ...], page_id: str) -> tuple[str, ...]:
    """Add an ``ai_enhanced`` LangSmith tag when the toggle is on."""
    if is_ai_uplift_on(page_id):
        return base_tags + ("ai_enhanced",)
    return base_tags
