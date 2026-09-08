"""The app's page registry — one place that knows where each page lives.

Streamlit identifies a page by its script path, so any cross-page link needs
that path as a literal. Keeping the three generation pages and the Assistant in
one registry means the scenario coordinator can offer "Open in Assistant", the
Assistant can offer "Back to scenario", and neither has to hard-code the other's
filename (emoji and all) at the call site.

``page_id`` is the same identifier the scenario coordinator namespaces its
session-state keys with, so a persisted result can always be traced back to the
page that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PageInfo:
    """One navigable page: its coordinator id, script path and labels."""

    page_id: str
    path: str
    label: str
    icon: str
    description: str


THREAT_GROUP_PAGE = PageInfo(
    page_id="threat_group",
    path="pages/1_🛡️_Threat_Group_Scenarios.py",
    label="Threat Group Scenarios",
    icon="🛡️",
    description="Build a scenario from a MITRE threat actor group or ATLAS case study.",
)

CUSTOM_PAGE = PageInfo(
    page_id="custom",
    path="pages/2_🛠️_Custom_Scenarios.py",
    label="Custom Scenarios",
    icon="🛠️",
    description="Build a scenario from your own selection of ATT&CK / ATLAS techniques.",
)

AI_INSIDER_PAGE = PageInfo(
    page_id="ai_insider",
    path="pages/3_🤖_AI_Insider_Threat_Scenarios.py",
    label="AI Insider Threat Scenarios",
    icon="🤖",
    description="Rehearse a frontier AI agent acting as an insider threat.",
)

ASSISTANT_PAGE = PageInfo(
    page_id="assistant",
    path="pages/4_💬_AttackGen_Assistant.py",
    label="AttackGen Assistant",
    icon="💬",
    description="Ask questions about, and refine, a generated scenario.",
)

SCENARIO_PAGES: tuple[PageInfo, ...] = (
    THREAT_GROUP_PAGE,
    CUSTOM_PAGE,
    AI_INSIDER_PAGE,
)
"""Every page that can produce a scenario, in navigation order."""

_BY_ID = {page.page_id: page for page in SCENARIO_PAGES + (ASSISTANT_PAGE,)}


def page_info(page_id: str) -> PageInfo | None:
    """Look up a page by its coordinator id, or ``None`` if it isn't one."""
    return _BY_ID.get(page_id)
