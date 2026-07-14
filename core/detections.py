"""Purple-team "Detection & Response" companion for generated scenarios.

Every scenario is built from a concrete set of technique IDs. Those same IDs are
the join key into the defensive half of the STIX bundle that AttackGen already
ships but never surfaces: ATT&CK v18+ *detection strategies* and *analytics*,
the *log sources* they read, and the *mitigations* that reduce the technique.
Joining them locally turns a red-only narrative into a purple-team exercise —
"here's the attack, and here's exactly what your SOC should have seen" — with no
LLM call and no external lookup for the deterministic section.

Two layers ship here:
  1. ``build_defense_report`` + ``defense_to_markdown`` — the deterministic,
     always-on join. Enterprise/ICS resolve full detection strategies via
     ``mitreattack-python``; ATLAS has no detection model, so it degrades to
     mitigations only.
  2. An optional LLM "purple narrative" pass (``build_narrative_messages``) that
     weaves the *supplied* detections/mitigations into a stage-by-stage
     defender's walkthrough of the scenario. Opt-in via a per-page toggle.

The join helper return shapes differ by object: detection strategies and
mitigations arrive as ``{"object": <stix2 obj>, ...}`` (attribute access);
analytics arrive as plain dicts (key access). ``_field`` papers over both.
"""

from __future__ import annotations

import streamlit as st

# Per-page toggle for the optional LLM narrative pass. The deterministic section
# is always on (it's free); the narrative costs a second model call, so it's
# opt-in — mirroring the AI-uplift toggle in core/ai_uplift.py.
DEFENSE_NARRATIVE_LABEL = "🟣 Purple-team narrative"
DEFENSE_NARRATIVE_HELP = (
    "After the scenario, make a second model call that walks it stage by stage "
    "from the defender's side — what telemetry should fire, what the SOC would "
    "see, and how to respond — grounded strictly in the MITRE detection "
    "strategies, analytics and mitigations for the scenario's techniques. Adds "
    "~20-40s and one extra model call."
)


def render_defense_narrative_toggle(page_id: str) -> bool:
    """Render the purple-team narrative toggle for a scenario page."""
    return st.toggle(
        DEFENSE_NARRATIVE_LABEL,
        key=f"{page_id}_defense_narrative",
        help=DEFENSE_NARRATIVE_HELP,
    )


def is_defense_narrative_on(page_id: str) -> bool:
    """Read the narrative toggle from session state without rendering it.

    Streamlit persists keyed-widget state across reruns, so the page can read
    the value at ``run_scenario_page`` call time even though the toggle widget
    is rendered later in the script. Defaults to ``False``.
    """
    return bool(st.session_state.get(f"{page_id}_defense_narrative", False))


# ---------------------------------------------------------------------------
# Deterministic join: technique IDs -> defensive STIX data
# ---------------------------------------------------------------------------


def _field(obj, name):
    """Read ``name`` from a STIX object that may be a stix2 object or a dict."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _analytic_log_sources(analytic) -> list[dict]:
    """Structured log sources an analytic reads: a ``name`` + optional ``channel``.

    MITRE stores a literal string ``"None"`` in the channel field for sources
    with no specific channel; treat that (and blanks) as absent so it never
    renders as an ugly ``"(None)"``.
    """
    sources: list[dict] = []
    for ref in _field(analytic, "x_mitre_log_source_references") or []:
        name = ref.get("name")
        if not name:
            continue
        channel = ref.get("channel")
        if not channel or str(channel).strip().lower() == "none":
            channel = None
        sources.append({"name": name, "channel": channel})
    return sources


def _format_log_source(source: dict) -> str:
    """Render one structured log source as ``"name (channel)"`` or ``"name"``."""
    return f"{source['name']} ({source['channel']})" if source["channel"] else source["name"]


def _enterprise_technique(data, external_id: str) -> dict | None:
    """Join one Enterprise/ICS technique to its detection strategies + mitigations."""
    obj = data.get_object_by_attack_id(external_id, "attack-pattern")
    if obj is None:
        return None

    strategies = []
    for item in data.get_detection_strategies_detecting_technique(obj.id):
        strategy = item["object"]
        analytics = [
            {
                "name": _field(analytic, "name") or "",
                "platforms": _field(analytic, "x_mitre_platforms") or [],
                "log_sources": _analytic_log_sources(analytic),
                "description": _field(analytic, "description") or "",
            }
            for analytic in data.get_analytics_by_detection_strategy(strategy.id)
        ]
        strategies.append(
            {
                "id": data.get_attack_id(strategy.id) or "",
                "name": _field(strategy, "name") or "",
                "analytics": analytics,
            }
        )

    mitigations = [
        {
            "id": data.get_attack_id(item["object"].id) or "",
            "name": _field(item["object"], "name") or "",
            "description": _field(item["object"], "description") or "",
        }
        for item in data.get_mitigations_mitigating_technique(obj.id)
    ]

    return {
        "id": external_id,
        "name": _field(obj, "name") or external_id,
        "detection_strategies": strategies,
        "mitigations": mitigations,
    }


def _atlas_technique(atlas_data, external_id: str) -> dict | None:
    """Join one ATLAS technique to its mitigations (ATLAS has no detections)."""
    tech = atlas_data.get_technique_by_id(external_id)
    if not tech:
        return None
    mitigations = [
        {
            "id": mitigation["external_id"],
            "name": mitigation["name"],
            "description": mitigation.get("description", ""),
        }
        for mitigation in atlas_data.get_mitigations_mitigating_technique(external_id)
    ]
    return {
        "id": external_id,
        "name": tech.get("name", external_id),
        "detection_strategies": [],
        "mitigations": mitigations,
    }


def _rollup_log_sources(techniques: list[dict]) -> list[str]:
    """Distinct log-source *names* across every analytic, first-seen order.

    Deduped by source name (not name+channel), so the roll-up reads as an
    actionable "enable these" checklist rather than a wall of near-duplicates —
    e.g. ``WinEventLog:Sysmon`` appears once, not once per EventCode. The
    per-channel specifics stay in each technique's analytic lines.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for technique in techniques:
        for strategy in technique["detection_strategies"]:
            for analytic in strategy["analytics"]:
                for source in analytic["log_sources"]:
                    name = source["name"]
                    if name not in seen:
                        seen.add(name)
                        ordered.append(name)
    return ordered


def build_defense_report(
    *,
    matrix: str,
    technique_ids: list[str],
    mitre_data=None,
    atlas_data=None,
) -> dict | None:
    """Build the structured defensive companion for a scenario's techniques.

    ``technique_ids`` are ATT&CK/ATLAS external IDs (e.g. ``"T1059"``,
    ``"AML.T0051"``); duplicates and blanks are dropped. Enterprise/ICS resolve
    detection strategies + mitigations via a ``MitreAttackData`` instance
    (``mitre_data``); ATLAS resolves mitigations via an ``ATLASData`` instance
    (``atlas_data``). Returns ``None`` when there's nothing to show — an unknown
    matrix, no IDs, or no defensive data — so callers can skip the section.
    """
    seen: set[str] = set()
    ids = [
        tid for tid in technique_ids if tid and not (tid in seen or seen.add(tid))
    ]
    if not ids:
        return None

    if matrix in ("Enterprise", "ICS"):
        if mitre_data is None:
            return None
        resolved = [_enterprise_technique(mitre_data, tid) for tid in ids]
    elif matrix == "ATLAS":
        if atlas_data is None:
            return None
        resolved = [_atlas_technique(atlas_data, tid) for tid in ids]
    else:
        return None

    techniques = [t for t in resolved if t]
    if not techniques:
        return None

    return {
        "matrix": matrix,
        "techniques": techniques,
        "log_sources": _rollup_log_sources(techniques),
    }


def defense_to_markdown(report: dict) -> str:
    """Render a defense report as the deterministic "Detection & Response" section."""
    lines = ["## 🛡️ Detection & Response", ""]
    if report["matrix"] == "ATLAS":
        lines += [
            "_MITRE ATLAS defines mitigations for these techniques but does not "
            + "yet publish detection strategies, so only mitigations are listed._",
            "",
        ]

    log_sources = report.get("log_sources") or []
    if log_sources:
        lines += ["### Log sources to enable", ""]
        lines += [f"- {source}" for source in log_sources]
        lines.append("")

    for technique in report["techniques"]:
        lines += [f"### {technique['name']} ({technique['id']})", ""]

        if technique["detection_strategies"]:
            lines += ["**Detection strategies**", ""]
            for strategy in technique["detection_strategies"]:
                suffix = f" ({strategy['id']})" if strategy["id"] else ""
                lines.append(f"- {strategy['name']}{suffix}")
                for analytic in strategy["analytics"]:
                    platforms = ", ".join(analytic["platforms"])
                    sources = "; ".join(
                        _format_log_source(s) for s in analytic["log_sources"]
                    )
                    detail = " — ".join(x for x in (platforms, sources) if x)
                    tail = f" — {detail}" if detail else ""
                    lines.append(f"    - {analytic['name']}{tail}")
            lines.append("")

        if technique["mitigations"]:
            lines += ["**Mitigations**", ""]
            for mitigation in technique["mitigations"]:
                suffix = f" ({mitigation['id']})" if mitigation["id"] else ""
                lines.append(f"- {mitigation['name']}{suffix}")
            lines.append("")

        if not technique["detection_strategies"] and not technique["mitigations"]:
            lines += [
                "_No ATT&CK detection strategies or mitigations are listed for "
                + "this technique._",
                "",
            ]

    return "\n".join(lines).strip() + "\n"


def defense_download_name(md_name: str) -> str:
    """Derive the companion download filename from the scenario's.

    ``"threat_group_20260714.md"`` -> ``"threat_group_20260714_detection.md"``.
    """
    stem = md_name[:-3] if md_name.endswith(".md") else md_name
    return f"{stem}_detection.md"


def assemble_defense_document(
    deterministic_md: str, narrative_md: str | None, *, title: str
) -> str:
    """Combine the narrative (if any) and the deterministic reference for download."""
    parts = [f"# Detection & Response — {title}", ""]
    if narrative_md:
        parts += [narrative_md.strip(), "", "---", "", "## Detection & Response Reference", ""]
    parts.append(deterministic_md.strip())
    return "\n".join(parts).strip() + "\n"


# ---------------------------------------------------------------------------
# Optional LLM narrative pass
# ---------------------------------------------------------------------------

DEFENSE_NARRATIVE_SYSTEM = (
    "You are a purple-team lead. You translate a red-team incident-response "
    "scenario into a defender's-eye 'Detection & Response' walkthrough, grounded "
    "strictly in the MITRE ATT&CK/ATLAS detection strategies, analytics, log "
    "sources and mitigations you are given. Format your response as Markdown."
)

DEFENSE_NARRATIVE_TEMPLATE = """Below is an incident-response scenario, followed by the MITRE detection and mitigation data for the techniques it uses.

Write a purple-team **Detection & Response walkthrough** that follows the scenario stage by stage. For each stage explain:
- what telemetry or analytics should fire — name the specific analytics and log sources from the data below, and do not invent detections that are not listed;
- what the SOC analyst would plausibly see, and how they would triage and confirm it;
- the response and mitigation actions, referencing the specific mitigations by name and ID from the data below.

Ground every detection and mitigation claim in the supplied data. Where the data lists no detection for a stage (for example ATLAS techniques, which have no detection strategies), say so plainly and fall back to the listed mitigations. Write in British English. Start at heading level 2 (`##`). Do not repeat the scenario back verbatim.

---
**Scenario:**
{scenario}

---
**MITRE detection & mitigation data:**
{detection_data}
"""


def build_narrative_messages(scenario_text: str, report: dict) -> list[dict]:
    """Build the chat messages for the optional purple-team narrative pass.

    The deterministic markdown *is* the grounding data handed to the model, so
    the narrative can only cite detections and mitigations that actually exist
    in the bundle.
    """
    user_content = DEFENSE_NARRATIVE_TEMPLATE.format(
        scenario=scenario_text,
        detection_data=defense_to_markdown(report),
    )
    return [
        {"role": "system", "content": DEFENSE_NARRATIVE_SYSTEM},
        {"role": "user", "content": user_content},
    ]
