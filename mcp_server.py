"""AttackGen MCP server — expose scenario generation to agentic workflows.

Two tiers of tools, one FastMCP server:

* **Tier A — data tools** (no LLM call, no API key). They surface AttackGen's
  MITRE data: threat groups / ATLAS case studies, a resolved kill chain, the
  purple-team Detection & Response join, an ATT&CK Navigator layer, and the AI
  insider-threat option catalogue. Each returns structured data and — where
  useful — a ready-to-run ``messages`` prompt, so a *client's* model can generate
  the scenario with no key on the server side. These are safe to host over HTTP.

* **Tier B — generate tools** (bring-your-own-key). They call an LLM
  server-side via ``core.llm.call_llm`` and return a finished Markdown scenario.
  The provider/model/key are passed as tool arguments; when ``api_key`` is
  omitted, LiteLLM falls back to the provider's env var (``OPENAI_API_KEY`` etc.)
  — i.e. the *caller's own* key. Keep these on local stdio; do not expose them on
  a shared host, since they read credentials from the process environment.

Run locally over stdio (default). Register in Claude Code with:
    claude mcp add attackgen -- python -m mcp_server

The heavy STIX bundles load lazily on first data-tool call (see
``core.attack_data``), so the stdio handshake is instant.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Literal

from mcp.server.fastmcp import FastMCP

from core import attack_data as ad
from core.detections import build_defense_report, defense_to_markdown
from core.llm import call_llm
from core.models import PROVIDERS, get_models_for_provider, get_provider
from core.navigator import build_layer, dumps, parse_technique_id
from core.prompts import (
    build_ai_insider_messages,
    build_campaign_messages,
    build_custom_messages,
    build_threat_group_messages,
)
from core.response import clean_model_response
from core.schemas import LLMConfig
from data import ai_insider_threats as aip

mcp = FastMCP("attackgen")

Matrix = Literal["Enterprise", "ICS", "ATLAS"]

# Provider/model defaults for the zero-arg happy path on the generate tools.
_DEFAULT_PROVIDER = os.getenv("ATTACKGEN_PROVIDER", "OpenAI API")
_DEFAULT_MODEL = os.getenv("ATTACKGEN_MODEL", "")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _resolve_kill_chain(matrix: str, group: str, seed: int | None) -> ad.KillChain:
    """Resolve a threat group (ATT&CK) or case study (ATLAS) to a kill chain."""
    if matrix == "ATLAS":
        return ad.resolve_case_study_kill_chain(group)
    return ad.resolve_threat_group_kill_chain(matrix, group, seed=seed)


def _defense_markdown(matrix: str, technique_ids: list[str]) -> str | None:
    """Build the Detection & Response Markdown for a set of technique IDs, if any."""
    if matrix == "ATLAS":
        report = build_defense_report(
            matrix=matrix, technique_ids=technique_ids, atlas_data=ad.atlas_data()
        )
    else:
        report = build_defense_report(
            matrix=matrix,
            technique_ids=technique_ids,
            mitre_data=ad.mitre_data_for_matrix(matrix),
        )
    return defense_to_markdown(report) if report else None


def _normalise_ids(techniques: list[str]) -> list[str]:
    """Accept ``"Name (ID)"`` display labels or bare IDs; return bare IDs."""
    return [parse_technique_id(t) or t.strip() for t in techniques if t and t.strip()]


def _make_config(
    provider: str,
    model: str,
    *,
    api_key: str | None,
    api_base: str | None,
    trace_name: str,
    trace_tags: tuple[str, ...],
) -> LLMConfig:
    """Build a validated LLMConfig for a generate tool.

    Validates the provider (and, except for Custom, the model) against the
    registry in ``core/models.py`` so a typo fails loudly instead of producing a
    confusing LiteLLM error. ``api_key=None`` leaves LiteLLM to read the
    provider's env var — the caller's own key.
    """
    if get_provider(provider) is None:
        raise ValueError(f"Unknown provider '{provider}'. Valid: {sorted(PROVIDERS)}")
    if not model:
        raise ValueError("A model name is required (or set ATTACKGEN_MODEL).")
    if provider != "Custom":
        valid = {m.model_id for m in get_models_for_provider(provider)}
        if model not in valid:
            raise ValueError(
                f"Model '{model}' is not registered for {provider}. Valid: {sorted(valid)}"
            )
    return LLMConfig(
        provider=provider,
        model_name=model,
        api_key=api_key or None,
        api_base=api_base or None,
        trace_name=trace_name,
        trace_tags=trace_tags,
    )


def _generate(config: LLMConfig, messages: list[dict]) -> str:
    """Call the model and return the cleaned scenario (thinking stripped)."""
    raw = call_llm(config, messages)
    _thinking, cleaned = clean_model_response(raw)
    return cleaned


def _maybe_controls(base_tags: tuple[str, ...], controls: str) -> tuple[str, ...]:
    """Add a ``control_overlay`` trace tag when a control description is supplied."""
    return base_tags + ("control_overlay",) if controls.strip() else base_tags


# ===========================================================================
# Tier A — data tools (no LLM, no API key)
# ===========================================================================


@mcp.tool()
def list_threat_groups(matrix: Matrix = "Enterprise") -> list[dict]:
    """List selectable threat actor groups (or ATLAS case studies).

    Returns ``[{"group", "url"}]`` for Enterprise/ICS threat groups, or the ATLAS
    case studies when ``matrix="ATLAS"``. Use a ``group`` value from here with
    ``get_kill_chain`` or ``generate_threat_group_scenario``.
    """
    return list(ad.list_threat_groups(matrix))


@mcp.tool()
def list_case_studies() -> list[dict]:
    """List MITRE ATLAS case studies with a short summary.

    Returns ``[{"group", "url", "summary", "case_study_type"}]``. The ``group``
    field is the case study name to pass to the ATLAS scenario tools.
    """
    return list(ad.list_case_studies())


@mcp.tool()
def list_campaigns(matrix: Matrix = "Enterprise") -> list[dict]:
    """List documented MITRE ATT&CK campaigns (Enterprise/ICS only).

    Returns ``[{"group", "url"}]`` — real-world, documented intrusions. Use a
    ``group`` value from here with ``generate_campaign_scenario``. Campaigns do
    not exist in the ATLAS matrix.
    """
    return list(ad.list_campaigns(matrix))


@mcp.tool()
def get_kill_chain(
    matrix: Matrix,
    group: str,
    seed: int | None = None,
    industry: str | None = None,
    company_size: str | None = None,
    controls: str = "",
) -> dict:
    """Resolve a threat group / case study to its kill chain (no LLM call).

    For Enterprise/ICS one technique is sampled per kill-chain phase (pass
    ``seed`` for a deterministic draw); ATLAS uses the case study's full
    documented procedure. When both ``industry`` and ``company_size`` are given,
    a ready-to-run ``messages`` prompt is included so a client model can generate
    the scenario directly; pass ``controls`` (a description of the org's security
    controls) to have that prompt assess the chain against them. Returns
    ``techniques``, ``all_techniques``, ``kill_chain_string`` and (optionally)
    ``messages``.
    """
    kc = _resolve_kill_chain(matrix, group, seed)
    result = asdict(kc)
    if industry and company_size and kc.techniques:
        result["messages"] = build_threat_group_messages(
            matrix=matrix,
            selected_group_alias=group,
            kill_chain_string=kc.kill_chain_string,
            industry=industry,
            company_size=company_size,
            controls=controls,
        )
    else:
        result["messages"] = None
    return result


@mcp.tool()
def get_detection_report(matrix: Matrix, technique_ids: list[str]) -> dict:
    """Get the purple-team Detection & Response join for a set of techniques.

    Joins the techniques to MITRE's detection strategies/analytics (Enterprise/
    ICS, ATT&CK v18+) and mitigations — no LLM call. ``technique_ids`` may be bare
    IDs (``"T1059"``, ``"AML.T0051"``) or ``"Name (ID)"`` labels. Returns
    ``{"matrix", "markdown"}``; ``markdown`` is ``None`` when there's no defensive
    data (e.g. ATLAS, which has mitigations only).
    """
    ids = _normalise_ids(technique_ids)
    return {"matrix": matrix, "markdown": _defense_markdown(matrix, ids)}


@mcp.tool()
def get_navigator_layer(
    matrix: Matrix,
    technique_ids: list[str],
    name: str = "AttackGen scenario",
    description: str = "",
) -> dict:
    """Build an ATT&CK/ATLAS Navigator layer JSON for a set of techniques.

    ``technique_ids`` may be bare IDs or ``"Name (ID)"`` labels. Returns
    ``{"layer_json"}`` (a JSON string ready to upload to the Navigator), or
    ``layer_json=None`` for a matrix with no Navigator domain.
    """
    ids = _normalise_ids(technique_ids)
    layer = build_layer(
        name=name,
        matrix=matrix,
        techniques=[(tid, None) for tid in ids],
        description=description,
    )
    return {"layer_json": dumps(layer) if layer else None}


@mcp.tool()
def list_ai_insider_options() -> dict:
    """List the option catalogue for AI insider-threat scenarios.

    Returns the valid values to pass to ``get_ai_insider_prompt`` /
    ``generate_ai_insider_scenario``: ``archetypes``, ``threat_categories``,
    ``stride`` threats, agent ``capabilities`` and quick-start ``templates``.
    """
    return {
        "archetypes": list(aip.DEPLOYMENT_ARCHETYPES.keys()),
        "threat_categories": list(aip.THREAT_CATEGORIES.keys()),
        "stride": aip.stride_options(),
        "capabilities": list(aip.AGENT_CAPABILITIES.keys()),
        "templates": list(aip.AI_INSIDER_TEMPLATES.keys()),
    }


@mcp.tool()
def get_ai_insider_prompt(
    archetype: str,
    categories: list[str],
    stride: list[str],
    capabilities: list[str],
    industry: str,
    company_size: str,
) -> dict:
    """Build the AI insider-threat prompt (no LLM call).

    Returns ``{"messages"}`` — a ready-to-run [system, user] prompt a client model
    can generate from. Use ``list_ai_insider_options`` for valid argument values.
    """
    messages = build_ai_insider_messages(
        archetype_name=archetype,
        selected_categories=categories,
        selected_stride=stride,
        selected_capabilities=capabilities,
        industry=industry,
        company_size=company_size,
    )
    return {"messages": messages}


# ===========================================================================
# Tier B — generate tools (bring-your-own-key; local stdio only)
# ===========================================================================


@mcp.tool()
def generate_threat_group_scenario(
    matrix: Matrix,
    group: str,
    industry: str,
    company_size: str,
    provider: str = _DEFAULT_PROVIDER,
    model: str = _DEFAULT_MODEL,
    api_key: str | None = None,
    api_base: str | None = None,
    ai_uplift: bool = False,
    controls: str = "",
    seed: int | None = None,
    include_detection: bool = False,
) -> str:
    """Generate a full incident-response scenario for a threat group / case study.

    Resolves the kill chain, calls the model (BYO-key), and returns finished
    Markdown. ``api_key`` omitted → the provider's env var is used. ``ai_uplift``
    adds the AI-enhanced-adversary framing; ``controls`` (a description of the
    org's security controls) makes the scenario assess the chain against them —
    what would be blocked, detected or missed; ``include_detection`` appends the
    purple-team Detection & Response section.
    """
    kc = _resolve_kill_chain(matrix, group, seed)
    if not kc.techniques:
        raise ValueError(f"No techniques found for '{group}' in the {matrix} matrix.")
    messages = build_threat_group_messages(
        matrix=matrix,
        selected_group_alias=group,
        kill_chain_string=kc.kill_chain_string,
        industry=industry,
        company_size=company_size,
        ai_uplift=ai_uplift,
        controls=controls,
    )
    config = _make_config(
        provider, model, api_key=api_key, api_base=api_base,
        trace_name="Threat Group Scenario (MCP)",
        trace_tags=_maybe_controls(("threat_group_scenario", "mcp"), controls),
    )
    scenario = _generate(config, messages)
    if include_detection:
        detection = _defense_markdown(matrix, [t["ATT&CK ID"] for t in kc.techniques])
        if detection:
            scenario = f"{scenario}\n\n---\n\n{detection}"
    return scenario


@mcp.tool()
def generate_campaign_scenario(
    matrix: Literal["Enterprise", "ICS"],
    campaign: str,
    industry: str,
    company_size: str,
    provider: str = _DEFAULT_PROVIDER,
    model: str = _DEFAULT_MODEL,
    api_key: str | None = None,
    api_base: str | None = None,
    ai_uplift: bool = False,
    controls: str = "",
    include_detection: bool = False,
) -> str:
    """Generate a scenario built around a real, documented ATT&CK campaign.

    Unlike a threat group (whose techniques are sampled), a campaign replays the
    *full* set of techniques observed in the actual intrusion. Campaigns exist
    only in the Enterprise and ICS matrices. Use a ``campaign`` value from
    ``list_campaigns``. Calls the model (BYO-key) and returns finished Markdown.
    ``controls`` makes the scenario assess the chain against your stated defences;
    ``include_detection`` appends the purple-team Detection & Response section.
    """
    kc = ad.resolve_campaign_kill_chain(matrix, campaign)
    if not kc.techniques:
        raise ValueError(f"No techniques found for campaign '{campaign}' in the {matrix} matrix.")
    messages = build_campaign_messages(
        matrix=matrix,
        campaign_name=campaign,
        kill_chain_string=kc.kill_chain_string,
        industry=industry,
        company_size=company_size,
        ai_uplift=ai_uplift,
        controls=controls,
    )
    config = _make_config(
        provider, model, api_key=api_key, api_base=api_base,
        trace_name="Campaign Scenario (MCP)",
        trace_tags=_maybe_controls(("campaign_scenario", "mcp"), controls),
    )
    scenario = _generate(config, messages)
    if include_detection:
        detection = _defense_markdown(matrix, [t["ATT&CK ID"] for t in kc.techniques])
        if detection:
            scenario = f"{scenario}\n\n---\n\n{detection}"
    return scenario


@mcp.tool()
def generate_custom_scenario(
    matrix: Matrix,
    techniques: list[str],
    industry: str,
    company_size: str,
    provider: str = _DEFAULT_PROVIDER,
    model: str = _DEFAULT_MODEL,
    api_key: str | None = None,
    api_base: str | None = None,
    template_info: str = "",
    ai_uplift: bool = False,
    controls: str = "",
    include_detection: bool = False,
) -> str:
    """Generate a custom scenario from a chosen set of ATT&CK / ATLAS techniques.

    ``techniques`` may be bare IDs or ``"Name (ID)"`` labels. Calls the model
    (BYO-key) and returns finished Markdown. ``controls`` (a description of the
    org's security controls) makes the scenario assess the chain against them;
    ``include_detection`` appends the purple-team Detection & Response section.
    """
    if not techniques:
        raise ValueError("At least one technique is required.")
    selected_techniques_string = "\n".join(techniques)
    messages = build_custom_messages(
        matrix=matrix,
        selected_techniques_string=selected_techniques_string,
        template_info=template_info,
        industry=industry,
        company_size=company_size,
        ai_uplift=ai_uplift,
        controls=controls,
    )
    config = _make_config(
        provider, model, api_key=api_key, api_base=api_base,
        trace_name="Custom Scenario (MCP)",
        trace_tags=_maybe_controls(("custom_scenario", "mcp"), controls),
    )
    scenario = _generate(config, messages)
    if include_detection:
        detection = _defense_markdown(matrix, _normalise_ids(techniques))
        if detection:
            scenario = f"{scenario}\n\n---\n\n{detection}"
    return scenario


@mcp.tool()
def generate_ai_insider_scenario(
    archetype: str,
    categories: list[str],
    stride: list[str],
    capabilities: list[str],
    industry: str,
    company_size: str,
    provider: str = _DEFAULT_PROVIDER,
    model: str = _DEFAULT_MODEL,
    api_key: str | None = None,
    api_base: str | None = None,
) -> str:
    """Generate an AI insider-threat tabletop scenario (BYO-key).

    Models a frontier AI agent deployed inside the organisation acting as an
    insider threat. Use ``list_ai_insider_options`` for valid argument values.
    Returns finished Markdown.
    """
    messages = build_ai_insider_messages(
        archetype_name=archetype,
        selected_categories=categories,
        selected_stride=stride,
        selected_capabilities=capabilities,
        industry=industry,
        company_size=company_size,
    )
    config = _make_config(
        provider, model, api_key=api_key, api_base=api_base,
        trace_name="AI Insider Threat Scenario (MCP)", trace_tags=("ai_insider_scenario", "mcp"),
    )
    return _generate(config, messages)


def main() -> None:
    """Entry point — run the MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
