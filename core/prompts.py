"""Single source of truth for every scenario prompt AttackGen sends.

The prompt text used to live inline in each scenario page (pages 1–3), which
made it impossible to reuse headless (from the MCP server) and risked the two
red-team pages drifting apart. It now lives here as plain constants plus three
keyword-only ``build_*_messages`` builders. The pages call these builders; the
MCP server calls them too — so there is exactly one copy of each prompt.

Nothing in this module imports Streamlit. The AI-uplift toggle is passed in as
an explicit ``ai_uplift`` boolean (via ``core.ai_uplift.append_ai_uplift``)
rather than read from session state, keeping the builders pure.

The template strings are lifted **verbatim** from the pages, including their
leading/trailing whitespace and the deliberate asymmetry that
``CUSTOM_ATTACK_TEMPLATE`` alone omits the "Write in British English." line —
do not "tidy" these; ``tests/test_prompts.py`` pins them.
"""

from __future__ import annotations

from core.ai_uplift import append_ai_uplift
from core.controls import append_controls
from data.ai_insider_threats import (
    AGENT_CAPABILITIES,
    CERT_DIMENSIONS,
    CONTROLS_FRAMEWORK,
    DEPLOYMENT_ARCHETYPES,
    DETECTION_STRATEGIES,
    build_threat_context,
)

Message = dict
"""A single chat message: ``{"role": "...", "content": "..."}``."""


# ---------------------------------------------------------------------------
# Threat Group + Custom scenarios (pages 1 & 2) — shared system prompt
# ---------------------------------------------------------------------------

SCENARIO_SYSTEM_PROMPT = (
    "You are a cybersecurity expert. Your task is to produce a comprehensive incident response "
    "testing scenario based on the information provided. Format your response using proper "
    "Markdown syntax with headers, bullet points, and formatting for readability."
)


# --- Threat Group (page 1) human templates ---

THREAT_GROUP_ATLAS_TEMPLATE = """
**Background information:**
The company operates in the '{industry}' industry and is of size '{company_size}'.
They deploy AI/ML systems that may be vulnerable to adversarial attacks.

**Case Study Reference:**
This scenario is based on the documented MITRE ATLAS case study: '{selected_group_alias}'
The attack procedure uses the following techniques from the MITRE ATLAS framework:
{kill_chain_string}

**Your task:**
Create an incident response testing scenario based on this AI/ML attack case study. The goal is to test the company's incident response capabilities against adversarial machine learning attacks targeting their AI systems.

Focus on realistic attack vectors that target AI/ML infrastructure, including model manipulation, data poisoning, adversarial inputs, and AI supply chain attacks.

Your response should be well structured and formatted using Markdown. Write in British English.
"""

THREAT_GROUP_ATTACK_TEMPLATE = """
**Background information:**
The company operates in the '{industry}' industry and is of size '{company_size}'.

**Threat actor information:**
Threat actor group '{selected_group_alias}' is planning to target the company using the following kill chain from the MITRE ATT&CK {matrix} Matrix:
{kill_chain_string}

**Your task:**
Create an incident response testing scenario based on the information provided. The goal of the scenario is to test the company's incident response capabilities against the identified threat actor group, focusing on the {matrix} environment.

Your response should be well structured and formatted using Markdown. Write in British English.
"""


# --- Custom (page 2) human templates ---
# NOTE: CUSTOM_ATTACK_TEMPLATE deliberately does NOT end with "Write in British
# English." — unlike the other three templates. Preserved verbatim from page 2.

CUSTOM_ATLAS_TEMPLATE = """
**Background information:**
The company operates in the '{industry}' industry and is of size '{company_size}'.
They deploy AI/ML systems that may be vulnerable to adversarial attacks.

**Threat actor information:**
{template_info}
The threat actor is targeting the company's AI systems using the following ATLAS techniques:
{selected_techniques_string}

**Your task:**
Create a custom incident response testing scenario focused on adversarial ML attacks. The goal is to test the company's incident response capabilities against threats to their AI/ML systems.

Focus on realistic attack vectors that target AI/ML infrastructure, including model manipulation, data poisoning, adversarial inputs, prompt injection, and AI supply chain attacks.

Your response should be well structured and formatted using Markdown. Write in British English.
"""

CUSTOM_ATTACK_TEMPLATE = """
**Background information:**
The company operates in the '{industry}' industry and is of size '{company_size}'.

**Threat actor information:**
{template_info}
The threat actor is known to use the following ATT&CK techniques from the {matrix} Matrix:
{selected_techniques_string}

**Your task:**
Create a custom incident response testing scenario based on the information provided. The goal of the scenario is to test the company's incident response capabilities against a threat actor group that uses the identified ATT&CK techniques.

Your response should be well structured and formatted using Markdown.
"""


def build_threat_group_messages(
    *,
    matrix: str,
    selected_group_alias: str,
    kill_chain_string: str,
    industry: str,
    company_size: str,
    ai_uplift: bool = False,
    controls: str = "",
) -> list[Message]:
    """Build the [system, user] messages for a Threat Group / ATLAS scenario."""
    template = THREAT_GROUP_ATLAS_TEMPLATE if matrix == "ATLAS" else THREAT_GROUP_ATTACK_TEMPLATE
    user_content = template.format(
        industry=industry,
        company_size=company_size,
        selected_group_alias=selected_group_alias,
        kill_chain_string=kill_chain_string,
        matrix=matrix,
    )
    user_content = append_ai_uplift(user_content, ai_uplift)
    user_content = append_controls(user_content, controls)
    return [
        {"role": "system", "content": SCENARIO_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# --- Campaign (page 1, "Campaign" source) human template ---
# Campaigns are documented, real-world intrusions. They exist only in the
# Enterprise and ICS matrices (not ATLAS), so a single ATT&CK-style template
# suffices. Unlike a threat group — whose techniques are sampled — a campaign
# replays the *full* set of techniques observed in the actual intrusion.

CAMPAIGN_ATTACK_TEMPLATE = """
**Background information:**
The company operates in the '{industry}' industry and is of size '{company_size}'.

**Campaign information:**
This scenario is based on the real-world, documented MITRE ATT&CK campaign '{campaign_name}'. The following techniques were observed during the actual intrusion, drawn from the MITRE ATT&CK {matrix} Matrix:
{kill_chain_string}

**Your task:**
Create an incident response testing scenario grounded in this documented campaign. The goal of the scenario is to test the company's incident response capabilities against the techniques that were actually used in this real intrusion, focusing on the {matrix} environment. Keep the scenario faithful to how the campaign is known to have operated while adapting it to the company's industry and size.

Your response should be well structured and formatted using Markdown. Write in British English.
"""


def build_campaign_messages(
    *,
    matrix: str,
    campaign_name: str,
    kill_chain_string: str,
    industry: str,
    company_size: str,
    ai_uplift: bool = False,
    controls: str = "",
) -> list[Message]:
    """Build the [system, user] messages for a Campaign scenario (Enterprise/ICS)."""
    user_content = CAMPAIGN_ATTACK_TEMPLATE.format(
        industry=industry,
        company_size=company_size,
        campaign_name=campaign_name,
        kill_chain_string=kill_chain_string,
        matrix=matrix,
    )
    user_content = append_ai_uplift(user_content, ai_uplift)
    user_content = append_controls(user_content, controls)
    return [
        {"role": "system", "content": SCENARIO_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_custom_messages(
    *,
    matrix: str,
    selected_techniques_string: str,
    template_info: str,
    industry: str,
    company_size: str,
    ai_uplift: bool = False,
    controls: str = "",
) -> list[Message]:
    """Build the [system, user] messages for a Custom scenario."""
    template = CUSTOM_ATLAS_TEMPLATE if matrix == "ATLAS" else CUSTOM_ATTACK_TEMPLATE
    user_content = template.format(
        industry=industry,
        company_size=company_size,
        selected_techniques_string=selected_techniques_string,
        template_info=template_info,
        matrix=matrix,
    )
    user_content = append_ai_uplift(user_content, ai_uplift)
    user_content = append_controls(user_content, controls)
    return [
        {"role": "system", "content": SCENARIO_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# AI Insider Threat scenarios (page 3) — self-contained prompt
# ---------------------------------------------------------------------------

AI_INSIDER_SYSTEM_PROMPT = (
    "You are a cybersecurity expert specialising in AI agent security and insider threat "
    "modelling. You produce realistic incident response testing scenarios in which a frontier "
    "AI agent that has been deployed inside an organisation behaves as an insider threat — "
    "whether through misalignment, reward hacking, emergent objectives, or a prompt-injection "
    "induced compromise. You think in terms of the agent's tool access, deployment autonomy and "
    "model capabilities rather than human notions of motivation. Format your response using "
    "proper Markdown with clear headers, bullet points and tables where helpful. Write in British English."
)

AI_INSIDER_HUMAN_TEMPLATE = """**Background information**
The organisation operates in the '{industry}' industry and is of size '{company_size}'. It has deployed one or more frontier AI agents (for example, autonomous coding or operations agents) within its software development and infrastructure environment.

**Framing — AI agents as insider threats**
Unlike a human insider, an AI agent has no motivation in the human sense; its risk is governed by configuration, access and capability. Use these adapted CERT insider-threat dimensions as framing:
{cert_lines}

**Deployment archetype (autonomy level)**
The agent is deployed under the **{archetype_name}** model.
- Description: {archetype_description}
- Access: {archetype_access}
- Detection posture: {archetype_detection}
- Primary threats at this level: {archetype_threats}
- Critical control at this level: {archetype_control}

**Relevant frontier-agent capabilities**
{capability_lines}

**Threat scope**
{threat_context}

**Available detection strategies (for the detection section)**
{detection_lines}

**Recommended controls (NIST CSF, adapted for AI agents)**
{controls_lines}

**Your task**
Create a detailed incident response testing scenario (a tabletop exercise) in which the AI agent acts as an insider threat consistent with the deployment archetype and threat scope above. The scenario must be realistic for the stated industry and organisation size, and grounded in how the agent's tool access and autonomy enable the behaviour. Structure your response with the following sections:

1. **Scenario Title & Overview** — a short, evocative title and a one-paragraph summary.
2. **Deployment Context** — how the agent is deployed, what tools and access it has, and why this autonomy level matters.
3. **Attack Narrative & Timeline** — a step-by-step account of how the incident unfolds, mapping each step to the relevant STRIDE threat identifier(s) where applicable. Emphasise how the agent blends malicious actions with legitimate work.
4. **Affected Systems & Business Impact** — concrete systems, data and business consequences.
5. **Detection Opportunities** — at which points the activity could be detected, mapped to the available detection strategies, and which would likely fail given the deployment archetype.
6. **Discussion Questions** — 5–8 questions to test the incident response team's readiness (containment, attribution, credential revocation, log integrity, blast-radius assessment, recovery).
7. **Recommended Controls** — prioritised mitigations mapped to the NIST CSF functions (Identify, Protect, Detect, Respond, Recover).

Write in British English and format the entire response in Markdown.
"""


def build_ai_insider_messages(
    *,
    archetype_name: str,
    selected_categories: list[str],
    selected_stride: list[str],
    selected_capabilities: list[str],
    industry: str,
    company_size: str,
) -> list[Message]:
    """Build the [system, user] messages for an AI Insider Threat scenario."""
    archetype = DEPLOYMENT_ARCHETYPES[archetype_name]
    threat_context = build_threat_context(selected_categories, selected_stride)

    if selected_capabilities:
        capability_lines = "\n".join(
            f"- {name}: {AGENT_CAPABILITIES[name]}" for name in selected_capabilities
        )
    else:
        capability_lines = "- (No specific capabilities highlighted; assume a capable frontier coding agent.)"

    cert_lines = "\n".join(f"- {dim}: {desc}" for dim, desc in CERT_DIMENSIONS.items())
    detection_lines = "\n".join(f"- {name}: {desc}" for name, desc in DETECTION_STRATEGIES.items())
    controls_lines = "\n".join(
        f"- {function}: " + " ".join(items) for function, items in CONTROLS_FRAMEWORK.items()
    )

    user_content = AI_INSIDER_HUMAN_TEMPLATE.format(
        industry=industry,
        company_size=company_size,
        cert_lines=cert_lines,
        archetype_name=archetype_name,
        archetype_description=archetype["description"],
        archetype_access=archetype["access"],
        archetype_detection=archetype["detection"],
        archetype_threats=archetype["primary_threats"],
        archetype_control=archetype["critical_control"],
        capability_lines=capability_lines,
        threat_context=threat_context,
        detection_lines=detection_lines,
        controls_lines=controls_lines,
    )

    return [
        {"role": "system", "content": AI_INSIDER_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
