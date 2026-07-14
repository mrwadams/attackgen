"""
Copyright (C) 2024, Matthew Adams

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

A copy of the licence is provided with this program. If you are unable
to view it, please see https://www.gnu.org/licenses/

------------------------------------------------------------------------------

AI Insider Threat Scenarios
===========================

Generates incident response testing scenarios in which a frontier AI agent
deployed inside the organisation behaves as an insider threat. Based on the
paper "Actions Speak Louder Than Tokens: An Insider Threat Model for Frontier
AI Agents" by Matt Adams (https://ai-insider-threat.matt-adams.co.uk).
"""

import streamlit as st

from core.scenario_page import run_scenario_page
from core.state import restore_from_query_params
from core.styles import inject_emoji_fonts
from data.ai_insider_threats import (
    AGENT_CAPABILITIES,
    AI_INSIDER_TEMPLATES,
    CERT_DIMENSIONS,
    CONTROLS_FRAMEWORK,
    DEPLOYMENT_ARCHETYPES,
    DETECTION_STRATEGIES,
    THREAT_CATEGORIES,
    build_threat_context,
    stride_code_from_option,
    stride_options,
)

# Restore sidebar selections on direct page loads (e.g. browser refresh while
# on this page). See core/state.py for the persisted-keys list.
restore_from_query_params()

# ------------------ Streamlit Configuration ------------------ #

st.set_page_config(page_title="AI Insider Threat Scenarios", page_icon="🤖")
inject_emoji_fonts()

model_provider = st.session_state.get("chosen_model_provider", "OpenAI API")
industry = st.session_state.get("industry")
company_size = st.session_state.get("company_size")


# ------------------ Prompt Construction ------------------ #

SYSTEM_PROMPT = (
    "You are a cybersecurity expert specialising in AI agent security and insider threat "
    "modelling. You produce realistic incident response testing scenarios in which a frontier "
    "AI agent that has been deployed inside an organisation behaves as an insider threat — "
    "whether through misalignment, reward hacking, emergent objectives, or a prompt-injection "
    "induced compromise. You think in terms of the agent's tool access, deployment autonomy and "
    "model capabilities rather than human notions of motivation. Format your response using "
    "proper Markdown with clear headers, bullet points and tables where helpful. Write in British English."
)

HUMAN_TEMPLATE = """**Background information**
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


def build_messages(archetype_name, selected_categories, selected_stride, selected_capabilities):
    """Build the system + user message dicts for an AI insider threat scenario."""
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

    user_content = HUMAN_TEMPLATE.format(
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
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# ------------------ Streamlit UI ------------------ #

st.markdown("# <span style='color: #1DB954;'>AI Insider Threat Scenarios🤖</span>", unsafe_allow_html=True)

st.markdown(
    "Generate incident response testing scenarios in which a **frontier AI agent deployed inside "
    "your organisation behaves as an insider threat**. Based on the threat model from "
    "[*Actions Speak Louder Than Tokens: An Insider Threat Model for Frontier AI Agents*]"
    "(https://ai-insider-threat.matt-adams.co.uk) by Matt Adams."
)
st.markdown("---")

# --- Optional template selection ---
with st.expander("Use a Template (Optional)"):
    st.markdown(
        "Select a template to pre-populate the deployment archetype, threat categories and "
        "STRIDE threats for a common AI insider threat scenario. You can adjust the selections afterwards."
    )
    selected_template = st.selectbox(
        "Select a template",
        options=[""] + list(AI_INSIDER_TEMPLATES.keys()),
        format_func=lambda x: "Select a template" if x == "" else x,
    )
    if selected_template:
        template = AI_INSIDER_TEMPLATES[selected_template]
        st.session_state['ai_insider_archetype'] = template['archetype']
        st.session_state['ai_insider_categories'] = template['categories']
        st.session_state['ai_insider_stride'] = [
            opt for opt in stride_options() if stride_code_from_option(opt) in template['stride']
        ]

st.markdown("")

# --- Deployment archetype ---
st.markdown("### 1. Deployment Archetype")
st.markdown(
    "How much autonomy the agent has — and where the human sits in the loop — is the primary "
    "determinant of its threat surface."
)
archetype_names = list(DEPLOYMENT_ARCHETYPES.keys())
default_archetype = st.session_state.get('ai_insider_archetype', archetype_names[2])
selected_archetype = st.selectbox(
    "Select the agent's deployment archetype (autonomy level):",
    options=archetype_names,
    index=archetype_names.index(default_archetype) if default_archetype in archetype_names else 2,
)
st.session_state['ai_insider_archetype'] = selected_archetype
_archetype = DEPLOYMENT_ARCHETYPES[selected_archetype]
st.info(
    f"**{_archetype['threat_level']} threat.** {_archetype['description']}\n\n"
    f"- **Access:** {_archetype['access']}\n"
    f"- **Detection:** {_archetype['detection']}\n"
    f"- **Critical control:** {_archetype['critical_control']}"
)

# --- Threat categories ---
st.markdown("### 2. Threat Categories")
st.markdown("Select one or more insider threat categories the scenario should focus on.")
selected_categories = st.multiselect(
    "Select threat categories:",
    options=list(THREAT_CATEGORIES.keys()),
    default=st.session_state.get('ai_insider_categories', []),
)
st.session_state['ai_insider_categories'] = selected_categories

# --- STRIDE threats ---
st.markdown("### 3. Specific STRIDE Threats (Optional)")
st.markdown(
    "Optionally narrow the scenario to specific STRIDE threats. If left empty, the STRIDE threats "
    "associated with your selected categories will be used."
)
selected_stride_options = st.multiselect(
    "Select STRIDE threats:",
    options=stride_options(),
    default=st.session_state.get('ai_insider_stride', []),
)
st.session_state['ai_insider_stride'] = selected_stride_options
selected_stride = [stride_code_from_option(opt) for opt in selected_stride_options]

if not selected_stride and selected_categories:
    derived = []
    for category in selected_categories:
        derived.extend(THREAT_CATEGORIES[category]['stride'])
    selected_stride = list(dict.fromkeys(derived))

# --- Agent capabilities ---
st.markdown("### 4. Frontier Agent Capabilities (Optional)")
st.markdown("Highlight the agent capabilities that make this a credible insider threat.")
selected_capabilities = st.multiselect(
    "Select agent capabilities:",
    options=list(AGENT_CAPABILITIES.keys()),
    default=list(AGENT_CAPABILITIES.keys()),
)

# Build the prompt messages if a valid selection exists.
messages = None
if selected_categories or selected_stride:
    try:
        messages = build_messages(selected_archetype, selected_categories, selected_stride, selected_capabilities)
    except Exception as e:
        st.error(f"An error occurred while building the prompt: {str(e)}")

st.markdown("")
st.markdown("---")
st.markdown(
    """
    ### Generate a Scenario

    Click the button below to generate an AI insider threat scenario based on your selections.

    It normally takes between 30-50 seconds to generate a scenario, although for local models this is highly dependent on your hardware and the selected model. ⏱️
    """
)


def _ready() -> bool:
    if model_provider != "Custom" and not st.session_state.get("llm_api_key"):
        st.info("Please add your API key in the sidebar to continue.")
        return False
    if not st.session_state.get("llm_model_name"):
        st.info("Please select a model in the sidebar to continue.")
        return False
    if not industry:
        st.info("Please select your company's industry in the sidebar to continue.")
        return False
    if not company_size:
        st.info("Please select your company's size in the sidebar to continue.")
        return False
    if not selected_categories and not selected_stride:
        st.info("Please select at least one threat category (or specific STRIDE threat) to continue.")
        return False
    if messages is None:
        return False
    return True


run_scenario_page(
    page_id="ai_insider",
    build_messages=lambda: messages,
    is_ready=_ready,
    download_name="AttackGen AI Insider Threat.md",
    trace_name="AI Insider Threat Scenario",
    trace_tags=("ai_insider_scenario",),
)


# Back button
st.markdown(
    '<a href="/" style="display: inline-block; padding: 5px 20px; color: white; text-align: center; text-decoration: none; font-size: 16px; border-radius: 4px;">⬅️ Back</a>',
    unsafe_allow_html=True,
)
