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

from core.prompts import build_ai_insider_messages
from core.scenario_page import run_scenario_page
from core.state import restore_from_query_params
from core.styles import inject_emoji_fonts
from data.ai_insider_threats import (
    AGENT_CAPABILITIES,
    AI_INSIDER_TEMPLATES,
    DEPLOYMENT_ARCHETYPES,
    THREAT_CATEGORIES,
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
# Prompt text lives in core/prompts.py (shared with the MCP server). This page
# only threads its own inputs into the shared builder.


def build_messages(archetype_name, selected_categories, selected_stride, selected_capabilities):
    """Build the system + user message dicts for an AI insider threat scenario."""
    return build_ai_insider_messages(
        archetype_name=archetype_name,
        selected_categories=selected_categories,
        selected_stride=selected_stride,
        selected_capabilities=selected_capabilities,
        industry=industry,
        company_size=company_size,
    )


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
