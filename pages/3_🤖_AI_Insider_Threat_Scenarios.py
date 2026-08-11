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
from core.sidebar import render_setup_blockers, render_setup_sidebar
from core.styles import inject_emoji_fonts
from data.ai_insider_threats import (
    AGENT_CAPABILITIES,
    AI_INSIDER_TEMPLATES,
    DEPLOYMENT_ARCHETYPES,
    THREAT_CATEGORIES,
    resolve_template,
    stride_code_from_option,
    stride_options,
)

# ------------------ Streamlit Configuration ------------------ #

st.set_page_config(page_title="AI Insider Threat Scenarios", page_icon="🤖")
inject_emoji_fonts()
setup = render_setup_sidebar()

industry = setup.industry
company_size = setup.company_size


# ------------------ Prompt Construction ------------------ #
# Prompt text lives in core/prompts.py (shared with the MCP server). This page
# only threads its own inputs into the shared builder.


def build_messages(snapshot):
    """Build messages exclusively from the Generate-time input snapshot."""
    return build_ai_insider_messages(
        archetype_name=snapshot["selected_entity"]["name"],
        selected_categories=snapshot["selected_categories"],
        selected_stride=snapshot["selected_stride"],
        selected_capabilities=snapshot["selected_capabilities"],
        industry=snapshot["organisation"]["industry"],
        company_size=snapshot["organisation"]["company_size"],
        scenario_seed=snapshot["scenario_seed"],
        required_decisions=snapshot["required_decisions"],
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
        "STRIDE threats for a common AI insider threat scenario, along with the decisions the "
        "exercise must force. Most templates leave the narrative to the model, so re-running "
        "one gives a fresh exercise; templates that rehearse a specific incident also fill in "
        "the scenario seed. You can adjust everything afterwards."
    )
    selected_template = st.selectbox(
        "Select a template",
        options=[""] + list(AI_INSIDER_TEMPLATES.keys()),
        format_func=lambda x: "Select a template" if x == "" else x,
    )
    if selected_template:
        template = resolve_template(selected_template)
        st.session_state['ai_insider_archetype'] = template['archetype']
        st.session_state['ai_insider_categories'] = template['categories']
        st.session_state['ai_insider_stride'] = [
            opt for opt in stride_options() if stride_code_from_option(opt) in template['stride']
        ]
        st.session_state['ai_insider_seed'] = template['brief']
        st.session_state['ai_insider_required_decisions'] = template['required_decisions']
    else:
        # Clearing the template drops its forced decisions — unlike the other
        # selections, they have no widget the user could edit them back out of.
        st.session_state['ai_insider_required_decisions'] = []

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

# --- Scenario seed ---
st.markdown("### 5. Scenario Seed (Optional)")
st.markdown(
    "Describe the specific situation you want to rehearse — the deployment, what goes wrong, "
    "and who is affected. The model builds the narrative around it rather than inventing its "
    "own premise. Leave it empty and the model invents a premise from your selections, which "
    "is usually what you want when re-running the same threat profile for a fresh exercise. "
    "Templates that rehearse a specific incident fill this in for you; edit it freely."
)
scenario_seed = st.text_area(
    "Scenario seed:",
    value=st.session_state.get('ai_insider_seed', ''),
    height=200,
    placeholder=(
        "e.g. An overnight evaluation run the team believed was network-isolated, where the "
        "agent's activity reaches a partner's production systems."
    ),
)
st.session_state['ai_insider_seed'] = scenario_seed

# Decisions the exercise must force. Supplied by a template only — there is no
# widget for these, so they persist until another template is chosen.
required_decisions = st.session_state.get('ai_insider_required_decisions', [])
if required_decisions:
    st.caption(
        "This template also requires the exercise to force decisions on: "
        + "; ".join(decision.split(" — ", 1)[0] for decision in required_decisions)
        + "."
    )

st.markdown("")
st.markdown("---")
st.markdown(
    """
    ### Generate a Scenario

    Click the button below to generate an AI insider threat scenario based on your selections.

    Generation often takes 30–50 seconds. Reasoning models and local models can take several minutes, depending on the selected model and hardware. Progress and elapsed time are shown below. ⏱️
    """
)


def _ready() -> bool:
    if not render_setup_blockers(setup):
        return False
    if not selected_categories and not selected_stride:
        st.info("Please select at least one threat category (or specific STRIDE threat) to continue.")
        return False
    return True


def _capture_inputs():
    return {
        "scenario_type": "ai_insider",
        "matrix": None,
        "organisation": {"industry": industry, "company_size": company_size},
        "selected_entity": {
            "type": "deployment_archetype",
            "name": selected_archetype,
        },
        "selected_techniques": [],
        "sampled_techniques": [],
        "selected_categories": list(selected_categories),
        "selected_stride": list(selected_stride),
        "selected_capabilities": list(selected_capabilities),
        "scenario_seed": scenario_seed,
        "required_decisions": list(required_decisions),
        "modifiers": {"template": selected_template or None},
    }


run_scenario_page(
    page_id="ai_insider",
    build_messages=build_messages,
    is_ready=_ready,
    download_name="AttackGen AI Insider Threat.md",
    trace_name="AI Insider Threat Scenario",
    trace_tags=("ai_insider_scenario",),
    capture_inputs=_capture_inputs,
)


# Back button
st.markdown(
    '<a href="/" style="display: inline-block; padding: 5px 20px; color: white; text-align: center; text-decoration: none; font-size: 16px; border-radius: 4px;">⬅️ Back</a>',
    unsafe_allow_html=True,
)
