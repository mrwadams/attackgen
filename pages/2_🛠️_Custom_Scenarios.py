import logging

import pandas as pd
import streamlit as st

from core.ai_uplift import is_ai_uplift_on, render_ai_uplift_toggle, uplift_trace_tags
from core.attack_data import list_technique_options, load_attack_data
from core.prompts import build_custom_messages
from core.detections import (
    is_defense_narrative_on,
    render_defense_narrative_toggle,
    resolve_defense_report,
)
from core.navigator import build_layer, dumps, normalise_technique_ids
from core.scenario_page import run_scenario_page
from core.sidebar import render_setup_sidebar
from core.styles import inject_emoji_fonts

logger = logging.getLogger(__name__)


# ------------------ Streamlit Configuration ------------------ #

st.set_page_config(page_title="Generate Custom Scenario", page_icon="🛠️")
inject_emoji_fonts()
setup = render_setup_sidebar()

industry = setup.industry
company_size = setup.company_size
matrix = setup.matrix or "Enterprise"


# ------------------ Incident Response Templates ------------------ #

incident_response_templates = {
    "Enterprise": {
        "Phishing Attack": ["Spearphishing Attachment (T1193)", "User Execution (T1204)", "Browser Extensions (T1176)", "Credentials from Password Stores (T1555)", "Input Capture (T1056)", "Exfiltration Over C2 Channel (T1041)"],
        "Ransomware Attack": ["Exploit Public-Facing Application (T1190)", "Windows Management Instrumentation (T1047)", "Create Account (T1136)", "Process Injection (T1055)", "Data Encrypted for Impact (T1486)"],
        "Malware Infection": ["Supply Chain Compromise (T1195)", "Command and Scripting Interpreter (T1059)", "Registry Run Keys / Startup Folder (T1060)", "Obfuscated Files or Information (T1027)", "Remote Services (T1021)", "Data Destruction (T1485)"],
        "Insider Threat": ["Valid Accounts (T1078)", "Account Manipulation (T1098)", "Exploitation for Privilege Escalation (T1068)", "Data Staged (T1074)", "Scheduled Transfer (T1029)", "Account Access Removal (T1531)"],
        "Cross-Site Scripting (XSS) Attack": ["Exploit Public-Facing Application (T1190)", "User Execution (T1204)", "Input Capture (T1056)", "Exfiltration Over Web Service (T1567)"],
        "SQL Injection Attack": ["Exploit Public-Facing Application (T1190)", "Exploitation for Credential Access (T1212)", "Exfiltration Over Web Service (T1567)"],
        "API Compromise": ["Active Scanning (T1595)", "Exploit Public-Facing Application (T1190)", "Exploitation for Client Execution (T1203)", "Valid Accounts (T1078)", "Application Layer Protocol (T1071)", "Data Manipulation (T1565)"],
    },
    "ICS": {
        "Remote Access Exploitation": ["External Remote Services (T0822)", "Exploit Public-Facing Application (T0819)", "Remote Services (T0886)", "Wireless Compromise (T0860)"],
        "ICS Data Manipulation": ["Modify Controller Tasking (T0821)", "Manipulate I/O Image (T0835)", "Modify Alarm Settings (T0838)", "Modify Parameter (T0836)"],
        "Denial of Service": ["Denial of Service (T0814)", "Activate Firmware Update Mode (T0800)", "Brute Force I/O (T0806)", "Block Command Message (T0803)"],
        "ICS Reconnaissance": ["Network Sniffing (T0842)", "Remote System Discovery (T0846)", "Network Connection Enumeration (T0840)", "Program Upload (T0845)"],
    },
    "ATLAS": {
        "Model Evasion Attack": ["Evade AI Model (AML.T0015)", "Craft Adversarial Data (AML.T0043)", "Black-Box Optimization (AML.T0043.001)", "Verify Attack (AML.T0042)"],
        "Data Poisoning Attack": ["Poison Training Data (AML.T0020)", "Publish Poisoned Datasets (AML.T0019)", "Poison AI Model (AML.T0018.000)", "AI Supply Chain Compromise (AML.T0010)"],
        "Model Extraction Attack": ["Discover AI Artifacts (AML.T0007)", "AI Model Inference API Access (AML.T0040)", "Exfiltration via AI Inference API (AML.T0024)", "Create Proxy AI Model (AML.T0005)"],
        "Prompt Injection Attack": ["Acquire Public AI Artifacts (AML.T0002)", "LLM Prompt Injection (AML.T0051)", "AI Agent Tool Invocation (AML.T0053)", "Exfiltration via AI Inference API (AML.T0024)"],
        "LLM Jailbreak": ["LLM Jailbreak (AML.T0054)", "LLM Prompt Injection (AML.T0051)", "LLM Prompt Crafting (AML.T0065)", "Spearphishing via Social Engineering LLM (AML.T0052.000)"],
        "AI Supply Chain Attack": ["AI Supply Chain Compromise (AML.T0010)", "Publish Poisoned Datasets (AML.T0019)", "Manipulate AI Model (AML.T0018)", "Publish Poisoned Models (AML.T0058)"],
    },
}


# ------------------ Data Loading ------------------ #
# Loaders + technique listing live in core/attack_data.py (shared with the MCP
# server). load_attack_data() is lazily cached there.

attack_data = load_attack_data()


def load_techniques():
    """Returns (techniques_df, error). ``error`` is ``None`` on success.

    An empty list is reported as a failure too: a matrix always has
    techniques, so an empty one means the bundle didn't parse as expected, and
    the picker has nothing to offer either way. The error is threaded back to
    ``_requirements`` so it can name the load failure instead of telling the
    user to select a technique from a picker that never rendered.
    """
    try:
        options = list_technique_options(matrix)
    except Exception as e:
        logger.error("Error in load_techniques: %s", e)
        return pd.DataFrame(), str(e)
    if not options:
        logger.error("Error in load_techniques: no %s techniques found", matrix)
        return pd.DataFrame(), "the technique list came back empty."
    return pd.DataFrame(options), None


techniques_df, techniques_load_error = load_techniques()


# ------------------ Prompt Construction ------------------ #
# Prompt text lives in core/prompts.py (shared with the MCP server). This page
# only threads its own inputs + the AI-uplift toggle into the shared builder.


def build_messages(snapshot):
    return build_custom_messages(
        matrix=snapshot["matrix"],
        selected_techniques_string="\n".join(snapshot["selected_techniques"]),
        template_info=snapshot["template_info"],
        industry=snapshot["organisation"]["industry"],
        company_size=snapshot["organisation"]["company_size"],
        ai_uplift=snapshot["modifiers"]["ai_uplift"],
    )


def build_layer_payload(snapshot):
    """Serialise the chosen techniques as an ATT&CK Navigator layer.

    The multiselect carries no phase information, so techniques are emitted
    without a tactic (valid — Navigator places them by ID). Returns the layer
    JSON, or ``None`` when the matrix has no Navigator.
    """
    technique_ids = normalise_technique_ids(snapshot["selected_techniques"])
    if not technique_ids:
        return None
    matrix = snapshot["matrix"]
    layer = build_layer(
        name=f"AttackGen: Custom Scenario ({matrix})",
        matrix=matrix,
        techniques=[(tid, None) for tid in technique_ids],
        description=f"Techniques selected for a custom AttackGen scenario ({matrix} matrix).",
    )
    if layer is None:
        return None
    return dumps(layer)


def build_defense_payload(snapshot):
    """Join the selected techniques to their detection strategies + mitigations.

    Normalises the same multiselect the prompt and layer were built from.
    Returns ``None`` when nothing is selected or there's no defensive data.
    """
    technique_ids = normalise_technique_ids(snapshot["selected_techniques"])
    if not technique_ids:
        return None
    return resolve_defense_report(snapshot["matrix"], technique_ids)


def _modifiers():
    render_ai_uplift_toggle("custom")
    render_defense_narrative_toggle("custom")


def template_selection(template, current_matrix):
    try:
        if template not in incident_response_templates[current_matrix]:
            st.write(f"Template {template} not found in {current_matrix} matrix")
            return

        template_techniques = incident_response_templates[current_matrix][template]
        if current_matrix == "ATLAS":
            matrix_techniques = attack_data["atlas"].get_techniques()
            matrix_technique_names = [f"{tech['name']} ({tech['external_id']})" for tech in matrix_techniques]
        else:
            matrix_techniques = attack_data[current_matrix.lower()].get_techniques()
            matrix_technique_names = [
                f"{technique['name']} ({attack_data[current_matrix.lower()].get_attack_id(technique['id'])})"
                for technique in matrix_techniques
            ]

        st.session_state['selected_techniques'] = [
            tech for tech in template_techniques if tech in matrix_technique_names
        ]
    except Exception as e:
        st.error(f"An error occurred in template_selection: {str(e)}")


# ------------------ Streamlit UI ------------------ #

st.markdown("# <span style='color: #1DB954;'>Generate Custom Scenario🛠️</span>", unsafe_allow_html=True)

if matrix == "ATLAS":
    st.markdown("### Select ATLAS Techniques")
else:
    st.markdown("### Select ATT&CK Techniques")

if techniques_load_error is not None:
    st.error(f"Could not load the {matrix} techniques: {techniques_load_error}")

with st.expander("Use a Template (Optional)"):
    selected_template = st.selectbox(
        "Select a template",
        options=[""] + list(incident_response_templates[matrix].keys()),
        format_func=lambda x: "Select a template" if x == "" else x,
    )
    if selected_template:
        template_selection(selected_template, matrix)

st.markdown("")

selected_techniques = []
if techniques_load_error is None:
    if matrix == "ATLAS":
        technique_options = techniques_df['Display Name'].tolist()
    else:
        techniques = attack_data[matrix.lower()].get_techniques()
        technique_options = [
            f"{technique['name']} ({attack_data[matrix.lower()].get_attack_id(technique['id'])})"
            for technique in techniques
        ]

    select_label = "Select ATLAS Techniques" if matrix == "ATLAS" else "Select ATT&CK Techniques"
    selected_techniques = st.multiselect(
        select_label,
        options=technique_options,
        default=st.session_state.get('selected_techniques', []),
    )

    if matrix == "Enterprise":
        st.info("📝 Techniques are searchable by either their name or technique ID (e.g. `T1556` or `Phishing`).")
    elif matrix == "ICS":
        st.info("📝 Techniques are searchable by either their name or technique ID (e.g. `T0814` or `Denial of Service`).")
    elif matrix == "ATLAS":
        st.info("📝 Techniques are searchable by either their name or technique ID (e.g. `AML.T0051` or `Prompt Injection`).")
    else:
        st.info("📝 Techniques are searchable by either their name or technique ID.")


st.markdown("")
st.markdown("### Generate a Scenario")


def _requirements() -> list[str]:
    """This page's own readiness blockers, in the order the form presents them."""
    label = "ATLAS" if matrix == "ATLAS" else "ATT&CK"
    if techniques_load_error is not None:
        return [f"Could not load the {label} techniques — see the error above."]
    if not selected_techniques:
        return [f"Select at least one {label} technique for the scenario."]
    return []


def _capture_inputs():
    return {
        "scenario_type": "custom",
        "matrix": matrix,
        "organisation": {"industry": industry, "company_size": company_size},
        "selected_entity": (
            {"type": "template", "name": selected_template}
            if selected_template
            else None
        ),
        "selected_techniques": list(selected_techniques),
        "sampled_techniques": list(selected_techniques),
        "template_info": (
            f"This is a '{selected_template}' scenario." if selected_template else ""
        ),
        "modifiers": {
            "ai_uplift": is_ai_uplift_on("custom"),
            "purple_team_narrative": is_defense_narrative_on("custom"),
        },
    }


run_scenario_page(
    page_id="custom",
    build_messages=build_messages,
    requirements=_requirements,
    setup=setup,
    download_name=f"AttackGen Custom {selected_template} {matrix}.md",
    trace_name="Custom Scenario",
    trace_tags=uplift_trace_tags(("custom_scenario",), page_id="custom"),
    render_modifiers=_modifiers,
    build_layer=build_layer_payload,
    build_defense=build_defense_payload,
    defense_narrative=is_defense_narrative_on("custom"),
    capture_inputs=_capture_inputs,
)


st.markdown(
    '<a href="/" style="display: inline-block; padding: 5px 20px; color: white; text-align: center; text-decoration: none; font-size: 16px; border-radius: 4px;">⬅️ Back</a>',
    unsafe_allow_html=True,
)
