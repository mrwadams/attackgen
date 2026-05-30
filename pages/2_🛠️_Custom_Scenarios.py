import os
import re

import pandas as pd
import streamlit as st
from langsmith import Client
from mitreattack.stix20 import MitreAttackData

from atlas_parser import ATLASData
from core.llm import call_llm
from core.schemas import LLMConfig
from core.state import restore_from_query_params

# Restore sidebar selections on direct page loads (e.g. browser refresh while
# on this page). See core/state.py for the persisted-keys list.
restore_from_query_params()


# ------------------ LangSmith Setup ------------------ #

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "AttackGen"

if "LANGCHAIN_API_KEY" in st.secrets:
    client = Client(api_key=st.secrets["LANGCHAIN_API_KEY"])
else:
    client = None


# ------------------ Streamlit Configuration ------------------ #

st.set_page_config(page_title="Generate Custom Scenario", page_icon="🛠️")

model_provider = st.session_state.get("chosen_model_provider", "OpenAI API")
industry = st.session_state.get("industry")
company_size = st.session_state.get("company_size")
matrix = st.session_state.get("matrix", "Enterprise")

if "custom_scenario_generated" not in st.session_state:
    st.session_state["custom_scenario_generated"] = False


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

@st.cache_resource
def load_attack_data():
    return {
        "enterprise": MitreAttackData("./data/enterprise-attack.json"),
        "ics": MitreAttackData("./data/ics-attack.json"),
        "atlas": ATLASData("./data/stix-atlas.json"),
    }


attack_data = load_attack_data()


@st.cache_resource
def load_techniques():
    try:
        current_matrix = st.session_state.get("matrix", "Enterprise")
        if current_matrix == "ATLAS":
            techniques = attack_data["atlas"].get_techniques()
            return pd.DataFrame([
                {
                    'id': tech['id'],
                    'Technique Name': tech['name'],
                    'External ID': tech['external_id'],
                    'Display Name': f"{tech['name']} ({tech['external_id']})",
                }
                for tech in techniques
            ])

        techniques = attack_data[current_matrix.lower()].get_techniques()
        rows = []
        for technique in techniques:
            for reference in technique.external_references:
                if "external_id" in reference:
                    rows.append({
                        'id': technique.id,
                        'Technique Name': technique.name,
                        'External ID': reference['external_id'],
                        'Display Name': f"{technique.name} ({reference['external_id']})",
                    })
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"Error in load_techniques: {e}")
        return pd.DataFrame()


techniques_df = load_techniques()


# ------------------ Prompt Construction ------------------ #

SYSTEM_PROMPT = (
    "You are a cybersecurity expert. Your task is to produce a comprehensive incident response "
    "testing scenario based on the information provided. Format your response using proper "
    "Markdown syntax with headers, bullet points, and formatting for readability."
)

ATLAS_HUMAN_TEMPLATE = """
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

ATTACK_HUMAN_TEMPLATE = """
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


def build_messages(selected_techniques_string, template_info):
    template = ATLAS_HUMAN_TEMPLATE if matrix == "ATLAS" else ATTACK_HUMAN_TEMPLATE
    user_content = template.format(
        industry=industry,
        company_size=company_size,
        selected_techniques_string=selected_techniques_string,
        template_info=template_info,
        matrix=matrix,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def post_process(scenario_text: str) -> tuple[str | None, str]:
    match = re.search(r'<think>(.*?)</think>', scenario_text, re.DOTALL)
    thinking = match.group(1).strip() if match else None
    cleaned = re.sub(r'<think>.*?</think>', '', scenario_text, flags=re.DOTALL).strip()
    cleaned = re.sub(r'^```\w*\n|```$', '', cleaned, flags=re.MULTILINE).strip()
    return thinking, cleaned


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

with st.expander("Use a Template (Optional)"):
    if matrix == "ATLAS":
        st.markdown("Select a template to quickly generate a custom scenario based on a predefined set of ATLAS techniques.")
    else:
        st.markdown("Select a template to quickly generate a custom scenario based on a predefined set of ATT&CK techniques.")

    selected_template = st.selectbox(
        "Select a template",
        options=[""] + list(incident_response_templates[matrix].keys()),
        format_func=lambda x: "Select a template" if x == "" else x,
    )
    if selected_template:
        template_selection(selected_template, matrix)

st.markdown("")

if matrix == "ATLAS":
    st.markdown("Use the multi-select box below to add or update the ATLAS techniques that you would like to include in a custom incident response testing scenario.")
else:
    st.markdown("Use the multi-select box below to add or update the ATT&CK techniques that you would like to include in a custom incident response testing scenario.")

selected_techniques = []
if not techniques_df.empty:
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


messages = None
try:
    if selected_techniques:
        selected_techniques_string = '\n'.join(selected_techniques)
        template_info = f"This is a '{selected_template}' scenario." if selected_template else ""
        messages = build_messages(selected_techniques_string, template_info)
except Exception as e:
    st.error(f"An error occurred: {str(e)}")

st.markdown("")
st.markdown(
    """
    ### Generate a Scenario

    Click the button below to generate a scenario based on the selected technique(s).

    It normally takes between 30-50 seconds to generate a scenario, although for local models this is highly dependent on your hardware and the selected model. ⏱️
    """
)


def _render_scenario(scenario_text: str):
    st.session_state['custom_scenario_generated'] = True
    st.session_state['custom_scenario_text'] = scenario_text
    st.markdown(scenario_text)
    st.download_button(
        label="Download Scenario",
        data=scenario_text,
        file_name="custom_scenario.md",
        mime="text/markdown",
    )
    st.session_state['last_scenario'] = True
    st.session_state['last_scenario_text'] = scenario_text


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
    if not selected_techniques:
        st.info("Please select at least one technique.")
        return False
    return True


try:
    if st.button('Generate Scenario', key='generate_custom_scenario'):
        if _ready() and messages is not None:
            config = LLMConfig.from_session_state(
                trace_name="Custom Scenario",
                trace_tags=("custom_scenario",),
            )
            scenario_text = None
            try:
                with st.status('Generating scenario...', expanded=True):
                    st.write("Calling the model.")
                    scenario_text = call_llm(config, messages)
                    st.write("Scenario generated successfully.")
            except Exception as e:
                st.error(f"An error occurred while generating the scenario: {str(e)}")

            st.markdown("---")
            if scenario_text:
                thinking, cleaned = post_process(scenario_text)
                if thinking:
                    with st.expander("View Model's Reasoning"):
                        st.markdown(thinking)
                _render_scenario(cleaned)
            elif 'custom_scenario_text' in st.session_state and st.session_state['custom_scenario_generated']:
                st.markdown("Displaying previously generated scenario:")
                st.markdown(st.session_state['custom_scenario_text'])
                st.download_button(
                    label="Download Scenario",
                    data=st.session_state['custom_scenario_text'],
                    file_name="custom_scenario.md",
                    mime="text/markdown",
                )

    if 'LANGCHAIN_API_KEY' not in st.secrets:
        st.info("ℹ️ No LangChain API key has been set. This run will not be logged to LangSmith.")

    feedback_placeholder = st.empty()
    st.markdown("---")
    if st.session_state.get('custom_scenario_generated', False) and client is not None:
        st.markdown("Rate the scenario to help improve this tool.")
        col1, col2, _ = st.columns([0.5, 0.5, 5])
        with col1:
            if st.button("👍", key="thumbs_up_custom"):
                try:
                    run_id = st.session_state.get('run_id')
                    if run_id:
                        feedback_record = client.create_feedback(run_id, "positive", score=1, comment="")
                        st.session_state.feedback = {"feedback_id": str(feedback_record.id), "score": 1}
                        feedback_placeholder.success("Feedback submitted. Thank you.")
                    else:
                        feedback_placeholder.warning("No run ID found. Please generate a scenario first.")
                except Exception as e:
                    feedback_placeholder.error(f"An error occurred while creating feedback: {str(e)}")
        with col2:
            if st.button("👎"):
                try:
                    run_id = st.session_state.get('run_id')
                    if run_id:
                        feedback_record = client.create_feedback(run_id, "negative", score=0, comment="")
                        st.session_state.feedback = {"feedback_id": str(feedback_record.id), "score": 0}
                        feedback_placeholder.success("Feedback submitted. Thank you.")
                    else:
                        feedback_placeholder.warning("No run ID found. Please generate a scenario first.")
                except Exception as e:
                    feedback_placeholder.error(f"An error occurred while creating feedback: {str(e)}")
except Exception as e:
    st.error("An error occurred: " + str(e))


st.markdown(
    '<a href="/" style="display: inline-block; padding: 5px 20px; color: white; text-align: center; text-decoration: none; font-size: 16px; border-radius: 4px;">⬅️ Back</a>',
    unsafe_allow_html=True,
)
