import pandas as pd
import streamlit as st
from mitreattack.stix20 import MitreAttackData

from atlas_parser import ATLASData, get_techniques_from_case_study_procedure
from core.scenario_page import run_scenario_page
from core.state import restore_from_query_params

# Restore sidebar selections on direct page loads (e.g. browser refresh while
# on this page). See core/state.py for the persisted-keys list.
restore_from_query_params()


# ------------------ Streamlit Configuration ------------------ #

st.set_page_config(page_title="Generate Scenario", page_icon="🛡️")

model_provider = st.session_state.get("chosen_model_provider", "OpenAI API")
industry = st.session_state.get("industry")
company_size = st.session_state.get("company_size")


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
def load_groups(matrix):
    if matrix == "Enterprise":
        return pd.read_json("./data/groups.json")
    if matrix == "ICS":
        return pd.read_json("./data/groups_ics.json")
    return pd.read_json("./data/atlas-case-studies.json")


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

**Case Study Reference:**
This scenario is based on the documented MITRE ATLAS case study: '{selected_group_alias}'
The attack procedure uses the following techniques from the MITRE ATLAS framework:
{kill_chain_string}

**Your task:**
Create an incident response testing scenario based on this AI/ML attack case study. The goal is to test the company's incident response capabilities against adversarial machine learning attacks targeting their AI systems.

Focus on realistic attack vectors that target AI/ML infrastructure, including model manipulation, data poisoning, adversarial inputs, and AI supply chain attacks.

Your response should be well structured and formatted using Markdown. Write in British English.
"""

ATTACK_HUMAN_TEMPLATE = """
**Background information:**
The company operates in the '{industry}' industry and is of size '{company_size}'.

**Threat actor information:**
Threat actor group '{selected_group_alias}' is planning to target the company using the following kill chain from the MITRE ATT&CK {matrix} Matrix:
{kill_chain_string}

**Your task:**
Create an incident response testing scenario based on the information provided. The goal of the scenario is to test the company's incident response capabilities against the identified threat actor group, focusing on the {matrix} environment.

Your response should be well structured and formatted using Markdown. Write in British English.
"""


def build_messages(matrix, selected_group_alias, kill_chain_string):
    template = ATLAS_HUMAN_TEMPLATE if matrix == "ATLAS" else ATTACK_HUMAN_TEMPLATE
    user_content = template.format(
        industry=industry,
        company_size=company_size,
        selected_group_alias=selected_group_alias,
        kill_chain_string=kill_chain_string,
        matrix=matrix,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# ------------------ Streamlit UI ------------------ #

st.markdown("# <span style='color: #1DB954;'>Generate Threat Group Scenario🛡️</span>", unsafe_allow_html=True)

matrix = st.session_state.get("matrix", "Enterprise")
groups = load_groups(matrix)

if matrix == "ATLAS":
    st.markdown(
        """
        ### Select a Case Study

        Use the drop-down selector below to select a case study from the MITRE ATLAS framework.

        You can then optionally view all of the ATLAS techniques associated with the case study and/or the case study's page on the MITRE ATLAS site.
        """
    )
    entity_label = "case study"
    select_placeholder = "Select Case Study"
else:
    st.markdown(
        f"""
        ### Select a Threat Actor Group

        Use the drop-down selector below to select a threat actor group from the MITRE ATT&CK framework.

        You can then optionally view all of the {matrix} ATT&CK techniques associated with the group and/or the group's page on the MITRE ATT&CK site.
        """
    )
    entity_label = "threat actor group"
    select_placeholder = "Select Group"

group_names = sorted(groups['group'].unique())
default_index = 0 if group_names else None

selected_group_alias = st.selectbox(
    f"Select a {entity_label} for the scenario",
    group_names,
    index=default_index,
    placeholder=select_placeholder,
    label_visibility="hidden",
)

if matrix == "ATLAS":
    phase_name_order = [
        'Reconnaissance', 'Resource Development', 'Initial Access', 'AI Model Access',
        'Execution', 'Persistence', 'Privilege Escalation', 'Defense Evasion',
        'Credential Access', 'Discovery', 'Lateral Movement', 'Collection',
        'AI Attack Staging', 'Command and Control', 'Exfiltration', 'Impact',
    ]
else:
    phase_name_order = [
        'Reconnaissance', 'Resource Development', 'Initial Access', 'Execution', 'Persistence',
        'Privilege Escalation', 'Defense Evasion', 'Credential Access', 'Discovery', 'Lateral Movement',
        'Collection', 'Command and Control', 'Exfiltration', 'Impact',
    ]

phase_name_category = pd.CategoricalDtype(categories=phase_name_order, ordered=True)


messages = None
techniques_df = pd.DataFrame()
selected_techniques_df = pd.DataFrame()

try:
    if selected_group_alias != select_placeholder:
        group_url = groups[groups['group'] == selected_group_alias]['url'].values[0]
        if matrix == "ATLAS":
            st.markdown(f"[View case study on atlas.mitre.org]({group_url})")
        else:
            st.markdown(f"[View {selected_group_alias}'s page on attack.mitre.org]({group_url})")

        if matrix == "ATLAS":
            case_study_row = groups[groups['group'] == selected_group_alias].iloc[0]
            procedure = case_study_row.get('procedure', [])

            if not procedure:
                st.warning(f"There are no ATLAS techniques associated with the case study: {selected_group_alias}")
                st.stop()

            techniques_list = get_techniques_from_case_study_procedure(procedure, attack_data["atlas"])
            if not techniques_list:
                st.warning(f"Could not extract techniques from the case study: {selected_group_alias}")
                st.stop()

            techniques_df = pd.DataFrame(techniques_list)
            techniques_df_llm = techniques_df.copy()
            techniques_df['Phase Name'] = techniques_df['Phase Name'].astype(phase_name_category)
            techniques_df_llm['Phase Name'] = techniques_df_llm['Phase Name'].astype(phase_name_category)
            techniques_df = techniques_df.sort_values('Phase Name')
            techniques_df_llm = techniques_df_llm.sort_values('Phase Name')

            selected_techniques_df = techniques_df_llm.copy()
            techniques_df = techniques_df[['Technique Name', 'ATT&CK ID', 'Phase Name']]
        else:
            group = attack_data[matrix.lower()].get_groups_by_alias(selected_group_alias)
            if group:
                group_stix_id = group[0].id
                techniques = attack_data[matrix.lower()].get_techniques_used_by_group(group_stix_id)
                if not techniques:
                    st.warning(f"There are no {matrix} ATT&CK techniques associated with the threat group: {selected_group_alias}")
                    st.stop()

                techniques_df = pd.DataFrame(techniques)
                techniques_df_llm = techniques_df.copy()
                techniques_df['Technique Name'] = techniques_df_llm['Technique Name'] = techniques_df['object'].apply(lambda x: x['name'])
                techniques_df['ATT&CK ID'] = techniques_df_llm['ATT&CK ID'] = techniques_df['object'].apply(lambda x: attack_data[matrix.lower()].get_attack_id(x['id']))
                techniques_df['Phase Name'] = techniques_df_llm['Phase Name'] = techniques_df['object'].apply(lambda x: x['kill_chain_phases'][0]['phase_name'])
                techniques_df = techniques_df.drop_duplicates(['Phase Name', 'Technique Name', 'ATT&CK ID'])

                techniques_df['Phase Name'] = techniques_df['Phase Name'].str.replace('-', ' ').str.title()
                techniques_df_llm['Phase Name'] = techniques_df_llm['Phase Name'].str.replace('-', ' ').str.title()
                techniques_df['Phase Name'] = techniques_df['Phase Name'].replace('Command And Control', 'Command and Control')
                techniques_df_llm['Phase Name'] = techniques_df_llm['Phase Name'].replace('Command And Control', 'Command and Control')
                techniques_df['Phase Name'] = techniques_df['Phase Name'].astype(phase_name_category)
                techniques_df_llm['Phase Name'] = techniques_df_llm['Phase Name'].astype(phase_name_category)
                techniques_df = techniques_df.sort_values('Phase Name')
                techniques_df_llm = techniques_df_llm.sort_values('Phase Name')

                selected_techniques_df = (
                    techniques_df_llm.groupby('Phase Name', observed=False)
                    .apply(
                        lambda x: x.sample(n=1) if not x.empty else pd.DataFrame(columns=x.columns),
                        include_groups=False,
                    )
                    .reset_index()
                )

                techniques_df = techniques_df.sort_values('Phase Name')
                techniques_df = techniques_df[['Technique Name', 'ATT&CK ID', 'Phase Name']]

        if not techniques_df.empty:
            expander_title = "Associated ATLAS Techniques" if matrix == "ATLAS" else "Associated ATT&CK Techniques"
            with st.expander(expander_title):
                st.dataframe(data=techniques_df, height=200, width='stretch', hide_index=True)

        kill_chain = [
            f"{row['Phase Name']}: {row['Technique Name']} ({row['ATT&CK ID']})"
            for _, row in selected_techniques_df.iterrows()
        ]
        kill_chain_string = "\n".join(kill_chain)
        messages = build_messages(matrix, selected_group_alias, kill_chain_string)
except Exception as e:
    st.error("An error occurred: " + str(e))


st.markdown("")

if matrix == "ATLAS":
    st.markdown(
        """
        ### Generate a Scenario

        Click the button below to generate a scenario based on the selected case study. The documented attack procedure from the case study will be used to generate the scenario.

        It normally takes between 30-50 seconds to generate a scenario, although for local models this is highly dependent on your hardware and the selected model. ⏱️
        """
    )
else:
    st.markdown(
        """
        ### Generate a Scenario

        Click the button below to generate a scenario based on the selected threat actor group. A selection of the group's known techniques will be chosen at random and used to generate the scenario.

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
    if techniques_df.empty:
        st.info(f"Please select a {entity_label} with associated techniques.")
        return False
    if messages is None:
        return False
    return True


run_scenario_page(
    page_id="threat_group",
    build_messages=lambda: messages,
    is_ready=_ready,
    download_name="threat_group_scenario.md",
    trace_name="Threat Group Scenario",
    trace_tags=("threat_group_scenario",),
)


st.markdown(
    '<a href="/" style="display: inline-block; padding: 5px 20px; color: white; text-align: center; text-decoration: none; font-size: 16px; border-radius: 4px;">⬅️ Back</a>',
    unsafe_allow_html=True,
)
