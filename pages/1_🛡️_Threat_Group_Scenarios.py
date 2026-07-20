import pandas as pd
import streamlit as st

from core.ai_uplift import is_ai_uplift_on, render_ai_uplift_toggle, uplift_trace_tags
from core.attack_data import (
    load_attack_data,
    resolve_case_study_kill_chain,
    resolve_threat_group_kill_chain,
)
from core.prompts import build_threat_group_messages
from core.detections import (
    build_defense_report,
    is_defense_narrative_on,
    render_defense_narrative_toggle,
)
from core.navigator import build_layer, dumps, tactic_shortname
from core.scenario_page import run_scenario_page
from core.state import restore_from_query_params
from core.styles import inject_emoji_fonts

# Restore sidebar selections on direct page loads (e.g. browser refresh while
# on this page). See core/state.py for the persisted-keys list.
restore_from_query_params()


# ------------------ Streamlit Configuration ------------------ #

st.set_page_config(page_title="Generate Scenario", page_icon="🛡️")
inject_emoji_fonts()

model_provider = st.session_state.get("chosen_model_provider", "OpenAI API")
industry = st.session_state.get("industry")
company_size = st.session_state.get("company_size")


# ------------------ Data Loading ------------------ #
# Loaders + kill-chain resolution live in core/attack_data.py (shared with the
# MCP server). load_attack_data() is lazily cached there.

attack_data = load_attack_data()


@st.cache_resource
def load_groups(matrix):
    if matrix == "Enterprise":
        return pd.read_json("./data/groups.json")
    if matrix == "ICS":
        return pd.read_json("./data/groups_ics.json")
    return pd.read_json("./data/atlas-case-studies.json")


# ------------------ Prompt Construction ------------------ #
# Prompt text lives in core/prompts.py (shared with the MCP server). This page
# only threads its own inputs + the AI-uplift toggle into the shared builder.


def build_messages(matrix, selected_group_alias, kill_chain_string):
    return build_threat_group_messages(
        matrix=matrix,
        selected_group_alias=selected_group_alias,
        kill_chain_string=kill_chain_string,
        industry=industry,
        company_size=company_size,
        ai_uplift=is_ai_uplift_on("threat_group"),
    )


def build_layer_payload():
    """Serialise the scenario's kill chain as an ATT&CK Navigator layer.

    Reads the same ``selected_techniques_df`` the prompt was built from, so the
    exported layer matches the techniques the model was given (this page samples
    one technique per phase, so the set differs run to run). Returns the layer
    JSON, or ``None`` when the matrix has no Navigator.
    """
    if selected_techniques_df.empty:
        return None
    techniques = [
        (row["ATT&CK ID"], tactic_shortname(str(row["Phase Name"])))
        for _, row in selected_techniques_df.iterrows()
    ]
    layer = build_layer(
        name=f"AttackGen: {selected_group_alias} ({matrix})",
        matrix=matrix,
        techniques=techniques,
        description=(
            f"Techniques used in the AttackGen scenario for "
            f"'{selected_group_alias}' ({matrix} matrix)."
        ),
    )
    if layer is None:
        return None
    return dumps(layer)


def build_defense_payload():
    """Join the scenario's techniques to their detection strategies + mitigations.

    Uses the same ``selected_techniques_df`` the prompt and layer were built
    from, so the Detection & Response companion matches the scenario's kill
    chain. Returns ``None`` when there's no defensive data.
    """
    if selected_techniques_df.empty:
        return None
    technique_ids = [str(row["ATT&CK ID"]) for _, row in selected_techniques_df.iterrows()]
    if matrix == "ATLAS":
        return build_defense_report(
            matrix=matrix, technique_ids=technique_ids, atlas_data=attack_data["atlas"]
        )
    return build_defense_report(
        matrix=matrix,
        technique_ids=technique_ids,
        mitre_data=attack_data[matrix.lower()],
    )


def _inline_controls():
    render_ai_uplift_toggle("threat_group")
    render_defense_narrative_toggle("threat_group")


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

        # Kill-chain resolution (incl. the per-phase sampling for ATT&CK) lives in
        # core.attack_data, shared with the MCP server. This page just renders it.
        if matrix == "ATLAS":
            kill_chain = resolve_case_study_kill_chain(selected_group_alias)
        else:
            kill_chain = resolve_threat_group_kill_chain(matrix, selected_group_alias)

        if not kill_chain.all_techniques:
            entity = "case study" if matrix == "ATLAS" else "threat group"
            st.warning(
                f"There are no {matrix} techniques associated with the {entity}: {selected_group_alias}"
            )
            st.stop()

        # Rebuild the DataFrames the rest of the page (expander, layer, defense)
        # expects, from the resolver's JSON-native records.
        techniques_df = pd.DataFrame(kill_chain.all_techniques)
        selected_techniques_df = pd.DataFrame(kill_chain.techniques)

        expander_title = "Associated ATLAS Techniques" if matrix == "ATLAS" else "Associated ATT&CK Techniques"
        with st.expander(expander_title):
            st.dataframe(data=techniques_df, height=200, width='stretch', hide_index=True)

        kill_chain_string = kill_chain.kill_chain_string
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
    download_name=f"AttackGen {selected_group_alias} {matrix}.md",
    trace_name="Threat Group Scenario",
    trace_tags=uplift_trace_tags(("threat_group_scenario",), page_id="threat_group"),
    inline_control=_inline_controls,
    build_layer=build_layer_payload,
    build_defense=build_defense_payload,
    defense_narrative=is_defense_narrative_on("threat_group"),
)


st.markdown(
    '<a href="/" style="display: inline-block; padding: 5px 20px; color: white; text-align: center; text-decoration: none; font-size: 16px; border-radius: 4px;">⬅️ Back</a>',
    unsafe_allow_html=True,
)
