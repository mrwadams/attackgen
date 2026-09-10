import pandas as pd
import streamlit as st

from core.ai_uplift import is_ai_uplift_on, render_ai_uplift_toggle, uplift_trace_tags
from core.attack_data import (
    list_usable_scenario_options,
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
from core.sidebar import render_setup_sidebar
from core.styles import inject_emoji_fonts


# ------------------ Streamlit Configuration ------------------ #

st.set_page_config(page_title="Generate Scenario", page_icon="🛡️")
inject_emoji_fonts()
setup = render_setup_sidebar()

industry = setup.industry
company_size = setup.company_size


# ------------------ Data Loading ------------------ #
# Loaders + kill-chain resolution live in core/attack_data.py (shared with the
# MCP server). load_attack_data() is lazily cached there.

attack_data = load_attack_data()


@st.cache_resource
def load_groups(matrix):
    return pd.DataFrame(
        list_usable_scenario_options(matrix),
        columns=["group", "url", "aliases", "label"],
    )


# ------------------ Prompt Construction ------------------ #
# Prompt text lives in core/prompts.py (shared with the MCP server). This page
# only threads its own inputs + the AI-uplift toggle into the shared builder.


def build_messages(snapshot):
    return build_threat_group_messages(
        matrix=snapshot["matrix"],
        selected_group_alias=snapshot["selected_entity"]["name"],
        kill_chain_string=snapshot["kill_chain_string"],
        industry=snapshot["organisation"]["industry"],
        company_size=snapshot["organisation"]["company_size"],
        ai_uplift=snapshot["modifiers"]["ai_uplift"],
    )


def build_layer_payload(snapshot):
    """Serialise the scenario's kill chain as an ATT&CK Navigator layer.

    Reads the same ``selected_techniques_df`` the prompt was built from, so the
    exported layer matches the techniques the model was given (this page samples
    one technique per phase, so the set differs run to run). Returns the layer
    JSON, or ``None`` when the matrix has no Navigator.
    """
    sampled_techniques = snapshot["sampled_techniques"]
    if not sampled_techniques:
        return None
    techniques = [
        (row["ATT&CK ID"], tactic_shortname(str(row["Phase Name"])))
        for row in sampled_techniques
    ]
    matrix = snapshot["matrix"]
    selected_group_alias = snapshot["selected_entity"]["name"]
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


def build_defense_payload(snapshot):
    """Join the scenario's techniques to their detection strategies + mitigations.

    Uses the same ``selected_techniques_df`` the prompt and layer were built
    from, so the Detection & Response companion matches the scenario's kill
    chain. Returns ``None`` when there's no defensive data.
    """
    sampled_techniques = snapshot["sampled_techniques"]
    if not sampled_techniques:
        return None
    technique_ids = [str(row["ATT&CK ID"]) for row in sampled_techniques]
    matrix = snapshot["matrix"]
    if matrix == "ATLAS":
        return build_defense_report(
            matrix=matrix, technique_ids=technique_ids, atlas_data=attack_data["atlas"]
        )
    return build_defense_report(
        matrix=matrix,
        technique_ids=technique_ids,
        mitre_data=attack_data[matrix.lower()],
    )


def _modifiers():
    render_ai_uplift_toggle("threat_group")
    render_defense_narrative_toggle("threat_group")


# ------------------ Streamlit UI ------------------ #

st.markdown("# <span style='color: #1DB954;'>Generate Threat Group Scenario🛡️</span>", unsafe_allow_html=True)

matrix = setup.matrix or "Enterprise"
groups = load_groups(matrix)

if matrix == "ATLAS":
    st.markdown("### Select a Case Study")
    entity_label = "case study"
    select_placeholder = "Select Case Study"
else:
    st.markdown("### Select a Threat Actor Group")
    entity_label = "threat actor group"
    select_placeholder = "Select Group"

# Options arrive already in natural-sorted order from list_usable_scenario_options
# (APT3, APT28, APT29, APT30…), so preserve that rather than re-sorting the labels
# as plain strings.
group_names = list(groups["group"].unique())
group_labels = dict(zip(groups["group"], groups["label"]))

selected_group_alias = st.selectbox(
    f"Select a {entity_label} for the scenario",
    group_names,
    index=None,
    placeholder=select_placeholder,
    format_func=group_labels.__getitem__,
    label_visibility="hidden",
)

kill_chain_string = ""
techniques_df = pd.DataFrame()
selected_techniques_df = pd.DataFrame()
lookup_error = None

try:
    if selected_group_alias is not None:
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
            # Deliberately not `st.stop()`: halting the script here would also
            # tear down the readiness summary and any result this page is
            # already holding. The empty technique set becomes a readiness
            # blocker instead (see `_requirements`).
            st.warning(
                f"There are no {matrix} techniques associated with the {entity}: {selected_group_alias}"
            )
        else:
            # Rebuild the DataFrames the rest of the page (expander, layer,
            # defense) expects, from the resolver's JSON-native records.
            techniques_df = pd.DataFrame(kill_chain.all_techniques)
            selected_techniques_df = pd.DataFrame(kill_chain.techniques)

            expander_title = "Associated ATLAS Techniques" if matrix == "ATLAS" else "Associated ATT&CK Techniques"
            with st.expander(expander_title):
                st.dataframe(data=techniques_df, height=200, width='stretch', hide_index=True)

            kill_chain_string = kill_chain.kill_chain_string
except Exception as e:
    # A failed lookup leaves the DataFrames at their empty defaults, which on
    # its own is indistinguishable from a group that genuinely has no
    # techniques. Record the failure so `_requirements` can say what actually
    # happened instead of contradicting the error above.
    lookup_error = str(e)
    st.error("An error occurred: " + lookup_error)


st.markdown("")

if matrix == "ATLAS":
    st.markdown("### Generate a Scenario")
    st.caption("Built from the case study's documented attack procedure.")
else:
    st.markdown("### Generate a Scenario")
    st.caption(
        "Built from a random selection of the group's techniques, so each run "
        "differs."
    )


def _requirements() -> list[str]:
    """This page's own readiness blockers, in the order the form presents them."""
    if selected_group_alias is None:
        return [f"Select a {entity_label} for the scenario."]
    if lookup_error is not None:
        return [
            f"Could not load the {matrix} techniques for "
            f"'{selected_group_alias}' — see the error above."
        ]
    if techniques_df.empty or not kill_chain_string:
        return [
            f"Select a {entity_label} with associated {matrix} techniques — "
            f"'{selected_group_alias}' has none."
        ]
    return []


def _capture_inputs():
    return {
        "scenario_type": "case_study" if matrix == "ATLAS" else "threat_group",
        "matrix": matrix,
        "organisation": {"industry": industry, "company_size": company_size},
        "selected_entity": {"type": entity_label, "name": selected_group_alias},
        "selected_techniques": techniques_df.to_dict(orient="records"),
        "sampled_techniques": selected_techniques_df.to_dict(orient="records"),
        "kill_chain_string": kill_chain_string,
        "modifiers": {
            "ai_uplift": is_ai_uplift_on("threat_group"),
            "purple_team_narrative": is_defense_narrative_on("threat_group"),
        },
    }


run_scenario_page(
    page_id="threat_group",
    build_messages=build_messages,
    requirements=_requirements,
    setup=setup,
    download_name=f"AttackGen {selected_group_alias} {matrix}.md",
    trace_name="Threat Group Scenario",
    trace_tags=uplift_trace_tags(("threat_group_scenario",), page_id="threat_group"),
    render_modifiers=_modifiers,
    build_layer=build_layer_payload,
    build_defense=build_defense_payload,
    defense_narrative=is_defense_narrative_on("threat_group"),
    capture_inputs=_capture_inputs,
)


st.markdown(
    '<a href="/" style="display: inline-block; padding: 5px 20px; color: white; text-align: center; text-decoration: none; font-size: 16px; border-radius: 4px;">⬅️ Back</a>',
    unsafe_allow_html=True,
)
