import logging

import pandas as pd
import streamlit as st

from core.ai_uplift import is_ai_uplift_on, render_ai_uplift_toggle, uplift_trace_tags
from core.attack_data import list_technique_options
from core.custom_templates import INCIDENT_RESPONSE_TEMPLATES, resolve_incident_template
from core.prompts import build_custom_messages
from core.detections import (
    is_defense_narrative_on,
    render_defense_narrative_toggle,
    resolve_defense_report,
)
from core.matrices import DEFAULT_MATRIX, MATRICES
from core.navigator import build_layer, dumps, normalise_technique_ids
from core.request import (
    CustomPayload,
    Modifiers,
    Organisation,
    ScenarioRequest,
    SelectedEntity,
)
from core.scenario_page import run_scenario_page
from core.sidebar import render_setup_sidebar
from core.styles import back_link, inject_emoji_fonts, page_heading

logger = logging.getLogger(__name__)


# ------------------ Streamlit Configuration ------------------ #

st.set_page_config(page_title="Generate Custom Scenario", page_icon="🛠️")
inject_emoji_fonts()
setup = render_setup_sidebar()

industry = setup.industry
company_size = setup.company_size
matrix = setup.matrix or DEFAULT_MATRIX
matrix_info = MATRICES[matrix]


# ------------------ Data Loading ------------------ #
# Loaders + technique listing live in core/attack_data.py (shared with the MCP
# server). The bundles are lazily cached there.


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


def build_messages(request: ScenarioRequest):
    return build_custom_messages(
        matrix=request.matrix,
        selected_techniques_string="\n".join(request.techniques),
        template_info=request.payload.template_info,
        industry=request.organisation.industry,
        company_size=request.organisation.company_size,
        ai_uplift=request.modifiers.ai_uplift,
    )


def build_layer_payload(request: ScenarioRequest):
    """Serialise the chosen techniques as an ATT&CK Navigator layer.

    The multiselect carries no phase information, so techniques are emitted
    without a tactic (valid — Navigator places them by ID). Returns the layer
    JSON, or ``None`` when the matrix has no Navigator.
    """
    technique_ids = normalise_technique_ids(request.techniques)
    if not technique_ids:
        return None
    matrix = request.matrix
    layer = build_layer(
        name=f"AttackGen: Custom Scenario ({matrix})",
        matrix=matrix,
        techniques=[(tid, None) for tid in technique_ids],
        description=f"Techniques selected for a custom AttackGen scenario ({matrix} matrix).",
    )
    if layer is None:
        return None
    return dumps(layer)


def build_defense_payload(request: ScenarioRequest):
    """Join the selected techniques to their detection strategies + mitigations.

    Normalises the same multiselect the prompt and layer were built from.
    Returns ``None`` when nothing is selected or there's no defensive data.
    """
    technique_ids = normalise_technique_ids(request.techniques)
    if not technique_ids:
        return None
    return resolve_defense_report(request.matrix, technique_ids)


def _modifiers():
    render_ai_uplift_toggle("custom")
    render_defense_narrative_toggle("custom")


def template_selection(template, current_matrix):
    try:
        kept, dropped = resolve_incident_template(current_matrix, template)
    except Exception as e:
        st.error(f"An error occurred in template_selection: {str(e)}")
        return
    st.session_state['selected_techniques'] = kept
    st.session_state['selected_techniques_matrix'] = current_matrix
    if dropped:
        st.warning(
            f"The {template} template left out techniques that are not active in the {current_matrix} "
            f"bundle: {', '.join(dropped)}."
        )


# ------------------ Streamlit UI ------------------ #

page_heading("Generate Custom Scenario🛠️")

st.markdown(f"### Select {matrix_info.framework} Techniques")

if techniques_load_error is not None:
    st.error(f"Could not load the {matrix} techniques: {techniques_load_error}")

with st.expander("Use a Template (Optional)"):
    selected_template = st.selectbox(
        "Select a template",
        options=[""] + list(INCIDENT_RESPONSE_TEMPLATES[matrix].keys()),
        format_func=lambda x: "Select a template" if x == "" else x,
    )
    if selected_template:
        template_selection(selected_template, matrix)

st.markdown("")

selected_techniques = []
if techniques_load_error is None:
    technique_options = techniques_df['Display Name'].tolist()

    # A template's techniques belong to the matrix it was picked on. After a
    # matrix switch they are not options in the new picker, so drop them.
    if st.session_state.get('selected_techniques_matrix') != matrix:
        st.session_state['selected_techniques'] = []

    selected_techniques = st.multiselect(
        f"Select {matrix_info.framework} Techniques",
        options=technique_options,
        default=st.session_state.get('selected_techniques', []),
    )

    st.info(
        "📝 Techniques are searchable by either their name or technique ID "
        f"(e.g. {matrix_info.technique_example})."
    )


st.markdown("")
st.markdown("### Generate a Scenario")


def _requirements() -> list[str]:
    """This page's own readiness blockers, in the order the form presents them."""
    if techniques_load_error is not None:
        # Phrased as something the user can act on: every other line in the
        # readiness box is a step they can complete, so a bare "could not
        # load" would read as a status report with no way forward. The whole
        # matrix's technique list failed, so — unlike page 1, where the user
        # can pick a different group — the step that changes what gets loaded
        # is switching framework in the sidebar. `load_techniques` isn't
        # cached, so the switch retries the load on the next rerun. Names
        # `matrix` rather than `matrix_info.framework`, which is "ATT&CK" for both Enterprise
        # and ICS and so wouldn't say which one failed. Reloading is
        # deliberately not offered: it starts a new session, dropping a key
        # typed into the sidebar and any scenario already generated.
        return [
            f"Choose a different MITRE framework in the Setup sidebar — "
            f"loading the {matrix} techniques failed (see the error above)."
        ]
    if not selected_techniques:
        return [f"Select at least one {matrix_info.framework} technique for the scenario."]
    return []


def _capture_inputs() -> ScenarioRequest:
    return ScenarioRequest(
        scenario_type="custom",
        matrix=matrix,
        organisation=Organisation(industry=industry, company_size=company_size),
        selected_entity=(
            SelectedEntity(type="template", name=selected_template)
            if selected_template
            else None
        ),
        techniques=tuple(selected_techniques),
        modifiers=Modifiers(
            ai_uplift=is_ai_uplift_on("custom"),
            purple_team_narrative=is_defense_narrative_on("custom"),
        ),
        payload=CustomPayload(
            template_info=(
                f"This is a '{selected_template}' scenario." if selected_template else ""
            ),
        ),
    )


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


back_link()
