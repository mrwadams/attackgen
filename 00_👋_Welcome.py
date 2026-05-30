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
"""

import os

import streamlit as st
from dotenv import load_dotenv

from core.models import PROVIDERS, get_models_for_provider
from core.state import restore_from_query_params, sync_to_query_params

# Load environment variables from .env file
load_dotenv()

# ------------------ Streamlit UI Configuration ------------------ #

st.set_page_config(
    page_title="AttackGen",
    page_icon="👾",
)

# Restore sidebar selections from ?p=...&m=... query params (set on previous
# visits via sync_to_query_params below). Runs before any widgets so that the
# selectboxes/radios pick up the restored values via their `key=`/`index=`.
restore_from_query_params()


# ------------------ Sidebar ------------------ #

with st.sidebar:
    st.sidebar.markdown("### <span style='color: #1DB954;'>Setup</span>", unsafe_allow_html=True)

    # NB: no `key=` on this (or any) widget that backs a persisted shadow key.
    # Streamlit clears widget-key entries from session_state when navigating to
    # a page that doesn't host the widget, which would lose the user's choices
    # the moment they leave Welcome. Seeding via `index=` from the shadow key
    # avoids that and keeps the URL → shadow → widget pipeline one-directional.
    provider_options = list(PROVIDERS.keys())
    persisted_provider = st.session_state.get("chosen_model_provider")
    provider_idx = (
        provider_options.index(persisted_provider)
        if persisted_provider in provider_options
        else 0
    )
    model_provider = st.selectbox(
        "Select your preferred model provider:",
        provider_options,
        index=provider_idx,
        help="Select the model provider you would like to use. This will determine the models available for selection.",
    )
    st.session_state["chosen_model_provider"] = model_provider

    provider_info = PROVIDERS[model_provider]

    # ---- API key ----
    if provider_info.needs_api_key:
        env_key = os.getenv(provider_info.env_var) if provider_info.env_var else None
        if env_key:
            st.session_state["llm_api_key"] = env_key
            st.success("API key loaded from .env")
        else:
            api_key_help = (
                f"You can find your API key at [{provider_info.api_key_url}]({provider_info.api_key_url})."
                if provider_info.api_key_url
                else "Enter the API key for your chosen provider."
            )
            st.session_state["llm_api_key"] = st.text_input(
                f"Enter your {provider_info.name} API key:",
                type="password",
                help=api_key_help,
            )
    else:
        # Optional key for Custom (some local endpoints accept any string)
        env_key = os.getenv(provider_info.env_var) if provider_info.env_var else None
        st.session_state["llm_api_key"] = st.text_input(
            "API key (optional):",
            type="password",
            value=env_key or "",
            help="Optional. Leave blank if your endpoint doesn't require authentication.",
        )

    # ---- Base URL (Custom only) ----
    if provider_info.needs_api_base:
        initial_base = (
            st.session_state.get("llm_api_base")
            or os.getenv("CUSTOM_BASE_URL")
            or provider_info.default_api_base
            or ""
        )
        st.session_state["llm_api_base"] = st.text_input(
            "Base URL:",
            value=initial_base,
            help="Base URL of your OpenAI-compatible endpoint. Example: http://localhost:11434/v1 for Ollama, http://localhost:1234/v1 for LM Studio.",
        )
    else:
        st.session_state["llm_api_base"] = None

    # ---- Model selection ----
    models = get_models_for_provider(model_provider)
    persisted_model = st.session_state.get("llm_model_name")
    if models:
        labels = [m.model_id for m in models]
        help_map = {m.model_id: m.help_text for m in models}
        # Seed the selectbox's value if the persisted model belongs to this
        # provider — otherwise (e.g. after switching providers) fall back to
        # the first option.
        default_idx = labels.index(persisted_model) if persisted_model in labels else 0
        chosen = st.selectbox(
            "Select the model you would like to use:",
            labels,
            index=default_idx,
            help="\n".join(f"**{mid}** — {help_map[mid] or 'No description.'}" for mid in labels),
        )
        st.session_state["llm_model_name"] = chosen
    else:
        # Custom: user types the model id
        initial_model = persisted_model or os.getenv("CUSTOM_MODEL_NAME") or ""
        st.session_state["llm_model_name"] = st.text_input(
            "Model name:",
            value=initial_model,
            help="Model identifier as expected by your endpoint (e.g. 'llama3.1', 'qwen3:32b').",
        )

    st.markdown("""---""")

    matrix_options = ["Enterprise", "ICS", "ATLAS"]
    persisted_matrix = st.session_state.get("matrix")
    matrix_idx = (
        matrix_options.index(persisted_matrix)
        if persisted_matrix in matrix_options
        else 0
    )
    matrix = st.sidebar.radio(
        "Select MITRE Framework:",
        matrix_options,
        index=matrix_idx,
        help="Enterprise and ICS are ATT&CK matrices for traditional IT and industrial control systems. ATLAS focuses on adversarial threats to AI/ML systems.",
    )
    st.session_state["matrix"] = matrix

    industries = sorted([
        'Aerospace / Defense', 'Agriculture / Food Services',
        'Automotive', 'Construction', 'Education',
        'Energy / Utilities', 'Finance / Banking',
        'Government / Public Sector', 'Healthcare',
        'Hospitality / Tourism', 'Insurance',
        'Legal Services', 'Manufacturing',
        'Media / Entertainment', 'Non-profit',
        'Real Estate', 'Retail / E-commerce',
        'Technology / IT', 'Telecommunication',
        'Transportation / Logistics',
    ])
    persisted_industry = st.session_state.get("industry")
    industry = st.selectbox(
        "Select your company's industry:",
        industries,
        index=industries.index(persisted_industry) if persisted_industry in industries else None,
        placeholder="Select Industry",
    )
    st.session_state["industry"] = industry

    sizes = [
        'Small (1-50 employees)',
        'Medium (51-200 employees)',
        'Large (201-1,000 employees)',
        'Enterprise (1,001-10,000 employees)',
        'Large Enterprise (10,000+ employees)',
    ]
    persisted_size = st.session_state.get("company_size")
    company_size = st.selectbox(
        "Select your company's size:",
        sizes,
        index=sizes.index(persisted_size) if persisted_size in sizes else None,
        placeholder="Select Company Size",
    )
    st.session_state["company_size"] = company_size

    # Mirror the sidebar selections into the URL so a refresh restores them.
    # API keys are intentionally excluded — see core/state.py.
    sync_to_query_params()

    st.sidebar.markdown("---")

    st.sidebar.markdown("### <span style='color: #1DB954;'>About</span>", unsafe_allow_html=True)

    st.sidebar.markdown("Created by [Matt Adams](https://www.linkedin.com/in/matthewrwadams)")

    st.sidebar.markdown(
        "⭐ Star on GitHub: [![Star on GitHub](https://img.shields.io/github/stars/mrwadams/attackgen?style=social)](https://github.com/mrwadams/attackgen)"
    )


# ------------------ Main App UI ------------------ #

st.markdown("# <span style='color: #1DB954;'>AttackGen 👾</span>", unsafe_allow_html=True)
st.markdown("<span style='color: #1DB954;'> **Use MITRE ATT&CK, ATLAS and Large Language Models to generate attack scenarios for incident response testing.**</span>", unsafe_allow_html=True)
st.markdown("---")

st.markdown("""
            ### Welcome to AttackGen!

            The MITRE ATT&CK and ATLAS frameworks are powerful tools for understanding the tactics, techniques, and procedures (TTPs) used by threat actors targeting traditional IT/OT systems and AI/ML systems respectively; however, it can be difficult to translate this information into realistic scenarios for testing.

            AttackGen solves this problem by using large language models to quickly generate attack scenarios based on threat actor groups, documented case studies, or custom technique selections.

            AttackGen also generates **AI Insider Threat Scenarios** — incident response exercises in which a frontier AI agent deployed inside your organisation behaves as an insider threat. These are based on the threat model from [*Actions Speak Louder Than Tokens: An Insider Threat Model for Frontier AI Agents*](https://ai-insider-threat.matt-adams.co.uk).
            """)

st.markdown("""
            ### Getting Started

            1. From the sidebar, pick your model provider, enter the API key (if required), and choose a model.
            2. Select your industry, company size, and MITRE framework (ATT&CK Enterprise, ICS, or ATLAS).
            3. Open the `Threat Group Scenarios` page to generate a scenario based on a threat actor group or ATLAS case study, or the `Custom Scenarios` page to generate one from your own selection of techniques.
            4. Use the `AttackGen Assistant` to refine the generated scenario, or to ask wider questions about incident response testing.

            **Running a local model?** Pick the **Custom** provider and point the base URL at your OpenAI-compatible endpoint (e.g. `http://localhost:11434/v1` for Ollama, `http://localhost:1234/v1` for LM Studio), then type the model name your runtime expects.
            """)

st.markdown("""
            💡 Looking to test your response to **AI agents acting as insider threats**? Head to the `AI Insider Threat Scenarios` page to generate exercises based on an agent's deployment autonomy, threat category, and STRIDE threats — no MITRE matrix selection required.
            """)
