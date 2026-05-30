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

# Load environment variables from .env file
load_dotenv()

# ------------------ Streamlit UI Configuration ------------------ #

st.set_page_config(
    page_title="AttackGen",
    page_icon="👾",
)


# ------------------ Sidebar ------------------ #

with st.sidebar:
    st.sidebar.markdown("### <span style='color: #1DB954;'>Setup</span>", unsafe_allow_html=True)

    model_provider = st.selectbox(
        "Select your preferred model provider:",
        list(PROVIDERS.keys()),
        key="chosen_model_provider",
        help="Select the model provider you would like to use. This will determine the models available for selection.",
    )

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
        env_base = os.getenv("CUSTOM_BASE_URL")
        st.session_state["llm_api_base"] = st.text_input(
            "Base URL:",
            value=env_base or provider_info.default_api_base or "",
            help="Base URL of your OpenAI-compatible endpoint. Example: http://localhost:11434/v1 for Ollama, http://localhost:1234/v1 for LM Studio.",
        )
    else:
        st.session_state["llm_api_base"] = None

    # ---- Model selection ----
    models = get_models_for_provider(model_provider)
    if models:
        labels = [m.model_id for m in models]
        help_map = {m.model_id: m.help_text for m in models}
        chosen = st.selectbox(
            "Select the model you would like to use:",
            labels,
            key="selected_model",
            help="\n".join(f"**{mid}** — {help_map[mid] or 'No description.'}" for mid in labels),
        )
        st.session_state["llm_model_name"] = chosen
    else:
        # Custom: user types the model id
        env_model = os.getenv("CUSTOM_MODEL_NAME")
        st.session_state["llm_model_name"] = st.text_input(
            "Model name:",
            value=env_model or "",
            help="Model identifier as expected by your endpoint (e.g. 'llama3.1', 'qwen3:32b').",
        )

    st.markdown("""---""")

    matrix = st.sidebar.radio(
        "Select MITRE Framework:",
        ["Enterprise", "ICS", "ATLAS"],
        key="selected_matrix",
        help="Enterprise and ICS are ATT&CK matrices for traditional IT and industrial control systems. ATLAS focuses on adversarial threats to AI/ML systems."
    )
    st.session_state["matrix"] = matrix

    industry = st.selectbox(
        "Select your company's industry:",
        sorted(['Aerospace / Defense', 'Agriculture / Food Services',
                'Automotive', 'Construction', 'Education',
                'Energy / Utilities', 'Finance / Banking',
                'Government / Public Sector', 'Healthcare',
                'Hospitality / Tourism', 'Insurance',
                'Legal Services', 'Manufacturing',
                'Media / Entertainment', 'Non-profit',
                'Real Estate', 'Retail / E-commerce',
                'Technology / IT', 'Telecommunication',
                'Transportation / Logistics']),
        placeholder="Select Industry",
    )
    st.session_state["industry"] = industry

    company_size = st.selectbox(
        "Select your company's size:",
        ['Small (1-50 employees)', 'Medium (51-200 employees)', 'Large (201-1,000 employees)', 'Enterprise (1,001-10,000 employees)', 'Large Enterprise (10,000+ employees)'],
        placeholder="Select Company Size",
    )
    st.session_state["company_size"] = company_size

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
