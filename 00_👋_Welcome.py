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

import streamlit as st

from core.sidebar import render_setup_sidebar
from core.styles import inject_emoji_fonts

# ------------------ Streamlit UI Configuration ------------------ #

st.set_page_config(
    page_title="AttackGen",
    page_icon="👾",
)
inject_emoji_fonts()
render_setup_sidebar()


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
