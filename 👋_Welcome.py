"""
Copyright (C) 2023, Matthew Adams

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


st.set_page_config(
    page_title="AttackGen",
    page_icon="👾",
)

with st.sidebar:
    st.sidebar.markdown("### <span style='color: #1DB954;'>Setup</span>", unsafe_allow_html=True)
    # Add toggle to select Azure OpenAI Service
    use_azure = st.toggle('Use Azure OpenAI Service', key='toggle_azure')
    st.session_state["use_azure"] = use_azure
    
    if use_azure:
        # Add Azure OpenAI API key input field to the sidebar
        st.session_state["AZURE_OPENAI_API_KEY"] = st.text_input(
            "Azure OpenAI API key:",
            type="password",
            help="You can find your Azure OpenAI API key on the [Azure portal](https://portal.azure.com/).",
        )
        
        # Add Azure OpenAI endpoint input field to the sidebar
        st.session_state["AZURE_OPENAI_ENDPOINT"] = st.text_input(
            "Azure OpenAI endpoint:",
            help="Example endpoint: https://YOUR_RESOURCE_NAME.openai.azure.com/",
        )

        # Add Azure OpenAI deployment name input field to the sidebar
        st.session_state["azure_deployment"] = st.text_input(
            "Deployment name:",
        )

        # Add API version dropdown selector to the sidebar
        st.session_state["openai_api_version"] = st.selectbox("API version:", ["2023-12-01-preview", "2023-05-15"], key="api_version", help="Select OpenAI API version used by your deployment.")

    else:     
        openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password", help="You can find your API key at https://platform.openai.com/account/api-keys")
        st.session_state["openai_api_key"] = openai_api_key

        # Add model selection input field to the sidebar
        model_name = st.selectbox(
            "Select the model you would like to use:",
            ["gpt-4-turbo-preview", "gpt-4", "gpt-3.5-turbo"],
            key="model",
            help="OpenAI have moved to continuous model upgrades so `gpt-3.5-turbo`, `gpt-4` and `gpt-4-turbo-preview` point to the latest available version of each model.",
        )
        st.session_state["model_name"] = model_name
    
    st.sidebar.markdown("---")

    # Add the drop-down selectors for Industry and Company Size
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
            'Transportation / Logistics'])
    , placeholder="Select Industry")
    st.session_state["industry"] = industry

    company_size = st.selectbox("Select your company's size:", ['Small (1-50 employees)', 'Medium (51-200 employees)', 'Large (201-1,000 employees)', 'Enterprise (1,001-10,000 employees)', 'Large Enterprise (10,000+ employees)'], placeholder="Select Company Size")
    st.session_state["company_size"] = company_size

    st.sidebar.markdown("---")

    st.sidebar.markdown("### <span style='color: #1DB954;'>About</span>", unsafe_allow_html=True)        
    
    st.sidebar.markdown("Created by [Matt Adams](https://www.linkedin.com/in/matthewrwadams)")

    st.sidebar.markdown(
        "⭐ Star on GitHub: [![Star on GitHub](https://img.shields.io/github/stars/mrwadams/attackgen?style=social)](https://github.com/mrwadams/attackgen)"
    )



st.markdown("# <span style='color: #1DB954;'>AttackGen 👾</span>", unsafe_allow_html=True)
st.markdown("<span style='color: #1DB954;'> **Use MITRE ATT&CK and Large Language Models to generate attack scenarios for incident response testing.**</span>", unsafe_allow_html=True)
st.markdown("---")

st.markdown("""          
            ### Welcome to AttackGen!
            
            The MITRE ATT&CK framework is a powerful tool for understanding the tactics, techniques, and procedures (TTPs) used by threat actors; however, it can be difficult to translate this information into realistic scenarios for testing.

            AttackGen solves this problem by using large language models to quickly generate attack scenarios based on a selection of a threat actor group's known techniques.
            """)

if st.session_state.get('use_azure', True):
    st.markdown("""          
            ### Getting Started

            1. Enter your Azure OpenAI Service API key, endpoint, and deployment name, then select your preferred model, industry, and company size from the sidebar. 
            2. Go to the `Threat Group Scenarios` page to generate a scenario based on a threat actor group's known techniques, or go to the `Custom Scenarios` page to generate a scenario based on your own selection of ATT&CK techniques.
            """)

else:
    st.markdown("""
            ### Getting Started

            1. Enter your OpenAI API key, then select your preferred model, industry, and company size from the sidebar. 
            2. Go to the `Threat Group Scenarios` page to generate a scenario based on a threat actor group's known techniques, or go to the `Custom Scenarios` page to generate a scenario based on your own selection of ATT&CK techniques.
            """)