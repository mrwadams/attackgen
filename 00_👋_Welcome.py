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
import requests
import os
from dotenv import load_dotenv

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
    # Add model selection input field to the sidebar
    model_provider = st.selectbox(
        "Select your preferred model provider:",
        ["OpenAI API", "Anthropic API", "Azure OpenAI Service", "Google AI API", "Mistral API", "Groq API", "Ollama", "Custom"],
        key="model_provider",
        help="Select the model provider you would like to use. This will determine the models available for selection.",
    )

    # Save the selected model provider to the session state
    st.session_state["chosen_model_provider"] = model_provider

    if model_provider == "Custom":
        # Add input fields for custom API configuration
        st.session_state["custom_api_key"] = st.text_input(
            "Enter your API key:",
            type="password",
            help="Enter the API key for your custom model provider.",
        )

        st.session_state["custom_model_name"] = st.text_input(
            "Enter the model name:",
            help="Enter the model name for your custom model provider.",
        )

        st.session_state["custom_base_url"] = st.text_input(
            "Enter the base URL:",
            help="Enter the base URL for your custom provider (e.g., http://localhost:1234/v1). Must be OpenAI API compatible.",
        )

    if model_provider == "OpenAI API":
        # Check if OpenAI API key is in environment variables
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            # Add OpenAI API key input field to the sidebar if not in environment
            st.session_state["openai_api_key"] = st.text_input(
                "Enter your OpenAI API key:",
                type="password",
                help="You can find your OpenAI API key on the [OpenAI dashboard](https://platform.openai.com/account/api-keys).",
            )
        else:
            st.session_state["openai_api_key"] = openai_api_key
            st.success("API key loaded from .env")

        # Add model selection input field to the sidebar
        model_name = st.selectbox(
            "Select the model you would like to use:",
            ["gpt-5.2", "gpt-5-mini", "gpt-5-nano", "gpt-5.2-pro", "gpt-5", "gpt-4.1"],
            key="selected_model",
            help="GPT-5.2 is the best model for coding and agentic tasks. GPT-5.2 pro produces smarter, more precise responses. GPT-5-mini and nano are faster, cost-efficient versions. GPT-4.1 is the smartest non-reasoning model with 1M token context.",
        )
        st.session_state["model_name"] = model_name

    if model_provider == "Anthropic API":
        # Check if Anthropic API key is in environment variables
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if not anthropic_api_key:
            # Add Anthropic API key input field to the sidebar if not in environment
            st.session_state["anthropic_api_key"] = st.text_input(
                "Enter your Anthropic API key:",
                type="password",
                help="You can find your Anthropic API key on the [Anthropic console](https://console.anthropic.com/account/keys).",
            )
        else:
            st.session_state["anthropic_api_key"] = anthropic_api_key
            st.success("API key loaded from .env")

        # Add model selection input field to the sidebar
        model_name = st.selectbox(
            "Select the model you would like to use:",
            ["claude-sonnet-4-5-20250929", "claude-haiku-4-5-20251001", "claude-opus-4-5-20251101"],
            key="selected_model",
            help="Claude Sonnet 4.5 is the best balance of performance and cost. Claude Haiku 4.5 is the fastest option. Claude Opus 4.5 is the most capable model for complex tasks.",
        )
        st.session_state["anthropic_model"] = model_name

    if model_provider == "Azure OpenAI Service":
        # Check if Azure OpenAI API key is in environment variables
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if not azure_api_key:
            # Add Azure OpenAI API key input field to the sidebar if not in environment
            st.session_state["AZURE_OPENAI_API_KEY"] = st.text_input(
                "Azure OpenAI API key:",
                type="password",
                help="You can find your Azure OpenAI API key on the [Azure portal](https://portal.azure.com/).",
            )
        else:
            st.session_state["AZURE_OPENAI_API_KEY"] = azure_api_key
            st.success("API key loaded from .env")
        
        # Check if Azure OpenAI endpoint is in environment variables
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if not azure_endpoint:
            # Add Azure OpenAI endpoint input field to the sidebar if not in environment
            st.session_state["AZURE_OPENAI_ENDPOINT"] = st.text_input(
                "Azure OpenAI endpoint:",
                help="Example endpoint: https://YOUR_RESOURCE_NAME.openai.azure.com/",
            )
        else:
            st.session_state["AZURE_OPENAI_ENDPOINT"] = azure_endpoint
            st.success("Endpoint loaded from .env")

        # Add Azure OpenAI deployment name input field to the sidebar
        azure_deployment = os.getenv("AZURE_DEPLOYMENT")
        if not azure_deployment:
            st.session_state["AZURE_DEPLOYMENT"] = st.text_input(
                "Deployment name:",
                help="This is the name of your Azure OpenAI deployment.",
            )
        else:
            st.session_state["AZURE_DEPLOYMENT"] = azure_deployment
            st.success("Deployment name loaded from .env")
        
        # Add API version dropdown selector to the sidebar
        st.session_state["openai_api_version"] = st.selectbox("API version:", ["2023-12-01-preview", "2023-05-15"], key="api_version", help="Select OpenAI API version used by your deployment.")

    if model_provider == "Google AI API":
        # Check if Google API key is in environment variables
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            # Add Google API key input field to the sidebar if not in environment
            st.session_state["GOOGLE_API_KEY"] = st.text_input(
                "Enter your Google AI API key:",
                type="password",
                help="You can generate a Google AI API key in the [Google AI Studio](https://makersuite.google.com/app/apikey).",
            )
        else:
            st.session_state["GOOGLE_API_KEY"] = google_api_key
            st.success("API key loaded from .env")

        # Add model selection input field to the sidebar
        st.session_state["google_model"] = st.selectbox(
            "Select the model you would like to use:",
            ["gemini-3-pro-preview", "gemini-3-flash-preview", "gemini-2.5-flash", "gemini-2.5-flash-lite"],
            key="selected_model",
            help="Gemini 3 Pro is the most capable model. Gemini 3 Flash offers a good balance. Gemini 2.5 Flash and Flash Lite are faster, cost-efficient options.",
        )

    if model_provider == "Groq API":
        # Check if Groq API key is in environment variables
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            # Add Groq API key input field to the sidebar if not in environment
            st.session_state["GROQ_API_KEY"] = st.text_input(
                "Enter your Groq API key:",
                type="password",
                help="You can find your Groq API key in the [Groq Console](https://console.groq.com/keys).",
            )
        else:
            st.session_state["GROQ_API_KEY"] = groq_api_key
            st.success("API key loaded from .env")

        # Add model selection input field to the sidebar
        st.session_state["groq_model"] = st.selectbox(
            "Select the model you would like to use:",
            ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
            key="selected_model",
            help="GPT-OSS 120B is the most capable model. GPT-OSS 20B is a smaller, faster option. Llama 3.3 70B offers strong performance. Llama 3.1 8B is the fastest option.",
        )

    if model_provider == "Mistral API":
        # Check if Mistral API key is in environment variables
        mistral_api_key = os.getenv("MISTRAL_API_KEY")
        if not mistral_api_key:
            # Add Mistral API key input field to the sidebar if not in environment
            st.session_state["MISTRAL_API_KEY"] = st.text_input(
                "Enter your Mistral API key:",
                type="password",
                help="You can generate a Mistral API key in the [Mistral console](https://console.mistral.ai/api-keys/).",
            )
        else:
            st.session_state["MISTRAL_API_KEY"] = mistral_api_key
            st.success("API key loaded from .env")

        # Add model selection input field to the sidebar
        st.session_state["mistral_model"] = st.selectbox(
            "Select the model you would like to use:",
            ["mistral-large-2512", "mistral-medium-2508", "mistral-small-2506", "ministral-14b-2512"],
            key="selected_model",
            help="Mistral Large is the most capable model. Mistral Medium and Small offer good performance at lower cost. Ministral 14B is a compact, efficient option.",
        )

    if model_provider == "Ollama":
        # Make a request to the Ollama API to get the list of available models
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            response.raise_for_status()  # Raises a HTTPError if the status is 4xx, 5xx
        except requests.exceptions.RequestException as e:
            st.error("Ollama endpoint not found, please select a different model provider.")
            response = None

        if response:
            data = response.json()
            available_models = [model["name"] for model in data["models"]]
            # Add model selection input field to the sidebar
            ollama_model = st.selectbox(
            "Select the model you would like to use:",
            available_models,
            key="selected_model",
            )
            st.session_state["ollama_model"] = ollama_model

    st.markdown("""---""")

    matrix = st.sidebar.radio(
        "Select MITRE Framework:",
        ["Enterprise", "ICS", "ATLAS"],
        key="selected_matrix",
        help="Enterprise and ICS are ATT&CK matrices for traditional IT and industrial control systems. ATLAS focuses on adversarial threats to AI/ML systems."
    )
    st.session_state["matrix"] = matrix

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


# ------------------ Main App UI ------------------ #

st.markdown("# <span style='color: #1DB954;'>AttackGen 👾</span>", unsafe_allow_html=True)
st.markdown("<span style='color: #1DB954;'> **Use MITRE ATT&CK, ATLAS and Large Language Models to generate attack scenarios for incident response testing.**</span>", unsafe_allow_html=True)
st.markdown("---")

st.markdown("""
            ### Welcome to AttackGen!

            The MITRE ATT&CK and ATLAS frameworks are powerful tools for understanding the tactics, techniques, and procedures (TTPs) used by threat actors targeting traditional IT/OT systems and AI/ML systems respectively; however, it can be difficult to translate this information into realistic scenarios for testing.

            AttackGen solves this problem by using large language models to quickly generate attack scenarios based on threat actor groups, documented case studies, or custom technique selections.
            """)

if st.session_state.get('chosen_model_provider') == "Azure OpenAI Service":
    st.markdown("""          
            ### Getting Started

            1. Enter the details of your Azure OpenAI Service model deployment, including the API key, endpoint, deployment name, and API version. 
            2. Select your industry, company size, and MITRE framework (ATT&CK Enterprise, ICS, or ATLAS) from the sidebar.
            3. Go to the `Threat Group Scenarios` page to generate a scenario based on a threat actor group or ATLAS case study, or go to the `Custom Scenarios` page to generate a scenario based on your own selection of techniques.
            4. Use `AttackGen Assistant` to refine / update the generated scenario, or ask more general questions about incident response testing.
            """)

elif st.session_state.get('chosen_model_provider') == "Anthropic API":
    st.markdown("""          
            ### Getting Started

            1. Enter your Anthropic API key, then select your preferred Claude model, industry, company size, and MITRE framework from the sidebar.
            2. Go to the `Threat Group Scenarios` page to generate a scenario based on a threat actor group or ATLAS case study, or go to the `Custom Scenarios` page to generate a scenario based on your own selection of techniques.
            3. Use `AttackGen Assistant` to refine / update the generated scenario, or ask more general questions about incident response testing.
            """)
    
elif st.session_state.get('chosen_model_provider') == "Google AI API":
    st.markdown("""          
            ### Getting Started

            1. Enter your Google AI API key, then select your preferred model, industry, company size, and MITRE framework from the sidebar.
            2. Go to the `Threat Group Scenarios` page to generate a scenario based on a threat actor group or ATLAS case study, or go to the `Custom Scenarios` page to generate a scenario based on your own selection of techniques.
            3. Use `AttackGen Assistant` to refine / update the generated scenario, or ask more general questions about incident response testing.
            """)
    
elif st.session_state.get('chosen_model_provider') == "Mistral API":
    st.markdown("""          
            ### Getting Started

            1. Enter your Mistral API key, then select your preferred model, industry, company size, and MITRE framework from the sidebar.
            2. Go to the `Threat Group Scenarios` page to generate a scenario based on a threat actor group or ATLAS case study, or go to the `Custom Scenarios` page to generate a scenario based on your own selection of techniques.
            3. Use `AttackGen Assistant` to refine / update the generated scenario, or ask more general questions about incident response testing.
            """)

elif st.session_state.get('chosen_model_provider') == "Ollama":
    st.markdown("""          
            ### Getting Started

            1. Select your locally hosted model from the sidebar, then select your industry, company size, and MITRE framework.
            2. Go to the `Threat Group Scenarios` page to generate a scenario based on a threat actor group or ATLAS case study, or go to the `Custom Scenarios` page to generate a scenario based on your own selection of techniques.
            3. Use `AttackGen Assistant` to refine / update the generated scenario, or ask more general questions about incident response testing.
            """)

elif st.session_state.get('chosen_model_provider') == "Groq API":
    st.markdown("""          
            ### Getting Started

            1. Enter your Groq API key, then select your preferred model, industry, company size, and MITRE framework from the sidebar.
            2. Go to the `Threat Group Scenarios` page to generate a scenario based on a threat actor group or ATLAS case study, or go to the `Custom Scenarios` page to generate a scenario based on your own selection of techniques.
            3. Use `AttackGen Assistant` to refine / update the generated scenario, or ask more general questions about incident response testing.
            """)
    
elif st.session_state.get('chosen_model_provider') == "Custom":
    st.markdown("""          
            ### Getting Started

            1. Enter your custom model provider's API key (if required), base URL, and model name.
            2. Select your industry, company size, and MITRE framework from the sidebar.
            3. Go to the `Threat Group Scenarios` page to generate a scenario based on a threat actor group or ATLAS case study, or go to the `Custom Scenarios` page to generate a scenario based on your own selection of techniques.
            4. Use `AttackGen Assistant` to refine / update the generated scenario, or ask more general questions about incident response testing.
            """)

else:
    st.markdown("""
            ### Getting Started

            1. Enter your OpenAI API key, then select your preferred model, industry, company size, and MITRE framework from the sidebar.
            2. Go to the `Threat Group Scenarios` page to generate a scenario based on a threat actor group or ATLAS case study, or go to the `Custom Scenarios` page to generate a scenario based on your own selection of techniques.
            3. Use `AttackGen Assistant` to refine / update the generated scenario, or ask more general questions about incident response testing.
            """)