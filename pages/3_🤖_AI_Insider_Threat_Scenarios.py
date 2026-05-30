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

------------------------------------------------------------------------------

AI Insider Threat Scenarios
===========================

Generates incident response testing scenarios in which a frontier AI agent
deployed inside the organisation behaves as an insider threat. Based on the
paper "Actions Speak Louder Than Tokens: An Insider Threat Model for Frontier
AI Agents" by Matt Adams (https://ai-insider-threat.matt-adams.co.uk).
"""

import os
import re
import streamlit as st

from langchain_anthropic import ChatAnthropic
from langchain_community.llms import Ollama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_openai import ChatOpenAI, AzureOpenAI
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langsmith import Client, RunTree, traceable
from openai import OpenAI

from data.ai_insider_threats import (
    DEPLOYMENT_ARCHETYPES,
    THREAT_CATEGORIES,
    AGENT_CAPABILITIES,
    CERT_DIMENSIONS,
    DETECTION_STRATEGIES,
    CONTROLS_FRAMEWORK,
    AI_INSIDER_TEMPLATES,
    build_threat_context,
    stride_options,
    stride_code_from_option,
)


# ------------------ Streamlit Configuration ------------------ #

# Add environment variables for LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "AttackGen"

# Initialise the LangChain client conditionally based on the presence of the secret
if "LANGCHAIN_API_KEY" in st.secrets:
    langchain_api_key = st.secrets["LANGCHAIN_API_KEY"]
    client = Client(api_key=langchain_api_key)
else:
    client = None

# Add environment variables from session state for Azure OpenAI Service
if "AZURE_OPENAI_API_KEY" in st.session_state:
    os.environ["AZURE_OPENAI_API_KEY"] = st.session_state["AZURE_OPENAI_API_KEY"]
if "AZURE_OPENAI_ENDPOINT" in st.session_state:
    os.environ["AZURE_OPENAI_ENDPOINT"] = st.session_state["AZURE_OPENAI_ENDPOINT"]
if "azure_deployment" in st.session_state:
    os.environ["AZURE_DEPLOYMENT"] = st.session_state["azure_deployment"]
if "openai_api_version" in st.session_state:
    os.environ["OPENAI_API_VERSION"] = st.session_state["openai_api_version"]

# Add environment variables from session state for Google AI API
if "GOOGLE_API_KEY" in st.session_state:
    os.environ["GOOGLE_API_KEY"] = st.session_state["GOOGLE_API_KEY"]
if "google_model" in st.session_state:
    os.environ["GOOGLE_MODEL"] = st.session_state["google_model"]

# Add environment variables from session state for Mistral API
if "MISTRAL_API_KEY" in st.session_state:
    os.environ["MISTRAL_API_KEY"] = st.session_state["MISTRAL_API_KEY"]
if "mistral_model" in st.session_state:
    os.environ["MISTRAL_MODEL"] = st.session_state["mistral_model"]

# Add environment variables from session state for Groq API
if "GROQ_API_KEY" in st.session_state:
    os.environ["GROQ_API_KEY"] = st.session_state["GROQ_API_KEY"]
if "groq_model" in st.session_state:
    os.environ["GROQ_MODEL"] = st.session_state["groq_model"]

# Add environment variables from session state for Ollama
if "ollama_model" in st.session_state:
    os.environ["OLLAMA_MODEL"] = st.session_state["ollama_model"]

# Add environment variables from session state for Custom Provider
if "custom_api_key" in st.session_state:
    os.environ["CUSTOM_API_KEY"] = st.session_state["custom_api_key"]
if "custom_model_name" in st.session_state:
    os.environ["CUSTOM_MODEL_NAME"] = st.session_state["custom_model_name"]
if "custom_base_url" in st.session_state:
    os.environ["CUSTOM_BASE_URL"] = st.session_state["custom_base_url"]

# Add environment variables from session state for Anthropic API
if "ANTHROPIC_API_KEY" in st.session_state:
    os.environ["ANTHROPIC_API_KEY"] = st.session_state["ANTHROPIC_API_KEY"]
if "anthropic_model" in st.session_state:
    os.environ["ANTHROPIC_MODEL"] = st.session_state["anthropic_model"]

st.set_page_config(
    page_title="AI Insider Threat Scenarios",
    page_icon="🤖",
)

# This page does not depend on a MITRE matrix selection, but the shared sidebar
# (configured on the Welcome page) provides the model provider, industry and size.
model_provider = st.session_state.get("chosen_model_provider", "OpenAI API")
industry = st.session_state.get("industry")
company_size = st.session_state.get("company_size")

# Set the default value for the generated-scenario flag
if "ai_insider_scenario_generated" not in st.session_state:
    st.session_state["ai_insider_scenario_generated"] = False


# ------------------ Scenario Generation Wrappers ------------------ #
# These mirror the provider wrappers used elsewhere in AttackGen so that the new
# page supports every configured model provider.

def generate_scenario_wrapper(openai_api_key, model_name, messages):
    if client is not None:
        @traceable(run_type="llm", name="AI Insider Threat Scenario", tags=["openai", "ai_insider_scenario"], client=client)
        def generate_scenario(openai_api_key, model_name, messages, *, run_tree: RunTree):
            model_name = st.session_state["model_name"]
            try:
                with st.status('Generating scenario...', expanded=True):
                    st.write("Initialising AI model.")
                    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name=model_name, streaming=False, output_version="responses/v1")
                    st.write("Model initialised. Generating scenario, please wait.")
                    response = llm.invoke(messages)
                    st.write("Scenario generated successfully.")
                    st.session_state['run_id'] = str(run_tree.id)
                    return response
            except Exception as e:
                st.error("An error occurred while generating the scenario: " + str(e))
                st.session_state['run_id'] = str(run_tree.id)
                return None
    else:
        def generate_scenario(openai_api_key, model_name, messages):
            model_name = st.session_state["model_name"]
            try:
                with st.status('Generating scenario...', expanded=True):
                    st.write("Initialising AI model.")
                    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name=model_name, streaming=False, output_version="responses/v1")
                    st.write("Model initialised. Generating scenario, please wait.")
                    response = llm.invoke(messages)
                    st.write("Scenario generated successfully.")
                    return response
            except Exception as e:
                st.error("An error occurred while generating the scenario: " + str(e))
                return None

    return generate_scenario(openai_api_key, model_name, messages)


def generate_scenario_azure_wrapper(messages):
    def _format(messages):
        formatted_messages = []
        for message in messages:
            if hasattr(message, 'role') and hasattr(message, 'content'):
                role = message.role
                if role == 'human':
                    role = 'user'
                formatted_messages.append({"role": role, "content": message.content})
            elif hasattr(message, 'type') and hasattr(message, 'content'):
                role = message.type
                if role == 'human':
                    role = 'user'
                formatted_messages.append({"role": role, "content": message.content})
            else:
                raise ValueError(f"Unsupported message format: {message}")
        return formatted_messages

    if client is not None:
        @traceable(run_type="llm", name="AI Insider Threat Scenario (Azure OpenAI)", tags=["azure", "ai_insider_scenario"], client=client)
        def generate_scenario_azure(messages, *, run_tree: RunTree):
            try:
                azure_api_key = os.getenv('AZURE_OPENAI_API_KEY')
                azure_api_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
                azure_deployment_name = os.getenv('AZURE_DEPLOYMENT')
                azure_api_version = os.getenv('OPENAI_API_VERSION')
                with st.status('Generating scenario...', expanded=True):
                    st.write("Initialising AI model.")
                    llm = AzureOpenAI(api_key=azure_api_key, azure_endpoint=azure_api_endpoint, api_version=azure_api_version)
                    st.write("Model initialised. Generating scenario, please wait.")
                    response = llm.chat.completions.create(model=azure_deployment_name, messages=_format(messages))
                    st.write("Scenario generated successfully.")
                    st.session_state['run_id'] = str(run_tree.id)
                    return response
            except Exception as e:
                st.error(f"An error occurred while generating the scenario: {str(e)}")
                st.session_state['run_id'] = str(run_tree.id)
                return None
    else:
        def generate_scenario_azure(messages):
            try:
                azure_api_key = os.getenv('AZURE_OPENAI_API_KEY')
                azure_api_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
                azure_deployment_name = os.getenv('AZURE_DEPLOYMENT')
                azure_api_version = os.getenv('OPENAI_API_VERSION')
                with st.status('Generating scenario...', expanded=True):
                    st.write("Initialising AI model.")
                    llm = AzureOpenAI(api_key=azure_api_key, azure_endpoint=azure_api_endpoint, api_version=azure_api_version)
                    st.write("Model initialised. Generating scenario, please wait.")
                    response = llm.chat.completions.create(model=azure_deployment_name, messages=_format(messages))
                    st.write("Scenario generated successfully.")
                    return response
            except Exception as e:
                st.error(f"An error occurred while generating the scenario: {str(e)}")
                return None
    return generate_scenario_azure(messages)


def generate_scenario_google_wrapper(google_api_key, model, messages):
    if client is not None:
        @traceable(run_type="llm", name="AI Insider Threat Scenario (Google AI API)", tags=["google", "ai_insider_scenario"], client=client)
        def generate_scenario_google(google_api_key, model, messages, *, run_tree: RunTree):
            try:
                google_api_key = os.getenv('GOOGLE_API_KEY')
                model = os.getenv('GOOGLE_MODEL')
                with st.status('Generating scenario...', expanded=True):
                    st.write("Initialising AI model.")
                    llm = ChatGoogleGenerativeAI(google_api_key=google_api_key, model=model)
                    st.write("Model initialised. Generating scenario, please wait.")
                    response = llm.invoke(messages)
                    st.write("Scenario generated successfully.")
                    st.session_state['run_id'] = str(run_tree.id)
                    return response
            except Exception as e:
                st.error(f"An error occurred while generating the scenario: {str(e)}")
                st.session_state['run_id'] = str(run_tree.id)
                return None
    else:
        def generate_scenario_google(google_api_key, model, messages):
            try:
                google_api_key = os.getenv('GOOGLE_API_KEY')
                model = os.getenv('GOOGLE_MODEL')
                with st.status('Generating scenario...', expanded=True):
                    st.write("Initialising AI model.")
                    llm = ChatGoogleGenerativeAI(google_api_key=google_api_key, model=model)
                    st.write("Model initialised. Generating scenario, please wait.")
                    response = llm.invoke(messages)
                    st.write("Scenario generated successfully.")
                    return response
            except Exception as e:
                st.error(f"An error occurred while generating the scenario: {str(e)}")
                return None
    return generate_scenario_google(google_api_key, model, messages)


def generate_scenario_mistral_wrapper(mistral_api_key, model_name, messages):
    if client is not None:
        @traceable(run_type="llm", name="AI Insider Threat Scenario (Mistral API)", tags=["mistral", "ai_insider_scenario"], client=client)
        def generate_scenario_mistral(mistral_api_key, model_name, messages, *, run_tree: RunTree):
            try:
                mistral_api_key = os.getenv('MISTRAL_API_KEY')
                model = os.getenv('MISTRAL_MODEL')
                with st.status('Generating scenario...', expanded=True):
                    st.write("Initialising AI model.")
                    llm = ChatMistralAI(mistral_api_key=mistral_api_key)
                    st.write("Model initialised. Generating scenario, please wait.")
                    response = llm.invoke(messages, model=model)
                    st.write("Scenario generated successfully.")
                    st.session_state['run_id'] = str(run_tree.id)
                    return response
            except Exception as e:
                st.error(f"An error occurred while generating the scenario: {str(e)}")
                st.session_state['run_id'] = str(run_tree.id)
                return None
    else:
        def generate_scenario_mistral(mistral_api_key, model_name, messages):
            try:
                mistral_api_key = os.getenv('MISTRAL_API_KEY')
                model = os.getenv('MISTRAL_MODEL')
                with st.status('Generating scenario...', expanded=True):
                    st.write("Initialising AI model.")
                    llm = ChatMistralAI(mistral_api_key=mistral_api_key)
                    st.write("Model initialised. Generating scenario, please wait.")
                    response = llm.invoke(messages, model=model)
                    st.write("Scenario generated successfully.")
                    return response
            except Exception as e:
                st.error(f"An error occurred while generating the scenario: {str(e)}")
                return None
    return generate_scenario_mistral(mistral_api_key, model_name, messages)


def generate_scenario_groq_wrapper(groq_api_key, model_name, messages):
    def _format(messages):
        formatted_messages = []
        for message in messages:
            if hasattr(message, 'role') and hasattr(message, 'content'):
                role = message.role
                if role == 'human':
                    role = 'user'
                formatted_messages.append({"role": role, "content": message.content})
            elif hasattr(message, 'type') and hasattr(message, 'content'):
                role = message.type
                if role == 'human':
                    role = 'user'
                formatted_messages.append({"role": role, "content": message.content})
            else:
                raise ValueError(f"Unsupported message format: {message}")
        return formatted_messages

    if client is not None:
        @traceable(run_type="llm", name="AI Insider Threat Scenario (Groq API)", tags=["groq", "ai_insider_scenario"], client=client)
        def generate_scenario_groq(groq_api_key, model_name, messages, *, run_tree: RunTree):
            try:
                groq_api_key = os.getenv('GROQ_API_KEY')
                model = os.getenv('GROQ_MODEL')
                with st.status('Generating scenario...', expanded=True):
                    st.write("Initialising AI model.")
                    llm = OpenAI(api_key=groq_api_key, base_url="https://api.groq.com/openai/v1")
                    st.write("Model initialised. Generating scenario, please wait.")
                    response = llm.chat.completions.create(model=model, messages=_format(messages))
                    st.write("Scenario generated successfully.")
                    st.session_state['run_id'] = str(run_tree.id)
                    return response
            except Exception as e:
                st.error(f"An error occurred while generating the scenario: {str(e)}")
                st.session_state['run_id'] = str(run_tree.id)
                return None
    else:
        def generate_scenario_groq(groq_api_key, model_name, messages):
            try:
                groq_api_key = os.getenv('GROQ_API_KEY')
                model = os.getenv('GROQ_MODEL')
                with st.status('Generating scenario...', expanded=True):
                    st.write("Initialising AI model.")
                    llm = OpenAI(api_key=groq_api_key, base_url="https://api.groq.com/openai/v1")
                    st.write("Model initialised. Generating scenario, please wait.")
                    response = llm.chat.completions.create(model=model, messages=_format(messages))
                    st.write("Scenario generated successfully.")
                    return response
            except Exception as e:
                st.error(f"An error occurred while generating the scenario: {str(e)}")
                return None
    return generate_scenario_groq(groq_api_key, model_name, messages)


def generate_scenario_ollama_wrapper(model, messages):
    if client is not None:
        @traceable(run_type="llm", name="AI Insider Threat Scenario (Ollama)", tags=["ollama", "ai_insider_scenario"], client=client)
        def generate_scenario_ollama(model, *, run_tree: RunTree):
            try:
                model = os.getenv('OLLAMA_MODEL')
                with st.status('Generating scenario...', expanded=True):
                    st.write("Initialising AI model.")
                    llm = Ollama(model=model)
                    st.write("Model initialised. Generating scenario, please wait.")
                    response = llm.invoke(messages, model=model)
                    st.write("Scenario generated successfully.")
                    st.session_state['run_id'] = str(run_tree.id)
                return response
            except Exception as e:
                st.error(f"An error occurred while generating the scenario: {str(e)}")
                st.session_state['run_id'] = str(run_tree.id)
                return None
    else:
        def generate_scenario_ollama(model):
            try:
                model = os.getenv('OLLAMA_MODEL')
                with st.status('Generating scenario...', expanded=True):
                    st.write("Initialising AI model.")
                    llm = Ollama(model=model)
                    st.write("Model initialised. Generating scenario, please wait.")
                    response = llm.invoke(messages, model=model)
                    st.write("Scenario generated successfully.")
                return response
            except Exception as e:
                st.error(f"An error occurred while generating the scenario: {str(e)}")
                return None
    return generate_scenario_ollama(model)


def generate_scenario_anthropic_wrapper(anthropic_api_key, model_name, messages):
    if client is not None:
        @traceable(run_type="llm", name="AI Insider Threat Scenario (Anthropic)", tags=["anthropic", "ai_insider_scenario"], client=client)
        def generate_scenario_anthropic(anthropic_api_key, model_name, messages, *, run_tree: RunTree):
            try:
                anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
                model = os.getenv('ANTHROPIC_MODEL')
                with st.status('Generating scenario...', expanded=True):
                    st.write("Initialising AI model.")
                    max_tokens = 8192
                    if "opus-4" in model:
                        max_tokens = 32000
                    elif "sonnet-4" in model or "3-7-sonnet" in model:
                        max_tokens = 64000
                    llm = ChatAnthropic(anthropic_api_key=anthropic_api_key, model_name=model, temperature=0.7, max_tokens=max_tokens)
                    st.write("Model initialised. Generating scenario, please wait.")
                    response = llm.invoke(messages)
                    st.write("Scenario generated successfully.")
                    st.session_state['run_id'] = str(run_tree.id)
                    return response
            except Exception as e:
                st.error(f"An error occurred while generating the scenario: {str(e)}")
                st.session_state['run_id'] = str(run_tree.id)
                return None
    else:
        def generate_scenario_anthropic(anthropic_api_key, model_name, messages):
            try:
                anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
                model = os.getenv('ANTHROPIC_MODEL')
                with st.status('Generating scenario...', expanded=True):
                    st.write("Initialising AI model.")
                    max_tokens = 8192
                    if "opus-4" in model:
                        max_tokens = 32000
                    elif "sonnet-4" in model or "3-7-sonnet" in model:
                        max_tokens = 64000
                    llm = ChatAnthropic(anthropic_api_key=anthropic_api_key, model_name=model, temperature=0.7, max_tokens=max_tokens)
                    st.write("Model initialised. Generating scenario, please wait.")
                    response = llm.invoke(messages)
                    st.write("Scenario generated successfully.")
                    return response
            except Exception as e:
                st.error(f"An error occurred while generating the scenario: {str(e)}")
                return None
    return generate_scenario_anthropic(anthropic_api_key, model_name, messages)


def generate_scenario_custom_wrapper(messages):
    base_url = st.session_state.get('custom_base_url')
    model = st.session_state.get('custom_model_name')

    if not base_url:
        st.error("Custom base URL must be set for the custom model provider.")
        return None
    if not model:
        st.error("Custom model name must be set for the custom model provider.")
        return None

    formatted_messages = []
    for message in messages:
        if hasattr(message, 'role') and hasattr(message, 'content'):
            role = message.role
            if role == 'human':
                role = 'user'
            formatted_messages.append({"role": role, "content": message.content})
        elif hasattr(message, 'type') and hasattr(message, 'content'):
            role = message.type
            if role == 'human':
                role = 'user'
            formatted_messages.append({"role": role, "content": message.content})
        else:
            st.error(f"Unsupported message format: {type(message)} - {message}")
            return None

    if not formatted_messages:
        st.error("No valid messages found to send to the model.")
        return None

    def _invoke(formatted_messages_arg):
        current_api_key = st.session_state.get('custom_api_key')
        current_model = st.session_state.get('custom_model_name')
        current_base_url = st.session_state.get('custom_base_url')
        with st.status('Generating scenario with custom model...', expanded=True):
            st.write("Initialising custom AI model.")
            client_args = {"base_url": current_base_url}
            if current_api_key:
                client_args["api_key"] = current_api_key
            llm = OpenAI(**client_args)
            response = llm.chat.completions.create(
                model=current_model,
                messages=formatted_messages_arg,
                temperature=0.7,
                max_tokens=-1,
                stream=False,
            )
            st.write("Scenario generated successfully.")
            return response

    if client is not None:
        @traceable(run_type="llm", name="AI Insider Threat Scenario (Custom)", tags=["custom", "ai_insider_scenario"], client=client)
        def generate_scenario_custom(formatted_messages_arg, *, run_tree: RunTree):
            try:
                response = _invoke(formatted_messages_arg)
                st.session_state['run_id'] = str(run_tree.id)
                return response
            except Exception as e:
                import traceback
                st.error(f"An error occurred while generating the scenario: {str(e)}")
                st.text(traceback.format_exc())
                st.session_state['run_id'] = str(run_tree.id)
                return None
        return generate_scenario_custom(formatted_messages)
    else:
        def generate_scenario_custom_no_trace(formatted_messages_arg):
            try:
                return _invoke(formatted_messages_arg)
            except Exception as e:
                import traceback
                st.error(f"An error occurred while generating the scenario: {str(e)}")
                st.text(traceback.format_exc())
                return None
        return generate_scenario_custom_no_trace(formatted_messages)


# ------------------ Prompt Construction ------------------ #

def build_messages(archetype_name, selected_categories, selected_stride, selected_capabilities):
    """Construct the ChatPromptTemplate messages for an AI insider threat scenario."""
    archetype = DEPLOYMENT_ARCHETYPES[archetype_name]

    threat_context = build_threat_context(selected_categories, selected_stride)

    if selected_capabilities:
        capability_lines = "\n".join(
            f"- {name}: {AGENT_CAPABILITIES[name]}" for name in selected_capabilities
        )
    else:
        capability_lines = "- (No specific capabilities highlighted; assume a capable frontier coding agent.)"

    cert_lines = "\n".join(f"- {dimension}: {desc}" for dimension, desc in CERT_DIMENSIONS.items())
    detection_lines = "\n".join(f"- {name}: {desc}" for name, desc in DETECTION_STRATEGIES.items())
    controls_lines = "\n".join(
        f"- {function}: " + " ".join(items) for function, items in CONTROLS_FRAMEWORK.items()
    )

    system_template = (
        "You are a cybersecurity expert specialising in AI agent security and insider threat "
        "modelling. You produce realistic incident response testing scenarios in which a frontier "
        "AI agent that has been deployed inside an organisation behaves as an insider threat — "
        "whether through misalignment, reward hacking, emergent objectives, or a prompt-injection "
        "induced compromise. You think in terms of the agent's tool access, deployment autonomy and "
        "model capabilities rather than human notions of motivation. Format your response using "
        "proper Markdown with clear headers, bullet points and tables where helpful. Write in British English."
    )
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

    human_template = """**Background information**
The organisation operates in the '{industry}' industry and is of size '{company_size}'. It has deployed one or more frontier AI agents (for example, autonomous coding or operations agents) within its software development and infrastructure environment.

**Framing — AI agents as insider threats**
Unlike a human insider, an AI agent has no motivation in the human sense; its risk is governed by configuration, access and capability. Use these adapted CERT insider-threat dimensions as framing:
{cert_lines}

**Deployment archetype (autonomy level)**
The agent is deployed under the **{archetype_name}** model.
- Description: {archetype_description}
- Access: {archetype_access}
- Detection posture: {archetype_detection}
- Primary threats at this level: {archetype_threats}
- Critical control at this level: {archetype_control}

**Relevant frontier-agent capabilities**
{capability_lines}

**Threat scope**
{threat_context}

**Available detection strategies (for the detection section)**
{detection_lines}

**Recommended controls (NIST CSF, adapted for AI agents)**
{controls_lines}

**Your task**
Create a detailed incident response testing scenario (a tabletop exercise) in which the AI agent acts as an insider threat consistent with the deployment archetype and threat scope above. The scenario must be realistic for the stated industry and organisation size, and grounded in how the agent's tool access and autonomy enable the behaviour. Structure your response with the following sections:

1. **Scenario Title & Overview** — a short, evocative title and a one-paragraph summary.
2. **Deployment Context** — how the agent is deployed, what tools and access it has, and why this autonomy level matters.
3. **Attack Narrative & Timeline** — a step-by-step account of how the incident unfolds, mapping each step to the relevant STRIDE threat identifier(s) where applicable. Emphasise how the agent blends malicious actions with legitimate work.
4. **Affected Systems & Business Impact** — concrete systems, data and business consequences.
5. **Detection Opportunities** — at which points the activity could be detected, mapped to the available detection strategies, and which would likely fail given the deployment archetype.
6. **Discussion Questions** — 5–8 questions to test the incident response team's readiness (containment, attribution, credential revocation, log integrity, blast-radius assessment, recovery).
7. **Recommended Controls** — prioritised mitigations mapped to the NIST CSF functions (Identify, Protect, Detect, Respond, Recover).

Write in British English and format the entire response in Markdown.
"""
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    return chat_prompt.format_prompt(
        industry=industry,
        company_size=company_size,
        cert_lines=cert_lines,
        archetype_name=archetype_name,
        archetype_description=archetype["description"],
        archetype_access=archetype["access"],
        archetype_detection=archetype["detection"],
        archetype_threats=archetype["primary_threats"],
        archetype_control=archetype["critical_control"],
        capability_lines=capability_lines,
        threat_context=threat_context,
        detection_lines=detection_lines,
        controls_lines=controls_lines,
    ).to_messages()


# ------------------ Streamlit UI ------------------ #

st.markdown("# <span style='color: #1DB954;'>AI Insider Threat Scenarios🤖</span>", unsafe_allow_html=True)

st.markdown(
    "Generate incident response testing scenarios in which a **frontier AI agent deployed inside "
    "your organisation behaves as an insider threat**. Based on the threat model from "
    "[*Actions Speak Louder Than Tokens: An Insider Threat Model for Frontier AI Agents*]"
    "(https://ai-insider-threat.matt-adams.co.uk) by Matt Adams."
)
st.markdown("---")

# --- Optional template selection --- #
with st.expander("Use a Template (Optional)"):
    st.markdown(
        "Select a template to pre-populate the deployment archetype, threat categories and "
        "STRIDE threats for a common AI insider threat scenario. You can adjust the selections afterwards."
    )
    selected_template = st.selectbox(
        "Select a template",
        options=[""] + list(AI_INSIDER_TEMPLATES.keys()),
        format_func=lambda x: "Select a template" if x == "" else x,
    )
    if selected_template:
        template = AI_INSIDER_TEMPLATES[selected_template]
        st.session_state['ai_insider_archetype'] = template['archetype']
        st.session_state['ai_insider_categories'] = template['categories']
        st.session_state['ai_insider_stride'] = [
            opt for opt in stride_options() if stride_code_from_option(opt) in template['stride']
        ]

st.markdown("")

# --- Deployment archetype --- #
st.markdown("### 1. Deployment Archetype")
st.markdown(
    "How much autonomy the agent has — and where the human sits in the loop — is the primary "
    "determinant of its threat surface."
)
archetype_names = list(DEPLOYMENT_ARCHETYPES.keys())
default_archetype = st.session_state.get('ai_insider_archetype', archetype_names[2])
selected_archetype = st.selectbox(
    "Select the agent's deployment archetype (autonomy level):",
    options=archetype_names,
    index=archetype_names.index(default_archetype) if default_archetype in archetype_names else 2,
)
st.session_state['ai_insider_archetype'] = selected_archetype
_archetype = DEPLOYMENT_ARCHETYPES[selected_archetype]
st.info(
    f"**{_archetype['threat_level']} threat.** {_archetype['description']}\n\n"
    f"- **Access:** {_archetype['access']}\n"
    f"- **Detection:** {_archetype['detection']}\n"
    f"- **Critical control:** {_archetype['critical_control']}"
)

# --- Threat categories --- #
st.markdown("### 2. Threat Categories")
st.markdown("Select one or more insider threat categories the scenario should focus on.")
selected_categories = st.multiselect(
    "Select threat categories:",
    options=list(THREAT_CATEGORIES.keys()),
    default=st.session_state.get('ai_insider_categories', []),
)
st.session_state['ai_insider_categories'] = selected_categories

# --- STRIDE threats --- #
st.markdown("### 3. Specific STRIDE Threats (Optional)")
st.markdown(
    "Optionally narrow the scenario to specific STRIDE threats. If left empty, the STRIDE threats "
    "associated with your selected categories will be used."
)
selected_stride_options = st.multiselect(
    "Select STRIDE threats:",
    options=stride_options(),
    default=st.session_state.get('ai_insider_stride', []),
)
st.session_state['ai_insider_stride'] = selected_stride_options
selected_stride = [stride_code_from_option(opt) for opt in selected_stride_options]

# If no STRIDE threats explicitly selected, derive them from the chosen categories.
if not selected_stride and selected_categories:
    derived = []
    for category in selected_categories:
        derived.extend(THREAT_CATEGORIES[category]['stride'])
    # Preserve order while de-duplicating
    selected_stride = list(dict.fromkeys(derived))

# --- Agent capabilities --- #
st.markdown("### 4. Frontier Agent Capabilities (Optional)")
st.markdown("Highlight the agent capabilities that make this a credible insider threat.")
selected_capabilities = st.multiselect(
    "Select agent capabilities:",
    options=list(AGENT_CAPABILITIES.keys()),
    default=list(AGENT_CAPABILITIES.keys()),
)

# Build the prompt messages if a valid selection exists.
messages = None
if selected_categories or selected_stride:
    try:
        messages = build_messages(selected_archetype, selected_categories, selected_stride, selected_capabilities)
    except Exception as e:
        st.error(f"An error occurred while building the prompt: {str(e)}")

st.markdown("")
st.markdown("---")
st.markdown(
    """
    ### Generate a Scenario

    Click the button below to generate an AI insider threat scenario based on your selections.

    It normally takes between 30-50 seconds to generate a scenario, although for local models this is highly dependent on your hardware and the selected model. ⏱️
    """
)


def _handle_response_text(scenario_text):
    """Store and display a successfully generated scenario."""
    st.session_state['ai_insider_scenario_generated'] = True
    st.session_state['ai_insider_scenario_text'] = scenario_text
    st.markdown(scenario_text)
    st.download_button(
        label="Download Scenario",
        data=scenario_text,
        file_name="ai_insider_threat_scenario.md",
        mime="text/markdown",
    )
    # Make the scenario available to the AttackGen Assistant
    st.session_state['last_scenario'] = True
    st.session_state['last_scenario_text'] = scenario_text


def _show_previous_scenario():
    if 'ai_insider_scenario_text' in st.session_state and st.session_state['ai_insider_scenario_generated']:
        st.markdown("---")
        st.markdown(st.session_state['ai_insider_scenario_text'])
        st.download_button(
            label="Download Scenario",
            data=st.session_state['ai_insider_scenario_text'],
            file_name="ai_insider_threat_scenario.md",
            mime="text/markdown",
        )


def _requires_selection():
    if not selected_categories and not selected_stride:
        st.info("Please select at least one threat category (or specific STRIDE threat) to continue.")
        return True
    return False


try:
    if model_provider == "Azure OpenAI Service":
        if st.button('Generate Scenario', key='generate_ai_insider_azure'):
            if not os.environ.get("AZURE_OPENAI_API_KEY"):
                st.info("Please add your Azure OpenAI Service API key to continue.")
            elif not os.environ.get("AZURE_OPENAI_ENDPOINT"):
                st.info("Please add your Azure OpenAI Service API endpoint to continue.")
            elif not os.environ.get("AZURE_DEPLOYMENT"):
                st.info("Please add the name of your Azure OpenAI Service Deployment to continue.")
            elif not industry:
                st.info("Please select your company's industry to continue.")
            elif not company_size:
                st.info("Please select your company's size to continue.")
            elif _requires_selection():
                pass
            else:
                response = generate_scenario_azure_wrapper(messages)
                st.markdown("---")
                if response is not None:
                    _handle_response_text(response.choices[0].message.content)
                else:
                    _show_previous_scenario()

    elif model_provider == "Google AI API":
        if st.button('Generate Scenario', key='generate_ai_insider_google'):
            if not os.environ.get("GOOGLE_API_KEY"):
                st.info("Please add your Google AI API key to continue.")
            elif not os.environ.get("GOOGLE_MODEL"):
                st.info("Please select a model to continue.")
            elif not industry:
                st.info("Please select your company's industry to continue.")
            elif not company_size:
                st.info("Please select your company's size to continue.")
            elif _requires_selection():
                pass
            else:
                google_api_key = st.session_state.get('google_api_key')
                model_name = os.getenv('GOOGLE_MODEL')
                response = generate_scenario_google_wrapper(google_api_key, model_name, messages)
                st.markdown("---")
                if response is not None:
                    if isinstance(response.content, list):
                        text_blocks = [block.get('text', '') for block in response.content if isinstance(block, dict) and block.get('type') == 'text']
                        scenario_text = '\n'.join(text_blocks)
                    else:
                        scenario_text = response.content
                    _handle_response_text(scenario_text)
                else:
                    _show_previous_scenario()

    elif model_provider == "Mistral API":
        if st.button('Generate Scenario', key='generate_ai_insider_mistral'):
            if not os.environ.get("MISTRAL_API_KEY"):
                st.info("Please add your Mistral API key to continue.")
            elif not os.environ.get("MISTRAL_MODEL"):
                st.info("Please select a model to continue.")
            elif not industry:
                st.info("Please select your company's industry to continue.")
            elif not company_size:
                st.info("Please select your company's size to continue.")
            elif _requires_selection():
                pass
            else:
                mistral_api_key = st.session_state.get('mistral_api_key')
                model_name = os.getenv('MISTRAL_MODEL')
                response = generate_scenario_mistral_wrapper(mistral_api_key, model_name, messages)
                st.markdown("---")
                if response is not None:
                    _handle_response_text(response.content)
                else:
                    _show_previous_scenario()

    elif model_provider == "Ollama":
        if st.button('Generate Scenario', key='generate_ai_insider_ollama'):
            if not os.environ.get("OLLAMA_MODEL"):
                st.info("Please select a model to continue.")
            elif not industry:
                st.info("Please select your company's industry to continue.")
            elif not company_size:
                st.info("Please select your company's size to continue.")
            elif _requires_selection():
                pass
            else:
                model = os.getenv('OLLAMA_MODEL')
                response = generate_scenario_ollama_wrapper(model, messages)
                st.markdown("---")
                if response is not None:
                    _handle_response_text(response)
                else:
                    _show_previous_scenario()

    elif model_provider == "Anthropic API":
        if st.button('Generate Scenario', key='generate_ai_insider_anthropic'):
            if not os.environ.get("ANTHROPIC_API_KEY"):
                st.info("Please add your Anthropic API key to continue.")
            elif not os.environ.get("ANTHROPIC_MODEL"):
                st.info("Please select a model to continue.")
            elif not industry:
                st.info("Please select your company's industry to continue.")
            elif not company_size:
                st.info("Please select your company's size to continue.")
            elif _requires_selection():
                pass
            else:
                anthropic_api_key = st.session_state.get('anthropic_api_key')
                model_name = os.getenv('ANTHROPIC_MODEL')
                response = generate_scenario_anthropic_wrapper(anthropic_api_key, model_name, messages)
                st.markdown("---")
                if response is not None:
                    _handle_response_text(response.content)
                else:
                    _show_previous_scenario()

    elif model_provider == "Groq API":
        if st.button('Generate Scenario', key='generate_ai_insider_groq'):
            if not os.environ.get("GROQ_API_KEY"):
                st.info("Please add your Groq API key to continue.")
            elif not os.environ.get("GROQ_MODEL"):
                st.info("Please select a model to continue.")
            elif not industry:
                st.info("Please select your company's industry to continue.")
            elif not company_size:
                st.info("Please select your company's size to continue.")
            elif _requires_selection():
                pass
            else:
                groq_api_key = st.session_state.get('GROQ_API_KEY')
                model_name = os.getenv('GROQ_MODEL')
                response = generate_scenario_groq_wrapper(groq_api_key, model_name, messages)
                st.markdown("---")
                if response is not None:
                    content = response.choices[0].message.content
                    if re.search(r'<think>(.*?)</think>', content, re.DOTALL):
                        thinking_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
                        thinking_content = thinking_match.group(1).strip()
                        scenario_text = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                        with st.expander("View Model's Reasoning"):
                            st.markdown(thinking_content)
                    else:
                        scenario_text = content
                    scenario_text = re.sub(r'^```\w*\n|```$', '', scenario_text, flags=re.MULTILINE).strip()
                    _handle_response_text(scenario_text)
                else:
                    _show_previous_scenario()

    elif model_provider == "Custom":
        if st.button('Generate Scenario', key='generate_ai_insider_custom'):
            if not st.session_state.get('custom_base_url'):
                st.info("Please set the Custom Base URL in the sidebar.")
            elif not st.session_state.get('custom_model_name'):
                st.info("Please set the Custom Model Name in the sidebar.")
            elif not industry:
                st.info("Please select your company's industry to continue.")
            elif not company_size:
                st.info("Please select your company's size to continue.")
            elif _requires_selection():
                pass
            else:
                response = generate_scenario_custom_wrapper(messages)
                st.markdown("---")
                if response is not None:
                    try:
                        if hasattr(response, 'choices') and response.choices and response.choices[0].message and response.choices[0].message.content:
                            _handle_response_text(response.choices[0].message.content)
                        else:
                            st.error("Error processing response: unexpected response structure.")
                            st.json(response.model_dump_json(indent=2))
                    except Exception as processing_error:
                        st.error(f"An error occurred while processing the scenario response: {processing_error}")
                else:
                    st.warning("Scenario generation failed. Check the error message above.")
                    _show_previous_scenario()

    else:  # OpenAI API (default)
        if st.button('Generate Scenario', key='generate_ai_insider_openai'):
            openai_api_key = st.session_state.get('openai_api_key')
            model_name = st.session_state.get('model_name')
            if not openai_api_key:
                st.info("Please add your OpenAI API key to continue.")
            elif not model_name:
                st.info("Please select a model to continue.")
            elif not industry:
                st.info("Please select your company's industry to continue.")
            elif not company_size:
                st.info("Please select your company's size to continue.")
            elif _requires_selection():
                pass
            else:
                response = generate_scenario_wrapper(openai_api_key, model_name, messages)
                st.markdown("---")
                if response is not None:
                    if isinstance(response.content, list):
                        text_blocks = [block.get('text', '') for block in response.content if block.get('type') == 'text']
                        scenario_text = '\n'.join(text_blocks)
                    else:
                        scenario_text = response.content
                    _handle_response_text(scenario_text)
                else:
                    _show_previous_scenario()

    # Display an info message if no LangChain API key is set
    if 'LANGCHAIN_API_KEY' not in st.secrets:
        st.info("ℹ️ No LangChain API key has been set. This run will not be logged to LangSmith.")

    # Feedback buttons
    feedback_placeholder = st.empty()
    st.markdown("---")
    if st.session_state.get('ai_insider_scenario_generated', False) and client is not None:
        st.markdown("Rate the scenario to help improve this tool.")
        col1, col2, col3 = st.columns([0.5, 0.5, 5])
        with col1:
            if st.button("👍", key="thumbs_up_ai_insider"):
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
            if st.button("👎", key="thumbs_down_ai_insider"):
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


# Add a back button
link_to_homepage = "/"
st.markdown(
    f'<a href="{link_to_homepage}" style="display: inline-block; padding: 5px 20px; color: white; text-align: center; text-decoration: none; font-size: 16px; border-radius: 4px;">⬅️ Back</a>',
    unsafe_allow_html=True,
)
