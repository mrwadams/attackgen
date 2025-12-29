import os
import pandas as pd
import streamlit as st
import re

from langchain_anthropic import ChatAnthropic
from langchain_community.llms import Ollama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_openai import ChatOpenAI, AzureOpenAI
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langsmith import Client, RunTree, traceable
from mitreattack.stix20 import MitreAttackData
from openai import OpenAI

from atlas_parser import ATLASData, load_atlas_case_studies, get_techniques_from_case_study_procedure


# ------------------ Streamlit UI Configuration ------------------ #

# Add environment variables for LangSmith
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "AttackGen"

# Initialise the LangSmith client conditionally based on the presence of an API key
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

# Add environment variables from session state for OpenAI API
if "OPENAI_API_KEY" in st.session_state:
    os.environ["OPENAI_API_KEY"] = st.session_state["OPENAI_API_KEY"]
if "openai_model" in st.session_state:
    os.environ["OPENAI_MODEL"] = st.session_state["openai_model"]
if "openai_base_url" in st.session_state:
    os.environ["OPENAI_BASE_URL"] = st.session_state["openai_base_url"]

# Add environment variables from session state for Anthropic API
if "ANTHROPIC_API_KEY" in st.session_state:
    os.environ["ANTHROPIC_API_KEY"] = st.session_state["ANTHROPIC_API_KEY"]
if "anthropic_model" in st.session_state:
    os.environ["ANTHROPIC_MODEL"] = st.session_state["anthropic_model"]

# Get the model provider and other required session state variables
model_provider = st.session_state["chosen_model_provider"]
industry = st.session_state["industry"]
company_size = st.session_state["company_size"]

# Set the default value for the scenario_generated session state variable
if "scenario_generated" not in st.session_state:
    st.session_state["scenario_generated"] = False

st.set_page_config(
    page_title="Generate Scenario",
    page_icon="🛡️",
)

# ------------------ Helper Functions ------------------ #

# Load and cache the MITRE ATT&CK and ATLAS data
@st.cache_resource
def load_attack_data():
    enterprise_data = MitreAttackData("./data/enterprise-attack.json")
    ics_data = MitreAttackData("./data/ics-attack.json")
    atlas_data = ATLASData("./data/stix-atlas.json")
    return {"enterprise": enterprise_data, "ics": ics_data, "atlas": atlas_data}

attack_data = load_attack_data()

@st.cache_resource
def load_groups(matrix):
    if matrix == "Enterprise":
        groups = pd.read_json("./data/groups.json")
    elif matrix == "ICS":
        groups = pd.read_json("./data/groups_ics.json")
    else:  # ATLAS
        groups = pd.read_json("./data/atlas-case-studies.json")
    return groups

def generate_scenario_wrapper(openai_api_key, model_name, messages):
    if client is not None:  # If LangChain client has been initialized
        @traceable(run_type="llm", name="Threat Group Scenario", tags=["openai", "threat_group_scenario"], client=client)
        def generate_scenario(openai_api_key, model_name, messages, *, run_tree: RunTree):
            model_name = st.session_state["model_name"]
            try:
                with st.status('Generating scenario...', expanded=True):
                    st.write("Initialising AI model.")
                    
                    # All models use the unified Responses API
                    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name=model_name, streaming=False, output_version="responses/v1")
                    
                    st.write("Model initialised. Generating scenario, please wait.")
                    response = llm.invoke(messages)
                    st.write("Scenario generated successfully.")
                    st.session_state['run_id'] = str(run_tree.id)  # Store the run ID in the session state
                    return response
            except Exception as e:
                st.error("An error occurred while generating the scenario: " + str(e))
                st.session_state['run_id'] = str(run_tree.id)  # Ensure run_id is updated even on failure
                return None
    else:  # If LangChain client has not been initialized
        def generate_scenario(openai_api_key, model_name, messages):
            model_name = st.session_state["model_name"]
            try:
                with st.status('Generating scenario...', expanded=True):
                    st.write("Initialising AI model.")
                    
                    # All models use the unified Responses API
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
    if client is not None:  # LangSmith client has been initialised
        @traceable(run_type="llm", name="Threat Group Scenario (Azure OpenAI)", tags=["azure", "threat_group_scenario"], client=client if client is not None else None)
        def generate_scenario_azure(messages, *, run_tree: RunTree):
            try:
                azure_api_key = os.getenv('AZURE_OPENAI_API_KEY')
                azure_api_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
                azure_deployment_name = os.getenv('AZURE_DEPLOYMENT')
                azure_api_version = os.getenv('OPENAI_API_VERSION')
                with st.status('Generating scenario...', expanded=True):
                    st.write("Initialising AI model.")
                    llm = AzureOpenAI(api_key=azure_api_key,
                                      azure_endpoint=azure_api_endpoint,
                                      api_version=azure_api_version)
                    st.write("Model initialised. Generating scenario, please wait.")
                    
                    # Convert message objects to the expected format
                    formatted_messages = []
                    for message in messages:
                        if hasattr(message, 'role') and hasattr(message, 'content'):
                            role = message.role
                            if role == 'human':
                                role = 'user'  # Replace 'human' with 'user'
                            formatted_messages.append({"role": role, "content": message.content})
                        elif hasattr(message, 'type') and hasattr(message, 'content'):
                            role = message.type
                            if role == 'human':
                                role = 'user'  # Replace 'human' with 'user'
                            formatted_messages.append({"role": role, "content": message.content})
                        else:
                            raise ValueError(f"Unsupported message format: {message}")
                    
                    response = llm.chat.completions.create(
                        model=azure_deployment_name,
                        messages=formatted_messages
                    )
                    st.write("Scenario generated successfully.")
                    st.session_state['run_id'] = str(run_tree.id)  # Store the run ID in the session state
                    return response
            except Exception as e:
                st.error(f"An error occurred while generating the scenario: {str(e)}")
                st.session_state['run_id'] = str(run_tree.id)  # Ensure run_id is updated even on failure
                return None
    else:  # LangSmith client has not been initialised
        def generate_scenario_azure(messages):
            try:
                azure_api_key = os.getenv('AZURE_OPENAI_API_KEY')
                azure_api_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
                azure_deployment_name = os.getenv('AZURE_DEPLOYMENT')
                azure_api_version = os.getenv('OPENAI_API_VERSION')
                with st.status('Generating scenario...', expanded=True):
                    st.write("Initialising AI model.")
                    llm = AzureOpenAI(api_key=azure_api_key,
                                      azure_endpoint=azure_api_endpoint,
                                      api_version=azure_api_version)
                    st.write("Model initialised. Generating scenario, please wait.")
                    
                    # Convert message objects to the expected format
                    formatted_messages = []
                    for message in messages:
                        if hasattr(message, 'role') and hasattr(message, 'content'):
                            role = message.role
                            if role == 'human':
                                role = 'user'  # Replace 'human' with 'user'
                            formatted_messages.append({"role": role, "content": message.content})
                        elif hasattr(message, 'type') and hasattr(message, 'content'):
                            role = message.type
                            if role == 'human':
                                role = 'user'  # Replace 'human' with 'user'
                            formatted_messages.append({"role": role, "content": message.content})
                        else:
                            raise ValueError(f"Unsupported message format: {message}")
                    
                    response = llm.chat.completions.create(
                        model=azure_deployment_name,
                        messages=formatted_messages
                    )
                    st.write("Scenario generated successfully.")
                    return response
            except Exception as e:
                st.error(f"An error occurred while generating the scenario: {str(e)}")
                return None
    return generate_scenario_azure(messages)

def generate_scenario_google_wrapper(google_api_key, model, messages):
    if client is not None: # If LangSmith client has been initialised
        @traceable(run_type="llm", name="Threat Group Scenario (Google AI API)", tags=["google", "threat_group_scenario"], client=client)
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
                    st.session_state['run_id'] = str(run_tree.id) # Store the run ID in the session state
                    return response
            except Exception as e:
                st.error(f"An error occurred while generating the scenario: {str(e)}")
                st.session_state['run_id'] = str(run_tree.id) # Ensure run_id is updated even on failure
                return None
    else: # If LangSmith client has not been initialised
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
    if client is not None: # If LangSmith client has been initialised
        @traceable(run_type="llm", name="Threat Group Scenario (Mistral API)", tags=["mistral", "threat_group_scenario"], client=client)
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
                    st.session_state['run_id'] = str(run_tree.id) # Store the run ID in the session state
                    return response
            except Exception as e:
                st.error(f"An error occurred while generating the scenario: {str(e)}")
                st.session_state['run_id'] = str(run_tree.id) # Ensure run_id is updated even on failure
                return None
    else: # If LangSmith client has not been initialised
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

def generate_scenario_ollama_wrapper(model):
    if client is not None: # If LangSmith client has been initialised
        @traceable(run_type="llm", name="Threat Group Scenario (Ollama)", tags=["ollama", "threat_group_scenario"], client=client)
        def generate_scenario_ollama(model, *, run_tree: RunTree):
            try:
                model = os.getenv('OLLAMA_MODEL')
                with st.status('Generating scenario...', expanded=True):
                    st.write("Initialising AI model.")
                    llm = Ollama(model=model)
                    st.write("Model initialised. Generating scenario, please wait.")
                    response = llm.invoke(messages, model=model)
                    st.write("Scenario generated successfully.")
                    st.session_state['run_id'] = str(run_tree.id)  # Store the run ID in the session state
                return response
            except Exception as e:
                st.error(f"An error occurred while generating the scenario: {str(e)}")
                st.session_state['run_id'] = str(run_tree.id)  # Ensure run_id is updated even on failure
                return None
    else: # If LangSmith client has not been initialised
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
    if client is not None:  # If LangSmith client has been initialised
        @traceable(run_type="llm", name="Threat Group Scenario (Anthropic)", tags=["anthropic", "threat_group_scenario"], client=client)
        def generate_scenario_anthropic(anthropic_api_key, model_name, messages, *, run_tree: RunTree):
            try:
                anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
                model = os.getenv('ANTHROPIC_MODEL')
                with st.status('Generating scenario...', expanded=True):
                    st.write("Initialising AI model.")
                    # Set max_tokens based on the model
                    max_tokens = 8192  # Default for Haiku
                    if "opus-4" in model:
                        max_tokens = 32000
                    elif "sonnet-4" in model or "3-7-sonnet" in model:
                        max_tokens = 64000
                    
                    llm = ChatAnthropic(anthropic_api_key=anthropic_api_key, model_name=model, temperature=0.7, max_tokens=max_tokens)
                    st.write("Model initialised. Generating scenario, please wait.")
                    response = llm.invoke(messages)
                    st.write("Scenario generated successfully.")
                    st.session_state['run_id'] = str(run_tree.id)  # Store the run ID in the session state
                    return response
            except Exception as e:
                st.error(f"An error occurred while generating the scenario: {str(e)}")
                st.session_state['run_id'] = str(run_tree.id)  # Ensure run_id is updated even on failure
                return None
    else:  # If LangSmith client has not been initialised
        def generate_scenario_anthropic(anthropic_api_key, model_name, messages):
            try:
                anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
                model = os.getenv('ANTHROPIC_MODEL')
                with st.status('Generating scenario...', expanded=True):
                    st.write("Initialising AI model.")
                    # Set max_tokens based on the model
                    max_tokens = 8192  # Default for Haiku
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

def generate_scenario_groq_wrapper(groq_api_key, model_name, messages):
    if client is not None: # If LangSmith client has been initialised
        @traceable(run_type="llm", name="Threat Group Scenario (Groq API)", tags=["groq", "threat_group_scenario"], client=client)
        def generate_scenario_groq(groq_api_key, model_name, messages, *, run_tree: RunTree):
            try:
                groq_api_key = os.getenv('GROQ_API_KEY')
                model = os.getenv('GROQ_MODEL')
                with st.status('Generating scenario...', expanded=True):
                    st.write("Initialising AI model.")
                    llm = OpenAI(
                        api_key=groq_api_key,
                        base_url="https://api.groq.com/openai/v1",
                    )
                    st.write("Model initialised. Generating scenario, please wait.")
                    
                    # Convert message objects to the expected format
                    formatted_messages = []
                    for message in messages:
                        if hasattr(message, 'role') and hasattr(message, 'content'):
                            role = message.role
                            if role == 'human':
                                role = 'user'  # Replace 'human' with 'user'
                            formatted_messages.append({"role": role, "content": message.content})
                        elif hasattr(message, 'type') and hasattr(message, 'content'):
                            role = message.type
                            if role == 'human':
                                role = 'user'  # Replace 'human' with 'user'
                            formatted_messages.append({"role": role, "content": message.content})
                        else:
                            raise ValueError(f"Unsupported message format: {message}")
                    
                    response = llm.chat.completions.create(
                        model=model,
                        messages=formatted_messages
                    )
                    st.write("Scenario generated successfully.")
                    st.session_state['run_id'] = str(run_tree.id)  # Store the run ID in the session state
                    return response
            except Exception as e:
                st.error(f"An error occurred while generating the scenario: {str(e)}")
                st.session_state['run_id'] = str(run_tree.id)  # Ensure run_id is updated even on failure
                return None
    else: # If LangSmith client has not been initialised
        def generate_scenario_groq(groq_api_key, model_name, messages):
            try:
                groq_api_key = os.getenv('GROQ_API_KEY')
                model = os.getenv('GROQ_MODEL')
                with st.status('Generating scenario...', expanded=True):
                    st.write("Initialising AI model.")
                    llm = OpenAI(
                        api_key=groq_api_key,
                        base_url="https://api.groq.com/openai/v1",
                    )
                    st.write("Model initialised. Generating scenario, please wait.")
                    
                    # Convert message objects to the expected format
                    formatted_messages = []
                    for message in messages:
                        if hasattr(message, 'role') and hasattr(message, 'content'):
                            role = message.role
                            if role == 'human':
                                role = 'user'  # Replace 'human' with 'user'
                            formatted_messages.append({"role": role, "content": message.content})
                        elif hasattr(message, 'type') and hasattr(message, 'content'):
                            role = message.type
                            if role == 'human':
                                role = 'user'  # Replace 'human' with 'user'
                            formatted_messages.append({"role": role, "content": message.content})
                        else:
                            raise ValueError(f"Unsupported message format: {message}")
                    
                    response = llm.chat.completions.create(
                        model=model,
                        messages=formatted_messages
                    )
                    st.write("Scenario generated successfully.")
                    return response
            except Exception as e:
                st.error(f"An error occurred while generating the scenario: {str(e)}")
                return None
    
    return generate_scenario_groq(groq_api_key, model_name, messages)

# Modify the OpenAI client to use a custom base URL if provided
def generate_scenario_openai_wrapper(messages):
    base_url = st.session_state.get('custom_base_url')
    openai_api_key = st.session_state.get('custom_api_key')
    model_name = st.session_state.get('model_name') or os.environ.get('OPENAI_MODEL')

    if not base_url:
        st.error("Custom base URL must be set for the custom model provider.")
        return None
    if not model_name:
        st.error("Custom model name must be set for the custom model provider.")
        return None
    # API key might be optional for some local endpoints, so no explicit check here

    # Convert LangChain message objects to the expected dictionary format
    formatted_messages = []
    for message in messages:
        if hasattr(message, 'role') and hasattr(message, 'content'):
            role = message.role
            if role == 'human':
                role = 'user'  # Replace 'human' with 'user'
            formatted_messages.append({"role": role, "content": message.content})
        elif hasattr(message, 'type') and hasattr(message, 'content'):
            role = message.type
            if role == 'human':
                role = 'user'  # Replace 'human' with 'user'
            formatted_messages.append({"role": role, "content": message.content})
        else:
            # Attempt to handle SystemMessagePromptTemplate and HumanMessagePromptTemplate if needed
            if hasattr(message, 'prompt') and hasattr(message.prompt, 'template'):
                 # This case might need more specific handling depending on how templates are used
                 # For now, we'll skip templates assuming 'messages' are already formatted instances
                 st.warning(f"Skipping message template of type {type(message)}")
                 continue
            else:
                 st.error(f"Unsupported message format: {type(message)} - {message}")
                 return None # Stop processing if message format is wrong

    if not formatted_messages:
        st.error("No valid messages found to send to the model.")
        return None

    if client is not None:  # If LangChain client has been initialized
        @traceable(run_type="llm", name="Threat Group Scenario (Custom)", tags=["custom", "threat_group_scenario"], client=client)
        def generate_scenario_custom(formatted_messages_arg, *, run_tree: RunTree):
            try:
                # Re-fetch variables inside traceable function scope if necessary, though session_state should be accessible
                custom_api_key = st.session_state.get('custom_api_key')
                custom_model = st.session_state.get('model_name') or os.environ.get('OPENAI_MODEL')
                custom_base_url = st.session_state.get('custom_base_url')

                with st.status('Generating scenario with custom model...', expanded=True):
                    st.write("Initialising custom AI model.")

                    # Conditionally prepare client arguments
                    client_args = {
                        "base_url": custom_base_url,
                    }
                    if custom_api_key: # Only add api_key if it's not empty
                        client_args["api_key"] = custom_api_key

                    llm = OpenAI(**client_args)

                    response = llm.chat.completions.create(
                        model=custom_model,
                        messages=formatted_messages_arg, # Use the formatted messages
                        temperature=0.7,
                        max_tokens=-1,
                        stream=False
                    )
                    st.write("Scenario generated successfully.")
                    st.session_state['run_id'] = str(run_tree.id)  # Store the run ID in the session state
                    return response
            except Exception as e:
                st.error(f"An error occurred while generating the scenario: {str(e)}")
                st.session_state['run_id'] = str(run_tree.id)  # Ensure run_id is updated even on failure
                return None
        return generate_scenario_custom(formatted_messages)

    else:  # If LangChain client has not been initialized
        def generate_scenario_custom_no_trace(formatted_messages_arg):
            try:
                # Fetch variables directly from session state
                current_api_key = st.session_state.get('custom_api_key')
                current_model = st.session_state.get('model_name') or os.environ.get('OPENAI_MODEL')
                current_base_url = st.session_state.get('custom_base_url')

                with st.status('Generating scenario with custom model...', expanded=True):
                    st.write("Initialising custom AI model.")

                    # Conditionally prepare client arguments
                    client_args = {
                        "base_url": current_base_url,
                    }
                    if current_api_key: # Only add api_key if it's not empty
                        client_args["api_key"] = current_api_key

                    llm = OpenAI(**client_args)

                    response = llm.chat.completions.create(
                        model=current_model,
                        messages=formatted_messages_arg,
                        temperature=0.7,
                        max_tokens=-1,
                        stream=False
                    )
                    st.write("Scenario generated successfully.")
                    return response
            except Exception as e:
                st.error(f"An error occurred while generating the scenario: {str(e)}")
                return None
        return generate_scenario_custom_no_trace(formatted_messages)

# ------------------ Streamlit UI ------------------ #

st.markdown("# <span style='color: #1DB954;'>Generate Threat Group Scenario🛡️</span>", unsafe_allow_html=True)

# Load groups based on the current matrix
matrix = st.session_state.get("matrix", "Enterprise")
groups = load_groups(matrix)

# Conditional UI text based on matrix selection
if matrix == "ATLAS":
    st.markdown("""
            ### Select a Case Study

            Use the drop-down selector below to select a case study from the MITRE ATLAS framework.

            You can then optionally view all of the ATLAS techniques associated with the case study and/or the case study's page on the MITRE ATLAS site.
            """)
    entity_label = "case study"
    select_placeholder = "Select Case Study"
else:
    st.markdown(f"""
            ### Select a Threat Actor Group

            Use the drop-down selector below to select a threat actor group from the MITRE ATT&CK framework.

            You can then optionally view all of the {matrix} ATT&CK techniques associated with the group and/or the group's page on the MITRE ATT&CK site.
            """)
    entity_label = "threat actor group"
    select_placeholder = "Select Group"

# Get the list of group names
group_names = sorted(groups['group'].unique())

# Set a default index that works for all matrices
default_index = 0 if len(group_names) > 0 else None

selected_group_alias = st.selectbox(
    f"Select a {entity_label} for the scenario",
    group_names,
    index=default_index,
    placeholder=select_placeholder,
    label_visibility="hidden"
)

# Define phase name order based on matrix
if matrix == "ATLAS":
    phase_name_order = ['Reconnaissance', 'Resource Development', 'Initial Access', 'AI Model Access',
                        'Execution', 'Persistence', 'Privilege Escalation', 'Defense Evasion',
                        'Credential Access', 'Discovery', 'Lateral Movement', 'Collection',
                        'AI Attack Staging', 'Command and Control', 'Exfiltration', 'Impact']
else:
    phase_name_order = ['Reconnaissance', 'Resource Development', 'Initial Access', 'Execution', 'Persistence',
                        'Privilege Escalation', 'Defense Evasion', 'Credential Access', 'Discovery', 'Lateral Movement',
                        'Collection', 'Command and Control', 'Exfiltration', 'Impact']

phase_name_category = pd.CategoricalDtype(categories=phase_name_order, ordered=True)



try:
    # Define techniques_df as an empty dataframe
    techniques_df = pd.DataFrame()

    # define selected_techniques_df as an empty dataframe before the if condition
    selected_techniques_df = pd.DataFrame()

    if selected_group_alias != select_placeholder:
        # Get the group by the selected alias
        matrix = st.session_state.get("matrix", "Enterprise")
        group_url = groups[groups['group'] == selected_group_alias]['url'].values[0]

        # Display the URL as a clickable link
        if matrix == "ATLAS":
            st.markdown(f"[View case study on atlas.mitre.org]({group_url})")
        else:
            st.markdown(f"[View {selected_group_alias}'s page on attack.mitre.org]({group_url})")

        if matrix == "ATLAS":
            # ATLAS: Extract techniques from case study procedure
            case_study_row = groups[groups['group'] == selected_group_alias].iloc[0]
            procedure = case_study_row.get('procedure', [])

            if not procedure:
                st.warning(f"There are no ATLAS techniques associated with the case study: {selected_group_alias}")
                st.stop()

            # Get techniques from the procedure
            techniques_list = get_techniques_from_case_study_procedure(procedure, attack_data["atlas"])

            if not techniques_list:
                st.warning(f"Could not extract techniques from the case study: {selected_group_alias}")
                st.stop()

            # Create DataFrame from techniques list
            techniques_df = pd.DataFrame(techniques_list)
            techniques_df_llm = techniques_df.copy()

            # Convert the 'Phase Name' column to the ordered category
            techniques_df['Phase Name'] = techniques_df['Phase Name'].astype(phase_name_category)
            techniques_df_llm['Phase Name'] = techniques_df_llm['Phase Name'].astype(phase_name_category)

            # Sort by Phase Name
            techniques_df = techniques_df.sort_values('Phase Name')
            techniques_df_llm = techniques_df_llm.sort_values('Phase Name')

            # For ATLAS, we use all techniques from the procedure (no random selection)
            # as they represent the documented attack chain
            selected_techniques_df = techniques_df_llm.copy()

            # Select display columns
            techniques_df = techniques_df[['Technique Name', 'ATT&CK ID', 'Phase Name']]

        else:
            # ATT&CK: Use standard group lookup
            group = attack_data[matrix.lower()].get_groups_by_alias(selected_group_alias)

            # Check if the group was found
            if group:
                # Get the STIX ID of the group
                group_stix_id = group[0].id

                # Get all techniques used by the group
                techniques = attack_data[matrix.lower()].get_techniques_used_by_group(group_stix_id)

                # Check if there are any techniques for the group
                if not techniques:
                    st.warning(f"There are no {matrix} ATT&CK techniques associated with the threat group: {selected_group_alias}")
                    st.stop()
                else:
                    # Update techniques_df with the techniques
                    techniques_df = pd.DataFrame(techniques)

                # Create a copy of the DataFrame for generating the LLM prompt
                techniques_df_llm = techniques_df.copy()

                # Add a 'Technique Name' column to both DataFrames
                techniques_df['Technique Name'] = techniques_df_llm['Technique Name'] = techniques_df['object'].apply(lambda x: x['name'])

                # Add a 'ATT&CK ID' column to both DataFrames
                techniques_df['ATT&CK ID'] = techniques_df_llm['ATT&CK ID'] = techniques_df['object'].apply(lambda x: attack_data[matrix.lower()].get_attack_id(x['id']))

                # Add a 'Phase Name' column to both DataFrames
                techniques_df['Phase Name'] = techniques_df_llm['Phase Name'] = techniques_df['object'].apply(lambda x: x['kill_chain_phases'][0]['phase_name'])

                # Drop duplicate techniques based on Phase Name, Technique Name and ATT&CK ID
                techniques_df = techniques_df.drop_duplicates(['Phase Name', 'Technique Name', 'ATT&CK ID'])

                # Convert the 'Phase Name' to title case and replace hyphens with spaces
                techniques_df['Phase Name'] = techniques_df['Phase Name'].str.replace('-', ' ').str.title()
                techniques_df_llm['Phase Name'] = techniques_df_llm['Phase Name'].str.replace('-', ' ').str.title()

                # Replace 'Command And Control' with 'Command and Control'
                techniques_df['Phase Name'] = techniques_df['Phase Name'].replace('Command And Control', 'Command and Control')
                techniques_df_llm['Phase Name'] = techniques_df_llm['Phase Name'].replace('Command And Control', 'Command and Control')

                # Convert the 'Phase Name' column to the ordered category in both DataFrames
                techniques_df['Phase Name'] = techniques_df['Phase Name'].astype(phase_name_category)
                techniques_df_llm['Phase Name'] = techniques_df_llm['Phase Name'].astype(phase_name_category)

                # Sort the DataFrame by the 'Phase Name' column
                techniques_df = techniques_df.sort_values('Phase Name')
                techniques_df_llm = techniques_df_llm.sort_values('Phase Name')

                # Group by 'Phase Name' and randomly select one technique from each group
                # Filter the groups to include only those that have at least one row in the LLM DataFrame
                selected_techniques_df = (
                    techniques_df_llm.groupby('Phase Name', observed=False)  # Use default group_keys behavior
                    .apply(
                        lambda x: x.sample(n=1) if not x.empty else pd.DataFrame(columns=x.columns),
                        include_groups=False  # This ensures x in lambda doesn't have grouping columns
                    )
                    .reset_index()  # This will bring 'Phase Name' from the index into a column
                )

                # Sort the DataFrame by the 'Phase Name' column
                techniques_df = techniques_df.sort_values('Phase Name')

                # Select only the 'Technique Name', 'ATT&CK ID', and 'Phase Name' columns
                techniques_df = techniques_df[['Technique Name', 'ATT&CK ID', 'Phase Name']]

        if not techniques_df.empty:
            # Create an expander for the techniques
            expander_title = "Associated ATLAS Techniques" if matrix == "ATLAS" else "Associated ATT&CK Techniques"
            with st.expander(expander_title):
                # Use the st.table function to display the DataFrame
                st.dataframe(data=techniques_df, height=200, width='stretch', hide_index=True)

        # Create an empty list to hold the kill chain information
        kill_chain = []

        # Iterate over the rows in the DataFrame
        for index, row in selected_techniques_df.iterrows():
            # Extract the phase and technique information
            phase_name = row['Phase Name']
            technique_name = row['Technique Name']
            attack_id = row['ATT&CK ID']

            # Add a string with this information to the kill chain list
            kill_chain.append(f"{phase_name}: {technique_name} ({attack_id})")

        # Convert the kill chain list to a string with newline characters between each item
        kill_chain_string = "\n".join(kill_chain)

        # Create System Message Template
        system_template = "You are a cybersecurity expert. Your task is to produce a comprehensive incident response testing scenario based on the information provided. Format your response using proper Markdown syntax with headers, bullet points, and formatting for readability."
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

        # Create Human Message Template - different for ATLAS vs ATT&CK
        if matrix == "ATLAS":
            human_template = ("""
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
""")
        else:
            human_template = ("""
**Background information:**
The company operates in the '{industry}' industry and is of size '{company_size}'.

**Threat actor information:**
Threat actor group '{selected_group_alias}' is planning to target the company using the following kill chain from the MITRE ATT&CK {matrix} Matrix:
{kill_chain_string}

**Your task:**
Create an incident response testing scenario based on the information provided. The goal of the scenario is to test the company's incident response capabilities against the identified threat actor group, focusing on the {matrix} environment.

Your response should be well structured and formatted using Markdown. Write in British English.
""")
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

        # Construct the ChatPromptTemplate
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

        # Format the prompt
        messages = chat_prompt.format_prompt(selected_group_alias=selected_group_alias, 
                                            kill_chain_string=kill_chain_string, 
                                            industry=industry, 
                                            company_size=company_size, 
                                            matrix=matrix).to_messages()

# Error handling for group selection
except Exception as e:
    st.error("An error occurred: " + str(e))

st.markdown("")

# Display the scenario generation section
if matrix == "ATLAS":
    st.markdown("""
            ### Generate a Scenario

            Click the button below to generate a scenario based on the selected case study. The documented attack procedure from the case study will be used to generate the scenario.

            It normally takes between 30-50 seconds to generate a scenario, although for local models this is highly dependent on your hardware and the selected model. ⏱️
            """)
else:
    st.markdown("""
            ### Generate a Scenario

            Click the button below to generate a scenario based on the selected threat actor group. A selection of the group's known techniques will be chosen at random and used to generate the scenario.

            It normally takes between 30-50 seconds to generate a scenario, although for local models this is highly dependent on your hardware and the selected model. ⏱️
            """)

try:
    if model_provider == "Azure OpenAI Service":
        if st.button('Generate Scenario', key='generate_scenario_azure'):
            if not os.environ["AZURE_OPENAI_API_KEY"]:
                st.info("Please add your Azure OpenAI Service API key to continue.")
            if not os.environ["AZURE_OPENAI_ENDPOINT"]:
                st.info("Please add your Azure OpenAI Service API endpoint to continue.")
            if not os.environ["AZURE_DEPLOYMENT"]:
                st.info("Please add the name of your Azure OpenAI Service Deployment to continue.")
            elif not industry:
                st.info("Please select your company's industry to continue.")
            elif not company_size:
                st.info("Please select your company's size to continue.")
            elif techniques_df.empty:
                st.info(f"Please select a {entity_label} with associated techniques.")
            else:
                response = generate_scenario_azure_wrapper(messages)
                st.markdown("---")
                if response is not None:
                    st.session_state['scenario_generated'] = True
                    scenario_text = response.choices[0].message.content
                    st.session_state['scenario_text'] = scenario_text  # Store the generated scenario in the session state
                    st.markdown(scenario_text)
                    st.download_button(label="Download Scenario", data=st.session_state['scenario_text'], file_name="threat_group_scenario.md", mime="text/markdown")

                    st.session_state['last_scenario'] = True
                    st.session_state['last_scenario_text'] = scenario_text # Store the last scenario in the session state for use by the Scenario Assistant

                else:
                    # If a scenario has been generated previously, display it
                    if 'scenario_text' in st.session_state and st.session_state['scenario_generated']:
                        st.markdown("---")
                        st.markdown(st.session_state['scenario_text'])
                        st.download_button(label="Download Scenario", data=st.session_state['scenario_text'], file_name="threat_group_scenario.md", mime="text/markdown")

    elif model_provider == "Google AI API":
        if st.button('Generate Scenario', key='generate_scenario_google'):
            if not os.environ["GOOGLE_API_KEY"]:
                st.info("Please add your Google AI API key to continue.")
            if not os.environ["GOOGLE_MODEL"]:
                st.info("Please select a model to continue.")
            elif not industry:
                st.info("Please select your company's industry to continue.")
            elif not company_size:
                st.info("Please select your company's size to continue.")
            elif techniques_df.empty:
                st.info(f"Please select a {entity_label} with associated techniques.")
            else:
                google_api_key = st.session_state.get('google_api_key')
                model_name = os.getenv('GOOGLE_MODEL')
                response = generate_scenario_google_wrapper(google_api_key, model_name, messages)
                st.markdown("---")
                if response is not None:
                    try:
                        # Extract text content from structured response if needed
                        if isinstance(response.content, list):
                            # Find text blocks in the structured response
                            text_blocks = [block.get('text', '') for block in response.content if isinstance(block, dict) and block.get('type') == 'text']
                            scenario_text = '\n'.join(text_blocks)
                        else:
                            scenario_text = response.content

                        st.session_state['scenario_generated'] = True
                        st.session_state['scenario_text'] = scenario_text
                        st.markdown(scenario_text)
                        st.download_button(label="Download Scenario", data=st.session_state['scenario_text'], file_name="threat_group_scenario.md", mime="text/markdown")

                        st.session_state['last_scenario'] = True
                        st.session_state['last_scenario_text'] = scenario_text
                    except Exception as processing_error:
                        st.error(f"An error occurred while processing the scenario response: {processing_error}")
                        st.text("Raw response content:")
                        st.json(str(response.content))
                else:
                    # If a scenario has been generated previously, display it
                    if 'scenario_text' in st.session_state and st.session_state['scenario_generated']:
                        st.markdown("---")
                        st.markdown(st.session_state['scenario_text'])
                        st.download_button(label="Download Scenario", data=st.session_state['scenario_text'], file_name="threat_group_scenario.md", mime="text/markdown")

    elif model_provider == "Mistral API":
        if st.button('Generate Scenario', key='generate_scenario_mistral'):
            if not os.environ["MISTRAL_API_KEY"]:
                st.info("Please add your Mistral API key to continue.")
            if not os.environ["MISTRAL_MODEL"]:
                st.info("Please select a model to continue.")
            elif not industry:
                st.info("Please select your company's industry to continue.")
            elif not company_size:
                st.info("Please select your company's size to continue.")
            elif techniques_df.empty:
                st.info(f"Please select a {entity_label} with associated techniques.")
            else:
                mistral_api_key = st.session_state.get('mistral_api_key')
                model_name = os.getenv('MISTRAL_MODEL')
                response = generate_scenario_mistral_wrapper(mistral_api_key, model_name, messages)
                st.markdown("---")
                if response is not None:
                    st.session_state['scenario_generated'] = True
                    scenario_text = response.content
                    st.session_state['scenario_text'] = scenario_text  # Store the generated scenario in the session state
                    st.markdown(scenario_text)
                    st.download_button(label="Download Scenario", data=st.session_state['scenario_text'], file_name="threat_group_scenario.md", mime="text/markdown")

                    st.session_state['last_scenario'] = True
                    st.session_state['last_scenario_text'] = scenario_text # Store the last scenario in the session state for use by the Scenario Assistant

                else:
                    # If a scenario has been generated previously, display it
                    if 'scenario_text' in st.session_state and st.session_state['scenario_generated']:
                        st.markdown("---")
                        st.markdown(st.session_state['scenario_text'])
                        st.download_button(label="Download Scenario", data=st.session_state['scenario_text'], file_name="threat_group_scenario.md", mime="text/markdown")
    
    elif model_provider == "Ollama":
        if st.button('Generate Scenario', key='generate_scenario_ollama'):
            if not os.environ["OLLAMA_MODEL"]:
                st.info("Please select a model to continue.")
            elif not industry:
                st.info("Please select your company's industry to continue.")
            elif not company_size:
                st.info("Please select your company's size to continue.")
            elif techniques_df.empty:
                st.info(f"Please select a {entity_label} with associated techniques.")
            else:
                model = os.getenv('OLLAMA_MODEL')
                response = generate_scenario_ollama_wrapper(model)
                st.markdown("---")
                if response is not None:
                    st.session_state['scenario_generated'] = True
                    scenario_text = response
                    st.session_state['scenario_text'] = scenario_text  # Store the generated scenario in the session state
                    st.markdown(scenario_text)
                    st.download_button(label="Download Scenario", data=st.session_state['scenario_text'], file_name="threat_group_scenario.md", mime="text/markdown")

                    st.session_state['last_scenario'] = True
                    st.session_state['last_scenario_text'] = scenario_text # Store the last scenario in the session state for use by the Scenario Assistant

                else:
                    # If a scenario has been generated previously, display it
                    if 'scenario_text' in st.session_state and st.session_state['scenario_generated']:
                        st.markdown("---")
                        st.markdown(st.session_state['scenario_text'])
                        st.download_button(label="Download Scenario", data=st.session_state['scenario_text'], file_name="threat_group_scenario.md", mime="text/markdown")

    elif model_provider == "Anthropic API":
        if st.button('Generate Scenario', key='generate_scenario_anthropic'):
            if not os.environ["ANTHROPIC_API_KEY"]:
                st.info("Please add your Anthropic API key to continue.")
            if not os.environ["ANTHROPIC_MODEL"]:
                st.info("Please select a model to continue.")
            elif not industry:
                st.info("Please select your company's industry to continue.")
            elif not company_size:
                st.info("Please select your company's size to continue.")
            elif techniques_df.empty:
                st.info(f"Please select a {entity_label} with associated techniques.")
            else:
                anthropic_api_key = st.session_state.get('anthropic_api_key')
                model_name = os.getenv('ANTHROPIC_MODEL')
                response = generate_scenario_anthropic_wrapper(anthropic_api_key, model_name, messages)
                st.markdown("---")
                if response is not None:
                    st.session_state['scenario_generated'] = True
                    scenario_text = response.content
                    st.session_state['scenario_text'] = scenario_text  # Store the generated scenario in the session state
                    st.markdown(scenario_text)
                    st.download_button(label="Download Scenario", data=st.session_state['scenario_text'], file_name="threat_group_scenario.md", mime="text/markdown")

                    st.session_state['last_scenario'] = True
                    st.session_state['last_scenario_text'] = scenario_text # Store the last scenario in the session state for use by the Scenario Assistant

                else:
                    # If a scenario has been generated previously, display it
                    if 'scenario_text' in st.session_state and st.session_state['scenario_generated']:
                        st.markdown("---")
                        st.markdown(st.session_state['scenario_text'])
                        st.download_button(label="Download Scenario", data=st.session_state['scenario_text'], file_name="threat_group_scenario.md", mime="text/markdown")

    elif model_provider == "Groq API":
        if st.button('Generate Scenario', key='generate_scenario_groq'):
            if not os.environ["GROQ_API_KEY"]:
                st.info("Please add your Groq API key to continue.")
            if not os.environ["GROQ_MODEL"]:
                st.info("Please select a model to continue.")
            elif not industry:
                st.info("Please select your company's industry to continue.")
            elif not company_size:
                st.info("Please select your company's size to continue.")
            elif techniques_df.empty:
                st.info(f"Please select a {entity_label} with associated techniques.")
            else:
                groq_api_key = st.session_state.get('GROQ_API_KEY')
                model_name = os.getenv('GROQ_MODEL')
                response = generate_scenario_groq_wrapper(groq_api_key, model_name, messages)
                st.markdown("---")
                if response is not None:
                    st.session_state['scenario_generated'] = True
                    content = response.choices[0].message.content
                    
                    # Check if this is DeepSeek output with thinking tags
                    if re.search(r'<think>(.*?)</think>', content, re.DOTALL):
                        # Extract the thinking content and the rest of the scenario
                        thinking_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
                        thinking_content = thinking_match.group(1).strip()
                        scenario_text = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                        
                        # Display thinking content in an expander
                        with st.expander("View Model's Reasoning"):
                            st.markdown(thinking_content)
                    else:
                        # If no thinking tags, use the entire content as the scenario
                        scenario_text = content
                    
                    # Clean up the scenario text by removing code block markers if present
                    scenario_text = re.sub(r'^```\w*\n|```$', '', scenario_text, flags=re.MULTILINE).strip()
                    
                    st.session_state['scenario_text'] = scenario_text  # Store the generated scenario in the session state
                    st.markdown(scenario_text)
                    st.download_button(label="Download Scenario", data=st.session_state['scenario_text'], file_name="threat_group_scenario.md", mime="text/markdown")

                    st.session_state['last_scenario'] = True
                    st.session_state['last_scenario_text'] = scenario_text # Store the last scenario in the session state for use by the Scenario Assistant

                else:
                    # If a scenario has been generated previously, display it
                    if 'scenario_text' in st.session_state and st.session_state['scenario_generated']:
                        st.markdown("---")
                        st.markdown(st.session_state['scenario_text'])
                        st.download_button(label="Download Scenario", data=st.session_state['scenario_text'], file_name="threat_group_scenario.md", mime="text/markdown")

    elif model_provider == "OpenAI API":
        if st.button('Generate Scenario', key='generate_scenario_openai'):
            openai_api_key = st.session_state.get('openai_api_key') or os.environ.get('OPENAI_API_KEY')
            model_name = st.session_state.get('model_name') or os.environ.get('OPENAI_MODEL')
            if not openai_api_key:
                st.info("Please add your OpenAI API key to continue.")
            elif not model_name:
                st.info("Please select a model to continue.")
            elif not industry:
                st.info("Please select your company's industry to continue.")
            elif not company_size:
                st.info("Please select your company's size to continue.")
            elif techniques_df.empty:
                st.info(f"Please select a {entity_label} with associated techniques.")
            else:
                response = generate_scenario_wrapper(openai_api_key, model_name, messages)
                st.markdown("---")
                if response is not None:
                    try:
                        # Extract text content from Responses API structured response
                        if isinstance(response.content, list):
                            # Find text blocks in the structured response
                            text_blocks = [block.get('text', '') for block in response.content if block.get('type') == 'text']
                            scenario_text = '\n'.join(text_blocks)
                        else:
                            scenario_text = response.content
                        
                        st.session_state['scenario_generated'] = True
                        st.session_state['scenario_text'] = scenario_text
                        st.markdown(scenario_text)
                        st.download_button(label="Download Scenario", data=st.session_state['scenario_text'], file_name="threat_group_scenario.md", mime="text/markdown")
                        st.session_state['last_scenario'] = True
                        st.session_state['last_scenario_text'] = scenario_text
                    except Exception as processing_error:
                        st.error(f"An error occurred while processing the scenario response: {processing_error}")
                        st.text("Raw response object:")
                        st.json(str(response))
                else:
                    st.warning("Scenario generation failed. Check the error message above.")
                    if 'scenario_text' in st.session_state and st.session_state['scenario_generated']:
                        st.markdown("Displaying previously generated scenario:")
                        st.markdown(st.session_state['scenario_text'])
                        st.download_button(label="Download Scenario", data=st.session_state['scenario_text'], file_name="threat_group_scenario.md", mime="text/markdown")

    elif model_provider == "Custom":
        if st.button('Generate Scenario', key='generate_scenario_custom'):
            # Check for required custom settings
            if not st.session_state.get('custom_base_url'):
                st.info("Please set the Custom Base URL in the sidebar.")
            elif not st.session_state.get('model_name'):
                st.info("Please set the Custom Model Name in the sidebar.")
            elif not industry:
                st.info("Please select your company\'s industry to continue.")
            elif not company_size:
                st.info("Please select your company\'s size to continue.")
            elif techniques_df.empty:
                st.info(f"Please select a {entity_label} with associated techniques.")
            else:
                # Call the wrapper function (it now reads custom settings from session_state)
                response = generate_scenario_openai_wrapper(messages)
                st.markdown("---")
                if response is not None:
                    try:
                        # Add more robust checking for the response structure
                        if hasattr(response, 'choices') and response.choices:
                            first_choice = response.choices[0]
                            if hasattr(first_choice, 'message') and first_choice.message:
                                if hasattr(first_choice.message, 'content') and first_choice.message.content:
                                    st.session_state['scenario_generated'] = True
                                    scenario_text = first_choice.message.content
                                    st.session_state['scenario_text'] = scenario_text
                                    st.markdown(scenario_text)
                                    st.download_button(label="Download Scenario", data=st.session_state['scenario_text'], file_name="threat_group_scenario.md", mime="text/markdown")

                                    st.session_state['last_scenario'] = True
                                    st.session_state['last_scenario_text'] = scenario_text
                                else:
                                    st.error("Error processing response: 'content' attribute missing from message.")
                                    st.json(response.model_dump_json(indent=2)) # Display raw response
                            else:
                                st.error("Error processing response: 'message' attribute missing from the first choice.")
                                st.json(response.model_dump_json(indent=2)) # Display raw response
                        else:
                            st.error("Error processing response: 'choices' list is missing or empty.")
                            st.json(response.model_dump_json(indent=2)) # Display raw response
                    except Exception as processing_error:
                        st.error(f"An error occurred while processing the scenario response: {processing_error}")
                        st.text("Raw response object:")
                        st.json(response.model_dump_json(indent=2)) # Display raw response on processing error

                else:
                    # Response was None, likely due to an error during generation (already logged by the wrapper)
                    st.warning("Scenario generation failed. Check the error message above.")
                    # Optionally display previous scenario if needed
                    if 'scenario_text' in st.session_state and st.session_state['scenario_generated']:
                        st.markdown("Displaying previously generated scenario:")
                        st.markdown(st.session_state['scenario_text'])
                        st.download_button(label="Download Scenario", data=st.session_state['scenario_text'], file_name="threat_group_scenario.md", mime="text/markdown")

    # Display an info message if no API key is set
    if 'LANGCHAIN_API_KEY' not in st.secrets:
        st.info("ℹ️ No LangChain API key has been set. This run will not be logged to LangSmith.")                

    # Create a placeholder for the feedback message
    feedback_placeholder = st.empty()

    # Show the thumbs_up and thumbs_down buttons only when a scenario has been generated
    st.markdown("---")
    # Show the feedback buttons only if a scenario has been generated and the LangSmith client is initialized
    if st.session_state.get('scenario_generated', False) and client is not None:
        st.markdown("Rate the scenario to help improve this tool.")
        col1, col2, col3 = st.columns([0.5,0.5,5])
        with col1:
            thumbs_up = st.button("👍")
            if thumbs_up:
                try:
                    run_id = st.session_state.get('run_id')
                    if run_id:
                        feedback_type_str = "positive"
                        score = 1  # or 0
                        comment = ""

                        # Record the feedback
                        feedback_record = client.create_feedback(
                            run_id,
                            feedback_type_str,
                            score=score,
                            comment=comment,
                        )
                        st.session_state.feedback = {
                            "feedback_id": str(feedback_record.id),
                            "score": score,
                        }
                        # Update the feedback message in the placeholder
                        feedback_placeholder.success("Feedback submitted. Thank you.")
                    else:
                        # Update the feedback message in the placeholder
                        feedback_placeholder.warning("No run ID found. Please generate a scenario first.")
                except Exception as e:
                    # Update the feedback message in the placeholder
                    feedback_placeholder.error(f"An error occurred while creating feedback: {str(e)}")

        with col2:
            thumbs_down = st.button("👎")
            if thumbs_down:
                try:
                    run_id = st.session_state.get('run_id')
                    if run_id:
                        feedback_type_str = "negative"
                        score = 0  # or 0
                        comment = ""

                        # Record the feedback
                        feedback_record = client.create_feedback(
                            run_id,
                            feedback_type_str,
                            score=score,
                            comment=comment,
                        )
                        st.session_state.feedback = {
                            "feedback_id": str(feedback_record.id),
                            "score": score,
                        }
                        # Update the feedback message in the placeholder
                        feedback_placeholder.success("Feedback submitted. Thank you.")
                    else:
                        # Update the feedback message in the placeholder
                        feedback_placeholder.warning("No run ID found. Please generate a scenario first.")
                except Exception as e:
                    # Update the feedback message in the placeholder
                    feedback_placeholder.error(f"An error occurred while creating feedback: {str(e)}")

except Exception as e:
    st.error("An error occurred: " + str(e))

# Add a back button
link_to_homepage = "/"

st.markdown(
    f'<a href="{link_to_homepage}" style="display: inline-block; padding: 5px 20px; color: white; text-align: center; text-decoration: none; font-size: 16px; border-radius: 4px;">⬅️ Back</a>',
    unsafe_allow_html=True
)