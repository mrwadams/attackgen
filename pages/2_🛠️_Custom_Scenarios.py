import os
import pandas as pd
import streamlit as st
import re

from langchain.callbacks.manager import collect_runs
from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_openai import ChatOpenAI, AzureOpenAI
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langsmith import Client, RunTree, traceable
from mitreattack.stix20 import MitreAttackData
from openai import OpenAI


# ------------------ Streamlit UI Configuration ------------------ #

# Add environment variables for LangSmith
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "AttackGen"

# Initialise the LangSmith client if an API key is available
api_key = os.getenv('LANGSMITH_API_KEY')    

client = Client(api_key=api_key) if api_key else None

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

# Get the model provider and other required session state variables
model_provider = st.session_state["chosen_model_provider"]
industry = st.session_state["industry"]
company_size = st.session_state["company_size"]
matrix = st.session_state["matrix"]

# Set the default value for the custom_scenario_generated session state variable
if "custom_scenario_generated" not in st.session_state:
    st.session_state["custom_scenario_generated"] = False

st.set_page_config(
    page_title="Generate Custom Scenario",
    page_icon="🛠️",
)

# ------------------ Incident Response Templates ------------------ #
incident_response_templates = {
    "Enterprise": {
        "Phishing Attack": ["Spearphishing Attachment (T1193)", "User Execution (T1204)", "Browser Extensions (T1176)", "Credentials from Password Stores (T1555)", "Input Capture (T1056)", "Exfiltration Over C2 Channel (T1041)"],
        "Ransomware Attack": ["Exploit Public-Facing Application (T1190)", "Windows Management Instrumentation (T1047)", "Create Account (T1136)", "Process Injection (T1055)", "Data Encrypted for Impact (T1486)"],
        "Malware Infection": ["Supply Chain Compromise (T1195)", "Command and Scripting Interpreter (T1059)", "Registry Run Keys / Startup Folder (T1060)", "Obfuscated Files or Information (T1027)", "Remote Services (T1021)", "Data Destruction (T1485)"],
        "Insider Threat": ["Valid Accounts (T1078)", "Account Manipulation (T1098)", "Exploitation for Privilege Escalation (T1068)", "Data Staged (T1074)", "Scheduled Transfer (T1029)", "Account Access Removal (T1531)"],
        "Cross-Site Scripting (XSS) Attack": ["Exploit Public-Facing Application (T1190)", "User Execution (T1204)", "Input Capture (T1056)", "Exfiltration Over Web Service (T1567)"],
        "SQL Injection Attack": ["Exploit Public-Facing Application (T1190)", "Exploitation for Credential Access (T1212)", "Exfiltration Over Web Service (T1567)"],
        "API Compromise": ["Active Scanning (T1595)", "Exploit Public-Facing Application (T1190)", "Exploitation for Client Execution (T1203)", "Valid Accounts (T1078)", "Application Layer Protocol (T1071)", "Data Manipulation (T1565)"],
    },
    "ICS": {
        "Remote Access Exploitation": ["External Remote Services (T0822)", "Exploit Public-Facing Application (T0819)", "Remote Services (T0886)", "Wireless Compromise (T0860)"],
        "ICS Data Manipulation": ["Modify Controller Tasking (T0821)", "Manipulate I/O Image (T0835)", "Modify Alarm Settings (T0838)", "Modify Parameter (T0836)"],
        "Denial of Service": ["Denial of Service (T0814)", "Activate Firmware Update Mode (T0800)", "Brute Force I/O (T0806)", "Block Command Message (T0803)"],
        "ICS Reconnaissance": ["Network Sniffing (T0842)", "Remote System Discovery (T0846)", "Network Connection Enumeration (T0840)", "Program Upload (T0845)"],
    }
}


# ------------------ Helper Functions ------------------ #

# Load and cache the MITRE ATT&CK data
@st.cache_resource
def load_attack_data():
    enterprise_data = MitreAttackData("./data/enterprise-attack.json")
    ics_data = MitreAttackData("./data/ics-attack.json")
    return {"enterprise": enterprise_data, "ics": ics_data}

attack_data = load_attack_data()

# Get all techniques
@st.cache_resource
def load_techniques():
    try:
        matrix = st.session_state.get("matrix", "Enterprise")
        techniques = attack_data[matrix.lower()].get_techniques()
        techniques_list = []
        for technique in techniques:
            for reference in technique.external_references:
                if "external_id" in reference:
                    techniques_list.append({
                        'id': technique.id,
                        'Technique Name': technique.name,
                        'External ID': reference['external_id'],
                        'Display Name': f"{technique.name} ({reference['external_id']})"
                    })
        techniques_df = pd.DataFrame(techniques_list)
        
        return techniques_df
    except Exception as e:
        print(f"Error in load_techniques: {e}")
        return pd.DataFrame() # Return an empty DataFrame

techniques_df = load_techniques()

def generate_scenario_wrapper(openai_api_key, model_name, messages):
    if client is not None:  # If LangChain client has been initialized
        @traceable(run_type="llm", name="Custom Scenario", tags=["openai", "custom_scenario"], client=client)
        def generate_scenario(openai_api_key, model_name, messages, *, run_tree: RunTree):
            model_name = st.session_state["model_name"]
            try:
                with st.status('Generating scenario...', expanded=True):
                    st.write("Initialising AI model.")
                    
                    # Check if the model is o1-preview or o1-mini
                    if model_name in ["o3-mini", "o1", "o1-mini"]:
                        llm = ChatOpenAI(openai_api_key=openai_api_key, model_name=model_name, streaming=False, temperature=1.0)
                        # Remove the 'system' message and combine it with the first 'human' message
                        human_messages = [msg for msg in messages if msg.type == 'human']
                        if human_messages:
                            system_content = next((msg.content for msg in messages if msg.type == 'system'), '')
                            human_messages[0].content = f"{system_content}\n\n{human_messages[0].content}"
                        messages = human_messages
                    else:
                        llm = ChatOpenAI(openai_api_key=openai_api_key, model_name=model_name, streaming=False)
                    
                    st.write("Model initialised. Generating scenario, please wait.")
                    response = llm.generate(messages=[messages])
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
                    
                    # Check if the model is o1-preview or o1-mini
                    if model_name in ["o3-mini", "o1", "o1-mini"]:
                        llm = ChatOpenAI(openai_api_key=openai_api_key, model_name=model_name, streaming=False, temperature=1.0)
                        # Remove the 'system' message and combine it with the first 'human' message
                        human_messages = [msg for msg in messages if msg.type == 'human']
                        if human_messages:
                            system_content = next((msg.content for msg in messages if msg.type == 'system'), '')
                            human_messages[0].content = f"{system_content}\n\n{human_messages[0].content}"
                        messages = human_messages
                    else:
                        llm = ChatOpenAI(openai_api_key=openai_api_key, model_name=model_name, streaming=False)
                    
                    st.write("Model initialised. Generating scenario, please wait.")
                    response = llm.generate(messages=[messages])
                    st.write("Scenario generated successfully.")
                    return response
            except Exception as e:
                st.error("An error occurred while generating the scenario: " + str(e))
                return None
    
    return generate_scenario(openai_api_key, model_name, messages)

def generate_scenario_azure_wrapper(messages):
    if client is not None:  # LangSmith client has been initialised
        @traceable(run_type="llm", name="Custom Scenario (Azure OpenAI)", tags=["azure", "custom_scenario"], client=client if client is not None else None)
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
        @traceable(run_type="llm", name="Custom Scenario (Google AI API)", tags=["google", "custom_scenario"], client=client)
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
        @traceable(run_type="llm", name="Custom Scenario (Mistral API)", tags=["mistral", "custom_scenario"], client=client)
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

def generate_scenario_groq_wrapper(groq_api_key, model_name, messages):
    if client is not None: # If LangSmith client has been initialised
        @traceable(run_type="llm", name="Custom Scenario (Groq API)", tags=["groq", "custom_scenario"], client=client)
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

# Wrapper for Custom OpenAI-compatible endpoints
def generate_custom_scenario_openai_wrapper(messages):
    base_url = st.session_state.get('custom_base_url')
    openai_api_key = st.session_state.get('custom_api_key')
    model = st.session_state.get('custom_model_name')

    if not base_url:
        st.error("Custom base URL must be set for the custom model provider.")
        return None
    if not model:
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
                 st.warning(f"Skipping message template of type {type(message)}")
                 continue
            else:
                 st.error(f"Unsupported message format: {type(message)} - {message}")
                 return None # Stop processing if message format is wrong

    if not formatted_messages:
        st.error("No valid messages found to send to the model.")
        return None

    if client is not None:  # If LangChain client has been initialized
        @traceable(run_type="llm", name="Custom Scenario (Custom)", tags=["custom", "custom_scenario"], client=client)
        def generate_scenario_custom(formatted_messages_arg, *, run_tree: RunTree):
            try:
                # Re-fetch variables inside traceable function scope if necessary
                current_api_key = st.session_state.get('custom_api_key')
                current_model = st.session_state.get('custom_model_name')
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
                        messages=formatted_messages_arg, # Use the formatted messages
                        temperature=0.7,
                        max_tokens=-1,
                        stream=False
                    )
                    st.write("Scenario generated successfully.")
                    st.session_state['run_id'] = str(run_tree.id)  # Store the run ID in the session state
                    return response
            except Exception as e:
                import traceback
                st.error(f"An error occurred while generating the scenario: {str(e)}")
                st.error("Traceback:")
                st.text(traceback.format_exc()) # Print the full traceback
                st.session_state['run_id'] = str(run_tree.id)  # Ensure run_id is updated even on failure
                return None
        return generate_scenario_custom(formatted_messages)

    else:  # If LangChain client has not been initialized
        def generate_scenario_custom_no_trace(formatted_messages_arg):
            try:
                # Fetch variables directly from session state
                current_api_key = st.session_state.get('custom_api_key')
                current_model = st.session_state.get('custom_model_name')
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
                import traceback
                st.error(f"An error occurred while generating the scenario: {str(e)}")
                st.error("Traceback:")
                st.text(traceback.format_exc()) # Print the full traceback
                return None
        return generate_scenario_custom_no_trace(formatted_messages)

def template_selection(template, matrix):
    try:
        if template in incident_response_templates[matrix]:
            template_techniques = incident_response_templates[matrix][template]
            matrix_techniques = attack_data[matrix.lower()].get_techniques()
            matrix_technique_names = [f"{technique['name']} ({attack_data[matrix.lower()].get_attack_id(technique['id'])})" for technique in matrix_techniques]
            selected_techniques = [tech for tech in template_techniques if tech in matrix_technique_names]
            st.session_state['selected_techniques'] = selected_techniques
        else:
            st.write(f"Template {template} not found in {matrix} matrix")
    except Exception as e:
        st.error(f"An error occurred in template_selection: {str(e)}")


# ------------------ Streamlit UI ------------------ #
    

st.markdown("# <span style='color: #1DB954;'>Generate Custom Scenario🛠️</span>", unsafe_allow_html=True)

st.markdown("""
            ### Select ATT&CK Techniques
            """)

with st.expander("Use a Template (Optional)"):
    st.markdown("""
                Select a template to quickly generate a custom scenario based on a predefined set of ATT&CK techniques.
                """)

    # Get the current matrix from the session state
    matrix = st.session_state.get("matrix", "Enterprise")

    # Dropdown for selecting the incident response template
    selected_template = st.selectbox(
        "Select a template",
        options=[""] + list(incident_response_templates[matrix].keys()),  # Add an empty option for no selection
        format_func=lambda x: "Select a template" if x == "" else x  # Display placeholder text
    )

    # Automatically update selected techniques when a template is chosen
    if selected_template:
        template_selection(selected_template, matrix)
st.markdown("")
st.markdown(""" 
            Use the multi-select box below to add or update the ATT&CK techniques that you would like to include in a custom incident response testing scenario.
            """)

selected_techniques = []
if not techniques_df.empty:
    matrix = st.session_state.get("matrix", "Enterprise")
    techniques = attack_data[matrix.lower()].get_techniques()
    technique_options = [f"{technique['name']} ({attack_data[matrix.lower()].get_attack_id(technique['id'])})" for technique in techniques]
    selected_techniques = st.multiselect("Select ATT&CK Techniques", options=technique_options, default=st.session_state.get('selected_techniques', []))
    if matrix == "Enterprise":
        st.info("📝 Techniques are searchable by either their name or technique ID (e.g. `T1556` or `Phishing`).")
    elif matrix == "ICS":
        st.info("📝 Techniques are searchable by either their name or technique ID (e.g. `T0814` or `Denial of Service`).")
    else:
        st.info("📝 Techniques are searchable by either their name or technique ID.")
    
try:
    if len(selected_techniques) > 0:
        selected_techniques_string = '\n'.join(selected_techniques)
        template_info = f"This is a '{selected_template}' scenario." if selected_template else ""

        # Create System Message Template
        system_template = "You are a cybersecurity expert. Your task is to produce a comprehensive incident response testing scenario based on the information provided."
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

        # Create Human Message Template
        human_template = ("""
**Background information:**
The company operates in the '{industry}' industry and is of size '{company_size}'.

**Threat actor information:**
{template_info}
The threat actor is known to use the following ATT&CK techniques from the {matrix} Matrix:
{selected_techniques_string}

**Your task:**
Create a custom incident response testing scenario based on the information provided. The goal of the scenario is to test the company's incident response capabilities against a threat actor group that uses the identified ATT&CK techniques. 

Your response should be well structured and formatted using Markdown.
""")
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

        # Construct the ChatPromptTemplate
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

        # Format the prompt
        messages = chat_prompt.format_prompt(selected_techniques_string=selected_techniques_string, 
                                            industry=industry, 
                                            company_size=company_size,
                                            template_info=template_info,
                                            matrix=matrix).to_messages()
except Exception as e:
    st.error(f"An error occurred: {str(e)}")

st.markdown("")

# Display the scenario generation section
st.markdown("""
            ### Generate a Scenario

            Click the button below to generate a scenario based on the selected technique(s).

            It normally takes between 30-50 seconds to generate a scenario, although for local models this is highly dependent on your hardware and the selected model. ⏱️
            """)
try:
        if model_provider == "Azure OpenAI Service":
            if st.button('Generate Scenario', key='generate_custom_scenario_azure'):
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
                else:
                        response = generate_scenario_azure_wrapper(messages)
                        st.markdown("---")
                        if response is not None:
                            st.session_state['custom_scenario_generated'] = True
                            custom_scenario_text = response.choices[0].message.content
                            st.session_state['custom_scenario_text'] = custom_scenario_text  # Store the generated scenario in the session state
                            st.markdown(custom_scenario_text)
                            st.download_button(label="Download Scenario", data=st.session_state['custom_scenario_text'], file_name="custom_scenario.md", mime="text/markdown")

                            st.session_state['last_scenario'] = True
                            st.session_state['last_scenario_text'] = custom_scenario_text # Store the last scenario in the session state for use by the Scenario Assistant

                        else:
                            # If a scenario has been generated previously, display it
                            if 'custom_scenario_text' in st.session_state and st.session_state['custom_scenario_generated']:
                                st.markdown("---")
                                st.markdown(st.session_state['custom_scenario_text'])
                                st.download_button(label="Download Scenario", data=st.session_state['custom_scenario_text'], file_name="custom_scenario.md", mime="text/markdown")

        elif model_provider == "Google AI API":
            if st.button('Generate Scenario', key='generate_custom_scenario_google'):
                if not os.environ["GOOGLE_API_KEY"]:
                    st.info("Please add your Google AI API key to continue.")
                if not os.environ["GOOGLE_MODEL"]:
                    st.info("Please select a model to continue.")
                elif not industry:
                    st.info("Please select your company's industry to continue.")
                elif not company_size:
                    st.info("Please select your company's size to continue.")
                else:
                    google_api_key = st.session_state.get('google_api_key')
                    model_name = os.getenv('GOOGLE_MODEL')
                    response = generate_scenario_google_wrapper(google_api_key, model_name, messages)
                    st.markdown("---")
                    if response is not None:
                        st.session_state['custom_scenario_generated'] = True
                        custom_scenario_text = response.content
                        st.session_state['custom_scenario_text'] = custom_scenario_text  # Store the generated scenario in the session state
                        st.markdown(custom_scenario_text)
                        st.download_button(label="Download Scenario", data=st.session_state['custom_scenario_text'], file_name="custom_scenario.md", mime="text/markdown")

                        st.session_state['last_scenario'] = True
                        st.session_state['last_scenario_text'] = custom_scenario_text # Store the last scenario in the session state for use by the Scenario Assistant

                    else:
                        # If a scenario has been generated previously, display it
                        if 'custom_scenario_text' in st.session_state and st.session_state['custom_scenario_generated']:
                            st.markdown("---")
                            st.markdown(st.session_state['custom_scenario_text'])
                            st.download_button(label="Download Scenario", data=st.session_state['custom_scenario_text'], file_name="custom_scenario.md", mime="text/markdown")

        elif model_provider == "Mistral API":
            if st.button('Generate Scenario', key='generate_custom_scenario_mistral'):
                if not os.environ["MISTRAL_API_KEY"]:
                    st.info("Please add your Mistral API key to continue.")
                if not os.environ["MISTRAL_MODEL"]:
                    st.info("Please select a model to continue.")
                elif not industry:
                    st.info("Please select your company's industry to continue.")
                elif not company_size:
                    st.info("Please select your company's size to continue.")
                else:
                    mistral_api_key = st.session_state.get('mistral_api_key')
                    model_name = os.getenv('MISTRAL_MODEL')
                    response = generate_scenario_mistral_wrapper(mistral_api_key, model_name, messages)
                    st.markdown("---")
                    if response is not None:
                        st.session_state['custom_scenario_generated'] = True
                        custom_scenario_text = response.content
                        st.session_state['custom_scenario_text'] = custom_scenario_text  # Store the generated scenario in the session state
                        st.markdown(custom_scenario_text)
                        st.download_button(label="Download Scenario", data=st.session_state['custom_scenario_text'], file_name="custom_scenario.md", mime="text/markdown")

                        st.session_state['last_scenario'] = True
                        st.session_state['last_scenario_text'] = custom_scenario_text # Store the last scenario in the session state for use by the Scenario Assistant

                    else:
                        # If a scenario has been generated previously, display it
                        if 'custom_scenario_text' in st.session_state and st.session_state['custom_scenario_generated']:
                            st.markdown("---")
                            st.markdown(st.session_state['custom_scenario_text'])
                            st.download_button(label="Download Scenario", data=st.session_state['custom_scenario_text'], file_name="custom_scenario.md", mime="text/markdown")
        
        elif model_provider == "Ollama":
            if st.button('Generate Scenario', key='generate_custom_scenario_ollama'):
                if not os.environ["OLLAMA_MODEL"]:
                    st.info("Please select a model to continue.")
                elif not industry:
                    st.info("Please select your company's industry to continue.")
                elif not company_size:
                    st.info("Please select your company's size to continue.")
                else:
                    model = os.getenv('OLLAMA_MODEL')
                    response = generate_scenario_ollama_wrapper(model)
                    st.markdown("---")
                    if response is not None:
                        st.session_state['custom_scenario_generated'] = True
                        custom_scenario_text = response
                        st.session_state['custom_scenario_text'] = custom_scenario_text  # Store the generated scenario in the session state
                        st.markdown(custom_scenario_text)
                        st.download_button(label="Download Scenario", data=st.session_state['custom_scenario_text'], file_name="custom_scenario.md", mime="text/markdown")

                        st.session_state['last_scenario'] = True
                        st.session_state['last_scenario_text'] = custom_scenario_text # Store the last scenario in the session state for use by the Scenario Assistant

                    else:
                        # If a scenario has been generated previously, display it
                        if 'custom_scenario_text' in st.session_state and st.session_state['custom_scenario_generated']:
                            st.markdown("---")
                            st.markdown(st.session_state['custom_scenario_text'])
                            st.download_button(label="Download Scenario", data=st.session_state['custom_scenario_text'], file_name="custom_scenario.md", mime="text/markdown")

        elif model_provider == "Groq API":
            if st.button('Generate Scenario', key='generate_custom_scenario_groq'):
                if not os.environ["GROQ_API_KEY"]:
                    st.info("Please add your Groq API key to continue.")
                if not os.environ["GROQ_MODEL"]:
                    st.info("Please select a model to continue.")
                elif not industry:
                    st.info("Please select your company's industry to continue.")
                elif not company_size:
                    st.info("Please select your company's size to continue.")
                else:
                    groq_api_key = st.session_state.get('GROQ_API_KEY')
                    model_name = os.getenv('GROQ_MODEL')
                    response = generate_scenario_groq_wrapper(groq_api_key, model_name, messages)
                    st.markdown("---")
                    if response is not None:
                        st.session_state['custom_scenario_generated'] = True
                        content = response.choices[0].message.content
                        
                        # Check if this is DeepSeek output with thinking tags
                        if re.search(r'<think>(.*?)</think>', content, re.DOTALL):
                            # Extract the thinking content and the rest of the scenario
                            thinking_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
                            thinking_content = thinking_match.group(1).strip()
                            custom_scenario_text = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                            
                            # Display thinking content in an expander
                            with st.expander("View Model's Reasoning"):
                                st.markdown(thinking_content)
                        else:
                            # If no thinking tags, use the entire content as the scenario
                            custom_scenario_text = content
                        
                        # Clean up the scenario text by removing code block markers if present
                        custom_scenario_text = re.sub(r'^```\w*\n|```$', '', custom_scenario_text, flags=re.MULTILINE).strip()
                        
                        st.session_state['custom_scenario_text'] = custom_scenario_text  # Store the generated scenario in the session state
                        st.markdown(custom_scenario_text)
                        st.download_button(label="Download Scenario", data=st.session_state['custom_scenario_text'], file_name="custom_scenario.md", mime="text/markdown")

                        st.session_state['last_scenario'] = True
                        st.session_state['last_scenario_text'] = custom_scenario_text # Store the last scenario in the session state for use by the Scenario Assistant

                    else:
                        # If a scenario has been generated previously, display it
                        if 'custom_scenario_text' in st.session_state and st.session_state['custom_scenario_generated']:
                            st.markdown("---")
                            st.markdown(st.session_state['custom_scenario_text'])
                            st.download_button(label="Download Scenario", data=st.session_state['custom_scenario_text'], file_name="custom_scenario.md", mime="text/markdown")
        elif model_provider == "Custom":
            if st.button('Generate Scenario', key='generate_custom_scenario_custom'):
                # Check for required custom settings
                if not st.session_state.get('custom_base_url'):
                    st.info("Please set the Custom Base URL in the sidebar.")
                elif not st.session_state.get('custom_model_name'):
                    st.info("Please set the Custom Model Name in the sidebar.")
                elif not industry:
                    st.info("Please select your company\'s industry to continue.")
                elif not company_size:
                    st.info("Please select your company\'s size to continue.")
                elif not selected_techniques:
                    st.info("Please select at least one ATT&CK technique.")
                else:
                    # Call the wrapper function
                    response = generate_custom_scenario_openai_wrapper(messages)
                    st.markdown("---")
                    if response is not None:
                        try:
                            # Robust checking for the response structure
                            if hasattr(response, 'choices') and response.choices:
                                first_choice = response.choices[0]
                                if hasattr(first_choice, 'message') and first_choice.message:
                                    if hasattr(first_choice.message, 'content') and first_choice.message.content:
                                        st.session_state['custom_scenario_generated'] = True
                                        custom_scenario_text = first_choice.message.content
                                        st.session_state['custom_scenario_text'] = custom_scenario_text
                                        st.markdown(custom_scenario_text)
                                        st.download_button(label="Download Scenario", data=st.session_state['custom_scenario_text'], file_name="custom_scenario.md", mime="text/markdown")

                                        st.session_state['last_scenario'] = True
                                        st.session_state['last_scenario_text'] = custom_scenario_text
                                    else:
                                        st.error("Error processing response: 'content' attribute missing from message.")
                                        st.json(response.model_dump_json(indent=2))
                                else:
                                    st.error("Error processing response: 'message' attribute missing from the first choice.")
                                    st.json(response.model_dump_json(indent=2))
                            else:
                                st.error("Error processing response: 'choices' list is missing or empty.")
                                st.json(response.model_dump_json(indent=2))
                        except Exception as processing_error:
                            st.error(f"An error occurred while processing the scenario response: {processing_error}")
                            st.text("Raw response object:")
                            st.json(response.model_dump_json(indent=2))
                    else:
                        # Response was None
                        st.warning("Scenario generation failed. Check the error message above.")
                        if 'custom_scenario_text' in st.session_state and st.session_state['custom_scenario_generated']:
                            st.markdown("Displaying previously generated scenario:")
                            st.markdown(st.session_state['custom_scenario_text'])
                            st.download_button(label="Download Scenario", data=st.session_state['custom_scenario_text'], file_name="custom_scenario.md", mime="text/markdown")

        else:  # OpenAI API (default)
            if st.button('Generate Scenario', key="generate_custom_scenario"):
                openai_api_key = st.session_state.get('openai_api_key')
                model_name = st.session_state.get('model_name')
                if not openai_api_key:
                    st.info("Please add your OpenAI API key to continue.")
                if not model_name:
                    st.info("Please select a model to continue.")
                elif not industry:
                    st.info("Please select your company's industry to continue.")
                elif not company_size:
                    st.info("Please select your company's size to continue.")
                else:
                    # Generate a scenario
                    response = generate_scenario_wrapper(openai_api_key, model_name, messages)
                    st.markdown("---")
                    if response is not None:
                        st.session_state['custom_scenario_generated'] = True
                        custom_scenario_text = response.generations[0][0].text
                        st.session_state['custom_scenario_text'] = custom_scenario_text  # Store the generated scenario in the session state
                        st.markdown(custom_scenario_text)
                        st.download_button(label="Download Scenario", data=custom_scenario_text, file_name="custom_scenario.md", mime="text/markdown")

                        st.session_state['last_scenario'] = True
                        st.session_state['last_scenario_text'] = custom_scenario_text # Store the last scenario in the session state for use by the Scenario Assistant
                    else:
                        # If a scenario has been generated previously, display it
                        if 'custom_scenario_text' in st.session_state and st.session_state['custom_scenario_generated']:
                            st.markdown("---")
                            st.markdown(st.session_state['custom_scenario_text'])
                            st.download_button(label="Download Scenario", data=st.session_state['custom_scenario_text'], file_name="custom_scenario.md", mime="text/markdown")

        # Display an info message if no API key is set
        if 'LANGCHAIN_API_KEY' not in st.secrets:
            st.info("ℹ️ No LangChain API key has been set. This run will not be logged to LangSmith.")

        # Create a placeholder for the feedback message
        feedback_placeholder = st.empty()

        # Show the thumbs_up and thumbs_down buttons only when a scenario has been generated
        st.markdown("---")
        # Ensure the condition checks if 'custom_scenario_generated' is True and client is initialized
        if st.session_state.get('custom_scenario_generated', False) and client is not None:
            st.markdown("Rate the scenario to help improve this tool.")
            col1, col2, col3 = st.columns([0.5, 0.5, 5])
            with col1:
                thumbs_up = st.button("👍", key="thumbs_up_custom")
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