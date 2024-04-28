import os
import pandas as pd
import streamlit as st
from langchain.callbacks.manager import collect_runs
from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_openai import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langsmith import Client, RunTree, traceable
from mitreattack.stix20 import MitreAttackData
from openai import AzureOpenAI

# ------------------ Streamlit UI ------------------ #

st.set_page_config(page_title="AttackGen Assistant", page_icon=":speech_balloon:")

st.markdown("# <span style='color: #1DB954;'>AttackGen Assistant💬</span>", unsafe_allow_html=True)

if 'last_scenario_text' in st.session_state and st.session_state['last_scenario']:
    input_scenario = st.session_state['last_scenario_text']
    with st.expander("Generated Scenario"):
        with st.container(height=400, border=True):
            st.markdown(input_scenario)

    # Create an empty container for the chat messages
    chat_container = st.empty()

    # Initialize st.session_state.messages if it doesn't exist
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi, I can help you update and ask questions about your incident response scenario."}
        ]

    # Display the chat messages in the empty container
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def generate_response(user_input, chat_history):
        model_provider = st.session_state["chosen_model_provider"]

        if model_provider == "Azure OpenAI Service":
            azure_api_key = os.getenv('AZURE_OPENAI_API_KEY')
            azure_api_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
            azure_deployment_name = os.getenv('AZURE_DEPLOYMENT')
            azure_api_version = os.getenv('OPENAI_API_VERSION')
            llm = AzureOpenAI(api_key=azure_api_key,
                              azure_endpoint=azure_api_endpoint,
                              api_version=azure_api_version)
            messages = [
                SystemMessagePromptTemplate.from_template("""
                You are an AI assistant that helps users update and ask questions about their incident response scenario.
                Only respond to questions or requests relating to the scenario, or incident response testing in general.
                """).format(),
                HumanMessagePromptTemplate.from_template(
                    "Here is the scenario that the user previously generated:\n\n{input_scenario}\n\nChat history:\n{chat_history}\n\nUser: {user_input}"
                ).format(input_scenario=input_scenario, chat_history=chat_history, user_input=user_input)
            ]
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
            response = llm.chat.completions.create(
                model=azure_deployment_name,
                messages=formatted_messages
            )
            return response.choices[0].message.content
        elif model_provider == "Google AI API":
            google_api_key = os.getenv('GOOGLE_API_KEY')
            model = os.getenv('GOOGLE_MODEL')
            llm = ChatGoogleGenerativeAI(google_api_key=google_api_key, model=model)
            messages = [
                SystemMessagePromptTemplate.from_template("""
                You are an AI assistant that helps users update and ask questions about their incident response scenario.
                Only respond to questions or requests relating to the scenario, or incident response testing in general.
                """).format(),
                HumanMessagePromptTemplate.from_template(
                    "Here is the scenario that the user previously generated:\n\n{input_scenario}\n\nChat history:\n{chat_history}\n\nUser: {user_input}"
                ).format(input_scenario=input_scenario, chat_history=chat_history, user_input=user_input)
            ]
            response = llm.invoke(messages)
            return response.content
        elif model_provider == "Mistral API":
            mistral_api_key = os.getenv('MISTRAL_API_KEY')
            model = os.getenv('MISTRAL_MODEL')
            llm = ChatMistralAI(mistral_api_key=mistral_api_key)
            messages = [
                SystemMessagePromptTemplate.from_template("""
                You are an AI assistant that helps users update and ask questions about their incident response scenario.
                Only respond to questions or requests relating to the scenario, or incident response testing in general.
                """).format(),
                HumanMessagePromptTemplate.from_template(
                    "Here is the scenario that the user previously generated:\n\n{input_scenario}\n\nChat history:\n{chat_history}\n\nUser: {user_input}"
                ).format(input_scenario=input_scenario, chat_history=chat_history, user_input=user_input)
            ]
            response = llm.invoke(messages, model=model)
            return response.content
        elif model_provider == "Ollama":
            model = os.getenv('OLLAMA_MODEL')
            llm = Ollama(model=model)
            prompt = (f"""
            You are an AI assistant that helps users update and ask questions about their incident response testing scenario.
            Only respond to questions or requests relating to the scenario, or incident response testing in general.\n\n
            Here is the scenario that the user previously generated:\n\n{input_scenario}\n\nChat history:\n{chat_history}\n\nUser: {user_input}"
            """)
            llm_result = llm.generate(prompts=[prompt])
            response = llm_result.generations[0][0].text
            return response
        else:
            openai_api_key = st.session_state.get('openai_api_key')
            model_name = st.session_state.get('model_name')
            llm = ChatOpenAI(openai_api_key=openai_api_key, model_name=model_name)
            messages = [
                SystemMessagePromptTemplate.from_template(
                    "You are an AI assistant that helps users update and ask questions about their incident response scenario."
                ).format(),
                HumanMessagePromptTemplate.from_template(
                    "Here is the scenario that the user previously generated:\n\n{input_scenario}\n\nChat history:\n{chat_history}\n\nUser: {user_input}"
                ).format(input_scenario=input_scenario, chat_history=chat_history, user_input=user_input)
            ]
            response = llm.generate(messages=[messages]).generations[0][0].text
            return response

    if prompt := st.chat_input("Type your message here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = generate_response(prompt, "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[:-1]]))
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Function to handle clearing the conversation
    def clear_conversation():
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi, I can help you update and ask questions about your incident response scenario."}
        ]
        # Clear the chat bubbles from the UI
        chat_container.empty()

        # Re-render the initial assistant message
        with chat_container:
            with st.chat_message("assistant"):
                st.markdown(st.session_state.messages[0]["content"])

    # Add the "Clear Conversation" button at the bottom of the screen
    with st.container():
        if st.button("Clear Conversation", key='clear_button'):
            clear_conversation()

else:
    st.info("No scenario found. Please generate a scenario first.")