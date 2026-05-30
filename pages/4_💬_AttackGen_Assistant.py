import os
import streamlit as st
import re

from langchain_anthropic import ChatAnthropic
from langchain_community.llms import Ollama
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_openai import ChatOpenAI, AzureOpenAI
from langchain_core.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate
from openai import OpenAI

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
                Format your responses using proper Markdown syntax with headers, bullet points, and formatting for readability.
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
                Format your responses using proper Markdown syntax with headers, bullet points, and formatting for readability.
                """).format(),
                HumanMessagePromptTemplate.from_template(
                    "Here is the scenario that the user previously generated:\n\n{input_scenario}\n\nChat history:\n{chat_history}\n\nUser: {user_input}"
                ).format(input_scenario=input_scenario, chat_history=chat_history, user_input=user_input)
            ]
            response = llm.invoke(messages)
            # Handle structured content format from newer Gemini models
            if isinstance(response.content, list):
                text_blocks = [block.get('text', '') for block in response.content if block.get('type') == 'text']
                return '\n'.join(text_blocks)
            return response.content
        elif model_provider == "Mistral API":
            mistral_api_key = os.getenv('MISTRAL_API_KEY')
            model = os.getenv('MISTRAL_MODEL')
            llm = ChatMistralAI(mistral_api_key=mistral_api_key)
            messages = [
                SystemMessagePromptTemplate.from_template("""
                You are an AI assistant that helps users update and ask questions about their incident response scenario.
                Only respond to questions or requests relating to the scenario, or incident response testing in general.
                Format your responses using proper Markdown syntax with headers, bullet points, and formatting for readability.
                """).format(),
                HumanMessagePromptTemplate.from_template(
                    "Here is the scenario that the user previously generated:\n\n{input_scenario}\n\nChat history:\n{chat_history}\n\nUser: {user_input}"
                ).format(input_scenario=input_scenario, chat_history=chat_history, user_input=user_input)
            ]
            response = llm.invoke(messages, model=model)
            return response.content
        elif model_provider == "Groq API":
            groq_api_key = os.getenv('GROQ_API_KEY')
            model = os.getenv('GROQ_MODEL')
            llm = OpenAI(
                api_key=groq_api_key,
                base_url="https://api.groq.com/openai/v1",
            )
            messages = [
                SystemMessagePromptTemplate.from_template("""
                You are an AI assistant that helps users update and ask questions about their incident response scenario.
                Only respond to questions or requests relating to the scenario, or incident response testing in general.
                Format your responses using proper Markdown syntax with headers, bullet points, and formatting for readability.
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
                model=model,
                messages=formatted_messages
            )
            content = response.choices[0].message.content
            
            # Check if this is DeepSeek output with thinking tags
            if re.search(r'<think>(.*?)</think>', content, re.DOTALL):
                # Extract the thinking content and the rest of the response
                thinking_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
                thinking_content = thinking_match.group(1).strip()
                response_text = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                
                # Display thinking content in an expander
                with st.expander("View Model's Reasoning"):
                    st.markdown(thinking_content)
                
                # Clean up the response text by removing code block markers if present
                response_text = re.sub(r'^```\w*\n|```$', '', response_text, flags=re.MULTILINE).strip()
                return response_text
            else:
                # If no thinking tags, clean up and return the entire content
                return re.sub(r'^```\w*\n|```$', '', content, flags=re.MULTILINE).strip()
        elif model_provider == "Ollama":
            model = os.getenv('OLLAMA_MODEL')
            llm = Ollama(model=model)
            prompt = (f"""
            You are an AI assistant that helps users update and ask questions about their incident response testing scenario.
            Only respond to questions or requests relating to the scenario, or incident response testing in general.\n\n
            Here is the scenario that the user previously generated:\n\n{input_scenario}\n\nChat history:\n{chat_history}\n\nUser: {user_input}"
            """)
            llm_result = llm.invoke(prompt)
            # Extract text content from Responses API structured response
            if isinstance(llm_result.content, list):
                # Find text blocks in the structured response
                text_blocks = [block.get('text', '') for block in llm_result.content if block.get('type') == 'text']
                response = '\n'.join(text_blocks)
            else:
                response = llm_result.content
            return response
        elif model_provider == "Anthropic API":
            anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
            model = os.getenv('ANTHROPIC_MODEL')
            # Set max_tokens based on the model
            max_tokens = 8192  # Default for Haiku
            if "opus-4" in model:
                max_tokens = 32000
            elif "sonnet-4" in model or "3-7-sonnet" in model:
                max_tokens = 64000
            
            llm = ChatAnthropic(anthropic_api_key=anthropic_api_key, model_name=model, temperature=0.7, max_tokens=max_tokens)
            messages = [
                SystemMessagePromptTemplate.from_template("""
                You are an AI assistant that helps users update and ask questions about their incident response scenario.
                Only respond to questions or requests relating to the scenario, or incident response testing in general.
                Format your responses using proper Markdown syntax with headers, bullet points, and formatting for readability.
                """).format(),
                HumanMessagePromptTemplate.from_template(
                    "Here is the scenario that the user previously generated:\n\n{input_scenario}\n\nChat history:\n{chat_history}\n\nUser: {user_input}"
                ).format(input_scenario=input_scenario, chat_history=chat_history, user_input=user_input)
            ]
            response = llm.invoke(messages)
            return response.content
        elif model_provider == "Custom":
            # Fetch custom provider details from session state
            custom_api_key = st.session_state.get('custom_api_key')
            custom_model_name = st.session_state.get('custom_model_name')
            custom_base_url = st.session_state.get('custom_base_url')

            if not custom_base_url:
                return "Error: Custom Base URL not set in sidebar."
            if not custom_model_name:
                return "Error: Custom Model Name not set in sidebar."

            # Conditionally prepare client arguments
            client_args = {
                "base_url": custom_base_url,
            }
            if custom_api_key:
                client_args["api_key"] = custom_api_key

            try:
                llm = OpenAI(**client_args)

                # Prepare messages in OpenAI format
                messages_for_api = [
                    {
                        "role": "system",
                        "content": "You are an AI assistant that helps users update and ask questions about their incident response scenario. Only respond to questions or requests relating to the scenario, or incident response testing in general."
                    },
                    {
                        "role": "user",
                        "content": f"Here is the scenario that the user previously generated:\n\n{input_scenario}\n\nChat history:\n{chat_history}\n\nUser: {user_input}"
                    }
                ]

                response = llm.chat.completions.create(
                    model=custom_model_name,
                    messages=messages_for_api,
                    temperature=0.7, # Match previous curl example
                    # max_tokens=-1, # Consider if needed, OpenAI default is usually fine
                    stream=False
                )
                return response.choices[0].message.content
            except Exception as e:
                # import traceback
                st.error(f"Error communicating with Custom API: {e}")
                # st.text(traceback.format_exc())
                return "Error: Failed to get response from Custom API."

        else: # Default to OpenAI API
            openai_api_key = st.session_state.get('openai_api_key')
            model_name = st.session_state.get('model_name')
            
            # All models use the unified Responses API
            llm = ChatOpenAI(openai_api_key=openai_api_key, model_name=model_name, output_version="responses/v1")
            
            messages = [
                SystemMessagePromptTemplate.from_template(
                    "You are an AI assistant that helps users update and ask questions about their incident response scenario. Format your responses using proper Markdown syntax with headers, bullet points, and formatting for readability."
                ).format(),
                HumanMessagePromptTemplate.from_template(
                    "Here is the scenario that the user previously generated:\n\n{input_scenario}\n\nChat history:\n{chat_history}\n\nUser: {user_input}"
                ).format(input_scenario=input_scenario, chat_history=chat_history, user_input=user_input)
            ]
            
            llm_result = llm.invoke(messages)
            # Extract text content from Responses API structured response
            if isinstance(llm_result.content, list):
                # Find text blocks in the structured response
                text_blocks = [block.get('text', '') for block in llm_result.content if block.get('type') == 'text']
                response = '\n'.join(text_blocks)
            else:
                response = llm_result.content
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