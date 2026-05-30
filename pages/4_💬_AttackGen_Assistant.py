import re

import streamlit as st

from core.llm import call_llm
from core.schemas import LLMConfig


SYSTEM_PROMPT = (
    "You are an AI assistant that helps users update and ask questions about their incident "
    "response scenario. Only respond to questions or requests relating to the scenario, or "
    "incident response testing in general. Format your responses using proper Markdown syntax "
    "with headers, bullet points, and formatting for readability."
)


st.set_page_config(page_title="AttackGen Assistant", page_icon=":speech_balloon:")

st.markdown("# <span style='color: #1DB954;'>AttackGen Assistant💬</span>", unsafe_allow_html=True)


def _post_process(text: str) -> tuple[str | None, str]:
    """Extract <think>...</think> reasoning blocks emitted by some models."""
    match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    thinking = match.group(1).strip() if match else None
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    cleaned = re.sub(r'^```\w*\n|```$', '', cleaned, flags=re.MULTILINE).strip()
    return thinking, cleaned


if 'last_scenario_text' in st.session_state and st.session_state.get('last_scenario'):
    input_scenario = st.session_state['last_scenario_text']
    with st.expander("Generated Scenario"):
        with st.container(height=400, border=True):
            st.markdown(input_scenario)

    chat_container = st.empty()

    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi, I can help you update and ask questions about your incident response scenario."}
        ]

    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def generate_response(user_input, chat_history):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Here is the scenario that the user previously generated:\n\n{input_scenario}\n\n"
                    f"Chat history:\n{chat_history}\n\nUser: {user_input}"
                ),
            },
        ]
        config = LLMConfig.from_session_state(
            trace_name="AttackGen Assistant",
            trace_tags=("assistant",),
        )
        try:
            raw = call_llm(config, messages)
        except Exception as e:
            return f"An error occurred while calling the model: {e}"

        thinking, cleaned = _post_process(raw)
        if thinking:
            with st.expander("View Model's Reasoning"):
                st.markdown(thinking)
        return cleaned

    if prompt := st.chat_input("Type your message here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            history = "\n".join(f"{m['role']}: {m['content']}" for m in st.session_state.messages[:-1])
            response = generate_response(prompt, history)
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

    def clear_conversation():
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi, I can help you update and ask questions about your incident response scenario."}
        ]
        chat_container.empty()
        with chat_container:
            with st.chat_message("assistant"):
                st.markdown(st.session_state.messages[0]["content"])

    with st.container():
        if st.button("Clear Conversation", key='clear_button'):
            clear_conversation()

else:
    st.info("No scenario found. Please generate a scenario first.")
