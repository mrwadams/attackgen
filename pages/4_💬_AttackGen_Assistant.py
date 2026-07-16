import streamlit as st

from core.llm import call_llm_stream
from core.response import clean_model_response, stream_filter_thinking
from core.schemas import LLMConfig
from core.state import restore_from_query_params
from core.styles import inject_emoji_fonts

# Restore sidebar selections on direct page loads (e.g. browser refresh while
# on this page). See core/state.py for the persisted-keys list.
restore_from_query_params()


SCENARIO_SYSTEM_PROMPT = (
    "You are an AI assistant that helps users update and ask questions about their incident "
    "response scenario. Only respond to questions or requests relating to the scenario, or "
    "incident response testing in general. Format your responses using proper Markdown syntax "
    "with headers, bullet points, and formatting for readability."
)

DEFENSE_SYSTEM_PROMPT = (
    "You are an AI assistant that helps users refine the purple-team Detection & Response "
    "narrative that accompanies their incident response scenario. The narrative walks the "
    "scenario from the defender's side — detection opportunities, log sources, and response "
    "actions, stage by stage. Only respond to questions or requests relating to the detection "
    "and response of this scenario, or purple-team testing in general. Keep your suggestions "
    "grounded in the scenario provided for reference. Format your responses using proper "
    "Markdown syntax with headers, bullet points, and formatting for readability."
)

BOTH_SYSTEM_PROMPT = (
    "You are an AI assistant that helps users refine an incident response scenario and its "
    "accompanying purple-team Detection & Response narrative together. When a requested change "
    "affects both — a different threat actor, industry, technique, or timeline — apply it "
    "consistently across the two so the attacker's scenario and the defender's walkthrough stay "
    "aligned, and make clear which output each part of your response applies to. Only respond to "
    "questions or requests relating to the scenario, its detection and response, or incident "
    "response testing in general. Format your responses using proper Markdown syntax with "
    "headers, bullet points, and formatting for readability."
)


st.set_page_config(page_title="AttackGen Assistant", page_icon=":speech_balloon:")
inject_emoji_fonts()

st.markdown("# <span style='color: #1DB954;'>AttackGen Assistant💬</span>", unsafe_allow_html=True)


scenario_text = (
    st.session_state["last_scenario_text"]
    if st.session_state.get("last_scenario") and "last_scenario_text" in st.session_state
    else None
)
defense_narrative = st.session_state.get("last_defense_narrative")

if not scenario_text:
    st.info("No scenario found. Please generate a scenario first.")
    st.stop()


# Pick what to edit. The Detection & Response and combined options only appear
# when a purple-team narrative was generated alongside the scenario (page 1/2
# toggle). The combined option refines both together so a change made to one can
# be carried consistently into the other.
if defense_narrative:
    choice = st.radio(
        "Editing:",
        ["Scenario", "Detection & Response", "Scenario + Detection & Response"],
        horizontal=True,
        key="assistant_target",
    )
else:
    choice = "Scenario"
target = {
    "Detection & Response": "defense",
    "Scenario + Detection & Response": "both",
}.get(choice, "scenario")

if target == "defense":
    panels = [("Detection & Response Narrative", defense_narrative)]
    system_prompt = DEFENSE_SYSTEM_PROMPT
    greeting = "Hi, I can help you refine the Detection & Response narrative for your scenario."
    trace_name = "AttackGen Assistant — Detection & Response"
    trace_tags = ("assistant", "purple_team_narrative")
elif target == "both":
    panels = [
        ("Generated Scenario", scenario_text),
        ("Detection & Response Narrative", defense_narrative),
    ]
    system_prompt = BOTH_SYSTEM_PROMPT
    greeting = (
        "Hi, I can help you refine the scenario and its Detection & Response narrative "
        "together, keeping changes consistent across both."
    )
    trace_name = "AttackGen Assistant — Scenario + Detection & Response"
    trace_tags = ("assistant", "purple_team_narrative")
else:
    panels = [("Generated Scenario", scenario_text)]
    system_prompt = SCENARIO_SYSTEM_PROMPT
    greeting = "Hi, I can help you update and ask questions about your incident response scenario."
    trace_name = "AttackGen Assistant"
    trace_tags = ("assistant",)


for label, content in panels:
    with st.expander(label):
        with st.container(height=400, border=True):
            st.markdown(content)

chat_container = st.empty()

# Keep a separate history per target so switching between the scenario and the
# Detection & Response narrative doesn't feed one artifact's chat into the other.
messages_key = f"assistant_messages_{target}"
if messages_key not in st.session_state:
    st.session_state[messages_key] = [{"role": "assistant", "content": greeting}]

with chat_container:
    for message in st.session_state[messages_key]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def generate_response(user_input, chat_history):
    if target == "both":
        context = (
            f"Here is the incident response scenario:\n\n{scenario_text}\n\n"
            f"Here is the accompanying Detection & Response narrative:\n\n{defense_narrative}\n\n"
            f"The user wants to refine both together. When a requested change affects both, "
            f"apply it consistently across them and show the update to each.\n\n"
            f"Chat history:\n{chat_history}\n\nUser: {user_input}"
        )
    elif target == "defense":
        context = (
            f"Here is the scenario, for reference:\n\n{scenario_text}\n\n"
            f"Here is the current Detection & Response narrative the user wants to refine:"
            f"\n\n{defense_narrative}\n\n"
            f"Chat history:\n{chat_history}\n\nUser: {user_input}"
        )
    else:
        context = (
            f"Here is the scenario that the user previously generated:\n\n{scenario_text}\n\n"
            f"Chat history:\n{chat_history}\n\nUser: {user_input}"
        )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": context},
    ]
    config = LLMConfig.from_session_state(
        trace_name=trace_name,
        trace_tags=trace_tags,
    )
    raw_chunks: list[str] = []

    def _tee(chunks):
        for chunk in chunks:
            raw_chunks.append(chunk)
            yield chunk

    try:
        yield from stream_filter_thinking(_tee(call_llm_stream(config, messages)))
    except Exception as e:
        yield f"\n\nAn error occurred while calling the model: {e}"
        st.session_state["_last_assistant_cleaned"] = (
            f"An error occurred while calling the model: {e}"
        )
        return

    raw = "".join(raw_chunks)
    thinking, cleaned = clean_model_response(raw)
    if thinking:
        with st.expander("View Model's Reasoning"):
            st.markdown(thinking)
    st.session_state["_last_assistant_cleaned"] = cleaned


if prompt := st.chat_input("Type your message here..."):
    st.session_state[messages_key].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        history = "\n".join(
            f"{m['role']}: {m['content']}" for m in st.session_state[messages_key][:-1]
        )
        st.write_stream(generate_response(prompt, history))

    st.session_state[messages_key].append(
        {"role": "assistant", "content": st.session_state.pop("_last_assistant_cleaned", "")}
    )


def clear_conversation():
    st.session_state[messages_key] = [{"role": "assistant", "content": greeting}]
    chat_container.empty()
    with chat_container:
        with st.chat_message("assistant"):
            st.markdown(st.session_state[messages_key][0]["content"])


with st.container():
    if st.button("Clear Conversation", key='clear_button'):
        clear_conversation()
