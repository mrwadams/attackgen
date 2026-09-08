import streamlit as st

from core.assistant import (
    CLEANED_REPLY_KEY,
    TARGETS,
    render_empty_state,
    render_scenario_identity,
    scenario_handoff,
    stream_assistant_reply,
)
from core.sidebar import render_setup_blockers, render_setup_sidebar
from core.styles import inject_emoji_fonts

st.set_page_config(page_title="AttackGen Assistant", page_icon=":speech_balloon:")
inject_emoji_fonts()
setup = render_setup_sidebar()

st.markdown("# <span style='color: #1DB954;'>AttackGen Assistant💬</span>", unsafe_allow_html=True)


# The scenario, its optional purple-team narrative and the metadata naming both
# are handed over by core.scenario_page when a base scenario is persisted.
scenario = scenario_handoff()

if scenario is None:
    # An empty state that can be acted on: the three ways to create a scenario,
    # rather than an instruction to go and find one.
    render_empty_state()
    st.stop()

render_scenario_identity(scenario)

# Pick what to edit. The Detection & Response and combined options only appear
# when a purple-team narrative was generated alongside the scenario (page 1/2
# toggle). The combined option refines both together so a change made to one can
# be carried consistently into the other.
if scenario.defense_narrative:
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
    panels = [("Detection & Response Narrative", scenario.defense_narrative)]
elif target == "both":
    panels = [
        ("Generated Scenario", scenario.text),
        ("Detection & Response Narrative", scenario.defense_narrative),
    ]
else:
    panels = [("Generated Scenario", scenario.text)]

greeting = TARGETS[target]["greeting"]

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


if not setup.complete:
    render_setup_blockers(setup)

if prompt := st.chat_input("Type your message here...", disabled=not setup.complete):
    st.session_state[messages_key].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        history = "\n".join(
            f"{m['role']}: {m['content']}" for m in st.session_state[messages_key][:-1]
        )
        st.write_stream(
            stream_assistant_reply(
                target=target,
                scenario=scenario,
                chat_history=history,
                user_input=prompt,
            )
        )

    st.session_state[messages_key].append(
        {"role": "assistant", "content": st.session_state.pop(CLEANED_REPLY_KEY, "")}
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
