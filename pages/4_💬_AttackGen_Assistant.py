import streamlit as st

from core.assistant import (
    CLEANED_REPLY_KEY,
    TARGETS,
    apply_summary_note,
    apply_write_back,
    render_empty_state,
    render_scenario_identity,
    resolve_download_artifacts,
    run_apply,
    scenario_handoff,
    stream_assistant_reply,
)
from core.scenario_page import applied_caption, mark_applied, stash_pre_apply
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

page_id = (scenario.meta or {}).get("page_id")
if page_id:
    caption = applied_caption(page_id)
    if caption:
        st.caption(caption)

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


st.markdown("---")

# The chat only ever refines a conversation; nothing reaches the downloadable
# artifact (or the Assistant's own panels above) until this dedicated call
# rewrites it. Gated on a real user turn so there is always something to apply.
has_user_turn = any(m["role"] == "user" for m in st.session_state[messages_key])
apply_clicked = st.button(
    "Apply changes to downloads",
    key=f"assistant_apply_{target}",
    disabled=not has_user_turn,
)
if not has_user_turn:
    st.caption(
        "Ask for at least one change in the chat above before applying it to "
        "the downloads."
    )

if apply_clicked:
    history = "\n".join(
        f"{m['role']}: {m['content']}" for m in st.session_state[messages_key]
    )
    try:
        with st.spinner("Applying changes..."):
            applied = run_apply(target=target, scenario=scenario, chat_history=history)
    except Exception as e:  # noqa: BLE001 - shown in the chat, no state changes
        with st.chat_message("assistant"):
            st.error(f"An error occurred while applying changes: {e}")
    else:
        if applied is None:
            # Refused, not applied: no session state changes, so this is shown
            # for this run only rather than appended to the chat history.
            with st.chat_message("assistant"):
                st.error(
                    "The model returned an empty response, so nothing was applied. "
                    "Try again, or ask a follow-up question first."
                )
        elif not page_id:
            with st.chat_message("assistant"):
                st.error("No originating page was found, so nothing was applied.")
        else:
            stash_pre_apply(page_id)
            for artifact in applied:
                apply_write_back(
                    page_id=page_id,
                    artifact=artifact.artifact,
                    revised_text=artifact.text,
                )
            mark_applied(page_id)
            st.session_state[messages_key].append(
                {"role": "assistant", "content": apply_summary_note(applied)}
            )
            st.rerun()

download_specs = []
artifacts = resolve_download_artifacts(scenario)
if target in ("scenario", "both") and "scenario" in artifacts:
    filename, data = artifacts["scenario"]
    download_specs.append(("Download Scenario", data, filename))
if target in ("defense", "both") and "defense" in artifacts:
    filename, data = artifacts["defense"]
    download_specs.append(("Download Detection & Response", data, filename))
if download_specs:
    for column, (label, data, filename) in zip(
        st.columns(len(download_specs)), download_specs
    ):
        with column:
            st.download_button(
                label=label,
                data=data,
                file_name=filename,
                mime="text/markdown",
                key=f"assistant_download_{filename}",
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
