import streamlit as st

from core.apply_recovery import applied_caption, mark_applied, stash_pre_apply
from core.assistant import (
    CLEANED_REPLY_KEY,
    CLEAR_BUTTON_KEY,
    Conversation,
    apply_summary_note,
    apply_write_back,
    assistant_downloads,
    choose_target,
    render_empty_state,
    render_scenario_identity,
    run_apply,
    scenario_handoff,
    stream_assistant_reply,
)
from core.sidebar import render_setup_blockers, render_setup_sidebar
from core.styles import inject_emoji_fonts, page_heading

st.set_page_config(page_title="AttackGen Assistant", page_icon=":speech_balloon:")
inject_emoji_fonts()
setup = render_setup_sidebar()

page_heading("AttackGen Assistant💬")


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

target = choose_target(scenario)

for label, content in target.panels(scenario):
    with st.expander(label):
        with st.container(height=400, border=True):
            st.markdown(content)

# The conversation belongs to this scenario and this editing target: a newly
# generated scenario starts afresh, and switching mode doesn't feed one
# artifact's chat into the other.
conversation = Conversation(scenario, target)

# One container, not an st.empty placeholder: a placeholder holds a single
# element, so each message would replace the last and only the final turn
# would show after a rerun.
with st.container():
    for message in conversation.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


if not setup.complete:
    render_setup_blockers(setup)

if prompt := st.chat_input("Type your message here...", disabled=not setup.complete):
    history = conversation.chat_history()
    conversation.append("user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        st.write_stream(
            stream_assistant_reply(
                target=target,
                scenario=scenario,
                chat_history=history,
                user_input=prompt,
            )
        )

    conversation.append("assistant", st.session_state.pop(CLEANED_REPLY_KEY, ""))


st.markdown("---")

# The chat only ever refines a conversation; nothing reaches the downloadable
# artifact (or the Assistant's own panels above) until this dedicated call
# rewrites it. Gated on a real user turn so there is always something to apply.
has_user_turn = any(m["role"] == "user" for m in conversation.messages)
apply_clicked = st.button(
    "Apply changes to downloads",
    key=target.apply_key,
    disabled=not has_user_turn,
)
if not has_user_turn:
    st.caption(
        "Ask for at least one change in the chat above before applying it to "
        "the downloads."
    )

if apply_clicked:
    history = conversation.chat_history()
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
            conversation.append("assistant", apply_summary_note(applied))
            st.rerun()

downloads = assistant_downloads(scenario, target)
if downloads:
    for column, download in zip(st.columns(len(downloads)), downloads):
        with column:
            st.download_button(
                label=download.label,
                data=download.data,
                file_name=download.filename,
                mime="text/markdown",
                key=download.key,
            )


def clear_conversation():
    conversation.reset()
    st.rerun()


with st.container():
    if st.button("Clear Conversation", key=CLEAR_BUTTON_KEY):
        clear_conversation()
