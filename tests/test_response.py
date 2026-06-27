"""Tests for `core.response.clean_model_response` and `stream_filter_thinking`."""

from __future__ import annotations

from core.response import clean_model_response, stream_filter_thinking


def _filter(*deltas: str) -> str:
    """Run the streaming filter over the given deltas and join the output."""
    return "".join(stream_filter_thinking(deltas))


def test_returns_text_unchanged_when_no_think_or_fences() -> None:
    text = "# Scenario\n\nA tabletop exercise."
    thinking, cleaned = clean_model_response(text)
    assert thinking is None
    assert cleaned == text


def test_extracts_single_think_block() -> None:
    text = "<think>plan the response</think>\n\n# Scenario\n\nBody."
    thinking, cleaned = clean_model_response(text)
    assert thinking == "plan the response"
    assert cleaned == "# Scenario\n\nBody."


def test_strips_multiline_think_block() -> None:
    text = "<think>\nstep 1\nstep 2\n</think>\nfinal answer"
    thinking, cleaned = clean_model_response(text)
    assert thinking == "step 1\nstep 2"
    assert cleaned == "final answer"


def test_strips_outer_markdown_code_fence() -> None:
    text = "```markdown\n# Scenario\n\nBody.\n```"
    thinking, cleaned = clean_model_response(text)
    assert thinking is None
    assert "```" not in cleaned
    assert cleaned.startswith("# Scenario")


def test_strips_bare_code_fence() -> None:
    text = "```\n# Scenario\n```"
    _, cleaned = clean_model_response(text)
    assert "```" not in cleaned


def test_handles_think_block_and_fences_together() -> None:
    text = "<think>reasoning</think>\n```markdown\n# Title\n```"
    thinking, cleaned = clean_model_response(text)
    assert thinking == "reasoning"
    assert "```" not in cleaned
    assert cleaned.startswith("# Title")


def test_returns_empty_cleaned_for_only_think_block() -> None:
    text = "<think>only thinking, no answer</think>"
    thinking, cleaned = clean_model_response(text)
    assert thinking == "only thinking, no answer"
    assert cleaned == ""


# ---------------------------------------------------------------------------
# stream_filter_thinking — the streaming counterpart that suppresses
# <think>...</think> regions across chunk boundaries.
# ---------------------------------------------------------------------------


def test_stream_passes_plain_text_unchanged() -> None:
    assert _filter("# Scenario\n\nA tabletop exercise.") == "# Scenario\n\nA tabletop exercise."


def test_stream_suppresses_think_block_with_surrounding_content() -> None:
    assert _filter("before<think>secret</think>after") == "beforeafter"


def test_stream_suppresses_tag_split_across_chunks() -> None:
    # The opening tag straddles two deltas; nothing inside it may leak.
    assert _filter("<thi", "nk>secret</think>visible") == "visible"


def test_stream_suppresses_close_tag_split_across_chunks() -> None:
    assert _filter("<think>secret</thi", "nk>visible") == "visible"


def test_stream_suppresses_multiple_think_blocks() -> None:
    assert _filter("<think>a</think>mid<think>b</think>end") == "midend"


def test_stream_does_not_swallow_stray_lessthan() -> None:
    # A literal "<" that never becomes a <think> tag must survive, even when
    # the buffer is long enough to trigger the partial-tag tail retention.
    text = "this is a < sign embedded in a reasonably long line of output"
    assert _filter(text) == text


def test_stream_drops_unterminated_think_block() -> None:
    # An opened-but-never-closed block emits nothing after the open tag.
    assert _filter("visible<think>secret never closed") == "visible"
