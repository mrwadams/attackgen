"""Tests for `core.response.clean_model_response`."""

from __future__ import annotations

from core.response import clean_model_response


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
