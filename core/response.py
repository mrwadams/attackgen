"""Cleaning helpers for raw LLM responses."""

from __future__ import annotations

import re


def clean_model_response(text: str) -> tuple[str | None, str]:
    """Pull out ``<think>...</think>`` reasoning and strip surrounding code fences.

    Some reasoning models wrap their chain-of-thought in ``<think>`` tags; we
    surface that separately so pages can show it in an expander while keeping
    the main scenario clean. The trailing fence strip handles models that wrap
    a Markdown scenario in a single outer ```` ```markdown ```` block.

    Returns ``(thinking_or_None, cleaned)``.
    """
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    thinking = match.group(1).strip() if match else None
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    cleaned = re.sub(r"^```\w*\n|```$", "", cleaned, flags=re.MULTILINE).strip()
    return thinking, cleaned
