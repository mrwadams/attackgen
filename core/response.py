"""Cleaning helpers for raw LLM responses."""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator

_OPEN_TAG = "<think>"
_CLOSE_TAG = "</think>"
_MAX_PARTIAL = max(len(_OPEN_TAG), len(_CLOSE_TAG)) - 1


def stream_filter_thinking(deltas: Iterable[str]) -> Iterator[str]:
    """Yield text deltas with ``<think>...</think>`` regions suppressed.

    Holds back up to ``len("</think>")-1`` trailing characters at any time so
    that a tag prefix spanning two chunks doesn't leak before we can decide
    whether it's the start of a tag. The caller is responsible for capturing
    the raw stream separately if it needs the unfiltered text (e.g. to run
    ``clean_model_response`` for the final cleaned + thinking pair).
    """
    buf = ""
    inside = False
    for delta in deltas:
        buf += delta
        while True:
            if inside:
                close_at = buf.find(_CLOSE_TAG)
                if close_at == -1:
                    # Keep enough tail to recognise a future "</think>".
                    if len(buf) > _MAX_PARTIAL:
                        buf = buf[-_MAX_PARTIAL:]
                    break
                buf = buf[close_at + len(_CLOSE_TAG):]
                inside = False
                continue
            open_at = buf.find(_OPEN_TAG)
            if open_at == -1:
                # Yield everything except a possible partial-tag tail.
                if len(buf) > _MAX_PARTIAL:
                    yield buf[:-_MAX_PARTIAL]
                    buf = buf[-_MAX_PARTIAL:]
                break
            if open_at > 0:
                yield buf[:open_at]
            buf = buf[open_at + len(_OPEN_TAG):]
            inside = True
    if not inside and buf:
        yield buf


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
