"""Shared CSS injected on every Streamlit page.

Streamlit's default font stack for widget labels and sidebar-nav entries does
not include any color emoji font, so on browsers/OSes that don't auto-fall-back
the system emoji font (notably stripped-down Linux containers, some Chromium
builds), emojis render as monochrome silhouettes or empty boxes. Appending the
three common platform emoji fonts is enough for browsers to pick one.
"""

from __future__ import annotations

import streamlit as st

_EMOJI_FONT_CSS = """
<style>
[data-testid="stWidgetLabel"] p,
[data-testid="stSidebarNav"] a span {
  font-family: "Source Sans Pro", -apple-system, BlinkMacSystemFont,
    "Segoe UI", Roboto, "Apple Color Emoji", "Segoe UI Emoji",
    "Noto Color Emoji", sans-serif;
}
</style>
"""


def inject_emoji_fonts() -> None:
    """Append color-emoji fonts to widget-label and sidebar-nav font stacks."""
    st.markdown(_EMOJI_FONT_CSS, unsafe_allow_html=True)
