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


def page_heading(title: str) -> None:
    """Render a page's title in the app's green."""
    st.markdown(f"# <span style='color: #1DB954;'>{title}</span>", unsafe_allow_html=True)


def back_link() -> None:
    """Render the link back to the Welcome page."""
    st.markdown(
        '<a href="/" style="display: inline-block; padding: 5px 20px; color: white; text-align: center; text-decoration: none; font-size: 16px; border-radius: 4px;">⬅️ Back</a>',
        unsafe_allow_html=True,
    )
