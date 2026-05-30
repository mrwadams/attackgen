"""Sidebar-state persistence via URL query parameters.

Streamlit's `st.session_state` is per-session and gets reset on browser refresh
or when the user lands on a page directly. To keep the sidebar selections
between refreshes (without storing API keys in the URL) we mirror a small set
of keys into the URL via `st.query_params`.

Every page calls `restore_from_query_params()` near the top so that a deep link
to e.g. page 3 also restores the provider/model/industry the user had selected.
The Welcome sidebar additionally calls `sync_to_query_params()` after rendering
its widgets so the URL stays in sync as the user changes selections.

API keys are intentionally excluded — the URL would expose them in browser
history, server logs, and any link the user shares.
"""

from __future__ import annotations

import streamlit as st

from core.models import PROVIDERS

# Mapping: session_state key → short query-param name
PERSISTED: dict[str, str] = {
    "chosen_model_provider": "p",
    "llm_model_name": "m",
    "llm_api_base": "b",
    "selected_matrix": "x",
    "industry": "i",
    "company_size": "s",
}

_QP_KEYS = set(PERSISTED.values())
_VALID_MATRICES = {"Enterprise", "ICS", "ATLAS"}


def restore_from_query_params() -> None:
    """Seed `st.session_state` from the URL on first script run.

    Safe to call multiple times — only sets keys that aren't already in
    session_state, so user-driven widget changes always win over the URL.
    """
    qp = st.query_params
    for ss_key, qp_key in PERSISTED.items():
        if ss_key in st.session_state:
            continue
        if qp_key not in qp:
            continue
        value = qp[qp_key]
        # Drop values that no longer correspond to a known provider/matrix —
        # otherwise a stale URL could put us in an invalid state.
        if ss_key == "chosen_model_provider" and value not in PROVIDERS:
            continue
        if ss_key == "selected_matrix" and value not in _VALID_MATRICES:
            continue
        st.session_state[ss_key] = value


def sync_to_query_params() -> None:
    """Mirror the current session_state into the URL.

    Only rewrites if the managed keys have actually changed — avoids gratuitous
    URL churn and any chance of a rerun loop.
    """
    desired = {
        qp_key: str(st.session_state[ss_key])
        for ss_key, qp_key in PERSISTED.items()
        if st.session_state.get(ss_key)
    }
    current = {k: v for k, v in st.query_params.items() if k in _QP_KEYS}
    if current == desired:
        return

    for k in list(st.query_params.keys()):
        if k in _QP_KEYS:
            del st.query_params[k]
    for k, v in desired.items():
        st.query_params[k] = v
