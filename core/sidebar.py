"""Shared Setup sidebar and its readiness contract.

All Streamlit pages call :func:`render_setup_sidebar`. Non-secret selections are
restored from and mirrored to the URL by ``core.state``; API credentials are
resolved from the environment or kept only in ``st.session_state``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from core.models import PROVIDERS, get_models_for_provider
from core.state import restore_from_query_params, sync_to_query_params

MATRIX_OPTIONS = ["Enterprise", "ICS", "ATLAS"]
INDUSTRIES = sorted(
    [
        "Aerospace / Defense",
        "Agriculture / Food Services",
        "Automotive",
        "Construction",
        "Education",
        "Energy / Utilities",
        "Finance / Banking",
        "Government / Public Sector",
        "Healthcare",
        "Hospitality / Tourism",
        "Insurance",
        "Legal Services",
        "Manufacturing",
        "Media / Entertainment",
        "Non-profit",
        "Real Estate",
        "Retail / E-commerce",
        "Technology / IT",
        "Telecommunication",
        "Transportation / Logistics",
    ]
)
COMPANY_SIZES = [
    "Small (1-50 employees)",
    "Medium (51-200 employees)",
    "Large (201-1,000 employees)",
    "Enterprise (1,001-10,000 employees)",
    "Large Enterprise (10,000+ employees)",
]


@dataclass(frozen=True)
class SetupState:
    """The resolved shared setup and every reason it is not ready."""

    provider: str | None
    api_key: str | None
    api_base: str | None
    model_name: str | None
    matrix: str | None
    industry: str | None
    company_size: str | None
    blockers: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.blockers


def get_setup_state(
    session_state: Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
) -> SetupState:
    """Resolve setup values and report all missing required fields.

    The optional mappings make this contract testable without rendering
    Streamlit widgets. Provider credentials and Custom defaults may come from
    the environment; API keys are read only and are never part of query-param
    persistence.
    """
    state = st.session_state if session_state is None else session_state
    env = os.environ if environ is None else environ

    provider = state.get("chosen_model_provider")
    provider_info = PROVIDERS.get(provider) if provider else None
    api_key = state.get("llm_api_key") or (
        env.get(provider_info.env_var) if provider_info and provider_info.env_var else None
    )
    api_base = state.get("llm_api_base") or (
        env.get("CUSTOM_BASE_URL") if provider_info and provider_info.needs_api_base else None
    )
    model_name = state.get("llm_model_name") or (
        env.get("CUSTOM_MODEL_NAME") if provider == "Custom" else None
    )
    matrix = state.get("matrix")
    industry = state.get("industry")
    company_size = state.get("company_size")

    blockers: list[str] = []
    if provider_info is None:
        blockers.append("Select a model provider in the Setup sidebar.")
    elif provider_info.needs_api_key and not api_key:
        blockers.append(f"Enter your {provider_info.name} API key in the Setup sidebar.")
    if provider_info and provider_info.needs_api_base and not api_base:
        blockers.append("Enter a base URL in the Setup sidebar.")
    if not model_name:
        blockers.append("Select or enter a model in the Setup sidebar.")
    if matrix not in MATRIX_OPTIONS:
        blockers.append("Select a MITRE framework in the Setup sidebar.")
    if not industry:
        blockers.append("Select your company's industry in the Setup sidebar.")
    if not company_size:
        blockers.append("Select your company's size in the Setup sidebar.")

    return SetupState(
        provider=provider,
        api_key=api_key or None,
        api_base=api_base or None,
        model_name=model_name or None,
        matrix=matrix,
        industry=industry,
        company_size=company_size,
        blockers=tuple(blockers),
    )


def render_setup_blockers(setup: SetupState) -> bool:
    """Show all setup blockers together and return whether setup is complete."""
    if setup.complete:
        return True
    st.info(
        "Complete Setup before continuing:\n\n"
        + "\n".join(f"- {item}" for item in setup.blockers)
    )
    return False


def render_setup_sidebar() -> SetupState:
    """Render the shared Setup sidebar and return its resolved readiness state."""
    load_dotenv()
    restore_from_query_params()

    with st.sidebar:
        st.markdown("### <span style='color: #1DB954;'>Setup</span>", unsafe_allow_html=True)

        # These widgets deliberately use the persisted values as their indexes
        # rather than using the shadow session keys as widget keys. The shadows
        # therefore survive page navigation and can also be mirrored to the URL.
        provider_options = list(PROVIDERS.keys())
        persisted_provider = st.session_state.get("chosen_model_provider")
        provider_idx = (
            provider_options.index(persisted_provider)
            if persisted_provider in provider_options
            else 0
        )
        provider = st.selectbox(
            "Select your preferred model provider:",
            provider_options,
            index=provider_idx,
            help=(
                "Select the model provider you would like to use. This will determine "
                "the models available for selection."
            ),
        )
        st.session_state["chosen_model_provider"] = provider
        provider_info = PROVIDERS[provider]

        # A key entered for one provider must not silently become another
        # provider's credential. The owner marker itself is session-only.
        previous_key_provider = st.session_state.get("_llm_api_key_provider")
        if previous_key_provider and previous_key_provider != provider:
            st.session_state.pop("llm_api_key", None)
        st.session_state["_llm_api_key_provider"] = provider

        env_key = os.getenv(provider_info.env_var) if provider_info.env_var else None
        existing_key = env_key or st.session_state.get("llm_api_key") or ""
        if provider_info.needs_api_key and env_key:
            st.session_state["llm_api_key"] = env_key
            st.success("API key loaded from .env")
        else:
            api_key_help = (
                f"You can find your API key at [{provider_info.api_key_url}]"
                f"({provider_info.api_key_url})."
                if provider_info.api_key_url
                else "Enter the API key for your chosen provider."
            )
            if not provider_info.needs_api_key:
                api_key_help = (
                    "Optional. Leave blank if your endpoint doesn't require authentication."
                )
            st.session_state["llm_api_key"] = st.text_input(
                (
                    f"Enter your {provider_info.name} API key:"
                    if provider_info.needs_api_key
                    else "API key (optional):"
                ),
                type="password",
                value=existing_key,
                help=api_key_help,
            )

        if provider_info.needs_api_base:
            initial_base = (
                st.session_state.get("llm_api_base")
                or os.getenv("CUSTOM_BASE_URL")
                or provider_info.default_api_base
                or ""
            )
            st.session_state["llm_api_base"] = st.text_input(
                "Base URL:",
                value=initial_base,
                help=(
                    "Base URL of your OpenAI-compatible endpoint. Example: "
                    "http://localhost:11434/v1 for Ollama, "
                    "http://localhost:1234/v1 for LM Studio."
                ),
            )
        else:
            st.session_state["llm_api_base"] = None

        models = get_models_for_provider(provider)
        persisted_model = st.session_state.get("llm_model_name")
        if models:
            labels = [model.model_id for model in models]
            help_map = {model.model_id: model.help_text for model in models}
            model_idx = labels.index(persisted_model) if persisted_model in labels else 0
            st.session_state["llm_model_name"] = st.selectbox(
                "Select the model you would like to use:",
                labels,
                index=model_idx,
                help="\n".join(
                    f"**{model_id}** — {help_map[model_id] or 'No description.'}"
                    for model_id in labels
                ),
            )
        else:
            initial_model = persisted_model or os.getenv("CUSTOM_MODEL_NAME") or ""
            st.session_state["llm_model_name"] = st.text_input(
                "Model name:",
                value=initial_model,
                help=(
                    "Model identifier as expected by your endpoint "
                    "(e.g. 'llama3.1', 'qwen3:32b')."
                ),
            )

        st.markdown("---")

        persisted_matrix = st.session_state.get("matrix")
        matrix_idx = (
            MATRIX_OPTIONS.index(persisted_matrix)
            if persisted_matrix in MATRIX_OPTIONS
            else 0
        )
        st.session_state["matrix"] = st.radio(
            "Select MITRE Framework:",
            MATRIX_OPTIONS,
            index=matrix_idx,
            help=(
                "Enterprise and ICS are ATT&CK matrices for traditional IT and industrial "
                "control systems. ATLAS focuses on adversarial threats to AI/ML systems."
            ),
        )

        persisted_industry = st.session_state.get("industry")
        st.session_state["industry"] = st.selectbox(
            "Select your company's industry:",
            INDUSTRIES,
            index=(
                INDUSTRIES.index(persisted_industry)
                if persisted_industry in INDUSTRIES
                else None
            ),
            placeholder="Select Industry",
        )

        persisted_size = st.session_state.get("company_size")
        st.session_state["company_size"] = st.selectbox(
            "Select your company's size:",
            COMPANY_SIZES,
            index=(
                COMPANY_SIZES.index(persisted_size)
                if persisted_size in COMPANY_SIZES
                else None
            ),
            placeholder="Select Company Size",
        )

        # The allow-list in core.state intentionally excludes llm_api_key.
        sync_to_query_params()

        st.markdown("---")
        st.markdown("### <span style='color: #1DB954;'>About</span>", unsafe_allow_html=True)
        st.markdown("Created by [Matt Adams](https://www.linkedin.com/in/matthewrwadams)")
        st.markdown(
            "⭐ Star on GitHub: [![Star on GitHub]"
            "(https://img.shields.io/github/stars/mrwadams/attackgen?style=social)]"
            "(https://github.com/mrwadams/attackgen)"
        )

    return get_setup_state()
