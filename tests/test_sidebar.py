"""State-contract tests for the shared Setup sidebar."""

from __future__ import annotations

import litellm

from core.sidebar import get_setup_state


def test_persisted_selections_and_environment_credential_are_complete() -> None:
    state = {
        "chosen_model_provider": "Anthropic API",
        "llm_model_name": "claude-sonnet-5",
        "matrix": "ICS",
        "industry": "Energy / Utilities",
        "company_size": "Large (201-1,000 employees)",
    }
    env = {"ANTHROPIC_API_KEY": "env-secret"}

    setup = get_setup_state(state, env)

    assert setup.complete is True
    assert setup.blockers == ()
    assert setup.api_key == "env-secret"


def test_missing_required_fields_report_every_blocker_without_model_call(
    monkeypatch,
) -> None:
    model_calls: list[None] = []

    def fail_if_called(*_args, **_kwargs):
        model_calls.append(None)
        raise AssertionError("setup validation must not invoke a model")

    monkeypatch.setattr(litellm, "completion", fail_if_called)

    setup = get_setup_state({"chosen_model_provider": "OpenAI API"}, {})

    assert setup.complete is False
    assert setup.blockers == (
        "Enter your OpenAI API key in the Setup sidebar.",
        "Select or enter a model in the Setup sidebar.",
        "Select a MITRE framework in the Setup sidebar.",
        "Select your company's industry in the Setup sidebar.",
        "Select your company's size in the Setup sidebar.",
    )
    assert model_calls == []


def test_custom_provider_requires_endpoint_and_model_but_not_api_key() -> None:
    setup = get_setup_state(
        {
            "chosen_model_provider": "Custom",
            "matrix": "Enterprise",
            "industry": "Technology / IT",
            "company_size": "Small (1-50 employees)",
        },
        {},
    )

    assert setup.blockers == (
        "Enter a base URL in the Setup sidebar.",
        "Select or enter a model in the Setup sidebar.",
    )
