"""Tests for the provider/model registry in `core.models`.

Scope: only behaviour that's specific to this project's registry decisions
(fallback shapes, provider references). Tautological lookups that just
re-derive expected values from MODELS/PROVIDERS are intentionally omitted.
"""

from __future__ import annotations

from core.models import (
    MODELS,
    PROVIDERS,
    get_litellm_prefix,
    get_model,
    get_models_for_provider,
    get_provider,
)


def test_every_model_references_a_known_provider() -> None:
    """Adding a model is a one-line edit to MODELS; a typo'd provider_key would
    otherwise route to an empty prefix and misfire only at call time."""
    for m in MODELS:
        assert m.provider_key in PROVIDERS, m.model_id


def test_custom_provider_has_no_static_models() -> None:
    # Custom is free-text: the user types the model name in the sidebar.
    assert get_models_for_provider("Custom") == []


def test_model_provider_pairs_are_unique() -> None:
    """A duplicate (provider, model) pair would be silently shadowed in the
    lookup table built in core.models."""
    pairs = [(m.provider_key, m.model_id) for m in MODELS]
    assert len(pairs) == len(set(pairs))


def test_get_provider_unknown_returns_none() -> None:
    assert get_provider("Nope") is None


def test_get_litellm_prefix_unknown_returns_empty_string() -> None:
    assert get_litellm_prefix("Nope") == ""


def test_get_models_for_provider_unknown_returns_empty() -> None:
    assert get_models_for_provider("Nope") == []


def test_get_model_unknown_returns_none() -> None:
    assert get_model("OpenAI API", "no-such-model") is None
    assert get_model("Nope", "gpt-5.5") is None
