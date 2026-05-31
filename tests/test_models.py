"""Tests for the provider/model registry in `core.models`.

Scope: only behaviour that's specific to this project's registry decisions
(fallback shapes, the OpenAI gpt-5 invariant). Tautological lookups that just
re-derive expected values from MODELS/PROVIDERS are intentionally omitted.
"""

from __future__ import annotations

from core.models import (
    MODELS,
    get_litellm_prefix,
    get_model,
    get_models_for_provider,
    get_provider,
    model_uses_completion_tokens,
)


def test_get_provider_unknown_returns_none() -> None:
    assert get_provider("Nope") is None


def test_get_litellm_prefix_unknown_returns_empty_string() -> None:
    assert get_litellm_prefix("Nope") == ""


def test_get_models_for_provider_unknown_returns_empty() -> None:
    assert get_models_for_provider("Nope") == []


def test_get_model_unknown_returns_none() -> None:
    assert get_model("OpenAI API", "no-such-model") is None
    assert get_model("Nope", "gpt-5.5") is None


def test_model_uses_completion_tokens_true_only_for_openai_gpt5() -> None:
    """Today only OpenAI gpt-5.x entries should set the flag. Adding a new
    model that flips this requires updating the assertion, which is the point —
    it's an intentional choice, not an accident."""
    for m in MODELS:
        if m.uses_max_completion_tokens:
            assert m.provider_key == "OpenAI API"
            assert m.model_id.startswith("gpt-5.")


def test_model_uses_completion_tokens_unknown_model_is_false() -> None:
    assert model_uses_completion_tokens("OpenAI API", "no-such-model") is False
