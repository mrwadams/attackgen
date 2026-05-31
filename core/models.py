"""Unified model registry — single source of truth for providers and models.

To add or update a model, edit the MODELS list below. Nothing else needs to change.
Mirrored from mrwadams/stride-gpt:stride_gpt/models.py (reviewed 2026-05-30).

Provider model listing pages:
  - Anthropic: https://docs.anthropic.com/en/docs/about-claude/models
  - OpenAI:    https://platform.openai.com/docs/models
  - Google AI: https://ai.google.dev/gemini-api/docs/models
  - Mistral:   https://docs.mistral.ai/getting-started/models
  - Groq:      https://console.groq.com/docs/models
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderInfo:
    name: str                          # Display name shown in the sidebar
    provider_key: str                  # Routing key used in LLMConfig.provider
    litellm_prefix: str                # Prefix prepended to model_name for LiteLLM
    env_var: str | None = None         # .env variable that supplies the API key
    needs_api_key: bool = True
    needs_api_base: bool = False
    default_api_base: str | None = None
    api_key_url: str = ""              # Where the user can generate a key


@dataclass(frozen=True)
class ModelInfo:
    model_id: str
    provider_key: str
    uses_max_completion_tokens: bool = False   # OpenAI reasoning models (gpt-5.x)
    supports_thinking: bool = False            # Claude/Gemini extended thinking
    help_text: str = ""


# ---------------------------------------------------------------------------
# Provider registry — order determines sidebar display order
# ---------------------------------------------------------------------------

PROVIDERS: dict[str, ProviderInfo] = {
    "OpenAI API": ProviderInfo(
        name="OpenAI",
        provider_key="OpenAI API",
        litellm_prefix="",
        env_var="OPENAI_API_KEY",
        api_key_url="https://platform.openai.com/account/api-keys",
    ),
    "Anthropic API": ProviderInfo(
        name="Anthropic",
        provider_key="Anthropic API",
        litellm_prefix="anthropic/",
        env_var="ANTHROPIC_API_KEY",
        api_key_url="https://console.anthropic.com/settings/keys",
    ),
    "Google AI API": ProviderInfo(
        name="Google AI",
        provider_key="Google AI API",
        litellm_prefix="gemini/",
        env_var="GOOGLE_API_KEY",
        api_key_url="https://makersuite.google.com/app/apikey",
    ),
    "Mistral API": ProviderInfo(
        name="Mistral",
        provider_key="Mistral API",
        litellm_prefix="mistral/",
        env_var="MISTRAL_API_KEY",
        api_key_url="https://console.mistral.ai/api-keys/",
    ),
    "Groq API": ProviderInfo(
        name="Groq",
        provider_key="Groq API",
        litellm_prefix="groq/",
        env_var="GROQ_API_KEY",
        api_key_url="https://console.groq.com/keys",
    ),
    "Custom": ProviderInfo(
        name="Custom (OpenAI-compatible)",
        provider_key="Custom",
        litellm_prefix="openai/",
        env_var="CUSTOM_API_KEY",
        needs_api_key=False,           # Many local endpoints don't need a key
        needs_api_base=True,
        default_api_base="http://127.0.0.1:1234/v1",
    ),
}


# ---------------------------------------------------------------------------
# Model registry — order determines UI display order per provider
# ---------------------------------------------------------------------------

MODELS: list[ModelInfo] = [
    # --- OpenAI ---
    ModelInfo(
        model_id="gpt-5.5",
        provider_key="OpenAI API",
        uses_max_completion_tokens=True,
        help_text="GPT-5.5 is OpenAI's latest flagship model.",
    ),
    ModelInfo(
        model_id="gpt-5.4",
        provider_key="OpenAI API",
        uses_max_completion_tokens=True,
        help_text="GPT-5.4 is OpenAI's previous flagship model with 1M+ context.",
    ),
    ModelInfo(
        model_id="gpt-5.4-pro",
        provider_key="OpenAI API",
        uses_max_completion_tokens=True,
        help_text="GPT-5.4 Pro produces smarter, more precise responses.",
    ),
    ModelInfo(
        model_id="gpt-5.4-mini",
        provider_key="OpenAI API",
        uses_max_completion_tokens=True,
        help_text="GPT-5.4 Mini is a fast, cost-efficient version.",
    ),
    ModelInfo(
        model_id="gpt-5.4-nano",
        provider_key="OpenAI API",
        uses_max_completion_tokens=True,
        help_text="GPT-5.4 Nano is the fastest and most affordable option.",
    ),
    # --- Anthropic ---
    ModelInfo(
        model_id="claude-sonnet-4-6",
        provider_key="Anthropic API",
        supports_thinking=True,
        help_text="Claude Sonnet 4.6 offers the best balance of performance and efficiency.",
    ),
    ModelInfo(
        model_id="claude-opus-4-8",
        provider_key="Anthropic API",
        supports_thinking=True,
        help_text="Claude Opus 4.8 is the most capable Claude model.",
    ),
    ModelInfo(
        model_id="claude-opus-4-7",
        provider_key="Anthropic API",
        supports_thinking=True,
        help_text="Claude Opus 4.7 is the previous-generation Opus.",
    ),
    ModelInfo(
        model_id="claude-haiku-4-5-20251001",
        provider_key="Anthropic API",
        help_text="Claude Haiku 4.5 is the fastest and most cost-effective Claude model.",
    ),
    # --- Google AI ---
    ModelInfo(
        model_id="gemini-3.1-pro-preview",
        provider_key="Google AI API",
        supports_thinking=True,
        help_text="Gemini 3.1 Pro is Google's most capable model with 1M context.",
    ),
    ModelInfo(
        model_id="gemini-3.5-flash",
        provider_key="Google AI API",
        supports_thinking=True,
        help_text="Gemini 3.5 Flash is Google's latest fast model with 1M context.",
    ),
    ModelInfo(
        model_id="gemini-3.1-flash-lite",
        provider_key="Google AI API",
        supports_thinking=True,
        help_text="Gemini 3.1 Flash Lite is the most cost-efficient option with 1M context.",
    ),
    ModelInfo(
        model_id="gemini-3-flash-preview",
        provider_key="Google AI API",
        supports_thinking=True,
        help_text="Gemini 3 Flash is optimized for speed with 1M context.",
    ),
    # --- Mistral ---
    ModelInfo(
        model_id="mistral-large-2512",
        provider_key="Mistral API",
        help_text="Mistral Large 3 offers premium capabilities.",
    ),
    ModelInfo(
        model_id="mistral-small-2603",
        provider_key="Mistral API",
        help_text="Mistral Small 4 merges reasoning, vision, and coding in a 256k-context model.",
    ),
    ModelInfo(
        model_id="mistral-medium-3-5",
        provider_key="Mistral API",
        help_text="Mistral Medium 3.5 provides balanced performance.",
    ),
    ModelInfo(
        model_id="mistral-medium-2508",
        provider_key="Mistral API",
        help_text="Mistral Medium 3.1 provides balanced performance.",
    ),
    ModelInfo(
        model_id="magistral-medium-2509",
        provider_key="Mistral API",
        help_text="Magistral Medium is a reasoning-focused model.",
    ),
    # --- Groq ---
    ModelInfo(
        model_id="openai/gpt-oss-120b",
        provider_key="Groq API",
        help_text="GPT-OSS 120B is an open-source reasoning model on Groq.",
    ),
    ModelInfo(
        model_id="openai/gpt-oss-20b",
        provider_key="Groq API",
        help_text="GPT-OSS 20B is a fast open-source model on Groq.",
    ),
    ModelInfo(
        model_id="llama-3.3-70b-versatile",
        provider_key="Groq API",
        help_text="Llama 3.3 70B excels at general-purpose tasks.",
    ),
    ModelInfo(
        model_id="qwen/qwen3-32b",
        provider_key="Groq API",
        help_text="Qwen3 32B delivers balanced performance.",
    ),
    # --- Custom: no static list; the user types the model name in the sidebar ---
]


_MODEL_LOOKUP: dict[tuple[str, str], ModelInfo] = {
    (m.provider_key, m.model_id): m for m in MODELS
}


def get_models_for_provider(provider_key: str) -> list[ModelInfo]:
    return [m for m in MODELS if m.provider_key == provider_key]


def get_model(provider_key: str, model_id: str) -> ModelInfo | None:
    return _MODEL_LOOKUP.get((provider_key, model_id))


def get_provider(provider_key: str) -> ProviderInfo | None:
    return PROVIDERS.get(provider_key)


def get_litellm_prefix(provider_key: str) -> str:
    provider = PROVIDERS.get(provider_key)
    return provider.litellm_prefix if provider else ""


def model_uses_completion_tokens(provider_key: str, model_id: str) -> bool:
    model = get_model(provider_key, model_id)
    return bool(model and model.uses_max_completion_tokens)
