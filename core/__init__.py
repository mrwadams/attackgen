"""Unified LLM wrapper for AttackGen.

All provider integration lives in `core/llm.py`. Never call `litellm.completion`
directly from outside this package — go through `call_llm`.
"""

from core.llm import call_llm
from core.models import PROVIDERS, MODELS, ModelInfo, ProviderInfo, get_models_for_provider
from core.schemas import LLMConfig

__all__ = [
    "call_llm",
    "LLMConfig",
    "PROVIDERS",
    "MODELS",
    "ModelInfo",
    "ProviderInfo",
    "get_models_for_provider",
]
