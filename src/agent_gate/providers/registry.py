"""Factory for constructing provider adapters from CLI flags or config."""

from __future__ import annotations

from pathlib import Path

from agent_gate.providers.anthropic import AnthropicProvider
from agent_gate.providers.base import Provider, ProviderError
from agent_gate.providers.mock import MockProvider
from agent_gate.providers.openai_compat import OpenAICompatibleProvider

SUPPORTED_PROVIDERS = (
    "mock",
    "openai",
    "azure-foundry",
    "azure",
    "anthropic",
    "openrouter",
    "ollama",
    "llamacpp",
)


def create_provider(
    provider_name: str,
    *,
    model: str | None = None,
    base_url: str | None = None,
    profile: str = "baseline",
    mock_fixture: str | Path | None = None,
    temperature: float = 0.0,
    timeout_seconds: float = 60.0,
) -> Provider:
    """Instantiate a configured Provider adapter by name."""
    normalized = provider_name.lower().strip()
    if normalized == "mock":
        return MockProvider(
            model=model or f"mock-{profile}",
            profile=profile,
            fixture_path=mock_fixture,
        )
    if normalized == "anthropic":
        return AnthropicProvider.from_env(
            model=model,
            base_url=base_url,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
        )
    if normalized in {
        "openai",
        "azure",
        "azure-foundry",
        "openrouter",
        "ollama",
        "llamacpp",
        "llama.cpp",
        "local",
    }:
        return OpenAICompatibleProvider.from_preset(
            normalized,
            model=model,
            base_url=base_url,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
        )
    raise ProviderError(
        f"Unknown provider '{provider_name}'. Supported providers: {', '.join(SUPPORTED_PROVIDERS)}."
    )
