"""OpenAI-compatible provider adapter.

Covers OpenAI, Azure AI Foundry, OpenRouter, Ollama, and llama.cpp behind one
clean HTTP client with zero SDK lock-in.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx

from agent_gate.models import EvalCase, ProviderResponse, TokenUsage, ToolCall
from agent_gate.providers.base import (
    Provider,
    ProviderError,
    format_case_messages,
    tools_to_openai_schema,
)


class OpenAICompatibleProvider(Provider):
    """Adapter for OpenAI Chat Completions API and compatible endpoints."""

    def __init__(
        self,
        *,
        provider_name: str,
        model: str,
        base_url: str,
        api_key: str | None = None,
        api_key_header: str = "Authorization",
        extra_headers: dict[str, str] | None = None,
        query_params: dict[str, str] | None = None,
        temperature: float = 0.0,
        timeout_seconds: float = 60.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.provider_name = provider_name
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.api_key_header = api_key_header
        self.extra_headers = dict(extra_headers or {})
        self.query_params = dict(query_params or {})
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self._client = client

    def _build_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self.extra_headers}
        if self.api_key:
            if self.api_key_header.lower() == "authorization":
                headers["Authorization"] = f"Bearer {self.api_key}"
            else:
                headers[self.api_key_header] = self.api_key
        return headers

    def _endpoint_url(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    def complete(self, case: EvalCase) -> ProviderResponse:
        system, user_text = format_case_messages(case)
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_text})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        tools = tools_to_openai_schema(case)
        if tools:
            payload["tools"] = tools

        start = time.perf_counter()
        try:
            if self._client is not None:
                response = self._client.post(
                    self._endpoint_url(),
                    headers=self._build_headers(),
                    params=self.query_params or None,
                    json=payload,
                    timeout=self.timeout_seconds,
                )
            else:
                with httpx.Client(timeout=self.timeout_seconds) as client:
                    response = client.post(
                        self._endpoint_url(),
                        headers=self._build_headers(),
                        params=self.query_params or None,
                        json=payload,
                    )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"{self.provider_name} request failed for model '{self.model}': {exc}"
            ) from exc
        except ValueError as exc:
            raise ProviderError(f"{self.provider_name} returned invalid JSON: {exc}") from exc

        latency_ms = round((time.perf_counter() - start) * 1000.0, 2)
        return self._parse_response(data, latency_ms)

    def _parse_response(self, data: dict[str, Any], latency_ms: float) -> ProviderResponse:
        choices = data.get("choices") or []
        if not choices:
            raise ProviderError(f"{self.provider_name} response contained no choices")

        message = choices[0].get("message") or {}
        text = message.get("content") or ""
        if isinstance(text, list):
            # Some OpenAI-compatible servers return structured text parts.
            text = "".join(
                str(part.get("text", "")) for part in text if isinstance(part, dict)
            )

        tool_calls: list[ToolCall] = []
        for raw_tc in message.get("tool_calls") or []:
            fn = raw_tc.get("function") or {}
            raw_args = fn.get("arguments", "{}")
            if isinstance(raw_args, str):
                try:
                    parsed_args = json.loads(raw_args) if raw_args.strip() else {}
                except json.JSONDecodeError:
                    parsed_args = {"_raw_arguments": raw_args}
            elif isinstance(raw_args, dict):
                parsed_args = raw_args
            else:
                parsed_args = {}
            tool_calls.append(
                ToolCall(
                    tool_name=str(fn.get("name", "")),
                    arguments=parsed_args,
                    call_id=str(raw_tc.get("id", "")),
                )
            )

        raw_usage = data.get("usage") or {}
        usage = TokenUsage(
            prompt_tokens=int(raw_usage.get("prompt_tokens", 0)),
            completion_tokens=int(raw_usage.get("completion_tokens", 0)),
        )
        return ProviderResponse(
            text=str(text),
            model=str(data.get("model") or self.model),
            provider=self.provider_name,
            latency_ms=latency_ms,
            usage=usage,
            tool_calls=tool_calls,
            raw=data,
        )

    @classmethod
    def from_preset(
        cls,
        provider_name: str,
        *,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        timeout_seconds: float = 60.0,
    ) -> OpenAICompatibleProvider:
        normalized = provider_name.lower().strip()
        if normalized == "openai":
            key = os.getenv("OPENAI_API_KEY")
            if not key:
                raise ProviderError("OPENAI_API_KEY is not set.")
            return cls(
                provider_name="openai",
                model=model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                base_url=base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                api_key=key,
                temperature=temperature,
                timeout_seconds=timeout_seconds,
            )

        if normalized in {"azure", "azure-foundry"}:
            key = os.getenv("AZURE_AI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
            endpoint = base_url or os.getenv("AZURE_AI_ENDPOINT") or os.getenv("AZURE_OPENAI_ENDPOINT")
            if not key or not endpoint:
                raise ProviderError(
                    "Azure AI Foundry requires AZURE_AI_API_KEY and AZURE_AI_ENDPOINT (or --base-url)."
                )
            api_version = os.getenv("AZURE_AI_API_VERSION")
            query = {"api-version": api_version} if api_version else None
            return cls(
                provider_name="azure-foundry",
                model=model or os.getenv("AZURE_AI_MODEL", "gpt-4o-mini"),
                base_url=endpoint,
                api_key=key,
                api_key_header="api-key",
                query_params=query,
                temperature=temperature,
                timeout_seconds=timeout_seconds,
            )

        if normalized == "openrouter":
            key = os.getenv("OPENROUTER_API_KEY")
            if not key:
                raise ProviderError("OPENROUTER_API_KEY is not set.")
            headers = {
                "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "https://github.com/gabeparra/agent-gate"),
                "X-Title": os.getenv("OPENROUTER_APP_NAME", "agent-gate"),
            }
            return cls(
                provider_name="openrouter",
                model=model or os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
                base_url=base_url or "https://openrouter.ai/api/v1",
                api_key=key,
                extra_headers=headers,
                temperature=temperature,
                timeout_seconds=timeout_seconds,
            )

        if normalized == "ollama":
            return cls(
                provider_name="ollama",
                model=model or os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
                base_url=base_url or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1"),
                api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
                temperature=temperature,
                timeout_seconds=timeout_seconds,
            )

        if normalized in {"llamacpp", "llama.cpp", "local"}:
            return cls(
                provider_name="llamacpp",
                model=model or os.getenv("LLAMACPP_MODEL", "local-model"),
                base_url=base_url or os.getenv("LLAMACPP_BASE_URL", "http://127.0.0.1:8080/v1"),
                api_key=os.getenv("LLAMACPP_API_KEY"),
                temperature=temperature,
                timeout_seconds=timeout_seconds,
            )

        raise ProviderError(f"Unsupported OpenAI-compatible preset: {provider_name}")
