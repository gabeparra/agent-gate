"""Anthropic Messages API provider adapter."""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

from agent_gate.models import EvalCase, ProviderResponse, TokenUsage, ToolCall
from agent_gate.providers.base import (
    Provider,
    ProviderError,
    format_case_messages,
    tools_to_anthropic_schema,
)


class AnthropicProvider(Provider):
    """Adapter for Anthropic Messages API (https://api.anthropic.com/v1/messages)."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str = "https://api.anthropic.com/v1",
        max_tokens: int = 1024,
        temperature: float = 0.0,
        timeout_seconds: float = 60.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.provider_name = "anthropic"
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self._client = client

    def _endpoint_url(self) -> str:
        if self.base_url.endswith("/messages"):
            return self.base_url
        return f"{self.base_url}/messages"

    def complete(self, case: EvalCase) -> ProviderResponse:
        system, user_text = format_case_messages(case)
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": user_text}],
        }
        if system:
            payload["system"] = system

        tools = tools_to_anthropic_schema(case)
        if tools:
            payload["tools"] = tools

        start = time.perf_counter()
        try:
            if self._client is not None:
                response = self._client.post(
                    self._endpoint_url(),
                    headers=headers,
                    json=payload,
                    timeout=self.timeout_seconds,
                )
            else:
                with httpx.Client(timeout=self.timeout_seconds) as client:
                    response = client.post(
                        self._endpoint_url(),
                        headers=headers,
                        json=payload,
                    )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"anthropic request failed for model '{self.model}': {exc}"
            ) from exc
        except ValueError as exc:
            raise ProviderError(f"anthropic returned invalid JSON: {exc}") from exc

        latency_ms = round((time.perf_counter() - start) * 1000.0, 2)
        return self._parse_response(data, latency_ms)

    def _parse_response(self, data: dict[str, Any], latency_ms: float) -> ProviderResponse:
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in data.get("content") or []:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                text_parts.append(str(block.get("text", "")))
            elif block_type == "tool_use":
                args = block.get("input")
                tool_calls.append(
                    ToolCall(
                        tool_name=str(block.get("name", "")),
                        arguments=args if isinstance(args, dict) else {},
                        call_id=str(block.get("id", "")),
                    )
                )

        raw_usage = data.get("usage") or {}
        usage = TokenUsage(
            prompt_tokens=int(raw_usage.get("input_tokens", 0)),
            completion_tokens=int(raw_usage.get("output_tokens", 0)),
        )
        return ProviderResponse(
            text="\n".join(part for part in text_parts if part).strip(),
            model=str(data.get("model") or self.model),
            provider=self.provider_name,
            latency_ms=latency_ms,
            usage=usage,
            tool_calls=tool_calls,
            raw=data,
        )

    @classmethod
    def from_env(
        cls,
        *,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        timeout_seconds: float = 60.0,
    ) -> AnthropicProvider:
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ProviderError("ANTHROPIC_API_KEY is not set.")
        return cls(
            model=model or os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
            api_key=key,
            base_url=base_url or os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1"),
            temperature=temperature,
            timeout_seconds=timeout_seconds,
        )
