"""Tests for multi-provider adapters using offline httpx.MockTransport."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from agent_gate.models import EvalCase, ToolSpec
from agent_gate.providers import (
    AnthropicProvider,
    MockProvider,
    OpenAICompatibleProvider,
    ProviderError,
    create_provider,
)


def _sample_case() -> EvalCase:
    return EvalCase(
        case_id="provider-test",
        category="general",
        prompt="Read config/app.yaml",
        system_prompt="Be concise.",
        tools=[
            ToolSpec(
                name="read_file",
                description="Read a workspace file",
                input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
            )
        ],
    )


def test_openai_compatible_adapter_parses_text_usage_and_tool_calls() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "message": {
                            "content": "Checking config/app.yaml now.",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": '{"path": "config/app.yaml"}',
                                    },
                                }
                            ],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 110, "completion_tokens": 25},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleProvider(
        provider_name="openai",
        model="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        client=client,
    )
    resp = provider.complete(_sample_case())

    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    assert captured["auth"] == "Bearer test-key"
    assert resp.text == "Checking config/app.yaml now."
    assert resp.usage.prompt_tokens == 110
    assert resp.usage.completion_tokens == 25
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].tool_name == "read_file"
    assert resp.tool_calls[0].arguments == {"path": "config/app.yaml"}


def test_azure_foundry_preset_uses_api_key_header_and_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_AI_API_KEY", "az-secret")
    monkeypatch.setenv("AZURE_AI_ENDPOINT", "https://example.services.ai.azure.com/models")
    monkeypatch.setenv("AZURE_AI_API_VERSION", "2024-05-01-preview")

    provider = create_provider("azure-foundry", model="gpt-4.1-mini")
    assert isinstance(provider, OpenAICompatibleProvider)
    headers = provider._build_headers()
    assert headers["api-key"] == "az-secret"
    assert "Authorization" not in headers
    assert provider.query_params == {"api-version": "2024-05-01-preview"}


def test_anthropic_adapter_parses_text_and_tool_use() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["api_key"] = request.headers.get("x-api-key")
        captured["version"] = request.headers.get("anthropic-version")
        return httpx.Response(
            200,
            json={
                "model": "claude-3-5-sonnet-latest",
                "content": [
                    {"type": "text", "text": "Inspecting workspace file."},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "read_file",
                        "input": {"path": "config/app.yaml"},
                    },
                ],
                "usage": {"input_tokens": 95, "output_tokens": 30},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = AnthropicProvider(
        model="claude-3-5-sonnet-latest",
        api_key="ant-key",
        client=client,
    )
    resp = provider.complete(_sample_case())

    assert captured["api_key"] == "ant-key"
    assert captured["version"] == "2023-06-01"
    assert resp.text == "Inspecting workspace file."
    assert resp.usage.prompt_tokens == 95
    assert resp.usage.completion_tokens == 30
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].tool_name == "read_file"


def test_missing_api_key_raises_provider_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ProviderError, match="OPENAI_API_KEY"):
        create_provider("openai")
    with pytest.raises(ProviderError, match="ANTHROPIC_API_KEY"):
        create_provider("anthropic")


def test_mock_provider_custom_fixture(tmp_path: Path) -> None:
    fixture = tmp_path / "replies.yaml"
    fixture.write_text(
        'custom-1:\n  text: "Answer is 42 ms [doc-1]."\n  latency_ms: 12.5\n',
        encoding="utf-8",
    )
    provider = MockProvider(fixture_path=fixture)
    resp = provider.complete(EvalCase(case_id="custom-1", category="numeric", prompt="Latency?"))
    assert resp.text == "Answer is 42 ms [doc-1]."
    assert resp.latency_ms == 12.5
