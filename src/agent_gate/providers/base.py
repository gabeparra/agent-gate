"""Base provider interface and prompt formatting helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agent_gate.models import EvalCase, ProviderResponse


class ProviderError(RuntimeError):
    """Raised when a model provider request fails or is misconfigured."""


class Provider(ABC):
    """Unified interface for all model providers."""

    provider_name: str
    model: str

    @abstractmethod
    def complete(self, case: EvalCase) -> ProviderResponse:
        """Run one evaluation case and return a normalized response."""


def format_case_messages(case: EvalCase) -> tuple[str, str]:
    """Build (system_prompt, user_message) including grounded context documents."""
    system = case.system_prompt.strip()
    sections: list[str] = []

    if case.context:
        doc_blocks = []
        for doc in case.context:
            doc_blocks.append(f"[{doc.doc_id}] {doc.title}\n{doc.content.strip()}")
        sections.append("Reference Context:\n" + "\n\n".join(doc_blocks))

    sections.append(case.prompt.strip())
    return system, "\n\n".join(sections)


def tools_to_openai_schema(case: EvalCase) -> list[dict[str, Any]]:
    """Convert EvalCase ToolSpecs to OpenAI function-calling schema."""
    converted: list[dict[str, Any]] = []
    for tool in case.tools:
        if not tool.allowed:
            continue
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema
                    or {"type": "object", "properties": {}, "additionalProperties": True},
                },
            }
        )
    return converted


def tools_to_anthropic_schema(case: EvalCase) -> list[dict[str, Any]]:
    """Convert EvalCase ToolSpecs to Anthropic tool_use schema."""
    converted: list[dict[str, Any]] = []
    for tool in case.tools:
        if not tool.allowed:
            continue
        converted.append(
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema
                or {"type": "object", "properties": {}, "additionalProperties": True},
            }
        )
    return converted
