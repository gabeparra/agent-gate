"""Base scorer protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agent_gate.models import EvalCase, ProviderResponse, ScoreResult


class Scorer(ABC):
    """Contract for deterministic and model-graded evaluation checks."""

    name: str
    critical: bool = False

    @abstractmethod
    def score(
        self,
        case: EvalCase,
        response: ProviderResponse,
        config: dict[str, Any],
    ) -> ScoreResult:
        """Evaluate a provider response against case-specific check configuration."""
