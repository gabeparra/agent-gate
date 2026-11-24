"""Tests for token cost calculation, pricing overrides, and latency percentiles."""

from __future__ import annotations

import json
from pathlib import Path

from agent_gate.metrics import estimate_cost_usd, load_pricing_table, percentile
from agent_gate.models import TokenUsage


def test_estimate_cost_and_pricing_override(tmp_path: Path) -> None:
    usage = TokenUsage(prompt_tokens=1_000_000, completion_tokens=500_000)
    # gpt-4o-mini default: $0.15 / 1M in, $0.60 / 1M out -> 0.15 + 0.30 = $0.45
    assert estimate_cost_usd("gpt-4o-mini", usage) == 0.45
    # Provider prefix fallback works (e.g. openrouter/gpt-4o-mini -> gpt-4o-mini)
    assert estimate_cost_usd("openrouter/gpt-4o-mini", usage) == 0.45

    pricing_file = tmp_path / "pricing.json"
    pricing_file.write_text(
        json.dumps({"custom-model": {"input_per_1m": 1.0, "output_per_1m": 4.0}}),
        encoding="utf-8",
    )
    table = load_pricing_table(pricing_file)
    assert estimate_cost_usd("custom-model", usage, table) == 3.0


def test_percentile_interpolation() -> None:
    assert percentile([], 95.0) == 0.0
    assert percentile([42.0], 95.0) == 42.0
    assert percentile([10.0, 20.0, 30.0, 40.0, 50.0], 50.0) == 30.0
    assert percentile([10.0, 20.0, 30.0, 40.0, 50.0], 95.0) == 48.0
