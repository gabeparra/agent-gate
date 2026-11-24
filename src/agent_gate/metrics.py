"""Cost estimation, latency percentile math, and run summary aggregation."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from agent_gate.models import CaseResult, RunSummary, TokenUsage

# Default pricing per 1,000,000 tokens: (input_usd_per_1m, output_usd_per_1m).
# Override any model rate via a JSON/YAML pricing file passed to the CLI.
DEFAULT_MODEL_PRICING_USD_PER_1M: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "o4-mini": (1.10, 4.40),
    "claude-3-5-sonnet-latest": (3.00, 15.00),
    "claude-3-7-sonnet-latest": (3.00, 15.00),
    "claude-3-5-haiku-latest": (0.80, 4.00),
    "openai/gpt-4o-mini": (0.15, 0.60),
    "anthropic/claude-3.5-sonnet": (3.00, 15.00),
    "llama3.1:8b": (0.0, 0.0),
    "local-model": (0.0, 0.0),
    # Synthetic realistic demo tiers for offline mock comparison reports.
    "mock-baseline": (2.50, 10.00),
    "mock-candidate": (0.40, 1.60),
    "mock-regressed": (0.15, 0.60),
}


def load_pricing_table(path: str | Path | None = None) -> dict[str, tuple[float, float]]:
    """Return default model pricing merged with optional user-supplied overrides."""
    table = dict(DEFAULT_MODEL_PRICING_USD_PER_1M)
    if path is None:
        return table

    pricing_path = Path(path)
    raw_text = pricing_path.read_text(encoding="utf-8")
    raw = (
        json.loads(raw_text)
        if pricing_path.suffix.lower() == ".json"
        else yaml.safe_load(raw_text)
    )
    if not isinstance(raw, dict):
        raise ValueError(f"Pricing file must map model names to input/output rates: {pricing_path}")

    for model_name, rates in raw.items():
        if isinstance(rates, dict):
            inp = float(rates.get("input_per_1m", rates.get("input", 0.0)))
            out = float(rates.get("output_per_1m", rates.get("output", 0.0)))
            table[str(model_name)] = (inp, out)
        elif isinstance(rates, (list, tuple)) and len(rates) == 2:
            table[str(model_name)] = (float(rates[0]), float(rates[1]))
    return table


def estimate_cost_usd(
    model: str,
    usage: TokenUsage,
    pricing_table: dict[str, tuple[float, float]] | None = None,
) -> float:
    """Estimate USD cost for a single model completion."""
    rates_map = pricing_table if pricing_table is not None else DEFAULT_MODEL_PRICING_USD_PER_1M
    rates = rates_map.get(model)
    if rates is None:
        # Match stripped provider prefix if present (e.g. openai/gpt-4o-mini -> gpt-4o-mini).
        short_name = model.split("/", 1)[-1]
        rates = rates_map.get(short_name, (0.0, 0.0))
    in_rate, out_rate = rates
    cost = (usage.prompt_tokens / 1_000_000.0) * in_rate + (
        usage.completion_tokens / 1_000_000.0
    ) * out_rate
    return round(cost, 8)


def percentile(values: list[float], pct: float) -> float:
    """Compute linear-interpolated percentile (0 to 100) for a non-empty list."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 2)
    rank = (pct / 100.0) * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return round(ordered[lower], 2)
    weight = rank - lower
    return round(ordered[lower] * (1.0 - weight) + ordered[upper] * weight, 2)


def summarize_cases(cases: list[CaseResult]) -> RunSummary:
    """Aggregate pass rates, critical safety rate, latency percentiles, and cost."""
    total = len(cases)
    if total == 0:
        return RunSummary(
            total_cases=0,
            passed_cases=0,
            pass_rate=0.0,
            mean_score=0.0,
            critical_total=0,
            critical_passed=0,
            critical_pass_rate=1.0,
            total_prompt_tokens=0,
            total_completion_tokens=0,
            total_cost_usd=0.0,
            cost_per_1k_runs_usd=0.0,
            mean_latency_ms=0.0,
            p50_latency_ms=0.0,
            p95_latency_ms=0.0,
        )

    passed_cases = sum(1 for c in cases if c.passed)
    mean_score = round(sum(c.score for c in cases) / total, 4)

    critical_scores = [s for c in cases for s in c.scores if s.critical]
    critical_total = len(critical_scores)
    critical_passed = sum(1 for s in critical_scores if s.passed)
    critical_pass_rate = (
        round(critical_passed / critical_total, 4) if critical_total > 0 else 1.0
    )

    total_prompt_tokens = sum(c.usage.prompt_tokens for c in cases)
    total_completion_tokens = sum(c.usage.completion_tokens for c in cases)
    total_cost_usd = round(sum(c.cost_usd for c in cases), 8)
    cost_per_1k = round((total_cost_usd / total) * 1000.0, 4)

    latencies = [c.latency_ms for c in cases]
    mean_latency = round(sum(latencies) / total, 2)
    p50_latency = percentile(latencies, 50.0)
    p95_latency = percentile(latencies, 95.0)

    cat_buckets: dict[str, list[CaseResult]] = defaultdict(list)
    scorer_buckets: dict[str, list[Any]] = defaultdict(list)
    for c in cases:
        cat_buckets[c.category].append(c)
        for s in c.scores:
            scorer_buckets[s.name].append(s)

    by_category = {
        cat: {
            "total": len(items),
            "passed": sum(1 for i in items if i.passed),
            "pass_rate": round(sum(1 for i in items if i.passed) / len(items), 4),
            "mean_score": round(sum(i.score for i in items) / len(items), 4),
        }
        for cat, items in sorted(cat_buckets.items())
    }
    by_scorer = {
        name: {
            "total": len(items),
            "passed": sum(1 for i in items if i.passed),
            "pass_rate": round(sum(1 for i in items if i.passed) / len(items), 4),
            "mean_score": round(sum(i.score for i in items) / len(items), 4),
        }
        for name, items in sorted(scorer_buckets.items())
    }

    return RunSummary(
        total_cases=total,
        passed_cases=passed_cases,
        pass_rate=round(passed_cases / total, 4),
        mean_score=mean_score,
        critical_total=critical_total,
        critical_passed=critical_passed,
        critical_pass_rate=critical_pass_rate,
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        total_cost_usd=total_cost_usd,
        cost_per_1k_runs_usd=cost_per_1k,
        mean_latency_ms=mean_latency,
        p50_latency_ms=p50_latency,
        p95_latency_ms=p95_latency,
        by_category=by_category,
        by_scorer=by_scorer,
    )
