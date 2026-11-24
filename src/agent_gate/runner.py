"""Suite execution engine connecting providers, sandbox guardrails, scorers, and metrics."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent_gate.guardrails import ActionHonestyScorer, MCPContractScorer, WorkspaceSandbox
from agent_gate.metrics import estimate_cost_usd, load_pricing_table, summarize_cases
from agent_gate.models import CaseResult, EvalCase, EvalSuite, RunResult, ScoreResult
from agent_gate.providers.base import Provider
from agent_gate.providers.mock import MockProvider
from agent_gate.scorers import (
    CitationGroundingScorer,
    JsonSchemaScorer,
    LLMJudgeScorer,
    NumericSpecificityScorer,
    PromptInjectionScorer,
    RefusalAndHedgingScorer,
    Scorer,
)


def build_default_scorers(judge_provider: Provider | None = None) -> dict[str, Scorer]:
    """Instantiate the built-in registry of deterministic, guardrail, and LLM-judge scorers."""
    judge = judge_provider or MockProvider(model="mock-judge", profile="baseline")
    scorers: list[Scorer] = [
        CitationGroundingScorer(),
        JsonSchemaScorer(),
        RefusalAndHedgingScorer(),
        NumericSpecificityScorer(),
        PromptInjectionScorer(),
        LLMJudgeScorer(judge),
        MCPContractScorer(),
        ActionHonestyScorer(),
    ]
    return {s.name: s for s in scorers}


def run_suite(
    suite: EvalSuite,
    provider: Provider,
    *,
    judge_provider: Provider | None = None,
    pricing_path: str | Path | None = None,
    tags: list[str] | None = None,
    run_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> RunResult:
    """Execute every matching case in an EvalSuite and return a serializable RunResult."""
    pricing_table = load_pricing_table(pricing_path)
    scorers = build_default_scorers(judge_provider=judge_provider)

    tag_filter = {t.strip().lower() for t in (tags or []) if t.strip()}
    selected_cases = [
        case
        for case in suite.cases
        if not tag_filter or tag_filter.intersection({t.lower() for t in case.tags})
    ]

    case_results: list[CaseResult] = []
    for case in selected_cases:
        case_results.append(
            _evaluate_case(case, provider, scorers=scorers, pricing_table=pricing_table)
        )

    summary = summarize_cases(case_results)
    return RunResult(
        run_id=run_id or f"run-{uuid.uuid4().hex[:10]}",
        suite_name=suite.name,
        suite_path=suite.source_path,
        provider=provider.provider_name,
        model=provider.model,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        summary=summary,
        cases=case_results,
        metadata=dict(metadata or {}),
    )


def _evaluate_case(
    case: EvalCase,
    provider: Provider,
    *,
    scorers: dict[str, Scorer],
    pricing_table: dict[str, tuple[float, float]],
) -> CaseResult:
    response = provider.complete(case)

    uses_sandbox = bool(
        case.sandbox_files
        or case.tools
        or "action_honesty" in case.checks
        or "mcp_contract" in case.checks
    )
    sandbox_report = (
        WorkspaceSandbox.from_case(case).replay(response.tool_calls) if uses_sandbox else None
    )

    score_results: list[ScoreResult] = []
    for check_name, check_config in case.checks.items():
        if check_config is False or check_config is None:
            continue
        cfg = check_config if isinstance(check_config, dict) else {}
        scorer = scorers.get(check_name)
        if scorer is None:
            raise ValueError(f"Unknown scorer '{check_name}' referenced in case '{case.case_id}'")
        if isinstance(scorer, ActionHonestyScorer):
            score_results.append(scorer.score(case, response, cfg, sandbox_report=sandbox_report))
        else:
            score_results.append(scorer.score(case, response, cfg))

    if score_results:
        mean_score = round(sum(s.score for s in score_results) / len(score_results), 4)
        case_passed = all(s.passed for s in score_results)
        critical_scores = [s for s in score_results if s.critical]
        critical_passed = all(s.passed for s in critical_scores)
    else:
        mean_score = 1.0
        case_passed = True
        critical_passed = True

    cost_usd = estimate_cost_usd(response.model, response.usage, pricing_table)
    recorded_calls = (
        list(sandbox_report.executed_calls)
        if sandbox_report is not None
        else list(response.tool_calls)
    )
    final_disk = dict(sandbox_report.final_files) if sandbox_report is not None else {}

    return CaseResult(
        case_id=case.case_id,
        category=case.category,
        tags=list(case.tags),
        prompt=case.prompt,
        response_text=response.text,
        passed=case_passed,
        score=mean_score,
        critical_passed=critical_passed,
        latency_ms=response.latency_ms,
        usage=response.usage,
        cost_usd=cost_usd,
        scores=score_results,
        tool_calls=recorded_calls,
        sandbox_state=final_disk,
    )


def save_run_result(result: RunResult, output_path: str | Path) -> Path:
    """Persist a RunResult to JSON."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    return out


def load_run_result(path: str | Path) -> RunResult:
    """Load a previously saved RunResult JSON file."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return RunResult.from_dict(raw)
