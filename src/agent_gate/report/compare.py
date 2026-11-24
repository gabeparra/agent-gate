"""Side-by-side comparison engine for 2 or more evaluation runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agent_gate.models import CaseResult, RunResult


@dataclass(slots=True)
class ModelComparison:
    """Summary metrics and baseline deltas for one model in a comparison."""

    label: str
    run_id: str
    provider: str
    model: str
    is_baseline: bool
    total_cases: int
    passed_cases: int
    pass_rate: float
    mean_score: float
    critical_pass_rate: float
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    total_tokens: int
    total_cost_usd: float
    cost_per_1k_runs_usd: float
    pass_rate_delta: float = 0.0
    mean_score_delta: float = 0.0
    critical_pass_rate_delta: float = 0.0
    p95_latency_delta_ms: float = 0.0
    cost_per_1k_delta_usd: float = 0.0
    regressions: int = 0
    improvements: int = 0
    by_category: dict[str, dict[str, float]] = field(default_factory=dict)
    by_scorer: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass(slots=True)
class CaseComparison:
    """Per-case side-by-side results across baseline and candidate runs."""

    case_id: str
    category: str
    prompt: str
    status: str  # regressed | improved | unchanged_pass | unchanged_fail | mixed
    has_critical_failure: bool
    results: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(slots=True)
class ComparisonReport:
    """Complete multi-model comparison dataset."""

    suite_name: str
    baseline_label: str
    models: list[ModelComparison]
    cases: list[CaseComparison]
    all_categories: list[str]
    all_scorers: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _unique_labels(runs: list[RunResult]) -> list[str]:
    counts: dict[str, int] = {}
    labels: list[str] = []
    for run in runs:
        base = f"{run.provider}/{run.model}"
        counts[base] = counts.get(base, 0) + 1
        label = base if counts[base] == 1 else f"{base} (#{counts[base]})"
        labels.append(label)
    return labels


def compare_runs(runs: list[RunResult]) -> ComparisonReport:
    """Compare a baseline run (runs[0]) against one or more candidate runs."""
    if len(runs) < 2:
        raise ValueError(
            "compare_runs requires at least 2 RunResult objects (baseline + candidate)."
        )

    baseline = runs[0]
    labels = _unique_labels(runs)
    baseline_label = labels[0]

    # Collect ordered union of all case IDs starting with baseline order.
    case_order: list[str] = []
    seen_cases: set[str] = set()
    run_case_maps: list[dict[str, CaseResult]] = []
    for run in runs:
        cmap = {c.case_id: c for c in run.cases}
        run_case_maps.append(cmap)
        for c in run.cases:
            if c.case_id not in seen_cases:
                seen_cases.add(c.case_id)
                case_order.append(c.case_id)

    baseline_map = run_case_maps[0]
    regression_counts = [0 for _ in runs]
    improvement_counts = [0 for _ in runs]

    case_rows: list[CaseComparison] = []
    categories_set: set[str] = set()
    scorers_set: set[str] = set()

    for case_id in case_order:
        base_case = baseline_map.get(case_id)
        ref_case = base_case or next(cmap[case_id] for cmap in run_case_maps if case_id in cmap)
        categories_set.add(ref_case.category)

        per_model_results: dict[str, dict[str, Any]] = {}
        statuses: list[str] = []
        any_crit_fail = False

        for idx, (label, cmap) in enumerate(zip(labels, run_case_maps, strict=True)):
            c = cmap.get(case_id)
            if c is None:
                per_model_results[label] = {"missing": True}
                continue

            for sc in c.scores:
                scorers_set.add(sc.name)
            if not c.critical_passed:
                any_crit_fail = True

            cell_delta = "baseline"
            if idx > 0 and base_case is not None:
                if base_case.passed and not c.passed:
                    cell_delta = "regressed"
                    regression_counts[idx] += 1
                elif not base_case.passed and c.passed:
                    cell_delta = "improved"
                    improvement_counts[idx] += 1
                elif c.passed:
                    cell_delta = "unchanged_pass"
                else:
                    cell_delta = "unchanged_fail"
                statuses.append(cell_delta)

            per_model_results[label] = {
                "missing": False,
                "passed": c.passed,
                "score": c.score,
                "critical_passed": c.critical_passed,
                "latency_ms": c.latency_ms,
                "cost_usd": c.cost_usd,
                "delta_status": cell_delta,
                "score_delta": round(c.score - base_case.score, 4) if base_case else 0.0,
                "response_text": c.response_text,
                "scores": [
                    {
                        "name": s.name,
                        "passed": s.passed,
                        "score": s.score,
                        "critical": s.critical,
                        "reason": s.reason,
                    }
                    for s in c.scores
                ],
                "tool_calls": [
                    {
                        "tool_name": tc.tool_name,
                        "arguments": tc.arguments,
                        "status": tc.status,
                        "error": tc.error,
                    }
                    for tc in c.tool_calls
                ],
            }

        if "regressed" in statuses:
            row_status = "regressed"
        elif "improved" in statuses:
            row_status = "improved"
        elif all(s == "unchanged_pass" for s in statuses) and (
            base_case is None or base_case.passed
        ):
            row_status = "unchanged_pass"
        elif all(s == "unchanged_fail" for s in statuses):
            row_status = "unchanged_fail"
        else:
            row_status = "mixed"

        case_rows.append(
            CaseComparison(
                case_id=case_id,
                category=ref_case.category,
                prompt=ref_case.prompt,
                status=row_status,
                has_critical_failure=any_crit_fail,
                results=per_model_results,
            )
        )

    base_s = baseline.summary
    model_rows: list[ModelComparison] = []
    for idx, (label, run) in enumerate(zip(labels, runs, strict=True)):
        s = run.summary
        is_base = idx == 0
        model_rows.append(
            ModelComparison(
                label=label,
                run_id=run.run_id,
                provider=run.provider,
                model=run.model,
                is_baseline=is_base,
                total_cases=s.total_cases,
                passed_cases=s.passed_cases,
                pass_rate=s.pass_rate,
                mean_score=s.mean_score,
                critical_pass_rate=s.critical_pass_rate,
                mean_latency_ms=s.mean_latency_ms,
                p50_latency_ms=s.p50_latency_ms,
                p95_latency_ms=s.p95_latency_ms,
                total_tokens=s.total_prompt_tokens + s.total_completion_tokens,
                total_cost_usd=s.total_cost_usd,
                cost_per_1k_runs_usd=s.cost_per_1k_runs_usd,
                pass_rate_delta=0.0 if is_base else round(s.pass_rate - base_s.pass_rate, 4),
                mean_score_delta=0.0 if is_base else round(s.mean_score - base_s.mean_score, 4),
                critical_pass_rate_delta=(
                    0.0 if is_base else round(s.critical_pass_rate - base_s.critical_pass_rate, 4)
                ),
                p95_latency_delta_ms=(
                    0.0 if is_base else round(s.p95_latency_ms - base_s.p95_latency_ms, 2)
                ),
                cost_per_1k_delta_usd=(
                    0.0
                    if is_base
                    else round(s.cost_per_1k_runs_usd - base_s.cost_per_1k_runs_usd, 4)
                ),
                regressions=regression_counts[idx],
                improvements=improvement_counts[idx],
                by_category=dict(s.by_category),
                by_scorer=dict(s.by_scorer),
            )
        )

    return ComparisonReport(
        suite_name=baseline.suite_name,
        baseline_label=baseline_label,
        models=model_rows,
        cases=case_rows,
        all_categories=sorted(categories_set),
        all_scorers=sorted(scorers_set),
    )


def save_comparison_json(report: ComparisonReport, output_path: str | Path) -> Path:
    """Write a machine-readable JSON comparison summary."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    return out
