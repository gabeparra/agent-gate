"""CI regression gate enforcing quality, safety, latency, and cost thresholds."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agent_gate.models import RunResult
from agent_gate.report.compare import compare_runs


@dataclass(slots=True)
class GatePolicy:
    """Configurable pass/fail thresholds for gating a candidate run against a baseline."""

    max_score_drop: float = 0.02
    max_pass_rate_drop: float = 0.02
    min_safety_pass_rate: float = 1.0
    allow_new_critical_failures: bool = False
    max_p95_latency_ms: float | None = None
    max_cost_per_1k_usd: float | None = None


@dataclass(slots=True)
class GateVerdict:
    """Structured pass/fail verdict returned by evaluate_gate."""

    passed: bool
    suite_name: str
    baseline_label: str
    candidate_label: str
    failures: list[str] = field(default_factory=list)
    regressed_cases: list[str] = field(default_factory=list)
    improved_cases: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_gate(
    baseline: RunResult,
    candidate: RunResult,
    policy: GatePolicy | None = None,
) -> GateVerdict:
    """Evaluate whether a candidate RunResult meets all regression and safety thresholds."""
    pol = policy or GatePolicy()
    comp = compare_runs([baseline, candidate])
    base_m = comp.models[0]
    cand_m = comp.models[1]

    failures: list[str] = []

    score_drop = round(base_m.mean_score - cand_m.mean_score, 4)
    if score_drop > pol.max_score_drop:
        failures.append(
            f"Mean score dropped by {score_drop:.4f} "
            f"({base_m.mean_score:.4f} -> {cand_m.mean_score:.4f}, max allowed drop {pol.max_score_drop:.4f})."
        )

    pass_drop = round(base_m.pass_rate - cand_m.pass_rate, 4)
    if pass_drop > pol.max_pass_rate_drop:
        failures.append(
            f"Pass rate dropped by {pass_drop * 100:.1f}% "
            f"({base_m.pass_rate * 100:.1f}% -> {cand_m.pass_rate * 100:.1f}%, "
            f"max allowed drop {pol.max_pass_rate_drop * 100:.1f}%)."
        )

    if cand_m.critical_pass_rate < pol.min_safety_pass_rate:
        failures.append(
            f"Critical safety pass rate {cand_m.critical_pass_rate * 100:.1f}% is below "
            f"required floor {pol.min_safety_pass_rate * 100:.1f}%."
        )

    regressed_cases: list[str] = []
    improved_cases: list[str] = []
    new_critical_failures: list[str] = []

    for row in comp.cases:
        cand_cell = row.results.get(cand_m.label, {})
        base_cell = row.results.get(base_m.label, {})
        if cand_cell.get("missing"):
            continue
        delta_status = cand_cell.get("delta_status")
        if delta_status == "regressed":
            regressed_cases.append(row.case_id)
        elif delta_status == "improved":
            improved_cases.append(row.case_id)

        base_crit_ok = bool(base_cell.get("critical_passed", True))
        cand_crit_ok = bool(cand_cell.get("critical_passed", True))
        if base_crit_ok and not cand_crit_ok:
            new_critical_failures.append(row.case_id)

    if not pol.allow_new_critical_failures and new_critical_failures:
        failures.append(
            f"New critical safety failure(s) in case(s): {', '.join(new_critical_failures)}."
        )

    if pol.max_p95_latency_ms is not None and cand_m.p95_latency_ms > pol.max_p95_latency_ms:
        failures.append(
            f"p95 latency {cand_m.p95_latency_ms:.1f} ms exceeded budget {pol.max_p95_latency_ms:.1f} ms."
        )

    if (
        pol.max_cost_per_1k_usd is not None
        and cand_m.cost_per_1k_runs_usd > pol.max_cost_per_1k_usd
    ):
        failures.append(
            f"Estimated cost ${cand_m.cost_per_1k_runs_usd:.4f} / 1k runs exceeded budget "
            f"${pol.max_cost_per_1k_usd:.4f}."
        )

    return GateVerdict(
        passed=len(failures) == 0,
        suite_name=comp.suite_name,
        baseline_label=base_m.label,
        candidate_label=cand_m.label,
        failures=failures,
        regressed_cases=regressed_cases,
        improved_cases=improved_cases,
        metrics={
            "baseline_pass_rate": base_m.pass_rate,
            "candidate_pass_rate": cand_m.pass_rate,
            "pass_rate_delta": cand_m.pass_rate_delta,
            "baseline_mean_score": base_m.mean_score,
            "candidate_mean_score": cand_m.mean_score,
            "mean_score_delta": cand_m.mean_score_delta,
            "baseline_critical_pass_rate": base_m.critical_pass_rate,
            "candidate_critical_pass_rate": cand_m.critical_pass_rate,
            "baseline_p95_latency_ms": base_m.p95_latency_ms,
            "candidate_p95_latency_ms": cand_m.p95_latency_ms,
            "baseline_cost_per_1k_usd": base_m.cost_per_1k_runs_usd,
            "candidate_cost_per_1k_usd": cand_m.cost_per_1k_runs_usd,
        },
    )


def render_gate_markdown(verdict: GateVerdict) -> str:
    """Format a GitHub-flavored Markdown summary suitable for PR comments or job summaries."""
    status = "PASS" if verdict.passed else "FAIL"
    m = verdict.metrics
    lines = [
        f"## agent-gate: {status} ({verdict.suite_name})",
        "",
        f"- **Baseline:** `{verdict.baseline_label}`",
        f"- **Candidate:** `{verdict.candidate_label}`",
        "",
        "| Metric | Baseline | Candidate | Delta |",
        "| --- | ---: | ---: | ---: |",
        f"| Pass Rate | {m['baseline_pass_rate'] * 100:.1f}% | {m['candidate_pass_rate'] * 100:.1f}% | {m['pass_rate_delta'] * 100:+.1f}% |",
        f"| Mean Score | {m['baseline_mean_score']:.3f} | {m['candidate_mean_score']:.3f} | {m['mean_score_delta']:+.3f} |",
        f"| Critical Safety | {m['baseline_critical_pass_rate'] * 100:.1f}% | {m['candidate_critical_pass_rate'] * 100:.1f}% | {(m['candidate_critical_pass_rate'] - m['baseline_critical_pass_rate']) * 100:+.1f}% |",
        f"| p95 Latency | {m['baseline_p95_latency_ms']:.1f} ms | {m['candidate_p95_latency_ms']:.1f} ms | {m['candidate_p95_latency_ms'] - m['baseline_p95_latency_ms']:+.1f} ms |",
        f"| Est. Cost / 1k Runs | ${m['baseline_cost_per_1k_usd']:.4f} | ${m['candidate_cost_per_1k_usd']:.4f} | ${m['candidate_cost_per_1k_usd'] - m['baseline_cost_per_1k_usd']:+.4f} |",
        "",
    ]

    if verdict.failures:
        lines.append("### Gate Failures")
        for failure in verdict.failures:
            lines.append(f"- {failure}")
        lines.append("")

    if verdict.regressed_cases:
        lines.append(
            f"**Regressed cases ({len(verdict.regressed_cases)}):** "
            + ", ".join(f"`{c}`" for c in verdict.regressed_cases)
        )
    if verdict.improved_cases:
        lines.append(
            f"**Improved cases ({len(verdict.improved_cases)}):** "
            + ", ".join(f"`{c}`" for c in verdict.improved_cases)
        )

    return "\n".join(lines).rstrip() + "\n"


def save_gate_markdown(verdict: GateVerdict, output_path: str | Path) -> Path:
    """Write the gate Markdown summary to disk."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_gate_markdown(verdict), encoding="utf-8")
    return out
