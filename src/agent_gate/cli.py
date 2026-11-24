"""Command-line entry point for agent-gate."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from agent_gate.gate import GatePolicy, evaluate_gate, render_gate_markdown, save_gate_markdown
from agent_gate.loader import SuiteLoadError, load_suite
from agent_gate.models import RunResult
from agent_gate.providers.base import ProviderError
from agent_gate.providers.registry import SUPPORTED_PROVIDERS, create_provider
from agent_gate.report import (
    ComparisonReport,
    compare_runs,
    save_comparison_html,
    save_comparison_json,
)
from agent_gate.runner import load_run_result, run_suite, save_run_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-gate",
        description="Deterministic evaluation, MCP tool guardrails, and CI regression harness.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_cmd = subparsers.add_parser("run", help="Run an evaluation suite against a provider.")
    run_cmd.add_argument("suite", help="Path to a .yaml, .yml, or .jsonl evaluation suite.")
    run_cmd.add_argument(
        "--provider",
        default="mock",
        choices=list(SUPPORTED_PROVIDERS),
        help="Model provider adapter to use (default: mock).",
    )
    run_cmd.add_argument("--model", default=None, help="Model identifier override.")
    run_cmd.add_argument(
        "--base-url",
        default=None,
        help="Custom OpenAI-compatible or Anthropic base URL.",
    )
    run_cmd.add_argument(
        "--profile",
        default="baseline",
        help="Built-in mock provider profile (baseline, candidate, regressed).",
    )
    run_cmd.add_argument(
        "--mock-fixture",
        default=None,
        help="Optional YAML/JSON file mapping case IDs to deterministic mock responses.",
    )
    run_cmd.add_argument(
        "--judge-provider",
        default=None,
        choices=list(SUPPORTED_PROVIDERS),
        help="Optional provider for llm_judge checks (defaults to mock).",
    )
    run_cmd.add_argument("--judge-model", default=None, help="Model override for llm_judge.")
    run_cmd.add_argument(
        "--pricing",
        default=None,
        help="Optional JSON/YAML file overriding model pricing per 1M tokens.",
    )
    run_cmd.add_argument(
        "--tag",
        action="append",
        default=[],
        help="Filter cases by tag (can be passed multiple times).",
    )
    run_cmd.add_argument(
        "--out",
        default=None,
        help="Path to write the JSON RunResult artifact.",
    )
    run_cmd.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any evaluation case fails.",
    )

    cmp_cmd = subparsers.add_parser(
        "compare",
        help="Compare two or more saved RunResult JSON files and generate HTML/JSON reports.",
    )
    cmp_cmd.add_argument(
        "runs",
        nargs="+",
        help="Two or more RunResult JSON paths (first is baseline, rest are candidates).",
    )
    cmp_cmd.add_argument(
        "--html-out",
        default=None,
        help="Path to write the self-contained static HTML comparison report.",
    )
    cmp_cmd.add_argument(
        "--json-out",
        default=None,
        help="Path to write the machine-readable JSON comparison diff.",
    )

    gate_cmd = subparsers.add_parser(
        "gate",
        help="Fail CI when a candidate run regresses below baseline quality, safety, or budget thresholds.",
    )
    gate_cmd.add_argument("--baseline", required=True, help="Path to baseline RunResult JSON.")
    gate_cmd.add_argument("--candidate", required=True, help="Path to candidate RunResult JSON.")
    gate_cmd.add_argument(
        "--max-score-drop",
        type=float,
        default=0.02,
        help="Maximum allowed drop in mean score (default: 0.02).",
    )
    gate_cmd.add_argument(
        "--max-pass-rate-drop",
        type=float,
        default=0.02,
        help="Maximum allowed drop in pass rate (default: 0.02).",
    )
    gate_cmd.add_argument(
        "--min-safety-rate",
        type=float,
        default=1.0,
        help="Minimum required pass rate on critical safety checks (default: 1.0).",
    )
    gate_cmd.add_argument(
        "--max-p95-ms",
        type=float,
        default=None,
        help="Optional p95 latency ceiling in milliseconds.",
    )
    gate_cmd.add_argument(
        "--max-cost-per-1k",
        type=float,
        default=None,
        help="Optional estimated USD cost ceiling per 1,000 runs.",
    )
    gate_cmd.add_argument(
        "--markdown-out",
        default=None,
        help="Optional path to write a GitHub-flavored Markdown PR summary.",
    )
    return parser


def _print_run_summary(result: RunResult) -> None:
    s = result.summary
    print(f"Suite:    {result.suite_name} ({s.total_cases} cases)")
    print(f"Provider: {result.provider} / {result.model}")
    print(
        f"Score:    {s.mean_score:.3f} avg | "
        f"Pass: {s.passed_cases}/{s.total_cases} ({s.pass_rate * 100:.1f}%) | "
        f"Critical safety: {s.critical_passed}/{s.critical_total} ({s.critical_pass_rate * 100:.1f}%)"
    )
    print(
        f"Perf:     mean {s.mean_latency_ms:.1f} ms | p50 {s.p50_latency_ms:.1f} ms | "
        f"p95 {s.p95_latency_ms:.1f} ms | est. ${s.cost_per_1k_runs_usd:.4f} / 1k runs"
    )
    print("")
    for case in result.cases:
        badge = "PASS" if case.passed else "FAIL"
        print(
            f"  [{badge}] {case.case_id} ({case.category}) score={case.score:.2f} {case.latency_ms:.0f}ms"
        )
        for sc in case.scores:
            if not sc.passed:
                crit = " [CRITICAL]" if sc.critical else ""
                print(f"         - {sc.name}{crit}: {sc.reason}")


def _print_comparison_summary(report: ComparisonReport) -> None:
    print(f"Comparison: {report.suite_name} ({len(report.cases)} cases)")
    print(
        f"{'Model':<28} {'Pass':>8} {'Score':>8} {'Safety':>8} {'p95 ms':>8} {'$/1k':>9} {'Regr':>6}"
    )
    print("-" * 81)
    for m in report.models:
        role = "*" if m.is_baseline else " "
        print(
            f"{role}{m.label:<27} {m.pass_rate * 100:>7.1f}% {m.mean_score:>8.3f} "
            f"{m.critical_pass_rate * 100:>7.1f}% {m.p95_latency_ms:>8.1f} "
            f"${m.cost_per_1k_runs_usd:>8.4f} {m.regressions:>6d}"
        )


def _handle_run(args: argparse.Namespace) -> int:
    suite = load_suite(args.suite)
    provider = create_provider(
        args.provider,
        model=args.model,
        base_url=args.base_url,
        profile=args.profile,
        mock_fixture=args.mock_fixture,
    )
    judge_provider = None
    if args.judge_provider:
        judge_provider = create_provider(args.judge_provider, model=args.judge_model)

    result = run_suite(
        suite,
        provider,
        judge_provider=judge_provider,
        pricing_path=args.pricing,
        tags=args.tag,
        metadata={"profile": args.profile} if args.provider == "mock" else {},
    )
    _print_run_summary(result)

    if args.out:
        saved = save_run_result(result, Path(args.out))
        print(f"\nSaved run artifact: {saved}")

    if args.strict and result.summary.passed_cases < result.summary.total_cases:
        return 1
    return 0


def _handle_compare(args: argparse.Namespace) -> int:
    if len(args.runs) < 2:
        raise ValueError("Provide at least two run JSON files to compare.")
    loaded = [load_run_result(p) for p in args.runs]
    report = compare_runs(loaded)
    _print_comparison_summary(report)

    if args.html_out:
        html_path = save_comparison_html(report, Path(args.html_out))
        print(f"\nSaved HTML report: {html_path}")
    if args.json_out:
        json_path = save_comparison_json(report, Path(args.json_out))
        print(f"Saved JSON diff:   {json_path}")
    return 0


def _handle_gate(args: argparse.Namespace) -> int:
    baseline = load_run_result(args.baseline)
    candidate = load_run_result(args.candidate)
    policy = GatePolicy(
        max_score_drop=args.max_score_drop,
        max_pass_rate_drop=args.max_pass_rate_drop,
        min_safety_pass_rate=args.min_safety_rate,
        max_p95_latency_ms=args.max_p95_ms,
        max_cost_per_1k_usd=args.max_cost_per_1k,
    )
    verdict = evaluate_gate(baseline, candidate, policy)
    print(render_gate_markdown(verdict))

    if args.markdown_out:
        saved = save_gate_markdown(verdict, Path(args.markdown_out))
        print(f"Saved gate summary: {saved}")

    return 0 if verdict.passed else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return _handle_run(args)
        if args.command == "compare":
            return _handle_compare(args)
        if args.command == "gate":
            return _handle_gate(args)
        parser.error(f"Unknown command: {args.command}")
        return 2
    except (SuiteLoadError, ProviderError, ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
