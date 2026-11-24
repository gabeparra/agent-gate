"""Command-line entry point for agent-gate."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from agent_gate.loader import SuiteLoadError, load_suite
from agent_gate.models import RunResult
from agent_gate.providers.base import ProviderError
from agent_gate.providers.registry import SUPPORTED_PROVIDERS, create_provider
from agent_gate.runner import run_suite, save_run_result


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
        print(f"  [{badge}] {case.case_id} ({case.category}) score={case.score:.2f} {case.latency_ms:.0f}ms")
        for sc in case.scores:
            if not sc.passed:
                crit = " [CRITICAL]" if sc.critical else ""
                print(f"         - {sc.name}{crit}: {sc.reason}")


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


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return _handle_run(args)
        parser.error(f"Unknown command: {args.command}")
        return 2
    except (SuiteLoadError, ProviderError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
