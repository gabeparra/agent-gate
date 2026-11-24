"""Tests for CI regression gate thresholds, exit codes, and Markdown summary output."""

from __future__ import annotations

from pathlib import Path

from agent_gate.cli import main
from agent_gate.gate import GatePolicy, evaluate_gate, render_gate_markdown
from agent_gate.loader import load_suite
from agent_gate.providers import MockProvider
from agent_gate.runner import run_suite, save_run_result


def test_gate_passes_for_faster_cheaper_candidate_and_fails_for_regressed() -> None:
    suite = load_suite("datasets/core_evals.yaml")
    baseline = run_suite(suite, MockProvider(model="mock-baseline", profile="baseline"))
    candidate = run_suite(suite, MockProvider(model="mock-candidate", profile="candidate"))
    regressed = run_suite(suite, MockProvider(model="mock-regressed", profile="regressed"))

    ok = evaluate_gate(baseline, candidate)
    assert ok.passed is True
    assert ok.failures == []

    bad = evaluate_gate(baseline, regressed)
    assert bad.passed is False
    assert any("Mean score dropped" in f for f in bad.failures)
    assert any("Critical safety pass rate" in f for f in bad.failures)
    assert len(bad.regressed_cases) == 6

    md = render_gate_markdown(bad)
    assert "## agent-gate: FAIL (core-evals)" in md
    assert "### Gate Failures" in md


def test_gate_enforces_latency_and_cost_budgets() -> None:
    suite = load_suite("datasets/core_evals.yaml")
    baseline = run_suite(suite, MockProvider(model="mock-baseline", profile="baseline"))
    candidate = run_suite(suite, MockProvider(model="mock-candidate", profile="candidate"))

    strict_budget = GatePolicy(max_p95_latency_ms=50.0, max_cost_per_1k_usd=0.01)
    verdict = evaluate_gate(baseline, candidate, strict_budget)
    assert verdict.passed is False
    assert any("p95 latency" in f for f in verdict.failures)
    assert any("Estimated cost" in f for f in verdict.failures)


def test_cli_gate_returns_zero_on_pass_and_one_on_fail(tmp_path: Path) -> None:
    suite = load_suite("datasets/agentic_guardrails.yaml")
    base_path = save_run_result(
        run_suite(suite, MockProvider(model="mock-baseline", profile="baseline")),
        tmp_path / "base.json",
    )
    cand_path = save_run_result(
        run_suite(suite, MockProvider(model="mock-candidate", profile="candidate")),
        tmp_path / "cand.json",
    )
    reg_path = save_run_result(
        run_suite(suite, MockProvider(model="mock-regressed", profile="regressed")),
        tmp_path / "reg.json",
    )
    md_out = tmp_path / "gate.md"

    assert (
        main(
            [
                "gate",
                "--baseline",
                str(base_path),
                "--candidate",
                str(cand_path),
                "--markdown-out",
                str(md_out),
            ]
        )
        == 0
    )
    assert md_out.exists()
    assert "PASS" in md_out.read_text(encoding="utf-8")

    assert (
        main(
            [
                "gate",
                "--baseline",
                str(base_path),
                "--candidate",
                str(reg_path),
            ]
        )
        == 1
    )
