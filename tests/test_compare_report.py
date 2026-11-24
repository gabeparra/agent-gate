"""Tests for side-by-side model comparison and self-contained HTML report generation."""

from __future__ import annotations

import json
from pathlib import Path

from agent_gate.cli import main
from agent_gate.loader import load_suite
from agent_gate.providers import MockProvider
from agent_gate.report import compare_runs, render_comparison_html
from agent_gate.runner import run_suite, save_run_result


def test_compare_runs_computes_deltas_and_regressions() -> None:
    suite = load_suite("datasets/core_evals.yaml")
    baseline = run_suite(suite, MockProvider(model="mock-baseline", profile="baseline"))
    candidate = run_suite(suite, MockProvider(model="mock-candidate", profile="candidate"))
    regressed = run_suite(suite, MockProvider(model="mock-regressed", profile="regressed"))

    report = compare_runs([baseline, candidate, regressed])
    assert report.suite_name == "core-evals"
    assert len(report.models) == 3

    cand_model = report.models[1]
    assert cand_model.regressions == 0
    assert cand_model.p95_latency_delta_ms < 0  # candidate is faster
    assert cand_model.cost_per_1k_delta_usd < 0  # candidate is cheaper

    reg_model = report.models[2]
    assert reg_model.regressions == 6
    assert reg_model.critical_pass_rate_delta == -1.0
    assert all(c.status == "regressed" for c in report.cases)


def test_html_report_is_self_contained_and_escapes_untrusted_text() -> None:
    suite = load_suite("datasets/core_evals.yaml")
    baseline = run_suite(suite, MockProvider(model="mock-baseline", profile="baseline"))
    candidate = run_suite(suite, MockProvider(model="mock-candidate", profile="candidate"))
    candidate.cases[0].response_text = '<script>alert("xss")</script> safe text'

    report = compare_runs([baseline, candidate])
    html_out = render_comparison_html(report)

    assert "<!DOCTYPE html>" in html_out
    assert "http://" not in html_out and "https://" not in html_out
    assert '<script>alert("xss")</script>' not in html_out
    assert "&lt;script&gt;alert(&quot;xss&quot;)&lt;/script&gt;" in html_out


def test_cli_compare_writes_html_and_json_artifacts(tmp_path: Path) -> None:
    suite = load_suite("datasets/agentic_guardrails.yaml")
    base_json = save_run_result(
        run_suite(suite, MockProvider(model="mock-baseline", profile="baseline")),
        tmp_path / "base.json",
    )
    cand_json = save_run_result(
        run_suite(suite, MockProvider(model="mock-candidate", profile="candidate")),
        tmp_path / "cand.json",
    )
    reg_json = save_run_result(
        run_suite(suite, MockProvider(model="mock-regressed", profile="regressed")),
        tmp_path / "reg.json",
    )

    html_path = tmp_path / "report.html"
    diff_path = tmp_path / "diff.json"
    code = main(
        [
            "compare",
            str(base_json),
            str(cand_json),
            str(reg_json),
            "--html-out",
            str(html_path),
            "--json-out",
            str(diff_path),
        ]
    )
    assert code == 0
    assert html_path.exists()
    assert diff_path.exists()

    parsed = json.loads(diff_path.read_text(encoding="utf-8"))
    assert len(parsed["models"]) == 3
    assert parsed["models"][2]["regressions"] == 4
