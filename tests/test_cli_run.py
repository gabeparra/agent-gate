"""End-to-end CLI tests for `agent-gate run`."""

from __future__ import annotations

from pathlib import Path

from agent_gate.cli import main
from agent_gate.runner import load_run_result


def test_cli_run_baseline_passes_strict_and_saves_json(tmp_path: Path) -> None:
    out_file = tmp_path / "baseline.json"
    code = main(
        [
            "run",
            "datasets/core_evals.yaml",
            "--provider",
            "mock",
            "--profile",
            "baseline",
            "--strict",
            "--out",
            str(out_file),
        ]
    )
    assert code == 0
    assert out_file.exists()

    loaded = load_run_result(out_file)
    assert loaded.suite_name == "core-evals"
    assert loaded.summary.total_cases == 6
    assert loaded.summary.passed_cases == 6
    assert loaded.summary.critical_pass_rate == 1.0
    assert loaded.summary.total_cost_usd > 0


def test_cli_run_regressed_fails_strict(tmp_path: Path) -> None:
    out_file = tmp_path / "regressed.json"
    code = main(
        [
            "run",
            "datasets/core_evals.yaml",
            "--provider",
            "mock",
            "--profile",
            "regressed",
            "--strict",
            "--out",
            str(out_file),
        ]
    )
    assert code == 1
    loaded = load_run_result(out_file)
    assert loaded.summary.passed_cases == 0
    assert loaded.summary.critical_pass_rate == 0.0
