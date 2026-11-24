"""Tests for YAML and JSONL evaluation suite loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_gate.loader import SuiteLoadError, load_suite


def test_load_yaml_core_suite() -> None:
    suite = load_suite(Path("datasets/core_evals.yaml"))
    assert suite.name == "core-evals"
    assert len(suite.cases) == 6
    assert suite.cases[0].case_id == "rag-sla-credits"
    assert suite.cases[0].system_prompt.startswith("You are an enterprise support")


def test_load_jsonl_suite(tmp_path: Path) -> None:
    jsonl_file = tmp_path / "quick.jsonl"
    rows = [
        {
            "id": "case-1",
            "category": "numeric",
            "prompt": "How many days?",
            "checks": {"numeric_specificity": {"expected_values": [30]}},
        },
        {
            "id": "case-2",
            "category": "refusal",
            "prompt": "Secret date?",
            "checks": {"refusal_hedging": {"expect_refusal": True}},
        },
    ]
    jsonl_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    suite = load_suite(jsonl_file)
    assert suite.name == "quick"
    assert [c.case_id for c in suite.cases] == ["case-1", "case-2"]


def test_load_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    bad = tmp_path / "dup.yaml"
    bad.write_text(
        "name: dup\ncases:\n  - id: a\n    prompt: hi\n  - id: a\n    prompt: bye\n",
        encoding="utf-8",
    )
    with pytest.raises(SuiteLoadError, match="Duplicate case id"):
        load_suite(bad)


def test_load_rejects_missing_file_and_bad_extension(tmp_path: Path) -> None:
    with pytest.raises(SuiteLoadError, match="not found"):
        load_suite(tmp_path / "missing.yaml")

    bad_ext = tmp_path / "suite.txt"
    bad_ext.write_text("hello", encoding="utf-8")
    with pytest.raises(SuiteLoadError, match="Unsupported suite format"):
        load_suite(bad_ext)
