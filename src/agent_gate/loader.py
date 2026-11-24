"""Suite loader supporting golden YAML and JSONL evaluation datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from agent_gate.models import EvalCase, EvalSuite


class SuiteLoadError(ValueError):
    """Raised when an evaluation dataset file is malformed or missing required fields."""


def load_suite(path: str | Path) -> EvalSuite:
    """Load an EvalSuite from a .yaml, .yml, or .jsonl file."""
    suite_path = Path(path)
    if not suite_path.exists():
        raise SuiteLoadError(f"Suite file not found: {suite_path}")

    suffix = suite_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return _load_yaml_suite(suite_path)
    if suffix == ".jsonl":
        return _load_jsonl_suite(suite_path)
    raise SuiteLoadError(f"Unsupported suite format '{suffix}'. Use .yaml, .yml, or .jsonl.")


def _load_yaml_suite(suite_path: Path) -> EvalSuite:
    raw = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SuiteLoadError(f"YAML suite must be a mapping at top level: {suite_path}")

    raw_cases = raw.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise SuiteLoadError(f"YAML suite must contain a non-empty 'cases' list: {suite_path}")

    default_system = str(raw.get("default_system_prompt", ""))
    cases = _parse_cases(raw_cases, default_system, suite_path)
    return EvalSuite(
        name=str(raw.get("name", suite_path.stem)),
        description=str(raw.get("description", "")),
        version=str(raw.get("version", "1.0")),
        default_system_prompt=default_system,
        cases=cases,
        source_path=str(suite_path),
    )


def _load_jsonl_suite(suite_path: Path) -> EvalSuite:
    raw_cases: list[dict[str, Any]] = []
    for line_no, line in enumerate(suite_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise SuiteLoadError(f"Invalid JSON on line {line_no} of {suite_path}: {exc}") from exc
        if not isinstance(obj, dict):
            raise SuiteLoadError(f"Line {line_no} of {suite_path} must be a JSON object")
        raw_cases.append(obj)

    if not raw_cases:
        raise SuiteLoadError(f"JSONL suite has no cases: {suite_path}")

    cases = _parse_cases(raw_cases, "", suite_path)
    return EvalSuite(
        name=suite_path.stem,
        description=f"JSONL evaluation suite loaded from {suite_path.name}",
        version="1.0",
        cases=cases,
        source_path=str(suite_path),
    )


def _parse_cases(
    raw_cases: list[Any],
    default_system_prompt: str,
    suite_path: Path,
) -> list[EvalCase]:
    cases: list[EvalCase] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(raw_cases):
        if not isinstance(item, dict):
            raise SuiteLoadError(f"Case #{index + 1} in {suite_path} must be a mapping")
        for required in ("id", "prompt"):
            if required not in item or not str(item[required]).strip():
                raise SuiteLoadError(
                    f"Case #{index + 1} in {suite_path} is missing required field '{required}'"
                )
        case = EvalCase.from_dict(item)
        if case.case_id in seen_ids:
            raise SuiteLoadError(f"Duplicate case id '{case.case_id}' in {suite_path}")
        seen_ids.add(case.case_id)
        if not case.system_prompt and default_system_prompt:
            case.system_prompt = default_system_prompt
        cases.append(case)
    return cases
