"""Numeric specificity and quantitative accuracy deterministic scorer."""

from __future__ import annotations

import re
from typing import Any

from agent_gate.models import EvalCase, ProviderResponse, ScoreResult
from agent_gate.scorers.base import Scorer

_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_])[-+]?\d+(?:,\d{3})*(?:\.\d+)?%?")


def _extract_numbers(text: str) -> list[tuple[str, float]]:
    results: list[tuple[str, float]] = []
    for raw in _NUMBER_RE.findall(text):
        cleaned = raw.replace(",", "").rstrip("%")
        try:
            results.append((raw, float(cleaned)))
        except ValueError:
            continue
    return results


class NumericSpecificityScorer(Scorer):
    """Checks concrete numeric values, tolerances, and required units."""

    name = "numeric_specificity"
    critical = False

    def score(
        self,
        case: EvalCase,
        response: ProviderResponse,
        config: dict[str, Any],
    ) -> ScoreResult:
        text = response.text or ""
        numbers = _extract_numbers(text)
        extracted_floats = [val for _, val in numbers]

        min_numbers = int(config.get("min_numbers", 1))
        expected_values = list(config.get("expected_values", []))
        tolerance = float(config.get("tolerance", 0.5))
        required_units = [str(u) for u in config.get("required_units", [])]

        failures: list[str] = []
        checks_total = 0
        checks_passed = 0

        # 1. Minimum numeric density.
        checks_total += 1
        if len(numbers) >= min_numbers:
            checks_passed += 1
        else:
            failures.append(
                f"Expected at least {min_numbers} concrete number(s), found {len(numbers)}"
            )

        # 2. Expected numeric targets within tolerance.
        matched_targets: list[Any] = []
        missing_targets: list[Any] = []
        for target in expected_values:
            checks_total += 1
            if isinstance(target, (int, float)):
                target_float = float(target)
                tol = tolerance
                if any(abs(val - target_float) <= tol for val in extracted_floats):
                    checks_passed += 1
                    matched_targets.append(target)
                else:
                    missing_targets.append(target)
                    failures.append(f"Missing expected value {target} (+/- {tol})")
            elif isinstance(target, dict):
                target_float = float(target["value"])
                tol = float(target.get("tolerance", tolerance))
                if any(abs(val - target_float) <= tol for val in extracted_floats):
                    checks_passed += 1
                    matched_targets.append(target_float)
                else:
                    missing_targets.append(target_float)
                    failures.append(f"Missing expected value {target_float} (+/- {tol})")
            else:
                target_str = str(target)
                if target_str.lower() in text.lower():
                    checks_passed += 1
                    matched_targets.append(target_str)
                else:
                    missing_targets.append(target_str)
                    failures.append(f"Missing expected numeric token '{target_str}'")

        # 3. Required units (ms, %, USD, days, etc.).
        lower_text = text.lower()
        for unit in required_units:
            checks_total += 1
            if unit.lower() in lower_text:
                checks_passed += 1
            else:
                failures.append(f"Missing required unit '{unit}'")

        score_val = round(checks_passed / checks_total, 4) if checks_total else 1.0
        passed = len(failures) == 0
        reason = (
            f"Verified {len(matched_targets)} expected numeric figure(s) across {len(numbers)} extracted values."
            if passed
            else "; ".join(failures)
        )
        return ScoreResult(
            name=self.name,
            passed=passed,
            score=score_val,
            reason=reason,
            critical=bool(config.get("critical", self.critical)),
            details={
                "extracted_numbers": [raw for raw, _ in numbers],
                "matched_targets": matched_targets,
                "missing_targets": missing_targets,
            },
        )
