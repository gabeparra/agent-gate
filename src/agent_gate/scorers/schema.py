"""JSON extraction and JSON Schema compliance scorer."""

from __future__ import annotations

import json
import re
from typing import Any

from jsonschema import Draft202012Validator

from agent_gate.models import EvalCase, ProviderResponse, ScoreResult
from agent_gate.scorers.base import Scorer

_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL | re.IGNORECASE)


def extract_json_payload(text: str) -> tuple[Any | None, str | None]:
    """Parse JSON from raw response text or a fenced markdown JSON block."""
    stripped = text.strip()
    if not stripped:
        return None, "Response is empty."

    try:
        return json.loads(stripped), None
    except json.JSONDecodeError:
        pass

    match = _FENCED_JSON_RE.search(stripped)
    if match:
        try:
            return json.loads(match.group(1)), None
        except json.JSONDecodeError as exc:
            return None, f"Invalid JSON inside code fence: {exc.msg}"

    # Last-resort balanced brace extraction for responses with brief surrounding prose.
    first_brace = stripped.find("{")
    last_brace = stripped.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        try:
            return json.loads(stripped[first_brace : last_brace + 1]), None
        except json.JSONDecodeError as exc:
            return None, f"JSON parse failed: {exc.msg}"

    return None, "No valid JSON payload found in response."


class JsonSchemaScorer(Scorer):
    """Validates structured model output against a JSON Schema contract."""

    name = "json_schema"
    critical = False

    def score(
        self,
        case: EvalCase,
        response: ProviderResponse,
        config: dict[str, Any],
    ) -> ScoreResult:
        schema = config.get("schema")
        strict_raw = bool(config.get("strict_raw_json", False))
        text = response.text or ""

        if strict_raw:
            try:
                parsed = json.loads(text.strip())
                err = None
            except json.JSONDecodeError as exc:
                parsed, err = None, f"Strict raw JSON parse failed: {exc.msg}"
        else:
            parsed, err = extract_json_payload(text)

        if err is not None:
            return ScoreResult(
                name=self.name,
                passed=False,
                score=0.0,
                reason=err,
                critical=bool(config.get("critical", self.critical)),
                details={"parse_error": err},
            )

        if not isinstance(schema, dict) or not schema:
            return ScoreResult(
                name=self.name,
                passed=True,
                score=1.0,
                reason="Response parsed as valid JSON.",
                critical=bool(config.get("critical", self.critical)),
            )

        validator = Draft202012Validator(schema)
        errors = sorted(validator.iter_errors(parsed), key=lambda e: list(e.path))
        if not errors:
            return ScoreResult(
                name=self.name,
                passed=True,
                score=1.0,
                reason="JSON matches schema contract.",
                critical=bool(config.get("critical", self.critical)),
            )

        formatted_errors = []
        for err_obj in errors[:5]:
            loc = ".".join(str(p) for p in err_obj.path) or "$"
            formatted_errors.append(f"{loc}: {err_obj.message}")

        # Partial credit if JSON parsed cleanly and only a subset of required fields failed.
        required_count = max(len(schema.get("required", [])), len(schema.get("properties", {})), 1)
        partial = max(0.0, round(1.0 - (len(errors) / (required_count + 1)), 4))
        return ScoreResult(
            name=self.name,
            passed=False,
            score=partial,
            reason=f"Schema validation failed ({len(errors)} error(s)): {'; '.join(formatted_errors)}",
            critical=bool(config.get("critical", self.critical)),
            details={"errors": formatted_errors},
        )
