"""Pluggable LLM-as-judge scorer using any configured Provider (including MockProvider)."""

from __future__ import annotations

from typing import Any

from agent_gate.models import EvalCase, ProviderResponse, ScoreResult
from agent_gate.providers.base import Provider
from agent_gate.scorers.base import Scorer
from agent_gate.scorers.schema import extract_json_payload


class LLMJudgeScorer(Scorer):
    """Grades nuanced responses against a rubric using a judge provider."""

    name = "llm_judge"
    critical = False

    def __init__(self, judge_provider: Provider) -> None:
        self.judge_provider = judge_provider

    def score(
        self,
        case: EvalCase,
        response: ProviderResponse,
        config: dict[str, Any],
    ) -> ScoreResult:
        rubric = str(config.get("rubric", "Grade accuracy, clarity, and adherence to constraints."))
        threshold = float(config.get("pass_threshold", 0.7))

        judge_prompt = (
            "You are a strict evaluation judge. Score the candidate response against the rubric.\n"
            "Return ONLY a JSON object with keys: score (0.0 to 1.0), passed (boolean), reason (string).\n\n"
            f"Original prompt:\n{case.prompt}\n\n"
            f"Rubric:\n{rubric}\n\n"
            f"Candidate response:\n{response.text}\n"
        )
        judge_case = EvalCase(
            case_id=f"{case.case_id}::judge",
            category="_llm_judge",
            prompt=judge_prompt,
            system_prompt="Return strict JSON with keys: score, passed, reason.",
            metadata={
                "candidate_response": response.text,
                "rubric_keywords": list(config.get("rubric_keywords", [])),
                "forbidden_keywords": list(config.get("forbidden_keywords", [])),
                "pass_threshold": threshold,
            },
        )
        judge_resp = self.judge_provider.complete(judge_case)
        parsed, err = extract_json_payload(judge_resp.text)
        if err or not isinstance(parsed, dict):
            return ScoreResult(
                name=self.name,
                passed=False,
                score=0.0,
                reason=f"Judge returned unparseable verdict: {err or 'not an object'}",
                critical=bool(config.get("critical", self.critical)),
                details={"raw_judge_output": judge_resp.text[:300]},
            )

        raw_score = float(parsed.get("score", 0.0))
        clamped_score = max(0.0, min(1.0, round(raw_score, 4)))
        passed = bool(parsed.get("passed", clamped_score >= threshold)) and clamped_score >= threshold
        reason = str(parsed.get("reason", "Judge completed evaluation."))
        return ScoreResult(
            name=self.name,
            passed=passed,
            score=clamped_score,
            reason=reason,
            critical=bool(config.get("critical", self.critical)),
            details={
                "judge_provider": self.judge_provider.provider_name,
                "judge_model": self.judge_provider.model,
                "threshold": threshold,
            },
        )
