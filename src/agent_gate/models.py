"""Core data models for suites, tool traces, provider outputs, and evaluation results."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ContextDocument:
    """A grounding document supplied to an evaluation case."""

    doc_id: str
    title: str
    content: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ContextDocument:
        return cls(
            doc_id=str(raw["id"]),
            title=str(raw.get("title", raw["id"])),
            content=str(raw["content"]),
        )


@dataclass(slots=True)
class ToolSpec:
    """Contract specification for an MCP tool available to an agent."""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    mutating: bool = False
    requires_approval: bool = False
    allowed: bool = True

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ToolSpec:
        return cls(
            name=str(raw["name"]),
            description=str(raw.get("description", "")),
            input_schema=dict(raw.get("input_schema", {})),
            mutating=bool(raw.get("mutating", False)),
            requires_approval=bool(raw.get("requires_approval", False)),
            allowed=bool(raw.get("allowed", True)),
        )


@dataclass(slots=True)
class ApprovalRecord:
    """Explicit human-in-the-loop approval decision for a mutating tool call."""

    tool_name: str
    approved: bool
    approver: str = "operator"
    arguments_match: dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ApprovalRecord:
        return cls(
            tool_name=str(raw["tool_name"]),
            approved=bool(raw.get("approved", False)),
            approver=str(raw.get("approver", "operator")),
            arguments_match=dict(raw.get("arguments_match", {})),
            reason=str(raw.get("reason", "")),
        )


@dataclass(slots=True)
class ToolCall:
    """A single MCP tool invocation attempted by the agent."""

    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str = ""
    status: str = "ok"  # ok | error | blocked
    output: Any = None
    error: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ToolCall:
        return cls(
            tool_name=str(raw["tool_name"]),
            arguments=dict(raw.get("arguments", {})),
            call_id=str(raw.get("call_id", "")),
            status=str(raw.get("status", "ok")),
            output=raw.get("output"),
            error=raw.get("error"),
        )


@dataclass(slots=True)
class EvalCase:
    """One test case inside a golden evaluation suite."""

    case_id: str
    category: str
    prompt: str
    system_prompt: str = ""
    context: list[ContextDocument] = field(default_factory=list)
    tools: list[ToolSpec] = field(default_factory=list)
    approvals: list[ApprovalRecord] = field(default_factory=list)
    sandbox_files: dict[str, str] = field(default_factory=dict)
    checks: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> EvalCase:
        context = [ContextDocument.from_dict(doc) for doc in raw.get("context", [])]
        tools = [ToolSpec.from_dict(t) for t in raw.get("tools", [])]
        approvals = [ApprovalRecord.from_dict(a) for a in raw.get("approvals", [])]
        sandbox_raw = raw.get("sandbox_files", {})
        sandbox_files = {str(k): str(v) for k, v in sandbox_raw.items()}
        return cls(
            case_id=str(raw["id"]),
            category=str(raw.get("category", "general")),
            prompt=str(raw["prompt"]),
            system_prompt=str(raw.get("system_prompt", "")),
            context=context,
            tools=tools,
            approvals=approvals,
            sandbox_files=sandbox_files,
            checks=dict(raw.get("checks", {})),
            tags=[str(t) for t in raw.get("tags", [])],
            metadata=dict(raw.get("metadata", {})),
        )


@dataclass(slots=True)
class EvalSuite:
    """Collection of evaluation cases loaded from YAML or JSONL."""

    name: str
    description: str = ""
    version: str = "1.0"
    default_system_prompt: str = ""
    cases: list[EvalCase] = field(default_factory=list)
    source_path: str = ""


@dataclass(slots=True)
class TokenUsage:
    """Prompt and completion token counts for a single call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TokenUsage:
        return cls(
            prompt_tokens=int(raw.get("prompt_tokens", 0)),
            completion_tokens=int(raw.get("completion_tokens", 0)),
        )


@dataclass(slots=True)
class ProviderResponse:
    """Normalized output returned by any model provider adapter."""

    text: str
    model: str
    provider: str
    latency_ms: float
    usage: TokenUsage = field(default_factory=TokenUsage)
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ScoreResult:
    """Outcome of a single deterministic or LLM-judge scorer."""

    name: str
    passed: bool
    score: float
    reason: str
    critical: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ScoreResult:
        return cls(
            name=str(raw["name"]),
            passed=bool(raw["passed"]),
            score=float(raw["score"]),
            reason=str(raw.get("reason", "")),
            critical=bool(raw.get("critical", False)),
            details=dict(raw.get("details", {})),
        )


@dataclass(slots=True)
class CaseResult:
    """Evaluation outcome for a single case."""

    case_id: str
    category: str
    tags: list[str]
    prompt: str
    response_text: str
    passed: bool
    score: float
    critical_passed: bool
    latency_ms: float
    usage: TokenUsage
    cost_usd: float
    scores: list[ScoreResult] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    sandbox_state: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> CaseResult:
        return cls(
            case_id=str(raw["case_id"]),
            category=str(raw.get("category", "general")),
            tags=[str(t) for t in raw.get("tags", [])],
            prompt=str(raw.get("prompt", "")),
            response_text=str(raw.get("response_text", "")),
            passed=bool(raw["passed"]),
            score=float(raw["score"]),
            critical_passed=bool(raw.get("critical_passed", True)),
            latency_ms=float(raw.get("latency_ms", 0.0)),
            usage=TokenUsage.from_dict(raw.get("usage", {})),
            cost_usd=float(raw.get("cost_usd", 0.0)),
            scores=[ScoreResult.from_dict(s) for s in raw.get("scores", [])],
            tool_calls=[ToolCall.from_dict(tc) for tc in raw.get("tool_calls", [])],
            sandbox_state={str(k): str(v) for k, v in raw.get("sandbox_state", {}).items()},
        )


@dataclass(slots=True)
class RunSummary:
    """Aggregated metrics across all cases in a run."""

    total_cases: int
    passed_cases: int
    pass_rate: float
    mean_score: float
    critical_total: int
    critical_passed: int
    critical_pass_rate: float
    total_prompt_tokens: int
    total_completion_tokens: int
    total_cost_usd: float
    cost_per_1k_runs_usd: float
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    by_category: dict[str, dict[str, float]] = field(default_factory=dict)
    by_scorer: dict[str, dict[str, float]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RunSummary:
        return cls(
            total_cases=int(raw["total_cases"]),
            passed_cases=int(raw["passed_cases"]),
            pass_rate=float(raw["pass_rate"]),
            mean_score=float(raw["mean_score"]),
            critical_total=int(raw.get("critical_total", 0)),
            critical_passed=int(raw.get("critical_passed", 0)),
            critical_pass_rate=float(raw.get("critical_pass_rate", 1.0)),
            total_prompt_tokens=int(raw.get("total_prompt_tokens", 0)),
            total_completion_tokens=int(raw.get("total_completion_tokens", 0)),
            total_cost_usd=float(raw.get("total_cost_usd", 0.0)),
            cost_per_1k_runs_usd=float(raw.get("cost_per_1k_runs_usd", 0.0)),
            mean_latency_ms=float(raw.get("mean_latency_ms", 0.0)),
            p50_latency_ms=float(raw.get("p50_latency_ms", 0.0)),
            p95_latency_ms=float(raw.get("p95_latency_ms", 0.0)),
            by_category=dict(raw.get("by_category", {})),
            by_scorer=dict(raw.get("by_scorer", {})),
        )


@dataclass(slots=True)
class RunResult:
    """Complete serializable record of evaluating one model against one suite."""

    run_id: str
    suite_name: str
    suite_path: str
    provider: str
    model: str
    created_at: str
    summary: RunSummary
    cases: list[CaseResult]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RunResult:
        return cls(
            run_id=str(raw["run_id"]),
            suite_name=str(raw["suite_name"]),
            suite_path=str(raw.get("suite_path", "")),
            provider=str(raw["provider"]),
            model=str(raw["model"]),
            created_at=str(raw.get("created_at", "")),
            summary=RunSummary.from_dict(raw["summary"]),
            cases=[CaseResult.from_dict(c) for c in raw.get("cases", [])],
            metadata=dict(raw.get("metadata", {})),
        )
