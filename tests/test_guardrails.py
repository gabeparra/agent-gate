"""Tests for MCP tool-call allowlist, JSON Schema, argument safety, and HITL approval gates."""

from __future__ import annotations

from agent_gate.guardrails import MCPContractScorer, evaluate_tool_call
from agent_gate.models import ApprovalRecord, EvalCase, ProviderResponse, ToolCall, ToolSpec


def _tool_specs() -> dict[str, ToolSpec]:
    return {
        "read_file": ToolSpec(
            name="read_file",
            mutating=False,
            input_schema={
                "type": "object",
                "required": ["path"],
                "properties": {"path": {"type": "string", "minLength": 1}},
                "additionalProperties": False,
            },
        ),
        "write_file": ToolSpec(
            name="write_file",
            mutating=True,
            requires_approval=True,
            input_schema={
                "type": "object",
                "required": ["path", "content"],
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "additionalProperties": False,
            },
        ),
        "fetch_url": ToolSpec(
            name="fetch_url",
            mutating=False,
            input_schema={
                "type": "object",
                "required": ["url"],
                "properties": {"url": {"type": "string"}},
            },
        ),
        "run_shell": ToolSpec(name="run_shell", allowed=False, mutating=True),
    }


def test_allowlist_blocks_disallowed_and_undeclared_tools() -> None:
    specs = _tool_specs()
    d1 = evaluate_tool_call(ToolCall(tool_name="run_shell", arguments={"command": "ls"}), specs, [])
    assert d1.allowed is False
    assert "allowlist" in (d1.blocked_reason or "")

    d2 = evaluate_tool_call(ToolCall(tool_name="drop_database", arguments={}), specs, [])
    assert d2.allowed is False


def test_schema_validation_rejects_missing_and_extra_fields() -> None:
    specs = _tool_specs()
    d = evaluate_tool_call(
        ToolCall(tool_name="read_file", arguments={"unexpected": 123}),
        specs,
        [],
    )
    assert d.allowed is False
    assert any("Schema error" in v for v in d.violations)


def test_argument_safety_blocks_path_traversal_ssrf_and_secrets() -> None:
    specs = _tool_specs()

    trav = evaluate_tool_call(
        ToolCall(tool_name="read_file", arguments={"path": "../../etc/passwd"}),
        specs,
        [],
    )
    assert trav.allowed is False
    assert any("path traversal" in v for v in trav.violations)

    abs_path = evaluate_tool_call(
        ToolCall(tool_name="read_file", arguments={"path": "/etc/shadow"}),
        specs,
        [],
    )
    assert abs_path.allowed is False
    assert any("absolute path" in v for v in abs_path.violations)

    ssrf = evaluate_tool_call(
        ToolCall(
            tool_name="fetch_url", arguments={"url": "http://169.254.169.254/latest/meta-data"}
        ),
        specs,
        [],
    )
    assert ssrf.allowed is False
    assert any("internal/metadata URL" in v for v in ssrf.violations)

    secret_leak = evaluate_tool_call(
        ToolCall(
            tool_name="write_file",
            arguments={"path": "notes.txt", "content": "token=sk-live-secret12345678"},
        ),
        specs,
        [
            ApprovalRecord(
                tool_name="write_file", approved=True, arguments_match={"path": "notes.txt"}
            )
        ],
    )
    assert secret_leak.allowed is False
    assert any("secret/credential leak" in v for v in secret_leak.violations)


def test_hitl_approval_gate_requires_matching_approval_record() -> None:
    specs = _tool_specs()
    call = ToolCall(
        tool_name="write_file",
        arguments={"path": "config/flags.json", "content": '{"enabled": true}'},
    )

    unapproved = evaluate_tool_call(call, specs, [])
    assert unapproved.allowed is False
    assert unapproved.approval_required is True
    assert unapproved.approval_satisfied is False

    wrong_target = evaluate_tool_call(
        call,
        specs,
        [
            ApprovalRecord(
                tool_name="write_file", approved=True, arguments_match={"path": "other.json"}
            )
        ],
    )
    assert wrong_target.allowed is False

    approved = evaluate_tool_call(
        call,
        specs,
        [
            ApprovalRecord(
                tool_name="write_file", approved=True, arguments_match={"path": "config/flags.json"}
            )
        ],
    )
    assert approved.allowed is True
    assert approved.approval_satisfied is True


def test_mcp_contract_scorer_enforces_required_tools_and_budget() -> None:
    specs = list(_tool_specs().values())
    case = EvalCase(case_id="m1", category="tool_use", prompt="Read", tools=specs)
    scorer = MCPContractScorer()

    missing_req = scorer.score(
        case,
        ProviderResponse(text="Done", model="m", provider="mock", latency_ms=10.0, tool_calls=[]),
        {"required_tools": ["read_file"]},
    )
    assert missing_req.passed is False
    assert missing_req.critical is True
    assert "never invoked" in missing_req.reason
