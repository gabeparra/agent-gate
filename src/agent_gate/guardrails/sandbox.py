"""Isolated tempdir workspace sandbox for replaying MCP tool calls against real disk state."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_gate.guardrails.mcp_contract import evaluate_tool_call
from agent_gate.models import ApprovalRecord, EvalCase, ToolCall, ToolSpec


@dataclass(slots=True)
class SandboxReport:
    """Ground-truth record of disk state and executed tool calls after sandbox replay."""

    initial_files: dict[str, str] = field(default_factory=dict)
    final_files: dict[str, str] = field(default_factory=dict)
    created_files: list[str] = field(default_factory=list)
    modified_files: list[str] = field(default_factory=list)
    deleted_files: list[str] = field(default_factory=list)
    executed_calls: list[ToolCall] = field(default_factory=list)
    tickets_created: list[dict[str, Any]] = field(default_factory=list)

    @property
    def mutated_files(self) -> set[str]:
        return set(self.created_files) | set(self.modified_files) | set(self.deleted_files)


class WorkspaceSandbox:
    """Temporary directory sandbox that executes allowed MCP tools and blocks unsafe ones."""

    def __init__(
        self,
        initial_files: dict[str, str],
        tool_specs: list[ToolSpec],
        approvals: list[ApprovalRecord],
        *,
        require_approval_for_mutating: bool = True,
        blocked_tools: set[str] | None = None,
    ) -> None:
        self.initial_files = {k.replace("\\", "/"): v for k, v in initial_files.items()}
        self.tool_specs = {spec.name: spec for spec in tool_specs}
        self.approvals = list(approvals)
        self.require_approval_for_mutating = require_approval_for_mutating
        self.blocked_tools = set(blocked_tools or set())

    @classmethod
    def from_case(cls, case: EvalCase) -> WorkspaceSandbox:
        mcp_cfg = case.checks.get("mcp_contract")
        req_approval = True
        blocked: set[str] = set()
        if isinstance(mcp_cfg, dict):
            req_approval = bool(mcp_cfg.get("require_approval_for_mutating", True))
            blocked = {str(t) for t in mcp_cfg.get("forbidden_tools", [])}
        return cls(
            initial_files=case.sandbox_files,
            tool_specs=case.tools,
            approvals=case.approvals,
            require_approval_for_mutating=req_approval,
            blocked_tools=blocked,
        )

    def replay(self, tool_calls: list[ToolCall]) -> SandboxReport:
        """Seed temp directory, replay tool calls with guardrail enforcement, and snapshot disk."""
        with tempfile.TemporaryDirectory(prefix="agent-gate-sandbox-") as tmp_dir:
            root = Path(tmp_dir).resolve()
            self._seed_files(root)

            executed: list[ToolCall] = []
            tickets: list[dict[str, Any]] = []
            for raw_call in tool_calls:
                call = ToolCall(
                    tool_name=raw_call.tool_name,
                    arguments=dict(raw_call.arguments),
                    call_id=raw_call.call_id,
                )
                decision = evaluate_tool_call(
                    call,
                    self.tool_specs,
                    self.approvals,
                    require_approval_for_mutating=self.require_approval_for_mutating,
                    blocked_tools=self.blocked_tools,
                )
                if not decision.allowed:
                    call.status = "blocked"
                    call.error = decision.blocked_reason
                    executed.append(call)
                    continue

                self._execute_tool(root, call, tickets)
                executed.append(call)

            final_files = self._snapshot_files(root)

        created = sorted(p for p in final_files if p not in self.initial_files)
        deleted = sorted(p for p in self.initial_files if p not in final_files)
        modified = sorted(
            p
            for p in final_files
            if p in self.initial_files and final_files[p] != self.initial_files[p]
        )
        return SandboxReport(
            initial_files=dict(self.initial_files),
            final_files=final_files,
            created_files=created,
            modified_files=modified,
            deleted_files=deleted,
            executed_calls=executed,
            tickets_created=tickets,
        )

    def _seed_files(self, root: Path) -> None:
        for rel_path, content in self.initial_files.items():
            target = self._safe_resolve(root, rel_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

    def _safe_resolve(self, root: Path, rel_path: str) -> Path:
        candidate = (root / rel_path).resolve()
        if not candidate.is_relative_to(root):
            raise ValueError(f"Resolved path escapes sandbox root: {rel_path}")
        return candidate

    def _execute_tool(
        self,
        root: Path,
        call: ToolCall,
        tickets: list[dict[str, Any]],
    ) -> None:
        name = call.tool_name
        args = call.arguments

        try:
            if name == "read_file":
                target = self._safe_resolve(root, str(args["path"]))
                if not target.is_file():
                    call.status = "error"
                    call.error = f"File not found: {args['path']}"
                    return
                call.status = "ok"
                call.output = target.read_text(encoding="utf-8")
                return

            if name == "write_file":
                target = self._safe_resolve(root, str(args["path"]))
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(str(args["content"]), encoding="utf-8")
                call.status = "ok"
                call.output = {"written": str(args["path"]), "bytes": len(str(args["content"]))}
                return

            if name == "delete_file":
                target = self._safe_resolve(root, str(args["path"]))
                if not target.exists():
                    call.status = "error"
                    call.error = f"Cannot delete missing file: {args['path']}"
                    return
                target.unlink()
                call.status = "ok"
                call.output = {"deleted": str(args["path"])}
                return

            if name == "list_files":
                prefix = str(args.get("directory", "")).strip()
                base = self._safe_resolve(root, prefix) if prefix else root
                if not base.exists():
                    call.status = "error"
                    call.error = f"Directory not found: {prefix}"
                    return
                files = sorted(
                    p.relative_to(root).as_posix() for p in base.rglob("*") if p.is_file()
                )
                call.status = "ok"
                call.output = files
                return

            if name == "create_ticket":
                ticket = {"id": f"TKT-{len(tickets) + 101}", **args}
                tickets.append(ticket)
                call.status = "ok"
                call.output = ticket
                return

            # Unknown allowed mock tool: record as cleanly executed no-op.
            call.status = "ok"
            call.output = {"executed": name}
        except (KeyError, ValueError, OSError) as exc:
            call.status = "error"
            call.error = str(exc)

    def _snapshot_files(self, root: Path) -> dict[str, str]:
        snapshot: dict[str, str] = {}
        for path in sorted(root.rglob("*")):
            if path.is_file():
                rel = path.relative_to(root).as_posix()
                snapshot[rel] = path.read_text(encoding="utf-8")
        return snapshot
