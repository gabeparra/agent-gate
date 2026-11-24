# agent-gate

**Deterministic evaluation, MCP tool guardrails, and CI regression harness for LLM agents.**

`agent-gate` is a lightweight Python CLI and library for testing production agents before shipping prompt, model, or tool changes. Instead of relying on spot checks or subjective grading alone, it pairs deterministic checks, sandboxed MCP tool verification, cost/latency accounting, side-by-side HTML comparison reports, and a strict GitHub Actions regression gate.

---

## Why This Exists

Most agent failures in production are not subtle style issues. They are concrete operational bugs:

- Fabricated citations or unsupported numbers in RAG answers
- Broken JSON payloads that crash downstream parsers
- Prompt injections hidden inside support tickets or retrieved documents
- Agents calling disallowed tools, escaping workspace paths, or mutating state without human approval
- Dishonest completions where the agent says "Done! I updated the file" even though the write tool failed or was never called

`agent-gate` turns those failure modes into repeatable, version-controlled assertions that run offline in CI at zero API cost and against live providers before a release.

---

## Architecture

```
  YAML / JSONL Golden Suite
              |
              v
  +-----------------------+     +----------------------------------------+
  |   Provider Adapter    | --> | OpenAI / Azure AI Foundry / OpenRouter |
  |   (Unified Contract)  |     | Anthropic / Ollama / llama.cpp / Mock  |
  +-----------------------+     +----------------------------------------+
              |
              v
  +-----------------------+     +----------------------------------------+
  |   Workspace Sandbox   | --> | MCP Allowlist + JSON Schema + Arg Safe |
  |   (Tempdir Replay)    |     | HITL Write Approval Gate               |
  +-----------------------+     +----------------------------------------+
              |
              v
  +-----------------------+     +----------------------------------------+
  |  Deterministic + LLM  | --> | Grounding, JSON Schema, Refusal/Hedge  |
  |       Scorers         |     | Numeric, Prompt Injection, Honesty     |
  +-----------------------+     +----------------------------------------+
              |
              v
  +-----------------------+     +----------------------------------------+
  |  Metrics & Reporting  | --> | Side-by-Side Static HTML + JSON Diff   |
  |    + CI Gate CLI      |     | Non-Zero Exit on Regression / Safety   |
  +-----------------------+     +----------------------------------------+
```

---

## 60-Second Offline Quickstart

No API keys are required. The built-in `mock` provider ships with deterministic `baseline`, `candidate`, and `regressed` profiles so you can run the full pipeline locally right away.

```bash
# 1. Install dependencies with uv
uv sync --group dev

# 2. Run the unit and integration test suite
uv run pytest

# 3. Run the core eval suite and the agentic MCP guardrails suite
uv run agent-gate run datasets/core_evals.yaml --provider mock --profile candidate --out artifacts/core_candidate.json
uv run agent-gate run datasets/core_evals.yaml --provider mock --profile regressed --out artifacts/core_regressed.json
uv run agent-gate run datasets/agentic_guardrails.yaml --provider mock --profile candidate --out artifacts/agentic_candidate.json
uv run agent-gate run datasets/agentic_guardrails.yaml --provider mock --profile regressed --out artifacts/agentic_regressed.json

# 4. Generate a self-contained side-by-side HTML comparison report
uv run agent-gate compare \
  baselines/agentic_guardrails.baseline.json \
  artifacts/agentic_candidate.json \
  artifacts/agentic_regressed.json \
  --html-out artifacts/agentic_comparison.html \
  --json-out artifacts/agentic_diff.json

# 5. Run the CI regression gate (exit 0 on pass, exit 1 on regression)
uv run agent-gate gate \
  --baseline baselines/agentic_guardrails.baseline.json \
  --candidate artifacts/agentic_candidate.json

# Try gating the regressed profile to see a failing PR summary:
uv run agent-gate gate \
  --baseline baselines/agentic_guardrails.baseline.json \
  --candidate artifacts/agentic_regressed.json
```

Open `artifacts/agentic_comparison.html` in any browser. It has zero external CDN dependencies and includes interactive filters for regressions, improvements, critical safety failures, and categories.

---

## Multi-Provider Configuration

All providers share one normalized `Provider` contract (`complete(case) -> ProviderResponse`) and read credentials strictly from environment variables.

| Provider | CLI Flag | Environment Variables |
| --- | --- | --- |
| Offline Mock | `--provider mock` | None (`--profile baseline\|candidate\|regressed` or `--mock-fixture file.yaml`) |
| OpenAI | `--provider openai` | `OPENAI_API_KEY` (optional `OPENAI_BASE_URL`, `OPENAI_MODEL`) |
| Azure AI Foundry | `--provider azure-foundry` | `AZURE_AI_API_KEY`, `AZURE_AI_ENDPOINT` (optional `AZURE_AI_API_VERSION`) |
| Anthropic | `--provider anthropic` | `ANTHROPIC_API_KEY` (optional `ANTHROPIC_MODEL`) |
| OpenRouter | `--provider openrouter` | `OPENROUTER_API_KEY` (optional `OPENROUTER_MODEL`) |
| Ollama (local) | `--provider ollama` | Optional `OLLAMA_BASE_URL` (default `http://127.0.0.1:11434/v1`) |
| llama.cpp (local) | `--provider llamacpp` | Optional `LLAMACPP_BASE_URL` (default `http://127.0.0.1:8080/v1`) |

Example comparing a local Ollama model against OpenAI on the same suite:

```bash
export OPENAI_API_KEY="..."
uv run agent-gate run datasets/core_evals.yaml --provider openai --model gpt-4o-mini --out artifacts/gpt4omini.json
uv run agent-gate run datasets/core_evals.yaml --provider ollama --model llama3.1:8b --out artifacts/llama31.json
uv run agent-gate compare artifacts/gpt4omini.json artifacts/llama31.json --html-out artifacts/model_bakeoff.html
```

---

## Scorers, Guardrails, and Tradeoffs

Every scorer returns a `ScoreResult(name, passed, score, reason, critical, details)`. Security and state-integrity checks are marked `critical=True` and enforce a hard safety floor in `agent-gate gate`.

| Check | Critical | What It Verifies | Engineering Tradeoff |
| --- | :---: | --- | --- |
| `citation_grounding` | No | Inline `[doc-id]` tags map to real context documents (zero fabricated IDs), required facts appear, and lexical overlap meets threshold. | Fast, zero-cost, and deterministic. Lexical overlap catches gross drift, though subtle semantic paraphrasing still benefits from pairing with `llm_judge`. |
| `json_schema` | No | Extracts raw or fenced JSON and validates against JSON Schema Draft 2020-12 with exact path-level diagnostics. | Guarantees downstream contract compatibility. Set `strict_raw_json: true` when consumers do not strip markdown fences. |
| `numeric_specificity` | No | Extracts figures, checks expected values within configurable tolerance, and verifies required units (`ms`, `%`, `days`). | Prevents vague hand-waving on operational telemetry. Tolerance avoids brittle rounding failures. |
| `refusal_hedging` | No | Confirms calibrated refusal on unanswerable/unsafe asks without fabricating dates or facts, and flags vague hedging on answerable asks. | Regex heuristics are transparent and fast, though unusual refusal phrasing may occasionally require tuning patterns. |
| `prompt_injection` | Yes | Verifies untrusted context cannot leak canary/secret tokens, trigger forbidden tools, or derail the legitimate user task. | Deterministic canary and task-preservation checks catch both direct hijacks and silent task drops without needing a second model call. |
| `mcp_contract` | Yes | Enforces MCP tool allowlists, JSON Schema input validation, path-traversal/shell/SSRF/secret argument safety, and HITL approval records for mutating tools. | Blocks unsafe tool calls before execution in the sandbox. Requires declaring explicit `ToolSpec` contracts in the suite. |
| `action_honesty` | Yes | Replays tool calls in an isolated temporary directory and compares final prose claims against real disk mutations and tool error/block statuses. | Catches hallucinated writes and concealed tool failures that text-only graders miss completely. Prose claim detection focuses on explicit file/action assertions. |
| `llm_judge` | No | Plugs any configured provider (or offline mock judge) into a structured JSON rubric prompt. | Useful for nuanced criteria that rules cannot capture; kept optional and separate from critical deterministic guardrails to avoid judge variance in CI. |

---

## CI Regression Gate

`agent-gate gate` compares a candidate `RunResult` against a committed baseline JSON and exits `1` if any policy fails:

- `--max-score-drop` (default `0.02`): blocks overall quality regressions
- `--max-pass-rate-drop` (default `0.02`): blocks case pass-rate regressions
- `--min-safety-rate` (default `1.0`): enforces zero tolerance on critical checks (`prompt_injection`, `mcp_contract`, `action_honesty`)
- `--max-p95-ms` and `--max-cost-per-1k`: optional latency and spend budgets
- `--markdown-out`: writes a GitHub-flavored Markdown summary table for `$GITHUB_STEP_SUMMARY` or PR comments

The workflow in [.github/workflows/eval-gate.yml](.github/workflows/eval-gate.yml) runs `ruff`, `pytest`, offline eval suites, HTML comparison generation, and baseline gating on every push and pull request without needing any repository secrets.

---

## Project Layout

```
agent-gate/
├── .github/workflows/eval-gate.yml   # Offline CI lint, test, report, and regression gate
├── baselines/                        # Committed baseline RunResult JSON artifacts
├── datasets/
│   ├── core_evals.yaml               # Grounding, refusal, JSON schema, numeric, injection
│   └── agentic_guardrails.yaml       # MCP allowlist, HITL approval, path safety, action honesty
├── src/agent_gate/
│   ├── cli.py                        # CLI commands: run, compare, gate
│   ├── gate.py                       # Policy evaluator and Markdown PR summary generator
│   ├── loader.py                     # YAML and JSONL suite loader
│   ├── metrics.py                    # Token pricing table, cost math, p50/p95 latency
│   ├── models.py                     # Core dataclasses and JSON serialization
│   ├── runner.py                     # Suite runner with sandbox replay and scorer orchestration
│   ├── guardrails/
│   │   ├── mcp_contract.py           # Allowlist, schema, arg safety, HITL approval gates
│   │   ├── sandbox.py                # Isolated tempdir workspace replay
│   │   └── honesty.py                # Disk and tool-log action-honesty verifier
│   ├── providers/
│   │   ├── openai_compat.py          # OpenAI, Azure AI Foundry, OpenRouter, Ollama, llama.cpp
│   │   ├── anthropic.py              # Anthropic Messages API adapter
│   │   ├── mock.py                   # Offline deterministic provider with comparison profiles
│   │   └── registry.py               # Provider factory
│   ├── report/
│   │   ├── compare.py                # Multi-model delta and regression calculator
│   │   └── html_report.py            # Self-contained static HTML report generator
│   └── scorers/                      # Grounding, JSON schema, refusal, numeric, injection, judge
└── tests/                            # Pytest suite covering all modules end-to-end
```

---

## License

MIT. See [LICENSE](LICENSE).
