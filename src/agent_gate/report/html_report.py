"""Self-contained static HTML generator for side-by-side evaluation comparison reports."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from agent_gate.report.compare import CaseComparison, ComparisonReport, ModelComparison


def render_comparison_html(report: ComparisonReport) -> str:
    """Render a zero-CDN static HTML comparison report with interactive filtering."""
    title = f"agent-gate Comparison Report: {report.suite_name}"
    cards_html = "\n".join(_render_model_card(m) for m in report.models)
    scorer_table_html = _render_scorer_matrix(report)
    case_rows_html = "\n".join(
        _render_case_row(c, [m.label for m in report.models]) for c in report.cases
    )
    cat_options = "\n".join(
        f'<option value="{html.escape(cat)}">{html.escape(cat)}</option>'
        for cat in report.all_categories
    )
    baseline_tag = ' <span class="tag tag-base">baseline</span>'
    model_headers = "\n".join(
        f"<th>{html.escape(m.label)}{baseline_tag if m.is_baseline else ''}</th>"
        for m in report.models
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #0f1117;
      --surface: #171a23;
      --surface-2: #1f2430;
      --border: #2c3344;
      --text: #e6e9f2;
      --muted: #98a2b8;
      --pass: #22c55e;
      --pass-bg: rgba(34, 197, 94, 0.14);
      --fail: #ef4444;
      --fail-bg: rgba(239, 68, 68, 0.16);
      --warn: #f59e0b;
      --warn-bg: rgba(245, 158, 11, 0.16);
      --accent: #38bdf8;
      --accent-bg: rgba(56, 189, 248, 0.14);
      --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 28px 24px 56px;
      background: var(--bg);
      color: var(--text);
      font-family: var(--font-sans);
      line-height: 1.5;
    }}
    .container {{ max-width: 1280px; margin: 0 auto; }}
    header {{
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: baseline;
      gap: 12px;
      border-bottom: 1px solid var(--border);
      padding-bottom: 16px;
      margin-bottom: 24px;
    }}
    h1 {{ margin: 0; font-size: 1.5rem; letter-spacing: -0.02em; }}
    h2 {{ margin: 28px 0 12px; font-size: 1.1rem; color: var(--text); }}
    .subtitle {{ color: var(--muted); font-size: 0.9rem; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 16px;
    }}
    .card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 16px 18px;
    }}
    .card-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      margin-bottom: 12px;
    }}
    .model-title {{ font-weight: 600; font-family: var(--font-mono); font-size: 0.92rem; }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 10px 14px;
    }}
    .kpi-label {{ font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }}
    .kpi-val {{ font-size: 1.15rem; font-weight: 600; font-family: var(--font-mono); }}
    .delta {{ font-size: 0.78rem; font-family: var(--font-mono); margin-left: 6px; }}
    .delta-pos {{ color: var(--pass); }}
    .delta-neg {{ color: var(--fail); }}
    .delta-flat {{ color: var(--muted); }}
    .tag {{
      display: inline-block;
      padding: 2px 7px;
      border-radius: 999px;
      font-size: 0.72rem;
      font-weight: 600;
      font-family: var(--font-mono);
      text-transform: uppercase;
    }}
    .tag-pass {{ background: var(--pass-bg); color: var(--pass); }}
    .tag-fail {{ background: var(--fail-bg); color: var(--fail); }}
    .tag-warn {{ background: var(--warn-bg); color: var(--warn); }}
    .tag-base {{ background: var(--accent-bg); color: var(--accent); }}
    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      margin: 14px 0;
    }}
    .filter-btn {{
      background: var(--surface);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 6px 12px;
      font-size: 0.85rem;
      cursor: pointer;
    }}
    .filter-btn.active {{
      background: var(--accent-bg);
      border-color: var(--accent);
      color: var(--accent);
    }}
    select {{
      background: var(--surface);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 6px 10px;
      font-size: 0.85rem;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 10px;
      overflow: hidden;
      font-size: 0.88rem;
    }}
    th, td {{
      text-align: left;
      padding: 10px 12px;
      border-bottom: 1px solid var(--border);
      vertical-align: top;
    }}
    th {{
      background: var(--surface-2);
      font-weight: 600;
      color: var(--muted);
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .bar-track {{
      width: 100%;
      height: 7px;
      background: var(--surface-2);
      border-radius: 4px;
      overflow: hidden;
      margin-top: 4px;
    }}
    .bar-fill {{ height: 100%; border-radius: 4px; }}
    .case-id {{ font-family: var(--font-mono); font-weight: 600; }}
    .prompt-preview {{ color: var(--muted); font-size: 0.82rem; margin-top: 4px; max-width: 340px; }}
    details {{ margin-top: 8px; }}
    summary {{ cursor: pointer; color: var(--accent); font-size: 0.8rem; user-select: none; }}
    pre {{
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 8px 10px;
      font-family: var(--font-mono);
      font-size: 0.78rem;
      white-space: pre-wrap;
      word-break: break-word;
      margin: 6px 0;
      max-height: 220px;
      overflow-y: auto;
    }}
    .scorer-list {{ list-style: none; padding: 0; margin: 6px 0; font-size: 0.8rem; }}
    .scorer-list li {{ margin-bottom: 4px; }}
    .tool-pill {{
      display: inline-block;
      margin: 2px 4px 2px 0;
      padding: 2px 6px;
      border-radius: 4px;
      font-family: var(--font-mono);
      font-size: 0.74rem;
      border: 1px solid var(--border);
      background: var(--surface-2);
    }}
    .hidden {{ display: none; }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <h1>agent-gate Comparison Report</h1>
        <div class="subtitle">Suite: <strong>{html.escape(report.suite_name)}</strong> | Baseline: <code>{html.escape(report.baseline_label)}</code> | Cases: {len(report.cases)}</div>
      </div>
    </header>

    <section class="grid">
      {cards_html}
    </section>

    <h2>Scorer Breakdown</h2>
    {scorer_table_html}

    <h2>Side-by-Side Case Diagnostics</h2>
    <div class="controls">
      <button class="filter-btn active" data-filter="all">All ({len(report.cases)})</button>
      <button class="filter-btn" data-filter="regressed">Regressions</button>
      <button class="filter-btn" data-filter="improved">Improvements</button>
      <button class="filter-btn" data-filter="critical">Critical Failures</button>
      <select id="categoryFilter">
        <option value="all">All categories</option>
        {cat_options}
      </select>
    </div>

    <table id="casesTable">
      <thead>
        <tr>
          <th style="width: 26%">Case</th>
          {model_headers}
        </tr>
      </thead>
      <tbody>
        {case_rows_html}
      </tbody>
    </table>
  </div>

  <script>
    (function() {{
      let activeStatus = "all";
      const buttons = document.querySelectorAll(".filter-btn");
      const catSelect = document.getElementById("categoryFilter");
      const rows = document.querySelectorAll("#casesTable tbody tr");

      function applyFilters() {{
        const cat = catSelect.value;
        rows.forEach((row) => {{
          const rowStatus = row.dataset.status;
          const rowCat = row.dataset.category;
          const isCrit = row.dataset.critical === "true";
          const statusOk =
            activeStatus === "all" ||
            (activeStatus === "critical" ? isCrit : rowStatus === activeStatus);
          const catOk = cat === "all" || rowCat === cat;
          row.classList.toggle("hidden", !(statusOk && catOk));
        }});
      }}

      buttons.forEach((btn) => {{
        btn.addEventListener("click", () => {{
          buttons.forEach((b) => b.classList.remove("active"));
          btn.classList.add("active");
          activeStatus = btn.dataset.filter;
          applyFilters();
        }});
      }});
      catSelect.addEventListener("change", applyFilters);
    }})();
  </script>
</body>
</html>
"""


def _fmt_delta(val: float, *, invert: bool = False, suffix: str = "", precision: int = 1) -> str:
    if abs(val) < 1e-6:
        return '<span class="delta delta-flat">0</span>'
    good = (val < 0) if invert else (val > 0)
    cls = "delta-pos" if good else "delta-neg"
    sign = "+" if val > 0 else ""
    return f'<span class="delta {cls}">{sign}{val:.{precision}f}{suffix}</span>'


def _render_model_card(m: ModelComparison) -> str:
    badge = (
        '<span class="tag tag-base">baseline</span>'
        if m.is_baseline
        else (
            f'<span class="tag tag-fail">{m.regressions} regressed</span>'
            if m.regressions > 0
            else '<span class="tag tag-pass">0 regressions</span>'
        )
    )
    pass_delta = "" if m.is_baseline else _fmt_delta(m.pass_rate_delta * 100.0, suffix="%", precision=1)
    score_delta = "" if m.is_baseline else _fmt_delta(m.mean_score_delta, precision=3)
    lat_delta = "" if m.is_baseline else _fmt_delta(m.p95_latency_delta_ms, invert=True, suffix="ms", precision=0)
    cost_delta = "" if m.is_baseline else _fmt_delta(m.cost_per_1k_delta_usd, invert=True, suffix="", precision=4)

    safety_cls = "tag-pass" if m.critical_pass_rate >= 1.0 else "tag-fail"
    return f"""
      <div class="card">
        <div class="card-header">
          <span class="model-title">{html.escape(m.label)}</span>
          {badge}
        </div>
        <div class="kpi-grid">
          <div>
            <div class="kpi-label">Pass Rate</div>
            <div class="kpi-val">{m.pass_rate * 100:.1f}%{pass_delta}</div>
          </div>
          <div>
            <div class="kpi-label">Mean Score</div>
            <div class="kpi-val">{m.mean_score:.3f}{score_delta}</div>
          </div>
          <div>
            <div class="kpi-label">Critical Safety</div>
            <div class="kpi-val"><span class="tag {safety_cls}">{m.critical_pass_rate * 100:.1f}%</span></div>
          </div>
          <div>
            <div class="kpi-label">p95 Latency</div>
            <div class="kpi-val">{m.p95_latency_ms:.0f} ms{lat_delta}</div>
          </div>
          <div>
            <div class="kpi-label">Est. Cost / 1k Runs</div>
            <div class="kpi-val">${m.cost_per_1k_runs_usd:.4f}{cost_delta}</div>
          </div>
          <div>
            <div class="kpi-label">Tokens</div>
            <div class="kpi-val">{m.total_tokens:,}</div>
          </div>
        </div>
      </div>"""


def _render_scorer_matrix(report: ComparisonReport) -> str:
    headers = "".join(f"<th>{html.escape(m.label)}</th>" for m in report.models)
    rows: list[str] = []
    for scorer_name in report.all_scorers:
        cells: list[str] = []
        for m in report.models:
            stats = m.by_scorer.get(scorer_name)
            if not stats:
                cells.append('<td class="subtitle">n/a</td>')
                continue
            rate = float(stats["pass_rate"]) * 100.0
            color = "var(--pass)" if rate >= 99.9 else ("var(--warn)" if rate >= 70.0 else "var(--fail)")
            cells.append(
                f"<td>"
                f"<div><strong>{rate:.0f}%</strong> ({int(stats['passed'])}/{int(stats['total'])})</div>"
                f'<div class="bar-track"><div class="bar-fill" style="width:{rate:.1f}%;background:{color}"></div></div>'
                f"</td>"
            )
        rows.append(f"<tr><td><code>{html.escape(scorer_name)}</code></td>{''.join(cells)}</tr>")

    return f"""
    <table>
      <thead>
        <tr>
          <th style="width: 24%">Scorer</th>
          {headers}
        </tr>
      </thead>
      <tbody>
        {"".join(rows)}
      </tbody>
    </table>"""


def _render_case_row(case: CaseComparison, model_labels: list[str]) -> str:
    status_badge = ""
    if case.status == "regressed":
        status_badge = '<span class="tag tag-fail">regressed</span>'
    elif case.status == "improved":
        status_badge = '<span class="tag tag-pass">improved</span>'

    cells: list[str] = []
    for label in model_labels:
        cell = case.results.get(label, {"missing": True})
        cells.append(_render_case_cell(cell))

    prompt_short = case.prompt if len(case.prompt) <= 130 else case.prompt[:127] + "..."
    return f"""
        <tr data-status="{html.escape(case.status)}" data-category="{html.escape(case.category)}" data-critical="{str(case.has_critical_failure).lower()}">
          <td>
            <div class="case-id">{html.escape(case.case_id)} {status_badge}</div>
            <div class="subtitle">{html.escape(case.category)}</div>
            <div class="prompt-preview">{html.escape(prompt_short)}</div>
          </td>
          {"".join(cells)}
        </tr>"""


def _render_case_cell(cell: dict[str, Any]) -> str:
    if cell.get("missing"):
        return '<td class="subtitle">missing</td>'

    passed = bool(cell["passed"])
    badge = '<span class="tag tag-pass">PASS</span>' if passed else '<span class="tag tag-fail">FAIL</span>'
    score_val = float(cell["score"])
    lat = float(cell["latency_ms"])

    scorer_items = []
    for sc in cell.get("scores", []):
        icon = "PASS" if sc["passed"] else "FAIL"
        sc_cls = "tag-pass" if sc["passed"] else "tag-fail"
        crit_str = " [CRIT]" if sc.get("critical") else ""
        scorer_items.append(
            f'<li><span class="tag {sc_cls}">{icon}</span> '
            f'<code>{html.escape(sc["name"])}{crit_str}</code>: '
            f'{html.escape(str(sc["reason"]))}</li>'
        )

    tool_pills = []
    for tc in cell.get("tool_calls", []):
        st = str(tc.get("status", "ok"))
        st_cls = "tag-pass" if st == "ok" else "tag-fail"
        tool_pills.append(
            f'<span class="tool-pill">{html.escape(str(tc["tool_name"]))} '
            f'<span class="tag {st_cls}">{html.escape(st)}</span></span>'
        )
    tools_html = f"<div>{''.join(tool_pills)}</div>" if tool_pills else ""

    response_text = html.escape(str(cell.get("response_text", "")))
    return f"""
          <td>
            <div>{badge} <strong>{score_val:.2f}</strong> <span class="subtitle">{lat:.0f} ms</span></div>
            {tools_html}
            <details>
              <summary>Details</summary>
              <ul class="scorer-list">{"".join(scorer_items)}</ul>
              <pre>{response_text}</pre>
            </details>
          </td>"""


def save_comparison_html(report: ComparisonReport, output_path: str | Path) -> Path:
    """Render and write the static HTML report to disk."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_comparison_html(report), encoding="utf-8")
    return out
