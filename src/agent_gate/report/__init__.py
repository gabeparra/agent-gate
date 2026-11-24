"""Side-by-side model comparison and static HTML report generation."""

from agent_gate.report.compare import (
    CaseComparison,
    ComparisonReport,
    ModelComparison,
    compare_runs,
    save_comparison_json,
)
from agent_gate.report.html_report import render_comparison_html, save_comparison_html

__all__ = [
    "CaseComparison",
    "ComparisonReport",
    "ModelComparison",
    "compare_runs",
    "render_comparison_html",
    "save_comparison_html",
    "save_comparison_json",
]
