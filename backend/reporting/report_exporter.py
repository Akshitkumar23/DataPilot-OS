"""
DataPilot OS — Phase 7 Report Exporter.
Exports AnalyticalReport models to standalone, publication-quality HTML and print-friendly documents.
Adheres strictly to DataPilot OS design language: dark modern slate theme, emerald/sky accents, and crisp print styling.
"""
from typing import Dict, Any
from backend.reporting.evidence_model import AnalyticalReport


class ReportExporter:
    """Exports structured reports to standalone HTML and JSON."""

    @classmethod
    def to_html(cls, report: AnalyticalReport) -> str:
        """Render self-contained HTML report with responsive styling and print media query."""
        sections_html = []
        for s in report.sections:
            metrics_html = ""
            if s.metrics:
                metric_badges = []
                for k, v in list(s.metrics.items())[:8]:
                    if isinstance(v, (int, float, str, bool)):
                        formatted_val = f"{v:,}" if isinstance(v, int) else (f"{v:.4f}" if isinstance(v, float) else str(v))
                        label = k.replace("_", " ").title()
                        metric_badges.append(
                            f'<div class="metric-card"><span class="metric-label">{label}</span>'
                            f'<span class="metric-value">{formatted_val}</span></div>'
                        )
                if metric_badges:
                    metrics_html = f'<div class="metrics-grid">{"".join(metric_badges)}</div>'

            tables_html = ""
            if s.tables:
                table_rows = []
                for row in s.tables[:10]:
                    cells = "".join(f"<td>{str(v)}</td>" for v in row.values())
                    table_rows.append(f"<tr>{cells}</tr>")
                headers = "".join(f"<th>{k.replace('_', ' ').title()}</th>" for k in s.tables[0].keys())
                tables_html = f'<div class="table-wrap"><table class="data-table"><thead><tr>{headers}</tr></thead><tbody>{"".join(table_rows)}</tbody></table></div>'

            sections_html.append(f"""
            <section class="report-section" id="{s.section_id}">
                <div class="section-header">
                    <h2>{s.title}</h2>
                    {f'<p class="section-subtitle">{s.subtitle}</p>' if s.subtitle else ''}
                </div>
                <div class="section-content">
                    <p class="narrative-text">{s.narrative}</p>
                    {metrics_html}
                    {tables_html}
                </div>
            </section>
            """)

        findings_list = "".join(f"<li>{f}</li>" for f in report.key_findings)
        limitations_list = "".join(f"<li>{l}</li>" for l in report.limitations)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{report.title} — DataPilot OS</title>
    <style>
        :root {{
            --bg: #090d16;
            --surface: #0f172a;
            --card-bg: #131b2e;
            --border: #1e293b;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent: #38bdf8;
            --accent-green: #34d399;
            --accent-purple: #a78bfa;
            --font-family: 'Inter', system-ui, -apple-system, sans-serif;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: var(--bg);
            color: var(--text-main);
            font-family: var(--font-family);
            line-height: 1.6;
            padding: 32px;
        }}
        .report-container {{
            max-width: 1000px;
            margin: 0 auto;
            display: flex;
            flex-direction: column;
            gap: 28px;
        }}
        .report-header {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 24px 32px;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }}
        .header-top {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid var(--border);
            padding-bottom: 12px;
        }}
        .brand-pill {{
            background: rgba(56, 189, 248, 0.1);
            color: var(--accent);
            border: 1px solid rgba(56, 189, 248, 0.2);
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .timestamp {{
            font-size: 12px;
            color: var(--text-muted);
        }}
        .report-title {{
            font-size: 24px;
            font-weight: 800;
            color: #ffffff;
        }}
        .meta-strip {{
            display: flex;
            flex-wrap: wrap;
            gap: 16px;
            font-size: 13px;
            color: var(--text-muted);
        }}
        .meta-strip strong {{ color: var(--text-main); }}
        .badge-stale {{
            background: rgba(239, 68, 68, 0.15);
            color: #f87171;
            padding: 2px 8px;
            border-radius: 4px;
            font-weight: 600;
        }}
        .exec-summary-card {{
            background: linear-gradient(135deg, rgba(14, 165, 233, 0.08), rgba(16, 185, 129, 0.08));
            border: 1px solid rgba(56, 189, 248, 0.3);
            border-radius: 10px;
            padding: 24px;
        }}
        .exec-summary-card h2 {{
            font-size: 16px;
            color: var(--accent);
            margin-bottom: 10px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .exec-summary-card p {{
            font-size: 14px;
            color: #e2e8f0;
            line-height: 1.7;
        }}
        .findings-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 20px 24px;
        }}
        .findings-card h3 {{
            font-size: 15px;
            color: var(--accent-green);
            margin-bottom: 12px;
            font-weight: 700;
        }}
        .findings-card ul {{
            padding-left: 20px;
            display: flex;
            flex-direction: column;
            gap: 8px;
            font-size: 13px;
            color: #cbd5e1;
        }}
        .report-section {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 22px 26px;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }}
        .section-header h2 {{
            font-size: 16px;
            color: #ffffff;
            font-weight: 700;
        }}
        .section-subtitle {{
            font-size: 12px;
            color: var(--text-muted);
            margin-top: 2px;
        }}
        .narrative-text {{
            font-size: 13px;
            color: #cbd5e1;
            line-height: 1.6;
            margin-bottom: 12px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 12px;
            margin-top: 10px;
        }}
        .metric-card {{
            background: var(--card-bg);
            border: 1px solid rgba(255, 255, 255, 0.05);
            padding: 12px 16px;
            border-radius: 8px;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }}
        .metric-label {{
            font-size: 11px;
            color: var(--text-muted);
            text-transform: uppercase;
            font-weight: 600;
            letter-spacing: 0.5px;
        }}
        .metric-value {{
            font-size: 18px;
            color: var(--accent);
            font-weight: 800;
            font-family: monospace;
        }}
        .table-wrap {{
            overflow-x: auto;
            margin-top: 10px;
            border: 1px solid var(--border);
            border-radius: 6px;
        }}
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
            text-align: left;
        }}
        .data-table th {{
            background: #1e293b;
            color: #94a3b8;
            padding: 8px 12px;
            font-weight: 700;
        }}
        .data-table td {{
            padding: 8px 12px;
            border-top: 1px solid var(--border);
            color: #e2e8f0;
            font-family: monospace;
        }}
        .limitations-card {{
            background: rgba(30, 41, 59, 0.5);
            border: 1px dashed #334155;
            border-radius: 8px;
            padding: 18px 22px;
        }}
        .limitations-card h3 {{
            font-size: 14px;
            color: #f59e0b;
            margin-bottom: 8px;
            font-weight: 700;
        }}
        .limitations-card ul {{
            padding-left: 20px;
            font-size: 12px;
            color: #94a3b8;
            display: flex;
            flex-direction: column;
            gap: 6px;
        }}
        .report-footer {{
            text-align: center;
            font-size: 12px;
            color: var(--text-muted);
            border-top: 1px solid var(--border);
            padding-top: 20px;
            margin-top: 20px;
        }}

        @media print {{
            body {{
                background: #ffffff;
                color: #0f172a;
                padding: 0;
            }}
            .report-container {{ max-width: 100%; }}
            .report-header, .report-section, .findings-card, .exec-summary-card {{
                background: #ffffff;
                color: #0f172a;
                border: 1px solid #cbd5e1;
                page-break-inside: avoid;
            }}
            .report-title {{ color: #0f172a; }}
            .narrative-text, .findings-card ul {{ color: #334155; }}
            .metric-card {{
                background: #f8fafc;
                border: 1px solid #e2e8f0;
            }}
            .metric-value {{ color: #0284c7; }}
            .data-table th {{ background: #f1f5f9; color: #475569; }}
            .data-table td {{ color: #1e293b; border-color: #e2e8f0; }}
        }}
    </style>
</head>
<body>
    <div class="report-container">
        <header class="report-header">
            <div class="header-top">
                <span class="brand-pill">DataPilot OS • Executive Report</span>
                <span class="timestamp">Generated: {report.generated_at}</span>
            </div>
            <h1 class="report-title">{report.title}</h1>
            <div class="meta-strip">
                <span>Dataset: <strong>{report.dataset_name}</strong></span>
                <span>ID: <strong>{report.dataset_id}</strong></span>
                <span>Fingerprint: <code>{report.dataset_fingerprint}</code></span>
                <span>Evidence Items: <strong>{report.total_evidence_count}</strong></span>
                {f'<span class="badge-stale">Stale Evidence Present</span>' if report.is_stale else ''}
            </div>
        </header>

        <section class="exec-summary-card">
            <h2>Executive Summary</h2>
            <p>{report.executive_summary}</p>
        </section>

        <section class="findings-card">
            <h3>Key Empirical Findings</h3>
            <ul>
                {findings_list}
            </ul>
        </section>

        {"".join(sections_html)}

        <section class="limitations-card">
            <h3>Methodological Limitations & Scientific Boundaries</h3>
            <ul>
                {limitations_list}
            </ul>
        </section>

        <footer class="report-footer">
            <p>DataPilot OS Kernel v2.0 • 100% Offline Analytical Governance • Zero Arbitrary Execution</p>
        </footer>
    </div>
</body>
</html>
"""
        return html

    to_standalone_html = to_html
