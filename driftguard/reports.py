"""CSV, JSON and HTML exports.

Exports contain aggregate statistics only - never raw rows. Identifier-like
columns are excluded from analysis and therefore from exported statistics.
"""

from __future__ import annotations

import html
import json
from typing import Any

import numpy as np
import pandas as pd

from .drift import DRIFT, INSUFFICIENT
from .methodology import DRIFT_VS_DEGRADATION, LIMITATIONS, METHODOLOGY, threshold_rows
from .pipeline import APP_VERSION, AnalysisResult
from .profiling import clean_number
from .schema import NUMERICAL


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.generic, float, int)) and not isinstance(obj, bool):
        return clean_number(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def results_dataframe(result: AnalysisResult) -> pd.DataFrame:
    rows = []
    for f in result.features:
        rs, cs = f.reference_stats, f.current_stats
        numeric = f.feature_type == NUMERICAL
        rows.append(
            {
                "feature": f.feature,
                "feature_type": f.feature_type,
                "status": f.status,
                "test": f.test,
                "n_reference": f.n_reference,
                "n_current": f.n_current,
                "statistic_name": f.statistic_name,
                "statistic": f.statistic,
                "p_value": f.p_value,
                "adjusted_p_value": f.adjusted_p_value,
                "effect_size_name": f.effect_size_name,
                "effect_size": f.effect_size,
                "psi": f.psi,
                "reference_center": rs.get("mean") if numeric else rs.get("mode"),
                "current_center": cs.get("mean") if numeric else cs.get("mode"),
                "reference_spread": rs.get("std") if numeric else rs.get("n_unique"),
                "current_spread": cs.get("std") if numeric else cs.get("n_unique"),
                "explanation": f.explanation,
            }
        )
    return pd.DataFrame(rows)


def results_csv(result: AnalysisResult) -> bytes:
    df = results_dataframe(result)
    df.insert(0, "analysis_id", result.analysis_id)
    df.insert(1, "timestamp_utc", result.timestamp)
    return df.to_csv(index=False).encode("utf-8")


def summary_dict(result: AnalysisResult) -> dict[str, Any]:
    features = []
    for f in result.features:
        d = f.to_dict()
        d["details"] = {k: v for k, v in d["details"].items() if k != "proportions"}
        features.append(d)
    return _json_safe(
        {
            "tool": "DriftGuard AI",
            "version": APP_VERSION,
            "analysis_id": result.analysis_id,
            "timestamp_utc": result.timestamp,
            "datasets": {
                "reference": result.reference_meta.to_dict(),
                "current": result.current_meta.to_dict(),
            },
            "configuration": result.config.to_dict(),
            "summary": result.summary(),
            "schema": {
                "common_columns": result.schema.common_columns,
                "missing_in_current": result.schema.missing_in_current,
                "new_in_current": result.schema.new_in_current,
                "type_mismatches": result.schema.type_mismatches,
            },
            "excluded_features": result.excluded_features,
            "features": features,
            "quality_issues": [i.to_dict() for i in result.quality_issues],
            "methodology": dict(METHODOLOGY),
            "limitations": LIMITATIONS,
            "notes": result.notes,
        }
    )


def summary_json(result: AnalysisResult) -> str:
    return json.dumps(summary_dict(result), indent=2, allow_nan=False)


def format_p(p: float | None) -> str:
    if p is None:
        return "n/a"
    if p < 1e-4:
        return "< 0.0001"
    return f"{p:.4f}"


def _num(v: float | None, digits: int = 3) -> str:
    return "n/a" if v is None else f"{v:.{digits}f}"


_CSS = """
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;color:#0f172a;max-width:1100px;margin:2rem auto;padding:0 1.25rem;line-height:1.5}
h1{font-size:1.6rem;margin-bottom:.25rem}h2{font-size:1.15rem;margin-top:2rem;border-bottom:1px solid #e2e8f0;padding-bottom:.35rem}
.muted{color:#475569}.banner{background:#fef3c7;border:1px solid #f59e0b;padding:.6rem .9rem;border-radius:6px;margin:1rem 0}
table{border-collapse:collapse;width:100%;font-size:.86rem;margin:.5rem 0}th,td{border:1px solid #e2e8f0;padding:.4rem .55rem;text-align:left;vertical-align:top}
th{background:#f1f5f9}.drift{color:#b91c1c;font-weight:600}.ok{color:#15803d}.na{color:#64748b}
.cards{display:flex;gap:1rem;flex-wrap:wrap}.card{border:1px solid #e2e8f0;border-radius:8px;padding:.75rem 1rem;min-width:160px}
.card b{display:block;font-size:1.4rem}
"""


def html_report(result: AnalysisResult) -> str:
    e = html.escape
    s = result.summary()
    q = s["quality_counts"]
    ref, cur = result.reference_meta, result.current_meta

    def status_class(status: str) -> str:
        return "drift" if status == DRIFT else "na" if status == INSUFFICIENT else "ok"

    feature_rows = "".join(
        f"<tr><td>{e(f.feature)}</td><td>{e(f.feature_type)}</td>"
        f"<td class='{status_class(f.status)}'>{e(f.status)}</td><td>{e(f.test)}</td>"
        f"<td>{format_p(f.p_value)}</td><td>{format_p(f.adjusted_p_value)}</td>"
        f"<td>{e(f.effect_size_name or '')} {_num(f.effect_size)}</td><td>{_num(f.psi)}</td>"
        f"<td>{e(f.explanation)}</td></tr>"
        for f in result.features
    )
    quality_rows = "".join(
        f"<tr><td>{e(i.severity)}</td><td>{e(i.check)}</td><td>{e(i.column or '-')}</td>"
        f"<td>{e(i.dataset)}</td><td>{e(i.message)}</td></tr>"
        for i in result.quality_issues
    ) or "<tr><td colspan='5'>No data quality issues found.</td></tr>"
    excluded_rows = "".join(
        f"<li><b>{e(c)}</b>: {e(r)}</li>" for c, r in result.excluded_features.items()
    ) or "<li>None</li>"
    thresholds = "".join(f"<tr><th>{e(k)}</th><td>{e(v)}</td></tr>" for k, v in threshold_rows(result.config))
    method = "".join(f"<h3>{e(t)}</h3><p>{e(d)}</p>" for t, d in METHODOLOGY)
    limits = "".join(f"<li>{e(x)}</li>" for x in LIMITATIONS)
    dvd = "".join(f"<tr><th>{e(a)}</th><td>{e(b)}</td></tr>" for a, b in DRIFT_VS_DEGRADATION)
    notes = "".join(f"<div class='banner'>{e(n)}</div>" for n in result.notes)

    def meta_row(label: str, m) -> str:
        return (
            f"<tr><th>{label}</th><td>{e(m.name)}</td><td>{m.n_rows:,}</td><td>{m.n_columns}</td>"
            f"<td>{e(m.source)}{' (synthetic)' if m.synthetic else ''}</td></tr>"
        )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DriftGuard AI report {e(result.analysis_id)}</title><style>{_CSS}</style></head>
<body>
<h1>DriftGuard AI - Data Drift Monitoring Report</h1>
<p class="muted">Analysis {e(result.analysis_id)} &middot; generated {e(result.timestamp)} (UTC) &middot; v{APP_VERSION}</p>
{notes}
<div class="banner">This report measures changes in input data distributions. It does not measure model accuracy;
no model was trained or evaluated.</div>
<h2>Summary</h2>
<p><b>{e(s["headline"])}</b></p>
<div class="cards">
<div class="card">Features analysed<b>{s["n_features_analyzed"]}</b></div>
<div class="card">Drift detected<b>{s["n_drifted"]}</b></div>
<div class="card">Insufficient data<b>{s["n_insufficient"]}</b></div>
<div class="card">Quality issues<b>{s["n_quality_issues"]}</b><span class="muted">{q["critical"]} critical, {q["warning"]} warning, {q["info"]} info</span></div>
</div>
<h2>Datasets</h2>
<table><tr><th></th><th>Name</th><th>Rows</th><th>Columns</th><th>Source</th></tr>
{meta_row("Reference", ref)}{meta_row("Current", cur)}</table>
<h2>Feature-level drift results</h2>
<table><tr><th>Feature</th><th>Type</th><th>Status</th><th>Test</th><th>p-value</th><th>Adjusted p</th><th>Effect size</th><th>PSI</th><th>Explanation</th></tr>
{feature_rows}</table>
<p class="muted">Excluded columns:</p><ul>{excluded_rows}</ul>
<h2>Data quality (separate from drift)</h2>
<table><tr><th>Severity</th><th>Check</th><th>Column</th><th>Dataset</th><th>Details</th></tr>{quality_rows}</table>
<h2>Configuration and thresholds</h2><table>{thresholds}</table>
<h2>Methodology</h2>{method}
<h2>Data drift vs. concept drift vs. performance degradation</h2><table>{dvd}</table>
<h2>Limitations</h2><ul>{limits}</ul>
<p class="muted">This report contains aggregate statistics only; no raw rows are included.</p>
</body></html>"""
