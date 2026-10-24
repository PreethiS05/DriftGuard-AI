"""Plotly figures. Every figure is computed from the active datasets/results."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .config import AnalysisConfig
from .drift import DRIFT, INSUFFICIENT, NO_DRIFT, FeatureDriftResult
from .profiling import DatasetProfile

REFERENCE_COLOR = "#64748B"
CURRENT_COLOR = "#0D9488"
STATUS_COLORS = {DRIFT: "#DC2626", NO_DRIFT: "#16A34A", INSUFFICIENT: "#A1A1AA"}
FONT = "Inter, system-ui, -apple-system, Segoe UI, sans-serif"


def _layout(fig: go.Figure, title: str | None = None, height: int = 360) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=15)) if title else None,
        height=height,
        margin=dict(l=10, r=10, t=50 if title else 20, b=10),
        font=dict(family=FONT, size=12, color="#0F172A"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hoverlabel=dict(font_family=FONT),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#E2E8F0", zeroline=False, linecolor="#CBD5E1")
    fig.update_yaxes(showgrid=True, gridcolor="#E2E8F0", zeroline=False, linecolor="#CBD5E1")
    return fig


def _finite(series: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def numeric_histogram(reference: pd.Series, current: pd.Series, feature: str, bins: int = 30) -> go.Figure:
    """Overlaid histograms on identical bin edges, normalised to % of each dataset."""
    ref, cur = _finite(reference), _finite(current)
    fig = go.Figure()
    if ref.size == 0 and cur.size == 0:
        return _layout(fig, f"{feature}: no numeric values")
    combined = np.concatenate([ref, cur])
    lo, hi = float(combined.min()), float(combined.max())
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    edges = np.linspace(lo, hi, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    width = edges[1] - edges[0]
    for name, values, color in (("Reference", ref, REFERENCE_COLOR), ("Current", cur, CURRENT_COLOR)):
        if values.size == 0:
            continue
        counts, _ = np.histogram(values, bins=edges)
        pct = counts / values.size * 100
        fig.add_bar(
            x=centers,
            y=pct,
            width=width,
            name=f"{name} (n={values.size:,})",
            marker_color=color,
            opacity=0.6,
            customdata=np.stack([edges[:-1], edges[1:], counts], axis=1),
            hovertemplate="%{customdata[0]:.4g} to %{customdata[1]:.4g}<br>%{y:.2f}% (%{customdata[2]} rows)<extra>"
            + name
            + "</extra>",
        )
    fig.update_layout(barmode="overlay", bargap=0)
    fig.update_yaxes(title_text="% of non-missing rows", rangemode="tozero")
    fig.update_xaxes(title_text=feature)
    return _layout(fig, "Distribution (shared bins)")


def ecdf_figure(reference: pd.Series, current: pd.Series, feature: str) -> go.Figure:
    """Empirical CDFs with the KS statistic location marked."""
    ref, cur = np.sort(_finite(reference)), np.sort(_finite(current))
    fig = go.Figure()
    if ref.size == 0 or cur.size == 0:
        return _layout(fig, "Empirical CDF: insufficient data")
    for name, values, color in (("Reference", ref, REFERENCE_COLOR), ("Current", cur, CURRENT_COLOR)):
        y = np.arange(1, values.size + 1) / values.size
        step = max(1, values.size // 2000)
        fig.add_scatter(
            x=values[::step], y=y[::step], mode="lines", line=dict(color=color, width=2, shape="hv"), name=name
        )
    grid = np.unique(np.concatenate([ref, cur]))
    ref_cdf = np.searchsorted(ref, grid, side="right") / ref.size
    cur_cdf = np.searchsorted(cur, grid, side="right") / cur.size
    i = int(np.argmax(np.abs(ref_cdf - cur_cdf)))
    d = abs(ref_cdf[i] - cur_cdf[i])
    fig.add_scatter(
        x=[grid[i], grid[i]],
        y=[ref_cdf[i], cur_cdf[i]],
        mode="lines+markers",
        line=dict(color="#DC2626", width=2, dash="dot"),
        marker=dict(size=6),
        name=f"Max gap D = {d:.3f}",
    )
    fig.update_yaxes(title_text="Cumulative share", range=[0, 1.02])
    fig.update_xaxes(title_text=feature)
    return _layout(fig, "Empirical CDF (KS statistic = largest vertical gap)")


def categorical_bar(result: FeatureDriftResult, top_n: int = 20) -> go.Figure:
    proportions = result.details.get("proportions") or {}
    fig = go.Figure()
    if not proportions:
        return _layout(fig, "No category distribution available")
    items = sorted(proportions.items(), key=lambda kv: -max(kv[1]["reference"], kv[1]["current"]))[:top_n]
    cats = [k for k, _ in items]
    ref = [v["reference"] * 100 for _, v in items]
    cur = [v["current"] * 100 for _, v in items]
    unseen = set(result.details.get("unseen_categories") or [])
    labels = [f"{c} (new)" if c in unseen else c for c in cats]
    fig.add_bar(x=labels, y=ref, name="Reference", marker_color=REFERENCE_COLOR, hovertemplate="%{x}: %{y:.2f}%<extra>Reference</extra>")
    fig.add_bar(x=labels, y=cur, name="Current", marker_color=CURRENT_COLOR, hovertemplate="%{x}: %{y:.2f}%<extra>Current</extra>")
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="% of non-missing rows", rangemode="tozero")
    title = "Category proportions"
    if len(proportions) > top_n:
        title += f" (top {top_n} of {len(proportions)})"
    return _layout(fig, title)


def missing_comparison(
    ref_profile: DatasetProfile, cur_profile: DatasetProfile, columns: list[str]
) -> go.Figure:
    cols = [c for c in columns if c in ref_profile.columns and c in cur_profile.columns]
    ref = [ref_profile.columns[c].pct_missing * 100 for c in cols]
    cur = [cur_profile.columns[c].pct_missing * 100 for c in cols]
    fig = go.Figure()
    fig.add_bar(y=cols, x=ref, name="Reference", orientation="h", marker_color=REFERENCE_COLOR, hovertemplate="%{y}: %{x:.2f}%<extra>Reference</extra>")
    fig.add_bar(y=cols, x=cur, name="Current", orientation="h", marker_color=CURRENT_COLOR, hovertemplate="%{y}: %{x:.2f}%<extra>Current</extra>")
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="% missing", rangemode="tozero")
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, "Missing values by column", height=max(260, 34 * len(cols) + 90))


def psi_overview(results: list[FeatureDriftResult], config: AnalysisConfig) -> go.Figure:
    rows = sorted((r for r in results if r.psi is not None), key=lambda r: r.psi or 0.0)
    fig = go.Figure()
    if not rows:
        return _layout(fig, "No PSI values available")
    fig.add_bar(
        y=[r.feature for r in rows],
        x=[r.psi for r in rows],
        orientation="h",
        marker_color=[STATUS_COLORS[r.status] for r in rows],
        customdata=[[r.status, r.feature_type] for r in rows],
        hovertemplate="%{y}<br>PSI %{x:.3f}<br>%{customdata[0]} (%{customdata[1]})<extra></extra>",
        showlegend=False,
    )
    fig.add_vline(x=0.1, line_dash="dot", line_color="#94A3B8", annotation_text="0.1", annotation_position="top")
    fig.add_vline(
        x=config.psi_threshold,
        line_dash="dash",
        line_color="#B45309",
        annotation_text=f"threshold {config.psi_threshold:g}",
        annotation_position="top",
    )
    fig.update_xaxes(title_text="PSI (heuristic distance, not a probability)", rangemode="tozero")
    return _layout(fig, "Population Stability Index by feature", height=max(280, 30 * len(rows) + 100))
