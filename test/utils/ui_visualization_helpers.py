# sentinel_project_root/test/utils/ui_visualization_helpers.py
# UI and Plotting helpers for Sentinel Health Co-Pilot Web Dashboards.

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import logging
import plotly.io as pio
import html
import os
import json
from typing import Optional, List, Dict, Any, Union

logger = logging.getLogger(__name__)

from config import app_config

# Fallback config values
COLOR_RISK_HIGH = getattr(app_config, 'COLOR_RISK_HIGH', '#D32F2F')
COLOR_RISK_MODERATE = getattr(app_config, 'COLOR_RISK_MODERATE', '#FFB300')
COLOR_RISK_LOW = getattr(app_config, 'COLOR_RISK_LOW', '#4CAF50')
COLOR_RISK_NEUTRAL = getattr(app_config, 'COLOR_RISK_NEUTRAL', '#757575')
COLOR_ACTION_PRIMARY = getattr(app_config, 'COLOR_ACTION_PRIMARY', '#1976D2')
COLOR_ACTION_SECONDARY = getattr(app_config, 'COLOR_ACTION_SECONDARY', '#F06292')
COLOR_POSITIVE_DELTA = getattr(app_config, 'COLOR_POSITIVE_DELTA', '#388E3C')
COLOR_NEGATIVE_DELTA = getattr(app_config, 'COLOR_NEGATIVE_DELTA', '#D32F2F')
COLOR_TEXT_DARK = getattr(app_config, 'COLOR_TEXT_DARK', '#212121')
COLOR_TEXT_HEADINGS_MAIN = getattr(app_config, 'COLOR_TEXT_HEADINGS_MAIN', '#1976D2')
COLOR_ACCENT_BRIGHT = getattr(app_config, 'COLOR_ACCENT_BRIGHT', '#FF5722')
COLOR_TEXT_LINK_DEFAULT = getattr(app_config, 'COLOR_TEXT_LINK_DEFAULT', '#0288D1')
COLOR_BACKGROUND_CONTENT = getattr(app_config, 'COLOR_BACKGROUND_CONTENT', '#FFFFFF')
COLOR_BACKGROUND_PAGE = getattr(app_config, 'COLOR_BACKGROUND_PAGE', '#F5F5F5')
COLOR_BORDER_LIGHT = getattr(app_config, 'COLOR_BORDER_LIGHT', '#E0E0E0')
COLOR_BORDER_MEDIUM = getattr(app_config, 'COLOR_BORDER_MEDIUM', '#B0BEC5')
COLOR_BACKGROUND_WHITE = getattr(app_config, 'COLOR_BACKGROUND_WHITE', '#FFFFFF')
MAP_DEFAULT_CENTER_LAT = getattr(app_config, 'MAP_DEFAULT_CENTER_LAT', 0.0)
MAP_DEFAULT_CENTER_LON = getattr(app_config, 'MAP_DEFAULT_CENTER_LON', 0.0)
MAP_DEFAULT_ZOOM = getattr(app_config, 'MAP_DEFAULT_ZOOM', 5)
WEB_PLOT_DEFAULT_HEIGHT = getattr(app_config, 'WEB_PLOT_DEFAULT_HEIGHT', 400)
WEB_PLOT_COMPACT_HEIGHT = getattr(app_config, 'WEB_PLOT_COMPACT_HEIGHT', 300)
WEB_MAP_DEFAULT_HEIGHT = getattr(app_config, 'WEB_MAP_DEFAULT_HEIGHT', 500)
MAPBOX_STYLE_WEB = getattr(app_config, 'MAPBOX_STYLE_WEB', 'open-street-map')
LEGACY_DISEASE_COLORS_WEB = getattr(app_config, 'LEGACY_DISEASE_COLORS_WEB', {})

# Mapbox Token Handling
MAPBOX_TOKEN_SET_FLAG = False
try:
    _SENTINEL_MAPBOX_ACCESS_TOKEN = os.getenv("MAPBOX_ACCESS_TOKEN")
    if _SENTINEL_MAPBOX_ACCESS_TOKEN and _SENTINEL_MAPBOX_ACCESS_TOKEN.strip() and len(_SENTINEL_MAPBOX_ACCESS_TOKEN) > 50:
        px.set_mapbox_access_token(_SENTINEL_MAPBOX_ACCESS_TOKEN)
        MAPBOX_TOKEN_SET_FLAG = True
        logger.info("Valid Mapbox access token set.")
    else:
        logger.warning("Invalid or missing Mapbox token. Using open styles.")
except Exception as e:
    logger.error(f"Error setting Mapbox token: {e}")

# Core Theming and Color Utilities
def _get_theme_color(index: Any = 0, fallback_color: Optional[str] = None, color_type: str = "general") -> str:
    color_map_direct = {
        "risk_high": COLOR_RISK_HIGH, "risk_moderate": COLOR_RISK_MODERATE,
        "risk_low": COLOR_RISK_LOW, "risk_neutral": COLOR_RISK_NEUTRAL,
        "action_primary": COLOR_ACTION_PRIMARY, "action_secondary": COLOR_ACTION_SECONDARY,
        "positive_delta": COLOR_POSITIVE_DELTA, "negative_delta": COLOR_NEGATIVE_DELTA,
        "text_dark": COLOR_TEXT_DARK, "headings_main": COLOR_TEXT_HEADINGS_MAIN,
        "accent_bright": COLOR_ACCENT_BRIGHT
    }
    if color_type in color_map_direct:
        return color_map_direct[color_type]
    final_fallback = fallback_color if fallback_color else COLOR_TEXT_LINK_DEFAULT
    try:
        if color_type == "disease" and LEGACY_DISEASE_COLORS_WEB:
            if isinstance(index, str) and index in LEGACY_DISEASE_COLORS_WEB:
                return LEGACY_DISEASE_COLORS_WEB[index]
        colorway = px.colors.qualitative.Plotly
        active_template = pio.templates.get(pio.templates.default)
        if active_template and hasattr(active_template.layout, 'colorway'):
            colorway = active_template.layout.colorway or colorway
        num_idx = index if isinstance(index, int) else abs(hash(str(index)))
        return colorway[num_idx % len(colorway)]
    except Exception as e:
        logger.warning(f"Error retrieving color (index: {index}, type: {color_type}): {e}")
        return final_fallback

def set_sentinel_plotly_theme_web():
    theme_font_family = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif'
    sentinel_colorway = [
        COLOR_ACTION_PRIMARY, COLOR_RISK_LOW, COLOR_RISK_MODERATE,
        COLOR_RISK_HIGH, COLOR_ACTION_SECONDARY, _get_theme_color(index=5, fallback_color="#00ACC1"),
        _get_theme_color(index=6, fallback_color="#5E35B1"),
        _get_theme_color(index=7, fallback_color="#FF7043")
    ]
    layout_config = {
        'font': dict(family=theme_font_family, size=11, color=COLOR_TEXT_DARK),
        'paper_bgcolor': COLOR_BACKGROUND_CONTENT,
        'plot_bgcolor': COLOR_BACKGROUND_PAGE,
        'colorway': sentinel_colorway,
        'xaxis': dict(gridcolor=COLOR_BORDER_LIGHT, linecolor=COLOR_BORDER_MEDIUM, zerolinecolor=COLOR_BORDER_MEDIUM, zerolinewidth=1, title_font_size=12, tickfont_size=10, automargin=True),
        'yaxis': dict(gridcolor=COLOR_BORDER_LIGHT, linecolor=COLOR_BORDER_MEDIUM, zerolinecolor=COLOR_BORDER_MEDIUM, zerolinewidth=1, title_font_size=12, tickfont_size=10, automargin=True),
        'title': dict(font=dict(family=theme_font_family, size=15, color=COLOR_TEXT_HEADINGS_MAIN), x=0.03, xanchor='left', y=0.95, yanchor='top', pad=dict(t=20)),
        'legend': dict(bgcolor='rgba(255,255,255,0.9)', bordercolor=COLOR_BORDER_LIGHT, borderwidth=0.5, orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font_size=10),
        'margin': dict(l=60, r=20, t=70, b=55)
    }
    effective_mapbox_style = MAPBOX_STYLE_WEB if MAPBOX_TOKEN_SET_FLAG else 'open-street-map'
    layout_config['mapbox'] = dict(
        style=effective_mapbox_style,
        center=dict(lat=MAP_DEFAULT_CENTER_LAT, lon=MAP_DEFAULT_CENTER_LON),
        zoom=MAP_DEFAULT_ZOOM
    )
    pio.templates["sentinel_web_theme"] = go.layout.Template(layout=go.Layout(**layout_config))
    pio.templates.default = "plotly+sentinel_web_theme"
    logger.info("Plotly theme 'sentinel_web_theme' set.")

set_sentinel_plotly_theme_web()

# HTML-Based UI Components
def render_web_kpi_card(title: str, value_str: str, icon: str = "●", status_level: str = "NEUTRAL",
                        delta: Optional[str] = None, delta_is_positive: Optional[bool] = None,
                        help_text: Optional[str] = None, units: Optional[str] = ""):
    st.markdown("""
    <style>
    .kpi-card { border: 1px solid #E0E0E0; padding: 10px; border-radius: 5px; background: #FFFFFF; }
    .status-high-risk { background-color: #FFF3E0; }
    .kpi-title { font-size: 14px; color: #1976D2; margin: 0; }
    .kpi-value { font-size: 18px; font-weight: bold; margin: 5px 0; }
    .kpi-units { font-size: 12px; color: #757575; }
    .kpi-delta.positive { color: #388E3C; font-size: 12px; }
    .kpi-delta.negative { color: #D32F2F; font-size: 12px; }
    </style>
    """, unsafe_allow_html=True)
    css_status_class = f"status-{status_level.lower().replace('_', '-')}"
    delta_html = f'<p class="kpi-delta {"positive" if delta_is_positive else "negative"}">{html.escape(str(delta))}</p>' if delta else ''
    tooltip_attr = f'title="{html.escape(str(help_text))}"' if help_text else ''
    units_html = f"<span class='kpi-units'>{html.escape(str(units))}</span>" if units else ""
    kpi_card_html = f"""
    <div class="kpi-card {css_status_class}" {tooltip_attr}>
        <div class="kpi-card-header">
            <div class="kpi-icon">{html.escape(str(icon))}</div>
            <h3 class="kpi-title">{html.escape(str(title)[:100])}</h3>
        </div>
        <div class="kpi-body">
            <p class="kpi-value">{html.escape(str(value_str))}{units_html}</p>
            {delta_html}
        </div>
    </div>
    """
    st.markdown(kpi_card_html, unsafe_allow_html=True)

def render_web_traffic_light_indicator(message: str, status_level: str, details_text: Optional[str] = None):
    st.markdown("""
    <style>
    .traffic-light-dot.status-high { background: #D32F2F; }
    .traffic-light-dot.status-low { background: #4CAF50; }
    .traffic-light-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
    .traffic-light-message { font-size: 12px; margin-left: 5px; }
    </style>
    """, unsafe_allow_html=True)
    css_dot_class = f"status-{status_level.lower().replace('_', '-')}"
    details_html = f'<span class="traffic-light-details">{html.escape(str(details_text))}</span>' if details_text else ""
    traffic_light_html = f"""
    <div class="traffic-light-indicator">
        <span class="traffic-light-dot {css_dot_class}"></span>
        <span class="traffic-light-message">{html.escape(message)}</span>
        {details_html}
    </div>
    """
    st.markdown(traffic_light_html, unsafe_allow_html=True)

# Plotly Chart Generation
def _create_empty_plot_figure(title_str: str, height_val: Optional[int], message: str = "No data available.") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title_text=f"{title_str}: {message}",
        height=height_val or WEB_PLOT_DEFAULT_HEIGHT,
        xaxis={'visible': False}, yaxis={'visible': False},
        annotations=[dict(text=message, xref="paper", yref="paper", showarrow=False, font=dict(size=12))]
    )
    return fig

def plot_annotated_line_chart_web(
    data_series: pd.Series, title: str, y_axis_label: str = "Value",
    line_color: Optional[str] = None,
    target_ref_line: Optional[float] = None, target_ref_label: Optional[str] = None,
    show_conf_interval: bool = False, lower_ci: Optional[pd.Series] = None, upper_ci: Optional[pd.Series] = None,
    chart_height: Optional[int] = None, show_anomalies: bool = False, anomaly_factor: float = 1.5,
    date_format: str = "%Y-%m-%d", y_is_count: bool = False
) -> go.Figure:
    final_height = chart_height or WEB_PLOT_COMPACT_HEIGHT
    if not isinstance(data_series, pd.Series) or data_series.empty:
        return _create_empty_plot_figure(title, final_height)
    series_clean = pd.to_numeric(data_series, errors='coerce')
    if series_clean.isnull().all():
        return _create_empty_plot_figure(title, final_height, "All data non-numeric.")
    fig = go.Figure()
    color = line_color or _get_theme_color(0)
    y_hover = 'd' if y_is_count else ',.1f'
    fig.add_trace(go.Scatter(
        x=series_clean.index, y=series_clean.values, mode="lines+markers", name=y_axis_label,
        line=dict(color=color, width=2), marker=dict(size=5),
        customdata=series_clean.values,
        hovertemplate=f'<b>Date</b>: %{{x|{date_format}}}<br><b>{y_axis_label}</b>: %{{y:{y_hover}}}<extra></extra>'
    ))
    if show_conf_interval and isinstance(lower_ci, pd.Series) and isinstance(upper_ci, pd.Series) and not lower_ci.empty and not upper_ci.empty:
        common_idx = series_clean.index.intersection(lower_ci.index).intersection(upper_ci.index)
        if not common_idx.empty:
            l_ci = pd.to_numeric(lower_ci.reindex(common_idx), errors='coerce')
            u_ci = pd.to_numeric(upper_ci.reindex(common_idx), errors='coerce')
            valid_ci = l_ci.notna() & u_ci.notna() & (u_ci >= l_ci)
            if valid_ci.any():
                x_vals = common_idx[valid_ci]
                y_upper = u_ci[valid_ci]
                y_lower = l_ci[valid_ci]
                fill_color = f"rgba({','.join(str(int(c,16)) for c in (color[1:3], color[3:5], color[5:7]))},0.15)" if color.startswith('#') and len(color)==7 else "rgba(100,100,100,0.15)"
                fig.add_trace(go.Scatter(
                    x=list(x_vals) + list(x_vals[::-1]),
                    y=list(y_upper.values) + list(y_lower.values[::-1]),
                    fill="toself", fillcolor=fill_color,
                    line=dict(width=0), name="Confidence Interval", hoverinfo="skip"
                ))
    if target_ref_line is not None:
        label = target_ref_label or f"Target: {target_ref_line:,.2f}"
        fig.add_hline(
            y=target_ref_line, line_dash="dash", line_color=COLOR_RISK_MODERATE,
            line_width=1.2, annotation_text=label,
            annotation_position="bottom right", annotation_font_size=9
        )
    if show_anomalies and len(series_clean.dropna()) > 7 and series_clean.nunique() > 2:
        q1, q3 = series_clean.quantile(0.25), series_clean.quantile(0.75)
        iqr = q3 - q1
        if pd.notna(iqr) and iqr > 0:
            upper_bound, lower_bound = q3 + anomaly_factor * iqr, q1 - anomaly_factor * iqr
            anomalies = series_clean[(series_clean < lower_bound) | (series_clean > upper_bound)]
            if not anomalies.empty:
                fig.add_trace(go.Scatter(
                    x=anomalies.index, y=anomalies.values, mode='markers',
                    marker=dict(color=COLOR_RISK_HIGH, size=7, symbol='circle-open', line=dict(width=1.5)),
                    name='Anomaly', customdata=anomalies.values,
                    hovertemplate=f'<b>Anomaly Date</b>: %{{x|{date_format}}}<br><b>Value</b>: %{{y:{y_hover}}}<extra></extra>'
                ))
    x_axis_title = series_clean.index.name or "Date/Time"
    y_axis_config = dict(title_text=y_axis_label, rangemode='tozero' if y_is_count and series_clean.min() >= 0 else 'normal')
    if y_is_count:
        y_axis_config['tickformat'] = 'd'
    fig.update_layout(
        title_text=title, xaxis_title=x_axis_title, yaxis=y_axis_config,
        height=final_height, hovermode="x unified", legend=dict(traceorder='normal')
    )
    return fig

def plot_bar_chart_web(
    df: pd.DataFrame, x_col: str, y_col: str, title: str,
    color_col: Optional[str] = None, barmode: str = 'group', orientation: str = 'v',
    y_axis_label: Optional[str] = None, x_axis_label: Optional[str] = None,
    chart_height: Optional[int] = None, text_auto: Union[bool, str] = True,
    sort_by: Optional[str] = None, sort_ascending: bool = True,
    text_format: Optional[str] = None, y_is_count: bool = False,
    color_map: Optional[Dict] = None
) -> go.Figure:
    final_height = chart_height or WEB_PLOT_DEFAULT_HEIGHT
    if not isinstance(df, pd.DataFrame) or df.empty or x_col not in df.columns or y_col not in df.columns:
        return _create_empty_plot_figure(title, final_height)
    df_plot = df.copy()
    df_plot[x_col] = df_plot[x_col].astype(str)
    df_plot[y_col] = pd.to_numeric(df_plot[y_col], errors='coerce')
    if y_is_count:
        df_plot[y_col] = df_plot[y_col].round().astype('Int64')
    df_plot.dropna(subset=[x_col, y_col], inplace=True)
    if df_plot.empty:
        return _create_empty_plot_figure(title, final_height, f"No valid data for x={x_col}, y={y_col}.")
    if sort_by and sort_by in df_plot.columns:
        df_plot.sort_values(by=sort_by, ascending=sort_ascending, inplace=True, na_position='last')
    effective_text_format = text_format or ('.0f' if y_is_count else '.1f')
    y_title_text = y_axis_label or y_col.replace('_', ' ').title()
    x_title_text = x_axis_label or x_col.replace('_', ' ').title()
    legend_title = color_col.replace('_', ' ').title() if color_col and color_col in df_plot.columns else None
    final_color_map = color_map or {
        str(val): _get_theme_color(str(val), color_type="disease", fallback_color=_get_theme_color(abs(hash(str(val))) % 10))
        for val in df_plot[color_col].dropna().unique()
    } if color_col and LEGACY_DISEASE_COLORS_WEB and any(str(val) in LEGACY_DISEASE_COLORS_WEB for val in df_plot[color_col]) else None
    fig = px.bar(
        df_plot, x=x_col, y=y_col, title=title, color=color_col, barmode=barmode,
        orientation=orientation, height=final_height,
        labels={y_col: y_title_text, x_col: x_title_text, color_col: legend_title or ''},
        text_auto=text_auto, color_discrete_map=final_color_map
    )
    hover_val_format = 'd' if y_is_count else effective_text_format
    if orientation == 'v':
        base_hover = f'<b>{x_title_text}</b>: %{{x}}<br><b>{y_title_text}</b>: %{{y:{hover_val_format}}}'
        text_template = f'%{{y:{effective_text_format}}}' if text_auto else None
    else:
        base_hover = f'<b>{y_title_text}</b>: %{{y}}<br><b>{x_title_text}</b>: %{{x:{hover_val_format}}}'
        text_template = f'%{{x:{effective_text_format}}}' if text_auto else None
    hover_text = base_hover
    custom_data_cols = []
    if color_col and color_col in df_plot.columns and legend_title:
        hover_text += f'<br><b>{legend_title}</b>: %{{customdata[0]}}'
        custom_data_cols.append(color_col)
    hover_text += '<extra></extra>'
    fig.update_traces(
        marker_line_width=0.5, marker_line_color='rgba(0,0,0,0.2)',
        textfont_size=9, textangle=0,
        textposition='auto' if orientation == 'v' else 'outside',
        cliponaxis=False, texttemplate=text_template,
        hovertemplate=hover_text, customdata=df_plot[custom_data_cols] if custom_data_cols else None
    )
    y_axis_config = {'title_text': y_title_text}
    x_axis_config = {'title_text': x_title_text}
    val_axis, cat_axis = (y_axis_config, x_axis_config) if orientation == 'v' else (x_axis_config, y_axis_config)
    if y_is_count:
        val_axis['tickformat'] = 'd'
        val_axis['rangemode'] = 'tozero'
    if sort_by == (x_col if orientation == 'v' else y_col):
        cat_axis['categoryorder'] = 'array'
        cat_axis['categoryarray'] = df_plot[x_col if orientation == 'v' else y_col].tolist()
    elif orientation == 'h' and (not sort_by or sort_by == y_col):
        cat_axis['categoryorder'] = 'total ascending' if sort_ascending else 'total descending'
    fig.update_layout(
        yaxis=y_axis_config, xaxis=x_axis_config,
        uniformtext_minsize=7, uniformtext_mode='hide',
        legend_title_text=legend_title
    )
    return fig

def plot_donut_chart_web(
    df: pd.DataFrame, labels_col: str, values_col: str, title: str,
    chart_height: Optional[int] = None, color_map: Optional[Dict] = None,
    pull_amount: float = 0.03, center_annotation_text: Optional[str] = None,
    values_are_absolute_counts: bool = True
) -> go.Figure:
    final_height = chart_height or (WEB_PLOT_COMPACT_HEIGHT + 40)
    if not isinstance(df, pd.DataFrame) or df.empty or labels_col not in df.columns or values_col not in df.columns:
        return _create_empty_plot_figure(title, final_height)
    df_plot = df.copy()
    df_plot[values_col] = pd.to_numeric(df_plot[values_col], errors='coerce').fillna(0)
    if values_are_absolute_counts:
        df_plot[values_col] = df_plot[values_col].round().astype('Int64')
    df_plot = df_plot[df_plot[values_col] > 0]
    if df_plot.empty:
        return _create_empty_plot_figure(title, final_height, "No positive data.")
    df_plot.sort_values(by=values_col, ascending=False, inplace=True)
    df_plot[labels_col] = df_plot[labels_col].astype(str)
    plot_colors = color_map or (
        [_get_theme_color(str(lbl), 'disease', _get_theme_color(i)) for i, lbl in enumerate(df_plot[labels_col])]
        if LEGACY_DISEASE_COLORS_WEB and any(lbl in LEGACY_DISEASE_COLORS_WEB for lbl in df_plot[labels_col])
        else [_get_theme_color(i) for i in range(len(df_plot[labels_col]))]
    )
    hover_val_format = 'd' if values_are_absolute_counts else '.2f'
    fig = go.Figure(data=[go.Pie(
        labels=df_plot[labels_col], values=df_plot[values_col],
        hole=0.55, pull=[pull_amount if i < min(3, len(df_plot)) else 0 for i in range(len(df_plot))],
        textinfo='label+percent', insidetextorientation='radial',
        hoverinfo='label+value+percent',
        hovertemplate=f'<b>%{{label}}</b><br>Value: %{{value:{hover_val_format}}}<br>Percent: %{{percent}}<extra></extra>',
        marker=dict(colors=plot_colors, line=dict(color=COLOR_BACKGROUND_WHITE, width=1.5)),
        sort=False
    )])
    annotations = [dict(text=str(center_annotation_text), x=0.5, y=0.5, font_size=14, showarrow=False, font_color=COLOR_TEXT_DARK)] if center_annotation_text else None
    fig.update_layout(
        title_text=title, height=final_height, showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="right", x=1.15, font_size=8),
        annotations=annotations, margin=dict(l=20, r=100, t=60, b=5)
    )
    return fig

def plot_heatmap_web(
    matrix_df: pd.DataFrame, title: str, chart_height: Optional[int] = None,
    colorscale: str = 'RdBu_r', z_midpoint: Optional[float] = None,
    show_text_on_cells: bool = True, text_display_format: str = '.1f2',
    show_colorbar_legend: bool = True
) -> go.Figure:
    final_height = chart_height or (WEB_PLOT_DEFAULT_HEIGHT + 50)
    if not isinstance(matrix_df, pd.DataFrame) or matrix_df.empty:
        return _create_empty_plot_figure(title, final_height)
    df_numeric = matrix_df.copy().apply(pd.to_numeric, errors='coerce')
    if df_numeric.isnull().all().all():
        return _create_empty_plot_figure(title, final_height, "All data non-numeric.")
    z_values = df_numeric.values
    text_values = np.vectorize(lambda x: f"{x:{text_display_format}}" if pd.notna(x) else '')(z_values) if show_text_on_cells else None
    final_zmid = z_midpoint
    z_flat = z_values[~np.isnan(z_values)]
    if len(z_flat) > 0 and not (np.any(z_flat < 0) and np.any(z_flat > z_flat 0)):
        final_zmid = None
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=z_values,
                x=df_numeric.columns.astype(str).tolist(),
                y=df_numeric.index.astype(str).tolist(),
                colorscale=colorscale,
                zmid=final_zmid,
                text=text_values,
                texttemplate="%{text}" if show_text_on_cells and text_values is not None else '',
                hoverongaps=False,
                xgap=1,
                ygap=1,
                colorbar=dict(
                    thickness=15,
                    len=0.85,
                    tickfont=dict(size=8),
                    title_side='right",
                    outlinewidth=0.5,
                    outlinecolor=COLOR_BORDER_MEDIUM
                ) if show_colorbar_legend else None
            )
        ]
    )
    x_tick_angle = -45 if len(df_numeric.columns) > 7 or max(len(str(c)) for c in df_numeric.columns) > 7 else 0
    fig.update_layout(
        title_text=title,
        height=final_height,
        xaxis=dict(showgrid=False, tickangle=x_tick_angle),
        yaxis=dict(showgrid=False, autorange='reversed'),
        plot_bgcolor=COLOR_BACKGROUND_WHITE
    )
    return fig

def plot_layered_map_web(
    df: pd.DataFrame, geojson_path: str, value_col: str, title: str,
    id_col: str = 'zone_id',
    color_scale: str = 'Viridis',
    hover_data: Optional[List[str]] = None,
    facility_points_df: Optional[pd.DataFrame] = None,
    facility_size_col: Optional[str] = None,
    facility_name_col: Optional[str] = None,
    facility_color: Optional[str] = None,
    height: Optional[int] = None,
    center_lat: Optional[float] = None,
    center_lon: Optional[float] = None,
    zoom: Optional[int] = None,
    map_style: Optional[str] = None
) -> go.Figure:
    final_map_height = height or WEB_MAP_DEFAULT_HEIGHT
    if not isinstance(df, pd.DataFrame) or df.empty or value_col not in df.columns or id_col not in df.columns:
        return _create_empty_plot_figure(title, final_map_height, "Invalid or missing columns.")
    try:
        with open(geojson_path, 'r') as f:
            geo_data = json.load(f)
        if not geo_data.get('features'):
            return _create_empty_plot_figure(title, final_map_height, "Invalid GeoJSON data.")
    except Exception as e:
        logger.error(f"Error loading GeoJSON: {geojson_path}: {e}"")
        return _create_empty_plot_figure(title, final_map_height, "GeoJSON file error.")
    df_plot = df.copy()
    df_plot[value_col] = pd.to_numeric(df_plot[value_col], errors='coerce')
    df_plot.dropna(subset=[value_col, id_col], inplace=True)
    if df_plot.empty:
        return _create_empty_plot_figure(title, final_map_height, f"No valid data for '{value_col}'.")
    hover_name_col = 'name' if 'name' in df_plot.columns else id_col
    final_hover_data = {col: True for col in hover_data if col in df_plot.columns} if hover_data else {value_col: True}
    if hover_name_col not in final_hover_data:
        final_hover_data[hover_name_col] = True
    effective_mapbox_style = map_style or 'open-street-map'
    fig = px.choropleth_mapbox_mapbox(
        df_plot,
        geojson=geo_data,
        locations=df_plot[id_col],
        featureidkey=f"properties.{id_col}",
        color=value_col,
        color_continuous_scale=color_scale,
        hover_name=hover_name_col,
        hover_data=final_map_data,
        mapbox_style=effective_mapbox_style,
        center=dict(lat=center_lat or MAP_DEFAULT_CENTER_LAT, lon=center_lon or MAP_DEFAULT_CENTER_LON),
        zoom=zoom or MAP_DEFAULT_ZOOM,
        opacity=0.7,
        height=final_map_height,
        title=title
    )
    fig.update_layout(
        margin=dict(l=0, r=20, t=45, b=0),
        mapbox_accesstoken=os.getenv("MAPBOX_ACCESS_TOKEN")
    )
    if isinstance(facility_points_df, pd.DataFrame) and not facility_points_df.empty and 'lat' in facility_points_df.columns and 'lon' in facility_points_df.columns:
        points_df = facility_points_df[facility_points_df[['lat', 'lon']].notna().all(axis=1)].copy()
        if not points_df.empty:
            marker_color = facility_color or COLOR_ACTION_SECONDARY
            size_col = facility_size_col if facility_size_col in points_df.columns and points_df[facility_size_col].notna().any() else None
            name_col = facility_name_col or ('name' if 'name' in points_df.columns else None)
            fig.add_trace(go.Scattermapbox(
                lat=points_df['lat'],
                lon=points_df['lon'],
                mode='markers',
                marker=go.scattermapbox.Marker(
                    size=points_df[size_col] if size_col else 4,
                    sizemin=4,
                    sizeref=0.5 if size_col else 1,
                    color=marker_color,
                    opacity=0.85
                ),
                text=points_df[name_col] if name_col else "Facility",
                hoverinfo='text',
                name="Facilities"
            ))
    fig.update_geos(fitbounds="locations", visible=False)
    return fig
