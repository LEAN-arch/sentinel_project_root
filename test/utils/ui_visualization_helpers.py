# sentinel_project_root/test/utils/ui_visualization_helpers.py
# UI and Plotting helpers for Sentinel Health Co-Pilot Web Dashboards.

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import logging
import plotly.io as pio
from config import app_config # Uses the fully refactored app_config
import html
import geopandas as gpd # For type hints and map data handling
import os
from typing import Optional, List, Dict, Any, Union # Added Union

logger = logging.getLogger(__name__)

# --- Mapbox Token Handling ---
MAPBOX_TOKEN_SET_FLAG = False
try:
    _SENTINEL_MAPBOX_ACCESS_TOKEN = os.getenv("MAPBOX_ACCESS_TOKEN")
    if _SENTINEL_MAPBOX_ACCESS_TOKEN and \
       _SENTINEL_MAPBOX_ACCESS_TOKEN.strip() and \
       "YOUR_MAPBOX_ACCESS_TOKEN".lower() not in _SENTINEL_MAPBOX_ACCESS_TOKEN.lower() and \
       len(_SENTINEL_MAPBOX_ACCESS_TOKEN) > 20: # Basic validity checks
        px.set_mapbox_access_token(_SENTINEL_MAPBOX_ACCESS_TOKEN)
        MAPBOX_TOKEN_SET_FLAG = True
        logger.info("Valid Mapbox access token found and set for Plotly Express.")
    else:
        if _SENTINEL_MAPBOX_ACCESS_TOKEN: # Token exists but is invalid
            logger.warning("Mapbox access token is a placeholder or invalid. Map styles requiring a token will default to open styles.")
        else: # Token not set at all
            logger.warning("MAPBOX_ACCESS_TOKEN environment variable not set. Map styles requiring a token will default to open styles.")
except Exception as e:
    logger.error(f"Error setting Mapbox token: {e}")

# --- I. Core Theming and Color Utilities ---

def _get_theme_color(index: Any = 0, fallback_color: Optional[str] = None, color_type: str = "general") -> str:
    """
    Safely retrieves a color based on type or index from app_config or Plotly theme.
    """
    # Direct mapping for specific semantic colors
    color_map_direct = {
        "risk_high": app_config.COLOR_RISK_HIGH, "risk_moderate": app_config.COLOR_RISK_MODERATE,
        "risk_low": app_config.COLOR_RISK_LOW, "risk_neutral": app_config.COLOR_RISK_NEUTRAL,
        "action_primary": app_config.COLOR_ACTION_PRIMARY, "action_secondary": app_config.COLOR_ACTION_SECONDARY,
        "positive_delta": app_config.COLOR_POSITIVE_DELTA, "negative_delta": app_config.COLOR_NEGATIVE_DELTA,
        "text_dark": app_config.COLOR_TEXT_DARK, "headings_main": app_config.COLOR_TEXT_HEADINGS_MAIN,
        "accent_bright": app_config.COLOR_ACCENT_BRIGHT
    }
    if color_type in color_map_direct:
        return color_map_direct[color_type]

    final_fallback = fallback_color if fallback_color else app_config.COLOR_TEXT_LINK_DEFAULT

    try:
        if color_type == "disease" and hasattr(app_config, 'LEGACY_DISEASE_COLORS_WEB'):
            if isinstance(index, str) and index in app_config.LEGACY_DISEASE_COLORS_WEB:
                return app_config.LEGACY_DISEASE_COLORS_WEB[index]
            # Fallback for disease type if index not in specific map
            logger.debug(f"Disease color for '{index}' not in LEGACY_DISEASE_COLORS_WEB, using general colorway.")

        # Use Plotly's active default template colorway for 'general' or unmapped types
        active_template_name = pio.templates.default
        colorway_to_use = px.colors.qualitative.Plotly # Default Plotly palette

        if active_template_name and active_template_name in pio.templates:
            current_template = pio.templates[active_template_name]
            if hasattr(current_template, 'layout') and hasattr(current_template.layout, 'colorway') and current_template.layout.colorway:
                colorway_to_use = current_template.layout.colorway
        
        if not colorway_to_use: # Should not happen if Plotly is working
            logger.warning("No colorway found in active Plotly template. Using Plotly default.")
            colorway_to_use = px.colors.qualitative.Plotly

        num_idx_for_color = index if isinstance(index, int) else abs(hash(str(index)))
        return colorway_to_use[num_idx_for_color % len(colorway_to_use)]
            
    except Exception as e:
        logger.warning(f"Error retrieving theme color (index/key:'{index}', type:'{color_type}'): {e}. Using fallback: {final_fallback}")
    return final_fallback


def set_sentinel_plotly_theme_web():
    """ Sets a custom Plotly theme for Sentinel web reports/dashboards. """
    theme_font_family = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Noto Color Emoji"'
    
    sentinel_colorway = [
        app_config.COLOR_ACTION_PRIMARY, app_config.COLOR_RISK_LOW, app_config.COLOR_RISK_MODERATE,
        app_config.COLOR_RISK_HIGH, app_config.COLOR_ACTION_SECONDARY, _get_theme_color(index=5, fallback_color="#00ACC1"), # Teal-like
        _get_theme_color(index=6, fallback_color="#5E35B1"), # Purple-like
        _get_theme_color(index=7, fallback_color="#FF7043"), # Orange-like
        _get_theme_color(index=8, fallback_color="#E0E0E0"), # Light Grey for neutral series
        _get_theme_color(index=9, fallback_color="#795548"), # Brown-ish
    ]
    # Ensure a minimum length for the colorway
    if len(sentinel_colorway) < 8: # Standard Plotly palettes often have 8-10
        sentinel_colorway.extend(px.colors.qualitative.Safe[len(sentinel_colorway):8])

    layout_config = {
        'font': dict(family=theme_font_family, size=11, color=app_config.COLOR_TEXT_DARK),
        'paper_bgcolor': app_config.COLOR_BACKGROUND_CONTENT, # Cards often have white bg, page is light grey
        'plot_bgcolor': app_config.COLOR_BACKGROUND_PAGE, # Plot area matches page bg for seamless look
        'colorway': sentinel_colorway,
        'xaxis': dict(gridcolor=app_config.COLOR_BORDER_LIGHT, linecolor=app_config.COLOR_BORDER_MEDIUM, zerolinecolor=app_config.COLOR_BORDER_MEDIUM, zerolinewidth=1, title_font_size=12, tickfont_size=10, automargin=True, title_standoff=10),
        'yaxis': dict(gridcolor=app_config.COLOR_BORDER_LIGHT, linecolor=app_config.COLOR_BORDER_MEDIUM, zerolinecolor=app_config.COLOR_BORDER_MEDIUM, zerolinewidth=1, title_font_size=12, tickfont_size=10, automargin=True, title_standoff=10),
        'title': dict(font=dict(family=theme_font_family, size=15, color=app_config.COLOR_TEXT_HEADINGS_MAIN), x=0.03, xanchor='left', y=0.95, yanchor='top', pad=dict(t=20, b=10, l=3)), # Slightly smaller title
        'legend': dict(bgcolor='rgba(255,255,255,0.9)', bordercolor=app_config.COLOR_BORDER_LIGHT, borderwidth=0.5, orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font_size=10),
        'margin': dict(l=60, r=20, t=70, b=55) # Adjusted margins
    }
    
    # Determine Mapbox style based on token availability and config
    effective_mapbox_style = app_config.MAPBOX_STYLE_WEB
    token_required_styles = ["satellite", "streets-v11", "outdoors-v11", "light-v10", "dark-v10"] # Keywords
    if not MAPBOX_TOKEN_SET_FLAG and any(style_keyword in effective_mapbox_style.lower() for style_keyword in token_required_styles):
        effective_mapbox_style = "carto-positron" # Fallback to a good open style
        logger.info(f"Mapbox token not set, defaulting map style to '{effective_mapbox_style}'. Original config: '{app_config.MAPBOX_STYLE_WEB}'.")
    
    layout_config['mapbox'] = dict(
        style=effective_mapbox_style,
        center=dict(lat=app_config.MAP_DEFAULT_CENTER_LAT, lon=app_config.MAP_DEFAULT_CENTER_LON),
        zoom=app_config.MAP_DEFAULT_ZOOM
    )
    
    pio.templates["sentinel_web_theme"] = go.layout.Template(layout=go.Layout(**layout_config))
    pio.templates.default = "plotly+sentinel_web_theme" # Combine with base Plotly for full coverage
    logger.info("Plotly theme 'sentinel_web_theme' set as default for web reports.")

set_sentinel_plotly_theme_web() # Initialize theme on module import

# --- II. HTML-Based UI Components ---
def render_web_kpi_card(title: str, value_str: str, icon: str = "●", status_level: str = "NEUTRAL",
                        delta: Optional[str] = None, delta_is_positive: Optional[bool] = None,
                        help_text: Optional[str] = None, units: Optional[str] = ""):
    """ Renders a KPI card using HTML and custom CSS. """
    # Convert Pythonic status_level (e.g., HIGH_RISK) to kebab-case CSS class (e.g., status-high-risk)
    css_status_class = f"status-{status_level.lower().replace('_', '-')}"

    delta_html = ""
    if delta is not None and str(delta).strip():
        delta_indicator_class = "neutral" # Default class for delta
        if delta_is_positive is True: delta_indicator_class = "positive"
        elif delta_is_positive is False: delta_indicator_class = "negative"
        delta_html = f'<p class="kpi-delta {delta_indicator_class}">{html.escape(str(delta))}</p>'

    tooltip_attr = f'title="{html.escape(str(help_text))}"' if help_text and str(help_text).strip() else ''
    units_html = f"<span class='kpi-units'>{html.escape(str(units))}</span>" if units and str(units).strip() else ""
    
    kpi_card_html = f"""
    <div class="kpi-card {css_status_class}" {tooltip_attr}>
        <div class="kpi-card-header">
            <div class="kpi-icon">{html.escape(str(icon))}</div>
            <h3 class="kpi-title">{html.escape(str(title))}</h3>
        </div>
        <div class="kpi-body">
            <p class="kpi-value">{html.escape(str(value_str))}{units_html}</p>
            {delta_html}
        </div>
    </div>
    """.replace("\n", " ").strip() # Minify HTML slightly
    st.markdown(kpi_card_html, unsafe_allow_html=True)

def render_web_traffic_light_indicator(message: str, status_level: str, details_text: Optional[str] = None):
    """ Renders a traffic light style indicator. """
    css_dot_class = f"status-{status_level.lower().replace('_', '-')}"
    details_html = f'<span class="traffic-light-details">{html.escape(str(details_text))}</span>' if details_text and str(details_text).strip() else ""
    
    traffic_light_html = f"""
    <div class="traffic-light-indicator">
        <span class="traffic-light-dot {css_dot_class}"></span>
        <span class="traffic-light-message">{html.escape(str(message))}</span>
        {details_html}
    </div>
    """.replace("\n", " ").strip()
    st.markdown(traffic_light_html, unsafe_allow_html=True)


# --- III. Plotly Chart Generation Functions ---
def _create_empty_plot_figure(title_str: str, height_val: Optional[int], message_str: str = "No data available to display.") -> go.Figure:
    """ Creates a standardized empty Plotly figure with a message. """
    fig = go.Figure()
    final_plot_height = height_val if height_val is not None else app_config.WEB_PLOT_DEFAULT_HEIGHT
    fig.update_layout(
        title_text=f"{title_str}: {message_str}", 
        height=final_plot_height,
        xaxis={'visible': False}, yaxis={'visible': False},
        annotations=[dict(
            text=message_str, xref="paper", yref="paper", 
            showarrow=False, font=dict(size=12, color=_get_theme_color(color_type="risk_neutral"))
        )]
    )
    return fig

def plot_annotated_line_chart_web(
    data_series: pd.Series, chart_title: str, y_axis_label: str = "Value",
    line_color: Optional[str] = None,
    target_ref_line: Optional[float] = None, target_ref_label: Optional[str] = None,
    show_conf_interval: bool = False, lower_ci: Optional[pd.Series] = None, upper_ci: Optional[pd.Series] = None,
    chart_height: Optional[int] = None, show_anomalies: bool = False, # Renamed anomaly param
    date_format: str = "%d-%b-%y", y_is_count: bool = False # Renamed params
) -> go.Figure:
    final_height = chart_height if chart_height is not None else app_config.WEB_PLOT_COMPACT_HEIGHT
    if not isinstance(data_series, pd.Series) or data_series.empty:
        return _create_empty_plot_figure(chart_title, final_height)
    
    series_clean = pd.to_numeric(data_series, errors='coerce') # Ensure numeric
    if series_clean.isnull().all():
        return _create_empty_plot_figure(chart_title, final_height, "All data non-numeric or NaN.")

    fig = go.Figure()
    color = line_color if line_color else _get_theme_color(0) # Default to first theme color
    
    y_hoverformat = 'd' if y_is_count else ',.1f'
    hovertemplate = f'<b>Date</b>: %{{x|{date_format}}}<br><b>{y_axis_label}</b>: %{{customdata:{y_hoverformat}}}<extra></extra>'
    
    fig.add_trace(go.Scatter(
        x=series_clean.index, y=series_clean.values, mode="lines+markers", name=y_axis_label,
        line=dict(color=color, width=2), marker=dict(size=5),
        customdata=series_clean.values, hovertemplate=hovertemplate
    ))

    if show_conf_interval and isinstance(lower_ci, pd.Series) and isinstance(upper_ci, pd.Series) and \
       not lower_ci.empty and not upper_ci.empty:
        common_idx_ci = series_clean.index.intersection(lower_ci.index).intersection(upper_ci.index)
        if not common_idx_ci.empty:
            l_ci = pd.to_numeric(lower_ci.reindex(common_idx_ci), errors='coerce')
            u_ci = pd.to_numeric(upper_ci.reindex(common_idx_ci), errors='coerce')
            valid_ci = l_ci.notna() & u_ci.notna() & (u_ci >= l_ci)
            if valid_ci.any():
                x_vals_ci, y_upper_ci, y_lower_ci = common_idx_ci[valid_ci], u_ci[valid_ci], l_ci[valid_ci]
                fill_color_rgba = f"rgba({','.join(str(int(c,16)) for c in (color[1:3], color[3:5], color[5:7]))},0.15)" if color.startswith('#') and len(color)==7 else "rgba(100,100,100,0.15)"
                fig.add_trace(go.Scatter(
                    x=list(x_vals_ci) + list(x_vals_ci[::-1]), 
                    y=list(y_upper_ci.values) + list(y_lower_ci.values[::-1]), 
                    fill="toself", fillcolor=fill_color_rgba, 
                    line=dict(width=0), name="Confidence Interval", hoverinfo='skip'
                ))
    
    if target_ref_line is not None:
        label = target_ref_label if target_ref_label else f"Target: {target_ref_line:,.2f}"
        fig.add_hline(y=target_ref_line, line_dash="dash", line_color=_get_theme_color(color_type="risk_moderate"), 
                      line_width=1.2, annotation_text=label, annotation_position="bottom right", annotation_font_size=9)

    if show_anomalies and len(series_clean.dropna()) > 7 and series_clean.nunique() > 2: # Basic conditions for anomaly detection
        q1, q3 = series_clean.quantile(0.25), series_clean.quantile(0.75)
        iqr = q3 - q1
        if pd.notna(iqr) and iqr > 1e-7: # Avoid issues with flat data
            upper_bound, lower_bound = q3 + 1.5 * iqr, q1 - 1.5 * iqr
            anomalies = series_clean[(series_clean < lower_bound) | (series_clean > upper_bound)]
            if not anomalies.empty:
                fig.add_trace(go.Scatter(
                    x=anomalies.index, y=anomalies.values, mode='markers', 
                    marker=dict(color=_get_theme_color(color_type="risk_high"), size=7, symbol='circle-open', line=dict(width=1.5)), 
                    name='Anomaly', customdata=anomalies.values, 
                    hovertemplate=(f'<b>Anomaly Date</b>: %{{x|{date_format}}}<br><b>Value</b>: %{{customdata:{y_hoverformat}}}<extra></extra>')
                ))

    x_axis_title = series_clean.index.name if series_clean.index.name and str(series_clean.index.name).strip() else "Date/Time"
    yaxis_config = dict(title_text=y_axis_label, rangemode='tozero' if y_is_count and series_clean.notna().any() and series_clean[series_clean.notna()].min() >= 0 else 'normal')
    if y_is_count: yaxis_config['tickformat'] = 'd'
    
    fig.update_layout(title_text=chart_title, xaxis_title=x_axis_title, yaxis=yaxis_config, 
                      height=final_height, hovermode="x unified", legend=dict(traceorder='normal'))
    return fig

def plot_bar_chart_web(
    df: pd.DataFrame, x_col: str, y_col: str, title: str,
    color_col: Optional[str] = None, barmode: str = 'group', orientation: str = 'v',
    y_axis_label: Optional[str] = None, x_axis_label: Optional[str] = None,
    chart_height: Optional[int] = None, text_auto: Union[bool, str] = True, # Allow True/False or format string
    sort_by: Optional[str] = None, sort_ascending: bool = True, # Renamed sort params
    text_format: Optional[str] = None, y_is_count: bool = False,
    color_map: Optional[Dict] = None # Renamed color_discrete_map
) -> go.Figure:
    final_height = chart_height if chart_height is not None else app_config.WEB_PLOT_DEFAULT_HEIGHT
    if not isinstance(df, pd.DataFrame) or df.empty or x_col not in df.columns or y_col not in df.columns:
        return _create_empty_plot_figure(title, final_height)
    
    df_plot = df.copy()
    df_plot[x_col] = df_plot[x_col].astype(str) # Ensure x-axis is categorical
    df_plot[y_col] = pd.to_numeric(df_plot[y_col], errors='coerce')
    if y_is_count: df_plot[y_col] = df_plot[y_col].round().astype('Int64') # Use nullable Int for counts
    df_plot.dropna(subset=[x_col, y_col], inplace=True)
    if df_plot.empty: return _create_empty_plot_figure(title, final_height, f"No valid data for x='{x_col}', y='{y_col}'.")
    
    # Sorting
    if sort_by and sort_by in df_plot.columns:
        try: 
            # Handle numeric or string sorting appropriately
            sort_key_func = lambda s: pd.to_numeric(s, errors='ignore') if df_plot[sort_by].dtype in [np.int64, np.float64, 'Int64'] else s.astype(str)
            df_plot.sort_values(by=sort_by, ascending=sort_ascending, inplace=True, na_position='last', key=sort_key_func if sort_by != x_col else None)
        except Exception as e: logger.warning(f"Bar chart ('{title}') sort failure on '{sort_by}': {e}")
    
    # Determine effective text format for display on bars
    effective_text_format = text_format if text_format else ('.0f' if y_is_count else '.1f') # d3-format string
    
    # Axis labels
    y_title_text = y_axis_label if y_axis_label else y_col.replace('_', ' ').title()
    x_title_text = x_axis_label if x_axis_label else x_col.replace('_', ' ').title()
    legend_title_text = color_col.replace('_',' ').title() if color_col and color_col in df_plot.columns else None

    # Smart color mapping
    final_color_map = color_map
    if not final_color_map and color_col and color_col in df_plot:
        unique_color_values = df_plot[color_col].dropna().unique()
        # Attempt to use legacy disease colors if applicable
        if hasattr(app_config,"LEGACY_DISEASE_COLORS_WEB") and any(str(val) in app_config.LEGACY_DISEASE_COLORS_WEB for val in unique_color_values):
            final_color_map = {str(val): _get_theme_color(str(val), color_type="disease", fallback_color=_get_theme_color(abs(hash(str(val)))%10)) for val in unique_color_values}

    fig = px.bar(df_plot, x=x_col, y=y_col, title=title, color=color_col, barmode=barmode,
                 orientation=orientation, height=final_height, labels={y_col:y_title_text, x_col:x_title_text, color_col:legend_title_text if legend_title_text else ""},
                 text_auto=text_auto, color_discrete_map=final_color_map)
    
    # Hovertemplate and texttemplate customization
    hover_val_format_str = 'd' if y_is_count else effective_text_format
    if orientation == 'v':
        base_hover = f'<b>{x_title_text}</b>: %{{x}}<br><b>{y_title_text}</b>: %{{y:{hover_val_format_str}}}'
        text_template = f'%{{y:{effective_text_format}}}' if text_auto else None
    else: # Horizontal
        base_hover = f'<b>{y_title_text}</b>: %{{y}}<br><b>{x_title_text}</b>: %{{x:{hover_val_format_str}}}'
        text_template = f'%{{x:{effective_text_format}}}' if text_auto else None
    
    hovertemplate_str = base_hover
    customdata_cols = []
    if color_col and color_col in df_plot.columns and legend_title_text:
        hovertemplate_str += f'<br><b>{legend_title_text}</b>: %{{customdata[0]}}'
        customdata_cols.append(color_col)
    hovertemplate_str += '<extra></extra>' # Remove trace info

    fig.update_traces(
        marker_line_width=0.5, marker_line_color='rgba(0,0,0,0.2)', 
        textfont_size=9, textangle=0, 
        textposition='auto' if orientation == 'v' else 'outside', 
        cliponaxis=False, 
        texttemplate=text_template, 
        hovertemplate=hovertemplate_str,
        customdata=df_plot[customdata_cols] if customdata_cols else None
    )
    
    # Axis configurations
    yaxis_cfg = {'title_text': y_title_text}
    xaxis_cfg = {'title_text': x_title_text}
    val_axis, cat_axis = (yaxis_cfg, xaxis_cfg) if orientation == 'v' else (xaxis_cfg, yaxis_cfg)
    
    if y_is_count: val_axis['tickformat'] = 'd'; val_axis['rangemode'] = 'tozero'
    # Category ordering for sorted charts
    if sort_by == (x_col if orientation=='v' else y_col): 
        cat_axis['categoryorder'] = 'array'
        cat_axis['categoryarray'] = df_plot[x_col if orientation=='v' else y_col].tolist()
    elif orientation == 'h' and (not sort_by or sort_by == y_col): # Default sort for horizontal bars
        cat_axis['categoryorder']='total ascending' if sort_ascending else 'total descending'
        
    fig.update_layout(yaxis=yaxis_cfg, xaxis=xaxis_cfg, uniformtext_minsize=7, uniformtext_mode='hide', legend_title_text=legend_title_text)
    return fig

def plot_donut_chart_web(
    df: pd.DataFrame, labels_col: str, values_col: str, title: str, 
    chart_height: Optional[int] = None, color_map: Optional[Dict] = None, 
    pull_amount: float = 0.03, # Renamed pull_segments
    center_annotation_text: Optional[str] = None, # Renamed center_text
    values_are_absolute_counts: bool = True # Renamed values_are_counts
) -> go.Figure:
    final_height = chart_height if chart_height is not None else app_config.WEB_PLOT_COMPACT_HEIGHT + 40 # Slightly taller for donut
    if not isinstance(df, pd.DataFrame) or df.empty or labels_col not in df.columns or values_col not in df.columns:
        return _create_empty_plot_figure(title, final_height)
    
    df_plot = df.copy()
    df_plot[values_col] = pd.to_numeric(df_plot[values_col], errors='coerce').fillna(0)
    if values_are_absolute_counts: 
        df_plot[values_col] = df_plot[values_col].round().astype('Int64')
    df_plot = df_plot[df_plot[values_col] > 0] # Only plot positive values
    if df_plot.empty: return _create_empty_plot_figure(title, final_height, "No positive data to display.")
    
    df_plot.sort_values(by=values_col, ascending=False, inplace=True) # Sort for consistent pull/display
    df_plot[labels_col] = df_plot[labels_col].astype(str)

    # Color mapping
    plot_colors = None
    if color_map:
        plot_colors = [color_map.get(str(lbl), _get_theme_color(i)) for i, lbl in enumerate(df_plot[labels_col])]
    elif hasattr(app_config,"LEGACY_DISEASE_COLORS_WEB") and any(str(lbl) in app_config.LEGACY_DISEASE_COLORS_WEB for lbl in df_plot[labels_col]):
        plot_colors = [_get_theme_color(str(lbl), color_type="disease", fallback_color=_get_theme_color(i)) for i,lbl in enumerate(df_plot[labels_col])]
    else: # Default theme colors
        plot_colors = [_get_theme_color(i) for i in range(len(df_plot[labels_col]))]

    hover_val_format_str = 'd' if values_are_absolute_counts else '.2f'
    hovertemplate_str = f'<b>%{{label}}</b><br>Value: %{{value:{hover_val_format_str}}}<br>Percent: %{{percent}}<extra></extra>'
    
    fig = go.Figure(data=[go.Pie(
        labels=df_plot[labels_col], values=df_plot[values_col], 
        hole=0.55, # Slightly larger hole for donut
        pull=[pull_amount if i < min(3, len(df_plot)) else 0 for i in range(len(df_plot))], # Pull top few segments
        textinfo='label+percent', insidetextorientation='radial',
        hoverinfo='label+value+percent', hovertemplate=hovertemplate_str,
        marker=dict(colors=plot_colors, line=dict(color=app_config.COLOR_BACKGROUND_WHITE, width=1.5)), # Clearer segment lines
        sort=False # Already sorted df_plot
    )])
    
    annotations_list = []
    if center_annotation_text:
        annotations_list.append(dict(text=str(center_annotation_text), x=0.5, y=0.5, font_size=14, showarrow=False, font_color=app_config.COLOR_TEXT_DARK))
    
    fig.update_layout(title_text=title, height=final_height, showlegend=True, 
                      legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="right", x=1.15, traceorder="normal", font_size=9), 
                      annotations=annotations_list if annotations_list else None,
                      margin=dict(l=20, r=100, t=60, b=20)) # Adjusted margins for legend
    return fig

def plot_heatmap_web(
    matrix_df: pd.DataFrame, title: str, chart_height: Optional[int] = None, # Renamed height
    colorscale_name: str = "RdBu_r", z_midpoint: Optional[float] = 0, # Renamed params
    show_text_on_cells: bool = True, text_display_format: str = ".2f", # Renamed params
    show_colorbar_legend: bool = True # Renamed param
) -> go.Figure:
    final_height = chart_height if chart_height is not None else app_config.WEB_PLOT_DEFAULT_HEIGHT + 50 # Heatmaps can be taller
    if not isinstance(matrix_df, pd.DataFrame) or matrix_df.empty:
        return _create_empty_plot_figure(title, final_height, "Invalid data for Heatmap.")
    
    df_numeric = matrix_df.copy().apply(pd.to_numeric, errors='coerce') # Ensure all data is numeric
    if df_numeric.isnull().all().all(): # If all values became NaN
        return _create_empty_plot_figure(title, final_height, "All heatmap data non-numeric or empty.")
    
    z_values = df_numeric.values
    text_values = None
    if show_text_on_cells:
        try:
            # Attempt to format text based on text_display_format
            # This requires z_values to be numeric, which they are after coercion.
            text_values = np.vectorize(lambda x: f"{x:{text_display_format}}" if pd.notna(x) else "")(z_values)
        except Exception as e_fmt:
            logger.warning(f"Could not apply text_display_format '{text_display_format}' to heatmap: {e_fmt}. Using default numeric text.")
            text_values = np.around(z_values, decimals=2) # Fallback if format string fails

    # Determine zmid: only set if data spans both positive and negative values significantly
    final_zmid = z_midpoint
    z_flat_no_nan = z_values[~np.isnan(z_values)]
    if len(z_flat_no_nan) > 0:
        if not (np.any(z_flat_no_nan < (0 - 1e-6)) and np.any(z_flat_no_nan > (0 + 1e-6))): # Check if data truly crosses zero
            final_zmid = None # Let Plotly auto-determine if not bipolar
    else: # All NaNs
        final_zmid = None
        
    fig = go.Figure(data=go.Heatmap(
        z=z_values, x=df_numeric.columns.astype(str).tolist(), y=df_numeric.index.astype(str).tolist(),
        colorscale=colorscale_name, zmid=final_zmid, 
        text=text_values if show_text_on_cells else None,
        texttemplate="%{text}" if show_text_on_cells and text_values is not None else None, # Use if text is pre-formatted
        hoverongaps=False, xgap=1, ygap=1,
        colorbar=dict(thickness=15, len=0.85, tickfont_size=9, title_side="right", outlinewidth=0.5, outlinecolor=app_config.COLOR_BORDER_MEDIUM) if show_colorbar_legend else None
    ))
    
    max_x_label_len = max((len(str(c)) for c in df_numeric.columns), default=0)
    x_tick_angle = -40 if len(df_numeric.columns) > 7 or max_x_label_len > 8 else 0 # More aggressive angle for long/many labels
    
    fig.update_layout(title_text=title, height=final_height, 
                      xaxis_showgrid=False, yaxis_showgrid=False,
                      xaxis_tickangle=x_tick_angle, yaxis_autorange='reversed',
                      plot_bgcolor=app_config.COLOR_BACKGROUND_WHITE) # Use white for clearer heatmap plot area
    return fig

def plot_layered_choropleth_map_web(
    gdf_data: gpd.GeoDataFrame, value_col_name: str, map_title: str,
    id_col_name: str = 'zone_id', # Assumed GeoJSON feature ID key path prefix
    color_scale_name: str = "Viridis_r", # Renamed color_scale
    hover_data_cols_list: Optional[List[str]] = None, # Renamed hover_data_cols
    facility_points_gdf: Optional[gpd.GeoDataFrame] = None, # For overlaying points
    facility_size_col: Optional[str] = None, # Renamed
    facility_hover_name_col: Optional[str] = None, # Renamed
    facility_color: Optional[str] = None, # Renamed
    map_height_val: Optional[int] = None, # Renamed
    map_center_lat: Optional[float] = None, map_center_lon: Optional[float] = None, # Renamed
    map_zoom_level: Optional[int] = None, mapbox_style: Optional[str] = None # Renamed
) -> go.Figure:
    final_map_height = map_height_val if map_height_val is not None else app_config.WEB_MAP_DEFAULT_HEIGHT
    if not isinstance(gdf_data, gpd.GeoDataFrame) or gdf_data.empty or 'geometry' not in gdf_data.columns:
        return _create_empty_plot_figure(map_title, final_map_height, "Geographic data (GDF) unavailable or invalid.")
    if value_col_name not in gdf_data.columns:
        return _create_empty_plot_figure(map_title, final_map_height, f"Value column '{value_col_name}' not found in geographic data.")
    if id_col_name not in gdf_data.columns:
         return _create_empty_plot_figure(map_title, final_map_height, f"ID column '{id_col_name}' for features not found in geographic data.")

    gdf_plot = gdf_data.copy()
    # Ensure value_col_name is numeric for color scale
    gdf_plot[value_col_name] = pd.to_numeric(gdf_plot[value_col_name], errors='coerce')
    # Drop rows where the value to be plotted is NaN, as choropleth cannot handle them
    gdf_plot.dropna(subset=[value_col_name, id_col_name, 'geometry'], inplace=True)
    if gdf_plot.empty:
        return _create_empty_plot_figure(map_title, final_map_height, f"No valid data remaining for '{value_col_name}' after cleaning.")

    # Convert GDF to GeoJSON for Plotly if not already (Plotly Express handles this internally, but manual go.Choroplethmapbox needs it)
    # For px.choropleth_mapbox, it can take GDF directly.
    
    # Determine hover data
    hover_name_col = 'name' if 'name' in gdf_plot.columns else id_col_name # Default hover name
    final_hover_data = {col: True for col in hover_data_cols_list if col in gdf_plot.columns} if hover_data_cols_list else {value_col_name: True}
    if hover_name_col not in final_hover_data : final_hover_data[hover_name_col] = True # Ensure name is in hover

    # Mapbox style determination
    effective_mapbox_style = mapbox_style if mapbox_style else pio.templates[pio.templates.default].layout.mapbox.style
    
    fig = px.choropleth_mapbox(
        gdf_plot,
        geojson=gdf_plot.geometry.__geo_interface__, # Pass GeoJSON features
        locations=gdf_plot[id_col_name],             # Link to GeoJSON features by ID
        featureidkey=f"properties.{id_col_name}",   # Path to ID in GeoJSON properties
        color=value_col_name,
        color_continuous_scale=color_scale_name,
        hover_name=hover_name_col,
        hover_data=final_hover_data,
        mapbox_style=effective_mapbox_style,
        center={"lat": map_center_lat if map_center_lat is not None else app_config.MAP_DEFAULT_CENTER_LAT, 
                "lon": map_center_lon if map_center_lon is not None else app_config.MAP_DEFAULT_CENTER_LON},
        zoom=map_zoom_level if map_zoom_level is not None else app_config.MAP_DEFAULT_ZOOM,
        opacity=0.7,
        height=final_map_height,
        title=map_title
    )
    fig.update_layout(margin={"r":0,"t":45,"l":0,"b":0}, mapbox_accesstoken=os.getenv("MAPBOX_ACCESS_TOKEN")) # Token here for px if not globally set

    # Overlay facility points if provided
    if isinstance(facility_points_gdf, gpd.GeoDataFrame) and not facility_points_gdf.empty and 'geometry' in facility_points_gdf.columns:
        points_plot = facility_points_gdf[facility_points_gdf.geometry.is_valid & ~facility_points_gdf.geometry.is_empty].copy()
        if not points_plot.empty:
            points_plot['lat_val'] = points_plot.geometry.y
            points_plot['lon_val'] = points_plot.geometry.x
            
            marker_color_actual = facility_color if facility_color else _get_theme_color(color_type="action_secondary")
            size_col_actual = facility_size_col if facility_size_col and facility_size_col in points_plot.columns and points_plot[facility_size_col].notna().any() else None
            
            hover_name_points = facility_hover_name_col if facility_hover_name_col and facility_hover_name_col in points_plot.columns else ('name' if 'name' in points_plot.columns else None)

            fig.add_trace(go.Scattermapbox(
                lat=points_plot['lat_val'], lon=points_plot['lon_val'],
                mode='markers',
                marker=go.scattermapbox.Marker(
                    size=points_plot[size_col_actual] if size_col_actual else 8, # Dynamic or fixed size
                    sizemin=4, sizeref= 0.5 if size_col_actual else 1, # Adjust sizeref if using size col
                    color=marker_color_actual,
                    opacity=0.85,
                ),
                text=points_plot[hover_name_points] if hover_name_points else "Facility", # Tooltip text
                hoverinfo='text', name="Facilities"
            ))
    fig.update_geos(fitbounds="locations", visible=False) # Fit to choropleth bounds
    return fig
