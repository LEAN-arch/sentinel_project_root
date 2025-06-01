# sentinel_project_root/test/tests/test_ui_visualization_helpers.py
# Pytest tests for UI visualization helpers in utils.ui_visualization_helpers.py for Sentinel.

import pytest
import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go
import plotly.io as pio # For accessing theme properties
from unittest.mock import patch # For mocking st.markdown
import html # For checking escaped strings in HTML output

# Functions and constants to be tested
from utils.ui_visualization_helpers import (
    set_sentinel_plotly_theme_web, # Theme setter
    _get_theme_color,
    render_web_kpi_card,
    render_web_traffic_light_indicator,
    _create_empty_plot_figure,
    plot_layered_choropleth_map_web,
    plot_annotated_line_chart_web,
    plot_bar_chart_web,
    plot_donut_chart_web,
    plot_heatmap_web,
    MAPBOX_TOKEN_SET_FLAG as SUT_MAPBOX_TOKEN_SET_FLAG # Import the flag from System Under Test
)
from config import app_config

# Fixtures (e.g., sample_series_data_sentinel, sample_bar_df_sentinel, etc.) are sourced from conftest.py

# Apply the Sentinel web theme once for all tests in this module.
@pytest.fixture(scope="module", autouse=True)
def apply_sentinel_theme_for_tests():
    set_sentinel_plotly_theme_web()

# --- Tests for Core Theming and Color Utilities ---
def test_get_theme_color_sentinel_specifics():
    assert _get_theme_color(color_type="risk_high") == app_config.COLOR_RISK_HIGH, "Risk high color mismatch."
    assert _get_theme_color(color_type="action_primary") == app_config.COLOR_ACTION_PRIMARY, "Action primary color mismatch."
    assert _get_theme_color(color_type="positive_delta") == app_config.COLOR_POSITIVE_DELTA, "Positive delta color mismatch."
    
    custom_fallback_hex = "#ABC123" # Unique fallback for test
    assert _get_theme_color(index=12345, fallback_color=custom_fallback_hex, color_type="some_unknown_type") == custom_fallback_hex, "Fallback color not used correctly."
    
    # Test LEGACY_DISEASE_COLORS_WEB
    if hasattr(app_config, 'LEGACY_DISEASE_COLORS_WEB') and app_config.LEGACY_DISEASE_COLORS_WEB and "TB" in app_config.LEGACY_DISEASE_COLORS_WEB:
        assert _get_theme_color(index="TB", color_type="disease") == app_config.LEGACY_DISEASE_COLORS_WEB["TB"], "Disease color for 'TB' mismatch."
    else:
        # If "TB" not in legacy colors (or legacy colors not defined), it should still return a string (theme default or final fallback)
        assert isinstance(_get_theme_color(index="TB", color_type="disease", fallback_color="#CCCCCC"), str), \
            "Disease color for 'TB' (when not in specific map) did not return a string."

    # Test getting a color from the currently set default Plotly theme's colorway
    active_template = pio.templates.get(pio.templates.default) # Use .get for safety
    if active_template and hasattr(active_template, 'layout') and hasattr(active_template.layout, 'colorway') and active_template.layout.colorway:
        expected_colorway_first = active_template.layout.colorway[0]
        assert _get_theme_color(index=0, color_type="general") == expected_colorway_first, "General theme color (index 0) mismatch."
    else:
        pytest.fail("Could not access colorway from default Plotly template. Theme setup might be an issue.")


# --- Tests for HTML Component Renderers ---
@patch('utils.ui_visualization_helpers.st.markdown') # Patch where st.markdown is called
def test_render_web_kpi_card_html_output(mock_st_markdown_kpi_card):
    kpi_title = "Active Patients"
    kpi_value = "150"
    kpi_icon = "🧑‍⚕️"
    kpi_status = "ACCEPTABLE" # Pythonic, will be converted to kebab-case for CSS
    kpi_units = "patients"
    kpi_help = "Total active patients under care."
    
    render_web_kpi_card(
        title=kpi_title, value_str=kpi_value, icon=kpi_icon, 
        status_level=kpi_status, units=kpi_units, help_text=kpi_help
    )
    mock_st_markdown_kpi_card.assert_called_once()
    html_args, _ = mock_st_markdown_kpi_card.call_args
    output_html = html_args[0]
    
    assert 'class="kpi-card status-acceptable"' in output_html, "KPI card status CSS class incorrect."
    assert f'<h3 class="kpi-title">{html.escape(kpi_title)}</h3>' in output_html, "KPI title rendering incorrect."
    assert f'<p class="kpi-value">{html.escape(kpi_value)}<span class=\'kpi-units\'>{html.escape(kpi_units)}</span></p>' in output_html, "KPI value/units rendering incorrect."
    assert f'title="{html.escape(kpi_help)}"' in output_html, "KPI help text (tooltip) incorrect."
    assert html.escape(kpi_icon) in output_html, "KPI icon rendering incorrect."

    mock_st_markdown_kpi_card.reset_mock()
    render_web_kpi_card(title="Risk Change", value_str="-5.2%", delta="-1.1%", delta_is_positive=False, status_level="HIGH_CONCERN")
    html_args_delta, _ = mock_st_markdown_kpi_card.call_args
    output_html_delta = html_args_delta[0]
    assert 'class="kpi-card status-high-concern"' in output_html_delta, "Delta KPI status class incorrect."
    assert f'<p class="kpi-delta negative">{html.escape("-1.1%")}</p>' in output_html_delta, "Negative delta rendering incorrect."

@patch('utils.ui_visualization_helpers.st.markdown')
def test_render_web_traffic_light_indicator_html_output(mock_st_markdown_traffic_light):
    tl_message = "System Connection"
    tl_status = "HIGH_RISK" # Pythonic, maps to status-high-risk
    tl_details = "Connection to central server lost."

    render_web_traffic_light_indicator(message=tl_message, status_level=tl_status, details_text=tl_details)
    mock_st_markdown_traffic_light.assert_called_once()
    html_args_tl, _ = mock_st_markdown_traffic_light.call_args
    output_html_tl = html_args_tl[0]
    
    assert 'class="traffic-light-dot status-high-risk"' in output_html_tl, "Traffic light dot CSS class incorrect."
    assert f'<span class="traffic-light-message">{html.escape(tl_message)}</span>' in output_html_tl, "Traffic light message rendering incorrect."
    assert f'<span class="traffic-light-details">{html.escape(tl_details)}</span>' in output_html_tl, "Traffic light details rendering incorrect."


# --- Tests for Plotting Functionality ---
def test_create_empty_plot_figure_properties():
    chart_title = "Empty Data Plot"
    height = 350
    message = "No data points to plot for this selection."
    fig = _create_empty_plot_figure(title_str=chart_title, height_val=height, message_str=message)
    
    assert isinstance(fig, go.Figure), "_create_empty_plot_figure did not return a Plotly Figure."
    assert fig.layout.title.text == f"{chart_title}: {message}", "Empty plot title incorrect."
    assert fig.layout.height == height, "Empty plot height incorrect."
    assert not fig.layout.xaxis.visible and not fig.layout.yaxis.visible, "Axes should be invisible for empty plot."
    assert len(fig.layout.annotations) == 1 and fig.layout.annotations[0].text == message, "Empty plot annotation incorrect."

def test_plot_annotated_line_chart_web_basic(sample_series_data_sentinel: pd.Series):
    chart_title_line = "Sample Time Series"
    fig = plot_annotated_line_chart_web(sample_series_data_sentinel, chart_title_line)
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == chart_title_line
    assert len(fig.data) >= 1 and fig.data[0].type == 'scatter' and 'lines' in fig.data[0].mode.lower(), "Line chart trace configuration incorrect."

    fig_empty = plot_annotated_line_chart_web(pd.Series(dtype=float), "Empty Series Line Chart")
    assert "Empty Series Line Chart: No data available" in fig_empty.layout.title.text, "Empty line chart title message incorrect."

def test_plot_bar_chart_web_basic(sample_bar_df_sentinel: pd.DataFrame):
    chart_title_bar = "Sample Bar Distribution"
    fig = plot_bar_chart_web(
        sample_bar_df_sentinel, x_col='category_label', y_col='value_count', 
        title=chart_title_bar, color_col='grouping_col'
    )
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == chart_title_bar
    assert len(fig.data) > 0 and fig.data[0].type == 'bar', "Bar chart trace configuration incorrect."

    fig_empty = plot_bar_chart_web(pd.DataFrame(columns=['x','y']), x_col='x', y_col='y', title="Empty Bar Data")
    assert "Empty Bar Data: No data available" in fig_empty.layout.title.text, "Empty bar chart title message incorrect."

def test_plot_donut_chart_web_basic(sample_donut_df_sentinel: pd.DataFrame):
    chart_title_donut = "Status Distribution Donut"
    fig = plot_donut_chart_web(
        sample_donut_df_sentinel, labels_col='risk_level_label', values_col='case_counts', title=chart_title_donut
    )
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == chart_title_donut
    assert len(fig.data) == 1 and fig.data[0].type == 'pie' and fig.data[0].hole is not None and fig.data[0].hole > 0.4, "Donut chart trace incorrect."

    fig_empty = plot_donut_chart_web(pd.DataFrame(columns=['l','v']), labels_col='l', values_col='v', title="Empty Donut Data")
    assert "Empty Donut Data: No data available" in fig_empty.layout.title.text, "Empty donut chart title message incorrect."

def test_plot_heatmap_web_basic(sample_heatmap_df_sentinel: pd.DataFrame):
    if sample_heatmap_df_sentinel.empty:
        pytest.skip("Sample heatmap data is empty for this test.")
    chart_title_heatmap = "Correlation Heatmap"
    fig = plot_heatmap_web(sample_heatmap_df_sentinel, title=chart_title_heatmap)
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == chart_title_heatmap
    assert len(fig.data) == 1 and fig.data[0].type == 'heatmap', "Heatmap trace incorrect."

    fig_empty = plot_heatmap_web(pd.DataFrame(), title="Empty Heatmap Data")
    # Message depends on internal checks of _create_empty_plot_figure
    assert "Empty Heatmap Data: Invalid data for Heatmap" in fig_empty.layout.title.text or \
           "Empty Heatmap Data: All heatmap data non-numeric" in fig_empty.layout.title.text, \
           "Empty heatmap title message incorrect."

def test_plot_layered_choropleth_map_web_basic(sample_choropleth_gdf_sentinel: gpd.GeoDataFrame):
    chart_title_map = "Zonal Risk Score Map"
    # sample_choropleth_gdf_sentinel fixture should have 'zone_id', 'name', 'geometry', 'sample_risk_value'
    if not isinstance(sample_choropleth_gdf_sentinel, gpd.GeoDataFrame) or \
       sample_choropleth_gdf_sentinel.empty or \
       'sample_risk_value' not in sample_choropleth_gdf_sentinel.columns or \
       'zone_id' not in sample_choropleth_gdf_sentinel.columns:
        pytest.skip("Sample GDF for choropleth map test is not correctly configured or empty.")

    fig = plot_layered_choropleth_map_web(
        gdf_data=sample_choropleth_gdf_sentinel,
        value_col_name='sample_risk_value', 
        map_title=chart_title_map,
        id_col_name='zone_id' 
    )
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == chart_title_map
    assert fig.layout.mapbox.style is not None, "Mapbox style should be set by theme or default."
    assert len(fig.data) >= 1 and fig.data[0].type == 'choroplethmapbox', "Choropleth trace type incorrect."

    # Test with an empty GeoDataFrame
    empty_gdf_for_map = gpd.GeoDataFrame(columns=['geometry', 'id_col', 'val_col'], geometry='geometry', crs=app_config.DEFAULT_CRS_STANDARD)
    fig_empty = plot_layered_choropleth_map_web(empty_gdf_for_map, value_col_name='val_col', map_title="Empty Geo Map", id_col_name='id_col')
    assert "Empty Geo Map: Geographic data unavailable" in fig_empty.layout.title.text, "Empty map message incorrect."


@patch('utils.ui_visualization_helpers.MAPBOX_TOKEN_SET_FLAG', False) # Mock the global flag in SUT to be False
def test_plot_map_web_no_token_falls_back_for_override(sample_choropleth_gdf_sentinel: gpd.GeoDataFrame):
    """Tests that if mapbox_style override requires a token but token flag is False, it falls back."""
    if not isinstance(sample_choropleth_gdf_sentinel, gpd.GeoDataFrame) or sample_choropleth_gdf_sentinel.empty or \
       'sample_risk_value' not in sample_choropleth_gdf_sentinel.columns or 'zone_id' not in sample_choropleth_gdf_sentinel.columns:
        pytest.skip("Sample GDF not configured for no-token map override test.")

    fig_no_token_override = plot_layered_choropleth_map_web(
        gdf_data=sample_choropleth_gdf_sentinel, value_col_name='sample_risk_value',
        map_title="No Token Fallback (Override Style)", id_col_name='zone_id',
        mapbox_style="mapbox://styles/mapbox/streets-v11" # A style that requires a token
    )
    # Plotly Express itself often defaults to 'carto-positron' if token invalid for a private style
    # Our theme setter aims for 'carto-positron' or 'open-street-map' as well.
    assert fig_no_token_override.layout.mapbox.style.lower() in ["carto-positron", "open-street-map"], \
           f"Map style did not fall back to an open style when token is False and override needs token. Got: {fig_no_token_override.layout.mapbox.style}"

@patch('utils.ui_visualization_helpers.MAPBOX_TOKEN_SET_FLAG', False) # Mock global flag in SUT
def test_plot_map_web_no_token_theme_uses_fallback(sample_choropleth_gdf_sentinel: gpd.GeoDataFrame, monkeypatch):
    """Tests that if the theme's default map style needs a token but token flag is False, it falls back."""
    if not isinstance(sample_choropleth_gdf_sentinel, gpd.GeoDataFrame) or sample_choropleth_gdf_sentinel.empty or \
       'sample_risk_value' not in sample_choropleth_gdf_sentinel.columns or 'zone_id' not in sample_choropleth_gdf_sentinel.columns:
        pytest.skip("Sample GDF not configured for no-token theme map test.")

    # Temporarily set app_config.MAPBOX_STYLE_WEB to a token-requiring style for this test
    monkeypatch.setattr(app_config, 'MAPBOX_STYLE_WEB', "mapbox://styles/mapbox/satellite-v9")
    # Re-apply the theme because app_config changed AND MAPBOX_TOKEN_SET_FLAG is mocked to False for this test scope
    set_sentinel_plotly_theme_web() 
    
    fig_theme_fallback = plot_layered_choropleth_map_web(
        gdf_data=sample_choropleth_gdf_sentinel, value_col_name='sample_risk_value',
        map_title="No Token Fallback (Theme Style)", id_col_name='zone_id',
        mapbox_style=None # Critical: let it use the theme's default mapbox_style
    )
    # The theme itself (set by set_sentinel_plotly_theme_web) should have chosen an open style
    # because MAPBOX_TOKEN_SET_FLAG is False within this test's context.
    theme_map_style = pio.templates[pio.templates.default].layout.mapbox.style
    assert theme_map_style.lower() in ["carto-positron", "open-street-map"], \
        f"Theme's default map style should be open when token is False. Got: {theme_map_style}"
    assert fig_theme_fallback.layout.mapbox.style == theme_map_style, \
        "Plot map style did not match theme's (fallback) map style when token is False."
