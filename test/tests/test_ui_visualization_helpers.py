# sentinel_project_root/test/tests/test_ui_visualization_helpers.py
# Pytest tests for UI visualization helpers in utils.ui_visualization_helpers.py for Sentinel.

import pytest
import pandas as pd
import geopandas as gpd
# import numpy as np # Not directly used here after fixture usage
import plotly.graph_objects as go
import plotly.io as pio # For accessing theme properties
from unittest.mock import patch # For mocking st.markdown
import html # For checking escaped strings

# Functions and constants to be tested
from utils.ui_visualization_helpers import (
    set_sentinel_plotly_theme_web,
    _get_theme_color,
    render_web_kpi_card,
    render_web_traffic_light_indicator,
    _create_empty_plot_figure,
    plot_layered_choropleth_map_web,
    plot_annotated_line_chart_web,
    plot_bar_chart_web,
    plot_donut_chart_web,
    plot_heatmap_web,
    MAPBOX_TOKEN_SET_FLAG as SUT_MAPBOX_TOKEN_SET_FLAG # Import the flag from SUT
)
from config import app_config

# Fixtures are sourced from conftest.py (sample_series_data_sentinel, etc.)

# Apply the Sentinel web theme once for all tests in this module
@pytest.fixture(scope="module", autouse=True)
def apply_sentinel_theme_for_tests():
    set_sentinel_plotly_theme_web()

# --- Tests for Core Theming and Color Utilities ---
def test_get_theme_color_sentinel_specifics():
    assert _get_theme_color(color_type="risk_high") == app_config.COLOR_RISK_HIGH, "Risk high color mismatch."
    assert _get_theme_color(color_type="action_primary") == app_config.COLOR_ACTION_PRIMARY, "Action primary color mismatch."
    assert _get_theme_color(color_type="positive_delta") == app_config.COLOR_POSITIVE_DELTA, "Positive delta color mismatch."
    
    custom_fallback_hex = "#123456"
    assert _get_theme_color(index=999, fallback_color=custom_fallback_hex, color_type="non_existent_color_type") == custom_fallback_hex, "Fallback color not used correctly."
    
    if hasattr(app_config, 'LEGACY_DISEASE_COLORS_WEB') and "TB" in app_config.LEGACY_DISEASE_COLORS_WEB:
        assert _get_theme_color(index="TB", color_type="disease") == app_config.LEGACY_DISEASE_COLORS_WEB["TB"], "Disease color for TB mismatch."
    else:
        # If TB not in legacy colors, it should still return a color (e.g., from general colorway or fallback)
        assert isinstance(_get_theme_color(index="TB", color_type="disease"), str), "Disease color for TB (fallback) did not return string."

    # Test getting a color from the currently set default Plotly theme's colorway
    active_template = pio.templates[pio.templates.default] # Should be "plotly+sentinel_web_theme"
    if hasattr(active_template, 'layout') and hasattr(active_template.layout, 'colorway') and active_template.layout.colorway:
        expected_theme_color_0 = active_template.layout.colorway[0]
        assert _get_theme_color(index=0, color_type="general") == expected_theme_color_0, "General theme color index 0 mismatch."
    else:
        pytest.fail("Could not access colorway from default Plotly template.")

# --- Tests for HTML Component Renderers ---
@patch('utils.ui_visualization_helpers.st.markdown') # Patch where st.markdown is called
def test_render_web_kpi_card_html_output(mock_streamlit_markdown_kpi):
    render_web_kpi_card(
        title="Total Active Cases", value_str="125", icon="😷", 
        status_level="MODERATE_CONCERN", units="patients", help_text="Active cases in the last 7 days."
    )
    mock_streamlit_markdown_kpi.assert_called_once()
    html_call_args, _ = mock_streamlit_markdown_kpi.call_args
    html_output = html_call_args[0]
    
    assert 'class="kpi-card status-moderate-concern"' in html_output, "KPI card status class incorrect."
    assert f'<h3 class="kpi-title">{html.escape("Total Active Cases")}</h3>' in html_output, "KPI title incorrect."
    assert f'<p class="kpi-value">{html.escape("125")}<span class=\'kpi-units\'>{html.escape("patients")}</span></p>' in html_output, "KPI value/units incorrect."
    assert f'title="{html.escape("Active cases in the last 7 days.")}"' in html_output, "KPI help text incorrect."
    assert html.escape("😷") in html_output, "KPI icon missing."

    mock_streamlit_markdown_kpi.reset_mock()
    render_web_kpi_card(title="Change in Rate", value_str="10.5%", delta="-2.1%", delta_is_positive=False)
    html_call_args_delta, _ = mock_streamlit_markdown_kpi.call_args
    html_output_delta = html_call_args_delta[0]
    assert '<p class="kpi-delta negative">-2.1%</p>' in html_output_delta, "Negative delta rendering incorrect."

@patch('utils.ui_visualization_helpers.st.markdown')
def test_render_web_traffic_light_indicator_html_output(mock_streamlit_markdown_tl):
    render_web_traffic_light_indicator(
        message="Data Sync Status", status_level="LOW_RISK", # Assuming LOW_RISK maps to status-low-risk
        details_text="Last sync: 5 mins ago."
    )
    mock_streamlit_markdown_tl.assert_called_once()
    html_call_args_tl, _ = mock_streamlit_markdown_tl.call_args
    html_output_tl_str = html_call_args_tl[0]
    
    assert 'class="traffic-light-dot status-low-risk"' in html_output_tl_str, "Traffic light dot class incorrect."
    assert f'<span class="traffic-light-message">{html.escape("Data Sync Status")}</span>' in html_output_tl_str, "Traffic light message incorrect."
    assert f'<span class="traffic-light-details">{html.escape("Last sync: 5 mins ago.")}</span>' in html_output_tl_str, "Traffic light details incorrect."

# --- Tests for Plotting Functionality ---
def test_create_empty_plot_figure_properties():
    test_title = "Empty Test Chart"
    test_height = 300
    test_message = "No data for this test chart."
    fig = _create_empty_plot_figure(title_str=test_title, height_val=test_height, message_str=test_message)
    
    assert isinstance(fig, go.Figure), "Did not return a Plotly Figure object."
    assert fig.layout.title.text == f"{test_title}: {test_message}", "Empty plot title incorrect."
    assert fig.layout.height == test_height, "Empty plot height incorrect."
    assert fig.layout.xaxis.visible is False and fig.layout.yaxis.visible is False, "Axes should be invisible for empty plot."
    assert len(fig.layout.annotations) == 1 and fig.layout.annotations[0].text == test_message, "Empty plot annotation incorrect."

def test_plot_annotated_line_chart_web_basic(sample_series_data_sentinel: pd.Series):
    fig = plot_annotated_line_chart_web(sample_series_data_sentinel, "Test Line Chart")
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Test Line Chart"
    assert len(fig.data) >= 1 and fig.data[0].type == 'scatter' and 'lines' in fig.data[0].mode, "Line chart trace incorrect."

    fig_empty = plot_annotated_line_chart_web(pd.Series(dtype=float), "Empty Line") # Empty series
    assert "Empty Line: No data available" in fig_empty.layout.title.text, "Empty line chart title incorrect."

def test_plot_bar_chart_web_basic(sample_bar_df_sentinel: pd.DataFrame):
    # Ensure fixture columns match expected parameter names by plot_bar_chart_web
    fig = plot_bar_chart_web(
        sample_bar_df_sentinel, x_col='category_axis', y_col='value_axis', 
        title="Test Bar Chart", color_col='grouping_col'
    )
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Test Bar Chart"
    assert len(fig.data) > 0 and fig.data[0].type == 'bar', "Bar chart trace incorrect."

    fig_empty = plot_bar_chart_web(pd.DataFrame(), x_col='x', y_col='y', title="Empty Bar")
    assert "Empty Bar: No data available" in fig_empty.layout.title.text, "Empty bar chart title incorrect."

def test_plot_donut_chart_web_basic(sample_donut_df_sentinel: pd.DataFrame):
    fig = plot_donut_chart_web(
        sample_donut_df_sentinel, labels_col='status_category', values_col='patient_counts', title="Test Donut Chart"
    )
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Test Donut Chart"
    assert len(fig.data) == 1 and fig.data[0].type == 'pie' and fig.data[0].hole > 0.4, "Donut chart trace incorrect."

    fig_empty = plot_donut_chart_web(pd.DataFrame(), labels_col='l', values_col='v', title="Empty Donut")
    assert "Empty Donut: No data available" in fig_empty.layout.title.text, "Empty donut chart title incorrect."

def test_plot_heatmap_web_basic(sample_heatmap_df_sentinel: pd.DataFrame): # Using the new fixture name
    if sample_heatmap_df_sentinel.empty:
        pytest.skip("Sample heatmap data is empty.")
    fig = plot_heatmap_web(sample_heatmap_df_sentinel, title="Test Heatmap")
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Test Heatmap"
    assert len(fig.data) == 1 and fig.data[0].type == 'heatmap', "Heatmap trace incorrect."

    fig_empty = plot_heatmap_web(pd.DataFrame(), title="Empty Heatmap")
    assert "Empty Heatmap: Invalid data for Heatmap" in fig_empty.layout.title.text, "Empty heatmap title incorrect." # Adjusted expected message

def test_plot_layered_choropleth_map_web_basic(sample_choropleth_gdf_sentinel: gpd.GeoDataFrame): # Using new fixture
    if not isinstance(sample_choropleth_gdf_sentinel, gpd.GeoDataFrame) or \
       sample_choropleth_gdf_sentinel.empty or \
       'risk_score' not in sample_choropleth_gdf_sentinel.columns or \
       'zone_id' not in sample_choropleth_gdf_sentinel.columns: # Ensure fixture provides these
        pytest.skip("Sample GDF for choropleth map test is not correctly configured or empty.")

    fig = plot_layered_choropleth_map_web(
        gdf_data=sample_choropleth_gdf_sentinel,
        value_col_name='risk_score', 
        map_title="Test Choropleth Map",
        id_col_name='zone_id'
    )
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Test Choropleth Map"
    assert fig.layout.mapbox.style is not None, "Mapbox style not set." # Theme should set a default
    assert len(fig.data) >= 1 and fig.data[0].type == 'choroplethmapbox', "Choropleth trace incorrect."

    fig_empty = plot_layered_choropleth_map_web(gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs=app_config.DEFAULT_CRS_STANDARD), value_col_name='v', map_title="Empty Geo Map", id_col_name='id')
    assert "Empty Geo Map: Geographic data unavailable" in fig_empty.layout.title.text or \
           "Empty Geo Map: Value column 'v' not found" in fig_empty.layout.title.text or \
           "Empty Geo Map: ID column 'id' for features not found" in fig_empty.layout.title.text, \
           "Empty map message incorrect." # Message can vary based on which check fails first

@patch('utils.ui_visualization_helpers.MAPBOX_TOKEN_SET_FLAG', False) # Mock the global flag in SUT
def test_plot_map_web_no_token_uses_fallback_style_override(sample_choropleth_gdf_sentinel: gpd.GeoDataFrame):
    """Test that if Mapbox token is False, a mapbox_style_override requiring token falls back."""
    if not isinstance(sample_choropleth_gdf_sentinel, gpd.GeoDataFrame) or sample_choropleth_gdf_sentinel.empty or \
       'risk_score' not in sample_choropleth_gdf_sentinel.columns or 'zone_id' not in sample_choropleth_gdf_sentinel.columns:
        pytest.skip("Sample GDF for no-token map test not correctly configured.")

    fig_map_override_no_token = plot_layered_choropleth_map_web(
        gdf_data=sample_choropleth_gdf_sentinel, value_col_name='risk_score',
        map_title="No Token Fallback Test (Override)", id_col_name='zone_id',
        mapbox_style="mapbox://styles/mapbox/satellite-streets-v11" # Style requiring token, passed as mapbox_style (new param name)
    )
    # The set_sentinel_plotly_theme_web function (and px itself) should handle this fallback.
    # The plot_layered_choropleth_map_web function passes mapbox_style to px.choropleth_mapbox.
    # Plotly Express itself will use 'carto-positron' if token is needed but not set via px.set_mapbox_access_token().
    # OR if px.set_mapbox_access_token("") was called.
    # Since we mocked MAPBOX_TOKEN_SET_FLAG, our theme setter uses "carto-positron" or "open-street-map"
    # if app_config.MAPBOX_STYLE_WEB was token-based.
    # If mapbox_style_override is given to the plot function, it overrides the theme.
    # Plotly Express's internal fallback for invalid token with token-requiring style is usually 'carto-positron'.
    assert fig_map_override_no_token.layout.mapbox.style == "carto-positron" or \
           fig_map_override_no_token.layout.mapbox.style == "open-street-map", \
           f"Map style did not fall back correctly. Got: {fig_map_override_no_token.layout.mapbox.style}"


@patch('utils.ui_visualization_helpers.MAPBOX_TOKEN_SET_FLAG', False) # Mock the global flag in SUT
def test_plot_map_web_no_token_theme_fallback(sample_choropleth_gdf_sentinel: gpd.GeoDataFrame, monkeypatch):
    """Test that if theme's MAPBOX_STYLE_WEB requires token but token is False, it falls back."""
    if not isinstance(sample_choropleth_gdf_sentinel, gpd.GeoDataFrame) or sample_choropleth_gdf_sentinel.empty or \
       'risk_score' not in sample_choropleth_gdf_sentinel.columns or 'zone_id' not in sample_choropleth_gdf_sentinel.columns:
        pytest.skip("Sample GDF for no-token theme map test not correctly configured.")

    # Temporarily set app_config.MAPBOX_STYLE_WEB to a token-requiring style
    monkeypatch.setattr(app_config, 'MAPBOX_STYLE_WEB', "mapbox://styles/mapbox/streets-v11")
    # Re-apply theme because app_config changed and MAPBOX_TOKEN_SET_FLAG is mocked to False for this test
    set_sentinel_plotly_theme_web() # This will now set a fallback style in the theme due to mocked flag
    
    fig_theme_no_token = plot_layered_choropleth_map_web(
        gdf_data=sample_choropleth_gdf_sentinel, value_col_name='risk_score',
        map_title="No Token Fallback Test (Theme)", id_col_name='zone_id',
        mapbox_style=None # Let it use the theme's mapbox_style
    )
    # The theme itself should have selected an open style because MAPBOX_TOKEN_SET_FLAG is False
    assert pio.templates[pio.templates.default].layout.mapbox.style == "carto-positron" or \
           pio.templates[pio.templates.default].layout.mapbox.style == "open-street-map"
    assert fig_theme_no_token.layout.mapbox.style == pio.templates[pio.templates.default].layout.mapbox.style, \
        "Map style did not fall back correctly based on theme with no token."
