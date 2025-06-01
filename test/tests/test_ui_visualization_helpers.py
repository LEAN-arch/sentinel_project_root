# sentinel_project_root/test/tests/test_ui_visualization_helpers.py
# Pytest tests for the refactored UI visualization helpers in utils.ui_visualization_helpers.py
# Aligned with "Sentinel Health Co-Pilot" redesign (primarily for Tiers 2/3 web views).

import pytest
import pandas as pd
import geopandas as gpd # For sample_choropleth_gdf fixture typing, not direct use unless checking gdf properties
import numpy as np
import plotly.graph_objects as go
from unittest.mock import patch, MagicMock # For mocking st.markdown and potentially other Streamlit elements

# Functions and constants to be tested (import the _web suffixed ones where applicable)
from utils.ui_visualization_helpers import (
    set_sentinel_plotly_theme_web, # Ensure theme is set before tests that check layout
    _get_theme_color,
    render_web_kpi_card,
    render_web_traffic_light_indicator,
    _create_empty_plot_figure,
    plot_layered_choropleth_map_web,
    plot_annotated_line_chart_web,
    plot_bar_chart_web,
    plot_donut_chart_web,
    plot_heatmap_web,
    # Import the module-level flag for easier mocking in tests if needed for specific mapbox scenarios
    MAPBOX_TOKEN_SET_FLAG as SUT_MAPBOX_TOKEN_SET_FLAG # SUT = System Under Test
)
# Import app_config to use its color definitions and other settings in assertions
from config import app_config # The NEW, redesigned app_config

# Fixtures are automatically sourced by pytest from conftest.py in the same directory or parent.
# Expected fixtures: sample_series_data_sentinel, sample_bar_df_sentinel,
#                    sample_donut_df_sentinel, sample_heatmap_df_sentinel,
#                    sample_choropleth_gdf (from original conftest.py, needs name and zone_id).

# Apply the Sentinel web theme once for all tests in this module.
# This ensures that plot layouts are tested against the intended theme.
@pytest.fixture(scope="module", autouse=True)
def apply_sentinel_web_theme():
    set_sentinel_plotly_theme_web()

# --- Tests for Core Theming and Color Utilities ---
def test_get_theme_color_sentinel_specifics():
    assert _get_theme_color(color_type="risk_high") == app_config.COLOR_RISK_HIGH
    assert _get_theme_color(color_type="action_primary") == app_config.COLOR_ACTION_PRIMARY
    assert _get_theme_color(color_type="positive_delta") == app_config.COLOR_POSITIVE_DELTA
    
    # Test fallback behavior for unknown color_type
    custom_fallback = "#ABCDEF"
    assert _get_theme_color(index=999, fallback_color=custom_fallback, color_type="non_existent_type") == custom_fallback
    
    # Test LEGACY_DISEASE_COLORS_WEB from app_config
    if app_config.LEGACY_DISEASE_COLORS_WEB and "TB" in app_config.LEGACY_DISEASE_COLORS_WEB:
        assert _get_theme_color(index="TB", color_type="disease") == app_config.LEGACY_DISEASE_COLORS_WEB["TB"]
    
    # Test getting a color from the currently set default Plotly theme's colorway
    # pio.templates.default should be "plotly+sentinel_web_theme"
    default_template_obj = go.layout.Template(pio.templates[pio.templates.default])
    expected_first_colorway = default_template_obj.layout.colorway[0]
    assert _get_theme_color(index=0, color_type="general") == expected_first_colorway


# --- Tests for HTML Component Renderers (using mocking st.markdown) ---
@patch('utils.ui_visualization_helpers.st.markdown') # Mock st.markdown in the module where it's called
def test_render_web_kpi_card_html_structure(mock_st_markdown_kpi):
    render_web_kpi_card(title="Test KPI A", value_str="250", icon="🚀", status_level="GOOD_PERFORMANCE", units="tests", help_text="A test help.")
    mock_st_markdown_kpi.assert_called_once()
    call_args_kpi, _ = mock_st_markdown_kpi.call_args
    html_output_kpi = call_args_kpi[0] # The HTML string
    
    assert 'class="kpi-card status-good-performance"' in html_output_kpi # Check status CSS class
    assert '<h3 class="kpi-title">Test KPI A</h3>' in html_output_kpi
    assert '<p class="kpi-value">250<span class=\'kpi-units\'>tests</span></p>' in html_output_kpi
    assert 'title="A test help."' in html_output_kpi

    mock_st_markdown_kpi.reset_mock()
    render_web_kpi_card(title="Delta KPI B", value_str="-10", delta="-2 (15%)", delta_is_positive=False, status_level="HIGH_CONCERN")
    call_args_delta, _ = mock_st_markdown_kpi.call_args
    html_output_delta = call_args_delta[0]
    assert 'class="kpi-card status-high-concern"' in html_output_delta
    assert '<p class="kpi-delta negative">-2 (15%)</p>' in html_output_delta


@patch('utils.ui_visualization_helpers.st.markdown')
def test_render_web_traffic_light_indicator_html_structure(mock_st_markdown_tl):
    render_web_traffic_light_indicator(message="System Stability", status_level="ACCEPTABLE", details_text="All parameters normal.")
    mock_st_markdown_tl.assert_called_once()
    call_args_tl, _ = mock_st_markdown_tl.call_args
    html_output_tl = call_args_tl[0]
    
    assert 'class="traffic-light-dot status-acceptable"' in html_output_tl # Check status CSS class for dot
    assert '<span class="traffic-light-message">System Stability</span>' in html_output_tl
    assert '<span class="traffic-light-details">All parameters normal.</span>' in html_output_tl


# --- Tests for Plotting Functionality (Primarily for correct Figure generation & empty data handling) ---
def test_create_empty_plot_figure_output():
    fig = _create_empty_plot_figure(title_str="Empty Chart Test", height_val=350, message_str="Custom empty message.")
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Empty Chart Test: Custom empty message."
    assert fig.layout.height == 350
    assert fig.layout.xaxis.visible is False and fig.layout.yaxis.visible is False
    assert len(fig.layout.annotations) == 1 and fig.layout.annotations[0].text == "Custom empty message."


def test_plot_annotated_line_chart_web_functionality(sample_series_data_sentinel):
    chart_title_line = "Sentinel Line Chart Test"
    fig_line = plot_annotated_line_chart_web(sample_series_data_sentinel, chart_title_line)
    assert isinstance(fig_line, go.Figure)
    assert fig_line.layout.title.text == chart_title_line
    assert len(fig_line.data) >= 1 # At least one scatter trace for the line
    
    # Test empty data handling
    fig_empty_line = plot_annotated_line_chart_web(pd.Series(dtype=float), "Empty Test Line")
    assert "Empty Test Line: No data available" in fig_empty_line.layout.title.text


def test_plot_bar_chart_web_functionality(sample_bar_df_sentinel):
    chart_title_bar = "Sentinel Bar Chart Test"
    # Ensure fixture columns match expected parameter names by plot_bar_chart_web
    fig_bar = plot_bar_chart_web(
        sample_bar_df_sentinel, x_col_bar='category_col', y_col_bar='value_col', title_bar=chart_title_bar, color_col_bar='group_col'
    )
    assert isinstance(fig_bar, go.Figure)
    assert fig_bar.layout.title.text == chart_title_bar
    assert len(fig_bar.data) > 0 and fig_bar.data[0].type == 'bar'

    fig_empty_bar = plot_bar_chart_web(pd.DataFrame(), x_col_bar='c', y_col_bar='v', title_bar="Empty Bar Test")
    assert "Empty Bar Test: No data available" in fig_empty_bar.layout.title.text


def test_plot_donut_chart_web_functionality(sample_donut_df_sentinel):
    chart_title_donut = "Sentinel Donut Chart Test"
    # Ensure fixture columns match expected parameter names
    fig_donut = plot_donut_chart_web(sample_donut_df_sentinel, labels_col='status_labels', values_col='counts_values', title=chart_title_donut)
    assert isinstance(fig_donut, go.Figure)
    assert fig_donut.layout.title.text == chart_title_donut
    assert len(fig_donut.data) == 1 and fig_donut.data[0].type == 'pie' and fig_donut.data[0].hole > 0

    fig_empty_donut = plot_donut_chart_web(pd.DataFrame(), labels_col='l', values_col='v', title="Empty Donut Test")
    assert "Empty Donut Test: No data available" in fig_empty_donut.layout.title.text


def test_plot_heatmap_web_functionality(sample_heatmap_df): # Assuming sample_heatmap_df fixture exists
    chart_title_heatmap = "Sentinel Heatmap Test"
    fig_heatmap = plot_heatmap_web(sample_heatmap_df, title=chart_title_heatmap)
    assert isinstance(fig_heatmap, go.Figure)
    assert fig_heatmap.layout.title.text == chart_title_heatmap
    assert len(fig_heatmap.data) == 1 and fig_heatmap.data[0].type == 'heatmap'

    fig_empty_heatmap = plot_heatmap_web(pd.DataFrame(), title="Empty Heatmap Test")
    assert "Empty Heatmap Test: Invalid data" in fig_empty_heatmap.layout.title.text # Message from _create_empty_plot_figure if df empty

# sample_choropleth_gdf needs 'zone_id', 'name' (for hover_name), and a value column for color.
# Ensure conftest.py's sample_choropleth_gdf provides these.
def test_plot_layered_choropleth_map_web_functionality(sample_choropleth_gdf):
    chart_title_map = "Sentinel Choropleth Map Test"
    if not isinstance(sample_choropleth_gdf, gpd.GeoDataFrame) or sample_choropleth_gdf.empty or \
       'risk_score' not in sample_choropleth_gdf.columns or 'zone_id' not in sample_choropleth_gdf.columns:
        pytest.skip("Sample GDF for choropleth map test is not correctly configured or empty.")

    fig_map = plot_layered_choropleth_map_web(
        gdf_data=sample_choropleth_gdf,
        value_col_name='risk_score', # Column from sample_choropleth_gdf
        map_title=chart_title_map,
        id_col_name='zone_id' # Assuming 'zone_id' is the ID column in GDF
    )
    assert isinstance(fig_map, go.Figure)
    assert fig_map.layout.title.text == chart_title_map
    assert fig_map.layout.mapbox.style is not None
    assert len(fig_map.data) >= 1 and fig_map.data[0].type == 'choroplethmapbox'

    fig_empty_map = plot_layered_choropleth_map_web(gpd.GeoDataFrame(), value_col_name='v', map_title="Empty Geo Map")
    assert "Empty Geo Map: Geographic data unavailable" in fig_empty_map.layout.title.text


@patch('utils.ui_visualization_helpers.MAPBOX_TOKEN_SET_FLAG', False) # Mock the global flag
def test_plot_map_web_no_token_uses_fallback_style(sample_choropleth_gdf, monkeypatch):
    """Test that if Mapbox token is not set, a non-token style is used for token-requiring styles."""
    if not isinstance(sample_choropleth_gdf, gpd.GeoDataFrame) or sample_choropleth_gdf.empty or \
       'risk_score' not in sample_choropleth_gdf.columns or 'zone_id' not in sample_choropleth_gdf.columns:
        pytest.skip("Sample GDF for no-token map test is not correctly configured.")

    # Simulate app_config.MAPBOX_STYLE_WEB requesting a token-based style,
    # but the (mocked) MAPBOX_TOKEN_SET_FLAG is False.
    # We need to ensure that the `plot_layered_choropleth_map_web` itself,
    # or the theme it inherits from, defaults to an open style.
    # The `set_sentinel_plotly_theme_web` already has this logic. This test acts as a secondary check
    # specific to the map function's behavior if it were to directly use a private style.

    # Force a token-requiring style directly in the call for testing this specific override logic.
    fig = plot_layered_choropleth_map_web(
        gdf_data=sample_choropleth_gdf, value_col_name='risk_score',
        map_title="No Token Map Fallback Test", id_col_name='zone_id',
        mapbox_style_override="mapbox://styles/mapbox/satellite-streets-v11" # Clearly token-requiring
    )
    assert fig.layout.mapbox.style == "open-street-map" # Should default to this

    # Test when default theme itself uses a token style (from app_config) but token is false
    # This implicitly tests the set_sentinel_plotly_theme_web fallback through the default template mechanism
    monkeypatch.setattr(app_config, 'MAPBOX_STYLE_WEB', "mapbox://styles/mapbox/streets-v11") # Force a token style via config
    # Re-apply theme with patched app_config. (Normally theme set once per module)
    # For more isolated test, one might mock pio.templates.default.layout.mapbox.style instead of full theme reload
    with patch('plotly.io.templates.default.layout.mapbox.style', "mapbox://styles/mapbox/streets-v11"):
        fig_theme_fallback = plot_layered_choropleth_map_web(
            gdf_data=sample_choropleth_gdf, value_col_name='risk_score',
            map_title="No Token Map Theme Fallback", id_col_name='zone_id',
            mapbox_style_override=None # Let it use the theme's mapbox_style
        )
        # Given MAPBOX_TOKEN_SET_FLAG is mocked to False, the effective map style
        # set by set_sentinel_plotly_theme_web (and thus inherited by the plot) should be open.
        assert fig_theme_fallback.layout.mapbox.style == "open-street-map"
