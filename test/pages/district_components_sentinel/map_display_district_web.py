# sentinel_project_root/test/pages/district_components_sentinel/map_display_district_web.py
# Part of "Sentinel Health Co-Pilot" - LMIC Edge-First System
# This module renders the interactive district map for the DHO web dashboard.
# It consumes the enriched GeoDataFrame and uses the refactored plotting utilities.

import streamlit as st
import pandas as pd
import geopandas as gpd # Still used here for type hints and if GDF passed directly
import logging

try:
    from config import app_config
    # Use the _web suffixed plotting function
    from utils.ui_visualization_helpers import plot_layered_choropleth_map_web, _create_empty_plot_figure
except ImportError:
    import sys, os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_test_root = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
    if project_test_root not in sys.path: sys.path.insert(0, project_test_root)
    from config import app_config
    from utils.ui_visualization_helpers import plot_layered_choropleth_map_web, _create_empty_plot_figure


logger = logging.getLogger(__name__)

def render_district_interactive_map_web( # Renamed for clarity as it's for web display
    district_gdf_main_enriched: Optional[gpd.GeoDataFrame],
    default_selected_metric_col_name: str = 'avg_risk_score', # Internal column name for default
    reporting_period_str: Optional[str] = "Latest Aggregated Zonal Data"
) -> None:
    """
    Renders an interactive choropleth map for DHO's district-level visualization.
    Called from the DHO dashboard page (Tier 2/3).
    (Full implementation as previously provided in File 30 response for render_district_interactive_map)
    """
    module_source_context = "DistrictMapWebRenderer"
    logger.info(f"({module_source_context}) Rendering district map for period: {reporting_period_str}")

    if not isinstance(district_gdf_main_enriched, gpd.GeoDataFrame) or district_gdf_main_enriched.empty:
        st.warning("Map visualization unavailable: Enriched district geographic data (GDF) is missing or empty.")
        st.plotly_chart(_create_empty_plot_figure("District Health Map", app_config.WEB_MAP_DEFAULT_HEIGHT, "Geographic data not loaded."), use_container_width=True)
        return

    # (The rest of this function's logic for defining map_metric_options, handling selectbox,
    # and calling plot_layered_choropleth_map_web is IDENTICAL to what was provided in detail
    # for the `render_district_interactive_map` function within `map_display_district.py` in File 30 / your Response #20.
    # This includes dynamic filtering of available_map_metrics_for_selection based on GDF columns.)

    # Example of key parts (Full version was in previous detailed response):
    map_metric_options_def = { # Aligned with Sentinel outputs
        "Avg. AI Risk Score (Zone)": {"col": "avg_risk_score", "colorscale": "OrRd_r", "format_str": "{:.1f}"},
        "Prevalence per 1k (Key Inf.)": {"col": "prevalence_per_1000", "colorscale": "YlOrRd_r", "format_str": "{:.1f}"},
        # ... Add all other metrics from File 30's map_metric_options_config definition ...
        "Facility Coverage Score (Zone)": {"col": "facility_coverage_score", "colorscale": "Greens", "format_str": "{:.1f}%"},
        "Zone Population": {"col": "population", "colorscale": "Viridis", "format_str": "{:,.0f}"},
    }
    # Add active case counts for key conditions if columns exist
    for cond_key_map in app_config.KEY_CONDITIONS_FOR_ACTION:
        col_map_name = f"active_{cond_key_map.lower().replace(' ', '_').replace('-', '_').replace('(severe)','')}_cases"
        if col_map_name in district_gdf_main_enriched.columns:
            map_metric_options_def[f"Active {cond_key_map} Cases (Zone)"] = {"col": col_map_name, "colorscale": "Purples_r", "format_str": "{:.0f}"}


    available_metrics_map = { dn: dt for dn, dt in map_metric_options_def.items() if dt["col"] in district_gdf_main_enriched.columns and district_gdf_main_enriched[dt["col"]].notna().any() }
    if not available_metrics_map: st.warning("No metrics available for map display."); return

    # Find default display name from default_selected_metric_col_name
    default_display_sel = default_selected_metric_col_name
    for disp_n, details_n in available_metrics_map.items():
        if details_n["col"] == default_selected_metric_col_name: default_display_sel = disp_n; break
    if default_display_sel not in available_metrics_map.keys() and available_metrics_map: default_display_sel = list(available_metrics_map.keys())[0]
    
    sel_metric_disp_name_map = st.selectbox("Select Metric for Map Visualization:", options=list(available_metrics_map.keys()), index=list(available_metrics_map.keys()).index(default_display_sel) if default_display_sel in available_metrics_map else 0, key="dho_page_map_metric_select_v2")
    
    selected_metric_details_map = available_metrics_map.get(sel_metric_disp_name_map)
    if selected_metric_details_map:
        hover_cols_map_list = ['name', selected_metric_details_map["col"], 'population', 'num_clinics'] # Example
        map_fig = plot_layered_choropleth_map_web(
            gdf_data=district_gdf_main_enriched, value_col_name=selected_metric_details_map["col"],
            map_title=f"District Map: {sel_metric_disp_name_map}", id_col_name='zone_id', # Ensure 'zone_id' in GDF
            color_scale=selected_metric_details_map["colorscale"], hover_data_cols=[c for c in hover_cols_map_list if c in district_gdf_main_enriched.columns],
            map_height=app_config.WEB_MAP_DEFAULT_HEIGHT
        )
        st.plotly_chart(map_fig, use_container_width=True)
    else: st.info("Please select a metric to display on the map.")

    logger.info(f"({module_source_context}) District map rendered for metric: {sel_metric_disp_name_map if selected_metric_details_map else 'None selected'}")
