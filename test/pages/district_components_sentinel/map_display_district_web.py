# sentinel_project_root/test/pages/district_components_sentinel/map_display_district_web.py
# Renders the interactive district map for the Sentinel DHO web dashboard.

import streamlit as st
import pandas as pd # Not directly used for GDF manipulation here, but good practice
import geopandas as gpd
import logging

# Standardized import block
try:
    from config import app_config
    from utils.ui_visualization_helpers import plot_layered_choropleth_map_web, _create_empty_plot_figure
except ImportError:
    import sys
    import os
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_for_utils = os.path.abspath(os.path.join(current_script_dir, os.pardir, os.pardir))
    if project_root_for_utils not in sys.path:
        sys.path.insert(0, project_root_for_utils)
    from config import app_config
    from utils.ui_visualization_helpers import plot_layered_choropleth_map_web, _create_empty_plot_figure

logger = logging.getLogger(__name__)

def _get_map_metric_options(
    district_gdf_sample: Optional[gpd.GeoDataFrame] = None
) -> Dict[str, Dict[str, str]]:
    """
    Defines metrics available for map display, checking against GDF sample columns.
    Returns: {Display Name: {"col": actual_col_name, "colorscale": "PlotlyScale", "format_str": "{:.1f}"}}
    """
    # These column names must align with the output of enrich_zone_geodata_with_health_aggregates
    all_map_metrics = {
        "Avg. AI Risk Score (Zone)": {"col": "avg_risk_score", "colorscale": "OrRd_r", "format_str": "{:.1f}"},
        "Key Disease Prevalence (/1k pop)": {"col": "prevalence_per_1000", "colorscale": "YlOrRd_r", "format_str": "{:.1f}"},
        "Facility Coverage Score (%)": {"col": "facility_coverage_score", "colorscale": "Greens_r", "format_str": "{:.1f}%"}, # Higher is better
        "Population (Total)": {"col": "population", "colorscale": "Blues", "format_str": "{:,.0f}"},
        "CHW Density (/10k pop)": {"col": "chw_density_per_10k", "colorscale": "Greens_r", "format_str": "{:.2f}"}, # Placeholder - needs 'chw_density_per_10k' in GDF
        "Avg. Clinic CO2 (ppm)": {"col": "zone_avg_co2", "colorscale": "Oranges_r", "format_str": "{:.0f}"},
        "Population Density (per sqkm)": {"col": "population_density", "colorscale": "Plasma_r", "format_str": "{:.1f}"},
        "Avg. Critical Test TAT (days)": {"col": "avg_test_turnaround_critical", "colorscale": "Reds_r", "format_str": "{:.1f}"},
        "% Critical Tests TAT Met": {"col": "perc_critical_tests_tat_met", "colorscale": "Greens_r", "format_str": "{:.1f}%"},
        "Total Patient Encounters (Zone)": {"col": "total_patient_encounters", "colorscale": "Purples", "format_str": "{:,.0f}"},
    }
    # Dynamically add active cases for key conditions
    for cond_key in app_config.KEY_CONDITIONS_FOR_ACTION:
        col_name_map = f"active_{cond_key.lower().replace(' ', '_').replace('-', '_').replace('(severe)','')}_cases"
        display_cond_name_map = cond_key.replace("(Severe)", "").strip()
        all_map_metrics[f"Active {display_cond_name_map} Cases (Zone)"] = {"col": col_name_map, "colorscale": "Inferno_r", "format_str": "{:.0f}"}

    if not isinstance(district_gdf_sample, gpd.GeoDataFrame) or district_gdf_sample.empty:
        logger.debug("No GDF sample for _get_map_metric_options, returning all defined metrics.")
        return all_map_metrics

    available_metrics = {}
    for display_name, details in all_map_metrics.items():
        col = details["col"]
        if col in district_gdf_sample.columns and district_gdf_sample[col].notna().any():
            available_metrics[display_name] = details
        else:
            logger.debug(f"Map metric '{display_name}' (col '{col}') excluded: column missing or all NaN in GDF sample.")
    return available_metrics


def render_district_interactive_map_web(
    district_gdf_main_enriched: Optional[gpd.GeoDataFrame],
    default_selected_metric_col_name: str = 'avg_risk_score', # Internal GDF column name for default
    reporting_period_str: str = "Latest Aggregated Zonal Data" # Default value
) -> None:
    """
    Renders an interactive choropleth map for DHO's district-level visualization.
    """
    module_log_prefix = "DistrictMapWebRenderer"
    logger.info(f"({module_log_prefix}) Rendering district map for period: {reporting_period_str}")

    if not isinstance(district_gdf_main_enriched, gpd.GeoDataFrame) or district_gdf_main_enriched.empty:
        st.warning("Map visualization unavailable: Enriched district geographic data (GDF) is missing or empty.")
        st.plotly_chart(
            _create_empty_plot_figure("District Health Map", app_config.WEB_MAP_DEFAULT_HEIGHT, "Geographic data not loaded."),
            use_container_width=True
        )
        return

    # Get available metrics for map selection based on the provided GDF
    available_metrics_for_map_selection = _get_map_metric_options(district_gdf_main_enriched.head())
    
    if not available_metrics_for_map_selection:
        st.warning("No metrics available for map display based on the current geographic data.")
        st.plotly_chart(
            _create_empty_plot_figure("District Health Map", app_config.WEB_MAP_DEFAULT_HEIGHT, "No metrics to display."),
            use_container_width=True
        )
        return

    # Determine default selection for the selectbox (user-friendly display name)
    default_display_name_selection = None
    for disp_name, details in available_metrics_for_map_selection.items():
        if details["col"] == default_selected_metric_col_name:
            default_display_name_selection = disp_name
            break
    if not default_display_name_selection: # Fallback if default col_name not found or not available
        default_display_name_selection = list(available_metrics_for_map_selection.keys())[0]

    # Create selectbox for metric choice
    selected_metric_display_name = st.selectbox(
        "Select Metric for Map Visualization:",
        options=list(available_metrics_for_map_selection.keys()),
        index=list(available_metrics_for_map_selection.keys()).index(default_display_name_selection),
        key="dho_map_metric_selector" # Unique key for widget
    )
    
    selected_metric_config = available_metrics_for_map_selection.get(selected_metric_display_name)

    if selected_metric_config:
        metric_col_to_plot = selected_metric_config["col"]
        
        # Define columns to show on hover (must exist in GDF)
        hover_cols_base = ['name', 'population', 'num_clinics'] # Basic info
        hover_cols_to_include = [col for col in hover_cols_base if col in district_gdf_main_enriched.columns]
        if metric_col_to_plot not in hover_cols_to_include : # Ensure the plotted metric is in hover data
            hover_cols_to_include.append(metric_col_to_plot)

        # Ensure 'zone_id' is present for linking features
        if 'zone_id' not in district_gdf_main_enriched.columns:
            st.error("Critical error: 'zone_id' column missing in geographic data for map rendering.")
            return

        map_figure = plot_layered_choropleth_map_web(
            gdf_data=district_gdf_main_enriched,
            value_col_name=metric_col_to_plot,
            map_title=f"District Map: {selected_metric_display_name}",
            id_col_name='zone_id', # This links to GeoJSON feature IDs
            color_scale_name=selected_metric_config["colorscale"],
            hover_data_cols_list=hover_cols_to_include,
            map_height_val=app_config.WEB_MAP_DEFAULT_HEIGHT
            # Facility points could be added here if facility_points_gdf was available and passed
        )
        st.plotly_chart(map_figure, use_container_width=True)
        logger.info(f"({module_log_prefix}) District map rendered for metric: '{selected_metric_display_name}' (column: '{metric_col_to_plot}')")
    else:
        st.info("Please select a valid metric to display on the map.")
        logger.warning(f"({module_log_prefix}) No valid metric configuration found for selected display name: '{selected_metric_display_name}'")
