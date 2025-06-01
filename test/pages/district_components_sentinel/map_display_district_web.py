# sentinel_project_root/test/pages/district_components_sentinel/map_display_district_web.py
# Renders the interactive district map for the Sentinel DHO web dashboard.

import streamlit as st
import pandas as pd # For type hinting of GDF as pd.DataFrame, though it's gpd.GeoDataFrame
import geopandas as gpd
import logging

# Standardized import block
try:
    from config import app_config
    from utils.ui_visualization_helpers import plot_layered_choropleth_map_web, _create_empty_plot_figure
except ImportError:
    import sys
    import os
    # Assumes this file is in sentinel_project_root/test/pages/district_components_sentinel/
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_test_root_dir = os.path.abspath(os.path.join(current_script_dir, os.pardir, os.pardir))
    if project_test_root_dir not in sys.path:
        sys.path.insert(0, project_test_root_dir)
    from config import app_config
    from utils.ui_visualization_helpers import plot_layered_choropleth_map_web, _create_empty_plot_figure

logger = logging.getLogger(__name__)

def _get_map_metric_options(
    district_gdf_sample_check: Optional[gpd.GeoDataFrame] = None # Use a more descriptive name
) -> Dict[str, Dict[str, str]]:
    """
    Defines metrics available for map display, checking against GDF sample columns.
    Returns: {Display Name: {"col": actual_col_name_in_gdf, "colorscale": "PlotlyScaleName", "format_str": "{:.1f}"}}
    Column names in 'col' must align with the output of `enrich_zone_geodata_with_health_aggregates`.
    """
    # Define all potential metrics that could be mapped
    all_map_metrics_definitions = {
        "Avg. AI Risk Score (Zone)": {"col": "avg_risk_score", "colorscale": "OrRd_r", "format_str": "{:.1f}"}, # Higher is worse
        "Key Disease Prevalence (/1k pop)": {"col": "prevalence_per_1000", "colorscale": "YlOrRd_r", "format_str": "{:.1f}"}, # Higher is worse
        "Facility Coverage Score (%)": {"col": "facility_coverage_score", "colorscale": "Greens", "format_str": "{:.0f}%"}, # Higher is better
        "Population (Total by Zone)": {"col": "population", "colorscale": "Blues", "format_str": "{:,.0f}"},
        # Placeholder: CHW Density - this column 'chw_density_per_10k' needs to be calculated during GDF enrichment.
        "CHW Density (/10k pop)": {"col": "chw_density_per_10k", "colorscale": "Greens", "format_str": "{:.2f}"},
        "Avg. Clinic CO2 (Zone Avg, ppm)": {"col": "zone_avg_co2", "colorscale": "Oranges_r", "format_str": "{:.0f}"}, # Higher is worse
        "Population Density (per sqkm)": {"col": "population_density", "colorscale": "Plasma_r", "format_str": "{:.1f}"}, # Higher is denser
        "Avg. Critical Test TAT (days)": {"col": "avg_test_turnaround_critical", "colorscale": "Reds_r", "format_str": "{:.1f}"}, # Higher is worse
        "% Critical Tests TAT Met": {"col": "perc_critical_tests_tat_met", "colorscale": "Greens", "format_str": "{:.0f}%"}, # Higher is better
        "Total Patient Encounters (Zone)": {"col": "total_patient_encounters", "colorscale": "Purples", "format_str": "{:,.0f}"},
        "Avg. Patient Daily Steps (Zone)": {"col": "avg_daily_steps_zone", "colorscale": "BuGn", "format_str": "{:,.0f}"} # Higher is better
    }
    # Dynamically add metrics for active cases of each key condition from app_config
    for condition_key_map_cfg in app_config.KEY_CONDITIONS_FOR_ACTION:
        # Construct column name exactly as it would be generated in enrich_zone_geodata_with_health_aggregates
        col_name_for_map_metric = f"active_{condition_key_map_cfg.lower().replace(' ', '_').replace('-', '_').replace('(severe)','')}_cases"
        display_condition_label_map = condition_key_map_cfg.replace("(Severe)", "").strip() # Cleaner label for UI
        all_map_metrics_definitions[f"Active {display_condition_label_map} Cases (Zone)"] = {
            "col": col_name_for_map_metric, 
            "colorscale": "Reds_r", # Default for disease burden: higher is worse
            "format_str": "{:.0f}"
        }

    if not isinstance(district_gdf_sample_check, gpd.GeoDataFrame) or district_gdf_sample_check.empty:
        logger.debug("No GDF sample provided to _get_map_metric_options, returning all defined potential metrics without validation.")
        return all_map_metrics_definitions

    # Filter metrics: only include if the required column exists in the GDF sample and has some non-null data
    available_metrics_for_map = {}
    for display_name_map_metric, metric_details_map in all_map_metrics_definitions.items():
        column_name_map_metric = metric_details_map["col"]
        if column_name_map_metric in district_gdf_sample_check.columns and \
           district_gdf_sample_check[column_name_map_metric].notna().any(): # Check if column has at least one non-NaN value
            available_metrics_for_map[display_name_map_metric] = metric_details_map
        else:
            logger.debug(f"Map metric '{display_name_map_metric}' (column '{column_name_map_metric}') excluded: column missing or all NaN in GDF sample.")
            
    return available_metrics_for_map


def render_district_interactive_map_web(
    district_gdf_main_enriched: Optional[gpd.GeoDataFrame],
    default_selected_metric_col_name: str = 'avg_risk_score', # Internal GDF column name for default selection
    reporting_period_str: str = "Latest Aggregated Zonal Data" # Default value for context
) -> None:
    """
    Renders an interactive choropleth map for DHO's district-level visualization.
    Manages its own metric selection dropdown.
    """
    module_log_prefix = "DistrictMapWebRenderer" # Consistent prefix
    logger.info(f"({module_log_prefix}) Rendering district interactive map for period: {reporting_period_str}")

    if not isinstance(district_gdf_main_enriched, gpd.GeoDataFrame) or district_gdf_main_enriched.empty or \
       'geometry' not in district_gdf_main_enriched.columns: # Critical check for geometry
        st.warning("Map visualization unavailable: Enriched district geographic data (GDF) is missing, empty, or lacks a geometry column.")
        st.plotly_chart(
            _create_empty_plot_figure("District Health Map", app_config.WEB_MAP_DEFAULT_HEIGHT, "Geographic data not loaded or invalid."),
            use_container_width=True
        )
        return

    # Get available metrics for map selection based on the columns present in the provided GDF
    # Pass a small sample (e.g., .head(2)) to check for column existence and if they contain any non-null data.
    map_metric_options_available = _get_map_metric_options(district_gdf_main_enriched.head(2)) 
    
    if not map_metric_options_available:
        st.warning("No metrics available for map display based on the current geographic data's columns or content.")
        st.plotly_chart(
            _create_empty_plot_figure("District Health Map", app_config.WEB_MAP_DEFAULT_HEIGHT, "No metrics available to display on map."),
            use_container_width=True
        )
        return

    # Determine default selection for the selectbox (user-friendly display name)
    default_selection_display_name = None
    # Try to find the display name corresponding to the default_selected_metric_col_name
    for display_name_option, details_option in map_metric_options_available.items():
        if details_option["col"] == default_selected_metric_col_name:
            default_selection_display_name = display_name_option
            break
    if not default_selection_display_name: # Fallback if default col_name not found or not available
        default_selection_display_name = list(map_metric_options_available.keys())[0] # First available metric

    # Create selectbox for user to choose which metric to visualize
    selected_metric_user_choice = st.selectbox(
        "Select Metric for Map Visualization:",
        options=list(map_metric_options_available.keys()),
        index=list(map_metric_options_available.keys()).index(default_selection_display_name), # Set default index
        key="dho_map_metric_selector_key" # Unique key for Streamlit widget
    )
    
    selected_metric_configuration = map_metric_options_available.get(selected_metric_user_choice)

    if selected_metric_configuration:
        metric_column_to_plot_on_map = selected_metric_configuration["col"]
        
        # Define a standard set of columns to show on hover, ensuring they exist in the GDF
        base_hover_cols = ['name', 'population', 'num_clinics', 'zone_id'] # Basic useful info
        hover_data_cols_for_map = [col_hover for col_hover in base_hover_cols if col_hover in district_gdf_main_enriched.columns]
        # Ensure the plotted metric itself is included in hover data if not already in base_hover_cols
        if metric_column_to_plot_on_map not in hover_data_cols_for_map and metric_column_to_plot_on_map in district_gdf_main_enriched.columns:
            hover_data_cols_for_map.append(metric_column_to_plot_on_map)

        # Critical check: 'zone_id' (or the chosen id_col_name) must be in GDF for linking map features
        id_col_for_map = 'zone_id' # Default, assuming enrich_zone_geodata outputs this
        if id_col_for_map not in district_gdf_main_enriched.columns:
            st.error(f"Critical error: ID column '{id_col_for_map}' for map features is missing in geographic data. Cannot render map.")
            logger.error(f"({module_log_prefix}) ID column '{id_col_for_map}' missing in GDF. Map rendering aborted.")
            return

        map_plot_figure = plot_layered_choropleth_map_web(
            gdf_data=district_gdf_main_enriched,
            value_col_name=metric_column_to_plot_on_map,
            map_title=f"District Map: {selected_metric_user_choice}",
            id_col_name=id_col_for_map, 
            color_scale_name=selected_metric_configuration["colorscale"],
            hover_data_cols_list=hover_data_cols_for_map, # Pass the filtered list
            map_height_val=app_config.WEB_MAP_DEFAULT_HEIGHT
            # Facility points could be an optional overlay if facility_points_gdf was passed here
        )
        st.plotly_chart(map_plot_figure, use_container_width=True)
        logger.info(f"({module_log_prefix}) District map rendered for metric: '{selected_metric_user_choice}' (using GDF column: '{metric_column_to_plot_on_map}')")
    else: # Should not happen if selectbox options are from available_metrics_for_map_selection
        st.info("Please select a valid metric from the dropdown to display on the map.")
        logger.warning(f"({module_log_prefix}) No valid metric configuration found for selected display name: '{selected_metric_user_choice}' after selectbox interaction.")
