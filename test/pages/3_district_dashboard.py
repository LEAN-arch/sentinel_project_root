# sentinel_project_root/test/pages/3_district_dashboard.py
# District Health Strategic Command Center for Sentinel Health Co-Pilot.

import streamlit as st
import pandas as pd
import geopandas as gpd
# import numpy as np # Not directly used after refactor
import os
import logging
from datetime import date, timedelta, datetime # Added datetime

# --- Sentinel System Imports ---
try:
    from config import app_config
    from utils.core_data_processing import (
        load_health_records, load_iot_clinic_environment_data, load_zone_data,
        enrich_zone_geodata_with_health_aggregates, get_district_summary_kpis,
        hash_geodataframe # For caching GeoDataFrames
    )
    from utils.ai_analytics_engine import apply_ai_models # For enriching health data before aggregation
    from utils.ui_visualization_helpers import (
        render_web_kpi_card,
        plot_annotated_line_chart_web,
        plot_bar_chart_web
    )
    # District Component specific data processors/renderers
    from .district_components_sentinel.kpi_structurer_district import structure_district_kpis_data
    from .district_components_sentinel.map_display_district_web import render_district_interactive_map_web
    from .district_components_sentinel.trend_calculator_district import calculate_district_trends_data
    from .district_components_sentinel.comparison_data_preparer_district import prepare_zonal_comparison_data
    from .district_components_sentinel.intervention_data_preparer_district import identify_priority_zones_for_intervention, get_intervention_criteria_options
except ImportError as e:
    st.error(f"Critical import error in DHO Dashboard: {e}. Ensure all modules are correctly placed and PYTHONPATH is set if running outside Streamlit's standard structure.")
    st.stop()

# --- Page Configuration ---
st.set_page_config(
    page_title=f"DHO Command Center - {app_config.APP_NAME}",
    layout="wide",
    initial_sidebar_state="expanded"
)
logger = logging.getLogger(__name__) # Page-specific logger

# --- Data Aggregation and Preparation for DHO View ---
@st.cache_data(
    ttl=app_config.CACHE_TTL_SECONDS_WEB_REPORTS,
    hash_funcs={gpd.GeoDataFrame: hash_geodataframe, pd.DataFrame: pd.util.hash_pandas_object},
    show_spinner="Aggregating and enriching district-level operational data..."
)
def get_dho_command_center_datasets_cached(): # Renamed for clarity of caching intent
    """
    Simulates the comprehensive data pipeline for the DHO Command Center.
    Loads raw data, applies AI enrichment, enriches zonal GDF, calculates district KPIs,
    and prepares intervention criteria options.
    """
    func_log_prefix = "GetDHODatasetsCached" # For logging
    logger.info(f"({func_log_prefix}) Initializing full data pipeline simulation...")
    
    health_df_raw = load_health_records(source_context=f"{func_log_prefix}/LoadHealth")
    iot_df_raw = load_iot_clinic_environment_data(source_context=f"{func_log_prefix}/LoadIoT")
    # load_zone_data returns a GeoDataFrame with attributes and geometries merged
    base_zone_gdf_loaded = load_zone_data(source_context=f"{func_log_prefix}/LoadZone")

    if not isinstance(base_zone_gdf_loaded, gpd.GeoDataFrame) or base_zone_gdf_loaded.empty:
        logger.error(f"({func_log_prefix}) Base zone geographic data (GDF) failed to load or is empty. DHO dashboard will be severely limited.")
        # Return empty structures with expected types to prevent downstream crashes
        return gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs=app_config.DEFAULT_CRS_STANDARD), \
               pd.DataFrame(), pd.DataFrame(), {}, {}

    # Enrich raw health data with AI model outputs (simulating central processing)
    full_health_df_enriched = pd.DataFrame()
    # Define expected columns if health_df_raw is empty or AI model doesn't run
    ai_added_cols_for_dho = ['ai_risk_score', 'ai_followup_priority_score']
    base_health_cols_for_dho = health_df_raw.columns.tolist() if isinstance(health_df_raw, pd.DataFrame) and not health_df_raw.empty else []

    if isinstance(health_df_raw, pd.DataFrame) and not health_df_raw.empty:
        enriched_ai_output_dho = apply_ai_models(health_df_raw.copy(), source_context=f"{func_log_prefix}/AIEnrichHealth")
        full_health_df_enriched = enriched_ai_output_dho[0] # [0] is the health_df
    else:
        logger.warning(f"({func_log_prefix}) Raw health data for DHO view is empty or invalid. AI enrichment stage skipped.")
        full_health_df_enriched = pd.DataFrame(columns=list(set(base_health_cols_for_dho + ai_added_cols_for_dho)))

    # Enrich the Zonal GeoDataFrame (Primary dataset for many DHO views)
    district_gdf_enriched_final = enrich_zone_geodata_with_health_aggregates(
        zone_gdf=base_zone_gdf_loaded, # This GDF already has base attributes
        health_df=full_health_df_enriched, # Pass the AI-enriched health data
        iot_df=iot_df_raw, 
        source_context=f"{func_log_prefix}/EnrichZoneGDF"
    )
    if not isinstance(district_gdf_enriched_final, gpd.GeoDataFrame) or district_gdf_enriched_final.empty:
        logger.warning(f"({func_log_prefix}) GDF enrichment resulted in empty/None. Defaulting to base zone GDF if available for map.")
        district_gdf_enriched_final = base_zone_gdf_loaded if isinstance(base_zone_gdf_loaded, gpd.GeoDataFrame) else \
                                      gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs=app_config.DEFAULT_CRS_STANDARD)

    # Calculate overall district summary KPIs from the enriched GDF
    dict_district_summary_kpis = get_district_summary_kpis(district_gdf_enriched_final, source_context=f"{func_log_prefix}/CalcDistrictKPIs")

    # Prepare filter/criteria options for Intervention Planning tab
    # Pass a sample of the GDF (e.g., .head()) to check for column availability.
    intervention_criteria_options = get_intervention_criteria_options(
        district_gdf_check_sample=district_gdf_enriched_final.head(2) if not district_gdf_enriched_final.empty else None
    )
    
    gdf_shape_log = district_gdf_enriched_final.shape if isinstance(district_gdf_enriched_final, gpd.GeoDataFrame) else 'N/A'
    logger.info(f"({func_log_prefix}) DHO data preparation complete. Enriched GDF shape: {gdf_shape_log}")
    return district_gdf_enriched_final, full_health_df_enriched, iot_df_raw, dict_district_summary_kpis, intervention_criteria_options

# --- Load All Data for DHO Console (Using Cache and Session State) ---
# This pattern ensures data is loaded once and persists across interactions within this page.
if 'dho_data_loaded_flag' not in st.session_state: # Use a more descriptive session state key
    st.session_state.district_gdf_dho_view, \
    st.session_state.historical_health_df_for_dho, \
    st.session_state.historical_iot_df_for_dho, \
    st.session_state.district_summary_kpis_for_dho, \
    st.session_state.intervention_criteria_options_dho = get_dho_command_center_datasets_cached()
    st.session_state.dho_data_loaded_flag = True

# Use data from session state for consistency
gdf_district_display = st.session_state.district_gdf_dho_view
df_health_historical_dho = st.session_state.historical_health_df_for_dho
df_iot_historical_dho = st.session_state.historical_iot_df_for_dho
dict_district_kpis_summary = st.session_state.district_summary_kpis_for_dho
dict_intervention_criteria = st.session_state.intervention_criteria_options_dho

# --- Page Title and Sidebar ---
st.title(f"🌍 {app_config.APP_NAME} - District Health Strategic Command Center")
# Use a fixed timestamp for "Data as of" if actual data refresh time isn't available from data source.
# In a real system, this would be the timestamp of the last ETL/data aggregation.
data_as_of_ts_display = pd.Timestamp('now') # Placeholder for actual data refresh time
st.markdown(f"**Aggregated Zonal Intelligence, Resource Allocation, and Public Health Program Monitoring.** (Data as of: {data_as_of_ts_display.strftime('%d %b %Y, %H:%M %Z')})")
st.divider()

if os.path.exists(app_config.APP_LOGO_SMALL):
    st.sidebar.image(app_config.APP_LOGO_SMALL, width=150)
st.sidebar.header("🗓️ Analysis Filters")

# Trend Date Range Selector (Primarily for "District-Wide Trends" tab)
min_date_dho_trends = data_as_of_ts_display.date() - timedelta(days=365 * 2) # Max 2 years history for trends
max_date_dho_trends = data_as_of_ts_display.date()

default_end_date_dho_trends = max_date_dho_trends
# Default to 90 days for DHO trend view (3x standard trend period)
default_start_date_dho_trends = default_end_date_dho_trends - timedelta(days=app_config.WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND * 3 - 1) 
if default_start_date_dho_trends < min_date_dho_trends: default_start_date_dho_trends = min_date_dho_trends

selected_start_date_dho_trend, selected_end_date_dho_trend = st.sidebar.date_input(
    "Select Date Range for Trend Analysis:",
    value=[default_start_date_dho_trends, default_end_date_dho_trends],
    min_value=min_date_dho_trends, max_value=max_date_dho_trends,
    key="dho_page_trend_date_range_picker" # Unique key
)
if selected_start_date_dho_trend > selected_end_date_dho_trend:
    st.sidebar.error("DHO Trends: Start date must be on or before the end date.")
    selected_start_date_dho_trend = selected_end_date_dho_trend # Auto-correct

# --- Main Section 1: District-Wide KPIs ---
st.header("📊 District Performance Dashboard")
# Check if summary dict has meaningful data, not just default values
if dict_district_kpis_summary and isinstance(dict_district_kpis_summary, dict) and \
   any(v is not None and not (isinstance(v, float) and np.isnan(v)) and v != 0 for k,v in dict_district_kpis_summary.items() if k != 'total_zones_in_gdf'): # Exclude simple zone count from check
    
    list_structured_dho_kpis = structure_district_kpis_data(
        dict_district_kpis_summary, 
        gdf_district_display, # Pass GDF for context like total zone count
        reporting_period_str=f"Snapshot as of {data_as_of_ts_display.strftime('%d %b %Y')}"
    )
    if list_structured_dho_kpis:
        num_dho_kpis_total = len(list_structured_dho_kpis)
        cols_per_kpi_row = 4 # Max KPIs per row for better layout
        for i_kpi_row in range(0, num_dho_kpis_total, cols_per_kpi_row):
            kpi_cols_in_row = st.columns(cols_per_kpi_row)
            for kpi_idx_in_this_row, kpi_data_item_val in enumerate(list_structured_dho_kpis[i_kpi_row : i_kpi_row + cols_per_kpi_row]):
                with kpi_cols_in_row[kpi_idx_in_this_row]: render_web_kpi_card(**kpi_data_item_val)
    else: 
        st.info("District KPIs could not be structured from the available summary data. Check KPI structurer logic or input summary.")
else:
    st.warning("District-wide summary KPIs are currently unavailable or contain no significant data. Please check data aggregation and enrichment pipeline.")
st.divider()

# --- Tabbed Interface for Detailed District Analysis ---
st.header("🔍 In-Depth District Analysis Modules")
dho_tab_titles_list_main = ["🗺️ Geospatial Overview", "📈 District Trends", "🆚 Zonal Comparison", "🎯 Intervention Planning"]
tab_map_dho_view, tab_trends_dho_view, tab_compare_dho_view, tab_intervene_dho_view = st.tabs(dho_tab_titles_list_main)

with tab_map_dho_view:
    st.subheader("Interactive District Health & Environmental Map")
    if isinstance(gdf_district_display, gpd.GeoDataFrame) and not gdf_district_display.empty:
        render_district_interactive_map_web( # This component manages its own metric selectbox
            district_gdf_main_enriched=gdf_district_display,
            default_selected_metric_col_name='avg_risk_score', # DHO often wants to see risk first
            reporting_period_str=f"Zonal Data as of {data_as_of_ts_display.strftime('%d %b %Y')}"
        )
    else:
        st.warning("Map visualization unavailable: Enriched district geographic data is missing or empty.")

with tab_trends_dho_view:
    trend_period_display_str = f"{selected_start_date_dho_trend.strftime('%d %b %Y')} - {selected_end_date_dho_trend.strftime('%d %b %Y')}"
    st.subheader(f"District-Wide Health & Environmental Trends ({trend_period_display_str})")
    
    # Filter historical_health_df and historical_iot_df for the selected trend period
    df_health_for_trends_tab = pd.DataFrame()
    if isinstance(df_health_historical_dho, pd.DataFrame) and not df_health_historical_dho.empty and 'encounter_date' in df_health_historical_dho.columns:
        df_health_for_trends_tab = df_health_historical_dho[
            (pd.to_datetime(df_health_historical_dho['encounter_date']).dt.date >= selected_start_date_dho_trend) &
            (pd.to_datetime(df_health_historical_dho['encounter_date']).dt.date <= selected_end_date_dho_trend)
        ].copy()

    df_iot_for_trends_tab = pd.DataFrame()
    if isinstance(df_iot_historical_dho, pd.DataFrame) and not df_iot_historical_dho.empty and 'timestamp' in df_iot_historical_dho.columns:
        df_iot_for_trends_tab = df_iot_historical_dho[
            (pd.to_datetime(df_iot_historical_dho['timestamp']).dt.date >= selected_start_date_dho_trend) &
            (pd.to_datetime(df_iot_historical_dho['timestamp']).dt.date <= selected_end_date_dho_trend)
        ].copy()

    if df_health_for_trends_tab.empty and df_iot_for_trends_tab.empty:
        st.info(f"No health or IoT data available for the selected trend period: {trend_period_display_str}")
    else:
        dict_district_trends_data = calculate_district_trends_data(
            df_health_for_trends_tab, df_iot_for_trends_tab,
            selected_start_date_dho_trend, selected_end_date_dho_trend, # For context to calculator
            trend_period_display_str 
        )
        
        # Display Disease Incidence Trends
        map_disease_incidence_trends = dict_district_trends_data.get("disease_incidence_trends", {})
        if map_disease_incidence_trends:
            st.markdown("###### Key Disease Incidence (Weekly New Cases - Unique Patients):")
            max_charts_per_row_disease = 2 
            disease_trend_chart_keys = list(map_disease_incidence_trends.keys())
            for i_disease_chart_row in range(0, len(disease_trend_chart_keys), max_charts_per_row_disease):
                cols_for_disease_trends = st.columns(max_charts_per_row_disease)
                for j_disease_chart_idx, key_disease_trend in enumerate(disease_trend_chart_keys[i_disease_chart_row : i_disease_chart_row + max_charts_per_row_disease]):
                    series_disease_data = map_disease_incidence_trends[key_disease_trend]
                    if isinstance(series_disease_data, pd.Series) and not series_disease_data.empty:
                        with cols_for_disease_trends[j_disease_chart_idx]:
                            st.plotly_chart(plot_annotated_line_chart_web(series_disease_data, f"{key_disease_trend} New Cases (Weekly)", y_is_count=True, y_axis_label="# Unique Patients"), use_container_width=True)
        
        # Display Other Key Trends
        other_district_trends_to_plot = {
            "Avg. Patient AI Risk Score Trend": (dict_district_trends_data.get("avg_patient_ai_risk_trend"), "AI Risk Score"),
            "Avg. Patient Daily Steps Trend": (dict_district_trends_data.get("avg_patient_daily_steps_trend"), "Steps/Day"),
            "Avg. Clinic CO2 Levels Trend": (dict_district_trends_data.get("avg_clinic_co2_trend"), "CO2 (ppm)")
        }
        for trend_chart_title, (trend_series_data, y_axis_lbl) in other_district_trends_to_plot.items():
            if isinstance(trend_series_data, pd.Series) and not trend_series_data.empty:
                # For steps, treat as count-like for y-axis formatting if appropriate
                y_is_count_trend = "Steps" in y_axis_lbl or "#" in y_axis_lbl 
                st.plotly_chart(plot_annotated_line_chart_web(trend_series_data, trend_chart_title, y_axis_label=y_axis_lbl, y_is_count=y_is_count_trend), use_container_width=True)
        
        if dict_district_trends_data.get("data_availability_notes"):
            for note_trend in dict_district_trends_data["data_availability_notes"]: st.caption(f"Trend Note: {note_trend}")

with tab_compare_dho_view:
    st.subheader("Comparative Zonal Analysis")
    if isinstance(gdf_district_display, gpd.GeoDataFrame) and not gdf_district_display.empty:
        dict_zonal_comparison_data = prepare_zonal_comparison_data(gdf_district_display, f"Data as of {data_as_of_ts_display.strftime('%d %b %Y')}")
        
        df_comparison_table_display = dict_zonal_comparison_data.get("zonal_comparison_table_df")
        if isinstance(df_comparison_table_display, pd.DataFrame) and not df_comparison_table_display.empty:
            st.markdown("###### **Aggregated Zonal Metrics Comparison Table:**")
            st.dataframe(df_comparison_table_display, height=min(600, len(df_comparison_table_display)*38 + 60) , use_container_width=True) # Index is zone name/ID
            
            # Example: Default bar chart for a key comparison metric like Avg. AI Risk Score by Zone
            map_comp_metrics_config = dict_zonal_comparison_data.get("comparison_metrics_config", {})
            default_bar_metric_display_name = "Avg. AI Risk Score (Zone)" # User-friendly name from config
            if default_bar_metric_display_name in map_comp_metrics_config:
                metric_details_for_bar = map_comp_metrics_config[default_bar_metric_display_name]
                actual_risk_col_name_bar = metric_details_for_bar["col"]
                # The comparison_df has zone name/ID as index; reset for plotting if x_col needs to be a column
                df_for_bar_chart_comp = df_comparison_table_display.reset_index() 
                # Determine which column is the zone identifier (was set as index)
                zone_identifier_col_for_bar = df_for_bar_chart_comp.columns[0] 
                
                if actual_risk_col_name_bar in df_for_bar_chart_comp.columns:
                    st.plotly_chart(plot_bar_chart_web(df_for_bar_chart_comp, x_col=zone_identifier_col_for_bar, y_col=actual_risk_col_name_bar, 
                                                       title=f"{default_bar_metric_display_name}", sort_by=actual_risk_col_name_bar, sort_ascending=False, 
                                                       x_axis_label="Zone", y_axis_label="Avg. AI Risk Score"), use_container_width=True)
        else:
            st.info("No data available for the zonal comparison table with current GDF content.")

        if dict_zonal_comparison_data.get("data_availability_notes"):
            for note_compare in dict_zonal_comparison_data["data_availability_notes"]: st.caption(f"Comparison Note: {note_compare}")
    else:
        st.warning("Zonal comparison cannot be performed: Enriched district geographic data is missing or empty.")

with tab_intervene_dho_view:
    st.subheader("Targeted Intervention Planning Assistant")
    if isinstance(gdf_district_display, gpd.GeoDataFrame) and not gdf_district_display.empty and dict_intervention_criteria:
        list_intervention_criteria_keys = list(dict_intervention_criteria.keys())
        # Default to selecting the first 1 or 2 criteria if available
        default_criteria_selection_list = list_intervention_criteria_keys[0:min(2, len(list_intervention_criteria_keys))] 
        
        selected_criteria_for_intervention = st.multiselect(
            "Select Criteria to Identify Priority Zones (zones meeting ANY selected criterion will be shown):",
            options=list_intervention_criteria_keys,
            default=default_criteria_selection_list,
            key="dho_intervention_criteria_multiselect" # Unique key
        )
        
        dict_intervention_results = identify_priority_zones_for_intervention(
            district_gdf_main_enriched=gdf_district_display,
            selected_criteria_display_names=selected_criteria_for_intervention,
            available_criteria_options=dict_intervention_criteria, # Pass the full options dict
            reporting_period_str=f"Data as of {data_as_of_ts_display.strftime('%d %b %Y')}"
        )
        
        df_priority_zones_for_display = dict_intervention_results.get("priority_zones_for_intervention_df")
        list_applied_criteria_names = dict_intervention_results.get("applied_criteria_names", [])

        if isinstance(df_priority_zones_for_display, pd.DataFrame) and not df_priority_zones_for_display.empty:
            st.markdown(f"###### **{len(df_priority_zones_for_display)} Zone(s) Flagged for Intervention Based on: {', '.join(list_applied_criteria_names) if list_applied_criteria_names else 'Selected Criteria'}**")
            # Make table scrollable, hide default pandas index if 'name' or 'zone_id' is a column
            st.dataframe(df_priority_zones_for_display, use_container_width=True, height=min(500, len(df_priority_zones_for_display)*40 + 60), hide_index=True)
        elif selected_criteria_for_intervention: # Criteria were selected, but no zones met them
             st.success(f"✅ No zones currently meet the selected criteria: {', '.join(list_applied_criteria_names) if list_applied_criteria_names else 'None (check criteria applicability or data)'}.")
        else: # No criteria selected by user yet
             st.info("Please select one or more criteria above to identify potential priority zones for intervention.")

        if dict_intervention_results.get("data_availability_notes"):
            for note_intervene in dict_intervention_results["data_availability_notes"]: st.caption(f"Intervention Note: {note_intervene}")
    else:
        st.warning("Intervention planning tools unavailable: District geographic data or intervention criteria definitions are missing/invalid.")

logger.info(f"DHO Strategic Command Center page loaded/refreshed. Data as of: {data_as_of_ts_display.isoformat()}")
