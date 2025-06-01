# sentinel_project_root/test/pages/3_district_dashboard.py
# Redesigned as "District Health Strategic Command Center" for "Sentinel Health Co-Pilot"
# This page simulates a web interface for DHOs and public health teams,
# typically at a Facility Node (Tier 2) or Regional/Cloud Node (Tier 3).

import streamlit as st
import pandas as pd
import geopandas as gpd # For type hinting and potential GDF operations on page
import numpy as np
import os
import logging
from datetime import date, timedelta

# --- Sentinel System Imports ---
from config import app_config # Uses the fully refactored app_config

# Core data loading & processing utilities
from utils.core_data_processing import (
    load_health_records, load_iot_clinic_environment_data, load_zone_data,
    enrich_zone_geodata_with_health_aggregates, get_district_summary_kpis,
    hash_geodataframe # For Streamlit caching of GeoDataFrames
)
# AI engine (simulation for enriching health data before aggregation)
from utils.ai_analytics_engine import apply_ai_models

# Refactored District Component data prep/rendering functions
# (Assumed to be in a _sentinel suffixed sub-package or renamed)
from pages.district_components_sentinel.kpi_structurer_district import structure_district_kpis_data
# map_display_district_web now directly renders the map using new plotters
from pages.district_components_sentinel.map_display_district_web import render_district_interactive_map_web
from pages.district_components_sentinel.trend_calculator_district import calculate_district_trends_data
from pages.district_components_sentinel.comparison_data_preparer_district import prepare_zonal_comparison_data
# get_intervention_criteria_options can be reused if applicable structure
from pages.district_components_sentinel.intervention_data_preparer_district import identify_priority_zones_for_intervention, get_intervention_criteria_options

# Refactored UI helpers for web display
from utils.ui_visualization_helpers import (
    render_web_kpi_card,
    plot_annotated_line_chart_web,
    plot_bar_chart_web
)

# --- Page Configuration ---
st.set_page_config(
    page_title=f"DHO Command Center - {app_config.APP_NAME}",
    layout="wide",
    initial_sidebar_state="expanded"
)
logger = logging.getLogger(__name__)
# CSS loaded by app_home.py

# --- Data Aggregation and Preparation for DHO View (Simulates Tier 2/3 Node) ---
@st.cache_data(
    ttl=app_config.CACHE_TTL_SECONDS_WEB_REPORTS,
    hash_funcs={gpd.GeoDataFrame: hash_geodataframe, pd.DataFrame: pd.util.hash_pandas_object},
    show_spinner="Aggregating and enriching district-level operational data..."
)
def get_dho_command_center_datasets():
    """
    Simulates the comprehensive data pipeline that would feed the DHO Command Center.
    - Loads raw health, IoT, and zone boundary/attribute data.
    - Applies AI enrichment to health data.
    - Enriches zonal GDF with aggregated health/IoT metrics.
    - Calculates overall district summary KPIs.
    - Prepares filter/criteria options.
    Returns tuple: (district_gdf_enriched, full_historical_health_df, full_historical_iot_df, district_summary_kpis, available_filter_criteria_dict)
    """
    logger.info("DHO Command Center: Initializing full data pipeline simulation...")
    
    # 1. Load raw base data (simulating access to a data lake or operational stores)
    health_df_raw = load_health_records(file_path=app_config.HEALTH_RECORDS_CSV, source_context="DHOData/LoadHealth")
    iot_df_raw = load_iot_clinic_environment_data(file_path=app_config.IOT_CLINIC_ENVIRONMENT_CSV, source_context="DHOData/LoadIoT")
    base_zone_gdf_from_load = load_zone_data(source_context="DHOData/LoadZone") # Already merges attributes + geoms

    # 2. Enrich raw health data with AI model outputs (simulating central processing)
    full_health_enriched = pd.DataFrame()
    if not health_df_raw.empty:
        full_health_enriched = apply_ai_models(health_df_raw, source_context="DHOData/AIEnrichHealth")[0]
    else:
        logger.warning("DHO Command Center: Raw health data empty, AI enrichment stage skipped.")
        # Fallback with schema expected by enrichment if main one is empty.
        # This schema should ideally match exactly the output of apply_ai_models.
        # For now, use raw cols + AI cols. A fixture or defined schema is better.
        base_cols_ai = health_df_raw.columns.tolist() if health_df_raw is not None else []
        ai_cols = ['ai_risk_score', 'ai_followup_priority_score']
        full_health_enriched = pd.DataFrame(columns=base_cols_ai + [c for c in ai_cols if c not in base_cols_ai])


    if base_zone_gdf_from_load is None or base_zone_gdf_from_load.empty:
        logger.error("DHO Command Center: Base zone geographic data (GDF) failed to load. Most views will be impacted.")
        # Return empty structures to prevent downstream crashes on this page
        return gpd.GeoDataFrame(), pd.DataFrame(), pd.DataFrame(), {}, {}

    # 3. Enrich the Zonal GeoDataFrame (This is the primary dataset for many DHO views)
    district_gdf_enriched_main = enrich_zone_geodata_with_health_aggregates(
        zone_gdf=base_zone_gdf_from_load,
        health_df=full_health_enriched,
        iot_df=iot_df_raw, # Pass raw IoT; enrichment function filters/aggregates by zone
        source_context="DHOData/EnrichZoneGDF"
    )
    if district_gdf_enriched_main is None or district_gdf_enriched_main.empty: # Should return base GDF if enrichment adds nothing but valid
        logger.warning("DHO Command Center: Full GDF enrichment resulted in empty/None. Defaulting to base zone GDF for map if possible.")
        district_gdf_enriched_main = base_zone_gdf_from_load if base_zone_gdf_from_load is not None else gpd.GeoDataFrame()

    # 4. Calculate overall district summary KPIs from the enriched GDF
    district_kpis_summary_main = get_district_summary_kpis(district_gdf_enriched_main, source_context="DHOData/CalcDistrictKPIs")

    # 5. Prepare filter/criteria options for tabs (e.g., for Intervention Planning)
    # Using get_intervention_criteria_options as it returns a useful structure: {display_name: {lambda_func, required_cols}}
    # This check is done on the head of the GDF to see which criteria are applicable.
    filter_criteria_options = get_intervention_criteria_options(
        district_gdf_check=district_gdf_enriched_main.head() if not district_gdf_enriched_main.empty else None
    )
    
    logger.info(f"DHO Command Center: Data preparation complete. Enriched GDF shape: {district_gdf_enriched_main.shape if district_gdf_enriched_main is not None else 'N/A'}")
    return district_gdf_enriched_main, full_health_enriched, iot_df_raw, district_kpis_summary_main, filter_criteria_options


# --- Load All Data for DHO Console ---
# This simulates the comprehensive data environment available to a DHO.
# Stored in session_state for persistence across tab interactions IF page doesn't fully rerun on tab switch.
# However, Streamlit usually reruns the whole script on widget interaction.
# For heavy data, caching is key.
if 'dho_data_loaded' not in st.session_state:
    st.session_state.district_gdf, \
    st.session_state.hist_health_df_dho, \
    st.session_state.hist_iot_df_dho, \
    st.session_state.district_kpis_summary_dho, \
    st.session_state.criteria_options_dho = get_dho_command_center_datasets()
    st.session_state.dho_data_loaded = True

# Use data from session state
district_gdf_dho = st.session_state.district_gdf
historical_health_df = st.session_state.hist_health_df_dho
historical_iot_df = st.session_state.hist_iot_df_dho
district_summary_kpis = st.session_state.district_kpis_summary_dho
all_available_criteria_options = st.session_state.criteria_options_dho


# --- Page Title and Sidebar ---
st.title(f"🌍 {app_config.APP_NAME} - District Health Strategic Command Center")
data_as_of_timestamp = pd.Timestamp('now') # In real scenario, this would be last data refresh time
st.markdown(f"**Aggregated Zonal Intelligence, Resource Allocation, and Public Health Program Monitoring. Data as of: {data_as_of_timestamp.strftime('%d %b %Y, %H:%M')}**")
st.markdown("---")

if os.path.exists(app_config.APP_LOGO_SMALL):
    st.sidebar.image(app_config.APP_LOGO_SMALL, width=180)
st.sidebar.header("🗓️ Analysis Filters")

# Trend Date Range Selector (Applies primarily to "District-Wide Trends" tab)
min_hist_date_for_dho_trends = date.today() - timedelta(days=365 * 2) # Allow up to 2 years back
max_hist_date_for_dho_trends = data_as_of_timestamp.date()

default_trend_end_dho_page = max_hist_date_for_dho_trends
default_trend_start_dho_page = default_trend_end_dho_page - timedelta(days=app_config.WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND * 3 -1) # Default 90 days
if default_trend_start_dho_page < min_hist_date_for_dho_trends: default_trend_start_dho_page = min_hist_date_for_dho_trends

selected_start_dho_trends_page, selected_end_dho_trends_page = st.sidebar.date_input(
    "Select Date Range for Trend Analysis:",
    value=[default_trend_start_dho_page, default_end_dho_trends_page],
    min_value=min_hist_date_for_dho_trends, max_value=max_hist_date_for_dho_trends,
    key="dho_page_trend_date_selector_v2"
)
if selected_start_dho_trends_page > selected_end_dho_trends_page:
    st.sidebar.error("DHO Trends: Start date must be before end date.")
    selected_start_dho_trends_page = selected_end_dho_trends_page


# --- Main Section 1: District-Wide KPIs ---
st.header("📊 District Performance Dashboard")
if district_summary_kpis and isinstance(district_summary_kpis, dict) and any(district_summary_kpis.values()):
    dho_kpis_structured_list = structure_district_kpis_data(
        district_summary_kpis, district_gdf_dho, # Pass GDF for context like total zone count
        reporting_period_str=f"Snapshot as of {data_as_of_timestamp.strftime('%d %b %Y')}"
    )
    if dho_kpis_structured_list:
        num_dho_kpis = len(dho_kpis_structured_list); cols_per_row_dho = 4
        for i in range(0, num_dho_kpis, cols_per_row_dho):
            kpi_display_cols = st.columns(cols_per_row_dho)
            for j, kpi_data_item in enumerate(dho_kpis_structured_list[i : i + cols_per_row_dho]):
                with kpi_display_cols[j]: render_web_kpi_card(**kpi_data_item) # Unpack dict
    else: st.info("District KPIs could not be structured from available summary.")
else:
    st.warning("District-wide summary KPIs are currently unavailable. Check data aggregation and enrichment pipeline.")
st.markdown("---")

# --- Tabbed Interface for Detailed District Analysis ---
st.header("🔍 In-Depth District Analysis Modules")
dho_tab_names_list = ["🗺️ Geospatial Overview", "📈 District Trends", "🆚 Zonal Comparison", "🎯 Intervention Planning"]
tab_map_dho, tab_trends_dho, tab_compare_dho, tab_intervene_dho = st.tabs(dho_tab_names_list)

with tab_map_dho:
    st.subheader("Interactive District Health & Environmental Map")
    if district_gdf_dho is not None and not district_gdf_dho.empty:
        render_district_interactive_map_web( # This component manages its own metric selectbox
            district_gdf_main_enriched=district_gdf_dho,
            default_selected_metric_key='avg_risk_score', # DHO often wants to see risk first
            reporting_period_str=f"Zonal Data as of {data_as_of_timestamp.strftime('%d %b %Y')}"
        )
    else:
        st.warning("Map visualization unavailable: Enriched district geographic data is missing or empty.")

with tab_trends_dho:
    trend_period_str = f"{selected_start_dho_trends_page.strftime('%d %b %Y')} - {selected_end_dho_trends_page.strftime('%d %b %Y')}"
    st.subheader(f"District-Wide Health & Environmental Trends ({trend_period_str})")
    
    # Filter historical_health_df and historical_iot_df for the selected trend period
    trend_health_data_for_dho = pd.DataFrame()
    if not historical_health_df.empty and 'encounter_date' in historical_health_df.columns:
        trend_health_data_for_dho = historical_health_df[
            (pd.to_datetime(historical_health_df['encounter_date']).dt.date >= selected_start_dho_trends_page) &
            (pd.to_datetime(historical_health_df['encounter_date']).dt.date <= selected_end_dho_trends_page)
        ].copy()

    trend_iot_data_for_dho = pd.DataFrame()
    if historical_iot_df is not None and not historical_iot_df.empty and 'timestamp' in historical_iot_df.columns:
        trend_iot_data_for_dho = historical_iot_df[
            (pd.to_datetime(historical_iot_df['timestamp']).dt.date >= selected_start_dho_trends_page) &
            (pd.to_datetime(historical_iot_df['timestamp']).dt.date <= selected_end_dho_trends_page)
        ].copy()

    district_trends_data_dict = calculate_district_trends_data(
        trend_health_data_for_dho, trend_iot_data_for_dho,
        selected_start_dho_trends_page, selected_end_dho_trends_page,
        trend_period_str
    )
    # Display logic for trends (iterate through dict, use plot_annotated_line_chart_web)
    # ... (Detailed display logic for each trend from previous refactor of district_trends_tab) ...
    # Example for disease incidence:
    if district_trends_data_dict.get("disease_incidence_trends"):
        st.markdown("###### Key Disease Incidence (Weekly):") # Assuming weekly calc
        disease_trends = district_trends_data_dict["disease_incidence_trends"]
        # Display in columns, max 2 charts per row for DHO view
        max_charts_per_row = 2 
        chart_cols_trends = st.columns(min(len(disease_trends), max_charts_per_row) or 1)
        current_col_idx = 0
        for cond, series in disease_trends.items():
            if series is not None and not series.empty:
                with chart_cols_trends[current_col_idx % max_charts_per_row]:
                    st.plotly_chart(plot_annotated_line_chart_web(series, f"{cond} New Cases", y_axis_is_count=True), use_container_width=True)
                current_col_idx += 1


with tab_compare_dho:
    st.subheader("Comparative Zonal Analysis")
    if district_gdf_dho is not None and not district_gdf_dho.empty:
        zonal_comparison_output = prepare_zonal_comparison_data(district_gdf_dho, current_data_timestamp_str)
        if zonal_comparison_output["zonal_comparison_table_df"] is not None:
            st.markdown("###### **Aggregated Zonal Metrics Comparison Table**")
            # The UI for selecting metric for bar chart or for detailed styling of table happens here.
            # For now, display the prepared DataFrame directly.
            st.dataframe(zonal_comparison_output["zonal_comparison_table_df"], height=min(500, len(zonal_comparison_output["zonal_comparison_table_df"])*38+60) , use_container_width=True)
            
            # Optional: Allow DHO to select a metric from comparison_metrics_config to plot a bar chart
            # This would involve a st.selectbox and then calling plot_bar_chart_web
            # Example: show one default bar chart (Avg AI Risk)
            comp_config = zonal_comparison_output.get("comparison_metrics_config",{})
            if "Avg. AI Risk Score (Zone)" in comp_config:
                risk_col_name = comp_config["Avg. AI Risk Score (Zone)"]["col"]
                # DataFrame needs 'name' or 'zone_id' as a column for x-axis. prepare_zonal_comparison_data sets index.
                df_for_bar_comp = zonal_comparison_output["zonal_comparison_table_df"].reset_index()
                zone_id_col_comp = df_for_bar_comp.columns[0] # Likely 'name' or 'zone_id'
                st.plotly_chart(plot_bar_chart_web(df_for_bar_comp, x_col_bar=zone_id_col_comp, y_col_bar=risk_col_name, title_bar="Avg. AI Risk Score by Zone", sort_values_by_web=risk_col_name), use_container_width=True)

        if zonal_comparison_output["data_availability_notes"]:
            for note in zonal_comparison_output["data_availability_notes"]: st.caption(note)
    else:
        st.warning("Zonal comparison data unavailable: Enriched district geographic data is missing.")

with tab_intervene_dho:
    st.subheader("Targeted Intervention Planning Assistant")
    if district_gdf_dho is not None and not district_gdf_dho.empty and all_available_criteria_options:
        # User selects criteria for intervention
        # `all_available_criteria_options` is from get_dho_command_center_datasets
        default_criteria_selection = list(all_available_criteria_options.keys())[0:min(2, len(all_available_criteria_options))] \
                                     if all_available_criteria_options else []
        
        selected_criteria_dho_intervene = st.multiselect(
            "Select Criteria to Identify Priority Zones (logical OR):",
            options=list(all_available_criteria_options.keys()),
            default=default_criteria_selection,
            key="dho_page_intervention_criteria_select_v1"
        )
        
        intervention_results_data = identify_priority_zones_for_intervention(
            district_gdf_main_enriched=district_gdf_dho,
            selected_criteria_display_names=selected_criteria_dho_intervene,
            available_criteria_options=all_available_criteria_options, # Pass the full options dict for lookup
            reporting_period_str=current_data_timestamp_str
        )
        
        if intervention_results_data["priority_zones_for_intervention_df"] is not None and \
           not intervention_results_data["priority_zones_for_intervention_df"].empty:
            st.markdown(f"###### **{len(intervention_results_data['priority_zones_for_intervention_df'])} Zone(s) Flagged for Intervention Based on: {', '.join(intervention_results_data.get('applied_criteria',[]))}**")
            st.dataframe(intervention_results_data["priority_zones_for_intervention_df"], use_container_width=True, height=min(450, len(intervention_results_data["priority_zones_for_intervention_df"])*40+50))
        elif selected_criteria_dho_intervene: # If criteria were selected but no zones met them
             st.success("✅ No zones currently meet the selected combination of high-priority criteria.")
        else:
             st.info("Please select one or more criteria above to identify priority zones for intervention.")

        if intervention_results_data["data_availability_notes"]:
            for note in intervention_results_data["data_availability_notes"]: st.caption(note)
    else:
        st.warning("Intervention planning tools unavailable: District geographic data or criteria definitions missing.")

logger.info("DHO Strategic Command Center page generated.")
