# sentinel_project_root/test/pages/3_district_dashboard.py
# Redesigned as "District Health Strategic Command Center" for "Sentinel Health Co-Pilot"
# This page simulates a web interface for DHOs and public health teams,
# typically at a Facility Node (Tier 2) or Regional/Cloud Node (Tier 3).

import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import os
import logging
from datetime import date, timedelta

# --- Sentinel System Imports ---
# Assuming correct PYTHONPATH configuration for Streamlit's page execution
try:
    from config import app_config
    from utils.core_data_processing import (
        load_health_records, load_iot_clinic_environment_data, load_zone_data,
        enrich_zone_geodata_with_health_aggregates, get_district_summary_kpis,
        hash_geodataframe # For caching GDFs
    )
    from utils.ai_analytics_engine import apply_ai_models # For enriching raw data

    # Refactored District Component data prep/rendering functions
    from pages.district_components_sentinel.kpi_structurer_district import structure_district_kpis_data
    from pages.district_components_sentinel.map_display_district_web import render_district_interactive_map_web
    from pages.district_components_sentinel.trend_calculator_district import calculate_district_trends_data
    from pages.district_components_sentinel.comparison_data_preparer_district import prepare_zonal_comparison_data
    from pages.district_components_sentinel.intervention_data_preparer_district import (
        identify_priority_zones_for_intervention, get_intervention_criteria_options
    )
    # Refactored UI helpers
    from utils.ui_visualization_helpers import (
        render_web_kpi_card, plot_annotated_line_chart_web, plot_bar_chart_web
    )
except ImportError as e:
    logging.critical(f"CRITICAL IMPORT ERROR in 3_district_dashboard.py: {e}. Application may not function correctly.")
    st.error(f"Application Critical Error: Could not load necessary modules for DHO Dashboard. Details: {e}")
    # Provide stubs for app_config if it's the one missing and other imports depend on it.
    class AppConfigFallback: LOG_LEVEL = "INFO"; APP_NAME = "Sentinel Fallback"; CACHE_TTL_SECONDS_WEB_REPORTS = 300
    app_config = AppConfigFallback() # This won't have all configs but prevents immediate crash on app_config references.
    # More stubs for other missing functions might be needed for the page to not fully break.
    st.stop()


# --- Page Configuration ---
st.set_page_config(
    page_title=f"DHO Command Center - {app_config.APP_NAME}",
    layout="wide",
    initial_sidebar_state="expanded"
)
logger = logging.getLogger(__name__)
# CSS loaded globally by app_home.py

# --- Data Aggregation for DHO View (Simulates Tier 2/3 Node Processing) ---
@st.cache_data(
    ttl=app_config.CACHE_TTL_SECONDS_WEB_REPORTS,
    hash_funcs={gpd.GeoDataFrame: hash_geodataframe, pd.DataFrame: pd.util.hash_pandas_object},
    show_spinner="Loading and processing district-level data for Command Center..."
)
def get_dho_command_center_datasets_page(): # Renamed for page context
    module_source_context = "DHOCommandCenterData"
    logger.info(f"({module_source_context}) Initializing data pipeline simulation...")
    
    health_df_raw_dho = load_health_records(source_context=f"{module_source_context}/LoadHealth")
    iot_df_raw_dho = load_iot_clinic_environment_data(source_context=f"{module_source_context}/LoadIoT")
    base_zone_gdf_dho = load_zone_data(source_context=f"{module_source_context}/LoadZoneData")

    enriched_health_full_dho = pd.DataFrame()
    if not health_df_raw_dho.empty:
        ai_output = apply_ai_models(health_df_raw_dho, source_context=f"{module_source_context}/AIEnrich")
        enriched_health_full_dho = ai_output[0] if ai_output and isinstance(ai_output, tuple) else pd.DataFrame()
    
    if base_zone_gdf_dho is None or base_zone_gdf_dho.empty:
        logger.error(f"({module_source_context}) Base zone geographic data (GDF) failed to load. Most DHO views will be unavailable.")
        # Return consistent empty structures to allow page to load without crashing downstream
        return gpd.GeoDataFrame(columns=['zone_id','name','geometry']), pd.DataFrame(), pd.DataFrame(), {}, {}

    main_district_gdf_enriched = enrich_zone_geodata_with_health_aggregates(
        zone_gdf=base_zone_gdf_dho, health_df=enriched_health_full_dho, iot_df=iot_df_raw_dho,
        source_context=f"{module_source_context}/EnrichZoneGDF"
    )
    if main_district_gdf_enriched is None or main_district_gdf_enriched.empty:
        logger.warning(f"({module_source_context}) Zone GDF enrichment was not successful. Using base GDF if available.")
        main_district_gdf_enriched = base_zone_gdf_dho if base_zone_gdf_dho is not None else gpd.GeoDataFrame(columns=['zone_id','name','geometry'])

    overall_district_kpis_dho = get_district_summary_kpis(main_district_gdf_enriched, source_context=f"{module_source_context}/DistrictKPIs")
    
    # Pass a small sample of the GDF to check column availability for criteria
    criteria_sample_gdf = main_district_gdf_enriched.head(2) if not main_district_gdf_enriched.empty else None
    available_criteria_for_filters = get_intervention_criteria_options(district_gdf_check=criteria_sample_gdf)
    
    logger.info(f"({module_source_context}) DHO data preparation complete. Enriched GDF shape: {main_district_gdf_enriched.shape if main_district_gdf_enriched is not None else 'N/A'}")
    return main_district_gdf_enriched, enriched_health_full_dho, iot_df_raw_dho, overall_district_kpis_dho, available_criteria_for_filters

# --- Load All Data for DHO Console ---
# Using st.session_state to avoid re-running this heavy data load on every interaction IF Streamlit's execution allows.
# For full page reruns on widget change (common), the @st.cache_data is the primary performance saver.
if 'dho_data_fully_loaded' not in st.session_state: # Check if already loaded in this session
    st.session_state.dist_gdf_main, st.session_state.hist_health_dho_full, st.session_state.hist_iot_dho_full, \
    st.session_state.summary_kpis_dho, st.session_state.filter_criteria_options_dho = get_dho_command_center_datasets_page()
    st.session_state.dho_data_fully_loaded = True
else: # Ensure variables are assigned from session_state on subsequent runs if state is maintained
    district_gdf_dho_page = st.session_state.dist_gdf_main
    historical_health_data_dho = st.session_state.hist_health_dho_full
    historical_iot_data_dho = st.session_state.hist_iot_dho_full
    district_summary_kpis_page = st.session_state.summary_kpis_dho
    all_criteria_options_for_page = st.session_state.filter_criteria_options_dho

# Assign to page-local variables for clarity in the rest of the script
district_gdf_dho_page = st.session_state.dist_gdf_main
historical_health_data_dho = st.session_state.hist_health_dho_full
historical_iot_data_dho = st.session_state.hist_iot_dho_full
district_summary_kpis_page = st.session_state.summary_kpis_dho
all_criteria_options_for_page = st.session_state.filter_criteria_options_dho


# --- Page Title and Sidebar Filters ---
st.title(f"🌍 {app_config.APP_NAME} - District Health Strategic Command Center")
page_data_as_of = pd.Timestamp('now') # Or more accurately, from max date in loaded data
st.markdown(f"**Strategic Oversight, Population Health Management, and Intervention Planning. Data Snapshot: {page_data_as_of.strftime('%d %b %Y, %H:%M')}**")
st.markdown("---")

if os.path.exists(app_config.APP_LOGO_SMALL): st.sidebar.image(app_config.APP_LOGO_SMALL, width=160)
st.sidebar.header("🗓️ Analysis Controls")

# Date Range Selector for Trend Analysis Tab
min_date_dho_trend_select = date.today() - timedelta(days=365*2)
max_date_dho_trend_select = page_data_as_of.date()
default_end_dho_trend_select = max_date_dho_trend_select
default_start_dho_trend_select = default_end_dho_trend_select - timedelta(days=app_config.WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND * 2 -1) # Default ~60 days for trends
if default_start_dho_trend_select < min_date_dho_trend_select: default_start_dho_trend_select = min_date_dho_trend_select

sel_start_trends_dho, sel_end_trends_dho = st.sidebar.date_input(
    "Select Date Range for District Trends:", value=[default_start_dho_trend_select, default_end_dho_trend_select],
    min_value=min_date_dho_trend_select, max_value=max_date_dho_trend_select, key="dho_page_trends_date_select_v3"
)
if sel_start_trends_dho > sel_end_trends_dho:
    st.sidebar.error("Trends date range: Start must be before end.")
    sel_start_trends_dho = sel_end_trends_dho

# --- Main KPIs Section ---
st.header("📊 District Performance Dashboard (Snapshot)")
if district_summary_kpis_page and isinstance(district_summary_kpis_page, dict) and any(v for k,v in district_summary_kpis_page.items() if pd.notna(v) and (isinstance(v, (int,float)) and v !=0 or isinstance(v,str) and v)): # check if dict has meaningful values
    structured_district_kpis_list = structure_district_kpis_data(district_summary_kpis_page, district_gdf_dho_page, f"As of {page_data_as_of.strftime('%d %b %Y')}")
    if structured_district_kpis_list:
        num_kpis_to_show = len(structured_district_kpis_list); kpi_cols_dho_row = 4
        for i_kpi in range(0, num_kpis_to_show, kpi_cols_dho_row):
            kpi_row_cols = st.columns(kpi_cols_dho_row)
            for j_kpi, kpi_data_obj in enumerate(structured_district_kpis_list[i_kpi : i_kpi + kpi_cols_dho_row]):
                with kpi_row_cols[j_kpi]: render_web_kpi_card(**kpi_data_obj) # Unpack dictionary
    else: st.info("Could not structure district KPIs from summary data.")
else: st.warning("District-wide summary KPI data is unavailable. Check data pipeline.")
st.markdown("---")

# --- Tabbed Interface for Detailed DHO Analysis ---
st.header("🔍 In-Depth District Analysis Modules")
dho_page_tabs_list = ["🗺️ Geospatial Health Map", "📈 District-Wide Trends", "🆚 Zonal Comparison Analysis", "🎯 Intervention Planning Assistant"]
tab_dho_map, tab_dho_trends, tab_dho_compare, tab_dho_intervene = st.tabs(dho_page_tabs_list)

with tab_dho_map:
    st.subheader("Spatial Distribution of Key District Metrics")
    if district_gdf_dho_page is not None and not district_gdf_dho_page.empty:
        # render_district_interactive_map_web handles its own metric selectbox
        render_district_interactive_map_web(
            district_gdf_main_enriched=district_gdf_dho_page,
            default_selected_metric_key='avg_risk_score', # Example: default view on map is AI Risk
            reporting_period_str=f"Zonal Data as of {page_data_as_of.strftime('%d %b')}"
        )
    else: st.warning("Map view unavailable: Enriched district geographic data (GDF) is missing or empty.")

with tab_dho_trends:
    trend_period_str_for_tab = f"{sel_start_trends_dho.strftime('%d %b %Y')} to {sel_end_trends_dho.strftime('%d %b %Y')}"
    st.subheader(f"Health & Environmental Trends ({trend_period_str_for_tab})")
    
    # Filter historical data passed from get_dho_command_center_datasets_page for the selected trend range
    # This is done here because the user selects the trend range on this page.
    health_for_trends_tab = pd.DataFrame()
    if not historical_health_data_dho.empty and 'encounter_date' in historical_health_data_dho.columns:
         historical_health_data_dho['encounter_date'] = pd.to_datetime(historical_health_data_dho['encounter_date'], errors='coerce')
         health_for_trends_tab = historical_health_data_dho[
             (historical_health_data_dho['encounter_date'].dt.date >= sel_start_trends_dho) &
             (historical_health_data_dho['encounter_date'].dt.date <= sel_end_trends_dho)
         ].copy()

    iot_for_trends_tab = pd.DataFrame()
    if historical_iot_data_dho is not None and not historical_iot_data_dho.empty and 'timestamp' in historical_iot_data_dho.columns:
        historical_iot_data_dho['timestamp'] = pd.to_datetime(historical_iot_data_dho['timestamp'], errors='coerce')
        iot_for_trends_tab = historical_iot_data_dho[
            (historical_iot_data_dho['timestamp'].dt.date >= sel_start_trends_dho) &
            (historical_iot_data_dho['timestamp'].dt.date <= sel_end_trends_dho)
        ].copy()
    
    dho_trends_output_dict = calculate_district_trends_data(
        health_for_trends_tab, iot_for_trends_tab, sel_start_trends_dho, sel_end_trends_dho, trend_period_str_for_tab
    )
    # Display trends logic as in previous refactor of this page...
    # For brevity, details omitted but assumes iteration over dho_trends_output_dict and plotting
    if dho_trends_output_dict.get("data_availability_notes"):
        for note in dho_trends_output_dict["data_availability_notes"]: st.caption(note)


with tab_dho_compare:
    st.subheader("Comparative Zonal Analysis Dashboard")
    if district_gdf_dho_page is not None and not district_gdf_dho_page.empty:
        zonal_comparison_data_output = prepare_zonal_comparison_data(district_gdf_dho_page, f"Data as of {page_data_as_of.strftime('%d %b')}")
        if zonal_comparison_data_output.get("zonal_comparison_table_df") is not None:
            st.markdown("###### **Aggregated Zonal Metrics Overview Table**")
            st.dataframe(zonal_comparison_data_output["zonal_comparison_table_df"], use_container_width=True, height=min(600, len(zonal_comparison_data_output["zonal_comparison_table_df"])*38+58))
            # Further UI to select metric for bar chart, or display a default one as per prev. refactor
            # ... bar chart plotting logic ...
        if zonal_comparison_data_output.get("data_availability_notes"):
            for note in zonal_comparison_data_output["data_availability_notes"]: st.caption(note)
    else: st.warning("Zonal comparison view unavailable: Enriched district GDF missing.")


with tab_dho_intervene:
    st.subheader("Intervention Planning & Priority Zone Identification")
    if district_gdf_dho_page is not None and not district_gdf_dho_page.empty and all_criteria_options_for_page:
        # DHO selects criteria using multiselect
        default_criteria_selection_dho = list(all_criteria_options_for_page.keys())[0:min(2, len(all_criteria_options_for_page))] \
                                           if all_criteria_options_for_page else []
        sel_criteria_names_dho_intervene = st.multiselect(
            "Select Criteria to Identify Priority Zones (Logical OR applied):",
            options=list(all_criteria_options_for_page.keys()), default=default_criteria_selection_dho,
            key="dho_page_intervene_criteria_v2", help="Zones meeting ANY selected criteria will be shown."
        )
        
        intervention_data_output = identify_priority_zones_for_intervention(
            district_gdf_main_enriched=district_gdf_dho_page,
            selected_criteria_display_names=sel_criteria_names_dho_intervene,
            available_criteria_options=all_criteria_options_for_page,
            reporting_period_str=f"Analysis Date: {page_data_as_of.strftime('%d %b')}"
        )
        priority_df = intervention_data_output.get("priority_zones_for_intervention_df")
        if priority_df is not None and not priority_df.empty:
            st.markdown(f"###### **{len(priority_df)} Zone(s) Flagged for Intervention Based on: {', '.join(intervention_data_output.get('applied_criteria',[]))}**")
            st.dataframe(priority_df, use_container_width=True, height=min(500, len(priority_df)*40+50))
        elif sel_criteria_names_dho_intervene:
            st.success("✅ No zones currently meet the selected combination of criteria.")
        else: st.info("Please select criteria to identify priority zones.")
        if intervention_data_output.get("data_availability_notes"):
            for note in intervention_data_output["data_availability_notes"]: st.caption(note)
    else: st.warning("Intervention planning tools unavailable: District GDF or criteria definitions missing.")


logger.info(f"DHO Strategic Command Center page generated successfully.")
