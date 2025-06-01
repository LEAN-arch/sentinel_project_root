# sentinel_project_root/test/pages/3_district_dashboard.py
# District Health Strategic Command Center for Sentinel Health Co-Pilot.

import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np # Kept for potential use
import os
import logging
from datetime import date, timedelta, datetime # Added datetime

# --- Sentinel System Imports ---
from config import app_config
from utils.core_data_processing import (
    load_health_records, load_iot_clinic_environment_data, load_zone_data,
    enrich_zone_geodata_with_health_aggregates, get_district_summary_kpis,
    hash_geodataframe # For caching GDFs
)
from utils.ai_analytics_engine import apply_ai_models # For enriching health data
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

# --- Page Configuration ---
st.set_page_config(
    page_title=f"DHO Command Center - {app_config.APP_NAME}",
    layout="wide",
    initial_sidebar_state="expanded"
)
logger = logging.getLogger(__name__)

# --- Data Aggregation and Preparation for DHO View ---
@st.cache_data(
    ttl=app_config.CACHE_TTL_SECONDS_WEB_REPORTS,
    hash_funcs={gpd.GeoDataFrame: hash_geodataframe, pd.DataFrame: pd.util.hash_pandas_object}, # Ensure GDFs are hashed correctly
    show_spinner="Aggregating district-level operational data..."
)
def get_dho_command_center_datasets_cached(): # Renamed for clarity that it's the cached one
    """
    Simulates the comprehensive data pipeline for the DHO Command Center.
    Loads, enriches (AI), aggregates to zones, and calculates district KPIs.
    Returns: Tuple (enriched_district_gdf, full_historical_health_df, full_historical_iot_df, district_summary_kpis, intervention_criteria_options_dict)
    """
    logger.info("DHO Command Center: Initializing full data pipeline simulation (cached)...")
    
    health_raw = load_health_records(source_context="DHOData/LoadHealth")
    iot_raw = load_iot_clinic_environment_data(source_context="DHOData/LoadIoT")
    base_zone_gdf = load_zone_data(source_context="DHOData/LoadZone")

    if not isinstance(base_zone_gdf, gpd.GeoDataFrame) or base_zone_gdf.empty:
        logger.error("DHO Command Center: Base zone geographic data (GDF) failed to load or is empty. Critical failure.")
        # Return empty structures with expected types
        return gpd.GeoDataFrame(), pd.DataFrame(), pd.DataFrame(), {}, {}

    # Enrich raw health data with AI model outputs
    full_health_enriched_df = pd.DataFrame()
    ai_added_cols_dho = ['ai_risk_score', 'ai_followup_priority_score'] # Expected from AI
    base_health_cols_dho = health_raw.columns.tolist() if isinstance(health_raw, pd.DataFrame) and not health_raw.empty else []

    if isinstance(health_raw, pd.DataFrame) and not health_raw.empty:
        enriched_output = apply_ai_models(health_raw, source_context="DHOData/AIEnrichHealth")
        full_health_enriched_df = enriched_output[0]
    else:
        logger.warning("DHO Command Center: Raw health data empty or invalid. AI enrichment skipped.")
        full_health_enriched_df = pd.DataFrame(columns=list(set(base_health_cols_dho + ai_added_cols_dho)))

    # Enrich the Zonal GeoDataFrame
    # This is the primary dataset for many DHO views.
    district_gdf_enriched = enrich_zone_geodata_with_health_aggregates(
        zone_gdf=base_zone_gdf, # This GDF already has base attributes from load_zone_data
        health_df=full_health_enriched_df, # Pass the AI-enriched health data
        iot_df=iot_raw, # Pass raw IoT; enrichment function filters/aggregates by zone
        source_context="DHOData/EnrichZoneGDF"
    )
    if not isinstance(district_gdf_enriched, gpd.GeoDataFrame) or district_gdf_enriched.empty:
        logger.warning("DHO Command Center: GDF enrichment resulted in empty/None. Defaulting to base zone GDF if available.")
        district_gdf_enriched = base_zone_gdf if isinstance(base_zone_gdf, gpd.GeoDataFrame) else gpd.GeoDataFrame()

    # Calculate overall district summary KPIs from the enriched GDF
    summary_kpis_district = get_district_summary_kpis(district_gdf_enriched, source_context="DHOData/CalcDistrictKPIs")

    # Prepare filter/criteria options for tabs (e.g., for Intervention Planning)
    # Pass a sample of the GDF to check column availability.
    intervention_criteria = get_intervention_criteria_options(
        district_gdf_check_sample=district_gdf_enriched.head() if not district_gdf_enriched.empty else None
    )
    
    logger.info(f"DHO Command Center: Data preparation complete. Enriched GDF shape: {district_gdf_enriched.shape if isinstance(district_gdf_enriched, gpd.GeoDataFrame) else 'N/A'}")
    return district_gdf_enriched, full_health_enriched_df, iot_raw, summary_kpis_district, intervention_criteria

# --- Load All Data for DHO Console (Using Cache and Session State) ---
# This simulates the data environment available to a DHO.
if 'dho_data_fully_loaded' not in st.session_state: # Use a more descriptive flag
    st.session_state.district_gdf, \
    st.session_state.historical_health_df_dho, \
    st.session_state.historical_iot_df_dho, \
    st.session_state.summary_kpis_dho, \
    st.session_state.intervention_options_dho = get_dho_command_center_datasets_cached()
    st.session_state.dho_data_fully_loaded = True # Set flag after successful load

# Use data from session state for consistency across interactions
district_gdf = st.session_state.district_gdf
health_data_hist_dho = st.session_state.historical_health_df_dho
iot_data_hist_dho = st.session_state.historical_iot_df_dho
district_kpis_summary = st.session_state.summary_kpis_dho
intervention_criteria_config = st.session_state.intervention_options_dho

# --- Page Title and Sidebar ---
st.title(f"🌍 {app_config.APP_NAME} - District Health Strategic Command Center")
# Use last data refresh time from a source if available, else use current time as placeholder
last_data_refresh_time = pd.Timestamp('now') # Placeholder
st.markdown(f"**Aggregated Zonal Intelligence, Resource Allocation, and Public Health Program Monitoring.** (Data as of: {last_data_refresh_time.strftime('%d %b %Y, %H:%M')})")
st.divider()

if os.path.exists(app_config.APP_LOGO_SMALL):
    st.sidebar.image(app_config.APP_LOGO_SMALL, width=150)
st.sidebar.header("🗓️ Analysis Filters")

# Trend Date Range Selector (Applies primarily to "District-Wide Trends" tab)
min_trend_date_dho = last_data_refresh_time.date() - timedelta(days=365 * 2) # Max 2 years historical
max_trend_date_dho = last_data_refresh_time.date()

default_trend_end_dho = max_trend_date_dho
# Default to 90 days for DHO trend view
default_trend_start_dho = default_trend_end_dho - timedelta(days=app_config.WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND * 3 - 1) 
if default_trend_start_dho < min_trend_date_dho: default_trend_start_dho = min_trend_date_dho

selected_trend_start_dho, selected_trend_end_dho = st.sidebar.date_input(
    "Select Date Range for Trend Analysis:",
    value=[default_trend_start_dho, default_trend_end_dho],
    min_value=min_trend_date_dho, max_value=max_trend_date_dho,
    key="dho_trend_date_selector"
)
if selected_trend_start_dho > selected_trend_end_dho:
    st.sidebar.error("DHO Trends: Start date must be on or before end date.")
    selected_trend_start_dho = selected_trend_end_dho

# --- Main Section 1: District-Wide KPIs ---
st.header("📊 District Performance Dashboard")
if district_kpis_summary and isinstance(district_kpis_summary, dict) and any(v is not None and not (isinstance(v, float) and np.isnan(v)) for v in district_kpis_summary.values()):
    # Pass GDF for context like total zone count if kpi_structurer needs it
    structured_dho_kpis = structure_district_kpis_data(
        district_kpis_summary, district_gdf, 
        reporting_period_str=f"Snapshot as of {last_data_refresh_time.strftime('%d %b %Y')}"
    )
    if structured_dho_kpis:
        num_kpis = len(structured_dho_kpis)
        cols_per_row = 4 # Max KPIs per row
        for i in range(0, num_kpis, cols_per_row):
            kpi_row_cols = st.columns(cols_per_row)
            for kpi_idx_in_row, kpi_data in enumerate(structured_dho_kpis[i : i + cols_per_row]):
                with kpi_row_cols[kpi_idx_in_row]: render_web_kpi_card(**kpi_data)
    else: 
        st.info("District KPIs could not be structured from the available summary data.")
else:
    st.warning("District-wide summary KPIs are currently unavailable. Please check data aggregation and enrichment pipeline.")
st.divider()

# --- Tabbed Interface for Detailed District Analysis ---
st.header("🔍 In-Depth District Analysis Modules")
tab_titles_dho = ["🗺️ Geospatial Overview", "📈 District Trends", "🆚 Zonal Comparison", "🎯 Intervention Planning"]
tab_map, tab_trends, tab_compare, tab_intervene = st.tabs(tab_titles_dho)

with tab_map:
    st.subheader("Interactive District Health & Environmental Map")
    if isinstance(district_gdf, gpd.GeoDataFrame) and not district_gdf.empty:
        render_district_interactive_map_web(
            district_gdf_main_enriched=district_gdf,
            default_selected_metric_col_name='avg_risk_score', # DHO often interested in risk first
            reporting_period_str=f"Zonal Data as of {last_data_refresh_time.strftime('%d %b %Y')}"
        )
    else:
        st.warning("Map visualization unavailable: Enriched district geographic data is missing or empty.")

with tab_trends:
    trend_period_display = f"{selected_trend_start_dho.strftime('%d %b %Y')} - {selected_trend_end_dho.strftime('%d %b %Y')}"
    st.subheader(f"District-Wide Health & Environmental Trends ({trend_period_display})")
    
    # Filter historical data for the selected trend period
    health_for_trends_tab = pd.DataFrame()
    if isinstance(health_data_hist_dho, pd.DataFrame) and not health_data_hist_dho.empty and 'encounter_date' in health_data_hist_dho.columns:
        health_for_trends_tab = health_data_hist_dho[
            (pd.to_datetime(health_data_hist_dho['encounter_date']).dt.date >= selected_trend_start_dho) &
            (pd.to_datetime(health_data_hist_dho['encounter_date']).dt.date <= selected_trend_end_dho)
        ].copy()

    iot_for_trends_tab = pd.DataFrame()
    if isinstance(iot_data_hist_dho, pd.DataFrame) and not iot_data_hist_dho.empty and 'timestamp' in iot_data_hist_dho.columns:
        iot_for_trends_tab = iot_data_hist_dho[
            (pd.to_datetime(iot_data_hist_dho['timestamp']).dt.date >= selected_trend_start_dho) &
            (pd.to_datetime(iot_data_hist_dho['timestamp']).dt.date <= selected_trend_end_dho)
        ].copy()

    if health_for_trends_tab.empty and iot_for_trends_tab.empty:
        st.info(f"No health or IoT data available for the selected trend period: {trend_period_display}")
    else:
        district_trends = calculate_district_trends_data(
            health_for_trends_tab, iot_for_trends_tab,
            selected_trend_start_dho, selected_trend_end_dho, # For context, actual filtering done above
            trend_period_display # For reporting_period key in output
        )
        
        # Display Disease Incidence Trends
        disease_incidence_map = district_trends.get("disease_incidence_trends", {})
        if disease_incidence_map:
            st.markdown("###### Key Disease Incidence (Weekly New Cases):")
            max_disease_charts_per_row = 2 
            disease_chart_keys = list(disease_incidence_map.keys())
            for i in range(0, len(disease_chart_keys), max_disease_charts_per_row):
                cols_disease_trend = st.columns(max_disease_charts_per_row)
                for j, key in enumerate(disease_chart_keys[i : i + max_disease_charts_per_row]):
                    series_data = disease_incidence_map[key]
                    if isinstance(series_data, pd.Series) and not series_data.empty:
                        with cols_disease_trend[j]:
                            st.plotly_chart(plot_annotated_line_chart_web(series_data, f"{key} New Cases", y_is_count=True, y_axis_label="# Cases"), use_container_width=True)
        
        # Display Other Trends (AI Risk, Steps, CO2)
        other_trends_to_plot = {
            "Avg. Patient AI Risk": (district_trends.get("avg_patient_ai_risk_trend"), "AI Risk Score"),
            "Avg. Patient Daily Steps": (district_trends.get("avg_patient_daily_steps_trend"), "Steps/Day"),
            "Avg. Clinic CO2 Levels": (district_trends.get("avg_clinic_co2_trend"), "CO2 (ppm)")
        }
        for trend_title, (trend_data, y_label) in other_trends_to_plot.items():
            if isinstance(trend_data, pd.Series) and not trend_data.empty:
                st.plotly_chart(plot_annotated_line_chart_web(trend_data, trend_title, y_axis_label=y_label, y_is_count= (y_label == "Steps/Day")), use_container_width=True) # Steps can be count-like
        
        if district_trends.get("data_availability_notes"):
            for note in district_trends["data_availability_notes"]: st.caption(f"Trend Note: {note}")


with tab_compare:
    st.subheader("Comparative Zonal Analysis")
    if isinstance(district_gdf, gpd.GeoDataFrame) and not district_gdf.empty:
        zonal_comparison = prepare_zonal_comparison_data(district_gdf, f"Data as of {last_data_refresh_time.strftime('%d %b %Y')}")
        
        comp_table_df = zonal_comparison.get("zonal_comparison_table_df")
        if isinstance(comp_table_df, pd.DataFrame) and not comp_table_df.empty:
            st.markdown("###### **Aggregated Zonal Metrics Comparison Table:**")
            # Make table scrollable if many zones/metrics
            st.dataframe(comp_table_df, height=min(500, len(comp_table_df)*38 + 60) , use_container_width=True) # Index is zone name/ID
            
            # Example: Default bar chart for a key comparison metric
            comp_metrics_cfg = zonal_comparison.get("comparison_metrics_config", {})
            default_bar_metric_display = "Avg. AI Risk Score (Zone)"
            if default_bar_metric_display in comp_metrics_cfg:
                metric_details_bar = comp_metrics_cfg[default_bar_metric_display]
                risk_col_bar = metric_details_bar["col"]
                # The comparison_df has zone name/ID as index, reset for plotting if needed by plot_bar_chart_web
                df_for_bar = comp_table_df.reset_index() 
                zone_id_col_for_bar = df_for_bar.columns[0] # First column is the reset index
                st.plotly_chart(plot_bar_chart_web(df_for_bar, x_col=zone_id_col_for_bar, y_col=risk_col_bar, title=f"{default_bar_metric_display}", sort_by=risk_col_bar, sort_ascending=False, x_axis_label="Zone"), use_container_width=True)
        else:
            st.info("No data available for zonal comparison table.")

        if zonal_comparison.get("data_availability_notes"):
            for note in zonal_comparison["data_availability_notes"]: st.caption(f"Comparison Note: {note}")
    else:
        st.warning("Zonal comparison data unavailable: Enriched district geographic data is missing or empty.")

with tab_intervene:
    st.subheader("Targeted Intervention Planning Assistant")
    if isinstance(district_gdf, gpd.GeoDataFrame) and not district_gdf.empty and intervention_criteria_config:
        default_criteria_keys = list(intervention_criteria_config.keys())
        # Select first 2 criteria as default if available
        default_selection = default_criteria_keys[0:min(2, len(default_criteria_keys))] 
        
        selected_intervention_criteria = st.multiselect(
            "Select Criteria to Identify Priority Zones (zones meeting ANY selected criterion will be shown):",
            options=default_criteria_keys,
            default=default_selection,
            key="dho_intervention_criteria_selector"
        )
        
        intervention_data = identify_priority_zones_for_intervention(
            district_gdf_main_enriched=district_gdf,
            selected_criteria_display_names=selected_intervention_criteria,
            available_criteria_options=intervention_criteria_config,
            reporting_period_str=f"Data as of {last_data_refresh_time.strftime('%d %b %Y')}"
        )
        
        priority_zones_df_intervene = intervention_data.get("priority_zones_for_intervention_df")
        applied_criteria_list = intervention_data.get("applied_criteria_names", [])

        if isinstance(priority_zones_df_intervene, pd.DataFrame) and not priority_zones_df_intervene.empty:
            st.markdown(f"###### **{len(priority_zones_df_intervene)} Zone(s) Flagged for Intervention Based on: {', '.join(applied_criteria_list) if applied_criteria_list else 'Selected Criteria'}**")
            # Make table scrollable
            st.dataframe(priority_zones_df_intervene, use_container_width=True, height=min(450, len(priority_zones_df_intervene)*40 + 50), hide_index=True)
        elif selected_intervention_criteria: # Criteria selected, but no zones met them
             st.success(f"✅ No zones currently meet the selected criteria: {', '.join(applied_criteria_list) if applied_criteria_list else 'None applied due to data issues'}.")
        else: # No criteria selected by user
             st.info("Please select one or more criteria above to identify priority zones for intervention.")

        if intervention_data.get("data_availability_notes"):
            for note in intervention_data["data_availability_notes"]: st.caption(f"Intervention Note: {note}")
    else:
        st.warning("Intervention planning tools unavailable: District geographic data or criteria definitions are missing.")

logger.info(f"DHO Strategic Command Center page generated. Data as of: {last_data_refresh_time.isoformat()}")
