# sentinel_project_root/test/pages/2_clinic_dashboard.py
# Redesigned as "Clinic Operations & Management Console" for "Sentinel Health Co-Pilot"
# This page simulates a web interface for Clinic Managers or Lead Clinicians
# at a Facility Node (Tier 2). It focuses on operational oversight, service quality,
# resource management, and local health/environmental conditions impacting the clinic.

import streamlit as st
import pandas as pd
import numpy as np
import os
import logging
from datetime import date, timedelta

# --- Sentinel System Imports ---
from config import app_config # Uses the new, redesigned app_config

# Core data loading utilities
from utils.core_data_processing import (
    load_health_records,
    load_iot_clinic_environment_data,
    get_clinic_summary # For overall clinic performance metrics
)
# AI engine for data enrichment (simulates processing at Facility Node)
from utils.ai_analytics_engine import apply_ai_models

# Refactored Clinic Component Data Preparation Functions
# These are assumed to be in a similar 'sentinel' suffixed sub-package or renamed.
# e.g., from test/pages/clinic_components_sentinel/your_module.py
from pages.clinic_components_sentinel.environmental_kpi_calculator import calculate_clinic_environmental_kpis
from pages.clinic_components_sentinel.main_kpi_structurer import structure_main_clinic_kpis_data, structure_disease_specific_kpis_data
from pages.clinic_components_sentinel.epi_data_calculator import calculate_clinic_epi_data
from pages.clinic_components_sentinel.environment_detail_preparer import prepare_clinic_environment_details_data
from pages.clinic_components_sentinel.patient_focus_data_preparer import prepare_clinic_patient_focus_data
from pages.clinic_components_sentinel.supply_forecast_generator import prepare_clinic_supply_forecast_data
from pages.clinic_components_sentinel.testing_insights_analyzer import prepare_clinic_testing_insights_data

# Refactored UI helpers for web reports
from utils.ui_visualization_helpers import (
    render_web_kpi_card,
    plot_annotated_line_chart_web,
    plot_bar_chart_web,
    plot_donut_chart_web
    # Add other plotters if used by specific tab visualizations
)

# --- Page Configuration ---
st.set_page_config(
    page_title=f"Clinic Console - {app_config.APP_NAME}",
    layout="wide",
    initial_sidebar_state="expanded"
)
logger = logging.getLogger(__name__)
# CSS loading would be handled by app_home.py or a @st.cache_resource here.

# --- Data Loading for Clinic Console (Simulates Facility Node Data Access) ---
@st.cache_data(
    ttl=app_config.CACHE_TTL_SECONDS_WEB_REPORTS,
    show_spinner="Loading clinic operational dataset for console..."
)
def get_clinic_console_comprehensive_data(selected_start_date: date, selected_end_date: date):
    """
    Simulates fetching, enriching, and preparing all data for the Clinic Console.
    - Loads raw health records and IoT data for the clinic's catchment/operation.
    - Applies AI enrichment (simulating Facility Node processing).
    - Filters data for the selected period.
    - Generates a core clinic summary for the period.
    Returns:
        full_historical_health_df: Enriched health data for all time (for historical rates in supply forecast).
        period_health_df: Enriched health data filtered for the selected date range.
        period_iot_df: IoT data filtered for the selected date range.
        period_clinic_summary_kpis: Output from get_clinic_summary for the selected period.
    """
    logger.info(f"Clinic Console: Loading data for period {selected_start_date} to {selected_end_date}")
    
    # 1. Load raw base data (simulating access to clinic's data store)
    health_df_raw_clinic = load_health_records(file_path=app_config.HEALTH_RECORDS_CSV, source_context="ClinicConsole/RawHealth")
    iot_df_raw_clinic = load_iot_clinic_environment_data(file_path=app_config.IOT_CLINIC_ENVIRONMENT_CSV, source_context="ClinicConsole/RawIoT")
    iot_data_globally_available = not iot_df_raw_clinic.empty # Flag if IoT source file exists

    # 2. Apply AI Models (simulates processing as data arrives or is batched at Facility Node)
    full_historical_health_df_enriched = pd.DataFrame()
    if not health_df_raw_clinic.empty:
        # apply_ai_models returns (enriched_df, supply_df_forecast_from_ai_model). We only need enriched_df here.
        full_historical_health_df_enriched = apply_ai_models(health_df_raw_clinic, source_context="ClinicConsole/AIEnrich")[0]
    else:
        logger.warning("Clinic Console: Raw health data empty, AI enrichment skipped.")
        # Create an empty df with expected columns if main data loading fails to avoid errors downstream
        # This schema should ideally match columns produced by apply_ai_models after enrichment.
        # For simplicity, using health_df_raw_clinic.columns; a fixed schema list is more robust.
        cols_health = health_df_raw_clinic.columns if health_df_raw_clinic is not None else []
        cols_ai = ['ai_risk_score', 'ai_followup_priority_score'] # AI adds these
        full_historical_health_df_enriched = pd.DataFrame(columns=list(cols_health) + [c for c in cols_ai if c not in cols_health])


    # 3. Filter data for the selected period
    period_health_df = pd.DataFrame(columns=full_historical_health_df_enriched.columns)
    if not full_historical_health_df_enriched.empty and 'encounter_date' in full_historical_health_df_enriched.columns:
        # Ensure encounter_date is datetime for filtering
        full_historical_health_df_enriched['encounter_date'] = pd.to_datetime(full_historical_health_df_enriched['encounter_date'], errors='coerce')
        
        period_health_df = full_historical_health_df_enriched[
            (full_historical_health_df_enriched['encounter_date'].dt.date >= selected_start_date) &
            (full_historical_health_df_enriched['encounter_date'].dt.date <= selected_end_date)
        ].copy()
    
    period_iot_df = pd.DataFrame()
    if not iot_df_raw_clinic.empty and 'timestamp' in iot_df_raw_clinic.columns:
        iot_df_raw_clinic['timestamp'] = pd.to_datetime(iot_df_raw_clinic['timestamp'], errors='coerce')
        period_iot_df = iot_df_raw_clinic[
            (iot_df_raw_clinic['timestamp'].dt.date >= selected_start_date) &
            (iot_df_raw_clinic['timestamp'].dt.date <= selected_end_date)
        ].copy()

    # 4. Generate core clinic summary KPIs for the selected period
    period_clinic_summary_kpis = {}
    if not period_health_df.empty:
        period_clinic_summary_kpis = get_clinic_summary(period_health_df, source_context="ClinicConsole/PeriodSummary")
    else:
        logger.info("Clinic Console: No health data in selected period to generate summary KPIs.")
        # Initialize with default structure if empty to prevent downstream errors
        period_clinic_summary_kpis = {"test_summary_details": {}} # Minimum for some components
        
    return full_historical_health_df_enriched, period_health_df, period_iot_df, period_clinic_summary_kpis, iot_data_globally_available


# --- Page Title & Sidebar Filters ---
st.title(f"🏥 {app_config.APP_NAME} - Clinic Operations & Management Console")
st.markdown(f"**Service Performance, Patient Care Quality, Resource Management, and Facility Environment Monitoring**")
st.markdown("---")

if os.path.exists(app_config.APP_LOGO_SMALL):
    st.sidebar.image(app_config.APP_LOGO_SMALL, width=180)
st.sidebar.header("🗓️ Console Filters")

# Date Range selection for Clinic Console
default_days_clinic = app_config.WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND # e.g., 30 days
min_date_clinic = date.today() - timedelta(days=365) # Max 1 year historical view
max_date_clinic = date.today()

default_end_dt_clinic = max_date_clinic
default_start_dt_clinic = default_end_dt_clinic - timedelta(days=default_days_clinic - 1)
if default_start_dt_clinic < min_date_clinic: default_start_dt_clinic = min_date_clinic

selected_start_dt_clinic_console, selected_end_dt_clinic_console = st.sidebar.date_input(
    "Select Date Range for Clinic Review:", value=[default_start_dt_clinic, default_end_dt_clinic],
    min_value=min_date_clinic, max_value=max_date_clinic, key="clinic_console_date_range_v2"
)
if selected_start_dt_clinic_console > selected_end_dt_clinic_console:
    st.sidebar.error("Console Date Range: Start date must be before end date.")
    selected_start_dt_clinic_console = selected_end_dt_clinic_console # Auto-correct

current_period_str_clinic = f"{selected_start_dt_clinic_console.strftime('%d %b %Y')} - {selected_end_dt_clinic_console.strftime('%d %b %Y')}"

# --- Load Data Based on Selections ---
full_hist_health_data, period_health_data, period_iot_data, period_summary_kpis, iot_available_flag = get_clinic_console_comprehensive_data(
    selected_start_dt_clinic_console, selected_end_dt_clinic_console
)
st.info(f"Displaying Clinic Console for period: **{current_period_str_clinic}**")


# --- Section 1: Top-Level KPIs (Overall Performance & Environment) ---
st.header("🚀 Performance & Environment Snapshot")
# Performance KPIs
if period_summary_kpis and period_summary_kpis.get("test_summary_details"): # Check if summary has data
    main_kpi_data_list = structure_main_clinic_kpis_data(period_summary_kpis, current_period_str_clinic)
    disease_kpi_data_list = structure_disease_specific_kpis_data(period_summary_kpis, current_period_str_clinic)
    
    if main_kpi_data_list:
        st.markdown("##### **Overall Service Performance:**")
        main_kpi_cols = st.columns(len(main_kpi_data_list) if len(main_kpi_data_list) <= 4 else 4)
        for i, kpi_item in enumerate(main_kpi_data_list):
            with main_kpi_cols[i % 4]: render_web_kpi_card(**kpi_item) # Unpack dict to args
    if disease_kpi_data_list:
        st.markdown("##### **Key Disease Testing & Supply Indicators:**")
        disease_kpi_cols = st.columns(len(disease_kpi_data_list) if len(disease_kpi_data_list) <=4 else 4)
        for i, kpi_item in enumerate(disease_kpi_data_list):
            with disease_kpi_cols[i % 4]: render_web_kpi_card(**kpi_item)
else:
    st.warning(f"Core clinic performance KPIs could not be generated for {current_period_str_clinic}. Ensure health data and processing are correct.")

# Environmental KPIs
st.markdown("##### **Clinic Environment Quick Check:**")
env_kpi_output_dict = calculate_clinic_environmental_kpis(period_iot_data, current_period_str_clinic)
if env_kpi_output_dict and any(pd.notna(v) for k,v in env_kpi_output_dict.items() if "avg_" in k or "count" in k or "flag" in k):
    env_kpi_cols = st.columns(4) # Show key env KPIs
    with env_kpi_cols[0]: render_web_kpi_card("Avg. CO2", f"{env_kpi_output_dict.get('avg_co2_ppm_overall', 'N/A'):.0f}" if pd.notna(env_kpi_output_dict.get('avg_co2_ppm_overall')) else "N/A", units="ppm", icon="💨", status_level=env_kpi_output_dict.get('co2_status_level',"NEUTRAL"))
    with env_kpi_cols[1]: render_web_kpi_card("Avg. PM2.5", f"{env_kpi_output_dict.get('avg_pm25_ugm3_overall', 'N/A'):.1f}" if pd.notna(env_kpi_output_dict.get('avg_pm25_ugm3_overall')) else "N/A", units="µg/m³", icon="🌫️", status_level=env_kpi_output_dict.get('pm25_status_level',"NEUTRAL"))
    with env_kpi_cols[2]: render_web_kpi_card("Avg. Waiting Occupancy", f"{env_kpi_output_dict.get('avg_waiting_room_occupancy_persons', 'N/A'):.1f}" if pd.notna(env_kpi_output_dict.get('avg_waiting_room_occupancy_persons')) else "N/A", units="ppl", icon="👨‍👩‍👧‍👦", status_level=env_kpi_output_dict.get('occupancy_status_level',"NEUTRAL"))
    with env_kpi_cols[3]: render_web_kpi_card("High Noise Alerts", str(env_kpi_output_dict.get('noise_rooms_at_high_alert_count',0)), units="rooms", icon="🔊", status_level="HIGH_CONCERN" if env_kpi_output_dict.get('noise_rooms_at_high_alert_count',0) > 0 else "ACCEPTABLE")
else:
    if iot_available_flag: st.info("No environmental IoT data available for this specific period to display snapshot KPIs.")
    else: st.caption("Environmental IoT data source not available for this clinic.")
st.markdown("---")


# --- Tabbed Interface for Detailed Operational Areas ---
st.header("🛠️ Operational Areas Deep Dive")
clinic_console_tab_names = [
    "📈 Local Epidemiology", "🔬 Testing & Diagnostics", "💊 Supply Management",
    "🧍 Patient Focus & Review", "🌿 Facility Environment Details"
]
tab_clinic_epi, tab_clinic_testing, tab_clinic_supply, tab_clinic_patients, tab_clinic_env_details = st.tabs(clinic_console_tab_names)

with tab_clinic_epi:
    st.subheader(f"Clinic-Level Epidemiological Intel ({current_period_str_clinic})")
    epi_data_for_tab = calculate_clinic_epi_data(period_health_data, current_period_str_clinic)
    # Logic to display data from epi_data_for_tab using _web plotters and st.dataframe...
    # (Example: Symptom Trends Bar Chart and Malaria RDT Positivity Line Chart from previous refactor)
    if epi_data_for_tab.get("symptom_trends_weekly_df") is not None and not epi_data_for_tab["symptom_trends_weekly_df"].empty:
        st.plotly_chart(plot_bar_chart_web(epi_data_for_tab["symptom_trends_weekly_df"], x_col_bar='week_start_date', y_col_bar='count', color_col_bar='symptom', title_bar="Weekly Symptom Frequency", barmode_web='group', y_axis_is_count=True), use_container_width=True)
    if epi_data_for_tab.get("malaria_rdt_positivity_weekly_series") is not None and not epi_data_for_tab["malaria_rdt_positivity_weekly_series"].empty:
        st.plotly_chart(plot_annotated_line_chart_web(epi_data_for_tab["malaria_rdt_positivity_weekly_series"], chart_title="Weekly Malaria RDT Positivity", y_axis_label="Positivity %", target_ref_line=app_config.TARGET_MALARIA_POSITIVITY_RATE), use_container_width=True)
    # Add display for demographics and referral funnel if needed from epi_data_for_tab

with tab_clinic_testing:
    st.subheader(f"Testing & Diagnostics Performance ({current_period_str_clinic})")
    # Provide UI to select a test group or default to critical.
    # For this example, default to "All Critical Tests Summary"
    # TODO: Implement selectbox for `selected_test_group_display_name_for_detail` based on available tests
    test_insights = prepare_clinic_testing_insights_data(
        period_health_data, period_summary_kpis, current_period_str_clinic,
        selected_test_group_display_name_for_detail="All Critical Tests Summary"
    )
    if test_insights.get("critical_tests_summary_df") is not None: st.dataframe(test_insights["critical_tests_summary_df"], use_container_width=True)
    if test_insights.get("overdue_pending_tests_df") is not None and not test_insights["overdue_pending_tests_df"].empty:
        st.markdown("###### Overdue Pending Tests:")
        st.dataframe(test_insights["overdue_pending_tests_df"].head(15), use_container_width=True)
    # Add more visualization from `test_insights` like rejection reasons donut chart...

with tab_clinic_supply:
    st.subheader(f"Medical Supply Forecast & Status")
    # Toggle for AI model (could be global or per-tab if resource intensive)
    use_ai_supply_fcst_clinic = st.checkbox("Use Advanced AI Supply Forecast (Simulated)", value=False, key="clinic_console_supply_ai_toggle_v1")
    supply_data = prepare_clinic_supply_forecast_data(
        clinic_historical_health_df=full_hist_health_data, # AI model needs more history potentially for rates
        reporting_period_str=current_period_str_clinic,
        use_ai_forecast_model=use_ai_supply_fcst_clinic,
        items_to_forecast=None # Default to KEY_DRUG_SUBSTRINGS_SUPPLY
    )
    st.markdown(f"**Forecast Model Used:** {supply_data.get('forecast_model_used', 'N/A')}")
    if supply_data.get("forecast_items_summary_list"):
        st.dataframe(pd.DataFrame(supply_data["forecast_items_summary_list"]), use_container_width=True, 
                        column_config={"estimated_stockout_date": st.column_config.DateColumn(format="YYYY-MM-DD")})
    # TODO: Add selectbox for item, then plot its detailed forecast from supply_data["forecast_data_df"] using plot_annotated_line_chart_web

with tab_clinic_patients:
    st.subheader(f"Patient Load & High-Interest Case Review ({current_period_str_clinic})")
    patient_focus_data = prepare_clinic_patient_focus_data(period_health_data, current_period_str_clinic)
    if patient_focus_data.get("patient_load_by_condition_df") is not None and not patient_focus_data["patient_load_by_condition_df"].empty:
        st.markdown("###### Patient Load by Key Condition:")
        st.plotly_chart(plot_bar_chart_web(patient_focus_data["patient_load_by_condition_df"], x_col_bar='period_start_date', y_col_bar='unique_patients_count', color_col_bar='condition', title_bar="Patient Load", barmode_web='stack', y_axis_is_count=True), use_container_width=True)
    if patient_focus_data.get("flagged_patients_for_review_df") is not None and not patient_focus_data["flagged_patients_for_review_df"].empty:
        st.markdown("###### Flagged Patients for Clinical Review (Top 15 by Priority):")
        st.dataframe(patient_focus_data["flagged_patients_for_review_df"].head(15), use_container_width=True) # Add column config for better display

with tab_clinic_env_details:
    st.subheader(f"Facility Environment Detailed Monitoring ({current_period_str_clinic})")
    env_details_output = prepare_clinic_environment_details_data(period_iot_data, iot_available_flag, current_period_str_clinic)
    if env_details_output.get("current_environmental_alerts_summary"):
        st.markdown("###### Current Environmental Alerts (Based on Latest in Period):")
        for alert in env_details_output["current_environmental_alerts_summary"]: st.markdown(f"- **{alert['alert_type']}**: {alert['message']} (Severity: {alert['level']})")
    
    co2_trend_clinic = env_details_output.get("hourly_avg_co2_trend_series")
    if co2_trend_clinic is not None and not co2_trend_clinic.empty: st.plotly_chart(plot_annotated_line_chart_web(co2_trend_clinic, "Hourly Avg. CO2 Levels", y_axis_label="CO2 (ppm)", date_display_format="%H:%M (%d-%b)"), use_container_width=True)
    
    if env_details_output.get("latest_sensor_readings_by_room_df") is not None and not env_details_output["latest_sensor_readings_by_room_df"].empty:
        st.markdown("###### Latest Sensor Readings by Room:")
        st.dataframe(env_details_output["latest_sensor_readings_by_room_df"], use_container_width=True)
    if env_details_output.get("data_availability_notes"):
        for note in env_details_output["data_availability_notes"]: st.caption(note)

logger.info(f"Clinic Operations & Management Console page generated for period: {current_period_str_clinic}")
