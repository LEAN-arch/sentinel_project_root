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
# Assuming correct PYTHONPATH for Streamlit page execution context
try:
    from config import app_config
    from utils.core_data_processing import (
        load_health_records, load_iot_clinic_environment_data, get_clinic_summary
    )
    from utils.ai_analytics_engine import apply_ai_models
    # Refactored Clinic Component Data Preparation Functions
    from pages.clinic_components_sentinel.environmental_kpi_calculator import calculate_clinic_environmental_kpis
    from pages.clinic_components_sentinel.main_kpi_structurer import structure_main_clinic_kpis_data, structure_disease_specific_kpis_data
    from pages.clinic_components_sentinel.epi_data_calculator import calculate_clinic_epi_data
    from pages.clinic_components_sentinel.environment_detail_preparer import prepare_clinic_environment_details_data
    from pages.clinic_components_sentinel.patient_focus_data_preparer import prepare_clinic_patient_focus_data
    from pages.clinic_components_sentinel.supply_forecast_generator import prepare_clinic_supply_forecast_data
    from pages.clinic_components_sentinel.testing_insights_analyzer import prepare_clinic_testing_insights_data
    # Refactored UI helpers for web reports
    from utils.ui_visualization_helpers import (
        render_web_kpi_card, plot_annotated_line_chart_web,
        plot_bar_chart_web, plot_donut_chart_web
    )
except ImportError as e:
    logging.critical(f"CRITICAL IMPORT ERROR in 2_clinic_dashboard.py: {e}. Ensure PYTHONPATH is correct. Application may not function.")
    st.error(f"Application Critical Error: Could not load necessary modules. Details: {e}")
    # Define stubs to prevent immediate crash for missing modules if in dev/demo mode without full setup
    def render_web_kpi_card(title,value, **kwargs): st.warning(f"KPI Card: {title}-{value} (render_web_kpi_card stub)")
    # Add stubs for other plotting functions as well if needed
    # For app_config, it's critical, so a hard stop is better
    # This is just illustrative error handling during dev if some paths are temporarily broken.
    st.stop()


# --- Page Configuration ---
st.set_page_config(
    page_title=f"Clinic Console - {app_config.APP_NAME}",
    layout="wide",
    initial_sidebar_state="expanded"
)
logger = logging.getLogger(__name__)
# CSS is assumed to be loaded globally by app_home.py

# --- Data Loading for Clinic Console (Simulates Facility Node Data Access) ---
@st.cache_data(
    ttl=app_config.CACHE_TTL_SECONDS_WEB_REPORTS,
    show_spinner="Loading and processing clinic operational dataset..."
)
def get_clinic_console_page_data( # Renamed for page context
    selected_start_date_page: date, 
    selected_end_date_page: date,
    clinic_id_filter_page: Optional[str] = None # Add clinic ID filter if app manages multiple clinics
):
    """
    Simulates fetching, enriching, and preparing all data for the Clinic Management Console.
    """
    module_source_context = "ClinicConsolePageData"
    logger.info(f"({module_source_context}) Loading data for period {selected_start_date_page} to {selected_end_date_page}")
    
    health_df_raw = load_health_records(file_path=app_config.HEALTH_RECORDS_CSV, source_context=f"{module_source_context}/RawHealth")
    iot_df_raw = load_iot_clinic_environment_data(file_path=app_config.IOT_CLINIC_ENVIRONMENT_CSV, source_context=f"{module_source_context}/RawIoT")
    iot_globally_available = not iot_df_raw.empty

    full_hist_health_enriched = pd.DataFrame()
    if not health_df_raw.empty:
        # Assuming apply_ai_models returns (enriched_df, supply_forecast_df_from_ai), we need enriched_df
        enriched_result = apply_ai_models(health_df_raw, source_context=f"{module_source_context}/AIEnrich")
        full_hist_health_enriched = enriched_result[0] if enriched_result and isinstance(enriched_result, tuple) else pd.DataFrame()
    
    # Filter by clinic_id if provided and available (simulating a multi-clinic node)
    if clinic_id_filter_page:
        if 'clinic_id' in full_hist_health_enriched.columns:
            full_hist_health_enriched = full_hist_health_enriched[full_hist_health_enriched['clinic_id'] == clinic_id_filter_page]
        if 'clinic_id' in iot_df_raw.columns:
            iot_df_raw = iot_df_raw[iot_df_raw['clinic_id'] == clinic_id_filter_page]


    period_health_df_page = pd.DataFrame(columns=full_hist_health_enriched.columns)
    if not full_hist_health_enriched.empty and 'encounter_date' in full_hist_health_enriched.columns:
        full_hist_health_enriched['encounter_date'] = pd.to_datetime(full_hist_health_enriched['encounter_date'], errors='coerce')
        period_health_df_page = full_hist_health_enriched[
            (full_hist_health_enriched['encounter_date'].dt.date >= selected_start_date_page) &
            (full_hist_health_enriched['encounter_date'].dt.date <= selected_end_date_page)
        ].copy()
    
    period_iot_df_page = pd.DataFrame()
    if not iot_df_raw.empty and 'timestamp' in iot_df_raw.columns:
        iot_df_raw['timestamp'] = pd.to_datetime(iot_df_raw['timestamp'], errors='coerce')
        period_iot_df_page = iot_df_raw[
            (iot_df_raw['timestamp'].dt.date >= selected_start_date_page) &
            (iot_df_raw['timestamp'].dt.date <= selected_end_date_page)
        ].copy()

    clinic_summary_for_period_page = {}
    if not period_health_df_page.empty:
        clinic_summary_for_period_page = get_clinic_summary(period_health_df_page, source_context=f"{module_source_context}/PeriodSummary")
    else: # Ensure minimal structure for downstream if no health data in period
        clinic_summary_for_period_page = {"test_summary_details": {}, "processing_notes": ["No health data for period."]}
        
    logger.info(f"({module_source_context}) Data loaded: Health(period): {len(period_health_df_page)}, IoT(period): {len(period_iot_df_page)}")
    return full_hist_health_enriched, period_health_df_page, period_iot_df_page, clinic_summary_for_period_page, iot_globally_available

# --- Page Title & Sidebar Filters ---
st.title(f"🏥 {app_config.APP_NAME} - Clinic Operations & Management Console")
st.markdown(f"**Performance Oversight, Patient Care Quality, Resource Management, and Facility Environment.**")
st.markdown("---")

if os.path.exists(app_config.APP_LOGO_SMALL): st.sidebar.image(app_config.APP_LOGO_SMALL, width=160)
st.sidebar.header("🗓️ Console Filters")

# TODO: Add Clinic ID selector if this console manages multiple clinics.
# clinic_ids_available = ["Clinic_A01", "Clinic_B02"] # Get from data
# selected_clinic_id_console = st.sidebar.selectbox("Select Clinic:", clinic_ids_available)
selected_clinic_id_console_mock = None # Placeholder - assumes single clinic context for now

min_date_console = date.today() - timedelta(days=180) # Default 6 months back
max_date_console = date.today()
def_end_console = max_date_console
def_start_console = def_end_console - timedelta(days=app_config.WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND - 1) # Default 30 days
if def_start_console < min_date_console: def_start_console = min_date_console

sel_start_dt_console, sel_end_dt_console = st.sidebar.date_input(
    "Select Date Range:", value=[def_start_console, def_end_console],
    min_value=min_date_console, max_value=max_date_console, key="clinic_page_date_range_v3"
)
if sel_start_dt_console > sel_end_dt_console:
    st.sidebar.error("Console Date Range: Start date must be before end date.")
    sel_start_dt_console = sel_end_dt_console

# --- Load Data Based on Filters ---
current_period_str_page = f"{sel_start_dt_console.strftime('%d %b %Y')} - {sel_end_dt_console.strftime('%d %b %Y')}"
all_hist_health_data_page, health_data_for_period_page, iot_data_for_period_page, \
clinic_summary_kpis_page, iot_globally_present_flag = get_clinic_console_page_data(
    sel_start_dt_console, sel_end_dt_console, selected_clinic_id_console_mock
)
st.info(f"Displaying Clinic Console for period: **{current_period_str_page}**")

if health_data_for_period_page.empty and iot_data_for_period_page.empty:
    st.warning(f"No operational or environmental data found for the selected period: {current_period_str_page}. Some console sections may be empty.")
    # Don't stop, allow tabs to show individual "no data" messages if needed.

# --- Main KPIs Section ---
st.header(f"🚀 Performance & Environment Snapshot")
# Performance KPIs from main_kpi_structurer
if clinic_summary_kpis_page and clinic_summary_kpis_page.get("test_summary_details"): # A key check if summary is meaningful
    main_perf_kpis = structure_main_clinic_kpis_data(clinic_summary_kpis_page, current_period_str_page)
    disease_supply_kpis = structure_disease_specific_kpis_data(clinic_summary_kpis_page, current_period_str_page)
    
    if main_perf_kpis:
        st.markdown("##### **Overall Service Delivery:**")
        cols_main_kpi_clinic = st.columns(min(len(main_perf_kpis), 4))
        for i, kpi_data in enumerate(main_perf_kpis):
            with cols_main_kpi_clinic[i]: render_web_kpi_card(**kpi_data)
    
    if disease_supply_kpis:
        st.markdown("##### **Key Testing & Supply Indicators:**")
        cols_disease_kpi_clinic = st.columns(min(len(disease_supply_kpis), 4))
        for i, kpi_data in enumerate(disease_supply_kpis):
            with cols_disease_kpi_clinic[i]: render_web_kpi_card(**kpi_data)
else:
    st.caption("Core clinic performance summary data (e.g., testing details) unavailable for this period to display top KPIs.")

# Environmental KPIs from environmental_kpi_calculator
st.markdown("##### **Clinic Environment Status:**")
env_kpi_data = calculate_clinic_environmental_kpis(iot_data_for_period_page, current_period_str_page)
if env_kpi_data and any(pd.notna(v) for k, v in env_kpi_data.items() if "avg_" in k or "count" in k or "flag" in k):
    cols_env_kpi_clinic = st.columns(4) # Ensure these keys match what calculate_clinic_environmental_kpis returns
    with cols_env_kpi_clinic[0]: render_web_kpi_card("Avg. CO2", f"{env_kpi_data.get('avg_co2_ppm_overall', 'N/A'):.0f}" if pd.notna(env_kpi_data.get('avg_co2_ppm_overall')) else "N/A", units="ppm", icon="💨", status_level=env_kpi_data.get('co2_status_level',"NEUTRAL"))
    with cols_env_kpi_clinic[1]: render_web_kpi_card("Avg. PM2.5", f"{env_kpi_data.get('avg_pm25_ugm3_overall', 'N/A'):.1f}" if pd.notna(env_kpi_data.get('avg_pm25_ugm3_overall')) else "N/A", units="µg/m³", icon="🌫️", status_level=env_kpi_data.get('pm25_status_level',"NEUTRAL"))
    with cols_env_kpi_clinic[2]: render_web_kpi_card("Avg. Waiting Occupancy", f"{env_kpi_data.get('avg_waiting_room_occupancy_persons', 'N/A'):.1f}" if pd.notna(env_kpi_data.get('avg_waiting_room_occupancy_persons')) else "N/A", units="persons", icon="👨‍👩‍👧‍👦", status_level=env_kpi_data.get('occupancy_status_level',"NEUTRAL"))
    with cols_env_kpi_clinic[3]: render_web_kpi_card("High Noise Alerts (Rooms)", str(env_kpi_data.get('noise_rooms_at_high_alert_count',0)), units="rooms", icon="🔊", status_level="HIGH_CONCERN" if env_kpi_data.get('noise_rooms_at_high_alert_count',0)>0 else "ACCEPTABLE")
else:
    st.caption("Environmental snapshot KPIs unavailable. IoT data might be missing for this period or source unavailable.")
st.markdown("---")

# --- Tabbed Interface for Operational Deep Dives ---
st.header("🛠️ Clinic Operational Areas - Detailed Review")
tab_names_clinic_page = ["📈 Local Epi Intel", "🔬 Testing & Diagnostics", "💊 Supply Management", "🧍 Patient Focus & Review", "🌿 Facility Environment Details"]
tab_epi_c, tab_test_c, tab_supply_c, tab_patient_c, tab_env_c = st.tabs(tab_names_clinic_page)

with tab_epi_c:
    # Calls calculate_clinic_epi_data and then displays its outputs (DFs, Series) using _web plotters
    # ... (Implementation from previous `2_clinic_dashboard.py` refactor for tab_epi content) ...
    st.markdown("*(Content for Clinic Epi Intel Tab: Symptom trends, positivity rates, demographics, referral funnel. Data from `calculate_clinic_epi_data`.)*")
    epi_output = calculate_clinic_epi_data(health_data_for_period_page, current_period_str_page)
    if epi_output.get("symptom_trends_weekly_top_n_df") is not None : st.plotly_chart(plot_bar_chart_web(epi_output["symptom_trends_weekly_top_n_df"], 'week_start_date','count','Weekly Symptom Freq (Top 5)', color_col_bar='symptom', barmode_web='group'), use_container_width=True)
    if epi_output.get("key_test_positivity_trends", {}).get("Malaria RDT") is not None: st.plotly_chart(plot_annotated_line_chart_web(epi_output["key_test_positivity_trends"]["Malaria RDT"], "Weekly Malaria RDT Positivity", y_axis_label="Positivity %"), use_container_width=True)
    # ... etc for other epi_output elements ...

with tab_test_c:
    # Calls prepare_clinic_testing_insights_data and displays its outputs
    # ... (Implementation from previous `2_clinic_dashboard.py` refactor for tab_tests_display content) ...
    st.markdown("*(Content for Testing Insights Tab: Critical tests summary, specific test KPIs/trends, overdue tests, rejection analysis. Data from `prepare_clinic_testing_insights_data`.)*")
    # Requires clinic_summary_kpis_page (has 'test_summary_details')
    # This assumes `selected_test_group_display_name_for_detail` would be set by a widget in a full UI
    testing_data = prepare_clinic_testing_insights_data(health_data_for_period_page, clinic_summary_kpis_page, current_period_str_page, "All Critical Tests Summary")
    if testing_data.get("all_critical_tests_summary_table_df") is not None: st.dataframe(testing_data["all_critical_tests_summary_table_df"])
    # ... etc for other testing_data elements ...


with tab_supply_c:
    # Calls prepare_clinic_supply_forecast_data and displays its outputs
    # ... (Implementation from previous `2_clinic_dashboard.py` refactor for tab_supplies_display content) ...
    st.markdown("*(Content for Supply Management Tab: Item stock summary, forecast chart for selected item. Data from `prepare_clinic_supply_forecast_data`.)*")
    # UI for model selection:
    ai_supply_model_clinic = st.checkbox("Use Advanced AI Forecast (Simulated)", key="clinic_page_supply_ai_toggle")
    supply_output = prepare_clinic_supply_forecast_data(all_hist_health_data_page, current_period_str_page, use_ai_forecast_model=ai_supply_model_clinic)
    if supply_output.get("forecast_items_overview_list"): st.dataframe(pd.DataFrame(supply_output["forecast_items_overview_list"]))
    # ... add selectbox for item and plot forecast_detail_df for that item ...

with tab_patient_c:
    # Calls prepare_clinic_patient_focus_data and displays its outputs
    # ... (Implementation from previous `2_clinic_dashboard.py` refactor for tab_patients_display content) ...
    st.markdown("*(Content for Patient Focus Tab: Patient load charts, flagged patient list. Data from `prepare_clinic_patient_focus_data`.)*")
    patient_focus_output = prepare_clinic_patient_focus_data(health_data_for_period_page, current_period_str_page)
    if patient_focus_output.get("patient_load_by_key_condition_df") is not None: st.plotly_chart(plot_bar_chart_web(patient_focus_output["patient_load_by_key_condition_df"], 'period_start_date','unique_patients_count','Patient Load by Condition', color_col_bar='condition', barmode_web='stack'), use_container_width=True)
    if patient_focus_output.get("flagged_patients_for_review_df") is not None: st.dataframe(patient_focus_output["flagged_patients_for_review_df"].head(20))


with tab_env_c:
    # Calls prepare_clinic_environment_details_data and displays its outputs
    # ... (Implementation from previous `2_clinic_dashboard.py` refactor for tab_environment_display content) ...
    st.markdown("*(Content for Environment Details Tab: Current env alerts, trends (CO2, Occupancy), latest room readings table. Data from `prepare_clinic_environment_details_data`.)*")
    env_details_output_clinic = prepare_clinic_environment_details_data(iot_data_for_period_page, iot_globally_present_flag, current_period_str_page)
    if env_details_output_clinic.get("current_environmental_alerts_list"): 
        st.markdown("###### Latest Environment Alerts:")
        for alert_env in env_details_output_clinic["current_environmental_alerts_list"]: st.markdown(f"- {alert_env['alert_type']}: {alert_env['message']} ({alert_env['level']})")
    # ... plot trends, show latest readings table from env_details_output_clinic ...


logger.info(f"Clinic Operations & Management Console page generated for period: {current_period_str_page}")
