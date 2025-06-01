# sentinel_project_root/test/pages/2_clinic_dashboard.py
# Clinic Operations & Management Console for Sentinel Health Co-Pilot.

import streamlit as st
import pandas as pd
import numpy as np # Kept for potential future use
import os
import logging
from datetime import date, timedelta, datetime # Added datetime
from typing import Optional, Dict, Any, Tuple # Added Tuple

# --- Sentinel System Imports ---
from config import app_config
from utils.core_data_processing import (
    load_health_records, load_iot_clinic_environment_data,
    get_clinic_summary # Core function for summary KPIs
)
from utils.ai_analytics_engine import apply_ai_models # For data enrichment simulation
from utils.ui_visualization_helpers import (
    render_web_kpi_card,
    plot_annotated_line_chart_web,
    plot_bar_chart_web,
    # plot_donut_chart_web # Import if used for e.g. test rejection reasons
)
# Clinic Component specific data processors
from .clinic_components_sentinel.environmental_kpi_calculator import calculate_clinic_environmental_kpis
from .clinic_components_sentinel.main_kpi_structurer import structure_main_clinic_kpis_data, structure_disease_specific_kpis_data
from .clinic_components_sentinel.epi_data_calculator import calculate_clinic_epi_data
from .clinic_components_sentinel.environment_detail_preparer import prepare_clinic_environment_details_data
from .clinic_components_sentinel.patient_focus_data_preparer import prepare_clinic_patient_focus_data
from .clinic_components_sentinel.supply_forecast_generator import prepare_clinic_supply_forecast_data
from .clinic_components_sentinel.testing_insights_analyzer import prepare_clinic_testing_insights_data

# --- Page Configuration ---
st.set_page_config(
    page_title=f"Clinic Console - {app_config.APP_NAME}",
    layout="wide",
    initial_sidebar_state="expanded"
)
logger = logging.getLogger(__name__)

# --- Data Loading for Clinic Console ---
@st.cache_data(
    ttl=app_config.CACHE_TTL_SECONDS_WEB_REPORTS,
    show_spinner="Loading comprehensive clinic dataset..."
)
def get_clinic_console_data(
    selected_start_date: date, selected_end_date: date
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Dict[str, Any], bool]:
    """
    Fetches, enriches, and prepares all data for the Clinic Console for the selected period.
    Returns: full_historical_health_enriched, period_health_enriched, period_iot, period_summary_kpis, iot_source_available_flag
    """
    logger.info(f"Clinic Console: Loading data for period {selected_start_date.isoformat()} to {selected_end_date.isoformat()}")
    
    health_raw_df = load_health_records(source_context="ClinicConsole/LoadRawHealth")
    iot_raw_df = load_iot_clinic_environment_data(source_context="ClinicConsole/LoadRawIoT")
    iot_available = not iot_raw_df.empty if isinstance(iot_raw_df, pd.DataFrame) else False

    full_health_enriched_df = pd.DataFrame()
    # Expected columns if health_raw_df is empty but AI model usually adds these
    ai_added_cols = ['ai_risk_score', 'ai_followup_priority_score']
    base_health_cols = health_raw_df.columns.tolist() if isinstance(health_raw_df, pd.DataFrame) and not health_raw_df.empty else []
    
    if isinstance(health_raw_df, pd.DataFrame) and not health_raw_df.empty:
        # Simulate AI enrichment on the full dataset (as if done by a Facility Node ETL)
        enriched_result = apply_ai_models(health_raw_df, source_context="ClinicConsole/AIEnrich")
        full_health_enriched_df = enriched_result[0] # [0] is the health_df
    else:
        logger.warning("Clinic Console: Raw health data empty or invalid. AI enrichment skipped.")
        # Ensure an empty DataFrame with expected schema (base + AI columns)
        full_health_enriched_df = pd.DataFrame(columns=list(set(base_health_cols + ai_added_cols)))

    # Filter enriched health data for the selected period
    period_health_df = pd.DataFrame(columns=full_health_enriched_df.columns) # Init with full schema
    if not full_health_enriched_df.empty and 'encounter_date' in full_health_enriched_df.columns:
        # Ensure encounter_date is datetime for filtering (load_health_records should handle this)
        if not pd.api.types.is_datetime64_any_dtype(full_health_enriched_df['encounter_date']):
            full_health_enriched_df['encounter_date'] = pd.to_datetime(full_health_enriched_df['encounter_date'], errors='coerce')
        
        period_health_df = full_health_enriched_df[
            (full_health_enriched_df['encounter_date'].dt.date >= selected_start_date) &
            (full_health_enriched_df['encounter_date'].dt.date <= selected_end_date)
        ].copy()
    
    # Filter IoT data for the selected period
    period_iot_df = pd.DataFrame() # Init empty
    if isinstance(iot_raw_df, pd.DataFrame) and not iot_raw_df.empty and 'timestamp' in iot_raw_df.columns:
        if not pd.api.types.is_datetime64_any_dtype(iot_raw_df['timestamp']): # Ensure datetime
             iot_raw_df['timestamp'] = pd.to_datetime(iot_raw_df['timestamp'], errors='coerce')
        period_iot_df = iot_raw_df[
            (iot_raw_df['timestamp'].dt.date >= selected_start_date) &
            (iot_raw_df['timestamp'].dt.date <= selected_end_date)
        ].copy()

    # Generate core clinic summary KPIs for the selected period using enriched period data
    summary_kpis_for_period = {}
    if not period_health_df.empty:
        summary_kpis_for_period = get_clinic_summary(period_health_df, source_context="ClinicConsole/PeriodSummary")
    else:
        logger.info("Clinic Console: No health data in selected period to generate summary KPIs.")
        # Initialize with minimal structure if empty to prevent downstream errors
        summary_kpis_for_period = {"test_summary_details": {}} # Critical for KPI structurer
        
    return full_health_enriched_df, period_health_df, period_iot_df, summary_kpis_for_period, iot_available

# --- Page Title & Sidebar Filters ---
st.title(f"🏥 {app_config.APP_NAME} - Clinic Operations & Management Console")
st.markdown(f"**Service Performance, Patient Care Quality, Resource Management, and Facility Environment Monitoring**")
st.divider()

if os.path.exists(app_config.APP_LOGO_SMALL):
    st.sidebar.image(app_config.APP_LOGO_SMALL, width=150)
st.sidebar.header("🗓️ Console Filters")

# Date Range selection
default_days = app_config.WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND
min_hist_date = date.today() - timedelta(days=365) # Max 1 year historical view for console
max_hist_date = date.today()

default_end_date = max_hist_date
default_start_date = default_end_date - timedelta(days=default_days - 1)
if default_start_date < min_hist_date: default_start_date = min_hist_date

selected_start_date_console, selected_end_date_console = st.sidebar.date_input(
    "Select Date Range for Clinic Review:", value=[default_start_date, default_end_date],
    min_value=min_hist_date, max_value=max_hist_date, key="clinic_console_date_range"
)
if selected_start_date_console > selected_end_date_console:
    st.sidebar.error("Console Date Range: Start date must be on or before end date.")
    selected_start_date_console = selected_end_date_console

current_period_display_str = f"{selected_start_date_console.strftime('%d %b %Y')} - {selected_end_date_console.strftime('%d %b %Y')}"

# --- Load Data Based on Selections ---
full_historical_health_data, \
period_health_data_for_tabs, \
period_iot_data_for_tabs, \
period_clinic_summary, \
iot_source_is_available = get_clinic_console_data(selected_start_date_console, selected_end_date_console)

st.info(f"Displaying Clinic Console for period: **{current_period_display_str}**")

# --- Section 1: Top-Level KPIs ---
st.header("🚀 Performance & Environment Snapshot")
# Performance KPIs
if period_clinic_summary and isinstance(period_clinic_summary.get("test_summary_details"), dict):
    main_kpis = structure_main_clinic_kpis_data(period_clinic_summary, current_period_display_str)
    disease_kpis = structure_disease_specific_kpis_data(period_clinic_summary, current_period_display_str)
    
    if main_kpis:
        st.markdown("##### **Overall Service Performance:**")
        cols_main_kpi = st.columns(min(len(main_kpis), 4)) # Max 4 cols
        for i, kpi in enumerate(main_kpis):
            with cols_main_kpi[i % 4]: render_web_kpi_card(**kpi)
    if disease_kpis:
        st.markdown("##### **Key Disease Testing & Supply Indicators:**")
        cols_disease_kpi = st.columns(min(len(disease_kpis), 4))
        for i, kpi in enumerate(disease_kpis):
            with cols_disease_kpi[i % 4]: render_web_kpi_card(**kpi)
else:
    st.warning(f"Core clinic performance KPIs could not be generated for {current_period_display_str}. Check data and processing.")

# Environmental KPIs
st.markdown("##### **Clinic Environment Quick Check:**")
env_kpis = calculate_clinic_environmental_kpis(period_iot_data_for_tabs, current_period_display_str)
if env_kpis and any(pd.notna(v) for k,v in env_kpis.items() if isinstance(v, (int, float)) and ("avg_" in k or "count" in k or "flag" in k) ): # Check for actual data
    cols_env_kpi = st.columns(4)
    with cols_env_kpi[0]: render_web_kpi_card("Avg. CO2", f"{env_kpis.get('avg_co2_ppm_overall', np.nan):.0f}", units="ppm", icon="💨", status_level=env_kpis.get('co2_status_level',"NEUTRAL"), help_text=f"Target < {app_config.ALERT_AMBIENT_CO2_HIGH_PPM}ppm")
    with cols_env_kpi[1]: render_web_kpi_card("Avg. PM2.5", f"{env_kpis.get('avg_pm25_ugm3_overall', np.nan):.1f}", units="µg/m³", icon="🌫️", status_level=env_kpis.get('pm25_status_level',"NEUTRAL"), help_text=f"Target < {app_config.ALERT_AMBIENT_PM25_HIGH_UGM3}µg/m³")
    with cols_env_kpi[2]: render_web_kpi_card("Avg. Waiting Occupancy", f"{env_kpis.get('avg_waiting_room_occupancy_persons', np.nan):.1f}", units="ppl", icon="👨‍👩‍👧‍👦", status_level=env_kpis.get('occupancy_status_level',"NEUTRAL"), help_text=f"Target < {app_config.TARGET_CLINIC_WAITING_ROOM_OCCUPANCY_MAX} persons")
    noise_alert_count = env_kpis.get('noise_rooms_at_high_alert_count',0)
    with cols_env_kpi[3]: render_web_kpi_card("High Noise Alerts", str(noise_alert_count), units="rooms", icon="🔊", status_level="HIGH_CONCERN" if noise_alert_count > 0 else "ACCEPTABLE", help_text=f"Rooms with noise > {app_config.ALERT_AMBIENT_NOISE_HIGH_DBA}dB")
else:
    if iot_source_is_available: st.info("No environmental IoT data available for this specific period to display snapshot KPIs.")
    else: st.caption("Environmental IoT data source generally unavailable for this clinic.")
st.divider()

# --- Tabbed Interface for Detailed Operational Areas ---
st.header("🛠️ Operational Areas Deep Dive")
tab_names = ["📈 Local Epi", "🔬 Testing", "💊 Supply Chain", "🧍 Patient Focus", "🌿 Environment"] # Shortened tab names
tab_epi, tab_testing, tab_supply, tab_patients, tab_env = st.tabs(tab_names)

with tab_epi:
    st.subheader(f"Clinic-Level Epidemiological Intel ({current_period_display_str})")
    if not period_health_data_for_tabs.empty:
        epi_data = calculate_clinic_epi_data(period_health_data_for_tabs, current_period_display_str)
        if epi_data.get("symptom_trends_weekly_top_n_df") is not None and not epi_data["symptom_trends_weekly_top_n_df"].empty:
            st.plotly_chart(plot_bar_chart_web(epi_data["symptom_trends_weekly_top_n_df"], x_col='week_start_date', y_col='count', color_col='symptom', title="Weekly Symptom Frequency", barmode='group', y_is_count=True, x_axis_label="Week Starting", y_axis_label="Symptom Count"), use_container_width=True)
        
        malaria_pos_trend = epi_data.get("key_test_positivity_trends",{}).get(app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get("RDT-Malaria",{}).get("display_name","Malaria RDT"))
        if isinstance(malaria_pos_trend, pd.Series) and not malaria_pos_trend.empty:
            st.plotly_chart(plot_annotated_line_chart_web(malaria_pos_trend, chart_title="Weekly Malaria RDT Positivity", y_axis_label="Positivity %", target_ref_line=app_config.TARGET_MALARIA_POSITIVITY_RATE), use_container_width=True)
        # Add more from epi_data as needed (demographics, referral funnel)
        if epi_data.get("calculation_notes"): st.caption("Epi Calc Notes: " + "; ".join(epi_data["calculation_notes"]))
    else: st.info("No health data in period for epidemiological analysis.")

with tab_testing:
    st.subheader(f"Testing & Diagnostics Performance ({current_period_display_str})")
    # TODO: Implement dynamic test group selection based on available tests in period_clinic_summary["test_summary_details"]
    selected_test_focus = "All Critical Tests Summary" # Default
    
    test_insights_data = prepare_clinic_testing_insights_data(
        period_health_data_for_tabs, period_clinic_summary, current_period_display_str, selected_test_focus
    )
    if test_insights_data.get("all_critical_tests_summary_table_df") is not None and not test_insights_data["all_critical_tests_summary_table_df"].empty:
        st.markdown("###### **Critical Tests Performance Summary:**")
        st.dataframe(test_insights_data["all_critical_tests_summary_table_df"], use_container_width=True, hide_index=True)
    
    overdue_df = test_insights_data.get("overdue_pending_tests_list_df")
    if isinstance(overdue_df, pd.DataFrame) and not overdue_df.empty:
        st.markdown("###### **Overdue Pending Tests (Top 15):**")
        st.dataframe(overdue_df.head(15), use_container_width=True, hide_index=True)
    # Add more from test_insights_data (rejection reasons chart, TAT/Volume trends for specific test if selected)
    if test_insights_data.get("processing_notes"): st.caption("Testing Insights Notes: " + "; ".join(test_insights_data["processing_notes"]))


with tab_supply:
    st.subheader(f"Medical Supply Forecast & Status")
    use_ai_supply = st.checkbox("Use Advanced AI Supply Forecast (Simulated)", value=False, key="clinic_supply_ai_toggle")
    supply_forecast_data = prepare_clinic_supply_forecast_data(
        full_historical_health_data, # AI model might need more history
        current_period_display_str,
        use_ai_forecast_model=use_ai_supply
    )
    st.markdown(f"**Forecast Model Used:** {supply_forecast_data.get('forecast_model_type_used', 'N/A')}")
    forecast_summary_list = supply_forecast_data.get("forecast_items_overview_list", [])
    if forecast_summary_list:
        st.dataframe(pd.DataFrame(forecast_summary_list), use_container_width=True, hide_index=True,
                        column_config={"estimated_stockout_date": st.column_config.DateColumn("Est. Stockout",format="YYYY-MM-DD")})
    # TODO: Add selectbox for item, then plot its detailed forecast from supply_forecast_data["forecast_detail_df"]
    if supply_forecast_data.get("data_processing_notes"): st.caption("Supply Forecast Notes: " + "; ".join(supply_forecast_data["data_processing_notes"]))


with tab_patients:
    st.subheader(f"Patient Load & High-Interest Case Review ({current_period_display_str})")
    if not period_health_data_for_tabs.empty:
        patient_focus_output = prepare_clinic_patient_focus_data(period_health_data_for_tabs, current_period_display_str)
        
        load_df = patient_focus_output.get("patient_load_by_key_condition_df")
        if isinstance(load_df, pd.DataFrame) and not load_df.empty:
            st.markdown("###### **Patient Load by Key Condition (Weekly):**")
            st.plotly_chart(plot_bar_chart_web(load_df, x_col='period_start_date', y_col='unique_patients_count', color_col='condition', title="Patient Load by Condition", barmode='stack', y_is_count=True, x_axis_label="Week Starting", y_axis_label="Unique Patients"), use_container_width=True)
        
        flagged_df = patient_focus_output.get("flagged_patients_for_review_df")
        if isinstance(flagged_df, pd.DataFrame) and not flagged_df.empty:
            st.markdown("###### **Flagged Patients for Clinical Review (Top by Priority):**")
            st.dataframe(flagged_df.head(15), use_container_width=True, hide_index=True) # Add column config for better display if needed
        elif isinstance(flagged_df, pd.DataFrame): # Empty but valid DF
             st.info("No patients flagged for review in this period based on current criteria.")
        if patient_focus_output.get("processing_notes"): st.caption("Patient Focus Notes: " + "; ".join(patient_focus_output["processing_notes"]))
    else: st.info("No health data in period for patient focus analysis.")


with tab_env:
    st.subheader(f"Facility Environment Detailed Monitoring ({current_period_display_str})")
    env_details = prepare_clinic_environment_details_data(period_iot_data_for_tabs, iot_source_is_available, current_period_display_str)
    
    current_env_alerts = env_details.get("current_environmental_alerts_list", [])
    if current_env_alerts:
        st.markdown("###### **Current Environmental Alerts (Based on Latest in Period):**")
        for alert in current_env_alerts:
            if alert.get("level") != "ACCEPTABLE": # Only show non-acceptable status alerts
                 render_web_traffic_light_indicator(message=alert['message'], status_level=alert['level'], details_text=alert.get('alert_type','Environmental Alert'))
            elif len(current_env_alerts) == 1 and alert.get("level") == "ACCEPTABLE": # Show if it's the only one
                 st.success(f"✅ {alert['message']}")


    co2_trend_data = env_details.get("hourly_avg_co2_trend")
    if isinstance(co2_trend_data, pd.Series) and not co2_trend_data.empty:
        st.plotly_chart(plot_annotated_line_chart_web(co2_trend_data, "Hourly Avg. CO2 Levels (Clinic)", y_axis_label="CO2 (ppm)", date_format="%H:%M (%d-%b)"), use_container_width=True)
    
    latest_sensors_df = env_details.get("latest_room_sensor_readings_df")
    if isinstance(latest_sensors_df, pd.DataFrame) and not latest_sensors_df.empty:
        st.markdown("###### **Latest Sensor Readings by Room:**")
        st.dataframe(latest_sensors_df, use_container_width=True, hide_index=True)
    
    if env_details.get("processing_notes"):
        for note in env_details["processing_notes"]: st.caption(note)
    if not iot_source_is_available and not period_iot_data_for_tabs: # If source globally missing
        st.warning("IoT environmental data source not available. Detailed monitoring is not possible.")

logger.info(f"Clinic Operations & Management Console page generated for period: {current_period_display_str}")
