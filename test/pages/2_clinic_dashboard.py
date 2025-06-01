# sentinel_project_root/test/pages/2_clinic_dashboard.py
# Clinic Operations & Management Console for Sentinel Health Co-Pilot.

import streamlit as st
import pandas as pd
# import numpy as np # Not directly used after refactor
import os
import logging
from datetime import date, timedelta # datetime not directly needed
from typing import Optional, Dict, Any, Tuple # Added Tuple

# --- Sentinel System Imports ---
try:
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
except ImportError as e:
    st.error(f"Critical import error in Clinic Dashboard: {e}. Ensure all modules are correctly placed and PYTHONPATH is set if running outside Streamlit's standard structure.")
    st.stop()

# --- Page Configuration ---
st.set_page_config(
    page_title=f"Clinic Console - {app_config.APP_NAME}",
    layout="wide",
    initial_sidebar_state="expanded"
)
logger = logging.getLogger(__name__) # Page-specific logger

# --- Data Loading for Clinic Console ---
@st.cache_data(
    ttl=app_config.CACHE_TTL_SECONDS_WEB_REPORTS,
    show_spinner="Loading comprehensive clinic operational dataset..."
)
def get_clinic_console_data( # Renamed from get_clinic_console_comprehensive_data
    selected_start_date: date, selected_end_date: date
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Dict[str, Any], bool]:
    """
    Fetches, enriches (simulated), and prepares all data for the Clinic Console for the selected period.
    Returns: 
        full_historical_health_enriched_df, 
        period_health_enriched_df, 
        period_iot_df, 
        period_clinic_summary_kpis_dict, 
        iot_source_exists_flag
    """
    func_log_prefix = "GetClinicConsoleData" # For logging context
    logger.info(f"({func_log_prefix}) Loading data for period: {selected_start_date.isoformat()} to {selected_end_date.isoformat()}")
    
    health_df_raw = load_health_records(source_context=f"{func_log_prefix}/LoadRawHealth")
    iot_df_raw = load_iot_clinic_environment_data(source_context=f"{func_log_prefix}/LoadRawIoT")
    iot_data_source_available = isinstance(iot_df_raw, pd.DataFrame) and not iot_df_raw.empty

    # Simulate AI enrichment on the full historical dataset
    full_health_enriched = pd.DataFrame()
    ai_cols_expected = ['ai_risk_score', 'ai_followup_priority_score']
    base_cols_health = health_df_raw.columns.tolist() if isinstance(health_df_raw, pd.DataFrame) and not health_df_raw.empty else []

    if isinstance(health_df_raw, pd.DataFrame) and not health_df_raw.empty:
        enriched_ai_output = apply_ai_models(health_df_raw.copy(), source_context=f"{func_log_prefix}/AIEnrich")
        full_health_enriched = enriched_ai_output[0] 
    else:
        logger.warning(f"({func_log_prefix}) Raw health data for clinic is empty or invalid. AI enrichment skipped.")
        full_health_enriched = pd.DataFrame(columns=list(set(base_cols_health + ai_cols_expected))) # Empty with schema

    # Filter enriched health data for the selected period
    df_period_health = pd.DataFrame(columns=full_health_enriched.columns) # Initialize with schema
    if not full_health_enriched.empty and 'encounter_date' in full_health_enriched.columns:
        # Ensure encounter_date is datetime (loader should do this, but good to be sure)
        if not pd.api.types.is_datetime64_any_dtype(full_health_enriched['encounter_date']):
            full_health_enriched['encounter_date'] = pd.to_datetime(full_health_enriched['encounter_date'], errors='coerce')
        
        df_period_health = full_health_enriched[
            (full_health_enriched['encounter_date'].dt.date >= selected_start_date) &
            (full_health_enriched['encounter_date'].dt.date <= selected_end_date)
        ].copy()
    
    # Filter IoT data for the selected period
    df_period_iot = pd.DataFrame() # Initialize empty
    if iot_data_source_available and 'timestamp' in iot_df_raw.columns: # iot_df_raw is already checked for DF type
        if not pd.api.types.is_datetime64_any_dtype(iot_df_raw['timestamp']):
             iot_df_raw['timestamp'] = pd.to_datetime(iot_df_raw['timestamp'], errors='coerce')
        df_period_iot = iot_df_raw[
            (iot_df_raw['timestamp'].dt.date >= selected_start_date) &
            (iot_df_raw['timestamp'].dt.date <= selected_end_date)
        ].copy()

    # Generate core clinic summary KPIs for the selected period using the enriched period data
    dict_period_clinic_summary = {}
    if not df_period_health.empty:
        dict_period_clinic_summary = get_clinic_summary(df_period_health, source_context=f"{func_log_prefix}/PeriodSummary")
    else:
        logger.info(f"({func_log_prefix}) No health data in selected period to generate clinic summary KPIs.")
        dict_period_clinic_summary = {"test_summary_details": {}} # Minimal structure for KPI structurers
        
    return full_health_enriched, df_period_health, df_period_iot, dict_period_clinic_summary, iot_data_source_available

# --- Page Title & Sidebar Filters ---
st.title(f"🏥 {app_config.APP_NAME} - Clinic Operations & Management Console")
st.markdown(f"**Service Performance, Patient Care Quality, Resource Management, and Facility Environment Monitoring**")
st.divider()

if os.path.exists(app_config.APP_LOGO_SMALL):
    st.sidebar.image(app_config.APP_LOGO_SMALL, width=150)
st.sidebar.header("🗓️ Console Filters")

# Date Range selection for Clinic Console
default_days_range_clinic = app_config.WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND # e.g., 30 days
min_date_clinic_console = date.today() - timedelta(days=365) # Max 1 year historical view
max_date_clinic_console = date.today()

default_end_date_clinic_val = max_date_clinic_console
default_start_date_clinic_val = default_end_date_clinic_val - timedelta(days=default_days_range_clinic - 1)
if default_start_date_clinic_val < min_date_clinic_console: default_start_date_clinic_val = min_date_clinic_console

selected_start_dt_console, selected_end_dt_console = st.sidebar.date_input(
    "Select Date Range for Clinic Review:", value=[default_start_date_clinic_val, default_end_date_clinic_val],
    min_value=min_date_clinic_console, max_value=max_date_clinic_console, key="clinic_console_date_range_picker"
)
if selected_start_dt_console > selected_end_dt_console:
    st.sidebar.error("Console Date Range: Start date must be on or before end date.")
    selected_start_dt_console = selected_end_dt_console # Auto-correct

current_period_display_str_clinic = f"{selected_start_dt_console.strftime('%d %b %Y')} - {selected_end_dt_console.strftime('%d %b %Y')}"

# --- Load Data Based on Selections ---
full_historical_health_df, \
period_health_df_for_tabs, \
period_iot_df_for_tabs, \
period_clinic_summary_kpis, \
iot_source_available_flag = get_clinic_console_data(selected_start_dt_console, selected_end_dt_console)

st.info(f"Displaying Clinic Console for period: **{current_period_display_str_clinic}**")


# --- Section 1: Top-Level KPIs (Overall Performance & Environment) ---
st.header("🚀 Performance & Environment Snapshot")
# Performance KPIs
if period_clinic_summary_kpis and isinstance(period_clinic_summary_kpis.get("test_summary_details"), dict):
    main_kpi_list_structured = structure_main_clinic_kpis_data(period_clinic_summary_kpis, current_period_display_str_clinic)
    disease_kpi_list_structured = structure_disease_specific_kpis_data(period_clinic_summary_kpis, current_period_display_str_clinic)
    
    if main_kpi_list_structured:
        st.markdown("##### **Overall Service Performance:**")
        cols_main_kpi_cards = st.columns(min(len(main_kpi_list_structured), 4)) # Max 4 cols for readability
        for i, kpi_item_main in enumerate(main_kpi_list_structured):
            with cols_main_kpi_cards[i % 4]: render_web_kpi_card(**kpi_item_main)
    if disease_kpi_list_structured:
        st.markdown("##### **Key Disease Testing & Supply Indicators:**")
        cols_disease_kpi_cards = st.columns(min(len(disease_kpi_list_structured), 4))
        for i, kpi_item_disease in enumerate(disease_kpi_list_structured):
            with cols_disease_kpi_cards[i % 4]: render_web_kpi_card(**kpi_item_disease)
else:
    st.warning(f"Core clinic performance KPIs could not be generated for {current_period_display_str_clinic}. Check data sources and processing pipeline.")

# Environmental KPIs
st.markdown("##### **Clinic Environment Quick Check:**")
env_kpi_dict_calculated = calculate_clinic_environmental_kpis(period_iot_df_for_tabs, current_period_display_str_clinic)
# Check if there's meaningful data in the calculated KPIs (not all NaN/0 for key metrics)
has_meaningful_env_data = env_kpi_dict_calculated and any(
    pd.notna(v) and (v != 0 if "count" in k else True) 
    for k, v in env_kpi_dict_calculated.items() 
    if isinstance(v, (int, float)) and ("avg_" in k or "count" in k or "flag" in k)
)

if has_meaningful_env_data:
    cols_env_kpi_cards = st.columns(4)
    with cols_env_kpi_cards[0]: render_web_kpi_card("Avg. CO2", f"{env_kpi_dict_calculated.get('avg_co2_ppm_overall', np.nan):.0f}", units="ppm", icon="💨", status_level=env_kpi_dict_calculated.get('co2_status_level',"NEUTRAL"), help_text=f"Target < {app_config.ALERT_AMBIENT_CO2_HIGH_PPM}ppm")
    with cols_env_kpi_cards[1]: render_web_kpi_card("Avg. PM2.5", f"{env_kpi_dict_calculated.get('avg_pm25_ugm3_overall', np.nan):.1f}", units="µg/m³", icon="🌫️", status_level=env_kpi_dict_calculated.get('pm25_status_level',"NEUTRAL"), help_text=f"Target < {app_config.ALERT_AMBIENT_PM25_HIGH_UGM3}µg/m³")
    with cols_env_kpi_cards[2]: render_web_kpi_card("Avg. Waiting Occupancy", f"{env_kpi_dict_calculated.get('avg_waiting_room_occupancy_persons', np.nan):.1f}", units="persons", icon="👨‍👩‍👧‍👦", status_level=env_kpi_dict_calculated.get('occupancy_status_level',"NEUTRAL"), help_text=f"Target < {app_config.TARGET_CLINIC_WAITING_ROOM_OCCUPANCY_MAX} persons")
    noise_alert_count_val = env_kpi_dict_calculated.get('noise_rooms_at_high_alert_count',0)
    with cols_env_kpi_cards[3]: render_web_kpi_card("High Noise Alerts", str(noise_alert_count_val), units="rooms", icon="🔊", status_level="HIGH_CONCERN" if noise_alert_count_val > 0 else "ACCEPTABLE", help_text=f"Rooms with noise > {app_config.ALERT_AMBIENT_NOISE_HIGH_DBA}dBA")
else:
    if iot_source_available_flag: st.info("No significant environmental IoT data available for this specific period to display snapshot KPIs.")
    else: st.caption("Environmental IoT data source appears generally unavailable for this clinic. Monitoring is limited.")
st.divider()

# --- Tabbed Interface for Detailed Operational Areas ---
st.header("🛠️ Operational Areas Deep Dive")
# Using shorter, more direct tab names
clinic_tab_titles = ["📈 Epi Intel", "🔬 Testing Insights", "💊 Supply Chain", "🧍 Patient Focus", "🌿 Environment Details"]
tab_clinic_epi_intel, tab_clinic_testing_insights, tab_clinic_supply_chain, tab_clinic_patient_focus, tab_clinic_env_details = st.tabs(clinic_tab_titles)

with tab_clinic_epi_intel:
    st.subheader(f"Local Epidemiological Intelligence ({current_period_display_str_clinic})")
    if not period_health_df_for_tabs.empty:
        epi_data_for_tab_display = calculate_clinic_epi_data(period_health_df_for_tabs, current_period_display_str_clinic)
        
        symptom_trends_df = epi_data_for_tab_display.get("symptom_trends_weekly_top_n_df")
        if isinstance(symptom_trends_df, pd.DataFrame) and not symptom_trends_df.empty:
            st.plotly_chart(plot_bar_chart_web(symptom_trends_df, x_col='week_start_date', y_col='count', color_col='symptom', title="Weekly Symptom Frequency (Top Reported)", barmode='group', y_is_count=True, x_axis_label="Week Starting", y_axis_label="Symptom Count"), use_container_width=True)
        
        # Example: Malaria RDT Positivity Trend
        malaria_rdt_display_name_cfg = app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get("RDT-Malaria",{}).get("display_name","Malaria RDT")
        malaria_positivity_trend_series = epi_data_for_tab_display.get("key_test_positivity_trends",{}).get(malaria_rdt_display_name_cfg)
        if isinstance(malaria_positivity_trend_series, pd.Series) and not malaria_positivity_trend_series.empty:
            st.plotly_chart(plot_annotated_line_chart_web(malaria_positivity_trend_series, chart_title=f"Weekly {malaria_rdt_display_name_cfg} Positivity Rate", y_axis_label="Positivity %", target_ref_line=app_config.TARGET_MALARIA_POSITIVITY_RATE), use_container_width=True)
        
        if epi_data_for_tab_display.get("calculation_notes"): 
            for note in epi_data_for_tab_display["calculation_notes"]: st.caption(f"Epi Note: {note}")
    else: st.info("No health data in the selected period for epidemiological analysis.")

with tab_clinic_testing_insights:
    st.subheader(f"Testing & Diagnostics Performance ({current_period_display_str_clinic})")
    # TODO: Implement dynamic test group selection from period_clinic_summary_kpis["test_summary_details"] keys
    selected_test_group_for_details = "All Critical Tests Summary" # Default view
    
    testing_insights_output = prepare_clinic_testing_insights_data(
        period_health_df_for_tabs, period_clinic_summary_kpis, current_period_display_str_clinic, selected_test_group_for_details
    )
    df_critical_tests_summary = testing_insights_output.get("all_critical_tests_summary_table_df")
    if isinstance(df_critical_tests_summary, pd.DataFrame) and not df_critical_tests_summary.empty:
        st.markdown("###### **Critical Tests Performance Summary:**")
        st.dataframe(df_critical_tests_summary, use_container_width=True, hide_index=True)
    
    df_overdue_tests = testing_insights_output.get("overdue_pending_tests_list_df")
    if isinstance(df_overdue_tests, pd.DataFrame) and not df_overdue_tests.empty:
        st.markdown("###### **Overdue Pending Tests (Top 15 by Days Pending):**")
        st.dataframe(df_overdue_tests.head(15), use_container_width=True, hide_index=True)
    elif isinstance(df_overdue_tests, pd.DataFrame): # Empty but valid DataFrame means no overdue tests
        st.success("✅ No tests currently flagged as overdue based on criteria.")

    # Add more from testing_insights_output (e.g., rejection reasons chart, TAT/Volume trends for specific test if selected)
    if testing_insights_output.get("processing_notes"): 
        for note in testing_insights_output["processing_notes"]: st.caption(f"Testing Note: {note}")


with tab_clinic_supply_chain:
    st.subheader(f"Medical Supply Forecast & Status")
    use_ai_supply_forecast_clinic_val = st.checkbox("Use Advanced AI Supply Forecast (Simulated)", value=False, key="clinic_supply_ai_toggle_key")
    
    supply_forecast_output_data = prepare_clinic_supply_forecast_data(
        full_historical_health_df, # AI model might need longer history for rate calculation
        current_period_display_str_clinic,
        use_ai_forecast_model=use_ai_supply_forecast_clinic_val
    )
    st.markdown(f"**Forecast Model Used:** `{supply_forecast_output_data.get('forecast_model_type_used', 'N/A')}`")
    list_forecast_summary_items = supply_forecast_output_data.get("forecast_items_overview_list", [])
    if list_forecast_summary_items:
        st.dataframe(pd.DataFrame(list_forecast_summary_items), use_container_width=True, hide_index=True,
                        column_config={"estimated_stockout_date": st.column_config.DateColumn("Est. Stockout",format="YYYY-MM-DD")})
    else:
        st.info("No supply forecast data generated for the selected items or model.")
    # TODO: Add selectbox for specific item, then plot its detailed forecast from supply_forecast_output_data["forecast_detail_df"]
    if supply_forecast_output_data.get("data_processing_notes"): 
        for note in supply_forecast_output_data["data_processing_notes"]: st.caption(f"Supply Note: {note}")


with tab_clinic_patient_focus:
    st.subheader(f"Patient Load & High-Interest Case Review ({current_period_display_str_clinic})")
    if not period_health_df_for_tabs.empty:
        patient_focus_output_data = prepare_clinic_patient_focus_data(period_health_df_for_tabs, current_period_display_str_clinic)
        
        df_patient_load_by_condition = patient_focus_output_data.get("patient_load_by_key_condition_df")
        if isinstance(df_patient_load_by_condition, pd.DataFrame) and not df_patient_load_by_condition.empty:
            st.markdown("###### **Patient Load by Key Condition (Weekly):**") # Assuming weekly aggregation from preparer
            st.plotly_chart(plot_bar_chart_web(
                df_patient_load_by_condition, x_col='period_start_date', y_col='unique_patients_count', 
                color_col='condition', title="Patient Load by Condition", barmode='stack', y_is_count=True, 
                x_axis_label="Week Starting", y_axis_label="Unique Patients"
            ), use_container_width=True)
        
        df_flagged_patients_for_review = patient_focus_output_data.get("flagged_patients_for_review_df")
        if isinstance(df_flagged_patients_for_review, pd.DataFrame) and not df_flagged_patients_for_review.empty:
            st.markdown("###### **Flagged Patients for Clinical Review (Top by Priority):**")
            st.dataframe(df_flagged_patients_for_review.head(15), use_container_width=True, hide_index=True) # Add column_config for better display if needed
        elif isinstance(df_flagged_patients_for_review, pd.DataFrame): # Empty but valid DF returned
             st.info("No patients flagged for clinical review in this period based on current criteria.")
        
        if patient_focus_output_data.get("processing_notes"): 
            for note in patient_focus_output_data["processing_notes"]: st.caption(f"Patient Focus Note: {note}")
    else: st.info("No health data in the selected period for patient focus analysis.")


with tab_clinic_env_details:
    st.subheader(f"Facility Environment Detailed Monitoring ({current_period_display_str_clinic})")
    env_details_data_output = prepare_clinic_environment_details_data(period_iot_df_for_tabs, iot_source_available_flag, current_period_display_str_clinic)
    
    list_current_env_alerts = env_details_data_output.get("current_environmental_alerts_list", [])
    if list_current_env_alerts:
        st.markdown("###### **Current Environmental Alerts (from Latest Readings in Period):**")
        non_acceptable_alerts_found = False
        for alert_item_env in list_current_env_alerts:
            if alert_item_env.get("level") != "ACCEPTABLE":
                 non_acceptable_alerts_found = True
                 render_web_traffic_light_indicator(
                     message=alert_item_env.get('message','Environmental issue detected.'), 
                     status_level=alert_item_env.get('level','UNKNOWN'), 
                     details_text=alert_item_env.get('alert_type','Environmental Alert')
                 )
        if not non_acceptable_alerts_found and len(list_current_env_alerts) == 1 and list_current_env_alerts[0].get("level") == "ACCEPTABLE":
             st.success(f"✅ {list_current_env_alerts[0].get('message', 'Environment appears normal.')}")
        elif not non_acceptable_alerts_found and len(list_current_env_alerts) > 1: # Multiple notes, but all acceptable
             st.info("Multiple environmental checks normal.")


    series_co2_trend_clinic = env_details_data_output.get("hourly_avg_co2_trend")
    if isinstance(series_co2_trend_clinic, pd.Series) and not series_co2_trend_clinic.empty:
        st.plotly_chart(plot_annotated_line_chart_web(series_co2_trend_clinic, "Hourly Avg. CO2 Levels (Clinic-wide)", y_axis_label="CO2 (ppm)", date_format="%H:%M (%d-%b)"), use_container_width=True)
    
    df_latest_sensor_readings = env_details_data_output.get("latest_room_sensor_readings_df")
    if isinstance(df_latest_sensor_readings, pd.DataFrame) and not df_latest_sensor_readings.empty:
        st.markdown("###### **Latest Sensor Readings by Room (End of Period):**")
        st.dataframe(df_latest_sensor_readings, use_container_width=True, hide_index=True)
    
    if env_details_data_output.get("processing_notes"):
        for note in env_details_data_output["processing_notes"]: st.caption(f"Env. Detail Note: {note}")
    
    if not iot_source_available_flag and (not isinstance(period_iot_df_for_tabs, pd.DataFrame) or period_iot_df_for_tabs.empty): # If source generally missing AND no data for period
        st.warning("IoT environmental data source appears unavailable. Detailed environmental monitoring is not possible.")

logger.info(f"Clinic Operations & Management Console page loaded/refreshed for period: {current_period_display_str_clinic}")
