# sentinel_project_root/test/pages/2_clinic_dashboard.py
# Clinic Operations & Management Console for Sentinel Health Co-Pilot.

import streamlit as st # Should be one ofthe first Streamlit imports
import sys # For path manipulation
import os # For path manipulation

# --- Robust Path Setup for Imports ---
_current_file_directory_clinic = os.path.dirname(os.path.abspath(__file__))
_project_test_root_directory_clinic = os.path.abspath(os.path.join(_current_file_directory_clinic, os.pardir))
if _project_test_root_directory_clinic not in sys.path:
    sys.path.insert(0, _project_test_root_directory_clinic)

# --- Sentinel System Imports (Now after path setup) ---
try:
    from config import app_config
    from utils.core_data_processing import (
        load_health_records, load_iot_clinic_environment_data,
        get_clinic_summary 
    )
    from utils.ai_analytics_engine import apply_ai_models
    from utils.ui_visualization_helpers import (
        render_web_kpi_card,
        plot_annotated_line_chart_web,
        plot_bar_chart_web,
        # plot_donut_chart_web # Uncomment if a donut chart (e.g., for rejection reasons) is added
    )
    # Clinic Component specific data processors
    # These relative imports require __init__.py in 'pages/' and 'clinic_components_sentinel/'
    from .clinic_components_sentinel.environmental_kpi_calculator import calculate_clinic_environmental_kpis
    from .clinic_components_sentinel.main_kpi_structurer import structure_main_clinic_kpis_data, structure_disease_specific_kpis_data
    from .clinic_components_sentinel.epi_data_calculator import calculate_clinic_epi_data
    from .clinic_components_sentinel.environment_detail_preparer import prepare_clinic_environment_details_data
    from .clinic_components_sentinel.patient_focus_data_preparer import prepare_clinic_patient_focus_data
    from .clinic_components_sentinel.supply_forecast_generator import prepare_clinic_supply_forecast_data
    from .clinic_components_sentinel.testing_insights_analyzer import prepare_clinic_testing_insights_data
except ImportError as e_import_clinic_dash: # Unique exception variable name
    error_msg_clinic_dash = (
        f"CRITICAL IMPORT ERROR in 2_clinic_dashboard.py: {e_import_clinic_dash}. "
        f"Current Python Path: {sys.path}. "
        f"Attempted to add to path: {_project_test_root_directory_clinic}. "
        "Ensure all modules are correctly placed and `__init__.py` files exist in 'pages' and component subdirectories. "
        "Also, ensure all external libraries (like geopandas, pandas, etc.) are installed in your environment (see requirements.txt)."
    )
    print(error_msg_clinic_dash, file=sys.stderr)
    if 'st' in globals() and hasattr(st, 'error'): # Check if Streamlit is available to show error
        st.error(error_msg_clinic_dash)
        st.stop()
    else: # If Streamlit isn't even loaded, re-raise to ensure script halts
        raise ImportError(error_msg_clinic_dash) from e_import_clinic_dash

import pandas as pd
import logging
from datetime import date, timedelta
from typing import Optional, Dict, Any, Tuple

# --- Page Configuration ---
st.set_page_config(
    page_title=f"Clinic Console - {app_config.APP_NAME}", # app_config must be available
    layout="wide",
    initial_sidebar_state="expanded"
)
logger = logging.getLogger(__name__)

# ... (rest of the file 2_clinic_dashboard.py as refactored in File 59) ...
# The content from "Data Loading for Clinic Console" onwards remains the same
# as the previously provided refactored version (File 59). I will paste it
# for completeness of this specific file output.

# --- Data Loading for Clinic Console ---
@st.cache_data(
    ttl=app_config.CACHE_TTL_SECONDS_WEB_REPORTS,
    show_spinner="Loading comprehensive clinic operational dataset..."
)
def get_clinic_console_data(
    selected_start_date: date, selected_end_date: date
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Dict[str, Any], bool]:
    func_log_prefix = "GetClinicConsoleData" 
    logger.info(f"({func_log_prefix}) Loading data for period: {selected_start_date.isoformat()} to {selected_end_date.isoformat()}")
    
    health_df_raw = load_health_records(source_context=f"{func_log_prefix}/LoadRawHealth")
    iot_df_raw = load_iot_clinic_environment_data(source_context=f"{func_log_prefix}/LoadRawIoT")
    iot_data_source_available = isinstance(iot_df_raw, pd.DataFrame) and not iot_df_raw.empty

    full_health_enriched = pd.DataFrame()
    ai_cols_expected_clinic = ['ai_risk_score', 'ai_followup_priority_score']
    base_cols_health_clinic = health_df_raw.columns.tolist() if isinstance(health_df_raw, pd.DataFrame) and not health_df_raw.empty else []

    if isinstance(health_df_raw, pd.DataFrame) and not health_df_raw.empty:
        enriched_ai_output_clinic = apply_ai_models(health_df_raw.copy(), source_context=f"{func_log_prefix}/AIEnrich")
        full_health_enriched = enriched_ai_output_clinic[0] 
    else:
        logger.warning(f"({func_log_prefix}) Raw health data for clinic is empty or invalid. AI enrichment skipped.")
        full_health_enriched = pd.DataFrame(columns=list(set(base_cols_health_clinic + ai_cols_expected_clinic)))

    df_period_health_clinic = pd.DataFrame(columns=full_health_enriched.columns) 
    if not full_health_enriched.empty and 'encounter_date' in full_health_enriched.columns:
        if not pd.api.types.is_datetime64_any_dtype(full_health_enriched['encounter_date']): 
            full_health_enriched['encounter_date'] = pd.to_datetime(full_health_enriched['encounter_date'], errors='coerce')
        
        df_period_health_clinic = full_health_enriched[
            (full_health_enriched['encounter_date'].dt.date >= selected_start_date) &
            (full_health_enriched['encounter_date'].dt.date <= selected_end_date)
        ].copy()
    
    df_period_iot_clinic = pd.DataFrame() 
    if iot_data_source_available and 'timestamp' in iot_df_raw.columns: 
        if not pd.api.types.is_datetime64_any_dtype(iot_df_raw['timestamp']):
             iot_df_raw['timestamp'] = pd.to_datetime(iot_df_raw['timestamp'], errors='coerce')
        df_period_iot_clinic = iot_df_raw[
            (iot_df_raw['timestamp'].dt.date >= selected_start_date) &
            (iot_df_raw['timestamp'].dt.date <= selected_end_date)
        ].copy()

    dict_period_summary_clinic = {}
    if not df_period_health_clinic.empty:
        dict_period_summary_clinic = get_clinic_summary(df_period_health_clinic, source_context=f"{func_log_prefix}/PeriodSummary")
    else:
        logger.info(f"({func_log_prefix}) No health data in selected period for clinic summary KPIs.")
        dict_period_summary_clinic = {"test_summary_details": {}}
        
    return full_health_enriched, df_period_health_clinic, df_period_iot_clinic, dict_period_summary_clinic, iot_data_source_available

# --- Page Title & Sidebar Filters ---
st.title(f"🏥 {app_config.APP_NAME} - Clinic Operations & Management Console")
st.markdown(f"**Service Performance, Patient Care Quality, Resource Management, and Facility Environment Monitoring**")
st.divider()

if os.path.exists(app_config.APP_LOGO_SMALL):
    st.sidebar.image(app_config.APP_LOGO_SMALL, width=150)
st.sidebar.header("🗓️ Console Filters")

default_days_console = app_config.WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND
min_date_console_picker = date.today() - timedelta(days=365) 
max_date_console_picker = date.today()

default_end_dt_console_val = max_date_console_picker
default_start_dt_console_val = default_end_dt_console_val - timedelta(days=default_days_console - 1)
if default_start_dt_console_val < min_date_console_picker: default_start_dt_console_val = min_date_console_picker

selected_start_date_clinic_console, selected_end_date_clinic_console = st.sidebar.date_input(
    "Select Date Range for Clinic Review:", value=[default_start_dt_console_val, default_end_dt_console_val],
    min_value=min_date_console_picker, max_value=max_date_console_picker, key="clinic_console_date_range_picker_main"
)
if selected_start_date_clinic_console > selected_end_date_clinic_console:
    st.sidebar.error("Console Date Range: Start date must be on or before end date.")
    selected_start_date_clinic_console = selected_end_date_clinic_console

current_period_str_for_display = f"{selected_start_date_clinic_console.strftime('%d %b %Y')} - {selected_end_date_clinic_console.strftime('%d %b %Y')}"

# --- Load Data Based on Selections ---
full_historical_health_df, \
period_health_data_clinic_tabs, \
period_iot_data_clinic_tabs, \
period_clinic_summary_kpis_data, \
iot_source_is_available_flag = get_clinic_console_data(selected_start_date_clinic_console, selected_end_date_clinic_console)

st.info(f"Displaying Clinic Console for period: **{current_period_str_for_display}**")

# --- Section 1: Top-Level KPIs ---
st.header("🚀 Performance & Environment Snapshot")
if period_clinic_summary_kpis_data and isinstance(period_clinic_summary_kpis_data.get("test_summary_details"), dict):
    main_kpi_list_clinic = structure_main_clinic_kpis_data(period_clinic_summary_kpis_data, current_period_str_for_display)
    disease_kpi_list_clinic = structure_disease_specific_kpis_data(period_clinic_summary_kpis_data, current_period_str_for_display)
    
    if main_kpi_list_clinic:
        st.markdown("##### **Overall Service Performance:**")
        cols_main_kpi_cards_clinic = st.columns(min(len(main_kpi_list_clinic), 4)) 
        for i, kpi_data_main in enumerate(main_kpi_list_clinic):
            with cols_main_kpi_cards_clinic[i % 4]: render_web_kpi_card(**kpi_data_main)
    if disease_kpi_list_clinic:
        st.markdown("##### **Key Disease Testing & Supply Indicators:**")
        cols_disease_kpi_cards_clinic = st.columns(min(len(disease_kpi_list_clinic), 4))
        for i, kpi_data_disease in enumerate(disease_kpi_list_clinic):
            with cols_disease_kpi_cards_clinic[i % 4]: render_web_kpi_card(**kpi_data_disease)
else:
    st.warning(f"Core clinic performance KPIs could not be generated for {current_period_str_for_display}. Check data sources and processing.")

st.markdown("##### **Clinic Environment Quick Check:**")
env_kpi_dict_clinic = calculate_clinic_environmental_kpis(period_iot_data_clinic_tabs, current_period_str_for_display)
has_meaningful_env_data_clinic = env_kpi_dict_clinic and any(
    pd.notna(v_env) and (v_env != 0 if "count" in k_env else True) 
    for k_env, v_env in env_kpi_dict_clinic.items() 
    if isinstance(v_env, (int, float)) and ("avg_" in k_env or "count" in k_env or "flag" in k_env)
)
if has_meaningful_env_data_clinic:
    cols_env_kpi_cards_clinic = st.columns(4)
    with cols_env_kpi_cards_clinic[0]: render_web_kpi_card("Avg. CO2", f"{env_kpi_dict_clinic.get('avg_co2_ppm_overall', np.nan):.0f}", units="ppm", icon="💨", status_level=env_kpi_dict_clinic.get('co2_status_level',"NEUTRAL"), help_text=f"Target < {app_config.ALERT_AMBIENT_CO2_HIGH_PPM}ppm")
    with cols_env_kpi_cards_clinic[1]: render_web_kpi_card("Avg. PM2.5", f"{env_kpi_dict_clinic.get('avg_pm25_ugm3_overall', np.nan):.1f}", units="µg/m³", icon="🌫️", status_level=env_kpi_dict_clinic.get('pm25_status_level',"NEUTRAL"), help_text=f"Target < {app_config.ALERT_AMBIENT_PM25_HIGH_UGM3}µg/m³")
    with cols_env_kpi_cards_clinic[2]: render_web_kpi_card("Avg. Waiting Occupancy", f"{env_kpi_dict_clinic.get('avg_waiting_room_occupancy_persons', np.nan):.1f}", units="persons", icon="👨‍👩‍👧‍👦", status_level=env_kpi_dict_clinic.get('occupancy_status_level',"NEUTRAL"), help_text=f"Target < {app_config.TARGET_CLINIC_WAITING_ROOM_OCCUPANCY_MAX} persons")
    noise_alert_count_clinic = env_kpi_dict_clinic.get('noise_rooms_at_high_alert_count',0)
    with cols_env_kpi_cards_clinic[3]: render_web_kpi_card("High Noise Alerts", str(noise_alert_count_clinic), units="rooms", icon="🔊", status_level="HIGH_CONCERN" if noise_alert_count_clinic > 0 else "ACCEPTABLE", help_text=f"Rooms with noise > {app_config.ALERT_AMBIENT_NOISE_HIGH_DBA}dBA")
else:
    if iot_source_is_available_flag: st.info("No significant environmental IoT data for this period to display snapshot KPIs.")
    else: st.caption("Environmental IoT data source is generally unavailable for this clinic. Monitoring limited.")
st.divider()

# --- Tabbed Interface for Detailed Operational Areas ---
st.header("🛠️ Operational Areas Deep Dive")
clinic_console_tab_names_list = ["📈 Local Epi", "🔬 Testing Insights", "💊 Supply Chain", "🧍 Patient Focus", "🌿 Environment Details"]
tab_clinic_epi_view, tab_clinic_testing_view, tab_clinic_supply_view, tab_clinic_patient_view, tab_clinic_env_view = st.tabs(clinic_console_tab_names_list)

# ... (The rest of the tab implementations remain the same as in File 59) ...
# For brevity, I'm not re-pasting the identical tab content.
# It includes calls to:
# calculate_clinic_epi_data, prepare_clinic_testing_insights_data, 
# prepare_clinic_supply_forecast_data, prepare_clinic_patient_focus_data,
# prepare_clinic_environment_details_data
# and renders their outputs using st.dataframe, st.plotly_chart, etc.

with tab_clinic_epi_view:
    st.subheader(f"Local Epidemiological Intelligence ({current_period_str_for_display})")
    if not period_health_data_clinic_tabs.empty:
        epi_data_for_display_tab = calculate_clinic_epi_data(period_health_data_clinic_tabs, current_period_str_for_display)
        
        df_symptom_trends_display = epi_data_for_display_tab.get("symptom_trends_weekly_top_n_df")
        if isinstance(df_symptom_trends_display, pd.DataFrame) and not df_symptom_trends_display.empty:
            st.plotly_chart(plot_bar_chart_web(df_symptom_trends_display, x_col='week_start_date', y_col='count', color_col='symptom', title="Weekly Symptom Frequency (Top Reported)", barmode='group', y_is_count=True, x_axis_label="Week Starting", y_axis_label="Symptom Encounters"), use_container_width=True)
        
        malaria_rdt_config_name = app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get("RDT-Malaria",{}).get("display_name","Malaria RDT")
        series_malaria_positivity_trend = epi_data_for_display_tab.get("key_test_positivity_trends",{}).get(malaria_rdt_config_name)
        if isinstance(series_malaria_positivity_trend, pd.Series) and not series_malaria_positivity_trend.empty:
            st.plotly_chart(plot_annotated_line_chart_web(series_malaria_positivity_trend, chart_title=f"Weekly {malaria_rdt_config_name} Positivity Rate", y_axis_label="Positivity %", target_ref_line=app_config.TARGET_MALARIA_POSITIVITY_RATE), use_container_width=True)
        
        if epi_data_for_display_tab.get("calculation_notes"): 
            for note_epi in epi_data_for_display_tab["calculation_notes"]: st.caption(f"Epi Note: {note_epi}")
    else: st.info("No health data available in the selected period for epidemiological analysis.")

with tab_clinic_testing_view:
    st.subheader(f"Testing & Diagnostics Performance ({current_period_str_for_display})")
    selected_test_group_for_details = "All Critical Tests Summary" 
    
    dict_testing_insights_data = prepare_clinic_testing_insights_data(
        period_health_data_clinic_tabs, period_clinic_summary_kpis_data, current_period_str_for_display, selected_test_group_for_details
    )
    df_critical_tests_summary_table = dict_testing_insights_data.get("all_critical_tests_summary_table_df")
    if isinstance(df_critical_tests_summary_table, pd.DataFrame) and not df_critical_tests_summary_table.empty:
        st.markdown("###### **Critical Tests Performance Summary:**")
        st.dataframe(df_critical_tests_summary_table, use_container_width=True, hide_index=True)
    
    df_overdue_pending_tests = dict_testing_insights_data.get("overdue_pending_tests_list_df")
    if isinstance(df_overdue_pending_tests, pd.DataFrame) and not df_overdue_pending_tests.empty:
        st.markdown("###### **Overdue Pending Tests (Top 15 by Days Pending):**")
        st.dataframe(df_overdue_pending_tests.head(15), use_container_width=True, hide_index=True)
    elif isinstance(df_overdue_pending_tests, pd.DataFrame): 
         st.success("✅ No tests currently flagged as overdue based on defined criteria.")

    if dict_testing_insights_data.get("processing_notes"): 
        for note_test in dict_testing_insights_data["processing_notes"]: st.caption(f"Testing Note: {note_test}")

with tab_clinic_supply_view:
    st.subheader(f"Medical Supply Forecast & Status")
    use_ai_supply_forecast_val = st.checkbox("Use Advanced AI Supply Forecast (Simulated)", value=False, key="clinic_console_supply_ai_toggle_main")
    
    dict_supply_forecast_data = prepare_clinic_supply_forecast_data(
        full_historical_health_df, 
        current_period_str_for_display,
        use_ai_forecast_model=use_ai_supply_forecast_val
    )
    st.markdown(f"**Forecast Model Used:** `{dict_supply_forecast_data.get('forecast_model_type_used', 'N/A')}`")
    list_forecast_items_summary = dict_supply_forecast_data.get("forecast_items_overview_list", [])
    if list_forecast_items_summary:
        st.dataframe(pd.DataFrame(list_forecast_items_summary), use_container_width=True, hide_index=True,
                        column_config={"estimated_stockout_date": st.column_config.DateColumn("Est. Stockout Date",format="YYYY-MM-DD")})
    else:
        st.info("No supply forecast data generated for the selected items or model type.")
    
    if dict_supply_forecast_data.get("data_processing_notes"): 
        for note_supply in dict_supply_forecast_data["data_processing_notes"]: st.caption(f"Supply Note: {note_supply}")

with tab_clinic_patient_view:
    st.subheader(f"Patient Load & High-Interest Case Review ({current_period_str_for_display})")
    if not period_health_data_clinic_tabs.empty:
        dict_patient_focus_data = prepare_clinic_patient_focus_data(period_health_data_clinic_tabs, current_period_str_for_display)
        
        df_patient_load_chart_data = dict_patient_focus_data.get("patient_load_by_key_condition_df")
        if isinstance(df_patient_load_chart_data, pd.DataFrame) and not df_patient_load_chart_data.empty:
            st.markdown("###### **Patient Load by Key Condition (Aggregated Weekly):**") 
            st.plotly_chart(plot_bar_chart_web(
                df_patient_load_chart_data, x_col='period_start_date', y_col='unique_patients_count', 
                color_col='condition', title="Patient Load by Key Condition", barmode='stack', y_is_count=True, 
                x_axis_label="Week Starting", y_axis_label="Unique Patients Seen"
            ), use_container_width=True)
        
        df_flagged_patients_display = dict_patient_focus_data.get("flagged_patients_for_review_df")
        if isinstance(df_flagged_patients_display, pd.DataFrame) and not df_flagged_patients_display.empty:
            st.markdown("###### **Flagged Patients for Clinical Review (Top by Priority):**")
            st.dataframe(df_flagged_patients_display.head(15), use_container_width=True, hide_index=True)
        elif isinstance(df_flagged_patients_display, pd.DataFrame): 
             st.info("No patients currently flagged for clinical review in this period.")
        
        if dict_patient_focus_data.get("processing_notes"): 
            for note_patient in dict_patient_focus_data["processing_notes"]: st.caption(f"Patient Focus Note: {note_patient}")
    else: st.info("No health data in the selected period for patient focus analysis.")

with tab_clinic_env_view:
    st.subheader(f"Facility Environment Detailed Monitoring ({current_period_str_for_display})")
    dict_env_details_display = prepare_clinic_environment_details_data(period_iot_data_clinic_tabs, iot_source_is_available_flag, current_period_str_for_display)
    
    list_current_env_alerts_display = dict_env_details_display.get("current_environmental_alerts_list", [])
    if list_current_env_alerts_display:
        st.markdown("###### **Current Environmental Alerts (from Latest Readings in Period):**")
        found_non_acceptable_env_alert = False
        for alert_item_env_display in list_current_env_alerts_display:
            if alert_item_env_display.get("level") != "ACCEPTABLE": 
                 found_non_acceptable_env_alert = True
                 render_web_traffic_light_indicator(
                     message=alert_item_env_display.get('message','Environmental issue detected.'), 
                     status_level=alert_item_env_display.get('level','UNKNOWN'), 
                     details_text=alert_item_env_display.get('alert_type','Environmental Alert')
                 )
        if not found_non_acceptable_env_alert and len(list_current_env_alerts_display) == 1 and list_current_env_alerts_display[0].get("level") == "ACCEPTABLE":
             st.success(f"✅ {list_current_env_alerts_display[0].get('message', 'Environment appears normal based on latest checks.')}")
        elif not found_non_acceptable_env_alert and len(list_current_env_alerts_display) > 1: 
             st.info("Multiple environmental parameters checked; all appear within acceptable limits.")

    series_co2_trend_clinic_display = dict_env_details_display.get("hourly_avg_co2_trend")
    if isinstance(series_co2_trend_clinic_display, pd.Series) and not series_co2_trend_clinic_display.empty:
        st.plotly_chart(plot_annotated_line_chart_web(series_co2_trend_clinic_display, "Hourly Avg. CO2 Levels (Clinic-wide)", y_axis_label="CO2 (ppm)", date_format="%H:%M (%d-%b)"), use_container_width=True)
    
    df_latest_sensor_readings_display = dict_env_details_display.get("latest_room_sensor_readings_df")
    if isinstance(df_latest_sensor_readings_display, pd.DataFrame) and not df_latest_sensor_readings_display.empty:
        st.markdown("###### **Latest Sensor Readings by Room (End of Period):**")
        st.dataframe(df_latest_sensor_readings_display, use_container_width=True, hide_index=True)
    
    if dict_env_details_display.get("processing_notes"):
        for note_env in dict_env_details_display["processing_notes"]: st.caption(f"Env. Detail Note: {note_env}")
    
    if not iot_source_is_available_flag and (not isinstance(period_iot_data_clinic_tabs, pd.DataFrame) or period_iot_data_clinic_tabs.empty):
        st.warning("IoT environmental data source appears unavailable. Detailed environmental monitoring is not possible.")

logger.info(f"Clinic Operations & Management Console page loaded/refreshed for period: {current_period_str_for_display}")
