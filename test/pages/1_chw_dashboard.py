# sentinel_project_root/test/pages/1_chw_dashboard.py
# CHW Supervisor Operations View for Sentinel Health Co-Pilot

import streamlit as st # Should be one of the first Streamlit imports
import sys # For path manipulation
import os # For path manipulation

# --- Robust Path Setup for Imports ---
# This ensures that 'config' and 'utils' can be imported correctly
# when Streamlit runs this page script from the 'pages' subdirectory.
# Assumes this file (1_chw_dashboard.py) is in sentinel_project_root/test/pages/
_current_file_directory = os.path.dirname(os.path.abspath(__file__))
# Navigate two levels up to sentinel_project_root/test/ (which contains config/, utils/)
_project_test_root_directory = os.path.abspath(os.path.join(_current_file_directory, os.pardir))

if _project_test_root_directory not in sys.path:
    sys.path.insert(0, _project_test_root_directory)

# --- Sentinel System Imports (Now after path setup) ---
try:
    from config import app_config
    from utils.core_data_processing import load_health_records
    from utils.ui_visualization_helpers import (
        render_web_kpi_card,
        render_web_traffic_light_indicator,
        plot_annotated_line_chart_web
    )
    # CHW Component specific data processors, imported relative to this 'pages' directory
    # This requires __init__.py in 'pages/' and 'chw_components_sentinel/' to treat them as packages
    from .chw_components_sentinel.summary_metrics_calculator import calculate_chw_daily_summary_metrics
    from .chw_components_sentinel.alert_generator import generate_chw_patient_alerts_from_data
    from .chw_components_sentinel.epi_signal_extractor import extract_chw_local_epi_signals
    from .chw_components_sentinel.task_processor import generate_chw_prioritized_tasks
    from .chw_components_sentinel.activity_trend_calculator import calculate_chw_activity_trends
except ImportError as e_import_chw:
    error_msg_chw = (
        f"CRITICAL IMPORT ERROR in 1_chw_dashboard.py: {e_import_chw}. "
        f"Current Python Path: {sys.path}. "
        f"Attempted to add to path: {_project_test_root_directory}. "
        "Ensure all modules are correctly placed and `__init__.py` files exist in 'pages' and component subdirectories (e.g., 'pages/chw_components_sentinel/')."
    )
    print(error_msg_chw, file=sys.stderr) # Print to stderr for server logs
    if 'st' in globals() and hasattr(st, 'error'): # Check if Streamlit is available to show error
        st.error(error_msg_chw)
        st.stop()
    else: # If Streamlit isn't even loaded, re-raise to ensure script halts
        raise ImportError(error_msg_chw) from e_import_chw

import pandas as pd
import geopandas as gpd
import logging
from datetime import date, timedelta
from typing import Optional, Dict, Any, Tuple, List
import inspect # For logging function names, useful for debugging

# --- Page Configuration ---
# This line was indicated by the traceback. The error is likely an import above it if app_config is not found.
st.set_page_config(
    page_title=f"CHW Supervisor View - {app_config.APP_NAME}", # app_config must be available here
    layout="wide",
    initial_sidebar_state="expanded" # Keep sidebar open by default for supervisor
)

logger = logging.getLogger(__name__) # Page-specific logger

# --- Data Loading Logic for Supervisor View (Simulation) ---
@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS_WEB_REPORTS, show_spinner="Loading CHW operational data for supervisor view...")
def get_chw_supervisor_dashboard_data(
    view_date: date,
    trend_start_date: date,
    trend_end_date: date,
    selected_chw_id: Optional[str] = None, 
    selected_zone_id: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]: # daily_df, period_df, pre_calc_kpis
    """
    Simulates fetching and preparing data needed for the CHW Supervisor View.
    """
    # Use inspect to get the current function name for logging context, if desired
    func_name_for_log = inspect.currentframe().f_code.co_name if inspect.currentframe() else "get_chw_supervisor_dashboard_data"
    
    health_df_all_synced = load_health_records(source_context=f"{func_name_for_log}/LoadSimulatedData")
    
    if health_df_all_synced.empty:
        logger.error(f"({func_name_for_log}) Base health records (simulating synced data) failed to load or are empty. Cannot proceed with dashboard data preparation.")
        return pd.DataFrame(), pd.DataFrame(), {}

    # Ensure 'encounter_date' is datetime (load_health_records should handle this, but defensive check)
    if 'encounter_date' not in health_df_all_synced.columns or \
       not pd.api.types.is_datetime64_any_dtype(health_df_all_synced['encounter_date']):
        logger.warning(f"({func_name_for_log}) 'encounter_date' column missing or not datetime in loaded health records. Attempting conversion.")
        health_df_all_synced['encounter_date'] = pd.to_datetime(health_df_all_synced.get('encounter_date'), errors='coerce')
        health_df_all_synced.dropna(subset=['encounter_date'], inplace=True)
        if health_df_all_synced.empty or 'encounter_date' not in health_df_all_synced.columns:
             logger.error(f"({func_name_for_log}) Failed to ensure valid 'encounter_date' column. Returning empty data.")
             return pd.DataFrame(), pd.DataFrame(), {}

    # Filter data for the daily snapshot
    daily_df_sup = health_df_all_synced[health_df_all_synced['encounter_date'].dt.date == view_date].copy()
    if selected_chw_id and 'chw_id' in daily_df_sup.columns:
        daily_df_sup = daily_df_sup[daily_df_sup['chw_id'] == selected_chw_id]
    if selected_zone_id and 'zone_id' in daily_df_sup.columns:
        daily_df_sup = daily_df_sup[daily_df_sup['zone_id'] == selected_zone_id]

    # Filter data for the trend period
    period_df_sup = health_df_all_synced[
        (health_df_all_synced['encounter_date'].dt.date >= trend_start_date) &
        (health_df_all_synced['encounter_date'].dt.date <= trend_end_date)
    ].copy()
    if selected_chw_id and 'chw_id' in period_df_sup.columns:
        period_df_sup = period_df_sup[period_df_sup['chw_id'] == selected_chw_id]
    if selected_zone_id and 'zone_id' in period_df_sup.columns:
        period_df_sup = period_df_sup[period_df_sup['zone_id'] == selected_zone_id]
        
    pre_calculated_daily_kpis = {}
    if selected_chw_id and not daily_df_sup.empty:
        # Example: Extract CHW self-reported fatigue if available in daily data for this CHW
        worker_self_check_df_today = daily_df_sup[
            (daily_df_sup.get('chw_id') == selected_chw_id) & 
            (daily_df_sup.get('encounter_type') == 'WORKER_SELF_CHECK')
        ]
        if not worker_self_check_df_today.empty and 'ai_followup_priority_score' in worker_self_check_df_today.columns:
            # Assuming 'ai_followup_priority_score' from WORKER_SELF_CHECK is used as the fatigue index for simulation
            # Take the max if multiple self-checks on the same day by the CHW
            pre_calculated_daily_kpis['worker_self_fatigue_index_today'] = worker_self_check_df_today['ai_followup_priority_score'].max() 

    logger.info(f"({func_name_for_log}) Data prepared for supervisor view: Daily records - {len(daily_df_sup)}, Period records - {len(period_df_sup)}.")
    return daily_df_sup, period_df_sup, pre_calculated_daily_kpis

# --- Page Title & Introduction ---
st.title("🧑‍🏫 CHW Supervisor Operations View")
st.markdown(f"**Team Performance Monitoring, Alert Triage, and Field Activity Oversight for {app_config.APP_NAME}**")
st.divider()

# --- Sidebar Filters ---
# Load minimal data just for filter options, or use static list if load is too slow / problematic here
# For this refactor, keeping dynamic population based on a light initial load.
_temp_filter_df = load_health_records(source_context="CHWDashboard/SidebarFilterPopulation")

if os.path.exists(app_config.APP_LOGO_SMALL):
    st.sidebar.image(app_config.APP_LOGO_SMALL, width=150) # Sidebar logo size
else:
    st.sidebar.markdown("🌍", unsafe_allow_html=True)
st.sidebar.header("🗓️ View Filters")

chw_id_options_list = ["All CHWs"]
if isinstance(_temp_filter_df, pd.DataFrame) and not _temp_filter_df.empty and 'chw_id' in _temp_filter_df.columns:
    chw_id_options_list.extend(sorted(_temp_filter_df['chw_id'].dropna().unique().tolist()))
else: # Fallback if dynamic options fail
    chw_id_options_list.extend(["CHW001", "CHW002", "CHW003", "CHW004", "CHW005"]) # From sample data

selected_chw_id_from_filter = st.sidebar.selectbox(
    "Filter by CHW ID:", options=chw_id_options_list, key="supervisor_chw_id_selectbox" # Unique key
)
chw_id_to_use_in_query = None if selected_chw_id_from_filter == "All CHWs" else selected_chw_id_from_filter

zone_id_options_list = ["All Zones"]
if isinstance(_temp_filter_df, pd.DataFrame) and not _temp_filter_df.empty and 'zone_id' in _temp_filter_df.columns:
    zone_id_options_list.extend(sorted(_temp_filter_df['zone_id'].dropna().unique().tolist()))
else: # Fallback
    zone_id_options_list.extend(["ZoneA", "ZoneB", "ZoneC", "ZoneD"])

selected_zone_id_from_filter = st.sidebar.selectbox(
    "Filter by Zone:", options=zone_id_options_list, key="supervisor_zone_id_selectbox" # Unique key
)
zone_id_to_use_in_query = None if selected_zone_id_from_filter == "All Zones" else selected_zone_id_from_filter

del _temp_filter_df # Release memory of the temporary DataFrame

# Date selection for "Daily Snapshot"
min_date_for_daily_snapshot = date.today() - timedelta(days=max(90, app_config.WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND * 3))
max_date_for_daily_snapshot = date.today()
selected_daily_view_date_input = st.sidebar.date_input(
    "View Daily Activity For:",
    value=max_date_for_daily_snapshot, 
    min_value=min_date_for_daily_snapshot, 
    max_value=max_date_for_daily_snapshot,
    key="supervisor_daily_snapshot_datepicker" # Unique key
)

# Date range for "Periodic Trends"
default_trend_end_date_val = selected_daily_view_date_input # Align trend end with snapshot date by default
default_trend_start_date_val = default_trend_end_date_val - timedelta(days=app_config.WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND -1)
if default_trend_start_date_val < min_date_for_daily_snapshot : default_trend_start_date_val = min_date_for_daily_snapshot

selected_trend_start_date_input, selected_trend_end_date_input = st.sidebar.date_input(
    "Select Date Range for Periodic Trends:",
    value=[default_trend_start_date_val, default_trend_end_date_val],
    min_value=min_date_for_daily_snapshot, 
    max_value=max_date_for_daily_snapshot, # Should be max_date_for_daily_snapshot
    key="supervisor_periodic_trend_datepicker" # Unique key
)
if selected_trend_start_date_input > selected_trend_end_date_input:
    st.sidebar.error("Trend start date must be on or before the end date.")
    selected_trend_start_date_input = selected_trend_end_date_input # Auto-correct

# --- Load Data Based on Filter Selections ---
daily_df_view_main, period_df_view_main, pre_calculated_kpis_for_view_main = get_chw_supervisor_dashboard_data(
    view_date=selected_daily_view_date_input,
    trend_start_date=selected_trend_start_date_input,
    trend_end_date=selected_trend_end_date_input,
    selected_chw_id=chw_id_to_use_in_query,
    selected_zone_id=zone_id_to_use_in_query
)

# Display context for the selected filters
filter_context_display_parts = [f"Date: **{selected_daily_view_date_input.strftime('%d %b %Y')}**"]
if chw_id_to_use_in_query: filter_context_display_parts.append(f"CHW: **{chw_id_to_use_in_query}**")
if zone_id_to_use_in_query: filter_context_display_parts.append(f"Zone: **{zone_id_to_use_in_query}**")
st.info(f"Displaying data for: {'  |  '.join(filter_context_display_parts)}")


# --- Section 1: Daily Performance Snapshot ---
st.header(f"📊 Daily Performance Snapshot")
if not daily_df_view_main.empty:
    chw_summary_metrics_today = calculate_chw_daily_summary_metrics(
        chw_daily_kpi_input_data=pre_calculated_kpis_for_view_main, 
        chw_daily_encounter_df=daily_df_view_main,
        for_date=selected_daily_view_date_input
    )
    
    cols_summary_kpi_cards = st.columns(4)
    with cols_summary_kpi_cards[0]: 
        render_web_kpi_card("Visits Today", str(chw_summary_metrics_today.get("visits_count", 0)), 
                            icon="👥", help_text="Total unique patients visited by selected CHW(s) today.")
    
    high_prio_followups_count = chw_summary_metrics_today.get("high_ai_prio_followups_count", 0)
    high_prio_status_level = "ACCEPTABLE" if high_prio_followups_count <= 2 else \
                             ("MODERATE_CONCERN" if high_prio_followups_count <= 5 else "HIGH_CONCERN")
    with cols_summary_kpi_cards[1]: 
        render_web_kpi_card("High Prio Follow-ups", str(high_prio_followups_count), 
                            icon="🎯", status_level=high_prio_status_level, 
                            help_text="Patients needing urgent follow-up based on AI priority score.")
    
    critical_spo2_cases_count = chw_summary_metrics_today.get("critical_spo2_cases_identified_count", 0)
    critical_spo2_status_level = "HIGH_CONCERN" if critical_spo2_cases_count > 0 else "ACCEPTABLE"
    with cols_summary_kpi_cards[2]: 
        render_web_kpi_card("Critical SpO2 Cases", str(critical_spo2_cases_count), 
                            icon="💨", status_level=critical_spo2_status_level, 
                            help_text=f"Patients identified with SpO2 < {app_config.ALERT_SPO2_CRITICAL_LOW_PCT}%.")
    
    high_fever_cases_count = chw_summary_metrics_today.get("high_fever_cases_identified_count", 0)
    high_fever_status_level = "HIGH_CONCERN" if high_fever_cases_count > 0 else "ACCEPTABLE"
    with cols_summary_kpi_cards[3]: 
        render_web_kpi_card("High Fever Cases", str(high_fever_cases_count), 
                            icon="🔥", status_level=high_fever_status_level, 
                            help_text=f"Patients identified with temperature ≥ {app_config.ALERT_BODY_TEMP_HIGH_FEVER_C}°C.")
else:
    st.markdown("_No CHW activity data found for the selected filters to display daily performance snapshot._")
st.divider()

# --- Section 2: Key Alerts & Actionable Task Summary ---
st.header("🚦 Key Alerts & Actionable Tasks Overview")

# Generate Alerts from daily activity data
list_patient_alerts_supervisor = generate_chw_patient_alerts_from_data(
    patient_encounter_data_df=daily_df_view_main, 
    for_date=selected_daily_view_date_input,
    chw_zone_context_str=zone_id_to_use_in_query or "All Supervised Zones",
    max_alerts_to_return=8 # Show a limited number of top alerts for supervisor
)
if list_patient_alerts_supervisor:
    st.subheader(f"Priority Patient Alerts (Requiring Supervisor Review/Action):")
    found_any_critical_alert = False
    for alert_data_item in list_patient_alerts_supervisor:
        if alert_data_item.get("alert_level") == "CRITICAL":
            found_any_critical_alert = True
            render_web_traffic_light_indicator(
                message=f"Pt. {alert_data_item.get('patient_id', 'N/A')}: {alert_data_item.get('primary_reason', 'Critical Alert')}",
                status_level="HIGH_RISK", # Map CRITICAL alert_level to HIGH_RISK style for traffic light
                details_text=f"Details: {alert_data_item.get('brief_details','N/A')} | Context: {alert_data_item.get('context_info','N/A')} | Suggested Action: {alert_data_item.get('suggested_action_code','REVIEW_IMMEDIATELY')}"
            )
    if not found_any_critical_alert:
        st.info("No CRITICAL patient alerts identified from field data for this selection. Checking for warnings...")
    
    # Display WARNING alerts if no CRITICAL ones, or always show them after critical ones
    warning_alerts_to_show = [a for a in list_patient_alerts_supervisor if a.get("alert_level") == "WARNING"]
    if warning_alerts_to_show and (not found_any_critical_alert or len(list_patient_alerts_supervisor) > sum(1 for a in list_patient_alerts_supervisor if a.get("alert_level") == "CRITICAL")): # Check if there are warnings to show
        st.markdown("###### Other Notable Warning Alerts:")
        for alert_data_item_warn in warning_alerts_to_show:
            render_web_traffic_light_indicator(
                message=f"Pt. {alert_data_item_warn.get('patient_id', 'N/A')}: {alert_data_item_warn.get('primary_reason', 'Warning')}",
                status_level="MODERATE_RISK", # Map WARNING alert_level to MODERATE_RISK style
                details_text=f"Details: {alert_data_item_warn.get('brief_details','N/A')} | Context: {alert_data_item_warn.get('context_info','N/A')}"
            )
    elif not found_any_critical_alert and not warning_alerts_to_show and list_patient_alerts_supervisor: # Only INFO alerts exist
        st.info("Only informational alerts were generated for this selection.")


elif not daily_df_view_main.empty : # Data exists but no alerts list was generated
    st.info("No specific patient alerts (Critical or Warning) generated from field data for this selection.")
else: # No data to generate alerts from
    st.markdown("_No CHW activity data available to generate alerts for this selection._")


# Generate Tasks from daily activity data
list_generated_tasks_supervisor = generate_chw_prioritized_tasks(
    source_patient_data_df=daily_df_view_main,
    for_date=selected_daily_view_date_input,
    chw_id_context=chw_id_to_use_in_query, 
    zone_context_str=zone_id_to_use_in_query or "All Supervised Zones",
    max_tasks_to_return_for_summary=10
)
if list_generated_tasks_supervisor:
    st.subheader(f"Top Priority Tasks from Today's Activities:")
    df_tasks_to_display_supervisor = pd.DataFrame(list_generated_tasks_supervisor)
    
    cols_for_task_display_supervisor = ['patient_id', 'task_description', 'priority_score', 'due_date', 'status', 'key_patient_context', 'assigned_chw_id', 'alert_source_info']
    actual_cols_in_tasks_df = [col for col in cols_for_task_display_supervisor if col in df_tasks_to_display_supervisor.columns]
    
    st.dataframe(
        df_tasks_to_display_supervisor[actual_cols_in_tasks_df], 
        use_container_width=True, 
        height=min(380, len(df_tasks_to_display_supervisor)*38 + 58), # Dynamic height with max, accounts for header row
        hide_index=True
    )
elif not daily_df_view_main.empty:
    st.info("No new high-priority tasks identified from today's activities based on the current filters.")
else:
    st.markdown("_No CHW activity data available to generate tasks for this selection._")
st.divider()


# --- Section 3: Local Epi Signals from Field ---
st.header("🔬 Local Epi Signals Watch")
if not daily_df_view_main.empty:
    dict_local_epi_signals = extract_chw_local_epi_signals(
        chw_daily_encounter_df=daily_df_view_main,
        pre_calculated_chw_kpis=pre_calculated_kpis_for_view_main, # From main data loading
        for_date=selected_daily_view_date_input,
        chw_zone_context=zone_id_to_use_in_query or "All Supervised Zones"
    )
    
    cols_epi_kpi_cards = st.columns(3)
    with cols_epi_kpi_cards[0]: render_web_kpi_card(title="Symptomatic (Key Cond.)", value=str(dict_local_epi_signals.get("symptomatic_patients_key_conditions_count", 0)), icon="🤒", units="cases today", help_text="Unique patients with key conditions reporting significant symptoms today.")
    with cols_epi_kpi_cards[1]: render_web_kpi_card(title="New Malaria Cases Today", value=str(dict_local_epi_signals.get("newly_identified_malaria_patients_count",0)), icon="🦟", units="cases", help_text="Malaria cases identified from today's CHW encounters.") # Removed "today" from units for brevity
    with cols_epi_kpi_cards[2]: render_web_kpi_card(title="Pending TB Contacts", value=str(dict_local_epi_signals.get("pending_tb_contact_tracing_tasks_count",0)), icon="👥", units="to trace", help_text="Number of TB contacts identified today needing follow-up by CHW team.")
    
    list_detected_symptom_clusters = dict_local_epi_signals.get("detected_symptom_clusters", [])
    if list_detected_symptom_clusters:
        st.markdown("###### Detected Symptom Clusters Today (Requires Supervisor Review):")
        for cluster_item_data_epi in list_detected_symptom_clusters:
            st.warning(f"⚠️ **Pattern: {cluster_item_data_epi.get('symptoms_pattern','Unknown Pattern')}**: {cluster_item_data_epi.get('patient_count','N/A')} cases in {cluster_item_data_epi.get('location_hint', 'area')}")
    elif 'patient_reported_symptoms' in daily_df_view_main.columns: # Check if symptoms data was even available
        st.info("No significant symptom clusters detected from today's CHW data based on current criteria.")
else:
    st.markdown("_No CHW activity data available to derive local epi signals for this selection._")
st.divider()

# --- Section 4: CHW Team Activity Trends (Periodic View) ---
st.header("📈 CHW Team Activity Trends")
trend_period_display_str_chw = f"{selected_trend_start_dt.strftime('%d %b %Y')} - {selected_trend_end_dt.strftime('%d %b %Y')}"
trend_filter_context_display_str = ""
if actual_chw_id_for_filtering: trend_filter_context_display_str += f" for CHW {actual_chw_id_for_filtering}"
if actual_zone_id_for_filtering: trend_filter_context_display_str += f" in Zone {actual_zone_id_for_filtering}"
st.markdown(f"Displaying trends from **{trend_period_display_str_chw}**{trend_filter_context_display_str if trend_filter_context_display_str else ' (All CHWs/Zones in period)'}.")

if not period_df_view_main.empty:
    dict_chw_activity_trends = calculate_chw_activity_trends(
        chw_historical_health_df=period_df_view_main, # This data is already filtered for period and optionally CHW/Zone
        trend_start_date_input=selected_trend_start_dt, 
        trend_end_date_input=selected_trend_end_dt,
        zone_filter=None, # Data is pre-filtered by zone if a supervisor selected one. Pass None to avoid re-filtering.
        time_period_aggregation='D' # Daily trends are suitable for supervisor dashboard
    )
    
    cols_trends_charts_chw = st.columns(2)
    with cols_trends_charts_chw[0]:
        series_patient_visits_trend = dict_chw_activity_trends.get("patient_visits_trend")
        if isinstance(series_patient_visits_trend, pd.Series) and not series_patient_visits_trend.empty:
            st.plotly_chart(plot_annotated_line_chart_web(
                series_patient_visits_trend, chart_title="Daily Patient Visits (Trend)",
                y_axis_label="# Patients Visited", y_is_count=True
            ), use_container_width=True)
        else: st.caption("No patient visit trend data available for this selection.")
        
    with cols_trends_charts_chw[1]:
        series_high_prio_followups_trend = dict_chw_activity_trends.get("high_priority_followups_trend")
        if isinstance(series_high_prio_followups_trend, pd.Series) and not series_high_prio_followups_trend.empty:
            st.plotly_chart(plot_annotated_line_chart_web(
                series_high_prio_followups_trend, chart_title="Daily High Prio. Follow-ups (Trend)",
                y_axis_label="# High Prio Follow-ups", y_is_count=True
            ), use_container_width=True)
        else: st.caption("No high-priority follow-up trend data available for this selection.")
else:
    st.markdown("_No historical data available for the selected filters/period to display activity trends._")

logger.info(f"CHW Supervisor View page loaded/refreshed for Date: {selected_daily_view_date_input}, CHW: {actual_chw_id_for_filtering or 'All'}, Zone: {actual_zone_id_for_filtering or 'All'}.")
