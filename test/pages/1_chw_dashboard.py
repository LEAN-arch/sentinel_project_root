# sentinel_project_root/test/pages/1_chw_dashboard.py
# CHW Supervisor Operations View for Sentinel Health Co-Pilot

import streamlit as st
import sys 
import os 

# --- Robust Path Setup for Imports from 'pages' subdirectory ---
# This script is in 'test/pages/'. We want to add 'test/' to sys.path.
_current_page_file_directory = os.path.dirname(os.path.abspath(__file__)) # .../test/pages
_app_root_directory_for_pages = os.path.abspath(os.path.join(_current_page_file_directory, os.pardir)) # .../test
if _app_root_directory_for_pages not in sys.path:
    sys.path.insert(0, _app_root_directory_for_pages)

# --- Sentinel System Imports (Now after path setup) ---
try:
    from config import app_config
    from utils.core_data_processing import load_health_records # Target of NameError
    from utils.ui_visualization_helpers import (
        render_web_kpi_card,
        render_web_traffic_light_indicator,
        plot_annotated_line_chart_web
    )
    # CHW Component specific data processors, imported relative to this 'pages' directory
    from .chw_components_sentinel.summary_metrics_calculator import calculate_chw_daily_summary_metrics
    from .chw_components_sentinel.alert_generator import generate_chw_patient_alerts_from_data
    from .chw_components_sentinel.epi_signal_extractor import extract_chw_local_epi_signals
    from .chw_components_sentinel.task_processor import generate_chw_prioritized_tasks
    from .chw_components_sentinel.activity_trend_calculator import calculate_chw_activity_trends
except ImportError as e_import_chw_dash: # Unique exception name
    error_msg_chw_dash = (
        f"CRITICAL IMPORT ERROR in 1_chw_dashboard.py: {e_import_chw_dash}. "
        f"Current Python Path: {sys.path}. "
        f"Attempted to add to path: {_app_root_directory_for_pages}. "
        "Ensure all modules are correctly placed and `__init__.py` files exist in 'pages' and component subdirectories. "
        "Also, ensure all external libraries (like geopandas, pandas, etc.) are installed in your environment."
    )
    print(error_msg_chw_dash, file=sys.stderr)
    if 'st' in globals() and hasattr(st, 'error'):
        st.error(error_msg_chw_dash)
        st.stop()
    else:
        raise ImportError(error_msg_chw_dash) from e_import_chw_dash

import pandas as pd
import logging
from datetime import date, timedelta
from typing import Optional, Dict, Any, Tuple, List
import inspect 

# --- Page Configuration ---
st.set_page_config(
    page_title=f"CHW Supervisor View - {app_config.APP_NAME}", 
    layout="wide",
    initial_sidebar_state="expanded"
)

logger = logging.getLogger(__name__)

# ... (rest of the file 1_chw_dashboard.py as refactored in File 67 / previous response) ...
# The content from "Data Loading Logic for Supervisor View" onwards remains the same.
# I will paste it for completeness.

@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS_WEB_REPORTS, show_spinner="Loading CHW operational data for supervisor view...")
def get_chw_supervisor_dashboard_data(
    view_date: date,
    trend_start_date: date,
    trend_end_date: date,
    selected_chw_id: Optional[str] = None, 
    selected_zone_id: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    func_name_for_log = inspect.currentframe().f_code.co_name if inspect.currentframe() else "get_chw_supervisor_dashboard_data"
    health_df_all_synced = load_health_records(source_context=f"{func_name_for_log}/LoadSimulatedData")
    if health_df_all_synced.empty:
        logger.error(f"({func_name_for_log}) Base health records failed to load or are empty.")
        return pd.DataFrame(), pd.DataFrame(), {}
    if 'encounter_date' not in health_df_all_synced.columns or \
       not pd.api.types.is_datetime64_any_dtype(health_df_all_synced['encounter_date']):
        logger.warning(f"({func_name_for_log}) 'encounter_date' column issue. Attempting conversion.")
        health_df_all_synced['encounter_date'] = pd.to_datetime(health_df_all_synced.get('encounter_date'), errors='coerce')
        health_df_all_synced.dropna(subset=['encounter_date'], inplace=True)
        if health_df_all_synced.empty or 'encounter_date' not in health_df_all_synced.columns:
             logger.error(f"({func_name_for_log}) Failed to ensure valid 'encounter_date'.")
             return pd.DataFrame(), pd.DataFrame(), {}
    daily_df_sup = health_df_all_synced[health_df_all_synced['encounter_date'].dt.date == view_date].copy()
    if selected_chw_id and 'chw_id' in daily_df_sup.columns:
        daily_df_sup = daily_df_sup[daily_df_sup['chw_id'] == selected_chw_id]
    if selected_zone_id and 'zone_id' in daily_df_sup.columns:
        daily_df_sup = daily_df_sup[daily_df_sup['zone_id'] == selected_zone_id]
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
        worker_self_check_df_today = daily_df_sup[
            (daily_df_sup.get('chw_id') == selected_chw_id) & 
            (daily_df_sup.get('encounter_type') == 'WORKER_SELF_CHECK')
        ]
        if not worker_self_check_df_today.empty and 'ai_followup_priority_score' in worker_self_check_df_today.columns:
            pre_calculated_daily_kpis['worker_self_fatigue_index_today'] = worker_self_check_df_today['ai_followup_priority_score'].max() 
    logger.info(f"({func_name_for_log}) Data prepared: Daily - {len(daily_df_sup)} recs, Period - {len(period_df_sup)} recs.")
    return daily_df_sup, period_df_sup, pre_calculated_daily_kpis

st.title("🧑‍🏫 CHW Supervisor Operations View")
st.markdown(f"**Team Performance Monitoring, Alert Triage, and Field Activity Oversight for {app_config.APP_NAME}**")
st.divider()

_temp_filter_df_chw_dash = load_health_records(source_context="CHWDashboard/SidebarFilterPopulation") # Corrected variable name
if os.path.exists(app_config.APP_LOGO_SMALL):
    st.sidebar.image(app_config.APP_LOGO_SMALL, width=150)
else:
    st.sidebar.markdown("🌍", unsafe_allow_html=True)
st.sidebar.header("🗓️ View Filters")
chw_ids_options_list_chw = ["All CHWs"]
if isinstance(_temp_filter_df_chw_dash, pd.DataFrame) and not _temp_filter_df_chw_dash.empty and 'chw_id' in _temp_filter_df_chw_dash.columns:
    chw_ids_options_list_chw.extend(sorted(_temp_filter_df_chw_dash['chw_id'].dropna().unique().tolist()))
else: 
    chw_ids_options_list_chw.extend(["CHW001", "CHW002", "CHW003", "CHW004", "CHW005"]) 
selected_chw_id_from_filter_chw = st.sidebar.selectbox(
    "Filter by CHW ID:", options=chw_ids_options_list_chw, key="supervisor_chw_id_selectbox_page1"
)
actual_chw_id_for_filtering_chw = None if selected_chw_id_from_filter_chw == "All CHWs" else selected_chw_id_from_filter_chw
zone_id_options_list_chw = ["All Zones"]
if isinstance(_temp_filter_df_chw_dash, pd.DataFrame) and not _temp_filter_df_chw_dash.empty and 'zone_id' in _temp_filter_df_chw_dash.columns:
    zone_id_options_list_chw.extend(sorted(_temp_filter_df_chw_dash['zone_id'].dropna().unique().tolist()))
else: 
    zone_id_options_list_chw.extend(["ZoneA", "ZoneB", "ZoneC", "ZoneD"])
selected_zone_id_from_filter_chw = st.sidebar.selectbox(
    "Filter by Zone:", options=zone_id_options_list_chw, key="supervisor_zone_id_selectbox_page1"
)
actual_zone_id_for_filtering_chw = None if selected_zone_id_from_filter_chw == "All Zones" else selected_zone_id_from_filter_chw
del _temp_filter_df_chw_dash
min_date_for_daily_snapshot_chw = date.today() - timedelta(days=max(90, app_config.WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND * 3))
max_date_for_daily_snapshot_chw = date.today()
selected_daily_view_date_input_chw = st.sidebar.date_input(
    "View Daily Activity For:",
    value=max_date_for_daily_snapshot_chw, 
    min_value=min_date_for_daily_snapshot_chw, 
    max_value=max_date_for_daily_snapshot_chw,
    key="supervisor_daily_snapshot_datepicker_page1"
)
default_trend_end_date_val_chw = selected_daily_view_date_input_chw
default_trend_start_date_val_chw = default_trend_end_date_val_chw - timedelta(days=app_config.WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND -1)
if default_trend_start_date_val_chw < min_date_for_daily_snapshot_chw : default_trend_start_date_val_chw = min_date_for_daily_snapshot_chw
selected_trend_start_date_input_chw, selected_trend_end_date_input_chw = st.sidebar.date_input(
    "Select Date Range for Periodic Trends:",
    value=[default_trend_start_date_val_chw, default_trend_end_date_val_chw],
    min_value=min_date_for_daily_snapshot_chw, 
    max_value=max_date_for_daily_snapshot_chw, 
    key="supervisor_periodic_trend_datepicker_page1"
)
if selected_trend_start_date_input_chw > selected_trend_end_date_input_chw:
    st.sidebar.error("Trend start date must be on or before the end date.")
    selected_trend_start_date_input_chw = selected_trend_end_date_input_chw
daily_df_view_main_chw, period_df_view_main_chw, pre_calculated_kpis_for_view_main_chw = get_chw_supervisor_dashboard_data(
    view_date=selected_daily_view_date_input_chw,
    trend_start_date=selected_trend_start_date_input_chw,
    trend_end_date=selected_trend_end_date_input_chw,
    selected_chw_id=actual_chw_id_for_filtering_chw,
    selected_zone_id=actual_zone_id_for_filtering_chw
)
filter_context_display_parts_chw = [f"Date: **{selected_daily_view_date_input_chw.strftime('%d %b %Y')}**"]
if actual_chw_id_for_filtering_chw: filter_context_display_parts_chw.append(f"CHW: **{actual_chw_id_for_filtering_chw}**")
if actual_zone_id_for_filtering_chw: filter_context_display_parts_chw.append(f"Zone: **{actual_zone_id_for_filtering_chw}**")
st.info(f"Displaying data for: {'  |  '.join(filter_context_display_parts_chw)}")

st.header(f"📊 Daily Performance Snapshot")
if not daily_df_view_main_chw.empty:
    chw_summary_metrics_today = calculate_chw_daily_summary_metrics(
        chw_daily_kpi_input_data=pre_calculated_kpis_for_view_main_chw, 
        chw_daily_encounter_df=daily_df_view_main_chw,
        for_date=selected_daily_view_date_input_chw
    )
    cols_summary_kpi_cards_chw = st.columns(4)
    with cols_summary_kpi_cards_chw[0]: 
        render_web_kpi_card("Visits Today", str(chw_summary_metrics_today.get("visits_count", 0)), 
                            icon="👥", help_text="Total unique patients visited by selected CHW(s) today.")
    high_prio_followups_count_chw = chw_summary_metrics_today.get("high_ai_prio_followups_count", 0)
    high_prio_status_level_chw = "ACCEPTABLE" if high_prio_followups_count_chw <= 2 else \
                             ("MODERATE_CONCERN" if high_prio_followups_count_chw <= 5 else "HIGH_CONCERN")
    with cols_summary_kpi_cards_chw[1]: 
        render_web_kpi_card("High Prio Follow-ups", str(high_prio_followups_count_chw), 
                            icon="🎯", status_level=high_prio_status_level_chw, 
                            help_text="Patients needing urgent follow-up based on AI priority score.")
    critical_spo2_cases_count_chw = chw_summary_metrics_today.get("critical_spo2_cases_identified_count", 0)
    critical_spo2_status_level_chw = "HIGH_CONCERN" if critical_spo2_cases_count_chw > 0 else "ACCEPTABLE"
    with cols_summary_kpi_cards_chw[2]: 
        render_web_kpi_card("Critical SpO2 Cases", str(critical_spo2_cases_count_chw), 
                            icon="💨", status_level=critical_spo2_status_level_chw, 
                            help_text=f"Patients identified with SpO2 < {app_config.ALERT_SPO2_CRITICAL_LOW_PCT}%.")
    high_fever_cases_count_chw = chw_summary_metrics_today.get("high_fever_cases_identified_count", 0)
    high_fever_status_level_chw = "HIGH_CONCERN" if high_fever_cases_count_chw > 0 else "ACCEPTABLE"
    with cols_summary_kpi_cards_chw[3]: 
        render_web_kpi_card("High Fever Cases", str(high_fever_cases_count_chw), 
                            icon="🔥", status_level=high_fever_status_level_chw, 
                            help_text=f"Patients identified with temperature ≥ {app_config.ALERT_BODY_TEMP_HIGH_FEVER_C}°C.")
else:
    st.markdown("_No CHW activity data found for the selected filters to display daily performance snapshot._")
st.divider()

st.header("🚦 Key Alerts & Actionable Tasks Overview")
list_patient_alerts_supervisor_chw = generate_chw_patient_alerts_from_data(
    patient_encounter_data_df=daily_df_view_main_chw, 
    for_date=selected_daily_view_date_input_chw,
    chw_zone_context_str=actual_zone_id_for_filtering_chw or "All Supervised Zones",
    max_alerts_to_return=8 
)
if list_patient_alerts_supervisor_chw:
    st.subheader(f"Priority Patient Alerts (Requiring Supervisor Review/Action):")
    found_any_critical_alert_chw = False
    for alert_data_item_chw in list_patient_alerts_supervisor_chw:
        if alert_data_item_chw.get("alert_level") == "CRITICAL":
            found_any_critical_alert_chw = True
            render_web_traffic_light_indicator(
                message=f"Pt. {alert_data_item_chw.get('patient_id', 'N/A')}: {alert_data_item_chw.get('primary_reason', 'Critical Alert')}",
                status_level="HIGH_RISK", 
                details_text=f"Details: {alert_data_item_chw.get('brief_details','N/A')} | Context: {alert_data_item_chw.get('context_info','N/A')} | Suggested Action: {alert_data_item_chw.get('suggested_action_code','REVIEW_IMMEDIATELY')}"
            )
    if not found_any_critical_alert_chw:
        st.info("No CRITICAL patient alerts identified from field data for this selection. Checking for warnings...")
    warning_alerts_to_show_chw = [a for a in list_patient_alerts_supervisor_chw if a.get("alert_level") == "WARNING"]
    if warning_alerts_to_show_chw and (not found_any_critical_alert_chw or len(list_patient_alerts_supervisor_chw) > sum(1 for a in list_patient_alerts_supervisor_chw if a.get("alert_level") == "CRITICAL")):
        st.markdown("###### Notable Warning Alerts:")
        for alert_data_item_warn_chw in warning_alerts_to_show_chw:
            render_web_traffic_light_indicator(
                message=f"Pt. {alert_data_item_warn_chw.get('patient_id', 'N/A')}: {alert_data_item_warn_chw.get('primary_reason', 'Warning')}",
                status_level="MODERATE_RISK",
                details_text=f"Details: {alert_data_item_warn_chw.get('brief_details','N/A')} | Context: {alert_data_item_warn_chw.get('context_info','N/A')}"
            )
    elif not found_any_critical_alert_chw and not warning_alerts_to_show_chw and list_patient_alerts_supervisor_chw: 
        st.info("Only informational alerts were generated for this selection.")
elif not daily_df_view_main_chw.empty : 
    st.info("No patient alerts (Critical or Warning) generated from field data for this selection.")
else: 
    st.markdown("_No CHW activity data available to generate alerts for this selection._")

list_generated_tasks_supervisor_chw = generate_chw_prioritized_tasks(
    source_patient_data_df=daily_df_view_main_chw,
    for_date=selected_daily_view_date_input_chw,
    chw_id_context=actual_chw_id_for_filtering_chw, 
    zone_context_str=actual_zone_id_for_filtering_chw or "All Supervised Zones",
    max_tasks_to_return_for_summary=10
)
if list_generated_tasks_supervisor_chw:
    st.subheader(f"Top Priority Tasks from Today's Activities:")
    df_tasks_to_display_supervisor_chw = pd.DataFrame(list_generated_tasks_supervisor_chw)
    cols_for_task_display_supervisor_chw = ['patient_id', 'task_description', 'priority_score', 'due_date', 'status', 'key_patient_context', 'assigned_chw_id', 'alert_source_info']
    actual_cols_in_tasks_df_chw = [col for col in cols_for_task_display_supervisor_chw if col in df_tasks_to_display_supervisor_chw.columns]
    st.dataframe(
        df_tasks_to_display_supervisor_chw[actual_cols_in_tasks_df_chw], 
        use_container_width=True, 
        height=min(380, len(df_tasks_to_display_supervisor_chw)*38 + 58), 
        hide_index=True
    )
elif not daily_df_view_main_chw.empty:
    st.info("No new high-priority tasks identified from today's activities based on current filters.")
else:
    st.markdown("_No CHW activity data available to generate tasks for this selection._")
st.divider()

st.header("🔬 Local Epi Signals Watch")
if not daily_df_view_main_chw.empty:
    dict_local_epi_signals_chw = extract_chw_local_epi_signals(
        chw_daily_encounter_df=daily_df_view_main_chw,
        pre_calculated_chw_kpis=pre_calculated_kpis_for_view_main_chw,
        for_date=selected_daily_view_date_input_chw,
        chw_zone_context=actual_zone_id_for_filtering_chw or "All Supervised Zones"
    )
    cols_epi_kpi_cards_chw = st.columns(3)
    with cols_epi_kpi_cards_chw[0]: render_web_kpi_card(title="Symptomatic (Key Cond.)", value=str(dict_local_epi_signals_chw.get("symptomatic_patients_key_conditions_count", 0)), icon="🤒", units="cases today", help_text="Unique patients with key conditions reporting significant symptoms today.")
    with cols_epi_kpi_cards_chw[1]: render_web_kpi_card(title="New Malaria Cases Today", value=str(dict_local_epi_signals_chw.get("newly_identified_malaria_patients_count",0)), icon="🦟", units="cases", help_text="Malaria cases identified from today's CHW encounters.")
    with cols_epi_kpi_cards_chw[2]: render_web_kpi_card(title="Pending TB Contacts", value=str(dict_local_epi_signals_chw.get("pending_tb_contact_tracing_tasks_count",0)), icon="👥", units="to trace", help_text="Number of TB contacts identified today needing follow-up by CHW team.")
    list_detected_symptom_clusters_chw = dict_local_epi_signals_chw.get("detected_symptom_clusters", [])
    if list_detected_symptom_clusters_chw:
        st.markdown("###### Detected Symptom Clusters Today (Requires Supervisor Review):")
        for cluster_item_data_epi_chw in list_detected_symptom_clusters_chw:
            st.warning(f"⚠️ **Pattern: {cluster_item_data_epi_chw.get('symptoms_pattern','Unknown Pattern')}**: {cluster_item_data_epi_chw.get('patient_count','N/A')} cases in {cluster_item_data_epi_chw.get('location_hint', 'area')}")
    elif 'patient_reported_symptoms' in daily_df_view_main_chw.columns: 
        st.info("No significant symptom clusters detected from today's CHW data based on current criteria.")
else:
    st.markdown("_No CHW activity data available to derive local epi signals for this selection._")
st.divider()

st.header("📈 CHW Team Activity Trends")
trend_period_display_str_chw_page = f"{selected_trend_start_date_input_chw.strftime('%d %b %Y')} - {selected_trend_end_date_input_chw.strftime('%d %b %Y')}"
trend_filter_context_display_str_chw = ""
if actual_chw_id_for_filtering_chw: trend_filter_context_display_str_chw += f" for CHW {actual_chw_id_for_filtering_chw}"
if actual_zone_id_for_filtering_chw: trend_filter_context_display_str_chw += f" in Zone {actual_zone_id_for_filtering_chw}"
st.markdown(f"Displaying trends from **{trend_period_display_str_chw_page}**{trend_filter_context_display_str_chw if trend_filter_context_display_str_chw else ' (All CHWs/Zones in period)'}.")

if not period_df_view_main_chw.empty:
    dict_chw_activity_trends_page = calculate_chw_activity_trends(
        chw_historical_health_df=period_df_view_main_chw, 
        trend_start_date_input=selected_trend_start_date_input_chw, 
        trend_end_date_input=selected_trend_end_date_input_chw,
        zone_filter=None, 
        time_period_aggregation='D'
    )
    cols_trends_charts_chw_page = st.columns(2)
    with cols_trends_charts_chw_page[0]:
        series_patient_visits_trend_page = dict_chw_activity_trends_page.get("patient_visits_trend")
        if isinstance(series_patient_visits_trend_page, pd.Series) and not series_patient_visits_trend_page.empty:
            st.plotly_chart(plot_annotated_line_chart_web(
                series_patient_visits_trend_page, chart_title="Daily Patient Visits (Trend)",
                y_axis_label="# Patients Visited", y_is_count=True
            ), use_container_width=True)
        else: st.caption("No patient visit trend data available for this selection.")
    with cols_trends_charts_chw_page[1]:
        series_high_prio_followups_trend_page = dict_chw_activity_trends_page.get("high_priority_followups_trend")
        if isinstance(series_high_prio_followups_trend_page, pd.Series) and not series_high_prio_followups_trend_page.empty:
            st.plotly_chart(plot_annotated_line_chart_web(
                series_high_prio_followups_trend_page, chart_title="Daily High Prio. Follow-ups (Trend)",
                y_axis_label="# High Prio Follow-ups", y_is_count=True
            ), use_container_width=True)
        else: st.caption("No high-priority follow-up trend data available for this selection.")
else:
    st.markdown("_No historical data available for the selected filters/period to display activity trends._")

logger.info(f"CHW Supervisor View page loaded/refreshed for Date: {selected_daily_view_date_input_chw}, CHW: {actual_chw_id_for_filtering_chw or 'All'}, Zone: {actual_zone_id_for_filtering_chw or 'All'}.")
