# sentinel_project_root/test/pages/1_chw_dashboard.py
# CHW Supervisor Operations View for Sentinel Health Co-Pilot

import streamlit as st
import pandas as pd
# import numpy as np # Not directly used after refactor
import os
import logging
from datetime import date, timedelta # datetime not directly needed here
from typing import Optional, Dict, Any, Tuple, List # Added List, Tuple
import inspect # For logging function names

# --- Sentinel System Imports ---
# Assuming 'test' directory (where app_home.py is) is the app root for Streamlit page execution.
# This makes 'config' and 'utils' top-level packages from the perspective of this page.
try:
    from config import app_config
    from utils.core_data_processing import load_health_records # For initial filter population & main data load
    from utils.ui_visualization_helpers import (
        render_web_kpi_card,
        render_web_traffic_light_indicator,
        plot_annotated_line_chart_web
    )
    # CHW Component specific data processors, imported relative to 'pages' directory
    from .chw_components_sentinel.summary_metrics_calculator import calculate_chw_daily_summary_metrics
    from .chw_components_sentinel.alert_generator import generate_chw_patient_alerts_from_data
    from .chw_components_sentinel.epi_signal_extractor import extract_chw_local_epi_signals
    from .chw_components_sentinel.task_processor import generate_chw_prioritized_tasks
    from .chw_components_sentinel.activity_trend_calculator import calculate_chw_activity_trends
except ImportError as e:
    st.error(f"Critical import error in CHW Dashboard: {e}. Ensure all modules are correctly placed and PYTHONPATH is set if running outside Streamlit's standard structure.")
    st.stop() # Stop execution if core components can't be imported


# --- Page Configuration ---
st.set_page_config(
    page_title=f"CHW Supervisor View - {app_config.APP_NAME}",
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
    selected_chw_id: Optional[str] = None, # If supervisor wants to drill down
    selected_zone_id: Optional[str] = None # If supervisor covers multiple zones
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]: # daily_df, period_df, pre_calc_kpis
    """
    Simulates fetching and preparing data needed for the CHW Supervisor View.
    In a real system, this would query an aggregated database at a Hub/Facility Node.
    """
    current_func_name = inspect.currentframe().f_code.co_name if inspect.currentframe() else "get_chw_supervisor_dashboard_data"
    
    # For simulation, load all health records. AI scores are assumed to be part of this data.
    health_df_all_synced = load_health_records(source_context=f"{current_func_name}/LoadSimData")
    
    if health_df_all_synced.empty:
        logger.error(f"({current_func_name}) Base health records (simulating synced data) failed to load or are empty. Cannot proceed.")
        return pd.DataFrame(), pd.DataFrame(), {} # Return empty structures

    # Ensure 'encounter_date' is datetime and handle potential errors (load_health_records should do this)
    if 'encounter_date' not in health_df_all_synced.columns or \
       not pd.api.types.is_datetime64_any_dtype(health_df_all_synced['encounter_date']):
        logger.error(f"({current_func_name}) 'encounter_date' column missing or not datetime in loaded health records.")
        # Attempt conversion again just in case, though loader should handle it
        health_df_all_synced['encounter_date'] = pd.to_datetime(health_df_all_synced.get('encounter_date'), errors='coerce')
        health_df_all_synced.dropna(subset=['encounter_date'], inplace=True)
        if health_df_all_synced.empty or 'encounter_date' not in health_df_all_synced.columns:
             return pd.DataFrame(), pd.DataFrame(), {}


    # Filter data based on supervisor's scope (date, CHW, Zone)
    # 1. Daily Snapshot Data
    daily_df_sup = health_df_all_synced[health_df_all_synced['encounter_date'].dt.date == view_date].copy()
    if selected_chw_id and 'chw_id' in daily_df_sup.columns:
        daily_df_sup = daily_df_sup[daily_df_sup['chw_id'] == selected_chw_id]
    if selected_zone_id and 'zone_id' in daily_df_sup.columns:
        daily_df_sup = daily_df_sup[daily_df_sup['zone_id'] == selected_zone_id]

    # 2. Period Data (for trends, typically broader than daily snapshot)
    period_df_sup = health_df_all_synced[
        (health_df_all_synced['encounter_date'].dt.date >= trend_start_date) &
        (health_df_all_synced['encounter_date'].dt.date <= trend_end_date)
    ].copy()
    if selected_chw_id and 'chw_id' in period_df_sup.columns:
        period_df_sup = period_df_sup[period_df_sup['chw_id'] == selected_chw_id]
    if selected_zone_id and 'zone_id' in period_df_sup.columns:
        period_df_sup = period_df_sup[period_df_sup['zone_id'] == selected_zone_id]
        
    # Pre-calculated KPIs for the daily summary (e.g., CHW's self-reported fatigue if available)
    # This would typically come from a separate data source or be part of daily_df_sup if it includes WORKER_SELF_CHECK for the CHW.
    pre_calculated_daily_kpis_for_supervisor = {}
    if selected_chw_id and not daily_df_sup.empty: # If a specific CHW is selected
        worker_self_check_df = daily_df_sup[
            (daily_df_sup.get('chw_id') == selected_chw_id) & 
            (daily_df_sup.get('encounter_type') == 'WORKER_SELF_CHECK')
        ]
        if not worker_self_check_df.empty and 'ai_followup_priority_score' in worker_self_check_df.columns:
            # Assuming 'ai_followup_priority_score' from WORKER_SELF_CHECK is the fatigue index
            pre_calculated_daily_kpis_for_supervisor['worker_self_fatigue_index_today'] = worker_self_check_df['ai_followup_priority_score'].max() # Take max if multiple self-checks

    logger.info(f"({current_func_name}) Data loaded for supervisor: Daily - {len(daily_df_sup)} recs, Period - {len(period_df_sup)} recs.")
    return daily_df_sup, period_df_sup, pre_calculated_daily_kpis_for_supervisor

# --- Page Title & Introduction ---
st.title("🧑‍🏫 CHW Supervisor Operations View")
st.markdown(f"**Team Performance Monitoring, Alert Triage, and Field Activity Oversight for {app_config.APP_NAME}**")
st.divider()

# --- Sidebar Filters ---
# Attempt a light load of health_records just for populating filter options dynamically
# This avoids loading the full dataset if only filter options are needed initially.
# However, the main data load function is cached, so repeated calls with same params are fast.
# For simplicity in this refactor, using a one-time load or placeholders if dynamic options are too slow.
_filter_options_df = load_health_records(source_context="CHWDashboard/FilterInit") # Light load for options

if os.path.exists(app_config.APP_LOGO_SMALL):
    st.sidebar.image(app_config.APP_LOGO_SMALL, width=150)
else:
    st.sidebar.markdown("🌍", unsafe_allow_html=True) # Fallback icon
st.sidebar.header("🗓️ View Filters")

chw_ids_options = ["All CHWs"]
if isinstance(_filter_options_df, pd.DataFrame) and not _filter_options_df.empty and 'chw_id' in _filter_options_df.columns:
    chw_ids_options.extend(sorted(_filter_options_df['chw_id'].dropna().unique().tolist()))
else: # Fallback if dynamic options fail
    chw_ids_options.extend(["CHW001", "CHW002", "CHW003", "CHW004", "CHW005"]) # From sample data

selected_chw_id_filter = st.sidebar.selectbox(
    "Filter by CHW ID:", options=chw_ids_options, key="supervisor_chw_id_filter_selectbox"
)
actual_chw_id_for_filtering = None if selected_chw_id_filter == "All CHWs" else selected_chw_id_filter

zone_ids_options = ["All Zones"]
if isinstance(_filter_options_df, pd.DataFrame) and not _filter_options_df.empty and 'zone_id' in _filter_options_df.columns:
    zone_ids_options.extend(sorted(_filter_options_df['zone_id'].dropna().unique().tolist()))
else: # Fallback
    zone_ids_options.extend(["ZoneA", "ZoneB", "ZoneC", "ZoneD"])

selected_zone_id_filter = st.sidebar.selectbox(
    "Filter by Zone:", options=zone_ids_options, key="supervisor_zone_id_filter_selectbox"
)
actual_zone_id_for_filtering = None if selected_zone_id_filter == "All Zones" else selected_zone_id_filter
del _filter_options_df # Clean up temp df

# Date selection for "Daily Snapshot"
min_date_for_snapshot = date.today() - timedelta(days=max(90, app_config.WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND * 3)) # Max 90 days or 3x trend range
max_date_for_snapshot = date.today()
selected_daily_view_date = st.sidebar.date_input(
    "View Daily Activity For:",
    value=max_date_for_snapshot, min_value=min_date_for_snapshot, max_value=max_date_for_snapshot,
    key="supervisor_daily_snapshot_date_picker"
)

# Date range for "Periodic Trends"
default_trend_end_date = selected_daily_view_date # Align trend end with snapshot date by default
default_trend_start_date = default_trend_end_date - timedelta(days=app_config.WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND -1)
if default_trend_start_date < min_date_for_snapshot : default_trend_start_date = min_date_for_snapshot

selected_trend_start_dt, selected_trend_end_dt = st.sidebar.date_input(
    "Select Date Range for Periodic Trends:",
    value=[default_trend_start_date, default_trend_end_date],
    min_value=min_date_for_snapshot, max_value=max_date_for_snapshot,
    key="supervisor_periodic_trend_date_range_picker"
)
if selected_trend_start_dt > selected_trend_end_dt:
    st.sidebar.error("Trend start date must be on or before the end date.")
    selected_trend_start_dt = selected_trend_end_dt # Auto-correct to valid range

# --- Load Data Based on Filter Selections ---
daily_df_for_view, period_df_for_view, pre_calculated_supervisor_kpis_view = get_chw_supervisor_dashboard_data(
    view_date=selected_daily_view_date,
    trend_start_date=selected_trend_start_dt,
    trend_end_date=selected_trend_end_dt,
    selected_chw_id=actual_chw_id_for_filtering,
    selected_zone_id=actual_zone_id_for_filtering
)

# Display context for the selected filters
filter_context_str_parts = [f"Date: **{selected_daily_view_date.strftime('%d %b %Y')}**"]
if actual_chw_id_for_filtering: filter_context_str_parts.append(f"CHW: **{actual_chw_id_for_filtering}**")
if actual_zone_id_for_filtering: filter_context_str_parts.append(f"Zone: **{actual_zone_id_for_filtering}**")
st.info(f"Displaying data for: {' | '.join(filter_context_str_parts)}")


# --- Section 1: Daily Performance Snapshot ---
st.header(f"📊 Daily Performance Snapshot")
if not daily_df_for_view.empty:
    summary_metrics_chw = calculate_chw_daily_summary_metrics(
        chw_daily_kpi_input_data=pre_calculated_supervisor_kpis_view, 
        chw_daily_encounter_df=daily_df_for_view,
        for_date=selected_daily_view_date
    )
    
    cols_summary_kpis = st.columns(4)
    with cols_summary_kpis[0]: render_web_kpi_card("Visits Today", str(summary_metrics_chw.get("visits_count", 0)), icon="👥", help_text="Total unique patients visited by selected CHW(s) today.")
    
    high_prio_followups = summary_metrics_chw.get("high_ai_prio_followups_count", 0)
    prio_status = "ACCEPTABLE" if high_prio_followups <= 2 else ("MODERATE_CONCERN" if high_prio_followups <= 5 else "HIGH_CONCERN") # Example thresholds
    with cols_summary_kpis[1]: render_web_kpi_card("High Prio Follow-ups", str(high_prio_followups), icon="🎯", status_level=prio_status, help_text="Patients needing urgent follow-up based on AI priority score.")
    
    critical_spo2_count_val = summary_metrics_chw.get("critical_spo2_cases_identified_count", 0)
    spo2_status = "HIGH_CONCERN" if critical_spo2_count_val > 0 else "ACCEPTABLE"
    with cols_summary_kpis[2]: render_web_kpi_card("Critical SpO2 Cases", str(critical_spo2_count_val), icon="💨", status_level=spo2_status, help_text=f"Patients identified with SpO2 < {app_config.ALERT_SPO2_CRITICAL_LOW_PCT}%.")
    
    high_fever_count_val = summary_metrics_chw.get("high_fever_cases_identified_count", 0) # Use specific high_fever key
    fever_status = "HIGH_CONCERN" if high_fever_count_val > 0 else "ACCEPTABLE"
    with cols_summary_kpis[3]: render_web_kpi_card("High Fever Cases", str(high_fever_count_val), icon="🔥", status_level=fever_status, help_text=f"Patients identified with temperature ≥ {app_config.ALERT_BODY_TEMP_HIGH_FEVER_C}°C.")
else:
    st.markdown("_No CHW activity data found for the selected filters to display daily performance snapshot._")
st.divider()

# --- Section 2: Key Alerts & Actionable Task Summary ---
st.header("🚦 Key Alerts & Actionable Tasks Overview")

# Generate Alerts from daily activity data
alerts_list_for_supervisor = generate_chw_patient_alerts_from_data(
    patient_encounter_data_df=daily_df_for_view, 
    for_date=selected_daily_view_date,
    chw_zone_context_str=actual_zone_id_for_filtering or "All Supervised Zones",
    max_alerts_to_return=8 # Show a few top alerts
)
if alerts_list_for_supervisor:
    st.subheader(f"Priority Patient Alerts (Requiring Review/Action):")
    found_critical_alert = False
    for alert_item_data in alerts_list_for_supervisor:
        if alert_item_data.get("alert_level") == "CRITICAL":
            found_critical_alert = True
            render_web_traffic_light_indicator(
                message=f"Pt. {alert_item_data.get('patient_id', 'N/A')}: {alert_item_data.get('primary_reason', 'Alert')}",
                status_level="HIGH_RISK", # Map CRITICAL to HIGH_RISK style for traffic light
                details_text=f"{alert_item_data.get('brief_details','N/A')} | Context: {alert_item_data.get('context_info','N/A')} | Suggested: {alert_item_data.get('suggested_action_code','REVIEW')}"
            )
    if not found_critical_alert and any(a.get("alert_level") == "WARNING" for a in alerts_list_for_supervisor):
        st.markdown("###### Notable Warning Alerts:")
        for alert_item_data in alerts_list_for_supervisor:
            if alert_item_data.get("alert_level") == "WARNING":
                 render_web_traffic_light_indicator(
                    message=f"Pt. {alert_item_data.get('patient_id', 'N/A')}: {alert_item_data.get('primary_reason', 'Warning')}",
                    status_level="MODERATE_RISK",
                    details_text=f"{alert_item_data.get('brief_details','N/A')} | Context: {alert_item_data.get('context_info','N/A')}"
                )
    elif not alerts_list_for_supervisor : # List is empty, but df might not have been
         st.info("No specific patient alerts generated from field data for this selection.")

elif not daily_df_for_view.empty : # Data exists but no alerts generated at all
    st.info("No patient alerts (Critical or Warning) identified from field data for this selection.")
else: # No data to generate alerts from
    st.markdown("_No CHW activity data to generate alerts for this selection._")


# Generate Tasks from daily activity data
tasks_list_for_supervisor = generate_chw_prioritized_tasks(
    source_patient_data_df=daily_df_for_view,
    for_date=selected_daily_view_date,
    chw_id_context=actual_chw_id_for_filtering, 
    zone_context_str=actual_zone_id_for_filtering or "All Supervised Zones",
    max_tasks_to_return_for_summary=10 # Show top N tasks
)
if tasks_list_for_supervisor:
    st.subheader(f"Top Priority Tasks from Today's Activities:")
    df_tasks_for_display = pd.DataFrame(tasks_list_for_supervisor)
    # Define columns for supervisor's task view
    task_display_cols_ordered = ['patient_id', 'task_description', 'priority_score', 'due_date', 'status', 'key_patient_context', 'assigned_chw_id', 'alert_source_info']
    actual_task_cols_for_display = [col for col in task_display_cols_ordered if col in df_tasks_for_display.columns]
    
    st.dataframe(
        df_tasks_for_display[actual_task_cols_for_display], 
        use_container_width=True, 
        height=min(380, len(df_tasks_for_display)*38 + 58), # Dynamic height with max, accounts for header
        hide_index=True
    )
elif not daily_df_for_view.empty:
    st.info("No new high-priority tasks identified from today's activities based on current filters.")
else:
    st.markdown("_No CHW activity data to generate tasks for this selection._")
st.divider()


# --- Section 3: Local Epi Signals from Field ---
st.header("🔬 Local Epi Signals Watch")
if not daily_df_for_view.empty:
    epi_signals_data = extract_chw_local_epi_signals(
        chw_daily_encounter_df=daily_df_for_view,
        pre_calculated_chw_kpis=pre_calculated_supervisor_kpis_view,
        for_date=selected_daily_view_date,
        chw_zone_context=actual_zone_id_for_filtering or "All Supervised Zones"
    )
    
    cols_epi_kpi = st.columns(3)
    with cols_epi_kpi[0]: render_web_kpi_card(title="Symptomatic (Key Cond.)", value=str(epi_signals_data.get("symptomatic_patients_key_conditions_count", 0)), icon="🤒", units="cases today", help_text="Unique patients with key conditions reporting symptoms today.")
    with cols_epi_kpi[1]: render_web_kpi_card(title="New Malaria Cases", value=str(epi_signals_data.get("newly_identified_malaria_patients_count",0)), icon="🦟", units="cases today", help_text="Malaria cases identified from today's CHW encounters.")
    with cols_epi_kpi[2]: render_web_kpi_card(title="Pending TB Contacts", value=str(epi_signals_data.get("pending_tb_contact_tracing_tasks_count",0)), icon="👥", units="to trace", help_text="Number of TB contacts identified today needing follow-up.")
    
    symptom_clusters_list = epi_signals_data.get("detected_symptom_clusters", [])
    if symptom_clusters_list:
        st.markdown("###### Detected Symptom Clusters Today:")
        for cluster_data_item in symptom_clusters_list:
            st.warning(f"⚠️ **Pattern: {cluster_data_item.get('symptoms_pattern','Unknown')}**: {cluster_data_item.get('patient_count','N/A')} cases in {cluster_data_item.get('location_hint', 'area')}")
    elif 'patient_reported_symptoms' in daily_df_for_view.columns: # Check if symptoms data was even available
        st.info("No significant symptom clusters detected from today's CHW data based on current criteria.")
else:
    st.markdown("_No CHW activity data to derive local epi signals for this selection._")
st.divider()

# --- Section 4: CHW Team Activity Trends (Periodic View) ---
st.header("📈 CHW Team Activity Trends")
trend_period_str_display = f"{selected_trend_start_dt.strftime('%d %b %Y')} - {selected_trend_end_dt.strftime('%d %b %Y')}"
trend_filter_context_str = ""
if actual_chw_id_for_filtering: trend_filter_context_str += f" for CHW {actual_chw_id_for_filtering}"
if actual_zone_id_for_filtering: trend_filter_context_str += f" in Zone {actual_zone_id_for_filtering}"
st.markdown(f"Displaying trends from **{trend_period_str_display}**{trend_filter_context_str if trend_filter_context_str else ' (All CHWs/Zones in period)'}.")

if not period_df_for_view.empty:
    chw_activity_trends_data_dict = calculate_chw_activity_trends(
        chw_historical_health_df=period_df_for_view, # This data is already filtered for period and CHW/Zone
        trend_start_date_input=selected_trend_start_dt, 
        trend_end_date_input=selected_trend_end_dt,
        zone_filter=None, # Data is already zone-filtered if a zone was selected by supervisor. Pass None to avoid re-filtering.
        time_period_aggregation='D' # Daily trends for supervisor view
    )
    
    cols_trends_charts = st.columns(2)
    with cols_trends_charts[0]:
        patient_visits_trend_series = chw_activity_trends_data_dict.get("patient_visits_trend")
        if isinstance(patient_visits_trend_series, pd.Series) and not patient_visits_trend_series.empty:
            st.plotly_chart(plot_annotated_line_chart_web(
                patient_visits_trend_series, chart_title="Daily Patient Visits (Trend)",
                y_axis_label="# Patients Visited", y_is_count=True
            ), use_container_width=True)
        else: st.caption("No patient visit trend data available for this selection.")
        
    with cols_trends_charts[1]:
        high_prio_followups_trend_series = chw_activity_trends_data_dict.get("high_priority_followups_trend")
        if isinstance(high_prio_followups_trend_series, pd.Series) and not high_prio_followups_trend_series.empty:
            st.plotly_chart(plot_annotated_line_chart_web(
                high_prio_followups_trend_series, chart_title="Daily High Prio. Follow-ups (Trend)",
                y_axis_label="# High Prio Follow-ups", y_is_count=True
            ), use_container_width=True)
        else: st.caption("No high-priority follow-up trend data available for this selection.")
else:
    st.markdown("_No historical data available for the selected filters/period to display activity trends._")

logger.info(f"CHW Supervisor View page loaded/refreshed for Date: {selected_daily_view_date}, CHW: {actual_chw_id_for_filtering or 'All'}, Zone: {actual_zone_id_for_filtering or 'All'}.")
