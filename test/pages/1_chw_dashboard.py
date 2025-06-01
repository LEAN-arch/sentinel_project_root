# sentinel_project_root/test/pages/1_chw_dashboard.py
# CHW Supervisor Operations View for Sentinel Health Co-Pilot

import streamlit as st
import pandas as pd
import numpy as np # Kept for potential future use, though not directly used now
import os
import logging
from datetime import date, timedelta, datetime # Added datetime for pd.to_datetime
from typing import Optional, Dict, Any, Tuple, List # Added List for return type

# --- Sentinel System Imports ---
# Assuming 'test' directory is the app root for Streamlit page execution.
from config import app_config
from utils.core_data_processing import load_health_records # For loading base data
from utils.ui_visualization_helpers import (
    render_web_kpi_card,
    render_web_traffic_light_indicator,
    plot_annotated_line_chart_web
)
# CHW Component specific data processors
from .chw_components_sentinel.summary_metrics_calculator import calculate_chw_daily_summary_metrics
from .chw_components_sentinel.alert_generator import generate_chw_patient_alerts_from_data
from .chw_components_sentinel.epi_signal_extractor import extract_chw_local_epi_signals
from .chw_components_sentinel.task_processor import generate_chw_prioritized_tasks
from .chw_components_sentinel.activity_trend_calculator import calculate_chw_activity_trends

# --- Page Configuration ---
st.set_page_config(
    page_title=f"CHW Supervisor View - {app_config.APP_NAME}",
    layout="wide",
    initial_sidebar_state="expanded"
)

logger = logging.getLogger(__name__) # Page-specific logger

# --- Data Loading Logic for Supervisor View (Simulation) ---
@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS_WEB_REPORTS, show_spinner="Loading CHW operational data...")
def get_chw_supervisor_dashboard_data(
    view_date: date,
    trend_start_date: date,
    trend_end_date: date,
    selected_chw_id: Optional[str] = None,
    selected_zone_id: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Simulates fetching and preparing data for the CHW Supervisor View.
    In a real system, this queries an aggregated Hub/Facility Node database.
    """
    # For simulation, load all health records.
    # AI scores are assumed to be part of this data (enriched upstream or by PEDs).
    health_df_all = load_health_records(source_context="CHWSupervisorDataSim") # Uses app_config.HEALTH_RECORDS_CSV internally
    
    if health_df_all.empty:
        logger.error(f"{inspect.currentframe().f_code.co_name}: Base health records (simulating synced data) failed to load or are empty.")
        return pd.DataFrame(), pd.DataFrame(), {}

    # Ensure 'encounter_date' is datetime and handle potential errors
    if 'encounter_date' not in health_df_all.columns:
        logger.error(f"{inspect.currentframe().f_code.co_name}: 'encounter_date' column missing from health records.")
        return pd.DataFrame(), pd.DataFrame(), {}
        
    health_df_all['encounter_date'] = pd.to_datetime(health_df_all['encounter_date'], errors='coerce')
    health_df_all.dropna(subset=['encounter_date'], inplace=True) # Remove rows where date conversion failed

    # Filter data for the daily snapshot
    daily_df = health_df_all[health_df_all['encounter_date'].dt.date == view_date].copy()
    if selected_chw_id and 'chw_id' in daily_df.columns:
        daily_df = daily_df[daily_df['chw_id'] == selected_chw_id]
    if selected_zone_id and 'zone_id' in daily_df.columns:
        daily_df = daily_df[daily_df['zone_id'] == selected_zone_id]

    # Filter data for the trend period
    period_df = health_df_all[
        (health_df_all['encounter_date'].dt.date >= trend_start_date) &
        (health_df_all['encounter_date'].dt.date <= trend_end_date)
    ].copy()
    if selected_chw_id and 'chw_id' in period_df.columns:
        period_df = period_df[period_df['chw_id'] == selected_chw_id]
    if selected_zone_id and 'zone_id' in period_df.columns:
        period_df = period_df[period_df['zone_id'] == selected_zone_id]
        
    # For this simulation, pre-calculated KPIs for daily summary are minimal.
    # `calculate_chw_daily_summary_metrics` will derive most from `daily_df`.
    # In a real system, some daily roll-ups might be pre-calculated (e.g., CHW's own fatigue score).
    pre_calculated_supervisor_kpis = {} 
    # Example: if CHW self-reported fatigue was part of a separate daily log
    # if selected_chw_id and not daily_df.empty:
    #    worker_fatigue_record = daily_df[(daily_df['chw_id'] == selected_chw_id) & (daily_df['encounter_type'] == 'WORKER_SELF_CHECK')]
    #    if not worker_fatigue_record.empty:
    #         pre_calculated_supervisor_kpis['worker_self_fatigue_index_today'] = worker_fatigue_record['ai_followup_priority_score'].iloc[0] # Assuming this field is used for fatigue

    return daily_df, period_df, pre_calculated_supervisor_kpis

# --- Page Title & Introduction ---
st.title("🧑‍🏫 CHW Supervisor Operations View")
st.markdown(f"**Team Performance, Alert Triage, and Field Activity Oversight for {app_config.APP_NAME}**")
st.divider() # More modern separator

# --- Sidebar Filters ---
if os.path.exists(app_config.APP_LOGO_SMALL): # Check existence before trying to display
    st.sidebar.image(app_config.APP_LOGO_SMALL, width=150) # Slightly smaller for sidebar
else:
    st.sidebar.markdown("🌍", unsafe_allow_html=True) # Fallback
st.sidebar.header("🗓️ View Filters")

# Dynamically populate CHW IDs and Zones if data is loaded, else use placeholders
# This would typically happen after initial data load if filters depend on data content.
# For this structure, we load data *after* filters are set. So, static placeholders are used.
# In a more complex app, initial data load might happen first, or filters are hardcoded/config-driven.
_placeholder_health_df = load_health_records(source_context="CHWSupervisorSidebarInit") # Light load for filter options
available_chw_ids = ["All CHWs"] + sorted(_placeholder_health_df['chw_id'].dropna().unique().tolist()) if not _placeholder_health_df.empty and 'chw_id' in _placeholder_health_df else ["All CHWs", "CHW01", "CHW02", "CHW03"]
selected_chw_filter_val = st.sidebar.selectbox(
    "Filter by CHW ID:", options=available_chw_ids, key="supervisor_chw_id_filter"
)
chw_to_filter = None if selected_chw_filter_val == "All CHWs" else selected_chw_filter_val

available_zones = ["All Zones"] + sorted(_placeholder_health_df['zone_id'].dropna().unique().tolist()) if not _placeholder_health_df.empty and 'zone_id' in _placeholder_health_df else ["All Zones", "ZoneA", "ZoneB", "ZoneC"]
selected_zone_filter_val = st.sidebar.selectbox(
    "Filter by Zone:", options=available_zones, key="supervisor_zone_filter"
)
zone_to_filter = None if selected_zone_filter_val == "All Zones" else selected_zone_filter_val
del _placeholder_health_df # Free memory

# Date selection for "Daily Snapshot"
min_snapshot_date = date.today() - timedelta(days=app_config.WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND * 3) # e.g., 90 days back
max_snapshot_date = date.today()
selected_daily_date = st.sidebar.date_input(
    "View Daily Activity For:",
    value=max_snapshot_date, min_value=min_snapshot_date, max_value=max_snapshot_date,
    key="supervisor_daily_snapshot_date"
)

# Date range for "Periodic Trends"
default_trend_end = selected_daily_date
default_trend_start = default_trend_end - timedelta(days=app_config.WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND -1)
if default_trend_start < min_snapshot_date : default_trend_start = min_snapshot_date

selected_trend_start_date, selected_trend_end_date = st.sidebar.date_input(
    "Select Date Range for Periodic Trends:",
    value=[default_trend_start, default_trend_end],
    min_value=min_snapshot_date, max_value=max_snapshot_date,
    key="supervisor_periodic_trend_range"
)
if selected_trend_start_date > selected_trend_end_date:
    st.sidebar.error("Trend start date must be on or before end date.")
    selected_trend_start_date = selected_trend_end_date # Auto-correct

# --- Load Data Based on Filter Selections ---
daily_activity_df, period_activity_df, pre_calculated_kpis = get_chw_supervisor_dashboard_data(
    view_date=selected_daily_date,
    trend_start_date=selected_trend_start_date,
    trend_end_date=selected_trend_end_date,
    selected_chw_id=chw_to_filter,
    selected_zone_id=zone_to_filter
)

# Display context for the selected filters
filter_context_parts = [f"Date: {selected_daily_date.strftime('%d %b %Y')}"]
if chw_to_filter: filter_context_parts.append(f"CHW: {chw_to_filter}")
if zone_to_filter: filter_context_parts.append(f"Zone: {zone_to_filter}")
st.info(f"**Displaying data for:** {' | '.join(filter_context_parts)}")


# --- Section 1: Daily Performance Snapshot ---
st.header(f"📊 Daily Performance Snapshot")
if not daily_activity_df.empty:
    # Pass pre_calculated_kpis which might contain CHW's own self-reported fatigue
    chw_summary_kpis = calculate_chw_daily_summary_metrics(
        chw_daily_kpi_input_data=pre_calculated_kpis, 
        chw_daily_encounter_df=daily_activity_df,
        for_date=selected_daily_date
    )
    
    cols_summary = st.columns(4)
    with cols_summary[0]: render_web_kpi_card("Visits", str(chw_summary_kpis.get("visits_count", 0)), icon="👥", help_text="Total unique patients visited.")
    # Example status logic for high priority follow-ups
    high_prio_count = chw_summary_kpis.get("high_ai_prio_followups_count", 0)
    high_prio_status = "ACCEPTABLE" if high_prio_count <= 2 else ("MODERATE_CONCERN" if high_prio_count <= 5 else "HIGH_CONCERN")
    with cols_summary[1]: render_web_kpi_card("High Prio Follow-ups", str(high_prio_count), icon="🎯", status_level=high_prio_status, help_text="Patients needing urgent follow-up based on AI score.")
    
    crit_spo2_count = chw_summary_kpis.get("critical_spo2_cases_identified_count", 0)
    crit_spo2_status = "HIGH_CONCERN" if crit_spo2_count > 0 else "ACCEPTABLE"
    with cols_summary[2]: render_web_kpi_card("Crit. SpO2 Cases", str(crit_spo2_count), icon="💨", status_level=crit_spo2_status, help_text=f"Patients with SpO2 < {app_config.ALERT_SPO2_CRITICAL_LOW_PCT}%.")
    
    high_fever_count = chw_summary_kpis.get("high_fever_cases_identified_count", 0) # Using the more specific key
    high_fever_status = "HIGH_CONCERN" if high_fever_count > 0 else "ACCEPTABLE"
    with cols_summary[3]: render_web_kpi_card("High Fever Cases", str(high_fever_count), icon="🔥", status_level=high_fever_status, help_text=f"Patients with temp ≥ {app_config.ALERT_BODY_TEMP_HIGH_FEVER_C}°C.")
else:
    st.markdown("_No CHW activity data found for the selected filters to display daily performance snapshot._")
st.divider()

# --- Section 2: Key Alerts & Actionable Task Summary ---
st.header("🚦 Key Alerts & Actionable Tasks Overview")

# Generate Alerts (focus on CRITICAL/WARNING for supervisor)
patient_alerts_list = generate_chw_patient_alerts_from_data(
    patient_encounter_data_df=daily_activity_df, # Use daily data which might trigger alerts
    for_date=selected_daily_date,
    chw_zone_context_str=zone_to_filter or "All Supervised Zones",
    max_alerts_to_return=10 # Supervisor sees a limited number of top alerts
)
if patient_alerts_list:
    st.subheader(f"Priority Patient Alerts (Requiring Review/Action):")
    critical_alert_found = False
    for alert_item in patient_alerts_list:
        if alert_item.get("alert_level") == "CRITICAL":
            critical_alert_found = True
            render_web_traffic_light_indicator(
                message=f"Pt. {alert_item.get('patient_id', 'N/A')}: {alert_item.get('primary_reason', 'Unknown Reason')}",
                status_level="HIGH_RISK", # Map CRITICAL to HIGH_RISK style
                details_text=f"{alert_item.get('brief_details','N/A')} | Context: {alert_item.get('context_info','N/A')} | Action: {alert_item.get('suggested_action_code','REVIEW')}"
            )
    if not critical_alert_found:
        st.info("No CRITICAL patient alerts identified from field data for this selection.")
    
    # Optionally, show WARNING alerts if space or if no CRITICAL ones
    warning_alerts_count = sum(1 for alert in patient_alerts_list if alert.get("alert_level") == "WARNING")
    if not critical_alert_found and warning_alerts_count > 0:
        st.markdown("###### Other Notable Warning Alerts:")
        for alert_item in patient_alerts_list:
            if alert_item.get("alert_level") == "WARNING":
                render_web_traffic_light_indicator(
                    message=f"Pt. {alert_item.get('patient_id', 'N/A')}: {alert_item.get('primary_reason', 'Unknown Reason')}",
                    status_level="MODERATE_RISK", # Map WARNING to MODERATE_RISK style
                    details_text=f"{alert_item.get('brief_details','N/A')} | Context: {alert_item.get('context_info','N/A')}"
                )
elif not daily_activity_df.empty : # Data exists but no alerts generated
    st.info("No specific patient alerts generated from field data for this selection.")
else: # No data to generate alerts from
    st.markdown("_No CHW activity data to generate alerts for this selection._")


# Generate Tasks (summary of high priority or overdue for team)
# Using daily_activity_df to show tasks generated *from today's events* or relevant to today's context.
# A true backlog might query a task database.
generated_tasks_list = generate_chw_prioritized_tasks(
    source_patient_data_df=daily_activity_df,
    for_date=selected_daily_date,
    chw_id_context=chw_to_filter, # Pass selected CHW or None for team
    zone_context_str=zone_to_filter or "All Supervised Zones",
    max_tasks_to_return_for_summary=10
)
if generated_tasks_list:
    st.subheader(f"Top Priority Tasks from Today's Activities:")
    tasks_to_display_df = pd.DataFrame(generated_tasks_list)
    # Select and reorder columns for supervisor display
    display_task_cols = ['patient_id', 'task_description', 'priority_score', 'due_date', 'status', 'key_patient_context', 'assigned_chw_id']
    actual_display_task_cols = [col for col in display_task_cols if col in tasks_to_display_df.columns]
    
    st.dataframe(
        tasks_to_display_df[actual_display_task_cols], 
        use_container_width=True, 
        height=min(350, len(tasks_to_display_df)*40 + 40), # Dynamic height with max
        hide_index=True
    )
elif not daily_activity_df.empty:
    st.info("No new high-priority tasks identified from today's activities based on current filters.")
else:
    st.markdown("_No CHW activity data to generate tasks for this selection._")
st.divider()


# --- Section 3: Local Epi Signals from Field ---
st.header("🔬 Local Epi Signals Watch")
if not daily_activity_df.empty:
    local_epi_signals = extract_chw_local_epi_signals(
        chw_daily_encounter_df=daily_activity_df,
        pre_calculated_chw_kpis=pre_calculated_kpis, # From data loading step
        for_date=selected_daily_date,
        chw_zone_context=zone_to_filter or "All Supervised Zones"
    )
    
    cols_epi = st.columns(3)
    with cols_epi[0]: render_web_kpi_card(title="Symptomatic (Key Cond.)", value=str(local_epi_signals.get("symptomatic_patients_key_conditions_count", 0)), icon="🤒", units="cases today", help_text="Patients with key conditions reporting symptoms.")
    with cols_epi[1]: render_web_kpi_card(title="New Malaria Cases", value=str(local_epi_signals.get("newly_identified_malaria_patients_count",0)), icon="🦟", units="cases today", help_text="Malaria cases identified from today's encounters.")
    with cols_epi[2]: render_web_kpi_card(title="Pending TB Contacts", value=str(local_epi_signals.get("pending_tb_contact_tracing_tasks_count",0)), icon="👥", units="to trace", help_text="Number of TB contacts needing follow-up.")
    
    detected_symptom_clusters = local_epi_signals.get("detected_symptom_clusters", [])
    if detected_symptom_clusters:
        st.markdown("###### Detected Symptom Clusters Today:")
        for cluster_item in detected_symptom_clusters:
            st.warning(f"⚠️ **{cluster_item.get('symptoms_pattern','Unknown Pattern')}**: {cluster_item.get('patient_count','N/A')} cases in {cluster_item.get('location_hint', 'area')}")
    elif 'patient_reported_symptoms' in daily_activity_df.columns: # If symptoms column exists but no clusters
        st.info("No significant symptom clusters detected from today's CHW data based on current criteria.")
else:
    st.markdown("_No CHW activity data to derive local epi signals for this selection._")
st.divider()

# --- Section 4: CHW Team Activity Trends (Periodic View) ---
st.header("📈 CHW Team Activity Trends")
trend_period_display_str = f"{selected_trend_start_date.strftime('%d %b %Y')} - {selected_trend_end_date.strftime('%d %b %Y')}"
filter_context_trend = ""
if chw_to_filter: filter_context_trend += f" for CHW {chw_to_filter}"
if zone_to_filter: filter_context_trend += f" in Zone {zone_to_filter}"
st.markdown(f"Displaying trends from **{trend_period_display_str}**{filter_context_trend if filter_context_trend else ' across All CHWs/Zones'}.")

if not period_activity_df.empty:
    chw_activity_trends = calculate_chw_activity_trends(
        chw_historical_health_df=period_activity_df, # Already filtered for period and optionally CHW/Zone
        trend_start_date_input=selected_trend_start_date, 
        trend_end_date_input=selected_trend_end_date,
        zone_filter=None, # Data is already zone-filtered if a zone was selected by supervisor
        time_period_aggregation='D' # Daily trends suitable for supervisor dashboard
    )
    
    cols_trends = st.columns(2)
    with cols_trends[0]:
        visits_trend = chw_activity_trends.get("patient_visits_trend")
        if isinstance(visits_trend, pd.Series) and not visits_trend.empty:
            st.plotly_chart(plot_annotated_line_chart_web(
                visits_trend, chart_title="Daily Patient Visits (Trend)",
                y_axis_label="# Patients", y_is_count=True
            ), use_container_width=True)
        else: st.caption("No patient visit trend data for this selection.")
        
    with cols_trends[1]:
        prio_followups_trend = chw_activity_trends.get("high_priority_followups_trend")
        if isinstance(prio_followups_trend, pd.Series) and not prio_followups_trend.empty:
            st.plotly_chart(plot_annotated_line_chart_web(
                prio_followups_trend, chart_title="Daily High Prio. Follow-ups (Trend)",
                y_axis_label="# Follow-ups", y_is_count=True
            ), use_container_width=True)
        else: st.caption("No high-priority follow-up trend data for this selection.")
else:
    st.markdown("_No historical data available for the selected filters to display activity trends._")

logger.info(f"CHW Supervisor View page generated for date: {selected_daily_date}, CHW: {chw_to_filter or 'All'}, Zone: {zone_to_filter or 'All'}.")
