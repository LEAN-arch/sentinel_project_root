# sentinel_project_root/test/pages/1_chw_dashboard.py
# Redesigned as "CHW Supervisor Operations View" for "Sentinel Health Co-Pilot"

import streamlit as st
import pandas as pd
import numpy as np
import os
import logging
from datetime import date, timedelta

# --- Sentinel System Imports ---
from config import app_config # Uses the new, redesigned app_config

# Core data loading for simulation (Supervisor would access aggregated/synced data)
from utils.core_data_processing import load_health_records

# Refactored CHW component data preparation functions
# Note: Adjust path if you placed "_sentinel" components differently.
# This assumes: test/pages/chw_components_sentinel/your_module.py
from pages.chw_components_sentinel.summary_metrics_calculator import calculate_chw_daily_summary_metrics
from pages.chw_components_sentinel.alert_generator import generate_chw_patient_alerts_from_data
from pages.chw_components_sentinel.epi_signal_extractor import extract_chw_local_epi_signals
from pages.chw_components_sentinel.task_processor import generate_chw_prioritized_tasks # Get full list for supervisor
from pages.chw_components_sentinel.activity_trend_calculator import calculate_chw_activity_trends

# Refactored UI helpers for web reports (Tier 1 Hub / Tier 2 Facility Node views)
from utils.ui_visualization_helpers import (
    render_web_kpi_card,
    render_web_traffic_light_indicator,
    plot_annotated_line_chart_web,
    plot_bar_chart_web # May not be used extensively for supervisor but available
)

# --- Page Configuration (Specific to this Supervisor View) ---
st.set_page_config(
    page_title=f"CHW Supervisor View - {app_config.APP_NAME}",
    layout="wide",
    initial_sidebar_state="expanded"
)

logger = logging.getLogger(__name__)
# CSS: Assuming loaded by app_home.py or use @st.cache_resource if needed here for standalone runs

# --- Data Loading Logic for Supervisor View (Simulation) ---
@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS_WEB_REPORTS, show_spinner="Loading CHW operational data for supervisor...")
def get_chw_supervisor_dashboard_data(
    view_date: date,
    trend_start_date: date,
    trend_end_date: date,
    # Future filters: supervisor_id, team_id, specific_chw_id, zone_filter
    selected_chw_id: Optional[str] = None, # If supervisor wants to drill down
    selected_zone_id: Optional[str] = None # If supervisor covers multiple zones
):
    """
    Simulates fetching and preparing data needed for the CHW Supervisor View.
    In a real system, this queries an aggregated database at a Hub/Facility Node,
    containing synced data from CHW PEDs. AI scores are assumed to be part of synced data.
    """
    # For simulation, load all health records (these are already AI-enriched by earlier a hypothetical process)
    # In production, this data source for supervisor view would be from a Tier 1/2 database
    # of *synced and processed* data from CHW PEDs.
    health_df_all_synced = load_health_records(file_path=app_config.HEALTH_RECORDS_CSV, source_context="CHWSupervisorDataSim")
    
    if health_df_all_synced.empty:
        logger.error("CHW Supervisor View: Base health records (simulating synced data) failed to load.")
        return pd.DataFrame(), pd.DataFrame(), {} # Empty daily_df, period_df, pre_calc_kpis

    # Ensure 'encounter_date' is datetime
    if 'encounter_date' not in health_df_all_synced.columns or \
       not pd.api.types.is_datetime64_any_dtype(health_df_all_synced['encounter_date']):
        health_df_all_synced['encounter_date'] = pd.to_datetime(health_df_all_synced['encounter_date'], errors='coerce')
    health_df_all_synced.dropna(subset=['encounter_date'], inplace=True)

    # Filter data based on supervisor's scope (date, CHW, Zone)
    # 1. Daily Snapshot Data
    daily_df_sup = health_df_all_synced[health_df_all_synced['encounter_date'].dt.date == view_date].copy()
    if selected_chw_id and 'chw_id' in daily_df_sup.columns: # Assuming 'chw_id' column exists
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
        
    # For this simulation, pre-calculated KPIs for the daily summary are minimal.
    # `calculate_chw_daily_summary_metrics` will derive most from `daily_df_sup`.
    # In a real system, some daily roll-ups for the CHW/team might be pre-calculated.
    pre_calculated_daily_kpis_for_supervisor = {} # e.g., if `get_chw_summary` from core_data ran for each CHW daily.

    return daily_df_sup, period_df_sup, pre_calculated_daily_kpis_for_supervisor

# --- Page Title & Introduction for CHW Supervisor ---
st.title("🧑‍🏫 CHW Supervisor Operations View")
st.markdown(f"**Team Performance Monitoring, Alert Triage, and Field Activity Oversight for {app_config.APP_NAME}**")
st.markdown("---")

# --- Sidebar Filters for Supervisor ---
if os.path.exists(app_config.APP_LOGO_SMALL):
    st.sidebar.image(app_config.APP_LOGO_SMALL, width=180)
st.sidebar.header("🗓️ Supervisor View Filters")

# TODO: Populate these from available CHW IDs and Zones in the data source
available_chw_ids_for_supervisor = ["All CHWs", "CHW01", "CHW02", "CHW03"] # Placeholder
selected_chw_supervisor_filter = st.sidebar.selectbox(
    "Filter by CHW ID:", options=available_chw_ids_for_supervisor, key="supervisor_chw_id_filter_v1"
)
actual_chw_filter = None if selected_chw_supervisor_filter == "All CHWs" else selected_chw_supervisor_filter

available_zones_for_supervisor = ["All Zones", "ZoneA", "ZoneB", "ZoneC"] # Placeholder
selected_zone_supervisor_filter = st.sidebar.selectbox(
    "Filter by Zone:", options=available_zones_for_supervisor, key="supervisor_zone_filter_v1"
)
actual_zone_filter = None if selected_zone_supervisor_filter == "All Zones" else selected_zone_supervisor_filter


# Date selection for "Daily Snapshot" section
# This affects the KPIs, alerts, epi-signals, and tasks for *that specific day*.
min_date_sup_snapshot = date.today() - timedelta(days=app_config.WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND * 2) # Example 60 days back for daily view
max_date_sup_snapshot = date.today()
selected_daily_view_date_sup = st.sidebar.date_input(
    "View Daily Activity For:",
    value=max_date_sup_snapshot, min_value=min_date_sup_snapshot, max_value=max_date_sup_snapshot,
    key="supervisor_daily_snapshot_date_v1"
)

# Date range for "Periodic Trends" section (defaults set based on daily view selection)
trend_end_date_sup_default = selected_daily_view_date_sup
trend_start_date_sup_default = trend_end_date_sup_default - timedelta(days=app_config.WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND -1)
if trend_start_date_sup_default < min_date_sup_snapshot : trend_start_date_sup_default = min_date_sup_snapshot

selected_trend_start_sup, selected_trend_end_sup = st.sidebar.date_input(
    "Select Date Range for Periodic Trends:",
    value=[trend_start_date_sup_default, trend_end_date_sup_default],
    min_value=min_date_sup_snapshot, max_value=max_date_sup_snapshot,
    key="supervisor_periodic_trend_range_v1"
)
if selected_trend_start_sup > selected_trend_end_sup:
    st.sidebar.error("Trend start date must be before end date.")
    selected_trend_start_sup = selected_trend_end_sup

# Load/Filter data based on selections
daily_data_for_view, period_data_for_view, pre_calc_kpis_for_view = get_chw_supervisor_dashboard_data(
    view_date=selected_daily_view_date_sup,
    trend_start_date=selected_trend_start_sup, # Pass trend dates for loading full period data once
    trend_end_date=selected_trend_end_sup,
    selected_chw_id=actual_chw_filter,
    selected_zone_id=actual_zone_filter
)

# Display context for the selected filters
filter_context_str = f"For: {selected_daily_view_date_sup.strftime('%d %b %Y')}"
if actual_chw_filter: filter_context_str += f" | CHW: {actual_chw_filter}"
if actual_zone_filter: filter_context_str += f" | Zone: {actual_zone_filter}"
st.info(filter_context_str)


# --- Section 1: Daily Performance Snapshot ---
st.header(f"📊 Daily Performance Snapshot")
if not daily_data_for_view.empty:
    chw_summary_metrics = calculate_chw_daily_summary_metrics(
        chw_daily_kpi_input_data=pre_calc_kpis_for_view, # This may have some aggregated data from the node
        chw_daily_encounter_df=daily_data_for_view, # Raw daily data for more specific calculations
        for_date=selected_daily_view_date_sup
    )
    
    cols_summary_kpi_sup = st.columns(4)
    with cols_summary_kpi_sup[0]: render_web_kpi_card("Visits", str(chw_summary_metrics.get("visits_count", 0)), icon="👥")
    with cols_summary_kpi_sup[1]: render_web_kpi_card("High Prio Follow-ups", str(chw_summary_metrics.get("high_ai_prio_followups_count", 0)), icon="🎯", status_level="MODERATE_CONCERN" if chw_summary_metrics.get("high_ai_prio_followups_count",0) > (app_config.TARGET_CLINIC_PATIENT_THROUGHPUT_MIN_PER_HOUR/2) else "ACCEPTABLE") # Example: if more than 2-3 high prio
    with cols_summary_kpi_sup[2]: render_web_kpi_card("Crit. SpO2 Cases", str(chw_summary_metrics.get("critical_spo2_cases_identified_count", 0)), icon="💨", status_level="HIGH_CONCERN" if chw_summary_metrics.get("critical_spo2_cases_identified_count",0) > 0 else "ACCEPTABLE")
    with cols_summary_kpi_sup[3]: render_web_kpi_card("High Fever Cases", str(chw_summary_metrics.get("patients_high_fever_today", 0)), icon="🔥", status_level="HIGH_CONCERN" if chw_summary_metrics.get("patients_high_fever_today",0) > 0 else "ACCEPTABLE") # Note: key updated in get_chw_summary was patients_high_fever_today
    
    # Add other relevant supervisor KPIs if needed, e.g., on CHW's own reported status (if synced)
else:
    st.markdown("_No CHW activity data found for the selected filters to display daily performance snapshot._")
st.markdown("---")

# --- Section 2: Key Alerts & Actionable Task Summary ---
st.header("🚦 Key Alerts & Actionable Tasks Overview")
# Data from `patient_alerts_tasks_df` which is `daily_data_for_view` in this supervisor context.
# Supervisor sees a summary of *actionable* items needing their attention or oversight.

# Generate Alerts (mostly for CRITICAL ones for supervisor)
alerts_list_sup = generate_chw_patient_alerts_from_data(
    patient_alerts_tasks_df=daily_data_for_view,
    chw_daily_encounter_df=daily_data_for_view, # Can be same if daily_data_for_view is already well-scoped
    for_date=selected_daily_view_date_sup,
    chw_zone_context=actual_zone_filter or "All Supervised Zones",
    max_alerts_to_return=7 # Supervisor dashboard shows a limited number of top alerts
)
if alerts_list_sup:
    st.subheader(f"Priority Patient Alerts (Requiring Review/Action):")
    critical_alerts_count = 0
    for alert in alerts_list_sup:
        if alert.get("alert_level") == "CRITICAL":
            critical_alerts_count += 1
            render_web_traffic_light_indicator(
                message=f"Pt. {alert['patient_id']}: {alert['primary_reason']}",
                status_level="HIGH_RISK", # Map CRITICAL to HIGH_RISK for styling
                details_text=f"{alert['brief_details']} | Context: {alert.get('context_info','N/A')} | Action: {alert.get('suggested_action_code','REVIEW')}"
            )
    if critical_alerts_count == 0:
        st.info("No CRITICAL patient alerts identified from field data for this selection.")
    # Optionally show WARNING alerts if no CRITICAL ones.
    elif len(alerts_list_sup) > critical_alerts_count :
        st.markdown("###### Other Notable Alerts:")
        for alert in alerts_list_sup:
            if alert.get("alert_level") == "WARNING":
                render_web_traffic_light_indicator(
                    message=f"Pt. {alert['patient_id']}: {alert['primary_reason']}",
                    status_level="MODERATE_RISK", # Map WARNING to MODERATE_RISK
                    details_text=f"{alert['brief_details']} | Context: {alert.get('context_info','N/A')}"
                )
else:
    st.info("No specific patient alerts generated from field data for this selection.")

# Generate Tasks (show summary of high priority or overdue for team)
# This should probably be more than just today's tasks if it's a backlog view for supervisor.
# For simplicity, use daily_data_for_view to show tasks generated *from today's events*.
tasks_list_sup = generate_chw_prioritized_tasks(
    patient_alerts_tasks_df=daily_data_for_view, # Use daily encounters that might trigger tasks
    chw_daily_encounter_df=daily_data_for_view,
    for_date=selected_daily_view_date_sup,
    chw_zone_context=actual_zone_filter or "All Supervised Zones",
    max_tasks_to_return=10 # Show top tasks for supervisor
)
if tasks_list_sup:
    st.subheader(f"Top Priority Tasks Generated from Today's Activities:")
    tasks_df_for_display = pd.DataFrame(tasks_list_sup)
    st.dataframe(tasks_df_for_display[[
        'patient_id', 'task_description', 'priority_score', 'due_date', 'status', 'key_patient_context'
    ]], use_container_width=True, height=min(300, len(tasks_df_for_display)*45 + 45))
else:
    st.info("No new high-priority tasks identified from today's activities based on current filters.")
st.markdown("---")


# --- Section 3: Local Epi Signals from Field ---
st.header("🔬 Local Epi Signals Watch")
if not daily_data_for_view.empty:
    epi_signals_output_sup = extract_chw_local_epi_signals(
        chw_daily_encounter_df=daily_data_for_view,
        pre_calculated_chw_kpis=pre_calc_kpis_for_view,
        for_date=selected_daily_view_date_sup,
        chw_zone_context=actual_zone_filter or "All Supervised Zones"
    )
    # Display key signals as simple text or small cards
    cols_epi_sup = st.columns(3)
    with cols_epi_sup[0]:
        render_web_kpi_card(
            title="Symptomatic (Key Cond.)",
            value=str(epi_signals_output_sup.get("new_symptomatic_cases_key_conditions_count", 0)),
            icon="🤒", units="cases today"
        )
    with cols_epi_sup[1]:
        render_web_kpi_card(
            title="New Malaria Cases",
            value=str(epi_signals_output_sup.get("new_malaria_cases_today_count",0)),
            icon="🦟", units="cases today"
        )
    with cols_epi_sup[2]:
        render_web_kpi_card(
            title="Pending TB Contacts",
            value=str(epi_signals_output_sup.get("pending_tb_contact_traces_count",0)),
            icon="👥", units="to trace"
        )
    
    if epi_signals_output_sup.get("reported_symptom_cluster_alerts"):
        st.markdown("###### Detected Symptom Clusters Today:")
        for cluster_alert in epi_signals_output_sup["reported_symptom_cluster_alerts"]:
            st.warning(f"⚠️ **{cluster_alert.get('symptoms')}**: {cluster_alert.get('count')} cases in {cluster_alert.get('location_hint', 'area')}")
else:
    st.markdown("_No CHW activity data to derive local epi signals for this selection._")
st.markdown("---")

# --- Section 4: CHW Team Activity Trends (Periodic View) ---
st.header("📈 CHW Team Activity Trends")
st.markdown(f"Displaying trends from **{selected_trend_start_sup.strftime('%d %b %Y')}** to **{selected_trend_end_sup.strftime('%d %b %Y')}** "
            f"{('for CHW ' + actual_chw_filter) if actual_chw_filter else ('for All CHWs')} "
            f"{('in Zone ' + actual_zone_filter) if actual_zone_filter else ('across All Zones')}.")

if not period_data_for_view.empty:
    chw_activity_trends_data = calculate_chw_activity_trends(
        chw_historical_health_df=period_data_for_view, # This is already filtered for the period
        trend_start_date=selected_trend_start_sup, # Used by internal filter if needed, and for context
        trend_end_date=selected_trend_end_sup,
        zone_filter=None, # period_data_for_view is already zone-filtered if actual_zone_filter was set
        time_period_agg='D' # Daily trends suitable for supervisor dashboard
    )
    cols_trends_page_sup = st.columns(2)
    with cols_trends_page_sup[0]:
        visits_trend_data_sup = chw_activity_trends_data.get("patient_visits_trend")
        if visits_trend_data_sup is not None and not visits_trend_data_sup.empty:
            st.plotly_chart(plot_annotated_line_chart_web(
                visits_trend_data_sup.squeeze(), chart_title="Daily Patient Visits (Trend)",
                y_axis_label="# Patients", y_axis_is_count=True
            ), use_container_width=True)
        else: st.caption("No patient visit trend data.")
    with cols_trends_page_sup[1]:
        prio_trend_data_sup = chw_activity_trends_data.get("high_priority_followups_trend")
        if prio_trend_data_sup is not None and not prio_trend_data_sup.empty:
            st.plotly_chart(plot_annotated_line_chart_web(
                prio_trend_data_sup.squeeze(), chart_title="Daily High Prio. Follow-ups (Trend)",
                y_axis_label="# Follow-ups", y_axis_is_count=True
            ), use_container_width=True)
        else: st.caption("No high-priority follow-up trend data.")
else:
    st.markdown("_No historical data available for selected filters to display trends._")

logger.info(f"CHW Supervisor View page generated for date: {selected_daily_view_date_sup}, user: Supervisor.")
