# sentinel_project_root/test/pages/1_chw_dashboard.py
# Redesigned as "CHW Supervisor Operations View" for "Sentinel Health Co-Pilot"
# This page simulates what a CHW Supervisor or Hub Coordinator might see on a
# web interface (Tablet/Laptop at Tier 1 Hub or Tier 2 Facility Node).
# It provides an overview of CHW activities, escalated alerts, and team performance.

import streamlit as st
import pandas as pd
import numpy as np
import os
import logging
from datetime import date, timedelta, datetime # Ensure datetime is imported

# --- Sentinel System Imports ---
# Attempt to import assuming 'test/' directory is effectively the app root for Streamlit page execution
# This structure requires __init__.py files in 'config', 'utils', 'pages', and 'pages/chw_components_sentinel'
try:
    from config import app_config
    from utils.core_data_processing import load_health_records # Used by data loader for this page
    from pages.chw_components_sentinel.summary_metrics_calculator import calculate_chw_daily_summary_metrics
    from pages.chw_components_sentinel.alert_generator import generate_chw_patient_alerts_from_data
    from pages.chw_components_sentinel.epi_signal_extractor import extract_chw_local_epi_signals
    from pages.chw_components_sentinel.task_processor import generate_chw_prioritized_tasks
    from pages.chw_components_sentinel.activity_trend_calculator import calculate_chw_activity_trends
    from utils.ui_visualization_helpers import (
        render_web_kpi_card,
        render_web_traffic_light_indicator,
        plot_annotated_line_chart_web,
        plot_bar_chart_web # Added in case epi signals use it
    )
except ImportError as e:
    # Fallback path adjustment for local development or specific execution contexts
    # This tries to add the 'test' directory (assumed parent of 'pages') to sys.path
    import sys
    current_script_path = os.path.dirname(os.path.abspath(__file__))
    project_app_root_dir = os.path.abspath(os.path.join(current_script_path, os.pardir)) # up to 'test/'
    if project_app_root_dir not in sys.path:
        sys.path.insert(0, project_app_root_dir)
    
    # Retry imports after path adjustment
    try:
        from config import app_config
        from utils.core_data_processing import load_health_records
        from pages.chw_components_sentinel.summary_metrics_calculator import calculate_chw_daily_summary_metrics
        from pages.chw_components_sentinel.alert_generator import generate_chw_patient_alerts_from_data
        from pages.chw_components_sentinel.epi_signal_extractor import extract_chw_local_epi_signals
        from pages.chw_components_sentinel.task_processor import generate_chw_prioritized_tasks
        from pages.chw_components_sentinel.activity_trend_calculator import calculate_chw_activity_trends
        from utils.ui_visualization_helpers import (
            render_web_kpi_card, render_web_traffic_light_indicator,
            plot_annotated_line_chart_web, plot_bar_chart_web
        )
        logging.warning(f"CHW Supervisor View: Used fallback sys.path adjustment for imports. Error was: {e}")
    except ImportError as final_e: # If still fails, something fundamental is wrong
        logging.critical(f"CRITICAL IMPORT ERROR in 1_chw_dashboard.py even after path adjustment: {final_e}. Application will not function correctly.")
        st.error(f"Application Critical Error: Could not load core modules for CHW Supervisor View. Details: {final_e}. Please check your project structure and PYTHONPATH.")
        # Define stubs for critical functions to prevent complete crash and allow some UI to render with error messages.
        # This part is for graceful degradation during development if some modules are missing temporarily.
        class AppConfigFallbackGlobal: LOG_LEVEL="ERROR"; APP_NAME="Sentinel (Import Error)"; CACHE_TTL_SECONDS_WEB_REPORTS=60; WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND=30; APP_LOGO_SMALL="None"
        app_config = AppConfigFallbackGlobal()
        def calculate_chw_daily_summary_metrics(*args, **kwargs): return {"notes":["Error: summary_metrics_calculator module not loaded"]}
        def generate_chw_patient_alerts_from_data(*args, **kwargs): return [{"notes":["Error: alert_generator module not loaded"]}]
        def extract_chw_local_epi_signals(*args, **kwargs): return {"notes":["Error: epi_signal_extractor module not loaded"]}
        def generate_chw_prioritized_tasks(*args, **kwargs): return [{"notes":["Error: task_processor module not loaded"]}]
        def calculate_chw_activity_trends(*args, **kwargs): return {"notes":["Error: activity_trend_calculator module not loaded"]}
        def render_web_kpi_card(title, value_str, **kwargs): st.error(f"KPI Card Display Error for '{title}'. UI Helper missing.")
        def render_web_traffic_light_indicator(message, **kwargs): st.error(f"Traffic Light Display Error for '{message}'. UI Helper missing.")
        def plot_annotated_line_chart_web(data, title, **kwargs): st.error(f"Line Chart Display Error for '{title}'. UI Helper missing.")
        def plot_bar_chart_web(data, title, **kwargs): st.error(f"Bar Chart Display Error for '{title}'. UI Helper missing.")


# --- Page Configuration (Specific to this Supervisor View) ---
st.set_page_config(
    page_title=f"CHW Supervisor Console - {app_config.APP_NAME}",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize logger for this page
logger = logging.getLogger(__name__)
# CSS is assumed to be loaded by app_home.py (or a @st.cache_resource call here if this page is the entry).

# --- Data Loading Function for Supervisor View (Simulation) ---
@st.cache_data(
    ttl=app_config.CACHE_TTL_SECONDS_WEB_REPORTS, # Using refined config key
    show_spinner="Loading CHW team operational data..."
)
def get_supervisor_view_data_final( # Unique name for this page's loader
    view_date: date,
    historical_start_date: date,
    historical_end_date: date,
    chw_id_for_filter: Optional[str] = None,
    zone_id_for_filter: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Simulates fetching and preparing data for the CHW Supervisor View.
    - `view_date` is for daily snapshot sections.
    - `historical_start_date` & `historical_end_date` define the window for data used in trends.
    """
    logger.info(f"CHW Sup. View: Loading data for snapshot={view_date}, trends={historical_start_date}-{historical_end_date}, CHW={chw_id_for_filter}, Zone={zone_id_for_filter}")
    
    # For simulation, this loads ALL health records and filters.
    # In a real Tier 1/2 system, it would query a DB already containing CHW-specific, synced, AI-enriched data.
    health_df_all = load_health_records(file_path=app_config.HEALTH_RECORDS_CSV, source_context="CHWSupView/LoadAll")
    if health_df_all.empty:
        logger.error("CHW Sup. View: Base health data load failed or returned empty.")
        return pd.DataFrame(), pd.DataFrame(), {}

    # Basic data cleaning and date object creation
    if 'encounter_date' not in health_df_all.columns:
        logger.error("CHW Sup. View: 'encounter_date' crucial column missing from health_df_all.")
        return pd.DataFrame(), pd.DataFrame(), {} # Cannot proceed
    health_df_all['encounter_date'] = pd.to_datetime(health_df_all['encounter_date'], errors='coerce')
    health_df_all.dropna(subset=['encounter_date'], inplace=True) # Remove rows with invalid dates
    health_df_all['encounter_date_only'] = health_df_all['encounter_date'].dt.date

    # 1. Data for Daily Snapshot section (for `view_date`)
    daily_df_result = health_df_all[health_df_all['encounter_date_only'] == view_date].copy()
    # Apply CHW ID filter if 'chw_id' column exists (from synced PED data)
    if chw_id_for_filter and 'chw_id' in daily_df_result.columns:
        daily_df_result = daily_df_result[daily_df_result['chw_id'] == chw_id_for_filter]
    if zone_id_for_filter and 'zone_id' in daily_df_result.columns:
        daily_df_result = daily_df_result[daily_df_result['zone_id'] == zone_id_for_filter]

    # 2. Data for Periodic Trends section (for `historical_start_date` to `historical_end_date`)
    # This broader dataset is used by calculate_chw_activity_trends which then internally filters to exact trend dates
    historical_df_for_trends_result = health_df_all[
        (health_df_all['encounter_date_only'] >= historical_start_date) &
        (health_df_all['encounter_date_only'] <= historical_end_date)
    ].copy()
    if chw_id_for_filter and 'chw_id' in historical_df_for_trends_result.columns:
        historical_df_for_trends_result = historical_df_for_trends_result[historical_df_for_trends_result['chw_id'] == chw_id_for_filter]
    if zone_id_for_filter and 'zone_id' in historical_df_for_trends_result.columns:
        historical_df_for_trends_result = historical_df_for_trends_result[historical_df_for_trends_result['zone_id'] == zone_id_for_filter]
        
    # This would contain pre-aggregated daily data per CHW/team from a Tier1/2 system, if available.
    # For this simulation, summary_metrics_calculator will derive most from daily_df_result.
    pre_aggregated_daily_kpis_placeholder = {}
    
    logger.info(f"CHW Sup. View Data: Daily snapshot records: {len(daily_df_result)}, Hist. for trends: {len(historical_df_for_trends_result)}")
    return daily_df_result, historical_df_for_trends_result, pre_aggregated_daily_kpis_placeholder


# --- Page Title & Sidebar Filters ---
st.title(f"🧑‍🏫 {app_config.APP_NAME} - CHW Supervisor Operations View")
st.markdown(f"**Team Activity Monitoring, Alert Triage, and Field Support Coordination**")
st.markdown("---")

# Sidebar Setup
if os.path.exists(app_config.APP_LOGO_SMALL): st.sidebar.image(app_config.APP_LOGO_SMALL, width=150)
st.sidebar.header("📊 View Filters")

# (Mock data for selectors - in real app, these come from available CHW/Zone data)
MOCK_CHW_IDS_FOR_PAGE = ["All CHWs"] + [f"CHW{i:03d}" for i in range(1, 6)]
selected_chw_id_page = st.sidebar.selectbox("Filter by CHW:", options=MOCK_CHW_IDS_FOR_PAGE, key="sup_page_chw_filter_final")
chw_filter_applied = None if selected_chw_id_page == "All CHWs" else selected_chw_id_page

MOCK_ZONES_FOR_PAGE = ["All Zones"] + [f"Zone{chr(65+i)}" for i in range(4)] # ZoneA-D
selected_zone_id_page = st.sidebar.selectbox("Filter by Operational Zone:", options=MOCK_ZONES_FOR_PAGE, key="sup_page_zone_filter_final")
zone_filter_applied = None if selected_zone_id_page == "All Zones" else selected_zone_id_page

# Date for Daily Snapshot section
MAX_DAYS_LOOKBACK_SNAPSHOT = 90 # Supervisor can look back up to 90 days for daily snapshot
min_snapshot_dt_page = date.today() - timedelta(days=MAX_DAYS_LOOKBACK_SNAPSHOT)
max_snapshot_dt_page = date.today()
snapshot_date_selected_page = st.sidebar.date_input(
    "Daily Snapshot For Date:", value=max_snapshot_dt_page,
    min_value=min_snapshot_dt_page, max_value=max_snapshot_dt_page,
    key="sup_page_snapshot_date_final"
)

# Date Range for Periodic Trends section
# Trends end on the snapshot date by default for context. Max range configurable.
MAX_DAYS_LOOKBACK_TRENDS = 180 # Max period for trend view
min_trend_dt_page = date.today() - timedelta(days=MAX_DAYS_LOOKBACK_TRENDS)

trend_range_end_page = snapshot_date_selected_page # Align trend end with snapshot date for coherence
trend_range_start_page = trend_range_end_page - timedelta(days=app_config.WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND - 1) # Default ~30 days trend
if trend_range_start_page < min_trend_dt_page: trend_range_start_page = min_trend_dt_page

selected_trend_start_date_page, selected_trend_end_date_page = st.sidebar.date_input(
    "Periodic Trends Range:", value=[trend_range_start_page, trend_range_end_page],
    min_value=min_trend_dt_page, max_value=max_snapshot_dt_page, # Cannot pick future trend end beyond snapshot
    key="sup_page_trend_range_final"
)
if selected_trend_start_date_page > selected_trend_end_date_page:
    st.sidebar.error("Trend range: Start date must be before end date.")
    selected_trend_start_date_page = selected_trend_end_date_page


# --- Load Data Based on Filters ---
daily_data_for_supervisor, period_data_for_supervisor, pre_calc_kpis_for_supervisor = get_chw_supervisor_view_data_final(
    view_date=snapshot_date_selected_page,
    historical_start_date=selected_trend_start_date_page, # Pass the full range needed for trends
    historical_end_date=selected_trend_end_date_page,
    chw_id_filter=chw_filter_applied,
    zone_filter=zone_filter_applied
)

# Display current filter context clearly
filter_context_str_page = f"Snapshot for: **{snapshot_date_selected_page.strftime('%A, %d %b %Y')}**"
if chw_filter_applied: filter_context_str_page += f" | CHW: **{chw_filter_applied}**"
if zone_filter_applied: filter_context_str_page += f" | Zone: **{zone_filter_applied}**"
st.info(filter_context_str_page)

if daily_data_for_supervisor.empty and period_data_for_supervisor.empty: # Check both as trends use period data
    st.warning("No CHW field activity data found for the selected filters and date(s). Please adjust sidebar filters or verify data sync from PEDs for this CHW/Zone/Date combination.")
    st.stop() # Stop if no data at all for any section

# --- Section 1: Daily Performance Snapshot ---
st.header(f"📊 Daily Performance Summary ({snapshot_date_selected_page.strftime('%d %b %Y')})")
if not daily_data_for_supervisor.empty:
    chw_summary_metrics_result = calculate_chw_daily_summary_metrics(
        chw_daily_kpi_input_data=pre_calc_kpis_for_supervisor,
        chw_daily_encounter_df=daily_data_for_supervisor,
        for_date=snapshot_date_selected_page
    )
    
    kpi_cols_sup_page_s1 = st.columns(4) # Ensure you have enough KPIs or adjust columns
    kpi_items_s1 = [
        {"title":"Total Visits", "value_str":str(chw_summary_metrics_result.get("visits_count",0)), "icon":"👥", "units":"visits"},
        {"title":"High Prio. Follow-ups", "value_str":str(chw_summary_metrics_result.get("high_ai_prio_followups_count",0)), "icon":"🎯", "units":"tasks", "status_level":"MODERATE_CONCERN" if chw_summary_metrics_result.get("high_ai_prio_followups_count",0) > 2 else "ACCEPTABLE"}, # Threshold for concern by supervisor
        {"title":"Critical SpO2 Cases", "value_str":str(chw_summary_metrics_result.get("critical_spo2_cases_identified_count",0)), "icon":"💨", "units":"patients", "status_level":"HIGH_CONCERN" if chw_summary_metrics_result.get("critical_spo2_cases_identified_count",0) > 0 else "ACCEPTABLE"},
        {"title":"High Fever Cases", "value_str":str(chw_summary_metrics_result.get("fever_cases_identified_count",0)), "icon":"🔥", "units":"patients", "status_level":"MODERATE_CONCERN" if chw_summary_metrics_result.get("fever_cases_identified_count",0) > 0 else "ACCEPTABLE"}
    ]
    for i, kpi_item_s1 in enumerate(kpi_items_s1):
        with kpi_cols_sup_page_s1[i % 4]: render_web_kpi_card(**kpi_item_s1)
else:
    st.markdown("_No CHW field activity recorded for selected day/filters to display daily performance summary._")
st.markdown("---")


# --- Section 2: Key Alerts & Actionable Tasks for Supervisor Review ---
st.header(f"🚦 Key Alerts & Task Oversight ({snapshot_date_selected_page.strftime('%d %b %Y')})")
# Alerts generated from today's data for this CHW/Zone
supervisor_alerts_list = generate_chw_patient_alerts_from_data(
    patient_encounter_data_df=daily_data_for_supervisor,
    chw_daily_context_df=daily_data_for_supervisor, # Can be the same if daily_data is comprehensive for context
    for_date=snapshot_date_selected_page,
    chw_zone_context_str=zone_filter_applied or "All Supervised Zones",
    max_alerts_to_return=7 # Supervisor dashboard shows top alerts
)
if supervisor_alerts_list:
    st.subheader(f"Priority Patient Alerts Escalated/Identified:")
    critical_alerts_found = False
    for alert_item_sup in supervisor_alerts_list:
        if alert_item_sup.get("alert_level") == "CRITICAL":
            critical_alerts_found = True
            render_web_traffic_light_indicator(
                message=f"Pt. {alert_item_sup.get('patient_id','N/A')}: **{alert_item_sup.get('primary_reason','N/A')}**",
                status_level="HIGH_RISK",
                details_text=f"Details: {alert_item_sup.get('brief_details','')} | Context: {alert_item_sup.get('context_info','N/A')} | Action Code: {alert_item_sup.get('suggested_action_code','REVIEW')}"
            )
    if not critical_alerts_found and supervisor_alerts_list: # Show warnings if no criticals
        st.markdown("###### Other Notable Warnings:")
        for alert_item_sup_warn in supervisor_alerts_list:
            if alert_item_sup_warn.get("alert_level") == "WARNING":
                 render_web_traffic_light_indicator(
                    message=f"Pt. {alert_item_sup_warn.get('patient_id','N/A')}: {alert_item_sup_warn.get('primary_reason','N/A')}",
                    status_level="MODERATE_RISK",
                    details_text=f"{alert_item_sup_warn.get('brief_details','')} | Context: {alert_item_sup_warn.get('context_info','N/A')}"
                )
else:
    st.info("No new critical or high-priority patient alerts identified from today's field activities for this selection.")

# Display High-Priority Tasks generated from today's activities, or all high-priority for the CHW/Zone (needs more data)
supervisor_tasks_generated_today = generate_chw_prioritized_tasks(
    source_patient_data_df=daily_data_for_supervisor, # Based on today's triggers
    for_date=snapshot_date_selected_page,
    chw_id_context=chw_filter_applied,
    zone_context_str=zone_filter_applied or "All Supervised Zones",
    max_tasks_to_return_for_summary=10
)
if supervisor_tasks_generated_today:
    st.subheader("Top Priority Tasks (Generated from Today's Activities):")
    tasks_df_for_sup_display = pd.DataFrame(supervisor_tasks_generated_today)
    cols_for_task_table_sup = ['patient_id', 'assigned_chw_id', 'task_description', 'priority_score', 'due_date', 'status', 'key_patient_context']
    if not chw_filter_applied and 'assigned_chw_id' not in tasks_df_for_sup_display.columns and 'chw_id' in daily_data_for_supervisor.columns: # Try to map chw_id
         tasks_df_for_sup_display['assigned_chw_id'] = tasks_df_for_sup_display['patient_id'].map(daily_data_for_supervisor.drop_duplicates('patient_id').set_index('patient_id')['chw_id']).fillna(chw_filter_applied or "Team")
    
    st.dataframe(tasks_df_for_sup_display[[c for c in cols_for_task_table_sup if c in tasks_df_for_sup_display.columns]],
                 use_container_width=True, height=min(350, len(tasks_df_for_sup_display)*40 + 45))
else:
    st.info("No new high-priority tasks identified from today's activities for immediate supervisor review.")
st.markdown("---")

# Section 3: Local Epi Signals Summary (from today's field reports)
st.header("🌿 Local Epi Signals Watch (Today)")
if not daily_data_for_supervisor.empty:
    epi_signals_sup_today = extract_chw_local_epi_signals(
        chw_daily_encounter_df=daily_data_for_supervisor,
        pre_calculated_chw_kpis=daily_kpis_inputs_sup, # Passed from data loading
        for_date=selected_snapshot_date_view,
        chw_zone_context=zone_filter_applied or "All Supervised Zones"
    )
    cols_epi_kpi_sup_page = st.columns(3)
    with cols_epi_kpi_sup_page[0]: render_web_kpi_card("Symptomatic (Key Cond.)", str(epi_signals_sup_today.get("symptomatic_patients_key_conditions_count",0)), icon="🤒", units="patients", help_text=f"Keywords: {epi_signals_sup_today.get('symptom_keywords_for_monitoring','N/A')}")
    with cols_epi_kpi_sup_page[1]: render_web_kpi_card("New Malaria Cases", str(epi_signals_sup_today.get("newly_identified_malaria_patients_count",0)), icon="🦟", units="patients")
    with cols_epi_kpi_sup_page[2]: render_web_kpi_card("Pending TB Contacts", str(epi_signals_sup_today.get("pending_tb_contact_tracing_tasks_count",0)), icon="👥", units="tasks", status_level="MODERATE_CONCERN" if epi_signals_sup_today.get("pending_tb_contact_tracing_tasks_count",0)>0 else "ACCEPTABLE")
    
    if epi_signals_sup_today.get("detected_symptom_clusters"):
        st.markdown("###### **Potential Symptom Clusters Detected Today:**")
        for cluster_info in epi_signals_sup_today["detected_symptom_clusters"]:
            st.warning(f"⚠️ Cluster: **{cluster_info.get('symptoms_pattern')}** ({cluster_info.get('patient_count')} cases) in {cluster_info.get('location_hint', 'area')}. Review Recommended.")
    if epi_signals_sup_today.get("calculation_notes"):
        for note in epi_signals_sup_today["calculation_notes"]: st.caption(note)
else:
    st.markdown("_No field activity data for selected day/filters to derive local epi signals._")
st.markdown("---")

# Section 4: CHW Team/Individual Activity Trends (Periodic View)
st.header("📈 Periodic Activity Trends")
st.markdown(f"Displaying trends from **{selected_trend_start_date_page.strftime('%d %b %Y')}** to **{selected_trend_end_date_page.strftime('%d %b %Y')}** "
            f"{('for CHW ' + chw_filter_applied) if chw_filter_applied else ('for Supervised Team/All CHWs')} "
            f"{('in Zone ' + zone_filter_applied) if zone_filter_applied else ('across All Assigned Zones')}.")

if not period_data_for_supervisor.empty:
    chw_activity_trends_output = calculate_chw_activity_trends(
        chw_historical_health_df=period_data_for_supervisor, # Already filtered for period & scope
        trend_start_date=selected_trend_start_date_page, # Contextual, actual filtering happened in loader
        trend_end_date=selected_trend_end_date_page,
        zone_filter=None, # Data already filtered if specific zone selected at page level
        time_period_aggregation='D', # Daily trends common for supervisor view
        source_context_log_prefix="CHWSupPageTrends"
    )
    cols_trends_sup_page_view = st.columns(2)
    with cols_trends_sup_page_view[0]:
        visits_trend = chw_activity_trends_output.get("patient_visits_trend")
        if visits_trend is not None and not visits_trend.empty:
            st.plotly_chart(plot_annotated_line_chart_web(
                visits_trend.squeeze(), chart_title="Daily Patient Visits Trend", y_axis_label="# Patients", y_axis_is_count=True, chart_height=app_config.WEB_PLOT_COMPACT_HEIGHT
            ), use_container_width=True)
        else: st.caption("No patient visit trend data available.")
    with cols_trends_sup_page_view[1]:
        high_prio_trend = chw_activity_trends_output.get("high_priority_followups_trend")
        if high_prio_trend is not None and not high_prio_trend.empty:
            st.plotly_chart(plot_annotated_line_chart_web(
                high_prio_trend.squeeze(), chart_title="Daily High Priority Follow-ups Trend", y_axis_label="# Tasks/Patients", y_axis_is_count=True, chart_height=app_config.WEB_PLOT_COMPACT_HEIGHT
            ), use_container_width=True)
        else: st.caption("No high-priority follow-up trend data available.")
else:
    st.markdown("_No CHW activity data available for the selected trend period and filters._")

logger.info(f"CHW Supervisor Operations View page generated for snapshot: {selected_snapshot_date_view}, trends: {selected_trend_start_date_page}-{selected_trend_end_date_page}")
