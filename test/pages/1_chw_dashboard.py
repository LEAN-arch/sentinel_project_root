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
from datetime import date, timedelta, datetime # Added datetime for parsing

# --- Sentinel System Imports ---
# Assuming app_config and utils are in PYTHONPATH or project root relative to this page's execution
# For Streamlit Cloud, if main app is in `test/app_home.py`, these imports should work if
# `test` is added to path or if Streamlit correctly handles subdir `pages`.
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
except ImportError as e:
    # If imports fail, likely due to path issues when running a page script directly.
    # Add parent of current dir ('test') to path. This helps if 'test' is the app root.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_app_root = os.path.abspath(os.path.join(current_dir, os.pardir)) # Goes to 'test'
    # project_true_root = os.path.abspath(os.path.join(project_app_root, os.pardir)) # Goes to 'sentinel_project_root'
    # Only add if not already present
    # if project_app_root not in sys.path: sys.path.insert(0, project_app_root)
    # For multi-page apps, Streamlit typically manages path based on main script location.
    # This specific path adjustment might be aggressive if not structured carefully for Cloud.
    # Re-raising error if it's not related to just finding config/utils directly.
    # st.error(f"Import Error in 1_chw_dashboard.py - ensure your PYTHONPATH is correctly set up. Details: {e}")
    # For robustness in provided file, just log and try to proceed with placeholder stubs if modules aren't found for demo.
    logging.error(f"CRITICAL IMPORT ERROR in 1_chw_dashboard.py: {e}. Some functionalities will be mocked or fail.")
    # Define stubs so app doesn't crash immediately if above modules aren't found.
    def calculate_chw_daily_summary_metrics(*args, **kwargs): return {"notes":["summary_metrics_calculator not loaded"]}
    def generate_chw_patient_alerts_from_data(*args, **kwargs): return [{"notes":["alert_generator not loaded"]}]
    def extract_chw_local_epi_signals(*args, **kwargs): return {"notes":["epi_signal_extractor not loaded"]}
    def generate_chw_prioritized_tasks(*args, **kwargs): return [{"notes":["task_processor not loaded"]}]
    def calculate_chw_activity_trends(*args, **kwargs): return {"notes":["activity_trend_calculator not loaded"]}
    def render_web_kpi_card(title, value, **kwargs): st.warning(f"KPI Card: {title} - {value} (render_web_kpi_card not fully loaded)")
    def render_web_traffic_light_indicator(message, **kwargs): st.warning(f"Traffic Light: {message} (render_web_traffic_light_indicator not fully loaded)")
    def plot_annotated_line_chart_web(data, title, **kwargs): st.warning(f"Line Chart: {title} (plot_annotated_line_chart_web not fully loaded)")
    # For config, it is critical, try one more specific path for typical project structures
    # where 'test' is not root but 'sentinel_project_root' is.
    try:
        sys.path.insert(0, os.path.abspath(os.path.join(project_app_root, os.pardir)))
        from config import app_config # Retry
    except ImportError:
         st.error("FATAL: app_config.py not found. Application cannot run.")
         st.stop()


# Page Configuration (Specific to this Supervisor View)
st.set_page_config(
    page_title=f"CHW Supervisor Console - {app_config.APP_NAME}",
    layout="wide",
    initial_sidebar_state="expanded"
)

logger = logging.getLogger(__name__)
# CSS is assumed to be loaded by app_home.py, or a @st.cache_resource call here
# for standalone page runs or specific styling for this page.

# --- Data Loading Logic for CHW Supervisor View (Simulation) ---
@st.cache_data(
    ttl=app_config.CACHE_TTL_SECONDS_WEB_REPORTS, # From refactored app_config
    show_spinner="Loading CHW team operational data for supervisor..."
)
def get_chw_supervisor_dashboard_data_page( # Renamed for page context
    view_date_page: date,
    trend_start_date_page: date,
    trend_end_date_page: date,
    chw_id_filter_page: Optional[str] = None,
    zone_filter_page: Optional[str] = None
):
    logger.info(f"Supervisor View Data: Loading for date={view_date_page}, trend={trend_start_date_page}-{trend_end_date_page}, CHW={chw_id_filter_page}, Zone={zone_filter_page}")
    
    # Simulate loading pre-enriched health records as if synced from PEDs
    # In reality, this would query a Tier 1/2 database that holds this processed data.
    health_df_all = load_health_records(file_path=app_config.HEALTH_RECORDS_CSV, source_context="CHWSupervisorPageData") # source_context for logs
    
    if health_df_all.empty:
        logger.error("CHW Supervisor Page: Base health data (simulating synced records) empty or failed to load.")
        return pd.DataFrame(), pd.DataFrame(), {}

    # Ensure 'encounter_date' is datetime and prepare 'encounter_date_only' for efficient filtering
    if 'encounter_date' not in health_df_all.columns:
        logger.error("CHW Supervisor Page: 'encounter_date' column missing from health data.")
        return pd.DataFrame(), pd.DataFrame(), {}
    health_df_all['encounter_date'] = pd.to_datetime(health_df_all['encounter_date'], errors='coerce')
    health_df_all.dropna(subset=['encounter_date'], inplace=True) # Critical for date operations
    health_df_all['encounter_date_only'] = health_df_all['encounter_date'].dt.date

    # 1. Data for Daily Snapshot View (KPIs, Today's Alerts/Tasks, Epi Signals for 'view_date_page')
    daily_snapshot_data_page = health_df_all[health_df_all['encounter_date_only'] == view_date_page].copy()
    
    # Apply CHW ID and/or Zone filters (if 'chw_id' exists)
    # Assuming 'chw_id' is a column that identifies the CHW who recorded the encounter
    if chw_id_filter_page and 'chw_id' in daily_snapshot_data_page.columns:
        daily_snapshot_data_page = daily_snapshot_data_page[daily_snapshot_data_page['chw_id'] == chw_id_filter_page]
    if zone_filter_page and 'zone_id' in daily_snapshot_data_page.columns:
        daily_snapshot_data_page = daily_snapshot_data_page[daily_snapshot_data_page['zone_id'] == zone_filter_page]

    # 2. Data for Periodic Trends View
    periodic_trends_data_page = health_df_all[
        (health_df_all['encounter_date_only'] >= trend_start_date_page) &
        (health_df_all['encounter_date_only'] <= trend_end_date_page)
    ].copy()
    if chw_id_filter_page and 'chw_id' in periodic_trends_data_page.columns:
        periodic_trends_data_page = periodic_trends_data_page[periodic_trends_data_page['chw_id'] == chw_id_filter_page]
    if zone_filter_page and 'zone_id' in periodic_trends_data_page.columns:
        periodic_trends_data_page = periodic_trends_data_page[periodic_trends_data_page['zone_id'] == zone_filter_page]
        
    # Pre-calculated daily KPIs input (typically empty in this sim, as components derive from daily_snapshot_data_page)
    daily_kpi_inputs_for_components = {} 
    
    logger.info(f"Supervisor View Data: Daily snapshot records: {len(daily_snapshot_data_page)}, Trend period records: {len(periodic_trends_data_page)}")
    return daily_snapshot_data_page, periodic_trends_data_page, daily_kpi_inputs_for_components

# --- Page Title & Sidebar Filters ---
st.title(f"🧑‍🏫 {app_config.APP_NAME} - CHW Supervisor Operations View")
st.markdown(f"**Team Performance Monitoring, Alert Triage, and Field Activity Oversight**")
st.markdown("---")

if os.path.exists(app_config.APP_LOGO_SMALL): st.sidebar.image(app_config.APP_LOGO_SMALL, width=160)
st.sidebar.header("📊 View Filters")

# Mock CHW IDs and Zones for supervisor selection (in a real app, populate from data)
MOCK_CHW_IDS_SUP = ["All CHWs"] + [f"CHW{i:03d}" for i in range(1, 6)]
selected_chw_for_view = st.sidebar.selectbox("Filter by CHW:", options=MOCK_CHW_IDS_SUP, key="sup_chw_filter_v3")
chw_filter_to_apply = None if selected_chw_for_view == "All CHWs" else selected_chw_for_view

MOCK_ZONES_SUP = ["All Zones"] + [f"Zone{chr(65+i)}" for i in range(4)] # ZoneA to ZoneD
selected_zone_for_view = st.sidebar.selectbox("Filter by Zone:", options=MOCK_ZONES_SUP, key="sup_zone_filter_v3")
zone_filter_to_apply = None if selected_zone_for_view == "All Zones" else selected_zone_for_view

min_date_for_sup_view = date.today() - timedelta(days=app_config.WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND * 2) # ~60 days back max
max_date_for_sup_view = date.today()
selected_snapshot_date_view = st.sidebar.date_input(
    "Daily Snapshot Date:", value=max_date_for_sup_view,
    min_value=min_date_for_sup_view, max_value=max_date_for_sup_view, key="sup_snapshot_date_v3"
)

trend_end_default_sup = selected_snapshot_date_view
trend_start_default_sup = trend_end_default_sup - timedelta(days=app_config.WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND -1)
if trend_start_default_sup < min_date_for_sup_view: trend_start_default_sup = min_date_for_sup_view
sel_trend_start_view, sel_trend_end_view = st.sidebar.date_input(
    "Trends Date Range:", value=[trend_start_default_sup, trend_end_default_sup],
    min_value=min_date_for_sup_view, max_value=max_date_for_sup_view, key="sup_trend_range_v3"
)
if sel_trend_start_view > sel_trend_end_view:
    st.sidebar.error("Trend range error: Start date must be after end date.")
    sel_trend_start_view = sel_trend_end_view

# --- Load Data Based on Filter Selections ---
daily_data_sup, period_data_sup, daily_kpis_inputs_sup = get_chw_supervisor_dashboard_data_page(
    view_date_page=selected_snapshot_date_view,
    trend_start_date_page=sel_trend_start_view, trend_end_date_page=sel_trend_end_view,
    chw_id_filter_page=chw_filter_to_apply, zone_filter_page=zone_filter_to_apply
)

filter_context_message = f"Displaying data for **{selected_snapshot_date_view.strftime('%A, %d %b %Y')}**"
if chw_filter_to_apply: filter_context_message += f" | CHW: **{chw_filter_to_apply}**"
if zone_filter_to_apply: filter_context_message += f" | Zone: **{zone_filter_to_apply}**"
st.info(filter_context_message)

# --- Main Content Display ---
if daily_data_sup.empty and period_data_sup.empty:
    st.warning(f"No CHW activity data found for the selected filters and date range. Please adjust sidebar filters or check data sync from PEDs.")
    st.stop()

# Section 1: Daily Performance Snapshot KPIs
st.header(f"📊 Daily Performance Summary ({selected_snapshot_date_view.strftime('%d %b')})")
if not daily_data_sup.empty:
    daily_summary_metrics_sup = calculate_chw_daily_summary_metrics(
        chw_daily_kpi_input_data=daily_kpis_inputs_sup, # Often empty, derived from daily_data_sup
        chw_daily_encounter_df=daily_data_sup,
        for_date=selected_snapshot_date_view
    )
    kpi_cols_sup = st.columns(4) # Display 4 summary KPIs in a row
    kpi_data_for_render = [
        {"title":"Total Visits", "value_str":str(daily_summary_metrics_sup.get("visits_count",0)), "icon":"👥", "units":"visits", "status_level": "NEUTRAL"},
        {"title":"High Prio Follow-ups", "value_str":str(daily_summary_metrics_sup.get("high_ai_prio_followups_count",0)), "icon":"🎯", "units":"tasks", "status_level":"MODERATE_CONCERN" if daily_summary_metrics_sup.get("high_ai_prio_followups_count",0) > 2 else "ACCEPTABLE"}, # Example threshold for supervisor
        {"title":"Critical SpO2 Cases", "value_str":str(daily_summary_metrics_sup.get("critical_spo2_cases_identified_count",0)), "icon":"💨", "units":"patients", "status_level":"HIGH_CONCERN" if daily_summary_metrics_sup.get("critical_spo2_cases_identified_count",0) > 0 else "ACCEPTABLE"},
        {"title":"High Fever Cases", "value_str":str(daily_summary_metrics_sup.get("fever_cases_identified_count",0)), "icon":"🔥", "units":"patients", "status_level":"MODERATE_CONCERN" if daily_summary_metrics_sup.get("fever_cases_identified_count",0) > 0 else "ACCEPTABLE"}
    ]
    for i, kpi_d in enumerate(kpi_data_for_render):
        with kpi_cols_sup[i % 4]: render_web_kpi_card(**kpi_d)
else:
    st.markdown("_No CHW field activity recorded for the selected day/filters to display performance summary._")
st.markdown("---")

# Section 2: Key Alerts & Tasks for Supervisor Review
st.header("🚦 Alerts & Task Oversight (Today)")
alert_data_for_sup_list = generate_chw_patient_alerts_from_data(
    patient_encounter_data_df=daily_data_sup, # Uses daily data to find today's alerts
    chw_daily_context_df=daily_data_sup, # Can be same if daily_data_sup is the full context needed
    for_date=selected_snapshot_date_view,
    chw_zone_context_str=zone_filter_to_apply or "All Supervised Zones",
    max_alerts_to_return=7 # Supervisor might want to see a few more top alerts
)
if alert_data_for_sup_list:
    st.subheader(f"Priority Patient Alerts from Field ({selected_snapshot_date_view.strftime('%d %b')}):")
    critical_alert_displayed = False
    for alert_obj in alert_data_for_sup_list:
        if alert_obj.get("alert_level") == "CRITICAL":
            critical_alert_displayed = True
            render_web_traffic_light_indicator(
                message=f"Patient {alert_obj.get('patient_id','N/A')}: **{alert_obj.get('primary_reason','N/A')}**",
                status_level="HIGH_RISK", # Map to CSS class
                details_text=f"Details: {alert_obj.get('brief_details','')} | Context: {alert_obj.get('context_info','')} | Prio: {alert_obj.get('raw_priority_score',0):.0f}"
            )
    if not critical_alert_displayed and alert_data_for_sup_list: # If no CRITICAL, show top WARNING ones
        st.markdown("###### Other Notable Warnings:")
        for alert_obj in alert_data_for_sup_list:
            if alert_obj.get("alert_level") == "WARNING":
                render_web_traffic_light_indicator(
                    message=f"Patient {alert_obj.get('patient_id','N/A')}: {alert_obj.get('primary_reason','N/A')}",
                    status_level="MODERATE_RISK",
                    details_text=f"{alert_obj.get('brief_details','')} | Context: {alert_obj.get('context_info','')}"
                )
else:
    st.info("No new critical or high-priority patient alerts identified from today's field activities.")

# Display a summary of tasks (e.g., tasks with high priority generated today)
tasks_for_supervisor_review_list = generate_chw_prioritized_tasks(
    source_patient_data_df=daily_data_sup, # Based on today's activities
    for_date=selected_snapshot_date_view,
    chw_id_context=chw_filter_to_apply,
    zone_context_str=zone_filter_to_apply or "All Supervised Zones",
    max_tasks_to_return_for_summary=10
)
if tasks_for_supervisor_review_list:
    st.subheader("Top Priority Tasks Generated/Flagged Today:")
    tasks_display_df_sup = pd.DataFrame(tasks_for_supervisor_review_list)
    cols_task_sup = ['patient_id', 'assigned_chw_id', 'task_description', 'priority_score', 'due_date', 'status', 'key_patient_context']
    if not chw_filter_to_apply: # if viewing all CHWs, 'assigned_chw_id' is relevant
        if 'assigned_chw_id' not in tasks_display_df_sup.columns and 'chw_id' in daily_data_sup.columns: # If source df had it
             tasks_display_df_sup['assigned_chw_id'] = tasks_display_df_sup['patient_id'].map(daily_data_sup.drop_duplicates('patient_id').set_index('patient_id')['chw_id'])

    st.dataframe(tasks_display_df_sup[[c for c in cols_task_sup if c in tasks_display_df_sup.columns]], use_container_width=True, height=300)
else:
    st.info("No new high-priority tasks generated from today's field activities for immediate review.")
st.markdown("---")

# Section 3: Local Epi Signals (as observed by the CHW team today)
st.header("🌿 Local Epi Signals from Field Reports")
if not daily_data_sup.empty:
    epi_signals_today_sup = extract_chw_local_epi_signals(
        chw_daily_encounter_df=daily_data_sup,
        pre_calculated_chw_kpis=daily_kpis_inputs_sup,
        for_date=selected_snapshot_date_view,
        chw_zone_context=zone_filter_to_apply or "All Supervised Zones"
    )
    cols_epi_kpi_sup = st.columns(3)
    with cols_epi_kpi_sup[0]: render_web_kpi_card("Symptomatic (Key Cond.)", str(epi_signals_today_sup.get("symptomatic_patients_key_conditions_count",0)), icon="🤒", units="patients", help_text=f"Keywords: {epi_signals_today_sup.get('symptom_keywords_for_monitoring','N/A')}")
    with cols_epi_kpi_sup[1]: render_web_kpi_card("New Malaria (Today)", str(epi_signals_today_sup.get("newly_identified_malaria_patients_count",0)), icon="🦟", units="patients")
    with cols_epi_kpi_sup[2]: render_web_kpi_card("Pending TB Contacts", str(epi_signals_today_sup.get("pending_tb_contact_tracing_tasks_count",0)), icon="👥", units="tasks")
    
    if epi_signals_today_sup.get("detected_symptom_clusters"):
        st.markdown("###### **Potential Symptom Clusters Detected Today:**")
        for cluster_item in epi_signals_today_sup["detected_symptom_clusters"]:
            st.warning(f"⚠️ Cluster: **{cluster_item.get('symptoms_pattern')}** ({cluster_item.get('patient_count')} cases) in {cluster_item.get('location_hint', 'area')}. Supervisor review recommended.")
    if epi_signals_today_sup.get("calculation_notes"):
        for note_epi in epi_signals_today_sup["calculation_notes"]: st.caption(note_epi)
else:
    st.markdown("_No data for selected day/filters to derive local epi signals._")
st.markdown("---")

# Section 4: CHW Activity Trends (Periodic View for supervisor)
st.header("📈 Team/CHW Activity Trends")
st.markdown(f"Trends from **{sel_trend_start_view.strftime('%d %b %Y')}** to **{sel_trend_end_view.strftime('%d %b %Y')}**.")
if not period_data_sup.empty:
    chw_trends_output = calculate_chw_activity_trends(
        chw_historical_health_df=period_data_sup, # Data already filtered by date range and CHW/Zone by get_supervisor_chw_view_data
        trend_start_date=sel_trend_start_view, # For context if get_trend_data needs it
        trend_end_date=sel_trend_end_view,
        zone_filter=None, # Data already filtered if zone was selected
        time_period_aggregation='D' # Daily trends often useful for supervisors
    )
    
    cols_trends_sup_page = st.columns(2)
    with cols_trends_sup_page[0]:
        visits_trend_series = chw_trends_output.get("patient_visits_trend")
        if visits_trend_series is not None and not visits_trend_series.empty:
            st.plotly_chart(plot_annotated_line_chart_web(visits_trend_series.squeeze(), chart_title="Daily Patient Visits Trend", y_axis_label="# Patients", y_axis_is_count=True, chart_height=app_config.WEB_PLOT_COMPACT_HEIGHT), use_container_width=True)
        else: st.caption("No patient visit trend data available for this scope/period.")
    with cols_trends_sup_page[1]:
        prio_tasks_trend_series = chw_trends_output.get("high_priority_followups_trend")
        if prio_tasks_trend_series is not None and not prio_tasks_trend_series.empty:
            st.plotly_chart(plot_annotated_line_chart_web(prio_tasks_trend_series.squeeze(), chart_title="Daily High Prio. Follow-ups Trend", y_axis_label="# Tasks", y_axis_is_count=True, chart_height=app_config.WEB_PLOT_COMPACT_HEIGHT), use_container_width=True)
        else: st.caption("No high-priority follow-up trend data available.")
else:
    st.markdown("_No CHW activity data available for the selected filters and trend period._")

logger.info(f"CHW Supervisor Operations View page generated successfully for view date: {selected_snapshot_date_view}")
