# sentinel_project_root/test/pages/1_chw_dashboard.py
# CHW Supervisor Operations View for Sentinel Health Co-Pilot

import streamlit as st
import pandas as pd
import logging
from datetime import date, timedelta
from typing import Optional, Dict, Any, Tuple, List
import inspect

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- Sentinel System Imports ---
try:
    from config import app_config_settings as app_config
    from utils.core_data_processing import load_health_records
    from utils.ui_visualization_widgets import (
        render_kpi_card,
        render_alert_indicator,
        plot_trend_chart
    )
    from .chw_components_sentinel.summary_metrics import calculate_daily_metrics
    from .chw_components_sentinel.alert_generator import generate_alerts
    from .chw_components_sentinel.epi_signal import extract_local_signals
    from .chw_components_sentinel.task_processor import generate_prioritized_tasks
    from .chw_components_sentinel.activity_trends import calculate_activity_trends
except ImportError as e:
    error_msg = f"Import Error: {e}. Ensure modules in test/utils and test/pages/chw_components_sentinel are accessible and __init__.py files exist."
    logger.error(error_msg)
    st.error(error_msg)
    st.stop()

# --- Validate App Config ---
required_configs = [
    'APP_NAME', 'CACHE_TTL_SECONDS', 'WEB_DASH_DEFAULT_TREND_DAYS',
    'ALERT_SPO2_CRITICAL', 'ALERT_BODY_TEMP_FEV'
]
missing_configs = [attr for attr in required_configs if not hasattr(app_config, attr)]
if missing_configs:
    st.error(f"Missing configurations: {', '.join(missing_configs)}")
    st.stop()

# --- Page Configuration ---
st.set_page_config(
    page_title=f"CHW Supervisor View - {app_config.APP_NAME}",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Utility Function for Filter Options ---
def create_filter_options(df: pd.DataFrame, column: str, default_options: List[str], display_name: str) -> List[str]:
    options = [f"All {display_name}"]
    if isinstance(df, pd.DataFrame) and not df.empty and column in df.columns:
        options.extend(sorted(df[column].dropna().unique().tolist()))
    else:
        logger.warning(f"No valid {column} found. Using default {display_name} options.")
        st.sidebar.warning(f"No {display_name} available.")
        options.extend(default_options)
    return options

# --- Data Loading Function ---
@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS, show_spinner="Loading CHW operational data...")
def get_chw_supervisor_dashboard_data(
    view_date: date,
    trend_start_date: date,
    trend_end_date: date,
    selected_chw_id: Optional[str] = None,
    selected_zone_id: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    func_name = inspect.currentframe().f_code.co_name
    try:
        health_df = load_health_records(
            source_context=f"{func_name}/LoadSimulatedData",
            columns=['encounter_date', 'chw_id', 'zone_id', 'encounter_type', 'ai_followup_priority_score']
        )
    except Exception as e:
        logger.error(f"({func_name}) Failed to load health records: {e}")
        st.error("Failed to load health data.")
        return pd.DataFrame(), pd.DataFrame(), {}

    if health_df.empty:
        logger.error(f"({func_name}) Health records empty.")
        st.error("No health data available.")
        return pd.DataFrame(), pd.DataFrame(), {}

    required_cols = ['encounter_date', 'chw_id', 'zone_id', 'encounter_type']
    if not all(col in health_df.columns for col in required_cols):
        logger.error(f"({func_name}) Missing columns: {set(required_cols) - set(health_df.columns)}")
        st.error("Missing required columns in health data.")
        return pd.DataFrame(), pd.DataFrame(), {}

    if not pd.api.types.is_datetime64_any_dtype(health_df['encounter_date']):
        health_df['encounter_date'] = pd.to_datetime(health_df['encounter_date'], errors='coerce')
        health_df.dropna(subset=['encounter_date'], inplace=True)
        if health_df.empty:
            logger.error(f"({func_name}) Invalid 'encounter_date' values.")
            st.error("Invalid date values in health data.")
            return pd.DataFrame(), pd.DataFrame(), {}

    daily_df = health_df[health_df['encounter_date'].dt.date == view_date].copy()
    if selected_chw_id:
        daily_df = daily_df[daily_df['chw_id'] == selected_chw_id]
    if selected_zone_id:
        daily_df = daily_df[daily_df['zone_id'] == selected_zone_id]

    period_df = health_df[
        (health_df['encounter_date'].dt.date >= trend_start_date) &
        (health_df['encounter_date'].dt.date <= trend_end_date)
    ].copy()
    if selected_chw_id:
        period_df = period_df[period_df['chw_id'] == selected_chw_id]
    if selected_zone_id:
        period_df = period_df[period_df['zone_id'] == selected_zone_id]

    pre_calculated_kpis = {}
    if selected_chw_id and not daily_df.empty:
        worker_self_check_df = daily_df[
            (daily_df['chw_id'] == selected_chw_id) &
            (daily_df['encounter_type'] == 'WORKER_SELF_CHECK')
        ]
        if not worker_self_check_df.empty and 'ai_followup_priority_score' in worker_self_check_df.columns:
            pre_calculated_kpis['worker_self_fatigue_index_today'] = worker_self_check_df['ai_followup_priority_score'].max()
        else:
            pre_calculated_kpis['worker_self_fatigue_index_today'] = 0
            logger.warning(f"({func_name}) No valid self-check data.")

    logger.info(f"({func_name}) Data loaded: Daily - {len(daily_df)} recs, Period - {len(period_df)} recs.")
    return daily_df, period_df, pre_calculated_kpis

# --- Main Page Content ---
st.title("🧑‍🏫 CHW Supervisor Operations View")
st.markdown(f"**Team Performance Monitoring for {app_config.APP_NAME}**")
st.divider()

# --- Sidebar Filters ---
@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS)
def load_filter_data():
    return load_health_records(source_context="CHWDashboard/SidebarFilterPopulation")

filter_df = load_filter_data()
st.sidebar.image(app_config.APP_LOGO_SMALL, width=150) if os.path.exists(app_config.APP_LOGO_SMALL) else st.sidebar.markdown("🌍")

st.sidebar.header("🗓️ View Filters")
chw_options = create_filter_options(filter_df, 'chw_id', ["CHW001", "CHW002", "CHW003", "CHW004", "CHW005"], "CHWs")
zone_options = create_filter_options(filter_df, 'zone_id', ["ZoneA", "ZoneB", "ZoneC", "ZoneD"], "Zones")

if 'chw_id_selectbox' not in st.session_state:
    st.session_state['chw_id_selectbox'] = "All CHWs"
selected_chw_id = st.sidebar.selectbox(
    "Filter by CHW ID:", options=chw_options, key="chw_id_selectbox"
)
actual_chw_id = None if selected_chw_id == "All CHWs" else selected_chw_id

if 'zone_id_selectbox' not in st.session_state:
    st.session_state['zone_id_selectbox'] = "All Zones"
selected_zone_id = st.sidebar.selectbox(
    "Filter by Zone:", options=zone_options, key="zone_id_selectbox"
)
actual_zone_id = None if selected_zone_id == "All Zones" else selected_zone_id

min_date = date.today() - timedelta(days=max(90, app_config.WEB_DASH_DEFAULT_TREND_DAYS * 3))
max_date = date.today()
selected_daily_date = st.sidebar.date_input(
    "View Daily Activity For:",
    value=max_date,
    min_value=min_date,
    max_value=max_date,
    key="daily_snapshot_datepicker"
)

default_trend_end = selected_daily_date
default_trend_start = default_trend_end - timedelta(days=app_config.WEB_DASH_DEFAULT_TREND_DAYS - 1)
if default_trend_start < min_date:
    default_trend_start = min_date

if 'trend_datepicker' not in st.session_state:
    st.session_state['trend_datepicker'] = [default_trend_start, default_trend_end]

trend_dates = st.sidebar.date_input(
    "Select Trend Date Range:",
    value=st.session_state['trend_datepicker'],
    min_value=min_date,
    max_value=max_date,
    key="trend_datepicker"
)

selected_trend_start, selected_trend_end = trend_dates if isinstance(trend_dates, tuple) else (trend_dates, trend_dates)
if selected_trend_start > selected_trend_end:
    st.sidebar.error("Start date must be on or before end date.")
    st.session_state['trend_datepicker'] = [default_trend_start, default_trend_end]
    st.rerun()

# --- Load Data ---
try:
    daily_df, period_df, pre_calculated_kpis = get_chw_supervisor_dashboard_data(
        view_date=selected_daily_date,
        trend_start_date=selected_trend_start,
        trend_end_date=selected_trend_end,
        selected_chw_id=actual_chw_id,
        selected_zone_id=actual_zone_id
    )
except Exception as e:
    logger.error(f"Failed to load dashboard data: {e}")
    st.error(f"Failed to load dashboard data: {str(e)}")
    st.stop()

filter_context = [f"Date: **{selected_daily_date.strftime('%d %b %Y')}**"]
if actual_chw_id:
    filter_context.append(f"CHW: **{actual_chw_id}**")
if actual_zone_id:
    filter_context.append(f"Zone: **{actual_zone_id}**")
st.info(f"Displaying data for: {'  |  '.join(filter_context)}")

# --- Daily Performance Snapshot ---
st.header("📊 Daily Performance Snapshot")
if not daily_df.empty:
    try:
        summary_metrics = calculate_daily_metrics(
            chw_daily_kpi_input_data=pre_calculated_kpis,
            chw_daily_encounter_df=daily_df,
            for_date=selected_daily_date
        )
    except Exception as e:
        logger.error(f"Error calculating daily metrics: {e}")
        st.error(f"Failed to calculate daily metrics: {str(e)}")
        summary_metrics = {}

    cols = st.columns(4)
    with cols[0]:
        render_kpi_card("Visits Today", str(summary_metrics.get("visits_count", 0)),
                        icon="👥", help_text="Total unique patients visited today.")
    high_prio_count = summary_metrics.get("high_ai_prio_followups_count", 0)
    high_prio_status = "ACCEPTABLE" if high_prio_count <= 2 else ("MODERATE_CONCERN" if high_prio_count <= 5 else "HIGH_CONCERN")
    with cols[1]:
        render_kpi_card("High Prio Follow-ups", str(high_prio_count),
                        icon="🎯", status_level=high_prio_status,
                        help_text="Patients needing urgent follow-up.")
    spo2_count = summary_metrics.get("critical_spo2_cases_identified_count", 0)
    spo2_status = "HIGH_CONCERN" if spo2_count > 0 else "ACCEPTABLE"
    with cols[2]:
        render_kpi_card("Critical SpO2 Cases", str(spo2_count),
                        icon="💨", status_level=spo2_status,
                        help_text=f"SpO2 < {app_config.ALERT_SPO2_CRITICAL}%.")
    fever_count = summary_metrics.get("high_fever_cases_identified_count", 0)
    fever_status = "HIGH_CONCERN" if fever_count > 0 else "ACCEPTABLE"
    with cols[3]:
        render_kpi_card("High Fever Cases", str(fever_count),
                        icon="🔥", status_level=fever_status,
                        help_text=f"Temperature ≥ {app_config.ALERT_BODY_TEMP_FEV}°C.")
else:
    st.markdown("_No activity data for daily performance snapshot._")
st.divider()

# --- Key Alerts & Actionable Tasks ---
st.header("🚦 Key Alerts & Actionable Tasks")
try:
    patient_alerts = generate_alerts(
        patient_encounter_data_df=daily_df,
        for_date=selected_daily_date,
        chw_zone_context_str=actual_zone_id or "All Zones",
        max_alerts_to_return=8
    )
except Exception as e:
    logger.error(f"Error generating alerts: {e}")
    st.error(f"Failed to generate alerts: {str(e)}")
    patient_alerts = []

if patient_alerts:
    st.subheader("Priority Patient Alerts:")
    found_critical = False
    for alert in patient_alerts:
        if alert.get("alert_level") == "CRITICAL":
            found_critical = True
            render_alert_indicator(
                message=f"Pt. {alert.get('patient_id', 'N/A')}: {alert.get('primary_reason', 'Critical Alert')}",
                status_level="HIGH_RISK",
                details_text=f"Details: {alert.get('brief_details','N/A')} | Context: {alert.get('context_info','N/A')} | Action: {alert.get('suggested_action_code','REVIEW_IMMEDIATELY')}"
            )
    if not found_critical:
        st.info("No CRITICAL alerts.")
    warnings = [a for a in patient_alerts if a.get("alert_level") == "WARNING"]
    if warnings:
        st.markdown("###### Warning Alerts:")
        for warn in warnings:
            render_alert_indicator(
                message=f"Pt. {warn.get('patient_id', 'N/A')}: {warn.get('primary_reason', 'Warning')}",
                status_level="MODERATE_RISK",
                details_text=f"Details: {warn.get('brief_details','N/A')} | Context: {warn.get('context_info','N/A')}"
            )
    elif not found_critical:
        st.info("Only informational alerts generated.")
elif not daily_df.empty:
    st.info("No alerts generated.")
else:
    st.markdown("_No data to generate alerts._")

try:
    tasks = generate_prioritized_tasks(
        source_patient_data_df=daily_df,
        for_date=selected_daily_date,
        chw_id_context=actual_chw_id,
        zone_context_str=actual_zone_id or "All Zones",
        max_tasks_to_return_for_summary=10
    )
except Exception as e:
    logger.error(f"Error generating tasks: {e}")
    st.error(f"Failed to generate tasks: {str(e)}")
    tasks = []

if tasks:
    st.subheader("Top Priority Tasks:")
    df_tasks = pd.DataFrame(tasks)
    cols = ['patient_id', 'task_description', 'priority_score', 'due_date', 'status', 'key_patient_context', 'assigned_chw_id', 'alert_source_info']
    actual_cols = [col for col in cols if col in df_tasks.columns]
    st.dataframe(
        df_tasks[actual_cols],
        use_container_width=True,
        height=min(380, len(df_tasks) * 38 + 58),
        hide_index=True
    )
elif not daily_df.empty:
    st.info("No high-priority tasks identified.")
else:
    st.markdown("_No data to generate tasks._")
st.divider()

# --- Local Epi Signals ---
st.header("🔬 Local Epi Signals Watch")
if not daily_df.empty:
    try:
        epi_signals = extract_local_signals(
            chw_daily_encounter_df=daily_df,
            pre_calculated_chw_kpis=pre_calculated_kpis,
            for_date=selected_daily_date,
            chw_zone_context=actual_zone_id or "All Zones"
        )
    except Exception as e:
        logger.error(f"Error extracting epi signals: {e}")
        st.error(f"Failed to extract epi signals: {str(e)}")
        epi_signals = {}

    cols = st.columns(3)
    with cols[0]:
        render_kpi_card("Symptomatic (Key Cond.)", str(epi_signals.get("symptomatic_patients_key_conditions_count", 0)),
                        icon="🤒", units="cases today", help_text="Patients with key conditions and symptoms.")
    with cols[1]:
        render_kpi_card("New Malaria Cases", str(epi_signals.get("newly_identified_malaria_patients_count", 0)),
                        icon="🦟", units="cases", help_text="Malaria cases identified today.")
    with cols[2]:
        render_kpi_card("Pending TB Contacts", str(epi_signals.get("pending_tb_contact_tracing_tasks_count", 0)),
                        icon="👥", units="to trace", help_text="TB contacts needing follow-up.")
    clusters = epi_signals.get("detected_symptom_clusters", [])
    if clusters:
        st.markdown("###### Detected Symptom Clusters:")
        for cluster in clusters:
            st.warning(f"⚠️ **{cluster.get('symptoms_pattern', 'Unknown Pattern')}**: {cluster.get('patient_count', 'N/A')} cases in {cluster.get('location_hint', 'area')}")
    elif 'patient_reported_symptoms' in daily_df.columns:
        st.info("No significant symptom clusters detected.")
else:
    st.markdown("_No data for epi signals._")
st.divider()

# --- CHW Team Activity Trends ---
st.header("📈 CHW Team Activity Trends")
trend_period = f"{selected_trend_start.strftime('%d %b %Y')} - {selected_trend_end.strftime('%d %b %Y')}"
context = f" for CHW {actual_chw_id}" if actual_chw_id else ""
context += f" in Zone {actual_zone_id}" if actual_zone_id else ""
st.markdown(f"Trends from **{trend_period}**{context or ' (All CHWs/Zones)'}.")

if not period_df.empty:
    try:
        activity_trends = calculate_activity_trends(
            chw_historical_health_df=period_df,
            trend_start_date_input=selected_trend_start,
            trend_end_date_input=selected_trend_end,
            zone_filter=actual_zone_id,
            time_period_aggregation='D'
        )
    except Exception as e:
        logger.error(f"Error calculating trends: {e}")
        st.error(f"Failed to calculate trends: {str(e)}")
        activity_trends = {}

    cols = st.columns(2)
    with cols[0]:
        visits_trend = activity_trends.get("patient_visits_trend")
        if isinstance(visits_trend, pd.Series) and not visits_trend.empty:
            st.plotly_chart(plot_trend_chart(
                visits_trend, chart_title="Daily Patient Visits",
                y_axis_label="# Patients Visited", y_is_count=True
            ), use_container_width=True)
        else:
            st.caption("No visit trend data.")
    with cols[1]:
        followups_trend = activity_trends.get("high_priority_followups_trend")
        if isinstance(followups_trend, pd.Series) and not followups_trend.empty:
            st.plotly_chart(plot_trend_chart(
                followups_trend, chart_title="Daily High Prio. Follow-ups",
                y_axis_label="# High Prio Follow-ups", y_is_count=True
            ), use_container_width=True)
        else:
            st.caption("No follow-up trend data.")
else:
    st.markdown("_No historical data for trends._")

logger.info(f"CHW Supervisor View loaded for Date: {selected_daily_date}, CHW: {actual_chw_id or 'All'}, Zone: {actual_zone_id or 'All'}.")
