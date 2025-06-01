# sentinel_project_root/test/pages/1_chw_dashboard.py
# Redesigned as "CHW Supervisor Operations View" for "Sentinel Health Co-Pilot"

import streamlit as st
import pandas as pd
import numpy as np
import os
import logging
from datetime import date, timedelta # Removed datetime as date objects are used

# --- Sentinel System Imports ---
# Standard try-except for module imports, robust for different execution contexts
# Ensure 'test' directory is treated as a source root by Streamlit/Python.
try:
    from config import app_config
    from utils.core_data_processing import load_health_records # Used by this page's data loader
    # Corrected import paths based on the directory structure defined
    from pages.chw_components_sentinel.summary_metrics_calculator import calculate_chw_daily_summary_metrics
    from pages.chw_components_sentinel.alert_generator import generate_chw_patient_alerts_from_data
    from pages.chw_components_sentinel.epi_signal_extractor import extract_chw_local_epi_signals
    from pages.chw_components_sentinel.task_processor import generate_chw_prioritized_tasks
    from pages.chw_components_sentinel.activity_trend_calculator import calculate_chw_activity_trends
    from utils.ui_visualization_helpers import (
        render_web_kpi_card,
        render_web_traffic_light_indicator,
        plot_annotated_line_chart_web
        # plot_bar_chart_web # Uncomment if used by specific epi displays
    )
except ImportError as e:
    # Fallback for local dev if PYTHONPATH not perfectly set for sub-pages
    import sys
    current_dir_page1 = os.path.dirname(os.path.abspath(__file__)) # .../pages
    project_app_root_page1 = os.path.abspath(os.path.join(current_dir_page1, os.pardir)) # .../test
    if project_app_root_page1 not in sys.path:
        sys.path.insert(0, project_app_root_page1)
    logging.warning(f"CHW Sup View: Added '{project_app_root_page1}' to sys.path due to: {e}")
    # Retry imports after path adjustment (essential for modules in 'test')
    from config import app_config
    from utils.core_data_processing import load_health_records
    from pages.chw_components_sentinel.summary_metrics_calculator import calculate_chw_daily_summary_metrics
    from pages.chw_components_sentinel.alert_generator import generate_chw_patient_alerts_from_data
    from pages.chw_components_sentinel.epi_signal_extractor import extract_chw_local_epi_signals
    from pages.chw_components_sentinel.task_processor import generate_chw_prioritized_tasks
    from pages.chw_components_sentinel.activity_trend_calculator import calculate_chw_activity_trends
    from utils.ui_visualization_helpers import (
        render_web_kpi_card, render_web_traffic_light_indicator,
        plot_annotated_line_chart_web
    )


# --- Page Configuration ---
st.set_page_config(
    page_title=f"CHW Supervisor Console - {app_config.APP_NAME}",
    layout="wide",
    initial_sidebar_state="expanded"
)
logger = logging.getLogger(__name__)
# CSS: Global styles loaded in app_home.py or add @st.cache_resource CSS loader here for standalone.

# --- Data Loading for CHW Supervisor View (Simulated) ---
@st.cache_data(
    ttl=app_config.CACHE_TTL_SECONDS_WEB_REPORTS,
    show_spinner="Loading CHW team operational data for supervisor..."
)
def get_chw_supervisor_page_data_final(
    view_date_p: date,
    historical_start_date_p: date,
    historical_end_date_p: date,
    chw_id_filter_p: Optional[str] = None,
    zone_filter_p: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    logger.info(f"CHW Sup Data: Loading for snapshot={view_date_p}, trends={historical_start_date_p}-{historical_end_date_p}, CHW={chw_id_filter_p}, Zone={zone_filter_p}")
    health_df_all_p = load_health_records(file_path=app_config.HEALTH_RECORDS_CSV, source_context="CHWSupViewPage/Data")
    if health_df_all_p.empty:
        logger.error("CHW Sup Page: Base health data load failed.")
        return pd.DataFrame(), pd.DataFrame(), {}

    if 'encounter_date' not in health_df_all_p.columns:
        logger.error("CHW Sup Page: 'encounter_date' missing from health_df_all_p.")
        return pd.DataFrame(), pd.DataFrame(), {}
    health_df_all_p['encounter_date'] = pd.to_datetime(health_df_all_p['encounter_date'], errors='coerce')
    health_df_all_p.dropna(subset=['encounter_date'], inplace=True)
    health_df_all_p['encounter_date_only'] = health_df_all_p['encounter_date'].dt.date

    # Daily snapshot data
    daily_df_p = health_df_all_p[health_df_all_p['encounter_date_only'] == view_date_p].copy()
    if chw_id_filter_p and 'chw_id' in daily_df_p.columns:
        daily_df_p = daily_df_p[daily_df_p['chw_id'] == chw_id_filter_p]
    if zone_filter_p and 'zone_id' in daily_df_p.columns:
        daily_df_p = daily_df_p[daily_df_p['zone_id'] == zone_filter_p]

    # Periodic data for trends
    period_df_p = health_df_all_p[
        (health_df_all_p['encounter_date_only'] >= historical_start_date_p) &
        (health_df_all_p['encounter_date_only'] <= historical_end_date_p)
    ].copy()
    if chw_id_filter_p and 'chw_id' in period_df_p.columns:
        period_df_p = period_df_p[period_df_p['chw_id'] == chw_id_filter_p]
    if zone_filter_p and 'zone_id' in period_df_p.columns:
        period_df_p = period_df_p[period_df_p['zone_id'] == zone_filter_p]
    
    pre_calc_kpis_placeholder = {} # Summary component derives from daily_df_p
    return daily_df_p, period_df_p, pre_calc_kpis_placeholder

# --- Page Title & Sidebar ---
st.title(f"🧑‍🏫 {app_config.APP_NAME} - CHW Supervisor Operations View")
st.markdown(f"**Team Activity Monitoring, Alert Triage, and Field Support Coordination**")
st.markdown("---")

if os.path.exists(app_config.APP_LOGO_SMALL): st.sidebar.image(app_config.APP_LOGO_SMALL, width=150)
st.sidebar.header("📊 View Filters")

# Placeholder filter options (in a real app, populate dynamically)
MOCK_CHWS = ["All CHWs"] + [f"CHW{i:03d}" for i in range(1, 6)]
sel_chw = st.sidebar.selectbox("Filter by CHW:", options=MOCK_CHWS, key="sup_page_chw_filter_v4")
chw_to_filter = None if sel_chw == "All CHWs" else sel_chw

MOCK_ZONES = ["All Zones"] + [f"Zone{chr(65+i)}" for i in range(4)] # ZoneA-D
sel_zone = st.sidebar.selectbox("Filter by Zone:", options=MOCK_ZONES, key="sup_page_zone_filter_v4")
zone_to_filter = None if sel_zone == "All Zones" else sel_zone

max_days_lookback = app_config.WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND * 2 # e.g., 60 days
min_snapshot_date = date.today() - timedelta(days=max_days_lookback)
max_snapshot_date = date.today()
snapshot_date_selected = st.sidebar.date_input(
    "Daily Snapshot For:", value=max_snapshot_date,
    min_value=min_snapshot_date, max_value=max_snapshot_date, key="sup_page_snapshot_date_v4"
)

trend_end_def = snapshot_date_selected
trend_start_def = trend_end_def - timedelta(days=app_config.WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND - 1)
if trend_start_def < min_snapshot_date : trend_start_def = min_snapshot_date
sel_trend_start_page_dt, sel_trend_end_page_dt = st.sidebar.date_input(
    "Periodic Trends Range:", value=[trend_start_def, trend_end_def],
    min_value=min_snapshot_date, max_value=max_snapshot_date, key="sup_page_trend_range_v4"
)
if sel_trend_start_page_dt > sel_trend_end_page_dt:
    st.sidebar.error("Trend range invalid: Start date must precede end date.")
    sel_trend_start_page_dt = sel_trend_end_page_dt

# --- Load Data based on Filters ---
daily_data_chw_sup_page, period_data_chw_sup_page, daily_kpi_inputs_sup_page = get_chw_supervisor_page_data_final(
    view_date_p=snapshot_date_selected,
    historical_start_date_p=sel_trend_start_page_dt, historical_end_date_p=sel_trend_end_date_page,
    chw_id_filter_p=chw_to_filter, zone_filter_p=zone_to_filter
)

filter_ctx_msg = f"Displaying data for snapshot date: **{snapshot_date_selected.strftime('%A, %d %b %Y')}**"
if chw_to_filter: filter_ctx_msg += f" | CHW ID: **{chw_to_filter}**"
if zone_to_filter: filter_ctx_msg += f" | Zone: **{zone_to_filter}**"
st.info(filter_ctx_msg)

if daily_data_chw_sup_page.empty and period_data_chw_sup_page.empty:
    st.warning("No CHW activity data found for the current filter selections and date range. Please adjust sidebar filters or check data synchronization from PEDs.")
    st.stop()

# === SECTION 1: Daily Performance Snapshot KPIs ===
st.header(f"📊 Daily Performance Metrics ({snapshot_date_selected.strftime('%d %b')})")
if not daily_data_chw_sup_page.empty:
    daily_summary_metrics = calculate_chw_daily_summary_metrics(
        chw_daily_kpi_input_data=daily_kpis_inputs_sup_page,
        chw_daily_encounter_df=daily_data_chw_sup_page,
        for_date=snapshot_date_selected
    )
    kpi_cols_s1 = st.columns(4)
    # Use more descriptive status levels relevant to supervisor
    kpi_definitions = [
        {"title":"Total Visits", "value_str":str(daily_summary_metrics.get("visits_count",0)), "icon":"👥", "units":"visits", "status_level":"NEUTRAL"},
        {"title":"High Prio Follow-ups", "value_str":str(daily_summary_metrics.get("high_ai_prio_followups_count",0)), "icon":"🎯", "units":"tasks", "status_level":"MODERATE_CONCERN" if daily_summary_metrics.get("high_ai_prio_followups_count",0) > (5 if not chw_to_filter else 2) else "ACCEPTABLE", "help_text":"High AI priority tasks generated."}, # Threshold different for team vs individual CHW
        {"title":"Critical SpO2 Cases", "value_str":str(daily_summary_metrics.get("critical_spo2_cases_identified_count",0)), "icon":"💨", "units":"patients", "status_level":"HIGH_CONCERN" if daily_summary_metrics.get("critical_spo2_cases_identified_count",0) > 0 else "ACCEPTABLE", "help_text":f"Patients with SpO2 < {app_config.ALERT_SPO2_CRITICAL_LOW_PCT}%"},
        {"title":"High Fever Cases", "value_str":str(daily_summary_metrics.get("fever_cases_identified_count",0)), "icon":"🔥", "units":"patients", "status_level":"MODERATE_CONCERN" if daily_summary_metrics.get("fever_cases_identified_count",0) > 1 else "ACCEPTABLE", "help_text":f"Patients with Temp >= {app_config.ALERT_BODY_TEMP_HIGH_FEVER_C}°C"}
    ]
    for i, kpi in enumerate(kpi_definitions):
        with kpi_cols_s1[i % 4]: render_web_kpi_card(**kpi)
else: st.caption("No activity data for daily performance snapshot for this selection.")
st.markdown("---")

# === SECTION 2: Key Alerts & Actionable Tasks for Supervisor Review ===
st.header(f"🚦 Key Alerts & Task Oversight ({snapshot_date_selected.strftime('%d %b')})")
alerts_for_supervisor = generate_chw_patient_alerts_from_data(
    patient_encounter_data_df=daily_data_chw_sup_page,
    chw_daily_context_df=daily_data_chw_sup_page, # Full daily context if needed
    for_date=snapshot_date_selected,
    chw_zone_context_str=zone_to_filter or "All Supervised Zones",
    max_alerts_to_return=8 # Max alerts to show in this list
)
if alerts_for_supervisor:
    st.subheader(f"Priority Patient Alerts Escalated or Identified:")
    num_crit_alerts = 0
    for alert in alerts_for_supervisor:
        if alert.get("alert_level") == "CRITICAL":
            num_crit_alerts +=1
            render_web_traffic_light_indicator(
                message=f"Pt {alert.get('patient_id','?')} ({alert.get('context_info','').split('|')[0].strip()}): **{alert.get('primary_reason','Unknown Reason')}**",
                status_level="HIGH_RISK",
                details_text=f"{alert.get('brief_details','')} | Action: {alert.get('suggested_action_code','Review Urgently')} | Prio: {alert.get('raw_priority_score',0):.0f}"
            )
    if num_crit_alerts == 0 and alerts_for_supervisor : # No CRITICAL, show top WARNINGS
        st.markdown("###### Other Notable Warnings:")
        for alert in alerts_for_supervisor:
            if alert.get("alert_level") == "WARNING":
                 render_web_traffic_light_indicator(message=f"Pt {alert.get('patient_id','?')}: {alert.get('primary_reason','')}", status_level="MODERATE_RISK", details_text=f"{alert.get('brief_details','')}")
    elif num_crit_alerts == 0:
        st.info("No CRITICAL or WARNING patient alerts flagged from today's activities based on current filters.")

else: st.info("No significant patient alerts generated from today's field activities for this selection.")

# Tasks for Supervisor Review (e.g., high priority, overdue if that data available)
tasks_for_supervisor_list = generate_chw_prioritized_tasks(
    source_patient_data_df=daily_data_chw_sup_page, # Tasks derived from today's encounters/alerts
    for_date=snapshot_date_selected,
    chw_id_context=chw_to_filter,
    zone_context_str=zone_to_filter or "All Supervised Zones",
    max_tasks_to_return_for_summary=10
)
if tasks_for_supervisor_list:
    st.subheader("Top Priority Tasks (Generated/Relevant for Today):")
    df_tasks_sup = pd.DataFrame(tasks_for_supervisor_list)
    # Define columns relevant for supervisor table
    cols_to_show = ['patient_id', 'task_description', 'priority_score', 'assigned_chw_id', 'status', 'due_date', 'key_patient_context']
    if not chw_to_filter and 'assigned_chw_id' not in df_tasks_sup.columns and 'chw_id' in daily_data_chw_sup_page.columns : # Attempt to map chw_id for team view
        chw_map = daily_data_chw_sup_page.drop_duplicates('patient_id').set_index('patient_id')['chw_id']
        df_tasks_sup['assigned_chw_id'] = df_tasks_sup['patient_id'].map(chw_map).fillna("Team")
    st.dataframe(df_tasks_sup[[c for c in cols_to_show if c in df_tasks_sup.columns]], use_container_width=True, height=300)
else: st.info("No new high-priority tasks identified for immediate review from today's activities.")
st.markdown("---")

# === SECTION 3: Local Epi Signals (Observed by team/CHW today) ===
st.header("🌿 Local Epi Signals ({})".format(snapshot_date_selected.strftime('%d %b')))
if not daily_data_chw_sup_page.empty:
    epi_signals_data = extract_chw_local_epi_signals(
        chw_daily_encounter_df=daily_data_chw_sup_page,
        pre_calculated_chw_kpis=daily_kpis_inputs_sup, # Contains summary from get_chw_summary potentially
        for_date=snapshot_date_selected,
        chw_zone_context=zone_to_filter or "All Supervised Zones"
    )
    cols_epi_cards_sup = st.columns(3)
    with cols_epi_cards_sup[0]: render_web_kpi_card("Symptomatic (Key Cond.)", str(epi_signals_data.get("symptomatic_patients_key_conditions_count",0)), icon="🤒", units="patients", help_text=f"Monitored: {epi_signals_data.get('symptom_keywords_for_monitoring','N/A')}")
    with cols_epi_cards_sup[1]: render_web_kpi_card("New Malaria Cases (Today)", str(epi_signals_data.get("newly_identified_malaria_patients_count",0)), icon="🦟", units="patients")
    with cols_epi_cards_sup[2]: render_web_kpi_card("Pending TB Contacts", str(epi_signals_data.get("pending_tb_contact_tracing_tasks_count",0)), icon="👥", units="tasks", status_level="MODERATE_CONCERN" if epi_signals_data.get("pending_tb_contact_tracing_tasks_count",0) > 0 else "ACCEPTABLE")

    if epi_signals_data.get("detected_symptom_clusters"):
        st.markdown("###### **Potential Symptom Clusters (Today):**")
        for cluster_data in epi_signals_data["detected_symptom_clusters"]:
            st.warning(f"⚠️ **Cluster:** {cluster_data.get('symptoms_pattern')} ({cluster_data.get('patient_count')} cases) in {cluster_data.get('location_hint', 'area')}. Advise follow-up/investigation.")
    if epi_signals_data.get("calculation_notes"):
        for note_item in epi_signals_data["calculation_notes"]: st.caption(f"Epi Note: {note_item}")
else: st.markdown("_No field activity data for selected day/filters to derive local epi signals._")
st.markdown("---")

# === SECTION 4: Periodic Activity Trends ===
st.header("📈 Team/CHW Periodic Activity Trends")
st.markdown(f"Trends for: **{sel_trend_start_page_dt.strftime('%d %b %Y')}** to **{sel_trend_end_page_dt.strftime('%d %b %Y')}**.")
if not period_data_chw_sup_page.empty:
    activity_trends_dict = calculate_chw_activity_trends(
        chw_historical_health_df=period_data_chw_sup_page, # Data is already filtered for date range & scope
        trend_start_date=sel_trend_start_page_dt, # Still useful for get_trend_data's internal ranging if needed
        trend_end_date=sel_trend_end_page_dt,
        zone_filter=None, # Already filtered in period_data_chw_sup_page
        time_period_aggregation='D', # Daily trends
        source_context_log_prefix="SupViewPageTrends"
    )
    trend_cols_viz = st.columns(2)
    with trend_cols_viz[0]:
        visits_trend_s = activity_trends_dict.get("patient_visits_trend")
        if visits_trend_s is not None and not visits_trend_s.empty:
            st.plotly_chart(plot_annotated_line_chart_web(
                visits_trend_s.squeeze(), chart_title="Daily Patient Visits", y_axis_label="# Patients",
                y_axis_is_count=True, chart_height=app_config.WEB_PLOT_COMPACT_HEIGHT), use_container_width=True)
        else: st.caption("No patient visit trend data found for this selection.")
    with trend_cols_viz[1]:
        prio_trend_s = activity_trends_dict.get("high_priority_followups_trend")
        if prio_trend_s is not None and not prio_trend_s.empty:
            st.plotly_chart(plot_annotated_line_chart_web(
                prio_trend_s.squeeze(), chart_title="Daily High Prio Follow-ups", y_axis_label="# Tasks/Pts",
                y_axis_is_count=True, chart_height=app_config.WEB_PLOT_COMPACT_HEIGHT), use_container_width=True)
        else: st.caption("No high-priority follow-up trend data found.")
else: st.markdown("_No historical activity data for selected filters and trend period._")

logger.info(f"CHW Supervisor View page generated for snapshot: {snapshot_date_selected}.")
