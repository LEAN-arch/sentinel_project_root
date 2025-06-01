# sentinel_project_root/test/pages/4_population_dashboard.py
# Redesigned as "Population Health Analytics & Research Console" for "Sentinel Health Co-Pilot"
# This page simulates a web interface for epidemiologists, researchers, and program managers,
# typically at a Tier 3 (Regional/Cloud Node) or an advanced Tier 2 (Facility Node)
# with access to broader, aggregated datasets for in-depth analysis.

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys # For existing path manipulation if necessary for your environment
import logging
from datetime import date, timedelta
import plotly.express as px # Keep for direct use for complex or custom plots if needed
import html # For escaping text in custom HTML KPIs

# --- Sentinel System Imports ---
# Assuming 'test' is the app root from Streamlit's perspective or added to sys.path
try:
    from config import app_config
    from utils.core_data_processing import (
        load_health_records, load_zone_data, get_trend_data
    )
    from utils.ai_analytics_engine import apply_ai_models # To simulate pre-enriched data
    from utils.ui_visualization_helpers import ( # Use _web suffixed functions
        plot_bar_chart_web, plot_donut_chart_web,
        plot_annotated_line_chart_web, _create_empty_plot_figure
    )
except ImportError as e:
    logging.critical(f"CRITICAL IMPORT ERROR in 4_population_dashboard.py: {e}. Application may not function.")
    st.error(f"Application Critical Error: Could not load modules for Population Analytics. Details: {e}")
    # Fallback for app_config if it's the one failing and others depend on it for the page to load partially
    class AppConfigPopFallback: LOG_LEVEL = "INFO"; APP_NAME = "Sentinel Fallback"; CACHE_TTL_SECONDS_WEB_REPORTS = 300; RISK_SCORE_HIGH_THRESHOLD = 75; APP_FOOTER_TEXT = "Fallback Footer."
    app_config = AppConfigPopFallback()
    st.stop() # Stop further execution if critical modules like config are missing

# --- Page Configuration ---
st.set_page_config(
    page_title=f"Population Analytics - {app_config.APP_NAME}",
    layout="wide",
    initial_sidebar_state="expanded"
)
logger = logging.getLogger(__name__)
# CSS loaded globally by app_home.py

# --- Data Loading for Population Analytics Console (Simulates Tier 3 Data Warehouse) ---
@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS_WEB_REPORTS, show_spinner="Loading and preparing population analytics dataset...")
def get_population_analytics_console_dataset_page(source_context="PopAnalyticsConsole/PageDataLoad"): # Renamed
    logger.info(f"({source_context}) Loading comprehensive population health records and zone attributes.")
    health_df_raw_pop_console = load_health_records(file_path=app_config.HEALTH_RECORDS_CSV, source_context=source_context)
    
    health_df_enriched_console = pd.DataFrame()
    if not health_df_raw_pop_console.empty:
        enriched_data_tuple = apply_ai_models(health_df_raw_pop_console, source_context=f"{source_context}/AIEnrich")
        health_df_enriched_console = enriched_data_tuple[0] if enriched_data_tuple and isinstance(enriched_data_tuple, tuple) else pd.DataFrame()
    else:
        logger.warning(f"({source_context}) Raw health records empty for population analytics. AI enrichment skipped.")
        # Fallback schema for an empty dataframe
        cols_h = health_df_raw_pop_console.columns.tolist() if health_df_raw_pop_console is not None else []
        cols_ai_add = ['ai_risk_score', 'ai_followup_priority_score']
        health_df_enriched_console = pd.DataFrame(columns=list(set(cols_h + cols_ai_add)))

    zone_gdf_full = load_zone_data(source_context=source_context) # Returns merged GDF of attributes and geoms
    zone_attributes_console_df = pd.DataFrame()
    if zone_gdf_full is not None and not zone_gdf_full.empty:
        geom_col_name = zone_gdf_full.geometry.name if hasattr(zone_gdf_full, 'geometry') else 'geometry'
        cols_to_drop = [geom_col_name] if geom_col_name in zone_gdf_full.columns else []
        zone_attributes_console_df = pd.DataFrame(zone_gdf_full.drop(columns=cols_to_drop, errors='ignore'))
        
        expected_sdoh_cols_list = ['zone_id', 'name', 'population', 'socio_economic_index', 'avg_travel_time_clinic_min']
        for col_sd in expected_sdoh_cols_list:
            if col_sd not in zone_attributes_console_df.columns:
                zone_attributes_console_df[col_sd] = np.nan
                logger.warning(f"({source_context}) Expected SDOH attr '{col_sd}' not in zone data. Added as NaN.")
        logger.info(f"({source_context}) Loaded {len(zone_attributes_console_df)} zone attributes.")
    else:
        logger.warning(f"({source_context}) Zone attribute data (from GDF) unavailable.")
        zone_attributes_console_df = pd.DataFrame(columns=['zone_id', 'name', 'population', 'socio_economic_index'])

    if health_df_enriched_console.empty:
        logger.error(f"({source_context}) Critical: Population health data is empty after all loading/processing steps.")
        
    return health_df_enriched_console, zone_attributes_console_df

# --- Load Main Datasets for the Console ---
health_df_pop_console_main, zone_attr_df_pop_console_main = get_population_analytics_console_dataset_page()

if health_df_pop_console_main.empty:
    st.error("🚨 **Critical Data Failure:** Could not load the primary health dataset for population analytics. Most console features will be unavailable. Please check data sources and ETL processes."); st.stop()

# --- Page Title, Intro & Sidebar Filters ---
st.title(f"📊 {app_config.APP_NAME} - Population Health Analytics & Research Console")
st.markdown("In-depth exploration of demographic distributions, epidemiological patterns, clinical trends, and health system factors using aggregated population-level data.")
st.markdown("---")

if os.path.exists(app_config.APP_LOGO_SMALL): st.sidebar.image(app_config.APP_LOGO_SMALL, width=160)
st.sidebar.header("🔎 Analytics Filters & Controls")

# Date Range Filter (analysts often need broad ranges)
min_date_pop_console_filt = health_df_pop_console_main['encounter_date'].min().date() if 'encounter_date' in health_df_pop_console_main.columns and health_df_pop_console_main['encounter_date'].notna().any() else date.today() - timedelta(days=365*5)
max_date_pop_console_filt = health_df_pop_console_main['encounter_date'].max().date() if 'encounter_date' in health_df_pop_console_main.columns and health_df_pop_console_main['encounter_date'].notna().any() else date.today()
if min_date_pop_console_filt > max_date_pop_console_filt: min_date_pop_console_filt = max_date_pop_console_filt

selected_start_date_console, selected_end_date_console = st.sidebar.date_input(
    "Select Date Range for Analysis:", value=[min_date_pop_console_filt, max_date_pop_console_filt], # Default to full available range
    min_value=min_date_pop_console_filt, max_value=max_date_pop_console_filt, key="pop_console_date_range_v2"
)
if selected_start_date_console > selected_end_date_console:
    st.sidebar.error("Date range error: Start date must be before end date.")
    selected_start_date_console = selected_end_date_console

# Apply Date Filter
analytics_df_for_display_tabs = pd.DataFrame(columns=health_df_pop_console_main.columns) # Init with schema
if 'encounter_date' in health_df_pop_console_main.columns:
    health_df_pop_console_main['encounter_date_obj_for_filt'] = pd.to_datetime(health_df_pop_console_main['encounter_date'], errors='coerce').dt.date
    analytics_df_for_display_tabs = health_df_pop_console_main[
        (health_df_pop_console_main['encounter_date_obj_for_filt'].notna()) &
        (health_df_pop_console_main['encounter_date_obj_for_filt'] >= selected_start_date_console) &
        (health_df_pop_console_main['encounter_date_obj_for_filt'] <= selected_end_date_console)
    ].copy()
else:
    st.error("Core 'encounter_date' column missing. Population analytics cannot proceed.")
    st.stop()

if analytics_df_for_display_tabs.empty:
    st.warning(f"No health data found for the selected period: {selected_start_date_console.strftime('%d %b %Y')} to {selected_end_date_console.strftime('%d %b %Y')}. Try adjusting the date range."); st.stop()

# Optional Condition & Zone Filters (similar to original, but use distinct keys)
cond_opts_pop_console = ["All Conditions (Aggregated)"] + sorted(analytics_df_for_display_tabs['condition'].dropna().unique().tolist())
sel_cond_pop_console_filt = st.sidebar.selectbox("Filter by Condition:", options=cond_opts_pop_console, index=0, key="pop_console_cond_filter_v2")
if sel_cond_pop_console_filt != "All Conditions (Aggregated)": analytics_df_for_display_tabs = analytics_df_for_display_tabs[analytics_df_for_display_tabs['condition'] == sel_cond_pop_console_filt]

zone_opts_pop_console = ["All Zones (Aggregated)"]
zone_id_name_map_pop = {} # For mapping display name back to ID if needed
if not zone_attr_df_pop_console_main.empty and 'zone_id' in zone_attr_df_pop_console_main and 'name' in zone_attr_df_pop_console_main:
    for _, row_zone in zone_attr_df_pop_console_main.drop_duplicates(subset=['zone_id']).iterrows():
        display_z = f"{row_zone['name']} ({row_zone['zone_id']})" if pd.notna(row_zone['name']) and row_zone['name'] != row_zone['zone_id'] else str(row_zone['zone_id'])
        zone_opts_pop_console.append(display_z)
        zone_id_name_map_pop[display_z] = str(row_zone['zone_id'])
elif not zone_attr_df_pop_console_main.empty and 'zone_id' in zone_attr_df_pop_console_main:
    zone_opts_pop_console.extend(sorted(zone_attr_df_pop_console_main['zone_id'].dropna().unique().astype(str).tolist()))

sel_zone_display_pop_console_filt = st.sidebar.selectbox("Filter by Zone:", options=zone_opts_pop_console, index=0, key="pop_console_zone_filter_v2")
if sel_zone_display_pop_console_filt != "All Zones (Aggregated)":
    actual_zone_id_to_filter = zone_id_name_map_pop.get(sel_zone_display_pop_console_filt, sel_zone_display_pop_console_filt) # Get ID if map exists
    if 'zone_id' in analytics_df_for_display_tabs.columns:
        analytics_df_for_display_tabs = analytics_df_for_display_tabs[analytics_df_for_display_tabs['zone_id'] == actual_zone_id_to_filter]

if analytics_df_for_display_tabs.empty and (sel_cond_pop_console_filt != "All Conditions (Aggregated)" or sel_zone_display_pop_console_filt != "All Zones (Aggregated)"):
    st.warning(f"No data found for the specific Condition/Zone filter combination. Broader data for period is shown where applicable, or adjust filters.")
    # analytics_df_for_display_tabs = ... (Logic to fall back to only date-filtered if strict filter empty, or just let tabs show no data)


# --- Top-Level Summary KPIs ---
st.subheader(f"Population Snapshot ({sel_start_pop_console.strftime('%d %b %Y')} - {sel_end_pop_console.strftime('%d %b %Y')})")
# Display selected filters for clarity
filter_str_parts_pop = [f"Cond: {sel_cond_pop_console_filt}", f"Zone: {sel_zone_display_pop_console_filt}"]
st.caption(f"Applied Filters: {', '.join(filter_str_parts_pop)}")

if analytics_df_for_display_tabs.empty:
    st.info("Insufficient data for filtered selection to display population summary KPIs.")
else:
    # Calculate KPI values
    unique_patients_val = analytics_df_for_display_tabs['patient_id'].nunique() if 'patient_id' in analytics_df_for_display_tabs else 0
    avg_risk_val = analytics_df_for_display_tabs['ai_risk_score'].mean() if 'ai_risk_score' in analytics_df_for_display_tabs and analytics_df_for_display_tabs['ai_risk_score'].notna().any() else np.nan
    high_risk_num = 0; high_risk_pct = 0.0
    if 'ai_risk_score' in analytics_df_for_display_tabs and unique_patients_val > 0:
        high_risk_df = analytics_df_for_display_tabs[analytics_df_for_display_tabs['ai_risk_score'] >= app_config.RISK_SCORE_HIGH_THRESHOLD]
        high_risk_num = high_risk_df['patient_id'].nunique() if 'patient_id' in high_risk_df else 0
        high_risk_pct = (high_risk_num / unique_patients_val) * 100
    top_cond_name, top_cond_count = "N/A", 0
    if 'condition' in analytics_df_for_display_tabs and analytics_df_for_display_tabs['condition'].notna().any() and analytics_df_for_display_tabs['condition'].nunique() > 0 :
        counts = analytics_df_for_display_tabs['condition'].value_counts(); top_cond_name, top_cond_count = counts.idxmax(), counts.max()
    
    # Display KPIs using custom markdown (assumes CSS from STYLE_CSS_PATH_WEB defines these)
    cols_kpi_pop_console = st.columns(4)
    with cols_kpi_pop_console[0]: st.markdown(f"""<div class="custom-markdown-kpi-box"><div class="custom-kpi-label-top-condition">Patients (Filtered Set)</div><div class="custom-kpi-value-large">{unique_patients_val:,}</div></div>""", unsafe_allow_html=True)
    with cols_kpi_pop_console[1]: st.markdown(f"""<div class="custom-markdown-kpi-box"><div class="custom-kpi-label-top-condition">Avg. AI Risk Score</div><div class="custom-kpi-value-large">{avg_risk_val:.1f if pd.notna(avg_risk_val) else 'N/A'}</div></div>""", unsafe_allow_html=True)
    with cols_kpi_pop_console[2]: st.markdown(f"""<div class="custom-markdown-kpi-box"><div class="custom-kpi-label-top-condition">% High AI Risk Pts</div><div class="custom-kpi-value-large">{high_risk_pct:.1f}%</div><div class="custom-kpi-subtext-small">({high_risk_num:,} patients)</div></div>""", unsafe_allow_html=True)
    with cols_kpi_pop_console[3]: st.markdown(f"""<div class="custom-markdown-kpi-box highlight-red-edge"><div class="custom-kpi-label-top-condition">Top Condition (Enc.)</div><div class="custom-kpi-value-large">{html.escape(str(top_cond_name))}</div><div class="custom-kpi-subtext-small">{top_cond_count:,} encounters</div></div>""", unsafe_allow_html=True)


# --- Tabbed Interface for Detailed Analytics ---
pop_console_tab_titles = [
    "📈 Epidemiological Overview", "🧑‍🤝‍🧑 Demographics & SDOH",
    "🔬 Clinical & Diagnostics", "⚙️ Health Systems & Equity"
]
tab_pop_console_epi, tab_pop_console_demog, tab_pop_console_clinical, tab_pop_console_systems = st.tabs(pop_console_tab_titles)

# Helper function placeholder to encapsulate data prep for each tab's content
def _prepare_tab_data_population(tab_name: str, df_input: pd.DataFrame, zone_attributes: pd.DataFrame):
    # This function would contain the specific pandas manipulations for the given tab.
    # For now, it's a placeholder concept. The logic from original tabs will be adapted here.
    logger.debug(f"Preparing data for Population Analytics Tab: {tab_name}")
    # Example: if tab_name == "EpiOverview": ... return dict of DataFrames/Series for that tab
    return {"notes": f"Data preparation logic for {tab_name} tab to be fully implemented here using df_input and zone_attributes."}


with tab_pop_console_epi:
    st.header(f"Population Epi Overview")
    if analytics_df_for_display_tabs.empty: st.info("No data after filtering for Epi Overview.")
    else:
        # Call data prep for this tab or perform inline then plot
        # ... Example: Condition counts plot logic from prior version (File 17) ...
        st.markdown("_(Epi Overview: Condition counts, AI Risk distribution, Incidence trends to be shown here)_")


with tab_pop_console_demog:
    st.header("Population Demographics & SDOH Context")
    if analytics_df_for_display_tabs.empty: st.info("No data after filtering for Demographics/SDOH.")
    else:
        # Call data prep or perform inline then plot
        # ... Example: Age distribution plot logic from prior version (File 17) ...
        st.markdown("_(Demographics: Age/Gender distributions. SDOH: Risk vs SES by zone, etc.)_")


with tab_pop_console_clinical:
    st.header("Clinical Insights & Diagnostic Patterns (Population Level)")
    if analytics_df_for_display_tabs.empty: st.info("No data after filtering for Clinical Insights.")
    else:
        # ... Top Symptoms, Test Result Distributions, Positivity Rate Trends ...
        st.markdown("_(Clinical Insights: Top symptoms, test result patterns, positivity trends)_")

with tab_pop_console_systems:
    st.header("Health Systems Performance & Equity (Population Lens)")
    if analytics_df_for_display_tabs.empty: st.info("No data after filtering for Systems/Equity.")
    else:
        # ... Encounters by Clinic Type/Level, Referral Pathways, AI Risk vs SDOH by Subgroups ...
        st.markdown("_(Systems & Equity: Service utilization patterns, referral success by demographics, risk disparities)_")

st.markdown("---")
st.caption(app_config.APP_FOOTER_TEXT)
logger.info("Population Health Analytics & Research Console page generated.")
