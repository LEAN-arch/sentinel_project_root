# sentinel_project_root/test/pages/4_population_dashboard.py
# Redesigned as "Population Health Analytics & Research Console" for "Sentinel Health Co-Pilot"
# This page simulates a web interface for epidemiologists, researchers, and program managers,
# typically at a Tier 3 (Regional/Cloud Node) or an advanced Tier 2 (Facility Node)
# with access to broader, aggregated datasets for in-depth analysis.

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys # For existing path manipulation, review if necessary based on Streamlit's PYTHONPATH for pages.
import logging
from datetime import date, timedelta
import plotly.express as px # Kept for custom/complex plots if _web helpers are insufficient.
import html # For escaping text in custom HTML KPIs

# --- Sentinel System Imports ---
# Ensure this path logic works for Streamlit's execution context of pages.
# Usually, Streamlit adds the repo root, and then if app_home.py is in 'test/', 'test/' might be effectively the root.
# So, 'from config...' should find 'test/config...'. If issues, use more robust relative/absolute pathing.
PROJECT_ROOT_SIMULATED_FROM_PAGE = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT_SIMULATED_FROM_PAGE not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_SIMULATED_FROM_PAGE)

from config import app_config # Uses new, redesigned app_config
from utils.core_data_processing import (
    load_health_records, # For loading base research dataset
    load_zone_data,      # For SDOH context from zone attributes
    get_trend_data       # General utility for trends
)
# AI models assumed to be applied upstream; this module consumes AI-enriched data.
from utils.ai_analytics_engine import apply_ai_models # Or directly load data pre-enriched.
from utils.ui_visualization_helpers import (
    plot_bar_chart_web,
    plot_donut_chart_web,
    plot_annotated_line_chart_web,
    _create_empty_plot_figure
)

# --- Page Configuration ---
st.set_page_config(
    page_title=f"Population Analytics Console - {app_config.APP_NAME}",
    layout="wide",
    initial_sidebar_state="expanded"
)
logger = logging.getLogger(__name__)
# CSS would be loaded by app_home.py

# --- Data Loading for Population Analytics Console (Simulates Tier 3 Data Warehouse Access) ---
@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS_WEB_REPORTS, show_spinner="Loading and preparing population analytics dataset...")
def get_population_analytics_console_dataset(source_context="PopAnalyticsConsole/DataLoad"):
    """
    Loads and prepares a potentially large, comprehensive dataset for population-level analytics.
    Simulates accessing an enriched data warehouse or data lake at a Tier 3 node.
    """
    logger.info(f"({source_context}) Loading population health records and zone attribute data.")
    
    # 1. Load comprehensive health records (assumed to be already AI-enriched in a real Tier 3 ETL)
    # For this simulation, we load raw and then apply AI models here.
    health_df_raw_pop_console = load_health_records(file_path=app_config.HEALTH_RECORDS_CSV, source_context=source_context)
    
    health_df_enriched_for_console = pd.DataFrame()
    if not health_df_raw_pop_console.empty:
        enriched_output = apply_ai_models(health_df_raw_pop_console, source_context=f"{source_context}/AIEnrich")
        health_df_enriched_for_console = enriched_output[0] # [0] is the enriched health_df
    else:
        logger.warning(f"({source_context}) Raw health records empty. Population analytics will be limited.")
        # Provide a DataFrame with expected schema if loading fails
        base_cols_health_pop = health_df_raw_pop_console.columns.tolist() if health_df_raw_pop_console is not None else []
        ai_cols_pop = ['ai_risk_score', 'ai_followup_priority_score']
        health_df_enriched_for_console = pd.DataFrame(columns=list(set(base_cols_health_pop + ai_cols_pop)))


    # 2. Load zone attributes for SDOH context (non-geometric part)
    zone_gdf_pop_console = load_zone_data(source_context=source_context) # Returns merged GDF
    zone_attributes_for_console = pd.DataFrame()
    if zone_gdf_pop_console is not None and not zone_gdf_pop_console.empty:
        geom_col = zone_gdf_pop_console.geometry.name if hasattr(zone_gdf_pop_console, 'geometry') else 'geometry'
        if geom_col in zone_gdf_pop_console.columns:
            zone_attributes_for_console = pd.DataFrame(zone_gdf_pop_console.drop(columns=[geom_col], errors='ignore'))
        else: # Assume it's already just attributes if no clear geometry column
            zone_attributes_for_console = pd.DataFrame(zone_gdf_pop_console)
        
        # Ensure key SDOH-related columns used in original logic are present
        expected_sdoh_cols = ['zone_id', 'name', 'population', 'socio_economic_index', 'avg_travel_time_clinic_min']
        for col_sdoh in expected_sdoh_cols:
            if col_sdoh not in zone_attributes_for_console.columns:
                zone_attributes_for_console[col_sdoh] = np.nan # Add as NaN if missing
                logger.warning(f"({source_context}) Expected SDOH attribute '{col_sdoh}' missing from zone data. Added as NaN.")
        logger.info(f"({source_context}) Loaded {len(zone_attributes_for_console)} zone attributes for SDOH context.")
    else:
        logger.warning(f"({source_context}) Zone attributes data unavailable for population analytics.")
        zone_attributes_for_console = pd.DataFrame(columns=['zone_id', 'name', 'population', 'socio_economic_index'])


    if health_df_enriched_for_console.empty:
        logger.error(f"({source_context}) Critical: Health data remains empty after all processing steps for population analytics.")
        
    return health_df_enriched_for_console, zone_attributes_for_console

# --- Load Datasets ---
health_df_analytics_main, zone_attr_df_analytics_main = get_population_analytics_console_dataset()

if health_df_analytics_main.empty:
    st.error("🚨 **Critical Data Failure:** Primary health dataset for population analytics could not be loaded or is empty. Most console features will be unavailable."); st.stop()

# --- Page Title and Introduction ---
st.title(f"📊 {app_config.APP_NAME} - Population Health Analytics & Research Console")
st.markdown("Deep-dive exploration of demographic distributions, epidemiological patterns, clinical trends, and health system factors using aggregated population-level data.")
st.markdown("---")

# --- Sidebar Filters ---
if os.path.exists(app_config.APP_LOGO_SMALL): st.sidebar.image(app_config.APP_LOGO_SMALL, width=180)
st.sidebar.header("🔎 Analytics Filters")

# Date Range (Allow broad range, default to all available if not specified)
min_date_pop_console = health_df_analytics_main['encounter_date'].min().date() if 'encounter_date' in health_df_analytics_main.columns and health_df_analytics_main['encounter_date'].notna().any() else date.today() - timedelta(days=365*5) # 5 years back if no data
max_date_pop_console = health_df_analytics_main['encounter_date'].max().date() if 'encounter_date' in health_df_analytics_main.columns and health_df_analytics_main['encounter_date'].notna().any() else date.today()
if min_date_pop_console > max_date_pop_console: min_date_pop_console = max_date_pop_console

# Default to full available range for analysts unless specified
sel_start_pop_console, sel_end_pop_console = st.sidebar.date_input(
    "Select Date Range for Analysis:", value=[min_date_pop_console, max_date_pop_console],
    min_value=min_date_pop_console, max_value=max_date_pop_console, key="pop_console_date_range"
)
if sel_start_pop_console > sel_end_pop_console:
    st.sidebar.error("Start date must be before end date for population analytics.")
    sel_start_pop_console = sel_end_pop_console

# Prepare encounter_date_obj if not present, then filter by date
if 'encounter_date' in health_df_analytics_main.columns:
    health_df_analytics_main['encounter_date_obj'] = pd.to_datetime(health_df_analytics_main['encounter_date'], errors='coerce').dt.date
    analytics_df_for_tabs = health_df_analytics_main[
        (health_df_analytics_main['encounter_date_obj'].notna()) &
        (health_df_analytics_main['encounter_date_obj'] >= sel_start_pop_console) &
        (health_df_analytics_main['encounter_date_obj'] <= sel_end_pop_console)
    ].copy()
else: # Critical column missing
    st.error("'encounter_date' column is essential and missing from the loaded health data.")
    analytics_df_for_tabs = pd.DataFrame() # Ensure it's an empty DF
    st.stop()


if analytics_df_for_tabs.empty:
    st.warning(f"No health data available for the selected period: {sel_start_pop_console.strftime('%d %b %Y')} to {sel_end_pop_console.strftime('%d %b %Y')}. Please adjust filters or verify data sources."); st.stop()

# Optional Condition Filter (Populated from the period-filtered data)
unique_conditions_pop = sorted(analytics_df_for_tabs['condition'].dropna().unique().tolist())
cond_list_pop_console = ["All Conditions (Aggregated)"] + unique_conditions_pop
sel_cond_pop_console = st.sidebar.selectbox(
    "Filter by Condition Group (Optional):", options=cond_list_pop_console, index=0,
    key="pop_console_condition_filter"
)
if sel_cond_pop_console != "All Conditions (Aggregated)":
    analytics_df_for_tabs = analytics_df_for_tabs[analytics_df_for_tabs['condition'] == sel_cond_pop_console]

# Optional Zone Filter
zone_names_pop = ["All Zones (Aggregated)"]
if not zone_attr_df_analytics_main.empty and 'name' in zone_attr_df_analytics_main and 'zone_id' in zone_attr_df_analytics_main:
    # Create "Display Name (ID)" for clarity if names not unique, map back to ID
    zone_map_pop = dict(zip(
        zone_attr_df_analytics_main.apply(lambda x: f"{x['name']} ({x['zone_id']})" if pd.notna(x['name']) and x['name']!=x['zone_id'] else x['zone_id'], axis=1),
        zone_attr_df_analytics_main['zone_id']
    ))
    zone_names_pop.extend(sorted(zone_map_pop.keys()))
elif not zone_attr_df_analytics_main.empty and 'zone_id' in zone_attr_df_analytics_main: # Fallback to zone_id if no 'name'
    zone_names_pop.extend(sorted(zone_attr_df_analytics_main['zone_id'].dropna().unique().tolist()))

sel_zone_display_pop_console = st.sidebar.selectbox(
    "Filter by Zone (Optional):", options=zone_names_pop, index=0, key="pop_console_zone_filter"
)
if sel_zone_display_pop_console != "All Zones (Aggregated)":
    actual_zone_id_filter = zone_map_pop.get(sel_zone_display_pop_console, sel_zone_display_pop_console) # Get ID from map or use value if it's already ID
    if 'zone_id' in analytics_df_for_tabs.columns:
        analytics_df_for_tabs = analytics_df_for_tabs[analytics_df_for_tabs['zone_id'] == actual_zone_id_filter]

if analytics_df_for_tabs.empty and (sel_cond_pop_console != "All Conditions (Aggregated)" or sel_zone_display_pop_console != "All Zones (Aggregated)"):
    st.warning(f"No data found for the specific combination of filters (Condition: {sel_cond_pop_console}, Zone: {sel_zone_display_pop_console}). Consider broadening your filter criteria or checking underlying data availability.");
    # To prevent full stop if some broader data exists:
    # analytics_df_for_tabs = analytics_df_period_filtered.copy() # Revert to only date-filtered, for example. This requires careful thought.


# --- Top-Level Population Summary KPIs (using custom HTML as per original, ensure CSS supports this) ---
st.subheader(f"Population Health Snapshot ({sel_start_pop_console.strftime('%d %b %Y')} - {sel_end_pop_console.strftime('%d %b %Y')}, Filter: {sel_cond_pop_console}, {sel_zone_display_pop_console})")
if analytics_df_for_tabs.empty:
    st.info("Insufficient data after filtering to display population summary KPIs.")
else:
    kpi_cols_pop_console = st.columns(4) # Number of high-level summary boxes
    # Calculate KPI values based on analytics_df_for_tabs
    unique_patients_pop_kpi = analytics_df_for_tabs['patient_id'].nunique() if 'patient_id' in analytics_df_for_tabs else 0
    avg_risk_pop_kpi = analytics_df_for_tabs['ai_risk_score'].mean() if 'ai_risk_score' in analytics_df_for_tabs and analytics_df_for_tabs['ai_risk_score'].notna().any() else np.nan
    
    high_risk_count_pop_kpi = 0; prop_high_risk_kpi = 0.0
    if 'ai_risk_score' in analytics_df_for_tabs and unique_patients_pop_kpi > 0:
        high_risk_kpi_df = analytics_df_for_tabs[analytics_df_for_tabs['ai_risk_score'] >= app_config.RISK_SCORE_HIGH_THRESHOLD]
        high_risk_count_pop_kpi = high_risk_kpi_df['patient_id'].nunique() if 'patient_id' in high_risk_kpi_df else 0
        prop_high_risk_kpi = (high_risk_count_pop_kpi / unique_patients_pop_kpi) * 100

    top_cond_pop_kpi_name, top_cond_pop_kpi_count = "N/A", 0
    if 'condition' in analytics_df_for_tabs and analytics_df_for_tabs['condition'].notna().any():
        counts_top_cond = analytics_df_for_tabs['condition'].value_counts()
        if not counts_top_cond.empty: top_cond_pop_kpi_name, top_cond_pop_kpi_count = counts_top_cond.idxmax(), counts_top_cond.max()

    # Render using the custom markdown KPI box style (assumes CSS is loaded and defines these classes)
    with kpi_cols_pop_console[0]: st.markdown(f"""<div class="custom-markdown-kpi-box"><div class="custom-kpi-label-top-condition">Total Unique Patients</div><div class="custom-kpi-value-large">{unique_patients_pop_kpi:,}</div></div>""", unsafe_allow_html=True)
    with kpi_cols_pop_console[1]: st.markdown(f"""<div class="custom-markdown-kpi-box"><div class="custom-kpi-label-top-condition">Avg. AI Risk Score</div><div class="custom-kpi-value-large">{avg_risk_pop_kpi:.1f if pd.notna(avg_risk_pop_kpi) else 'N/A'}</div></div>""", unsafe_allow_html=True)
    with kpi_cols_pop_console[2]: st.markdown(f"""<div class="custom-markdown-kpi-box"><div class="custom-kpi-label-top-condition">% High AI Risk Patients</div><div class="custom-kpi-value-large">{prop_high_risk_kpi:.1f}%</div><div class="custom-kpi-subtext-small">({high_risk_count_pop_kpi:,} patients)</div></div>""", unsafe_allow_html=True)
    with kpi_cols_pop_console[3]: st.markdown(f"""<div class="custom-markdown-kpi-box highlight-red-edge"><div class="custom-kpi-label-top-condition">Top Condition (Encounters)</div><div class="custom-kpi-value-large">{html.escape(str(top_cond_pop_kpi_name))}</div><div class="custom-kpi-subtext-small">{top_cond_pop_kpi_count:,} encounters</div></div>""", unsafe_allow_html=True)


# --- Tabbed Interface for Detailed Population Analytics ---
population_analytics_tab_names = [ # Using clearer names from previous iteration
    "📈 Epidemiological Overview", "🧑‍🤝‍🧑 Demographics & SDOH Context",
    "🔬 Clinical Insights & Diagnostics", "⚙️ Health Systems & Equity Lens"
]
tab_pop_epi, tab_pop_demog, tab_pop_clinical, tab_pop_systems = st.tabs(population_analytics_tab_names)

# --- Data Preparation and Display Logic for Each Tab ---
# Each tab will now ideally:
# 1. Call a dedicated data preparation function (local to this file or imported)
#    that takes `analytics_df_for_tabs` and `zone_attr_df_analytics_main` (if needed for SDOH).
# 2. Use the returned structured data to call the `_web` suffixed plotting functions.
# For this refactoring, the data manipulation might still be inline but use the new plotters.

with tab_pop_epi:
    st.header(f"Epidemiological Overview (Filter: {sel_cond_pop_console} | {sel_zone_display_pop_console})")
    if not analytics_df_for_tabs.empty:
        # Example: Condition Case Counts (Unique Patients)
        if 'condition' in analytics_df_for_tabs and 'patient_id' in analytics_df_for_tabs:
            cond_counts_unique_patients_df = analytics_df_for_tabs.groupby('condition')['patient_id'].nunique().nlargest(12).reset_index(name='unique_patients_count')
            if not cond_counts_unique_patients_df.empty:
                st.plotly_chart(plot_bar_chart_web(cond_counts_unique_patients_df, x_col_bar='condition', y_col_bar='unique_patients_count', title_bar="Top Conditions by Unique Patient Count", orientation_web='h', y_axis_is_count=True, chart_height=450), use_container_width=True)
            else: st.caption("No aggregated condition counts for unique patients.")
        
        # Example: AI Risk Score Distribution (Histogram)
        if 'ai_risk_score' in analytics_df_for_tabs and analytics_df_for_tabs['ai_risk_score'].notna().any():
            # For histogram with Plotly Express (px.histogram) directly, as no _web helper for histogram specifically
            # Ensure theme consistency.
            fig_risk_hist = px.histogram(analytics_df_for_tabs.dropna(subset=['ai_risk_score']), x="ai_risk_score", nbins=25, title="Patient AI Risk Score Distribution (Filtered Set)")
            fig_risk_hist.update_layout(bargap=0.1, height=app_config.WEB_PLOT_COMPACT_HEIGHT) # Use app_config for height
            st.plotly_chart(fig_risk_hist, use_container_width=True)
        
        # Placeholder for Incidence Trends - requires careful data prep for "new cases" definition
        # weekly_new_cases_df = prepare_pop_incidence_trend_data(analytics_df_for_tabs, ...)
        # st.plotly_chart(plot_annotated_line_chart_web(weekly_new_cases_df, ...))
        st.caption("Note: Incidence trends for specific conditions would require further definition of 'new case' for this population view.")
    else:
        st.info("Insufficient data after filtering for Epidemiological Overview.")

with tab_pop_demog:
    st.header("Demographics & Social Determinants of Health (SDOH) Context")
    if not analytics_df_for_tabs.empty:
        # Example: Age Distribution
        # ... (logic for age_dist_data_pop from previous refactor) ...
        # st.plotly_chart(plot_bar_chart_web(age_dist_data_pop, ...))
        st.markdown("_(Demographic charts: Age Distribution, Gender Distribution - data prep & plotting to be fully implemented here using analytics_df_for_tabs)_")

        # Example: SDOH context if zone attributes available
        if not zone_attr_df_analytics_main.empty and 'zone_id' in analytics_df_for_tabs.columns:
            # Merge filtered health data with zone attributes for SDOH plots
            # data_for_sdoh_plots = pd.merge(analytics_df_for_tabs.drop_duplicates(subset=['patient_id','zone_id']), zone_attr_df_analytics_main, on='zone_id', how='left')
            # Example: Scatter plot AI Risk vs Socio-Economic Index by Zone (if relevant metrics available)
            # fig_sdoh_scatter = px.scatter(data_for_sdoh_plots, x='socio_economic_index', y='ai_risk_score', color='zone_name', ...)
            # st.plotly_chart(fig_sdoh_scatter, use_container_width=True)
            st.markdown("_(SDOH charts: Linking health outcomes like AI Risk to Zone SES, Travel Time - requires merging health_df with zone_attr_df and careful aggregation)_")
    else:
        st.info("Insufficient data after filtering for Demographics & SDOH analysis.")


with tab_pop_clinical:
    st.header("Clinical Insights & Diagnostic Patterns")
    # ... (Data prep and plotting for Top Symptoms, Test Result Distributions, Test Positivity Trends) ...
    st.markdown("_(Clinical Insights: Top Symptoms, Test Result Distributions, Test Positivity Trends - data prep & plotting to be implemented here using analytics_df_for_tabs)_")

with tab_pop_systems:
    st.header("Health Systems Performance & Equity Lens")
    # ... (Data prep and plotting for Encounters by Clinic, Referral Status, AI Risk by SDOH variables like SES) ...
    st.markdown("_(Systems & Equity: Encounters by Clinic, Referral Funnel, AI Risk vs SDOH factors - data prep & plotting to be implemented here)_")


st.markdown("---"); st.caption(app_config.APP_FOOTER_TEXT)
logger.info(f"Population Health Analytics & Research Console page generated for period: {sel_start_pop_console.strftime('%Y-%m-%d')} to {sel_end_pop_console.strftime('%Y-%m-%d')}")
