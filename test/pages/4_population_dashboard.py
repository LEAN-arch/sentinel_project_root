# sentinel_project_root/test/pages/4_population_dashboard.py
# Population Health Analytics & Research Console for Sentinel Health Co-Pilot.

import streamlit as st
import pandas as pd
import numpy as np
import os
# import sys # sys.path manipulation removed, assuming Streamlit handles root correctly
import logging
from datetime import date, timedelta, datetime # Added datetime
import plotly.express as px # For direct use of complex plots like histograms
import html # For escaping text in custom HTML elements

# --- Sentinel System Imports ---
from config import app_config
from utils.core_data_processing import (
    load_health_records, load_zone_data # For SDOH context from zone attributes
    # get_trend_data might be used by specific analyses within tabs
)
from utils.ai_analytics_engine import apply_ai_models # Assumes data is enriched, or apply here
from utils.ui_visualization_helpers import (
    plot_bar_chart_web,
    # plot_donut_chart_web, # Import if used
    # plot_annotated_line_chart_web, # Import if used
    _create_empty_plot_figure # For graceful empty states
)

# --- Page Configuration ---
st.set_page_config(
    page_title=f"Population Analytics - {app_config.APP_NAME}", # Shortened title
    layout="wide",
    initial_sidebar_state="expanded"
)
logger = logging.getLogger(__name__)

# --- Data Loading for Population Analytics Console ---
@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS_WEB_REPORTS, show_spinner="Loading population analytics dataset...")
def get_population_analytics_dataset(source_context: str = "PopAnalyticsConsole/Load"):
    """
    Loads and prepares a comprehensive dataset for population-level analytics.
    Simulates accessing an enriched data warehouse or data lake.
    """
    logger.info(f"({source_context}) Loading population health records and zone attribute data.")
    
    health_raw = load_health_records(source_context=f"{source_context}/Health")
    
    enriched_health_df = pd.DataFrame()
    ai_cols = ['ai_risk_score', 'ai_followup_priority_score'] # Expected from AI
    base_health_cols_pop = health_raw.columns.tolist() if isinstance(health_raw, pd.DataFrame) and not health_raw.empty else []

    if isinstance(health_raw, pd.DataFrame) and not health_raw.empty:
        enriched_output = apply_ai_models(health_raw, source_context=f"{source_context}/AIEnrich")
        enriched_health_df = enriched_output[0] 
    else:
        logger.warning(f"({source_context}) Raw health records empty or invalid. Population analytics will be limited.")
        enriched_health_df = pd.DataFrame(columns=list(set(base_health_cols_pop + ai_cols)))

    # Load zone attributes for SDOH context (non-geometric part)
    zone_gdf = load_zone_data(source_context=f"{source_context}/Zone") # Returns merged GDF
    zone_attributes_df = pd.DataFrame()
    expected_sdoh_cols = ['zone_id', 'name', 'population', 'socio_economic_index', 'avg_travel_time_clinic_min', 'predominant_hazard_type', 'primary_livelihood']

    if isinstance(zone_gdf, gpd.GeoDataFrame) and not zone_gdf.empty:
        # Drop geometry to get attributes
        geom_col_name = zone_gdf.geometry.name if hasattr(zone_gdf, 'geometry') else 'geometry'
        zone_attributes_df = pd.DataFrame(zone_gdf.drop(columns=[geom_col_name], errors='ignore'))
        
        # Ensure key SDOH columns are present, add as NaN if missing
        for col_sdoh in expected_sdoh_cols:
            if col_sdoh not in zone_attributes_df.columns:
                zone_attributes_df[col_sdoh] = np.nan
                logger.debug(f"({source_context}) Expected SDOH attribute '{col_sdoh}' missing, added as NaN.")
        logger.info(f"({source_context}) Loaded {len(zone_attributes_df)} zone attributes for SDOH context.")
    else:
        logger.warning(f"({source_context}) Zone attributes data (GDF) unavailable. SDOH analytics limited.")
        zone_attributes_df = pd.DataFrame(columns=expected_sdoh_cols) # Empty schema

    if enriched_health_df.empty:
        logger.error(f"({source_context}) CRITICAL: Health data remains empty after processing for population analytics.")
        
    return enriched_health_df, zone_attributes_df

# --- Load Datasets ---
health_data_main_pop, zone_attributes_main_pop = get_population_analytics_dataset()

if health_data_main_pop.empty:
    st.error("🚨 **Critical Data Failure:** Primary health dataset for population analytics could not be loaded or is empty. Most console features will be unavailable.")
    st.stop() # Hard stop if no health data

# --- Page Title and Introduction ---
st.title(f"📊 {app_config.APP_NAME} - Population Health Analytics & Research Console")
st.markdown("Deep-dive exploration of demographic distributions, epidemiological patterns, clinical trends, and health system factors using aggregated population-level data.")
st.divider()

# --- Sidebar Filters ---
if os.path.exists(app_config.APP_LOGO_SMALL): st.sidebar.image(app_config.APP_LOGO_SMALL, width=150)
st.sidebar.header("🔎 Analytics Filters")

# Date Range
min_data_date = pd.NaT
max_data_date = pd.NaT
if 'encounter_date' in health_data_main_pop.columns and health_data_main_pop['encounter_date'].notna().any():
    min_data_date = health_data_main_pop['encounter_date'].min().date()
    max_data_date = health_data_main_pop['encounter_date'].max().date()

if pd.isna(min_data_date) or pd.isna(max_data_date): # Fallback if no valid dates in data
    logger.warning("Could not determine min/max dates from health data for population console.")
    max_data_date = date.today()
    min_data_date = max_data_date - timedelta(days=365*2) # Default 2 years back

selected_start_date_pop, selected_end_date_pop = st.sidebar.date_input(
    "Select Date Range for Analysis:", value=[min_data_date, max_data_date],
    min_value=min_data_date, max_value=max_data_date, key="pop_console_date_range_filter"
)
if selected_start_date_pop > selected_end_date_pop:
    st.sidebar.error("Start date must be on or before end date.")
    selected_start_date_pop = selected_end_date_pop # Auto-correct

# Filter by Date
if 'encounter_date' not in health_data_main_pop.columns:
    st.error("Critical error: 'encounter_date' column is missing from the health dataset. Cannot proceed.")
    st.stop()

# Ensure 'encounter_date' is datetime before trying to access .dt.date
if not pd.api.types.is_datetime64_any_dtype(health_data_main_pop['encounter_date']):
    health_data_main_pop['encounter_date'] = pd.to_datetime(health_data_main_pop['encounter_date'], errors='coerce')

df_filtered_by_date_pop = health_data_main_pop[
    (health_data_main_pop['encounter_date'].notna()) &
    (health_data_main_pop['encounter_date'].dt.date >= selected_start_date_pop) &
    (health_data_main_pop['encounter_date'].dt.date <= selected_end_date_pop)
].copy()

# --- Apply further filters progressively ---
analytics_df_current = df_filtered_by_date_pop

# Condition Filter
if 'condition' in analytics_df_current.columns:
    unique_conditions_list = ["All Conditions (Aggregated)"] + sorted(analytics_df_current['condition'].dropna().unique().tolist())
    selected_condition_filter_pop = st.sidebar.selectbox(
        "Filter by Condition Group (Optional):", options=unique_conditions_list, index=0,
        key="pop_console_condition_selector"
    )
    if selected_condition_filter_pop != "All Conditions (Aggregated)":
        analytics_df_current = analytics_df_current[analytics_df_current['condition'] == selected_condition_filter_pop]
else:
    selected_condition_filter_pop = "All Conditions (Aggregated)" # Set default if column missing
    st.sidebar.caption("Condition filter unavailable (column missing).")


# Zone Filter
zone_filter_options_pop = ["All Zones (Aggregated)"]
zone_id_to_display_map = {} # For mapping display name back to actual zone_id if needed

if isinstance(zone_attributes_main_pop, pd.DataFrame) and not zone_attributes_main_pop.empty and \
   'zone_id' in zone_attributes_main_pop.columns:
    # Create display names, handling missing 'name'
    for _, row_zone in zone_attributes_main_pop.iterrows():
        zid = str(row_zone['zone_id'])
        zname = str(row_zone.get('name', zid)) # Use zone_id if name is missing/NaN
        display_name = f"{zname} ({zid})" if zname != zid and zname != "Unknown" else zid
        zone_filter_options_pop.append(display_name)
        zone_id_to_display_map[display_name] = zid
    zone_filter_options_pop = sorted(list(set(zone_filter_options_pop))) # Ensure unique and sorted
else:
    st.sidebar.caption("Zone filter options limited (zone attributes missing).")

selected_zone_display_filter_pop = st.sidebar.selectbox(
    "Filter by Zone (Optional):", options=zone_filter_options_pop, index=0, 
    key="pop_console_zone_selector"
)
if selected_zone_display_filter_pop != "All Zones (Aggregated)":
    actual_zone_id_to_filter = zone_id_to_display_map.get(selected_zone_display_filter_pop, selected_zone_display_filter_pop) # Get ID from map
    if 'zone_id' in analytics_df_current.columns:
        analytics_df_current = analytics_df_current[analytics_df_current['zone_id'] == actual_zone_id_to_filter]
    else:
        st.sidebar.caption("Cannot filter by zone ('zone_id' column missing in health data).")

# Feedback if filters result in empty data
if analytics_df_current.empty:
    st.warning(f"No health data available for the selected filters (Period: {selected_start_date_pop.strftime('%d %b %Y')} to {selected_end_date_pop.strftime('%d %b %Y')}, Condition: {selected_condition_filter_pop}, Zone: {selected_zone_display_filter_pop}). Please broaden your filter criteria.")
    # Don't st.stop() here, allow user to change filters. Tabs will handle empty analytics_df_current.

# --- Top-Level Population Summary KPIs ---
st.subheader(f"Population Health Snapshot ({selected_start_date_pop.strftime('%d %b %Y')} - {selected_end_date_pop.strftime('%d %b %Y')}, Filter: {selected_condition_filter_pop}, {selected_zone_display_filter_pop})")
if analytics_df_current.empty:
    st.info("Insufficient data after filtering to display population summary KPIs.")
else:
    cols_kpi_pop = st.columns(4)
    
    total_unique_patients = analytics_df_current['patient_id'].nunique() if 'patient_id' in analytics_df_current.columns else 0
    avg_ai_risk = np.nan
    if 'ai_risk_score' in analytics_df_current.columns and analytics_df_current['ai_risk_score'].notna().any():
        avg_ai_risk = analytics_df_current['ai_risk_score'].mean()
    
    high_risk_patient_count = 0; percent_high_risk = 0.0
    if 'ai_risk_score' in analytics_df_current.columns and total_unique_patients > 0:
        high_risk_df = analytics_df_current[analytics_df_current['ai_risk_score'] >= app_config.RISK_SCORE_HIGH_THRESHOLD]
        high_risk_patient_count = high_risk_df['patient_id'].nunique() if 'patient_id' in high_risk_df.columns else 0
        percent_high_risk = (high_risk_patient_count / total_unique_patients) * 100 if total_unique_patients > 0 else 0.0

    top_condition_name, top_condition_encounters = "N/A", 0
    if 'condition' in analytics_df_current.columns and analytics_df_current['condition'].notna().any():
        condition_counts = analytics_df_current['condition'].value_counts() # Counts encounters per condition
        if not condition_counts.empty:
            top_condition_name = condition_counts.idxmax()
            top_condition_encounters = condition_counts.max()

    with cols_kpi_pop[0]: st.markdown(f"""<div class="custom-markdown-kpi-box"><div class="custom-kpi-label-top-condition">Total Unique Patients</div><div class="custom-kpi-value-large">{total_unique_patients:,}</div></div>""", unsafe_allow_html=True)
    with cols_kpi_pop[1]: st.markdown(f"""<div class="custom-markdown-kpi-box"><div class="custom-kpi-label-top-condition">Avg. AI Risk Score</div><div class="custom-kpi-value-large">{avg_ai_risk:.1f if pd.notna(avg_ai_risk) else 'N/A'}</div></div>""", unsafe_allow_html=True)
    with cols_kpi_pop[2]: st.markdown(f"""<div class="custom-markdown-kpi-box"><div class="custom-kpi-label-top-condition">% High AI Risk Patients</div><div class="custom-kpi-value-large">{percent_high_risk:.1f}%</div><div class="custom-kpi-subtext-small">({high_risk_patient_count:,} patients)</div></div>""", unsafe_allow_html=True)
    with cols_kpi_pop[3]: st.markdown(f"""<div class="custom-markdown-kpi-box highlight-red-edge"><div class="custom-kpi-label-top-condition">Top Condition (Encounters)</div><div class="custom-kpi-value-large">{html.escape(str(top_condition_name))}</div><div class="custom-kpi-subtext-small">{top_condition_encounters:,} encounters</div></div>""", unsafe_allow_html=True)

# --- Tabbed Interface for Detailed Population Analytics ---
tab_titles_pop = ["📈 Epidemiological Overview", "🧑‍🤝‍🧑 Demographics & SDOH", "🔬 Clinical Insights", "⚙️ Health Systems Lens"]
tab_pop_epi, tab_pop_demog, tab_pop_clinical, tab_pop_systems = st.tabs(tab_titles_pop)

with tab_pop_epi:
    st.header(f"Epidemiological Overview (Filter: {selected_condition_filter_pop} | {selected_zone_display_filter_pop})")
    if analytics_df_current.empty:
        st.info("No data available for Epidemiological Overview with current filters.")
    else:
        # Condition Case Counts (Unique Patients)
        if 'condition' in analytics_df_current.columns and 'patient_id' in analytics_df_current.columns:
            condition_patient_counts = analytics_df_current.groupby('condition')['patient_id'].nunique().nlargest(12).reset_index(name='unique_patients')
            if not condition_patient_counts.empty:
                st.plotly_chart(plot_bar_chart_web(
                    condition_patient_counts, x_col='condition', y_col='unique_patients', 
                    title="Top Conditions by Unique Patient Count (Filtered Set)", 
                    orientation='h', y_is_count=True, chart_height=450,
                    x_axis_label="Unique Patient Count", y_axis_label="Condition"
                ), use_container_width=True)
            else: st.caption("No aggregated condition counts available for unique patients with current filters.")
        
        # AI Risk Score Distribution (Histogram)
        if 'ai_risk_score' in analytics_df_current.columns and analytics_df_current['ai_risk_score'].notna().any():
            fig_risk_dist = px.histogram(
                analytics_df_current.dropna(subset=['ai_risk_score']), x="ai_risk_score", nbins=30,
                title="Patient AI Risk Score Distribution (Filtered Set)",
                labels={'ai_risk_score': 'AI Risk Score', 'count': 'Number of Patients/Encounters'} # Adjust based on what rows represent
            )
            fig_risk_dist.update_layout(bargap=0.1, height=app_config.WEB_PLOT_COMPACT_HEIGHT)
            st.plotly_chart(fig_risk_dist, use_container_width=True)
        
        st.caption("Note: Incidence/prevalence trends require careful definition of 'new case' vs 'active case' for population view. This section provides overview counts and distributions.")

with tab_pop_demog:
    st.header("Demographics & Social Determinants of Health (SDOH) Context")
    if analytics_df_current.empty:
        st.info("No data available for Demographics & SDOH analysis with current filters.")
    else:
        st.markdown("_(Placeholder: Detailed charts for Age Distribution, Gender Distribution. For SDOH, merge `analytics_df_current` with `zone_attributes_main_pop` on `zone_id` to correlate health outcomes like AI Risk with Zone SES, Travel Time, etc.)_")
        if zone_attributes_main_pop.empty:
            st.caption("Zone attribute data (for SDOH) is unavailable.")

with tab_pop_clinical:
    st.header("Clinical Insights & Diagnostic Patterns")
    if analytics_df_current.empty:
        st.info("No data available for Clinical Insights with current filters.")
    else:
        st.markdown("_(Placeholder: Analyses for Top Reported Symptoms (frequency/trends), Test Result Distributions (e.g., % Positive for key tests over time or by demographic), Test Positivity Trends by specific conditions or risk groups.)_")

with tab_pop_systems:
    st.header("Health Systems Performance & Equity Lens")
    if analytics_df_current.empty:
        st.info("No data available for Health Systems & Equity analysis with current filters.")
    else:
        st.markdown("_(Placeholder: Analyses for Patient Encounters by Clinic/Zone, Referral Pathway Completion Rates (if data allows), AI Risk Score Variations by SDOH factors like Zone SES or predominant livelihood to explore equity.)_")

st.divider()
st.caption(app_config.APP_FOOTER_TEXT)
logger.info(f"Population Health Analytics Console page generated for period: {selected_start_date_pop.isoformat()} to {selected_end_date_pop.isoformat()}, Cond: {selected_condition_filter_pop}, Zone: {selected_zone_display_filter_pop}")
