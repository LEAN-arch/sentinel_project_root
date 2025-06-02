# sentinel_project_root/test/pages/4_population_dashboard.py
# Population Health Analytics & Research Console for Sentinel Health Co-Pilot.

import streamlit as st # Should be one of the first Streamlit imports
import sys # For path manipulation
import os # For path manipulation

# --- Robust Path Setup for Imports ---
_current_file_directory_pop = os.path.dirname(os.path.abspath(__file__))
_project_test_root_directory_pop = os.path.abspath(os.path.join(_current_file_directory_pop, os.pardir))
if _project_test_root_directory_pop not in sys.path:
    sys.path.insert(0, _project_test_root_directory_pop)

# --- Sentinel System Imports (Now after path setup) ---
try:
    # This is where the 'No module named 'plotly.express'' likely occurs if plotly is not installed
    import plotly.express as px # For direct use of complex plots like histograms
    import pandas as pd
    import numpy as np
    import logging
    from datetime import date, timedelta 
    import html 

    from config import app_config
    from utils.core_data_processing import (
        load_health_records, load_zone_data
    )
    from utils.ai_analytics_engine import apply_ai_models
    from utils.ui_visualization_helpers import (
        plot_bar_chart_web,
        _create_empty_plot_figure
    )
except ImportError as e_import_pop_dash: # Unique exception variable name
    error_msg_pop_dash = (
        f"CRITICAL IMPORT ERROR in 4_population_dashboard.py: {e_import_pop_dash}. "
        f"Current Python Path: {sys.path}. "
        f"Attempted to add to path: {_project_test_root_directory_pop}. "
        "Ensure all modules are correctly placed and `__init__.py` files exist in 'pages' and component subdirectories. "
        "Most importantly, ensure all external libraries (like plotly, pandas, etc.) are installed in your Python environment (see requirements.txt)."
    )
    print(error_msg_pop_dash, file=sys.stderr)
    if 'st' in globals() and hasattr(st, 'error'): 
        st.error(error_msg_pop_dash)
        st.stop()
    else:
        raise ImportError(error_msg_pop_dash) from e_import_pop_dash

# --- Page Configuration ---
st.set_page_config(
    page_title=f"Population Analytics - {app_config.APP_NAME}", # app_config must be available
    layout="wide",
    initial_sidebar_state="expanded"
)
logger = logging.getLogger(__name__)

# ... (rest of the file 4_population_dashboard.py as refactored in File 61) ...
# The content from "Data Loading for Population Analytics Console" onwards remains the same
# as the previously provided refactored version (File 61). I will paste it
# for completeness of this specific file output.

# --- Data Loading for Population Analytics Console ---
@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS_WEB_REPORTS, show_spinner="Loading and preparing population analytics dataset...")
def get_population_analytics_dataset(source_context_log: str = "PopAnalyticsConsole/LoadData"):
    logger.info(f"({source_context_log}) Loading population health records and zone attribute data.")
    
    health_df_raw_pop = load_health_records(source_context=f"{source_context_log}/HealthRecs")
    
    df_enriched_health_pop = pd.DataFrame()
    ai_cols_expected_pop = ['ai_risk_score', 'ai_followup_priority_score'] 
    base_health_cols_pop_schema = health_df_raw_pop.columns.tolist() if isinstance(health_df_raw_pop, pd.DataFrame) and not health_df_raw_pop.empty else []

    if isinstance(health_df_raw_pop, pd.DataFrame) and not health_df_raw_pop.empty:
        enriched_ai_output_pop = apply_ai_models(health_df_raw_pop.copy(), source_context=f"{source_context_log}/AIEnrich")
        df_enriched_health_pop = enriched_ai_output_pop[0] 
    else:
        logger.warning(f"({source_context_log}) Raw health records for population analytics are empty or invalid. AI enrichment skipped.")
        df_enriched_health_pop = pd.DataFrame(columns=list(set(base_health_cols_pop_schema + ai_cols_expected_pop))) 

    gdf_zone_data_pop = load_zone_data(source_context=f"{source_context_log}/ZoneData") 
    df_zone_attributes_pop = pd.DataFrame()
    expected_sdoh_cols_list = ['zone_id', 'name', 'population', 'socio_economic_index', 
                               'avg_travel_time_clinic_min', 'predominant_hazard_type', 
                               'primary_livelihood', 'water_source_main', 'area_sqkm']

    if isinstance(gdf_zone_data_pop, gpd.GeoDataFrame) and not gdf_zone_data_pop.empty: # Check if it's a GeoDataFrame
        active_geom_col_name = gdf_zone_data_pop.geometry.name if hasattr(gdf_zone_data_pop, 'geometry') else 'geometry'
        df_zone_attributes_pop = pd.DataFrame(gdf_zone_data_pop.drop(columns=[active_geom_col_name], errors='ignore'))
        
        for sdoh_col_name in expected_sdoh_cols_list:
            if sdoh_col_name not in df_zone_attributes_pop.columns:
                df_zone_attributes_pop[sdoh_col_name] = np.nan
                logger.debug(f"({source_context_log}) Expected SDOH attribute '{sdoh_col_name}' was missing from loaded zone data, added as NaN column.")
        logger.info(f"({source_context_log}) Loaded {len(df_zone_attributes_pop)} zone attributes for SDOH context.")
    else:
        logger.warning(f"({source_context_log}) Zone attributes data (GDF) unavailable or empty. SDOH analytics will be limited.")
        df_zone_attributes_pop = pd.DataFrame(columns=expected_sdoh_cols_list) 

    if df_enriched_health_pop.empty: 
        logger.error(f"({source_context_log}) CRITICAL FAILURE: Health data remains empty after all processing steps for population analytics console.")
        
    return df_enriched_health_pop, df_zone_attributes_pop

# --- Load Datasets ---
df_health_main_population, df_zone_attributes_population = get_population_analytics_dataset()

if df_health_main_population.empty: 
    st.error("🚨 **Critical Data Failure:** Primary health dataset for population analytics could not be loaded or is empty. Most console features will be unavailable. Please check data sources and logs.")
    st.stop()

# --- Page Title and Introduction ---
st.title(f"📊 {app_config.APP_NAME} - Population Health Analytics & Research Console")
st.markdown("In-depth exploration of demographic distributions, epidemiological patterns, clinical trends, and health system factors using aggregated population-level data.")
st.divider()

# --- Sidebar Filters ---
if os.path.exists(app_config.APP_LOGO_SMALL): st.sidebar.image(app_config.APP_LOGO_SMALL, width=150)
st.sidebar.header("🔎 Analytics Filters")

min_data_date_pop = date.today() - timedelta(days=365*3) 
max_data_date_pop = date.today() 
if 'encounter_date' in df_health_main_population.columns and df_health_main_population['encounter_date'].notna().any():
    if not pd.api.types.is_datetime64_any_dtype(df_health_main_population['encounter_date']):
        df_health_main_population['encounter_date'] = pd.to_datetime(df_health_main_population['encounter_date'], errors='coerce')
    
    if df_health_main_population['encounter_date'].notna().any(): 
        min_data_date_pop = df_health_main_population['encounter_date'].min().date()
        max_data_date_pop = df_health_main_population['encounter_date'].max().date()

if min_data_date_pop > max_data_date_pop: 
    min_data_date_pop = max_data_date_pop 

selected_start_date_population, selected_end_date_population = st.sidebar.date_input(
    "Select Date Range for Analysis:", value=[min_data_date_pop, max_data_date_pop],
    min_value=min_data_date_pop, max_value=max_data_date_pop, key="population_console_date_range_picker"
)
if selected_start_date_population > selected_end_date_population:
    st.sidebar.error("Start date must be on or before end date.")
    selected_start_date_population = selected_end_date_population 

if 'encounter_date' not in df_health_main_population.columns:
    st.error("Critical error: 'encounter_date' column is missing from the health dataset. Cannot filter by date.")
    analytics_df_for_display = pd.DataFrame() 
    st.stop()
else:
    analytics_df_for_display = df_health_main_population[
        (df_health_main_population['encounter_date'].notna()) & 
        (df_health_main_population['encounter_date'].dt.date >= selected_start_date_population) &
        (df_health_main_population['encounter_date'].dt.date <= selected_end_date_population)
    ].copy()

selected_condition_filter_population = "All Conditions (Aggregated)" 
if 'condition' in analytics_df_for_display.columns:
    unique_conditions_list_pop = ["All Conditions (Aggregated)"] + sorted(analytics_df_for_display['condition'].dropna().unique().tolist())
    selected_condition_filter_population = st.sidebar.selectbox(
        "Filter by Condition Group (Optional):", options=unique_conditions_list_pop, index=0,
        key="population_console_condition_selector"
    )
    if selected_condition_filter_population != "All Conditions (Aggregated)":
        analytics_df_for_display = analytics_df_for_display[analytics_df_for_display['condition'] == selected_condition_filter_population]
else:
    st.sidebar.caption("Condition filter unavailable ('condition' column missing).")

selected_zone_display_population = "All Zones (Aggregated)" 
zone_display_to_id_map_pop = {}
zone_options_for_filter_pop = ["All Zones (Aggregated)"]
if isinstance(df_zone_attributes_population, pd.DataFrame) and not df_zone_attributes_population.empty and \
   'zone_id' in df_zone_attributes_population.columns:
    for _, zone_row_filter in df_zone_attributes_population.iterrows():
        zone_id_val = str(zone_row_filter['zone_id'])
        zone_name_val = str(zone_row_filter.get('name', zone_id_val)) 
        display_option = f"{zone_name_val} ({zone_id_val})" if zone_name_val != zone_id_val and zone_name_val != "Unknown" else zone_id_val
        if display_option not in zone_display_to_id_map_pop: 
            zone_options_for_filter_pop.append(display_option)
            zone_display_to_id_map_pop[display_option] = zone_id_val
    zone_options_for_filter_pop = sorted(list(set(zone_options_for_filter_pop))) 
else:
    st.sidebar.caption("Zone filter options limited (zone attributes data missing).")

if len(zone_options_for_filter_pop) > 1 : 
    selected_zone_display_population = st.sidebar.selectbox(
        "Filter by Zone (Optional):", options=zone_options_for_filter_pop, index=0, 
        key="population_console_zone_selector"
    )
    if selected_zone_display_population != "All Zones (Aggregated)":
        actual_zone_id_for_filtering_pop = zone_display_to_id_map_pop.get(selected_zone_display_population, selected_zone_display_population)
        if 'zone_id' in analytics_df_for_display.columns:
            analytics_df_for_display = analytics_df_for_display[analytics_df_for_display['zone_id'] == actual_zone_id_for_filtering_pop]
        else:
            st.sidebar.caption("Cannot filter by zone ('zone_id' missing in health data).")

if analytics_df_for_display.empty and (selected_start_date_population != min_data_date_pop or selected_end_date_population != max_data_date_pop or selected_condition_filter_population != "All Conditions (Aggregated)" or selected_zone_display_population != "All Zones (Aggregated)"):
    st.warning(f"No health data available for the selected filters. Please broaden your filter criteria.")

st.subheader(f"Population Health Snapshot ({selected_start_date_population.strftime('%d %b %Y')} - {selected_end_date_population.strftime('%d %b %Y')}, Cond: {selected_condition_filter_population}, Zone: {selected_zone_display_population})")
if analytics_df_for_display.empty:
    st.info("Insufficient data after filtering to display population summary KPIs.")
else:
    cols_kpi_population_summary = st.columns(4)
    num_total_unique_patients = analytics_df_for_display['patient_id'].nunique() if 'patient_id' in analytics_df_for_display.columns else 0
    mean_ai_risk_score_pop = np.nan
    if 'ai_risk_score' in analytics_df_for_display.columns and analytics_df_for_display['ai_risk_score'].notna().any():
        mean_ai_risk_score_pop = analytics_df_for_display['ai_risk_score'].mean()
    count_high_risk_patients_pop = 0; percent_high_risk_patients_pop = 0.0
    if 'ai_risk_score' in analytics_df_for_display.columns and num_total_unique_patients > 0:
        df_high_risk_pop = analytics_df_for_display[analytics_df_for_display['ai_risk_score'] >= app_config.RISK_SCORE_HIGH_THRESHOLD]
        count_high_risk_patients_pop = df_high_risk_pop['patient_id'].nunique() if 'patient_id' in df_high_risk_pop.columns else 0
        percent_high_risk_patients_pop = (count_high_risk_patients_pop / num_total_unique_patients) * 100 if num_total_unique_patients > 0 else 0.0
    str_top_condition_name_pop, num_top_condition_encounters_pop = "N/A", 0
    if 'condition' in analytics_df_for_display.columns and analytics_df_for_display['condition'].notna().any():
        series_condition_counts_pop = analytics_df_for_display['condition'].value_counts()
        if not series_condition_counts_pop.empty:
            str_top_condition_name_pop = series_condition_counts_pop.idxmax()
            num_top_condition_encounters_pop = series_condition_counts_pop.max()
    with cols_kpi_population_summary[0]: st.markdown(f"""<div class="custom-markdown-kpi-box"><div class="custom-kpi-label-top-condition">Total Unique Patients</div><div class="custom-kpi-value-large">{num_total_unique_patients:,}</div></div>""", unsafe_allow_html=True)
    with cols_kpi_population_summary[1]: st.markdown(f"""<div class="custom-markdown-kpi-box"><div class="custom-kpi-label-top-condition">Avg. AI Risk Score</div><div class="custom-kpi-value-large">{mean_ai_risk_score_pop:.1f if pd.notna(mean_ai_risk_score_pop) else 'N/A'}</div></div>""", unsafe_allow_html=True)
    with cols_kpi_population_summary[2]: st.markdown(f"""<div class="custom-markdown-kpi-box"><div class="custom-kpi-label-top-condition">% High AI Risk Patients</div><div class="custom-kpi-value-large">{percent_high_risk_patients_pop:.1f}%</div><div class="custom-kpi-subtext-small">({count_high_risk_patients_pop:,} patients)</div></div>""", unsafe_allow_html=True)
    with cols_kpi_population_summary[3]: st.markdown(f"""<div class="custom-markdown-kpi-box highlight-red-edge"><div class="custom-kpi-label-top-condition">Top Condition (Encounters)</div><div class="custom-kpi-value-large">{html.escape(str(str_top_condition_name_pop))}</div><div class="custom-kpi-subtext-small">{num_top_condition_encounters_pop:,} encounters</div></div>""", unsafe_allow_html=True)

pop_analytics_tab_titles = ["📈 Epi Overview", "🧑‍🤝‍🧑 Demographics & SDOH", "🔬 Clinical Insights", "⚙️ Systems & Equity"]
tab_pop_epi_view, tab_pop_demog_sdoh_view, tab_pop_clinical_insights_view, tab_pop_systems_equity_view = st.tabs(pop_analytics_tab_titles)

# ... (The rest of the tab implementations remain the same placeholder st.markdown as in File 61) ...
# For brevity, I'm not re-pasting that identical tab placeholder content here.
# It includes the st.header and the st.info/st.markdown placeholders for each tab.

with tab_pop_epi_view:
    st.header(f"Epidemiological Overview (Filters: {selected_condition_filter_population} | {selected_zone_display_population})")
    if analytics_df_for_display.empty:
        st.info("No data available for Epidemiological Overview with the current filter selections.")
    else:
        if 'condition' in analytics_df_for_display.columns and 'patient_id' in analytics_df_for_display.columns:
            df_condition_unique_patient_counts = analytics_df_for_display.groupby('condition')['patient_id'].nunique().nlargest(12).reset_index(name='unique_patients')
            if not df_condition_unique_patient_counts.empty:
                st.plotly_chart(plot_bar_chart_web(
                    df_condition_unique_patient_counts, x_col='condition', y_col='unique_patients', 
                    title="Top Conditions by Unique Patient Count (Filtered Set)", 
                    orientation='h', y_is_count=True, chart_height=450,
                    x_axis_label="Unique Patient Count", y_axis_label="Condition"
                ), use_container_width=True)
            else: st.caption("No aggregated condition counts found for unique patients with current filters.")
        
        if 'ai_risk_score' in analytics_df_for_display.columns and analytics_df_for_display['ai_risk_score'].notna().any():
            fig_ai_risk_distribution = px.histogram(
                analytics_df_for_display.dropna(subset=['ai_risk_score']), x="ai_risk_score", nbins=25, 
                title="Patient AI Risk Score Distribution (Filtered Set)",
                labels={'ai_risk_score': 'AI Risk Score', 'count': 'Number of Records'} 
            )
            fig_ai_risk_distribution.update_layout(bargap=0.1, height=app_config.WEB_PLOT_COMPACT_HEIGHT)
            st.plotly_chart(fig_ai_risk_distribution, use_container_width=True)
        
        st.caption("Note: True incidence/prevalence trends require careful definition of 'new case' vs 'active case' and appropriate denominators for this population-level view. This section provides overview counts and distributions based on available encounter data.")

with tab_pop_demog_sdoh_view:
    st.header("Demographics & Social Determinants of Health (SDOH) Context")
    if analytics_df_for_display.empty:
        st.info("No data available for Demographics & SDOH analysis with the current filter selections.")
    else:
        st.markdown("_(Placeholder: Detailed charts for Age Distribution, Gender Distribution. For SDOH, merge `analytics_df_for_display` with `df_zone_attributes_population` on `zone_id` to correlate health outcomes like AI Risk with Zone SES, Travel Time, etc.)_")
        if df_zone_attributes_population.empty:
            st.caption("Zone attribute data (required for SDOH analysis) is currently unavailable.")

with tab_pop_clinical_insights_view:
    st.header("Clinical Insights & Diagnostic Patterns")
    if analytics_df_for_display.empty:
        st.info("No data available for Clinical Insights with the current filter selections.")
    else:
        st.markdown("_(Placeholder: Analyses for Top Reported Symptoms (frequency, trends if data supports), Test Result Distributions (e.g., % Positive for key tests over time or by demographic strata), and deeper dives into Test Positivity Trends for specific conditions or risk groups.)_")

with tab_pop_systems_equity_view:
    st.header("Health Systems Performance & Equity Lens")
    if analytics_df_for_display.empty:
        st.info("No data available for Health Systems & Equity analysis with the current filter selections.")
    else:
        st.markdown("_(Placeholder: Analyses on Patient Encounters by Clinic/Zone (if applicable), Referral Pathway Completion Rates and Bottlenecks (if referral outcome data is comprehensive), and investigation of AI Risk Score Variations when stratified by SDOH factors like Zone SES or Primary Livelihood to explore potential health equity considerations.)_")


st.divider()
st.caption(app_config.APP_FOOTER_TEXT)
logger.info(f"Population Health Analytics Console page loaded/refreshed for Period: {selected_start_date_population.isoformat()} to {selected_end_date_population.isoformat()}, Cond: {selected_condition_filter_population}, Zone: {selected_zone_display_population}")
