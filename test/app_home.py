# sentinel_project_root/test/app_home.py
# Main landing page for the "Sentinel Health Co-Pilot" System Overview / Demonstrator.
# This Streamlit application primarily showcases supervisor/manager/DHO views (Tiers 1-3).
# Frontline CHW interaction happens on native Personal Edge Devices (PEDs).

import streamlit as st
import os
from config import app_config # Uses the fully redesigned and finalized app_config
import logging

# --- Page Configuration (Reflects Sentinel system identity) ---
# Ensure APP_LOGO_SMALL exists, or provide a valid character/emoji.
# This should point to a file that will be present in your assets directory.
page_icon_to_use = app_config.APP_LOGO_SMALL
if not os.path.exists(page_icon_to_use): # Check if the configured path actually exists
    logger.warning(f"Page icon logo not found at {page_icon_to_use}, using fallback emoji.")
    page_icon_to_use = "🌍"

st.set_page_config(
    page_title=f"{app_config.APP_NAME} - System Overview",
    page_icon=page_icon_to_use,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': f"mailto:{app_config.SUPPORT_CONTACT_INFO}?subject=Help Request - {app_config.APP_NAME}",
        'Report a bug': f"mailto:{app_config.SUPPORT_CONTACT_INFO}?subject=Bug Report - {app_config.APP_NAME} v{app_config.APP_VERSION}",
        'About': f"""
        ### {app_config.APP_NAME} (v{app_config.APP_VERSION})
        An Edge-First Health Intelligence & Action Co-Pilot for LMIC Environments.
        {app_config.APP_FOOTER_TEXT}

        **System Overview:**
        The Sentinel Health Co-Pilot prioritizes offline-first operations via Personal Edge Devices (PEDs) for frontline workers, 
        translating real-time sensor and contextual data into actionable guidance. 
        Data is opportunistically synced to higher tiers for broader operational oversight and strategic planning.

        **This Web Demonstrator:**
        The views navigated from this page simulate web-based interfaces primarily intended for:
        - **Supervisors (Tier 1 Hubs):** Team performance, escalated alerts.
        - **Clinic Managers (Tier 2 Facility Nodes):** Operational efficiency, quality of care, resource management.
        - **District Health Officers (DHOs @ Tier 2/3 Nodes):** Strategic oversight, population health insights.
        - **Analysts (Tier 3 / Advanced Tier 2):** In-depth epidemiological research.

        The actual PED interface for frontline workers is a specialized native application, not shown here.
        """
    }
)

# --- Logging Setup ---
# Configure logging once at the application's entry point.
# This basicConfig should ideally not be repeated in sub-pages if they inherit the logger.
if not logging.getLogger().hasHandlers(): # Avoid adding multiple handlers if app reloads
    logging.basicConfig(
        level=getattr(logging, app_config.LOG_LEVEL.upper(), logging.INFO),
        format=app_config.LOG_FORMAT,
        datefmt=app_config.LOG_DATE_FORMAT,
        handlers=[logging.StreamHandler()] # Ensure output to Streamlit's log console
    )
logger = logging.getLogger(__name__)

# --- CSS Loading (for Web Views - Tier 2/3) ---
@st.cache_resource # Cache the loading of CSS
def load_sentinel_web_styling(css_file_path_web: str):
    if os.path.exists(css_file_path_web):
        try:
            with open(css_file_path_web, encoding="utf-8") as f:
                st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
            logger.info(f"Sentinel web styling loaded from {css_file_path_web}")
        except Exception as e_css:
            logger.error(f"Error reading Sentinel web CSS {css_file_path_web}: {e_css}")
    else:
        logger.warning(f"Sentinel web CSS file not found: {css_file_path_web}. Default styles will be used.")

load_sentinel_web_styling(app_config.STYLE_CSS_PATH_WEB)


# --- App Header ---
header_cols = st.columns([0.10, 0.90]) # Adjusted for potentially different logo proportions
with header_cols[0]:
    # Prefer APP_LOGO_LARGE for the main landing page for more impact.
    main_logo_path = app_config.APP_LOGO_LARGE
    if not os.path.exists(main_logo_path): # Fallback to small logo if large one is missing
        main_logo_path = app_config.APP_LOGO_SMALL
    
    if os.path.exists(main_logo_path):
        st.image(main_logo_path, width=90) # Adjusted width for a prominent but not overwhelming logo
    else: # Ultimate fallback if no image files found
        logger.warning(f"Main page logo(s) not found at {app_config.APP_LOGO_LARGE} or {app_config.APP_LOGO_SMALL}. Using icon.")
        st.markdown("🌍", unsafe_allow_html=True)
      
with header_cols[1]:
    st.title(app_config.APP_NAME)
    st.caption(f"Version {app_config.APP_VERSION}  |  Empowering Frontline Health with Edge Intelligence")
st.markdown("---") 

# --- App Introduction & Navigation Expanders ---
st.markdown(f"""
    #### Welcome to the **{app_config.APP_NAME}** System Overview
    
    This platform demonstrates key functionalities and data views of the Sentinel Health Co-Pilot, 
    an integrated system designed for robust health monitoring and response in challenging LMIC settings. 
    It emphasizes:
    - **Real-time, On-Device Intelligence** for frontline health workers via Personal Edge Devices (PEDs).
    - **Actionable, Context-Aware Recommendations** fitting diverse operational workflows.
    - **Resilient Data Management** with offline-first capabilities and flexible synchronization.

    The following sections provide access to simulated web-based dashboards representing views for
    supervisory, clinical management, district health oversight, and population analytics roles
    (typically Tiers 1 Hub, Tier 2 Facility Node, or Tier 3 Regional/Cloud Node).
""")
st.success("👈 **Use the sidebar to navigate to the specific role-based views.**")

st.subheader("Simulated Web Views for System Tiers & Roles:")

with st.expander("🧑‍⚕️ **CHW Field Operations & Support View (Supervisor @ Tier 1 Hub / Tier 2 Node)**", expanded=False):
    st.markdown("""
    This view simulates a web interface for **CHW Supervisors or Hub Coordinators**. It focuses on:
    - **Operational Oversight:** Aggregated CHW team performance, task completion rates.
    - **Alert Management:** Review of critical patient alerts escalated from the field by CHWs' PEDs.
    - **Resource & Support Needs:** Identifying CHW kit resupply needs, potential CHW fatigue/stress indicators (based on synced data).
    - **Local Epi Signals:** Early detection of unusual symptom clusters or disease patterns from CHW reports.
    *Note: Frontline CHWs use a dedicated, offline-first native app on their Personal Edge Device (PED).*
    """)
    if st.button("Go to CHW Supervisor View", key="nav_chw_supervisor_sentinel_v2", type="primary"): # Ensure unique key
        st.switch_page("pages/1_chw_dashboard.py")

with st.expander("🏥 **Clinic Operations & Management Console (Clinic Manager @ Tier 2 Facility Node)**", expanded=False):
    st.markdown("""
    Simulates a web-based console for **Clinic Managers and Lead Clinicians** at a Facility Node. Key areas include:
    - **Service Delivery:** Monitoring patient flow, testing turnaround times (TAT), and quality of care indicators.
    - **Resource Management:** Tracking critical medical supply levels (drugs, consumables), forecasting needs, and overseeing diagnostic capacity.
    - **Environmental Safety:** Reviewing IoT sensor data (CO2, PM2.5, occupancy) to ensure a safe clinic environment.
    - **Local Epidemiology:** Analyzing clinic-level disease patterns, symptom trends, and test positivity rates.
    """)
    if st.button("Go to Clinic Management Console", key="nav_clinic_manager_sentinel_v2", type="primary"):
        st.switch_page("pages/2_clinic_dashboard.py")

with st.expander("🗺️ **District Health Strategic Command Center (DHO @ Tier 2/3 Node)**", expanded=False):
    st.markdown("""
    Presents a strategic web dashboard for **District Health Officers (DHOs) and public health teams**. Features include:
    - **Population Health Insights:** District-wide health KPIs, geospatial visualization of zonal health disparities (risk, burden, access).
    - **Resource Allocation:** Tools for comparative zonal analysis to inform equitable resource distribution.
    - **Intervention Planning:** Data-driven identification of priority zones for targeted public health interventions.
    - **Program Monitoring:** Tracking key health trends and the impact of ongoing programs.
    """)
    if st.button("Go to DHO Command Center", key="nav_dho_strategic_sentinel_v2", type="primary"):
        st.switch_page("pages/3_district_dashboard.py")

with st.expander("📊 **Population Health Analytics & Research Console (Analyst @ Tier 3 / Adv. Tier 2)**", expanded=False): # Default to collapsed for home page
    st.markdown("""
    A specialized web interface for **Epidemiologists, Health Data Analysts, and Researchers**. Provides tools for:
    - **In-depth Investigations:** Detailed analysis of demographic impacts, SDOH correlations, clinical pathways, and diagnostic trends using comprehensive datasets.
    - **Health Systems Research:** Evaluating system performance, access to care, and health equity across defined populations.
    - **Program Evaluation & Modeling:** Assessing the impact of public health interventions and modeling future health scenarios.
    """)
    if st.button("Go to Population Analytics Console", key="nav_pop_analytics_sentinel_v2", type="primary"):
        st.switch_page("pages/4_population_dashboard.py")

st.markdown("---")
st.subheader(f"{app_config.APP_NAME} - Core System Capabilities")
# Updated capabilities section using columns for better layout
key_caps_cols = st.columns(3)
with key_caps_cols[0]:
    st.markdown("##### 🛡️ **Edge Intelligence & Action**")
    st.markdown("<small>PED-based real-time vital/environmental analysis, fatigue/risk scoring, context-aware alerts, and JIT protocol guidance for frontline workers, fully offline.</small>", unsafe_allow_html=True)
with key_caps_cols[1]:
    st.markdown("##### 🩺 **Comprehensive Health Monitoring**")
    st.markdown("<small>Integrated monitoring from individual (wearable/PED) to community (IoT/CHW reports) and facility levels (operational/clinical data).</small>", unsafe_allow_html=True)
with key_caps_cols[2]:
    st.markdown("##### 🔗 **Resilient Data Ecosystem**")
    st.markdown("<small>Modular data flow with opportunistic sync (BT, QR, SD, SMS, IP) across tiers, supporting data aggregation for strategic insights and interoperability (FHIR/IHE).</small>", unsafe_allow_html=True)


with st.expander("📜 **System Glossary & Terminology** - Definitions for Sentinel Co-Pilot context", expanded=False):
    st.markdown("""
    - Understand key terminology for the **Sentinel Health Co-Pilot** system, including concepts like Personal Edge Devices (PEDs), Facility/Hub Nodes, Edge AI, lean data inputs, and LMIC-specific health metrics.
    - Useful for all users to clarify technical definitions and operational terms.
    """)
    if st.button("Go to System Glossary", key="nav_glossary_sentinel_v2", type="secondary"):
        st.switch_page("pages/5_Glossary.py")

# --- Sidebar Content ---
st.sidebar.header(f"{app_config.APP_NAME}") # Simpler sidebar header
if os.path.exists(app_config.APP_LOGO_SMALL):
    st.sidebar.image(app_config.APP_LOGO_SMALL, width=160) # Adjusted width
st.sidebar.caption(f"Version: {app_config.APP_VERSION}")
st.sidebar.markdown("---")
st.sidebar.markdown("##### **System Overview**") # More direct title
st.sidebar.info(
    "This web application demonstrates simulated higher-tier views (Supervisor, Clinic Manager, DHO, Analyst) "
    "of the Sentinel system. Frontline interaction occurs on dedicated Personal Edge Devices (PEDs)."
)
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Support & Information:**<br/>{app_config.ORGANIZATION_NAME}<br/>"
                    f"Contact: <a href='mailto:{app_config.SUPPORT_CONTACT_INFO}'>{app_config.SUPPORT_CONTACT_INFO}</a>", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.caption(app_config.APP_FOOTER_TEXT)

logger.info(f"Sentinel Health Co-Pilot system overview page ({app_config.APP_NAME} v{app_config.APP_VERSION}) loaded.")
