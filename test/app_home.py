# sentinel_project_root/test/app_home.py
# Main landing page for the "Sentinel Health Co-Pilot" System Overview / Demonstrator.
# This Streamlit application primarily showcases supervisor/manager/DHO views (Tiers 1-3).
# Frontline CHW interaction happens on native Personal Edge Devices (PEDs).

import streamlit as st
import os
import logging # IMPORT LOGGING EARLIER
import sys   # For path adjustments if needed

# --- Path Setup for local dev (Streamlit can be tricky with subdirs for pages) ---
# This ensures 'test/' (where config, utils, pages are) is on the Python path
# when Streamlit executes this script or pages within it.
current_file_dir_app_home = os.path.dirname(os.path.abspath(__file__)) # .../test/
if current_file_dir_app_home not in sys.path:
    sys.path.insert(0, current_file_dir_app_home)
# If config/utils were truly one level UP from 'test/', it would be:
# project_root_app_home = os.path.abspath(os.path.join(current_file_dir_app_home, os.pardir))
# if project_root_app_home not in sys.path:
#    sys.path.insert(0, project_root_app_home)
# For this structure, assuming 'config' and 'utils' are siblings to 'pages' under 'test'.

from config import app_config # Uses the fully redesigned and finalized app_config

# --- Logging Setup (DEFINED BEFORE FIRST USE) ---
# Configure logging once at the application's entry point.
# This basicConfig should ideally not be repeated in sub-pages if they inherit the logger.
if not logging.getLogger().hasHandlers(): # Avoid adding multiple handlers if app reloads
    logging.basicConfig(
        level=getattr(logging, str(app_config.LOG_LEVEL).upper(), logging.INFO), # Use app_config for level
        format=app_config.LOG_FORMAT,                                       # Use app_config for format
        datefmt=app_config.LOG_DATE_FORMAT,                                 # Use app_config for date format
        handlers=[logging.StreamHandler()] # Ensure logs go to console/Streamlit log viewer
    )
logger = logging.getLogger(__name__) # Define logger for this module (app_home.py)

# --- Page Configuration (Reflects Sentinel system identity) ---
page_icon_to_use_app_home = app_config.APP_LOGO_SMALL
if not os.path.exists(page_icon_to_use_app_home):
    logger.warning(f"Page icon logo not found at '{page_icon_to_use_app_home}', using fallback emoji '🌍'.") # Logger is now defined
    page_icon_to_use_app_home = "🌍"

st.set_page_config(
    page_title=f"{app_config.APP_NAME} - System Overview",
    page_icon=page_icon_to_use_app_home,
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

        The actual PED interface for frontline health workers is a specialized native application, not shown here.
        """
    }
)

# --- CSS Loading (for Web Views - Tier 2/3) ---
@st.cache_resource
def load_sentinel_web_styling_app_home(css_file_path_web: str): # Unique function name
    if os.path.exists(css_file_path_web):
        try:
            with open(css_file_path_web, encoding="utf-8") as f:
                st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
            logger.info(f"Sentinel web styling loaded for app_home from {css_file_path_web}")
        except Exception as e_css_load_ah: # Unique exception var name
            logger.error(f"Error reading Sentinel web CSS for app_home {css_file_path_web}: {e_css_load_ah}")
    else:
        logger.warning(f"Sentinel web CSS file not found for app_home: {css_file_path_web}. Default styles will be used.")

load_sentinel_web_styling_app_home(app_config.STYLE_CSS_PATH_WEB)


# --- App Header ---
header_cols_ah_main = st.columns([0.12, 0.88]) # Unique var name
with header_cols_ah_main[0]:
    main_logo_display_path = app_config.APP_LOGO_LARGE
    if not os.path.exists(main_logo_display_path): main_logo_display_path = app_config.APP_LOGO_SMALL
    
    if os.path.exists(main_logo_display_path):
        st.image(main_logo_display_path, width=110)
    else:
        logger.warning(f"Main page logo(s) not found. Using fallback icon.")
        st.markdown("🌍", unsafe_allow_html=True)
      
with header_cols_ah_main[1]:
    st.title(app_config.APP_NAME)
    st.caption(f"Version {app_config.APP_VERSION}  |  Empowering Frontline Health with Edge Intelligence")
st.markdown("---") 

# --- App Introduction & Navigation Expanders ---
st.markdown(f"""
    #### Welcome to the **{app_config.APP_NAME}** System Overview!
    
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

# Navigation Expanders (content previously refined, keys made unique)
with st.expander("🧑‍⚕️ **CHW Field Operations & Support View (Supervisor @ Tier 1 Hub / Tier 2 Node)**", expanded=False):
    st.markdown("""
    Simulates a web interface for **CHW Supervisors or Hub Coordinators**. Focuses on:
    - Team performance, escalated patient alerts, CHW kit supply needs, early epi signals.
    *(Frontline CHWs use a dedicated, offline-first native app on their Personal Edge Device (PED).)*
    """)
    if st.button("Go to CHW Supervisor View", key="nav_chw_supervisor_final_v3", type="primary"):
        st.switch_page("pages/1_chw_dashboard.py")

with st.expander("🏥 **Clinic Operations & Management Console (Clinic Manager @ Tier 2 Facility Node)**", expanded=False):
    st.markdown("""
    Simulates a web console for **Clinic Managers/Leads**. Focuses on:
    - Service delivery (patient flow, TAT), resource management (supplies, tests), facility environment safety, local epi trends.
    """)
    if st.button("Go to Clinic Management Console", key="nav_clinic_manager_final_v3", type="primary"):
        st.switch_page("pages/2_clinic_dashboard.py")

with st.expander("🗺️ **District Health Strategic Command Center (DHO @ Tier 2/3 Node)**", expanded=False):
    st.markdown("""
    A strategic web dashboard for **District Health Officers (DHOs)**. Features include:
    - Population health KPIs, geospatial zonal analysis, resource allocation tools, intervention planning aids.
    """)
    if st.button("Go to DHO Command Center", key="nav_dho_strategic_final_v3", type="primary"):
        st.switch_page("pages/3_district_dashboard.py")

with st.expander("📊 **Population Health Analytics & Research Console (Analyst @ Tier 3 / Adv. Tier 2)**", expanded=False): # Default to collapsed
    st.markdown("""
    Specialized web interface for **Epidemiologists, Data Analysts, and Researchers**. Provides tools for:
    - In-depth investigation of health dynamics, SDOH impacts, clinical patterns, health systems performance, and equity.
    """)
    if st.button("Go to Population Analytics Console", key="nav_pop_analytics_final_v3", type="primary"):
        st.switch_page("pages/4_population_dashboard.py")

st.markdown("---")
st.subheader(f"{app_config.APP_NAME} - Core System Capabilities")
key_capabilities_cols = st.columns(3) # Unique var name
with key_capabilities_cols[0]:
    st.markdown("##### 🛡️ **Edge Intelligence & Action**")
    st.markdown("<small>PED-based real-time monitoring, personalized alerts, and JIT guidance for frontline workers, fully offline.</small>", unsafe_allow_html=True)
with key_capabilities_cols[1]:
    st.markdown("##### 🧠 **Human-Centered UX (PED)**")
    st.markdown("<small>Pictogram-driven native PED UIs, local languages, voice/tap inputs, and haptic/audio cues for high-stress LMIC contexts.</small>", unsafe_allow_html=True)
with key_capabilities_cols[2]:
    st.markdown("##### 🔗 **Resilient & Scalable Data**")
    st.markdown("<small>Modular data flow with opportunistic sync (BT, QR, SD, SMS, IP) across tiers, supporting aggregation and interoperability (FHIR/IHE).</small>", unsafe_allow_html=True)


with st.expander("📜 **System Glossary & Terminology**", expanded=False):
    st.markdown("Understand key terms and concepts for the Sentinel Health Co-Pilot system.")
    if st.button("Go to System Glossary", key="nav_glossary_final_v3", type="secondary"):
        st.switch_page("pages/5_Glossary.py")

# --- Sidebar Content ---
st.sidebar.header(f"{app_config.APP_NAME}")
if os.path.exists(app_config.APP_LOGO_SMALL): st.sidebar.image(app_config.APP_LOGO_SMALL, width=150) # Consistent small logo in sidebar
st.sidebar.caption(f"v{app_config.APP_VERSION}")
st.sidebar.markdown("---")
st.sidebar.markdown("##### **Web Demonstrator Overview**") # Title clarified
st.sidebar.info(
    "This app simulates higher-tier views (Supervisor, Clinic, DHO, Analyst) of the Sentinel system. "
    "Frontline worker interaction occurs on dedicated Personal Edge Devices (PEDs)."
)
st.sidebar.markdown("---")
st.sidebar.markdown(f"**{app_config.ORGANIZATION_NAME}**")
st.sidebar.markdown(f"Support: <a href='mailto:{app_config.SUPPORT_CONTACT_INFO}'>{app_config.SUPPORT_CONTACT_INFO}</a>", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.caption(app_config.APP_FOOTER_TEXT)

logger.info(f"Sentinel Health Co-Pilot system overview page (app_home.py) loaded successfully (v{app_config.APP_VERSION}).")
