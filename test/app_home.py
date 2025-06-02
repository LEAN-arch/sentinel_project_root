# sentinel_project_root/test/app_home.py
# Main landing page for the "Sentinel Health Co-Pilot" System Overview.

import streamlit as st
import sys
import os
import logging

# --- Robust Path Setup for Imports ---
_current_file_directory_home = os.path.dirname(os.path.abspath(__file__))
_app_module_root_home = os.path.abspath(os.path.join(_current_file_directory_home, os.pardir)) 
if _app_module_root_home not in sys.path:
    sys.path.insert(0, _app_module_root_home)

try:
    from config import app_config 
except ImportError as e_import_home_main:
    error_msg_home_main = (
        f"CRITICAL IMPORT ERROR in app_home.py: {e_import_home_main}. "
        f"Current Python Path: {sys.path}. "
        f"Attempted to ensure '{_app_module_root_home}' is on path. "
        "Ensure 'config.py' is in 'test/config/' and Streamlit is run from 'sentinel_project_root'."
    )
    print(error_msg_home_main, file=sys.stderr)
    raise ImportError(error_msg_home_main) from e_import_home_main

# --- Page Configuration ---
page_icon_path_home = "🌍" 
if hasattr(app_config, 'APP_LOGO_SMALL') and app_config.APP_LOGO_SMALL and os.path.exists(app_config.APP_LOGO_SMALL):
    page_icon_path_home = app_config.APP_LOGO_SMALL
else:
    logging.warning(f"App logo small not found at: {app_config.APP_LOGO_SMALL if hasattr(app_config, 'APP_LOGO_SMALL') else 'N/A'}. Using fallback icon.")

st.set_page_config(
    page_title=f"{app_config.APP_NAME} - System Overview",
    page_icon=page_icon_path_home,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': f"mailto:{app_config.SUPPORT_CONTACT_INFO}?subject=Help Request - {app_config.APP_NAME}",
        'Report a bug': f"mailto:{app_config.SUPPORT_CONTACT_INFO}?subject=Bug Report - {app_config.APP_NAME} v{app_config.APP_VERSION}",
        'About': f"""
### {app_config.APP_NAME} (v{app_config.APP_VERSION})
{app_config.APP_FOOTER_TEXT}
An Edge-First Health Intelligence Co-Pilot for LMIC Environments. 
Web views are for supervisor/management tiers. Frontline workers use dedicated PED native apps.
"""
    }
)

# --- Logging Setup ---
log_level_cfg_home = getattr(logging, str(app_config.LOG_LEVEL).upper(), logging.INFO)
logging.basicConfig(
    level=log_level_cfg_home, format=app_config.LOG_FORMAT, datefmt=app_config.LOG_DATE_FORMAT,
    handlers=[logging.StreamHandler()] 
)
logger = logging.getLogger(__name__)

# --- CSS Loading ---
@st.cache_resource 
def load_global_styles_app_home(css_file_path_global: str):
    if os.path.exists(css_file_path_global):
        try:
            with open(css_file_path_global, encoding="utf-8") as f_css_global:
                st.markdown(f'<style>{f_css_global.read()}</style>', unsafe_allow_html=True)
            logger.info(f"Global web CSS loaded: {css_file_path_global}")
        except Exception as e_css_global:
            logger.error(f"Error reading global CSS {css_file_path_global}: {e_css_global}")
            st.error(f"Critical error: Could not load application styles.")
    else:
        logger.warning(f"Global web CSS not found: {css_file_path_global}. Default styles will apply.")
        st.warning(f"Application styles not found. Display may be affected.")

if hasattr(app_config, 'STYLE_CSS_PATH_WEB'):
    load_global_styles_app_home(app_config.STYLE_CSS_PATH_WEB)

# --- App Header ---
header_cols_main_app = st.columns([0.15, 0.85]) 
with header_cols_main_app[0]:
    logo_path_main_header = app_config.APP_LOGO_LARGE
    if not os.path.exists(logo_path_main_header): logo_path_main_header = app_config.APP_LOGO_SMALL
    if os.path.exists(logo_path_main_header): st.image(logo_path_main_header, width=100)
    else: st.markdown("🌍", unsafe_allow_html=True)
with header_cols_main_app[1]:
    st.title(app_config.APP_NAME)
    st.caption(f"Version {app_config.APP_VERSION}  |  Edge-First Health Intelligence for LMICs")
st.divider()

# --- Welcome Message & System Description (As provided by user) ---
st.markdown(f"""
    ### Welcome to the Sentinel Health Co-Pilot System Overview
    The Sentinel Health Co-Pilot is an **edge-first health intelligence system** redesigned for 
    maximum clinical and operational actionability in resource-limited, high-risk environments. 
    It bridges advanced technology with real-world field utility by converting wearable, IoT, 
    and contextual data into life-saving, workflow-integrated decisions, **even with minimal 
    or no internet connectivity.**
""")

st.markdown("#### Core Principles:")
st.markdown("""
    - **Offline-First Operations:** On-device Edge AI on Personal Edge Devices (PEDs) ensures functionality without continuous connectivity.
    - **Action-Oriented Intelligence:** Every insight aims to trigger a clear, targeted response relevant to frontline workflows.
    - **Human-Centered Design:** Interfaces are optimized for low-literacy, high-stress users, prioritizing immediate understanding.
    - **Resilience & Scalability:** Modular design allows scaling from individual PEDs to facility and regional views, with robust, flexible data synchronization.
""")

st.markdown("""
    👈 The sidebar provides navigation to simulated views for different operational tiers. 
    These views primarily represent what supervisors, clinic managers, or District Health 
    Officers (DHOs) might see at a Facility Node (Tier 2) or Regional/Cloud Node (Tier 3). 
    The primary interface for frontline workers (e.g., CHWs) is a native mobile/wearable 
    application on their Personal Edge Device (PED), which is designed for their specific 
    high-stress, low-resource context and is not fully replicated here.
""")
st.info(
    "💡 **Note:** This web application serves as a demonstrator for the system's data processing "
    "capabilities and higher-level reporting views."
)
st.divider()

# --- Simulated Role-Specific Views Section (As provided by user) ---
st.header("Simulated Role-Specific Views (Facility/Regional Level)")

st.subheader("🧑‍⚕️ CHW Operations Summary & Field Support View (Supervisor/Hub Level)")
st.markdown("""
This view simulates how a CHW Supervisor or a Hub coordinator might access summarized data from CHW Personal Edge Devices (PEDs).

- **Focus (Tier 1-2):** Team performance monitoring, targeted support for CHWs, localized outbreak signal detection based on aggregated CHW reports.
- **Key Data Points:** CHW activity summaries (visits, tasks completed), patient alert escalations, critical supply needs for CHW kits, early epidemiological signals from specific zones.
- **Objective:** Enable supervisors to manage CHW teams effectively, provide timely support, identify emerging health issues quickly, and coordinate local responses. The CHW's primary tool is their offline-first native app on their PED, providing real-time alerts & task management.
""")
if st.button("Go to CHW Supervisor View", key="nav_chw_supervisor_main_v2", type="primary"): # Updated key for uniqueness
    st.switch_page("pages/1_chw_dashboard.py")
st.markdown("---") 

st.subheader("🏥 Clinic Operations & Environmental Safety View (Facility Node Level)")
st.markdown("""
Simulates a dashboard for Clinic Managers at a Facility Node (Tier 2), providing insights into service efficiency, care quality, resource management, and environmental conditions.

- **Focus (Tier 2):** Optimizing clinic workflows, ensuring quality patient care, managing supplies and testing backlogs, monitoring clinic environment for safety and infection control.
- **Key Data Points:** Clinic performance KPIs (e.g., test TAT, patient throughput), supply stock forecasts, IoT sensor data summaries (CO2, PM2.5, occupancy), clinic-level epidemiological trends, flagged patient cases for review.
- **Objective:** Enhance operational efficiency, support clinical decision-making, maintain resource availability, and ensure a safe clinic environment.
""")
if st.button("Go to Clinic Operations Console", key="nav_clinic_ops_main_v2", type="primary"):
    st.switch_page("pages/2_clinic_dashboard.py")
st.markdown("---")

st.subheader("🗺️ District Health Strategic Overview (DHO at Facility/Regional Node Level)")
st.markdown("""
Presents a strategic dashboard for District Health Officers (DHOs), typically accessed at a Facility Node (Tier 2) or a Regional/Cloud Node (Tier 3).

- **Focus (Tier 2-3):** Population health insights, resource allocation across zones, monitoring environmental well-being, and planning targeted interventions.
- **Key Data Points:** District-wide health KPIs, interactive maps for zonal comparisons (risk, disease burden, resources), trend analyses, intervention planning tools based on aggregated data.
- **Objective:** Support evidence-based strategic planning, public health interventions, program monitoring, and policy development for the district.
""")
if st.button("Go to DHO Strategic Overview", key="nav_dho_overview_main_v2", type="primary"):
    st.switch_page("pages/3_district_dashboard.py")
st.markdown("---")

st.subheader("📊 Population Health Analytics Deep Dive (Epidemiologist/Analyst View - Tier 3)")
st.markdown("""
A view designed for detailed epidemiological and health systems analysis, typically used by analysts or program managers at a Regional/Cloud Node (Tier 3) with access to more comprehensive, aggregated datasets.

- **Focus (Tier 3):** In-depth analysis of demographic patterns, SDOH impacts, clinical trends, health system performance, and equity across broader populations.
- **Key Data Points:** Stratified disease burden, AI risk distributions by various factors, aggregated test positivity trends, comorbidity analysis, referral pathway performance, health equity metrics.
- **Objective:** Provide robust analytical capabilities to understand population health dynamics, evaluate interventions, identify areas for research, and inform large-scale public health strategy.
""")
if st.button("Go to Population Analytics Console", key="nav_pop_analytics_main_v2", type="primary"):
    st.switch_page("pages/4_population_dashboard.py")
st.divider()

# --- UPDATED "Key Capabilities Reimagined" Section ---
st.header(f"{app_config.APP_NAME} - Key Capabilities Reimagined")
capabilities_reimagined_updated = [
    ("🛡️ Frontline Worker Safety & Support", "Real-time vitals/environmental monitoring, fatigue detection, and safety nudges on Personal Edge Devices (PEDs)."),
    ("🌍 Offline-First Edge AI", "On-device intelligence for alerts, prioritization, and guidance with zero reliance on continuous connectivity."),
    ("⚡ Actionable, Contextual Insights", "From raw data to clear, role-specific recommendations that integrate into field workflows."),
    ("🤝 Human-Centered & Accessible UX", "Pictogram-based UIs, voice/tap commands, and local language support for low-literacy, high-stress users on PEDs."),
    ("📡 Resilient Data Synchronization", "Flexible data sharing (Bluetooth, QR, SD card, SMS, opportunistic IP) across PEDs, Hubs, and Nodes."),
    ("🌱 Scalable & Interoperable Architecture", "Modular design from personal to national levels, with FHIR/HL7 compliance for system integration.")
]
for icon_title_cap, desc_cap in capabilities_reimagined_updated:
    st.markdown(f"##### {icon_title_cap}")
    st.markdown(f"<small>{desc_cap}</small>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True) # Visual spacer

st.divider()

# --- Link to the Glossary page ---
with st.expander("📜 **Sentinel System Glossary** - Definitions for terms, metrics, and system components.", expanded=False):
    st.markdown("""
    - Understand terminology specific to the **Sentinel Health Co-Pilot** system.
    - Clarify technical definitions and operational terms used throughout the platform.
    """)
    if st.button("Go to System Glossary", key="nav_glossary_main_page_v2", type="secondary"): # Unique key
        st.switch_page("pages/5_Glossary.py")

# --- Sidebar Content ---
st.sidebar.header(f"{app_config.APP_NAME}") 
st.sidebar.divider()
st.sidebar.markdown("#### About This Demonstrator:")
st.sidebar.info(
    "This web app simulates higher-level views (Supervisor, Clinic Manager, DHO, Analyst) "
    "of the Sentinel System. Frontline interaction is via dedicated Personal Edge Devices (PEDs)."
)
st.sidebar.divider()
st.sidebar.markdown(f"**{app_config.ORGANIZATION_NAME}**")
st.sidebar.markdown(f"Support: [{app_config.SUPPORT_CONTACT_INFO}](mailto:{app_config.SUPPORT_CONTACT_INFO})")
st.sidebar.divider()
st.sidebar.caption(app_config.APP_FOOTER_TEXT)

logger.info(f"{app_config.APP_NAME} (v{app_config.APP_VERSION}) - System Overview page loaded successfully with updated capabilities section.")
