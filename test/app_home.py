# sentinel_project_root/test/app_home.py
# Main landing page for the "Sentinel Health Co-Pilot" System Overview / Demonstrator.
# This Streamlit application primarily showcases supervisor/manager/DHO views (Tiers 1-3).
# Frontline CHW interaction happens on native Personal Edge Devices (PEDs).

import streamlit as st
import os
# pandas is not directly used for display logic here, but app_config may use it.
from config import app_config # Uses the fully redesigned app_config
import logging

# --- Page Configuration (Reflects Sentinel system identity) ---
# Ensure APP_LOGO_SMALL exists, or provide a valid character/emoji.
page_icon_path = app_config.APP_LOGO_SMALL
if not os.path.exists(page_icon_path):
    page_icon_path = "🌍" # Fallback emoji if logo file not found

st.set_page_config(
    page_title=f"{app_config.APP_NAME} - System Overview",
    page_icon=page_icon_path,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': f"mailto:{app_config.SUPPORT_CONTACT_INFO}?subject=Help Request - {app_config.APP_NAME}",
        'Report a bug': f"mailto:{app_config.SUPPORT_CONTACT_INFO}?subject=Bug Report - {app_config.APP_NAME} v{app_config.APP_VERSION}",
        'About': f"""
        ### {app_config.APP_NAME}
        **Version:** {app_config.APP_VERSION}

        An Edge-First Health Intelligence & Action Co-Pilot for LMIC Environments.
        {app_config.APP_FOOTER_TEXT}

        This platform (Sentinel Health Co-Pilot) prioritizes offline-first operations,
        actionable insights for frontline workers, and resilient data systems.
        The web views demonstrated here primarily represent supervisor, clinic manager,
        or District Health Officer (DHO) perspectives (typically at Facility or
        Regional/Cloud Nodes). The primary interface for frontline health workers
        is a native mobile/wearable application on their Personal Edge Device (PED),
        which is designed for their specific high-stress, low-resource context.
        """
    }
)

# --- Logging Setup (Initialize once at the entry point of the app) ---
# Note: If running individual pages directly, they might need their own logger setup or inherit.
# Streamlit Cloud usually handles a basic logging config.
logging.basicConfig(
    level=getattr(logging, app_config.LOG_LEVEL.upper(), logging.INFO),
    format=app_config.LOG_FORMAT,
    datefmt=app_config.LOG_DATE_FORMAT,
    handlers=[logging.StreamHandler()] # Ensure logs go to console/Streamlit log viewer
)
logger = logging.getLogger(__name__) # Logger for this main app_home page

# --- CSS Loading (for Web Views - Tier 2/3) ---
# This CSS is for the overall Streamlit app shell and components.
@st.cache_resource # Use cache_resource for functions loading assets
def load_sentinel_web_css(css_file_path: str):
    if os.path.exists(css_file_path):
        try:
            with open(css_file_path, encoding="utf-8") as f:
                st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
            logger.info(f"Sentinel web CSS loaded successfully from {css_file_path}")
        except Exception as e_css_load:
            logger.error(f"Error reading Sentinel web CSS file {css_file_path}: {e_css_load}")
    else:
        logger.warning(f"Sentinel web CSS file not found: {css_file_path}. Default Streamlit styles will apply.")

load_sentinel_web_css(app_config.STYLE_CSS_PATH_WEB)


# --- App Header ---
header_cols_app_home = st.columns([0.12, 0.88]) # Adjust ratio for logo size
with header_cols_app_home[0]:
    logo_to_display_main = app_config.APP_LOGO_LARGE
    if not os.path.exists(logo_to_display_main): # Fallback to small if large isn't there
        logo_to_display_main = app_config.APP_LOGO_SMALL
    
    if os.path.exists(logo_to_display_main):
        st.image(logo_to_display_main, width=110) # Consistent but slightly prominent width for main page logo
    else:
        st.markdown("🌍", unsafe_allow_html=True) # Fallback icon if no logos

with header_cols_app_home[1]:
    st.title(app_config.APP_NAME)
    st.caption(f"Version {app_config.APP_VERSION}  |  Transforming Data into Lifesaving Action at the Edge.")
st.markdown("---")

# --- Introduction to Sentinel Health Co-Pilot System ---
st.markdown(f"""
    #### Welcome to the **{app_config.APP_NAME}** System Overview!
    
    This platform demonstrates key aspects of an **edge-first health intelligence system** designed for
    maximum clinical and operational actionability in resource-limited, high-risk LMIC environments.
    It aims to bridge advanced technology with real-world field utility, converting diverse data sources
    (wearables, IoT, contextual inputs) into life-saving, workflow-integrated decisions,
    with a strong emphasis on **offline-first capabilities** for frontline workers.

    **Core Principles Guiding the Sentinel System:**
    - **Edge-First, Offline Capable:** Intelligence and core functions reside on Personal Edge Devices (PEDs) for continuous operation without internet.
    - **Action-Driven Insights:** Focus on clear, targeted recommendations and automated alerts, not just data display.
    - **Human-Centered UX for Frontline:** PED interfaces are pictogram-based, use local languages, and support voice/tap interaction for high-stress, low-literacy environments.
    - **Resilient & Scalable Data Flow:** Modular architecture allows data synchronization from PEDs to Hubs, Facility Nodes, and potentially Regional/Cloud instances using appropriate technologies (Bluetooth, QR, SD card, SMS, opportunistic IP).

    👈 **The sidebar provides navigation to simulated web-based views for different operational tiers.**
    These views primarily represent what **Supervisors (Tier 1 Hubs), Clinic Managers (Tier 2 Facility Nodes),
    or District Health Officers (DHOs at Tier 2/3 Nodes)** would access. They are *not* representative of the
    native mobile/wearable application used by frontline health workers on their PEDs.
""")
st.info(
    "💡 **Note:** This web application serves as a high-level demonstrator for the system's data processing capabilities "
    "and the types of aggregated views available to management and strategic personnel."
)

st.subheader("Explore Simulated Views for Different System Tiers & Roles:")

# --- Navigation Expanders to Other "Pages" (Simulated Tier 1-3 Web Views) ---
# Descriptions refined to match new roles and system context.

with st.expander("🧑‍⚕️ **CHW Field Operations & Support View (Supervisor @ Tier 1 Hub / Tier 2 Node)**", expanded=False):
    st.markdown("""
    Simulates a web interface for **CHW Supervisors or Hub Coordinators**. This view focuses on:
    - **Focus:** Aggregated CHW team performance, escalated patient alerts from the field, critical supply needs for CHW kits, and early epidemiological signals derived from CHW data.
    - **Key Features:** Summaries of daily CHW activities (visits, tasks), lists of patients requiring urgent supervisor attention, CHW team well-being indicators (simulated), and localized risk patterns.
    - **Objective:** Enable effective team management, provide targeted support to CHWs, facilitate rapid response to critical field situations, and contribute to local health surveillance.
    *(Frontline CHWs use a dedicated, offline-first native app on their Personal Edge Device (PED) for real-time alerts & task management.)*
    """)
    if st.button("Go to CHW Supervisor View", key="nav_chw_supervisor_sentinel", type="primary"):
        st.switch_page("pages/1_chw_dashboard.py")

with st.expander("🏥 **Clinic Operations & Management Console (Clinic Manager @ Tier 2 Facility Node)**", expanded=False):
    st.markdown("""
    Simulates a web-based console for **Clinic Managers and Lead Clinicians** at a Facility Node. This view focuses on:
    - **Focus:** Optimizing clinic service delivery, ensuring quality of care, managing resources (staff, tests, supplies), monitoring the clinic's environment for safety, and responding to local epidemiological trends.
    - **Key Features:** Clinic performance KPIs (e.g., test TAT, patient flow), supply stock forecasts & alerts, IoT environmental sensor summaries, local disease patterns, and lists of flagged patient cases needing clinical review.
    - **Objective:** Support data-driven clinic management, enhance operational efficiency, ensure patient and staff safety, and maintain high standards of care.
    """)
    if st.button("Go to Clinic Management Console", key="nav_clinic_manager_sentinel", type="primary"):
        st.switch_page("pages/2_clinic_dashboard.py")

with st.expander("🗺️ **District Health Strategic Command Center (DHO @ Tier 2/3 Node)**", expanded=False):
    st.markdown("""
    Presents a strategic web dashboard for **District Health Officers (DHOs) and public health teams**, enabling comprehensive oversight.
    - **Focus:** Population health management across multiple zones, equitable resource allocation, strategic intervention planning, monitoring environmental health risks, and evaluating program impact.
    - **Key Features:** District-wide health & operational KPIs, interactive maps for visualizing zonal disparities (risk, burden, access), comparative zonal analytics, district-level trend analysis, and data-driven tools for identifying priority areas for intervention.
    - **Objective:** Empower DHOs with actionable intelligence for evidence-based strategic planning, effective resource deployment, public health emergency response, and continuous improvement of health outcomes at the district level.
    """)
    if st.button("Go to DHO Command Center", key="nav_dho_strategic_sentinel", type="primary"):
        st.switch_page("pages/3_district_dashboard.py")

with st.expander("📊 **Population Health Analytics & Research Console (Analyst @ Tier 3 / Adv. Tier 2)**", expanded=True): # Kept expanded as an example
    st.markdown("""
    A specialized web interface for **Epidemiologists, Health Data Analysts, and Researchers**, typically at a Regional/Cloud Node or an advanced Facility Node.
    - **Focus:** In-depth investigation of population health dynamics, demographic and SDOH impacts, clinical and diagnostic patterns, health systems performance, and health equity considerations using broader, aggregated datasets.
    - **Key Features:** Tools for stratified analysis of disease burden, exploration of AI risk score distributions, advanced test positivity and comorbidity trend analysis, evaluation of referral pathways, and investigation into health disparities.
    - **Objective:** Provide robust analytical capabilities to generate new insights, support health systems research, evaluate public health programs, and inform evidence-based policy formulation for long-term population health improvement.
    """)
    if st.button("Go to Population Analytics Console", key="nav_pop_analytics_sentinel", type="primary"):
        st.switch_page("pages/4_population_dashboard.py")

st.markdown("---")
st.subheader(f"{app_config.APP_NAME} - Core Capabilities") # Updated to match system name
col_cap1, col_cap2, col_cap3 = st.columns(3)
with col_cap1:
    st.markdown("##### 🛡️ **Frontline Safety & Action**")
    st.markdown("<small>PED-based real-time monitoring, personalized alerts (vitals, environment, fatigue), and JIT guidance for CHWs.</small>", unsafe_allow_html=True)
    st.markdown("##### 📈 **Multi-Tiered Analytics**")
    st.markdown("<small>From immediate PED insights to supervisor summaries, clinic operational dashboards, and DHO strategic views.</small>", unsafe_allow_html=True)
with col_cap2:
    st.markdown("##### 🧠 **Edge-First AI & Logic**")
    st.markdown("<small>On-device intelligence (TinyML) for risk stratification, task prioritization, and alert generation, fully offline capable.</small>", unsafe_allow_html=True)
    st.markdown("##### 🤝 **Human-Centered & Accessible UX**")
    st.markdown("<small>Pictogram-driven native PED UIs, local languages, voice/tap inputs, and haptic/audio cues for high-stress LMIC contexts.</small>", unsafe_allow_html=True)
with col_cap3:
    st.markdown("##### 📡 **Resilient & Flexible Data Sync**")
    st.markdown("<small>Opportunistic data transfer (Bluetooth, QR, SD, SMS, Wi-Fi) designed for constrained connectivity environments.</small>", unsafe_allow_html=True)
    st.markdown("##### 🌐 **Interoperable & Scalable Design**")
    st.markdown("<small>Modular architecture supporting FHIR/IHE standards for integration with national health systems and future expansion.</small>", unsafe_allow_html=True)

# Link to the Glossary page
with st.expander("📜 **Sentinel System Glossary** - Definitions for terms, metrics, and system components.", expanded=False):
    st.markdown("""
    - Understand terminology specific to the **Sentinel Health Co-Pilot** system, including concepts like Personal Edge Devices (PEDs), Facility Nodes, Edge AI, and LMIC-focused metrics.
    - Clarify technical definitions and operational terms used throughout the system's various views and documentation.
    """)
    if st.button("Go to System Glossary", key="nav_glossary_sentinel", type="secondary"):
        st.switch_page("pages/5_Glossary.py")


# --- Sidebar Content (Contextual for App Home) ---
st.sidebar.header(f"{app_config.APP_NAME} Navigation")
if os.path.exists(app_config.APP_LOGO_SMALL):
    st.sidebar.image(app_config.APP_LOGO_SMALL, width=180)
st.sidebar.caption(f"Version {app_config.APP_VERSION}")
st.sidebar.markdown("---")
st.sidebar.markdown("#### About This Demonstrator")
st.sidebar.info(
    "This web application simulates higher-level views (Supervisor, Clinic Manager, DHO, Analyst) "
    "of the Sentinel Health Co-Pilot system. Frontline worker interaction occurs on dedicated Personal Edge Devices (PEDs)."
)
st.sidebar.markdown("---")
st.sidebar.markdown(f"**{app_config.ORGANIZATION_NAME}**")
st.sidebar.markdown(f"Support: [{app_config.SUPPORT_CONTACT_INFO}](mailto:{app_config.SUPPORT_CONTACT_INFO})", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.caption(app_config.APP_FOOTER_TEXT)

logger.info(f"Sentinel Health Co-Pilot system overview page ({app_config.APP_NAME}) loaded successfully.")
