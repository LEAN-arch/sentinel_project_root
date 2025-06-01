# sentinel_project_root/test/app_home.py
# Main landing page for the "Sentinel Health Co-Pilot" System Overview.

import streamlit as st
import os
import logging # For setting up root logger
from config import app_config # Uses the fully redesigned app_config

# --- Page Configuration ---
# Determine page icon path carefully
page_icon_to_use = "🌍" # Default emoji
if hasattr(app_config, 'APP_LOGO_SMALL') and app_config.APP_LOGO_SMALL:
    if os.path.exists(app_config.APP_LOGO_SMALL):
        page_icon_to_use = app_config.APP_LOGO_SMALL
    else:
        logging.warning(f"App logo small configured but not found at: {app_config.APP_LOGO_SMALL}. Using fallback icon.")

st.set_page_config(
    page_title=f"{app_config.APP_NAME} - System Overview",
    page_icon=page_icon_to_use,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': f"mailto:{app_config.SUPPORT_CONTACT_INFO}?subject=Help Request - {app_config.APP_NAME}",
        'Report a bug': f"mailto:{app_config.SUPPORT_CONTACT_INFO}?subject=Bug Report - {app_config.APP_NAME} v{app_config.APP_VERSION}",
        'About': f"""
### {app_config.APP_NAME}
**Version:** {app_config.APP_VERSION}

{app_config.APP_FOOTER_TEXT}

This platform demonstrates an Edge-First Health Intelligence & Action Co-Pilot 
designed for resource-limited environments, prioritizing offline capabilities 
and actionable insights for frontline health workers (FHWs). 
Web views shown here are for supervisor/management tiers. 
FHWs use a dedicated native application on their Personal Edge Device (PED).
"""
    }
)

# --- Logging Setup (Initialize once at the app entry point) ---
# Configure root logger for the Streamlit application
log_level_from_config = getattr(logging, str(app_config.LOG_LEVEL).upper(), logging.INFO)
logging.basicConfig(
    level=log_level_from_config,
    format=app_config.LOG_FORMAT,
    datefmt=app_config.LOG_DATE_FORMAT,
    handlers=[logging.StreamHandler()] # Ensure logs go to console/Streamlit log viewer
)
logger = logging.getLogger(__name__) # Logger for this app_home.py

# --- CSS Loading ---
@st.cache_resource # Cache resource loading functions
def load_global_styles(css_path: str):
    if os.path.exists(css_path):
        try:
            with open(css_path, encoding="utf-8") as f_css:
                st.markdown(f'<style>{f_css.read()}</style>', unsafe_allow_html=True)
            logger.info(f"Global web CSS loaded successfully from: {css_path}")
        except Exception as e:
            logger.error(f"Error reading global web CSS file {css_path}: {e}")
            st.error(f"Critical error: Could not load application styles from {os.path.basename(css_path)}.")
    else:
        logger.warning(f"Global web CSS file not found: {css_path}. Default Streamlit styles will apply.")
        st.warning(f"Application styles not found at {os.path.basename(css_path)}. Display may be affected.")

if hasattr(app_config, 'STYLE_CSS_PATH_WEB'):
    load_global_styles(app_config.STYLE_CSS_PATH_WEB)

# --- App Header ---
header_cols = st.columns([0.15, 0.85]) # Adjust ratio for logo vs title
with header_cols[0]:
    logo_path_header = app_config.APP_LOGO_LARGE
    if not os.path.exists(logo_path_header): # Fallback to small logo
        logo_path_header = app_config.APP_LOGO_SMALL
    
    if os.path.exists(logo_path_header):
        st.image(logo_path_header, width=100) # Adjusted width
    else:
        st.markdown("🌍", unsafe_allow_html=True) # Fallback if no logos found

with header_cols[1]:
    st.title(app_config.APP_NAME)
    st.caption(f"Version {app_config.APP_VERSION}  |  Transforming Data into Lifesaving Action at the Edge")
st.divider()

# --- Introduction ---
st.markdown(f"""
    #### Welcome to the **{app_config.APP_NAME}** System Overview!
    
    This platform demonstrates key aspects of an **edge-first health intelligence system** 
    designed for maximum clinical and operational actionability in resource-limited, 
    high-risk LMIC environments. It aims to bridge advanced technology with real-world 
    field utility, emphasizing **offline-first capabilities** for frontline workers.

    **Core Principles:**
    - **Edge-First, Offline Capable:** Intelligence on Personal Edge Devices (PEDs).
    - **Action-Driven Insights:** Clear, targeted recommendations and alerts.
    - **Human-Centered UX for Frontline:** Pictogram-based, local language PED UIs.
    - **Resilient & Scalable Data Flow:** Modular architecture for flexible data sync.

    👈 **Navigate using the sidebar to explore simulated web-based views for different operational tiers.**
    These views primarily represent **Supervisor, Clinic Manager, or District Health Officer (DHO)** perspectives. 
    They are *not* the native mobile/wearable application used by frontline health workers on their PEDs.
""")
st.info(
    "💡 **Note:** This web application is a high-level demonstrator of the system's data processing "
    "and aggregated views for management and strategic personnel."
)

st.subheader("Explore Simulated Views:")

# --- Navigation Expanders ---
expander_details = [
    ("🧑‍⚕️ **CHW Supervisor View** (Tier 1 Hub / Tier 2 Node)", 
     "Focus: CHW team performance, escalated field alerts, CHW supply needs, local epi signals.", 
     "pages/1_chw_dashboard.py", "nav_to_chw_dashboard"),
    ("🏥 **Clinic Operations Console** (Clinic Manager @ Tier 2 Node)", 
     "Focus: Clinic service delivery, care quality, resource management, facility environment.", 
     "pages/2_clinic_dashboard.py", "nav_to_clinic_dashboard"),
    ("🗺️ **DHO Strategic Command Center** (DHO @ Tier 2/3 Node)", 
     "Focus: Population health, resource allocation, intervention planning, environmental risks.", 
     "pages/3_district_dashboard.py", "nav_to_district_dashboard"),
    ("📊 **Population Analytics Console** (Analyst @ Tier 3 / Adv. Tier 2)", 
     "Focus: In-depth epidemiological investigation, SDOH impacts, clinical patterns, health equity.", 
     "pages/4_population_dashboard.py", "nav_to_population_dashboard"),
    ("📜 **System Glossary**",
     "Focus: Definitions for terms, metrics, and system components used in Sentinel.",
     "pages/5_Glossary.py", "nav_to_glossary")
]

for title, description, page_path, button_key in expander_details:
    # Expand the first non-glossary item by default for a better landing experience
    is_expanded = "Population Analytics" in title if "Population Analytics" in title else False 
    with st.expander(title, expanded=is_expanded):
        st.markdown(f"<small>{description}</small>", unsafe_allow_html=True)
        if st.button(f"Go to {title.split('(')[0].strip()}", key=button_key, type="primary" if "Glossary" not in title else "secondary"):
            st.switch_page(page_path)

st.divider()
st.subheader(f"{app_config.APP_NAME} - Core Capabilities")
cols_caps = st.columns(3)
capabilities_list = [
    ("🛡️ **Frontline Safety & Action**", "PED-based real-time monitoring, personalized alerts, and JIT guidance."),
    ("🧠 **Edge-First AI & Logic**", "On-device intelligence for offline risk stratification, task prioritization, and alerts."),
    ("📡 **Resilient Data Sync**", "Opportunistic data transfer (Bluetooth, QR, SD, SMS, Wi-Fi) for constrained connectivity."),
    ("📈 **Multi-Tiered Analytics**", "From PED insights to supervisor summaries, clinic dashboards, and DHO strategic views."),
    ("🤝 **Human-Centered UX**", "Pictogram-driven native PED UIs, local languages, voice/tap inputs for LMIC contexts."),
    ("🌐 **Interoperable Design**", "Modular architecture supporting standards like FHIR for health system integration.")
]
for i, (cap_title, cap_desc) in enumerate(capabilities_list):
    with cols_caps[i % 3]:
        st.markdown(f"##### {cap_title}")
        st.markdown(f"<small>{cap_desc}</small>", unsafe_allow_html=True)
        if (i + 1) % 3 != 0 and i < len(capabilities_list) -1 : st.empty() # Visual spacer if not last in row

# --- Sidebar Content ---
st.sidebar.header(f"{app_config.APP_NAME}") # Simplified header
# Logo and version already handled by page_icon and menu_items or main header
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

logger.info(f"{app_config.APP_NAME} (v{app_config.APP_VERSION}) - System Overview page loaded.")
