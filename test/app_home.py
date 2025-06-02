# sentinel_project_root/test/app_home.py
# Main landing page for the "Sentinel Health Co-Pilot" System Overview.

import streamlit as st
import sys 
import os 
import logging 

# --- Robust Path Setup for Imports ---
_current_file_dir_home = os.path.dirname(os.path.abspath(__file__))
_app_module_root_for_home = _current_file_dir_home 
if _app_module_root_for_home not in sys.path:
    sys.path.insert(0, _app_module_root_for_home)

try:
    from config import app_config 
except ImportError as e_import_home:
    _project_root_candidate_home = os.path.abspath(os.path.join(_current_file_dir_home, os.pardir, os.pardir)) # sentinel_project_root
    _app_path_to_add_home = os.path.join(_project_root_candidate_home, "test") # sentinel_project_root/test
    if _app_path_to_add_home not in sys.path: # Try adding project_root/test if direct fails
        sys.path.insert(0, _app_path_to_add_home)
    try: # Retry import
        from config import app_config
    except ImportError as e_retry_home:
        error_msg_app_home_final = (
            f"CRITICAL IMPORT ERROR in app_home.py: {e_retry_home}. Python Path: {sys.path}. "
            f"Tried adding: '{_app_module_root_for_home}' and '{_app_path_to_add_home}'. "
            "Ensure 'config/app_config.py' is correctly placed relative to your app root "
            "(expected 'sentinel_project_root/test/config/app_config.py') and Streamlit is run from 'sentinel_project_root'."
        )
        print(error_msg_app_home_final, file=sys.stderr)
        raise ImportError(error_msg_app_home_final) from e_retry_home

# --- Page Configuration ---
page_icon_to_use_main_app = "🌍" 
if hasattr(app_config, 'APP_LOGO_SMALL') and app_config.APP_LOGO_SMALL and os.path.exists(app_config.APP_LOGO_SMALL):
    page_icon_to_use_main_app = app_config.APP_LOGO_SMALL
else:
    print(f"WARNING (app_home.py): App logo small not found. Using fallback icon.")

st.set_page_config(
    page_title=f"{app_config.APP_NAME} - System Overview",
    page_icon=page_icon_to_use_main_app,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': f"mailto:{app_config.SUPPORT_CONTACT_INFO}?subject=Help Request - {app_config.APP_NAME}",
        'Report a bug': f"mailto:{app_config.SUPPORT_CONTACT_INFO}?subject=Bug Report - {app_config.APP_NAME} v{app_config.APP_VERSION}",
        'About': f"""
### {app_config.APP_NAME} (v{app_config.APP_VERSION})
{app_config.APP_FOOTER_TEXT}

An Edge-First Health Intelligence Co-Pilot designed for resource-limited environments.
This demonstrator showcases higher-level views. Frontline workers use dedicated native PED apps.
"""
    }
)

# --- Logging Setup ---
log_level_main_app_config = getattr(logging, str(app_config.LOG_LEVEL).upper(), logging.INFO)
logging.basicConfig(
    level=log_level_main_app_config, format=app_config.LOG_FORMAT, datefmt=app_config.LOG_DATE_FORMAT,
    handlers=[logging.StreamHandler(sys.stdout)], force=True 
)
logger = logging.getLogger(__name__)

# --- CSS Loading ---
@st.cache_resource 
def load_app_home_styles_final(css_file_path: str): # Renamed to avoid potential streamlit cache key collision
    if os.path.exists(css_file_path):
        try:
            with open(css_file_path, encoding="utf-8") as f_css_final:
                st.markdown(f'<style>{f_css_final.read()}</style>', unsafe_allow_html=True)
            logger.info(f"Global web CSS loaded by app_home: {css_file_path}")
        except Exception as e_css_final:
            logger.error(f"Error reading global CSS {css_file_path} in app_home: {e_css_final}")
            st.error("Critical error loading application styles.")
    else:
        logger.warning(f"Global web CSS file not found by app_home: {css_file_path}. Default styles apply.")
        st.warning("Application styles file not found. Display may be affected.")

if hasattr(app_config, 'STYLE_CSS_PATH_WEB'):
    load_app_home_styles_final(app_config.STYLE_CSS_PATH_WEB)

# --- App Header ---
cols_header_main_app = st.columns([0.12, 0.88]) 
with cols_header_main_app[0]:
    path_logo_header_main = app_config.APP_LOGO_LARGE
    if not os.path.exists(path_logo_header_main): path_logo_header_main = app_config.APP_LOGO_SMALL
    
    if os.path.exists(path_logo_header_main):
        st.image(path_logo_header_main, width=90)
    else:
        st.markdown("<h3>🌍</h3>", unsafe_allow_html=True) 
with cols_header_main_app[1]:
    st.title(app_config.APP_NAME)
    st.subheader("Transforming Data into Lifesaving Action at the Edge")
st.divider()

# --- Enhanced Welcome & System Description ---
st.markdown(f"""
    ## Welcome to the Sentinel Health Co-Pilot
    
    Sentinel is an **edge-first health intelligence system** designed for **maximum clinical and 
    operational actionability** in resource-limited, high-risk LMIC environments. It converts 
    diverse data sources—wearables, IoT, contextual inputs—into life-saving, workflow-integrated 
    decisions, even with **minimal or no internet connectivity.**
""")

st.markdown("#### Core Principles Guiding Sentinel:")
# Using columns for a slightly more visual layout of principles with Emojis
cols_core_principles = st.columns(2)
core_principles_list = [
    ("📶 **Offline-First Operations**", "On-device Edge AI on Personal Edge Devices (PEDs) ensures critical functionality without continuous connectivity."),
    ("🎯 **Action-Oriented Intelligence**", "Every insight aims to trigger a clear, targeted response relevant to frontline workflows."),
    ("🧑‍🤝‍🧑 **Human-Centered Design**", "Interfaces are optimized for low-literacy, high-stress users, prioritizing immediate understanding and ease of use on PEDs."),
    ("🔗 **Resilience & Scalability**", "Modular design allows scaling from individual PEDs to facility and regional views, with robust, flexible data synchronization mechanisms.")
]
for i_principle, (title_principle, desc_principle) in enumerate(core_principles_list):
    with cols_core_principles[i_principle % 2]:
        st.markdown(f"##### {title_principle}")
        st.markdown(f"<small>{desc_principle}</small>", unsafe_allow_html=True)
        st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True) # Add some space


st.markdown("""
    ---
    👈 **Navigate via the sidebar** to explore simulated web dashboards for various operational tiers. 
    These views represent perspectives of **Supervisors, Clinic Managers, or District Health Officers (DHOs)**. 
    The primary interface for frontline workers (e.g., CHWs) is a dedicated native application on their 
    Personal Edge Device (PED), tailored for their specific operational context.
""")
st.info(
    "💡 **Note:** This web application serves as a high-level demonstrator for the Sentinel system's "
    "data processing capabilities and the types of aggregated views available to management and strategic personnel."
)
st.divider()

# --- Simulated Role-Specific Views Section ---
st.header("Explore Simulated Role-Specific Dashboards")
st.caption("These views demonstrate the information available at higher tiers (Facility/Regional Nodes).")

role_view_nav_details_list = [
    ("🧑‍⚕️ CHW Operations Summary & Field Support View (Supervisor/Hub Level)", 
     "This view simulates how a CHW Supervisor or a Hub coordinator might access summarized data from CHW Personal Edge Devices (PEDs).\n\n- **Focus (Tier 1-2):** Team performance monitoring, targeted support for CHWs, localized outbreak signal detection based on aggregated CHW reports.\n- **Key Data Points:** CHW activity summaries (visits, tasks completed), patient alert escalations, critical supply needs for CHW kits, early epidemiological signals from specific zones.\n- **Objective:** Enable supervisors to manage CHW teams effectively, provide timely support, identify emerging health issues quickly, and coordinate local responses. The CHW's primary tool is their offline-first native app on their PED, providing real-time alerts & task management.", 
     "pages/1_chw_dashboard.py", "nav_chw_ops_main_final"),
    ("🏥 Clinic Operations & Environmental Safety View (Facility Node Level)", 
     "Simulates a dashboard for Clinic Managers at a Facility Node (Tier 2), providing insights into service efficiency, care quality, resource management, and environmental conditions.\n\n- **Focus (Tier 2):** Optimizing clinic workflows, ensuring quality patient care, managing supplies and testing backlogs, monitoring clinic environment for safety and infection control.\n- **Key Data Points:** Clinic performance KPIs (e.g., test TAT, patient throughput), supply stock forecasts, IoT sensor data summaries (CO2, PM2.5, occupancy), clinic-level epidemiological trends, flagged patient cases for review.\n- **Objective:** Enhance operational efficiency, support clinical decision-making, maintain resource availability, and ensure a safe clinic environment.", 
     "pages/2_clinic_dashboard.py", "nav_clinic_ops_main_final"),
    ("🗺️ District Health Strategic Overview (DHO at Facility/Regional Node Level)", 
     "Presents a strategic dashboard for District Health Officers (DHOs), typically accessed at a Facility Node (Tier 2) or a Regional/Cloud Node (Tier 3).\n\n- **Focus (Tier 2-3):** Population health insights, resource allocation across zones, monitoring environmental well-being, and planning targeted interventions.\n- **Key Data Points:** District-wide health KPIs, interactive maps for zonal comparisons (risk, disease burden, resources), trend analyses, intervention planning tools based on aggregated data.\n- **Objective:** Support evidence-based strategic planning, public health interventions, program monitoring, and policy development for the district.", 
     "pages/3_district_dashboard.py", "nav_dho_main_final"),
    ("📊 Population Health Analytics Deep Dive (Epidemiologist/Analyst View - Tier 3)", 
     "A view designed for detailed epidemiological and health systems analysis, typically used by analysts or program managers at a Regional/Cloud Node (Tier 3) with access to more comprehensive, aggregated datasets.\n\n- **Focus (Tier 3):** In-depth analysis of demographic patterns, SDOH impacts, clinical trends, health system performance, and equity across broader populations.\n- **Key Data Points:** Stratified disease burden, AI risk distributions by various factors, aggregated test positivity trends, comorbidity analysis, referral pathway performance, health equity metrics.\n- **Objective:** Provide robust analytical capabilities to understand population health dynamics, evaluate interventions, identify areas for research, and inform large-scale public health strategy.", 
     "pages/4_population_dashboard.py", "nav_pop_analytics_main_final"),
]

cols_role_nav_display = st.columns(2)
for i_role, (role_nav_title, role_nav_desc, role_nav_path, role_nav_key) in enumerate(role_view_nav_details_list):
    with cols_role_nav_display[i_role % 2]:
        with st.container(border=True):
            st.subheader(role_nav_title)
            st.markdown(f"<small>{role_nav_desc}</small>", unsafe_allow_html=True)
            # Cleaner button label by removing parenthetical role info
            button_label_role = f"Explore {role_nav_title.split('(')[0].strip().replace('**','')}"
            if st.button(button_label_role, key=role_nav_key, type="primary", use_container_width=True):
                st.switch_page(role_nav_path)
            st.markdown("<div style='margin-bottom: 0.5rem;'></div>", unsafe_allow_html=True) # Small bottom margin inside card
st.divider()


# --- Key Capabilities Reimagined Section (3x2 Layout) ---
st.header(f"{app_config.APP_NAME} - Key Capabilities Reimagined")
capabilities_reimagined_data_list = [
    ("🛡️ Frontline Worker Safety & Support", "Real-time vitals/environmental monitoring, fatigue detection, and safety nudges on Personal Edge Devices (PEDs)."),
    ("🌍 Offline-First Edge AI", "On-device intelligence for alerts, prioritization, and guidance with zero reliance on continuous connectivity."),
    ("⚡ Actionable, Contextual Insights", "From raw data to clear, role-specific recommendations that integrate into field workflows."),
    ("🤝 Human-Centered & Accessible UX", "Pictogram-based UIs, voice/tap commands, and local language support for low-literacy, high-stress users on PEDs."),
    ("📡 Resilient Data Synchronization", "Flexible data sharing (Bluetooth, QR, SD card, SMS, opportunistic IP) across PEDs, Hubs, and Nodes."),
    ("🌱 Scalable & Interoperable Architecture", "Modular design from personal to national levels, with FHIR/HL7 compliance for system integration.")
]

cap_row1_cols_display = st.columns(3)
cap_row2_cols_display = st.columns(3)

for i_cap_item, (cap_item_title, cap_item_desc) in enumerate(capabilities_reimagined_data_list):
    target_col_for_capability = cap_row1_cols_display[i_cap_item] if i_cap_item < 3 else cap_row2_cols_display[i_cap_item-3]
    with target_col_for_capability:
        st.markdown(f"##### {cap_item_title}")
        st.markdown(f"<small>{cap_item_desc}</small>", unsafe_allow_html=True)
        # Add consistent bottom margin for items, especially if text length varies greatly
        st.markdown("<div style='margin-bottom: 1.2rem;'></div>", unsafe_allow_html=True) 
st.divider()

# --- Link to the Glossary page ---
with st.expander("📜 **System Glossary** - Understand Sentinel's terminology and metrics.", expanded=False):
    st.markdown("Explore definitions for terms, metrics, and system components specific to the Sentinel Health Co-Pilot.")
    if st.button("Go to Glossary", key="nav_glossary_from_app_home", type="secondary"): # Unique key
        st.switch_page("pages/5_Glossary.py")

# --- Sidebar Content ---
st.sidebar.header(f"{app_config.APP_NAME}") 
st.sidebar.divider()
st.sidebar.markdown("#### About This Demonstrator:")
st.sidebar.info(
    "This web app simulates higher-level dashboards for Supervisors, Clinic Managers, DHOs, and Analysts. "
    "Frontline health worker interaction occurs on dedicated Personal Edge Devices (PEDs) with native applications."
)
st.sidebar.markdown("---")
st.sidebar.page_link("pages/5_Glossary.py", label="📜 System Glossary", icon="📚") 
st.sidebar.divider()
st.sidebar.markdown(f"**{app_config.ORGANIZATION_NAME}**")
st.sidebar.markdown(f"Support: [{app_config.SUPPORT_CONTACT_INFO}](mailto:{app_config.SUPPORT_CONTACT_INFO})")
st.sidebar.divider()
st.sidebar.caption(app_config.APP_FOOTER_TEXT)

logger.info(f"{app_config.APP_NAME} (v{app_config.APP_VERSION}) - System Overview page (app_home.py) loaded successfully with UI/UX enhancements including 3x2 capabilities layout.")
