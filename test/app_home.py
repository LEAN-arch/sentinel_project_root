# sentinel_project_root/test/app_home.py
# Main landing page for the "Sentinel Health Co-Pilot" System Overview.

import streamlit as st
import sys 
import os 
import logging 

# --- Path Setup (Simplified for app_home.py) ---
# When running `streamlit run test/app_home.py` from `sentinel_project_root`,
# the 'test' directory is typically added to sys.path by Streamlit,
# allowing direct imports of 'config' and 'utils' if they are in 'test/config' and 'test/utils'.
# Explicit path manipulation here is primarily for robustness or non-standard execution.
# For this iteration, we'll rely on Streamlit's default behavior for app_home.py,
# assuming 'config' and 'utils' are discoverable.
# The robust path setup IS crucial for files within the 'pages' subdirectory.

try:
    from config import app_config 
except ImportError as e_import_home:
    # This error suggests 'test' is not in sys.path when app_home.py is run,
    # or config.py is not in test/config/.
    # Attempting a more robust path addition if initial import fails.
    _current_file_dir_app_home = os.path.dirname(os.path.abspath(__file__)) # .../test
    if _current_file_dir_app_home not in sys.path:
        sys.path.insert(0, _current_file_dir_app_home)
    
    # Try importing again
    try:
        from config import app_config
    except ImportError as e_retry_import:
        error_msg_app_home = (
            f"CRITICAL IMPORT ERROR in app_home.py: {e_retry_import}. "
            f"Python Path: {sys.path}. Current file dir: {_current_file_dir_app_home}. "
            "Ensure 'config/app_config.py' exists within your app root (expected to be 'test/')."
        )
        print(error_msg_app_home, file=sys.stderr)
        # Cannot use st.error reliably here.
        raise ImportError(error_msg_app_home) from e_retry_import

# --- Page Configuration ---
page_icon_to_use_main = "🌍" 
if hasattr(app_config, 'APP_LOGO_SMALL') and app_config.APP_LOGO_SMALL and os.path.exists(app_config.APP_LOGO_SMALL):
    page_icon_to_use_main = app_config.APP_LOGO_SMALL
else:
    # Logging might not be fully set up if app_config had issues, so print.
    print(f"WARNING (app_home.py): App logo small not found at: {app_config.APP_LOGO_SMALL if hasattr(app_config, 'APP_LOGO_SMALL') else 'N/A'}. Using fallback icon.")

st.set_page_config(
    page_title=f"{app_config.APP_NAME} - System Overview",
    page_icon=page_icon_to_use_main,
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
log_level_main_app = getattr(logging, str(app_config.LOG_LEVEL).upper(), logging.INFO)
logging.basicConfig(
    level=log_level_main_app, format=app_config.LOG_FORMAT, datefmt=app_config.LOG_DATE_FORMAT,
    handlers=[logging.StreamHandler(sys.stdout)], force=True 
)
logger = logging.getLogger(__name__)

# --- CSS Loading ---
@st.cache_resource 
def load_app_home_styles(css_file_path: str):
    if os.path.exists(css_file_path):
        try:
            with open(css_file_path, encoding="utf-8") as f_css:
                st.markdown(f'<style>{f_css.read()}</style>', unsafe_allow_html=True)
            logger.info(f"Global web CSS loaded by app_home: {css_file_path}")
        except Exception as e:
            logger.error(f"Error reading global CSS {css_file_path} in app_home: {e}")
            st.error("Critical error loading application styles.")
    else:
        logger.warning(f"Global web CSS file not found by app_home: {css_file_path}. Default styles apply.")
        st.warning("Application styles file not found. Display may be affected.")

if hasattr(app_config, 'STYLE_CSS_PATH_WEB'):
    load_app_home_styles(app_config.STYLE_CSS_PATH_WEB)

# --- App Header ---
cols_header_app_home = st.columns([0.12, 0.88]) # Adjusted for slightly smaller logo if desired
with cols_header_app_home[0]:
    path_logo_header = app_config.APP_LOGO_LARGE
    if not os.path.exists(path_logo_header): path_logo_header = app_config.APP_LOGO_SMALL
    
    if os.path.exists(path_logo_header):
        st.image(path_logo_header, width=90) # Slightly smaller width
    else:
        st.markdown("<h3>🌍</h3>", unsafe_allow_html=True) # Larger emoji if no logo
with cols_header_app_home[1]:
    st.title(app_config.APP_NAME)
    st.subheader("Transforming Data into Lifesaving Action at the Edge") # More impactful sub-headline
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
# Using columns for a slightly more visual layout of principles
cols_principles = st.columns(2)
principles_list = [
    ("オフラインファーストの運用 (Offline-First Operations)", "On-device Edge AI on Personal Edge Devices (PEDs) ensures critical functionality without continuous connectivity."),
    ("行動志向のインテリジェンス (Action-Oriented Intelligence)", "Every insight aims to trigger a clear, targeted response relevant to frontline workflows."),
    ("人間中心のデザイン (Human-Centered Design)", "Interfaces are optimized for low-literacy, high-stress users, prioritizing immediate understanding and ease of use on PEDs."),
    ("回復力とスケーラビリティ (Resilience & Scalability)", "Modular design allows scaling from individual PEDs to facility and regional views, with robust, flexible data synchronization mechanisms.")
]
for i, (title, desc) in enumerate(principles_list):
    with cols_principles[i % 2]:
        st.markdown(f"##### {title}")
        st.markdown(f"<small>{desc}</small>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)


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

role_view_navigation_details = [
    ("🧑‍⚕️ CHW Supervisor Dashboard", 
     "Monitors CHW team performance, manages escalated field alerts, oversees CHW supply needs, and detects local epidemiological signals from aggregated CHW reports.", 
     "pages/1_chw_dashboard.py", "nav_chw_dashboard_main_page"),
    ("🏥 Clinic Operations Console", 
     "Provides Clinic Managers with insights into service efficiency, care quality, resource management (supplies, tests), and environmental safety within the facility.", 
     "pages/2_clinic_dashboard.py", "nav_clinic_dashboard_main_page"),
    ("🗺️ DHO Strategic Command Center", 
     "Offers District Health Officers a strategic overview for population health management, resource allocation, environmental risk monitoring, and intervention planning.", 
     "pages/3_district_dashboard.py", "nav_district_dashboard_main_page"),
    ("📊 Population Analytics Console", 
     "Enables Epidemiologists and Analysts to conduct in-depth investigations into health dynamics, SDOH impacts, clinical patterns, and health equity using broader datasets.", 
     "pages/4_population_dashboard.py", "nav_population_dashboard_main_page"),
]

# Display role views in two columns for a more balanced layout
cols_role_nav = st.columns(2)
for i, (role_title, role_desc, role_page_path, role_key) in enumerate(role_view_navigation_details):
    with cols_role_nav[i % 2]:
        with st.container(border=True): # Add a border to each role card
            st.subheader(role_title)
            st.markdown(f"<small>{role_desc}</small>", unsafe_allow_html=True)
            if st.button(f"Open {role_title.split('(')[0].strip().replace('**','')}", key=role_key, type="primary", use_container_width=True):
                st.switch_page(role_page_path)
            st.markdown("<br>", unsafe_allow_html=True) # Add some space at the bottom of the card
st.divider()


# --- Key Capabilities Reimagined Section (3x2 Layout) ---
st.header(f"{app_config.APP_NAME} - Key Capabilities Reimagined")
capabilities_reimagined_data = [
    ("🛡️ Frontline Worker Safety & Support", "Real-time vitals/environmental monitoring, fatigue detection, and safety nudges on Personal Edge Devices (PEDs)."),
    ("🌍 Offline-First Edge AI", "On-device intelligence for alerts, prioritization, and guidance with zero reliance on continuous connectivity."),
    ("⚡ Actionable, Contextual Insights", "From raw data to clear, role-specific recommendations that integrate into field workflows."),
    ("🤝 Human-Centered & Accessible UX", "Pictogram-based UIs, voice/tap commands, and local language support for low-literacy, high-stress users on PEDs."),
    ("📡 Resilient Data Synchronization", "Flexible data sharing (Bluetooth, QR, SD card, SMS, opportunistic IP) across PEDs, Hubs, and Nodes."),
    ("🌱 Scalable & Interoperable Architecture", "Modular design from personal to national levels, with FHIR/HL7 compliance for system integration.")
]

# Create two rows of three columns
cap_row1_cols = st.columns(3)
cap_row2_cols = st.columns(3)

for i, (cap_title_val, cap_desc_val) in enumerate(capabilities_reimagined_data):
    target_col_for_cap = cap_row1_cols[i] if i < 3 else cap_row2_cols[i-3]
    with target_col_for_cap:
        st.markdown(f"##### {cap_title_val}")
        st.markdown(f"<small>{cap_desc_val}</small>", unsafe_allow_html=True)
        # Add consistent bottom margin for items in columns if not last in a conceptual row of 3
        if (i % 3 != 2) or (i < 3 and len(capabilities_reimagined_data) > 3) : # Add if not last in its row OR if it's in first row and there's a second row
             st.markdown("<div style='margin-bottom: 1.2rem;'></div>", unsafe_allow_html=True) 
st.divider()

# --- Link to the Glossary page ---
with st.expander("📜 **System Glossary** - Understand Sentinel's terminology and metrics.", expanded=False):
    st.markdown("Explore definitions for terms, metrics, and system components specific to the Sentinel Health Co-Pilot.")
    if st.button("Go to Glossary", key="nav_glossary_from_home", type="secondary"): 
        st.switch_page("pages/5_Glossary.py")

# --- Sidebar Content ---
st.sidebar.header(f"{app_config.APP_NAME}") 
st.sidebar.divider()
st.sidebar.markdown("#### About This Demonstrator")
st.sidebar.info(
    "This web application simulates higher-level dashboards for Supervisors, Clinic Managers, DHOs, and Analysts. "
    "Frontline health worker interaction occurs on dedicated Personal Edge Devices (PEDs) with native applications."
)
st.sidebar.markdown("---")
st.sidebar.page_link("pages/5_Glossary.py", label="📜 System Glossary", icon="📚") # Direct link in sidebar
st.sidebar.divider()
st.sidebar.markdown(f"**{app_config.ORGANIZATION_NAME}**")
st.sidebar.markdown(f"Support: [{app_config.SUPPORT_CONTACT_INFO}](mailto:{app_config.SUPPORT_CONTACT_INFO})")
st.sidebar.divider()
st.sidebar.caption(app_config.APP_FOOTER_TEXT)

logger.info(f"{app_config.APP_NAME} (v{app_config.APP_VERSION}) - System Overview page (app_home.py) loaded successfully with UI/UX enhancements.")
