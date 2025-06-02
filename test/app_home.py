# sentinel_project_root/test/app_home.py
# Main landing page for the "Sentinel Health Co-Pilot" System Overview.

import streamlit as st
import sys # For path manipulation
import os # For path manipulation
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
        "Ensure 'config.py'You are absolutely correct! My apologies. I inadvertently included the file numbering and "Summary is in 'test/config/' and Streamlit is run from 'sentinel_project_root'."
    )
    print(error_msg_home_main, file=sys.stderr)
    raise ImportError(error_msg_home_main) from e_import_home_main

# --- Page Configuration ---
page_icon_path_home = "🌍" 
if hasattr(app_config, 'APP_LOGO_SMALL') and app_config.APP_LOGO_SMALL and os.path.exists(app_config.APP_LOGO_SMALL):
    page_icon_path_home = app_config.APP_LOGO_SMALL
else:
    logging.warning(f"App logo small not found at: {app of Changes" comment block *inside* the Python code block for `app_home.py` in my previous response. That is a clear `SyntaxError`.

Let me provide the corrected `app_home.py` (File 71) without the extraneous comment block at the top. The actual Python code logic for the 3x2 capabilities layout and other content updates should be correct.

---

**File 71 of X (Corrected `app_home.py` - SyntaxError Fixed): `sentinel_project_root/test/app_home.py`**

```python
# sentinel_project_root/test/app_config.APP_LOGO_SMALL if hasattr(app_config, 'APP_LOGO_SMALL') else 'N/A'}. Using fallback icon.")

st.set_page_config(
    page_title=f"{app_config.APP_NAME} - System Overview",
    page_icon=page_icon_path_home,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': f"mailto:{app_config.SUPPORT_CONTACT_INFO}?subject=Help Request - {app_config.APP_NAME}",
        'Report a bug': f"mailto:{app_config.SUPPORT_CONTACT_INFO}?subject=Bug Report - {app_config.APP_NAME} v{app_config.APP_VERSION}",_home.py
# Main landing page for the "Sentinel Health Co-Pilot" System Overview.

import streamlit as st
import sys # For path manipulation
import os # For path manipulation
import logging # For setting up root logger

# --- Robust Path Setup for Imports ---
# This ensures that 'config' and 'utils' can be imported correctly
# when Streamlit runs this page script from the 'pages' subdirectory.
# app_home.py is in sentinel_project_root/test/, so test/ is the root for these imports.
_current_file_directory_home = os.path.dirname(os.path.abspath(__file__))
# _app_module_root_home is the 'test' directory itself if app_home.py is directly in 'test'
_app_module_root_home = _current_file_directory_home # If app_home.py IS in 'test'
# If app_home.py was in a sub
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
            logger.error(f"Error reading global CSS {css_file_path_global}:-directory of 'test', then you'd go os.pardir
# Based on the path /mount/src/sentinel_project_root/test/app_home.py, _current_file_directory_home is 'test'

# The goal is to add 'sentinel_project_root/test' to sys.path if it's not already there
# when running `streamlit run test/app_home.py` from `sentinel_project_root`
# However, if streamlit runs app_home.py, its own directory ('test') is usually on path.
# For robustness if scripts are invoked differently or for submodule imports from pages:
_project_root_candidate = os.path.abspath(os.path.join(_current_file_directory_home, os.pardir)) # sentinel_project_root
_app_path_to_add = os.path.join(_project_root_candidate, "test") # sentinel_project_root/test

if _app_path_to_add not in sys.path:
    sys.path.insert(0, _app_path_to_add)


# --- Sentinel System Imports (Now after path setup) ---
try:
    from config import app_config 
except ImportError as e_import_home_main:
    error_msg_home_main = (
        f"CRITICAL IMPORT ERROR in app_home.py: {e_import_home_main}. "
        f"Current Python Path: {sys.path {e_css_global}")
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
st.header("Simulated Role-Specific Views}. "
        f"Attempted to ensure '{_app_path_to_add}' is on path. " # Used the correct path var here
        "Ensure 'config.py' is in 'test/config/' directory and Streamlit is run correctly."
    )
    print(error_msg_home_main, file=sys.stderr)
    # Cannot use st.error here yet as Streamlit might not be fully loaded if config fails
    raise ImportError(error_msg_home_main) from e_import_home_main


# --- Page Configuration ---
page_icon_path_home = "🌍" 
if hasattr(app_config, 'APP_LOGO_SMALL') and app_config.APP_LOGO_SMALL and os.path.exists(app_config.APP_LOGO_SMALL):
    page_icon_path_home = app_config.APP_LOGO_SMALL
else:
    # (Facility/Regional Level)")

expander_details_home = [
    ("🧑‍⚕️ CHW Operations Summary & Field Support View (Supervisor/Hub Level)", 
     "Focus: CHW team performance, targeted support for CHWs, localized outbreak signal detection based on aggregated CHW reports.", 
     "pages/1_chw_dashboard.py", "nav_to_chw_dashboard_home_v3"),
    ("🏥 Clinic Operations & Environmental Safety View (Facility Node Level)", 
     "Focus: Optimizing clinic workflows, ensuring quality patient care, managing supplies and testing backlogs, monitoring clinic environment for safety and infection control.", 
     "pages/2_clinic_dashboard.py", "nav_to_clinic_dashboard_home_v3"),
    ("🗺️ District Health Strategic Overview (DHO at Facility/Regional Node Level)", 
     "Focus: Population health insights, resource allocation across zones, monitoring environmental well-being, and planning targeted interventions.", 
     "pages/3_district_dashboard.py", "nav_to_district_dashboard_home_v3"),
    ("📊 Population Health Analytics Deep Dive (Epidemiologist/Analyst View - Tier 3)", 
     "Focus: In-depth analysis of demographic patterns, SDOH impacts, clinical trends, health system performance, and equity across broader populations.", 
     "pages/4_population_dashboard.py", "nav_to_population_dashboard_home_v3"),
]
for title_exp, description_exp, page_path_exp, button_key_exp in expander_details_home:
    st.subheader(title_exp) 
    st.markdown(f"<small>{description_exp}</small>", unsafe_allow_html=True)
    button_label = f"Explore {title_exp.split('(')[0].strip().replace('**','')}"
    if st.button(button_label, key=button_key_exp, type="primary"):
        st.switch_page(page_path_exp)
    st.markdown("---") if "Population Health Analytics" not in title_exp else st.divider()

st.divider()


# --- UPDATED "Key Capabilities Reimagined" Section in 3x2 Layout ---
st.header(f"{app_config.APP_NAME} - Key Capabilities Reimagined")
capabilities_reimagined_list_updated = [
    ("🛡️ Frontline Worker Safety & Support", "Real-time vitals/environmental monitoring, fatigue detection, and safety nudges on Personal Edge Devices (PEDs)."),
    ("🌍 Offline-First Edge AI", "On-device intelligence for alerts, prioritization, and guidance with zero reliance on continuous connectivity."),
    ("⚡ Actionable, Contextual Insights", "From raw data to clear, role-specific recommendations that integrate into field workflows."),
    ("🤝 Human-Centered & Accessible UX", "Pictogram-based UIs, voice/tap commands, and local language support for low-literacy, high-stress users on PEDs."),
    ("📡 Resilient Data Synchronization", "Flexible data sharing (Bluetooth, QR, SD card, SMS, opportunistic IP) across PEDs, Hubs, and Nodes."),
    ("🌱 Scalable & Interoperable Architecture", "Modular design from personal to national levels, with FHIR/HL7 compliance for system integration.")
]

# Display in 2 rows of 3 columns
row1_cols_caps = st.columns(3)
row2_cols_caps = st Logging might not be configured yet if app_config failed, print for now.
    print(f"WARNING: App logo small not found at: {app_config.APP_LOGO_SMALL if hasattr(app_config, 'APP_LOGO_SMALL') else 'N/A'}. Using fallback icon.")

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
# It's safer to configure logging after st.set_page_config and successful app_config import
log_level_cfg_home = getattr(logging, str(app_config.LOG_LEVEL).upper(), logging.INFO)
logging.basicConfig(
    level=log_level_cfg_home, format=app_config.LOG_FORMAT, datefmt=app_config.LOG_DATE_FORMAT,
    handlers=[logging.StreamHandler()], force=True # force=True can help reconfigure if already set by Streamlit default
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
        # st.warning(f"Application styles not found. Display may be affected.") # Less alarming for user

if hasattr(app_config, 'STYLE_CSS_PATH_WEB'):
    load_global_styles_app_home(app_config.STYLE_CSS_PATH_WEB)

# --- App Header ---
header_cols_main_app = st.columns([0.15, 0.85]) 
with header_cols_main_app[0]:
    logo_path_main_header = app_config.APP_LOGO_LARGE
    if not os.path.exists(logo_path_main_header): logo_path_main_header = app_config.APP_LOGO_SMALL
    if os.path.exists(logo_path_main_header): st.image(logo_path_main_header, width=100)
    else: st.markdown("🌍", unsafe_allow_html=True) # Fallback if no logos
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

expander_details_home = [
    ("🧑‍⚕️ CHW Operations Summary & Field Support View (Supervisor/Hub Level)", 
     "This view simulates how a CHW Supervisor or a Hub coordinator might access summarized data from CHW Personal Edge Devices (PEDs).\n\n- **Focus (Tier 1-2):** Team.columns(3)

for i_cap, (cap_title_item, cap_desc_item) in enumerate(capabilities_reimagined_list_updated):
    target_col_caps = row1_cols_caps[i_cap] if i_cap < 3 else row2_cols_caps[i_cap-3]
    with target_col_caps:
        st.markdown(f"##### {cap_title_item}")
        st.markdown(f"<small>{cap_desc_item}</small>", unsafe_allow_html=True)
        # Adding a bit more vertical space within the column if needed, or rely on CSS
        if i_cap < 3 : st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True) # Add space for items in the first row

st.divider()
# --- End of UPDATED Key Capabilities Section ---


# --- Link to the Glossary page ---
with st.expander("📜 **Sentinel System Glossary** - Definitions for terms, metrics, and system components.", expanded=False):
    st.markdown("""
    - Understand terminology specific to the **Sentinel Health Co-Pilot** system.
    - Clarify technical definitions and operational terms used throughout the platform.
    """)
    if st.button("Go to System Glossary", key="nav_glossary_main_page_v3", type="secondary"): 
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

logger.info(f"{app_config.APP_NAME} (v{app_config.APP_VERSION}) - System Overview page loaded successfully with updated capabilities layout.")
