# sentinel_project_root/test/pages/5_Glossary.py
# Glossary of Terms for the "Sentinel Health Co-Pilot" System

import streamlit as st
import os
import sys
import logging
from typing import Optional, List, Dict, Any # Ensure all used types are imported

# --- Sentinel System Imports & Path Setup ---
# This block attempts to make 'config' importable if the script is run in various contexts.
# For Streamlit Cloud, if 'test/app_home.py' is the main script, 'test/' is often added to sys.path.
try:
    from config import app_config
except ImportError:
    current_dir_glossary = os.path.dirname(os.path.abspath(__file__)) # .../pages
    project_app_root_glossary = os.path.abspath(os.path.join(current_dir_glossary, os.pardir)) # .../test
    if project_app_root_glossary not in sys.path:
        sys.path.insert(0, project_app_root_glossary)
    
    try: # Retry import after path adjustment
        from config import app_config
        logging.warning(f"Glossary Page: Used fallback sys.path adjustment for 'config' import.")
    except ImportError as e_cfg_glossary: # Still fails, app_config is critical.
        logging.critical(f"CRITICAL IMPORT ERROR in 5_Glossary.py: Cannot find 'config.app_config'. Error: {e_cfg_glossary}")
        # Define a very basic fallback for app_config to allow the page to at least try to load with errors.
        class AppConfigFallbackGlossary:
            APP_NAME="Sentinel App (Config Error)"
            APP_FOOTER_TEXT="System Glossary - Configuration Error"
            LOG_LEVEL="DEBUG" # So warnings are seen
            ALERT_SPO2_CRITICAL_LOW_PCT=90 # Example, so related_config doesn't fail if used
            FATIGUE_INDEX_HIGH_THRESHOLD=80
            RISK_SCORE_HIGH_THRESHOLD=75
            KEY_CONDITIONS_FOR_ACTION=["TB","Malaria"] # Minimal example for display_term_sentinel related_config
        app_config = AppConfigFallbackGlossary()
        st.error("Critical Error: Application configuration could not be loaded. Glossary content may be incomplete or incorrect.")

# --- Page Configuration ---
st.set_page_config(
    page_title=f"Glossary - {app_config.APP_NAME}",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Logger setup specific to this page, ensures it works even if app_home didn't run first
# or if logging basicConfig from app_home didn't take for page modules.
logger = logging.getLogger(__name__)
if not logger.hasHandlers(): # Check if handlers are already set for this logger instance
    # This ensures that if app_home set up global logging, we don't add duplicate handlers here.
    # But if this page is run standalone, it gets some basic logging.
    log_level_from_config = getattr(logging, str(app_config.LOG_LEVEL).upper(), logging.INFO)
    logging.basicConfig(level=log_level_from_config, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# CSS is assumed to be loaded globally by app_home.py. If this page
# needs to run truly standalone with its own styles, add CSS loading here:
# from utils.ui_visualization_helpers import load_sentinel_web_styling # if function name is that
# load_sentinel_web_styling(app_config.STYLE_CSS_PATH_WEB)


# --- Glossary Content ---
st.title(f"📜 {app_config.APP_NAME} - System Glossary")
st.markdown(
    "This page provides definitions for common terms, abbreviations, metrics, and system-specific concepts "
    "used throughout the Sentinel Health Co-Pilot platform and its documentation."
)
st.markdown("---")

# Helper function for consistent term formatting
def display_term_sentinel_glossary(term: str, definition: str, related_config_key: Optional[str] = None):
    st.markdown(f"#### {term}") # Using H4 for better visual hierarchy for terms
    st.markdown(f": _{definition}_") # Italicize definition
    if related_config_key:
        # Attempt to get the value from app_config if the key exists
        config_value = getattr(app_config, related_config_key, None)
        if config_value is not None:
            st.caption(f"*(Related configuration value: `{related_config_key}` = **`{config_value}`**)*")
        else:
            st.caption(f"*(Related configuration: `app_config.{related_config_key}`)*")
    st.markdown("---") # Separator


# --- I. Sentinel Health Co-Pilot System Concepts ---
st.header("🌐 System Architecture & Core Concepts")
display_term_sentinel_glossary("Sentinel Health Co-Pilot", "An edge-first health intelligence and action support system designed for resource-limited, high-risk LMIC environments. It prioritizes offline functionality, actionable insights for frontline workers, and resilient data flow.", "APP_NAME")
display_term_sentinel_glossary("Personal Edge Device (PED)", "A ruggedized smartphone, wearable sensor, or low-power System-on-Chip (SoC) device used by frontline health workers. It runs native applications with on-device Edge AI for real-time monitoring, alerts, task management, and Just-In-Time (JIT) guidance, primarily offline.")
display_term_sentinel_glossary("Edge AI / TinyML", "Artificial Intelligence models (e.g., using TensorFlow Lite, MicroPython for microcontrollers, or optimized libraries for mobile) designed to run directly on PEDs or local hubs with minimal computational resources, enabling offline decision support.")
display_term_sentinel_glossary("Supervisor Hub (Tier 1)", "An optional intermediary device (e.g., a more capable smartphone or tablet) used by a team leader. It locally aggregates data from team PEDs via short-range communication (Bluetooth, Wi-Fi Direct) for localized oversight, basic team analytics, and can facilitate batched data transfer to a Facility Node.")
display_term_sentinel_glossary("Facility Node (Tier 2)", "A local server, PC, or robust device (e.g., Raspberry Pi, Intel NUC) typically situated at a clinic or community health center. It aggregates data from multiple Supervisor Hubs or directly from PEDs (via opportunistic sync). It performs more complex local analytics, can interface with local EMRs/HIS (if they exist), generates reports for clinic management and DHOs, and acts as a staging point for wider data synchronization to regional or cloud systems.")
display_term_sentinel_glossary("Regional/Cloud Node (Tier 3)", "Optional centralized infrastructure (e.g., cloud servers or national data centers) for population-level analytics, epidemiological surveillance, advanced AI model refinement, and national health reporting. Receives batched, processed data from Facility Nodes.")
display_term_sentinel_glossary("Lean Data Inputs", f"A core design principle focusing on collecting only the minimum viable data elements that have maximum predictive power for Edge AI and actionable decision-making. This is crucial for environments with constrained input capabilities and bandwidth. Examples include critical vital sign thresholds (e.g., SpO2 < {getattr(app_config, 'ALERT_SPO2_CRITICAL_LOW_PCT', 'N/A')}%), key symptom flags, basic demographics like age group, and high-impact contextual factors.", "Section IV of app_config.py (conceptual list)")
display_term_sentinel_glossary("Action Code / Suggested Action Code", "A system-internal alphanumeric code (e.g., 'ACT_SPO2_URGENT_OXYGEN') generated by an alert or assigned to a task. On a PED, this code is mapped by the native application to a specific pictogram, a Just-In-Time (JIT) guidance sequence (e.g., simplified SOP steps), an automated communication trigger (like an SMS template), or a pre-defined checklist.")
display_term_sentinel_glossary("Opportunistic Sync", "A data synchronization strategy where PEDs, Hubs, and Nodes attempt to transfer data only when a viable, low-cost, or available communication channel is detected. This can include Bluetooth, local Wi-Fi Direct, physical SD card transfer ('sneaker-net'), batched SMS for critical summaries, or brief periods of cellular/internet connectivity.")
display_term_sentinel_glossary("Pictogram-based UI", "User Interface on the PED that heavily relies on simple, universally understandable icons and symbols rather than extensive text, catering to users with varying literacy levels operating in high-stress situations.")


# --- II. Clinical, Epidemiological & Operational Terms ---
st.header("🩺 Clinical, Epidemiological & Operational Terms")
display_term_sentinel_glossary("AI Risk Score", f"A simulated algorithmic score (typically 0-100) that estimates a patient's or worker's general health risk or likelihood of adverse outcomes. This score is derived from a combination of vital signs, reported symptoms, demographic data, and contextual factors. A higher score indicates higher risk (e.g., scores ≥ {getattr(app_config, 'RISK_SCORE_HIGH_THRESHOLD', 'N/A')} are considered high risk).", "RISK_SCORE_HIGH_THRESHOLD")
display_term_sentinel_glossary("AI Follow-up / Task Priority Score", f"A simulated composite score (typically 0-100) generated by AI or rule-based logic to help prioritize which patients require more urgent follow-up by CHWs, or which operational tasks need immediate attention by clinic/district staff. Scores exceeding a threshold (e.g., ≥ {getattr(app_config, 'FATIGUE_INDEX_HIGH_THRESHOLD', 'N/A')} for high priority tasks) are flagged.", "FATIGUE_INDEX_HIGH_THRESHOLD") # Using FATIGUE_INDEX_HIGH_THRESHOLD as an example placeholder for a general high priority task threshold
display_term_sentinel_glossary("Ambient Heat Index (°C)", f"A measure indicating how hot it feels when relative humidity is combined with air temperature. The Sentinel system may trigger alerts at specific risk levels, e.g., Risk ≥ {getattr(app_config, 'ALERT_AMBIENT_HEAT_INDEX_RISK_C', 'N/A')}°C, Danger ≥ {getattr(app_config, 'ALERT_AMBIENT_HEAT_INDEX_DANGER_C', 'N/A')}°C.", "ALERT_AMBIENT_HEAT_INDEX_RISK_C")
display_term_sentinel_glossary("Key Conditions for Action", f"A curated list of high-priority health conditions that the Sentinel system is specifically configured to monitor and trigger actions for. Examples include: {', '.join(getattr(app_config, 'KEY_CONDITIONS_FOR_ACTION', ['TB, Malaria'])[:3])}...", "KEY_CONDITIONS_FOR_ACTION")
display_term_sentinel_glossary("SpO₂ (Peripheral Capillary Oxygen Saturation)", f"An estimate of blood oxygen levels. Critical Low: < {getattr(app_config, 'ALERT_SPO2_CRITICAL_LOW_PCT', 'N/A')}%. Warning Low: < {getattr(app_config, 'ALERT_SPO2_WARNING_LOW_PCT', 'N/A')}%.", "ALERT_SPO2_CRITICAL_LOW_PCT")
display_term_sentinel_glossary("TAT (Test Turnaround Time)", "Time from sample collection/receipt to result availability. Overall facility target might be around {getattr(app_config, 'TARGET_TEST_TURNAROUND_DAYS', 'N/A')} days, with specific targets for critical tests.", "TARGET_TEST_TURNAROUND_DAYS")
display_term_sentinel_glossary("HRV (Heart Rate Variability)", "Variation in time between heartbeats, an indicator of stress/fatigue. Low HRV (e.g., RMSSD < {getattr(app_config, 'STRESS_HRV_LOW_THRESHOLD_MS', 'N/A')}ms) can signal high physiological stress.", "STRESS_HRV_LOW_THRESHOLD_MS")
display_term_sentinel_glossary("Facility Coverage Score (Zonal)", f"A proxy metric (0-100%) indicating health facility accessibility/capacity for a zone. Scores below {getattr(app_config, 'DISTRICT_INTERVENTION_FACILITY_COVERAGE_LOW_PCT', 'N/A')}% may flag a zone for review.", "DISTRICT_INTERVENTION_FACILITY_COVERAGE_LOW_PCT")
# Add more terms such as: Antigen Test, PCR Test, DOTS, Syndromic Surveillance, Endemic, Epidemic, Pandemic, etc.
# Ensure they are relevant to LMIC context and the system's scope.

# --- III. Technical & Data Format Terms ---
st.header("💻 Technical, Data & Platform")
display_term_sentinel_glossary("API (Application Programming Interface)", "A defined way for different software components to communicate, e.g., PED to Hub, or Facility Node to a national Health Information System (HIS).")
display_term_sentinel_glossary("CSV (Comma-Separated Values)", "A plain text file format for storing tabular data, often used for data import/export and simple logging.")
display_term_sentinel_glossary("FHIR (Fast Healthcare Interoperability Resources)", "A global standard (from HL7) for exchanging healthcare information electronically. The Sentinel system aims for FHIR compliance at Tiers 2 and 3 for data integration.")
display_term_sentinel_glossary("GeoJSON", "An open standard format for encoding geographic data features (points, lines, polygons) and their attributes. Used for zone boundaries and mapping.")
display_term_sentinel_glossary("GDF (GeoDataFrame)", "A data structure from the Python GeoPandas library, like a Pandas DataFrame but with added capabilities for handling and analyzing geospatial data (geometries).")
display_term_sentinel_glossary("IoT (Internet of Things)", "Network of physical devices (e.g., environmental sensors in clinics, wearable sensors) that collect and transmit data.")
display_term_sentinel_glossary("JSON (JavaScript Object Notation)", "A lightweight, human-readable data-interchange format. Used for configurations (e.g., pictogram maps, haptic patterns) and API data transfer.")
display_term_sentinel_glossary("SQLite", "A lightweight, file-based SQL database engine, suitable for local data storage on PEDs and Supervisor Hubs where a full database server is not feasible.")
display_term_sentinel_glossary("TFLite (TensorFlow Lite)", "A set of tools and an optimized runtime for running TensorFlow machine learning models on mobile, embedded, and IoT devices (Edge AI).")
display_term_sentinel_glossary("UI/UX (User Interface / User Experience)", "Design of the interaction between users and the Sentinel system, heavily focused on simplicity, intuitiveness, and actionability for frontline workers (Human-Centered Design).")
# Add other terms like: MQTT, Bluetooth LE, Wi-Fi Direct, SD Card Sync, QR Data Packet if detailing sync methods.

st.markdown("---")
st.caption(app_config.APP_FOOTER_TEXT)
logger.info(f"Glossary page for {app_config.APP_NAME} loaded successfully.")
