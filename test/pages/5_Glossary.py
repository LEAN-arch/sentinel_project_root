# sentinel_project_root/test/pages/5_Glossary.py
# Glossary of Terms for the "Sentinel Health Co-Pilot" System

import streamlit as st
import os
# import sys # sys.path manipulation removed, Streamlit handles page imports
from config import app_config # For APP_NAME, APP_FOOTER_TEXT, and specific thresholds
import logging
from typing import Optional # For type hinting in helper function

# --- Page Configuration ---
# Page icon can be set here if a specific one for Glossary is desired,
# otherwise, it inherits from app_home.py or Streamlit's default.
st.set_page_config(
    page_title=f"Glossary - {app_config.APP_NAME}",
    layout="wide", # Consistent layout
    initial_sidebar_state="expanded" # Consistent sidebar state
)

logger = logging.getLogger(__name__) # Page-specific logger
# CSS loading is handled globally by app_home.py

# --- Glossary Content ---
st.title(f"📜 {app_config.APP_NAME} - Glossary of Terms")
st.markdown(
    "This page provides definitions for common terms, abbreviations, metrics, and system-specific concepts "
    "used throughout the Sentinel Health Co-Pilot platform, its dashboards, and documentation."
)
st.divider() # Modern separator for sections

# Helper function for consistent term formatting and display
def display_term_sentinel(term: str, definition: str, related_config_var: Optional[str] = None):
    """ Displays a glossary term and its definition in a standardized format. """
    st.markdown(f"#### {term}") # Using H4 for individual terms for better semantic structure
    st.markdown(f"*{definition}*") # Italicize the definition for emphasis
    if related_config_var:
        # Attempt to display the actual value from app_config if the variable name is valid
        try:
            config_val_display = getattr(app_config, related_config_var, None)
            if config_val_display is not None:
                st.caption(f"*(Related config: `app_config.{related_config_var}` = `{config_val_display}`)*")
            else: # Variable name not found in app_config
                st.caption(f"*(Related configuration in `app_config.py`: `{related_config_var}`)*")
        except AttributeError: # Should not happen if getattr used with default
             st.caption(f"*(Related configuration in `app_config.py`: `{related_config_var}`)*")
    st.markdown("---") # Visual separator after each term definition


# --- I. Sentinel Health Co-Pilot System Concepts ---
st.header("🌐 System Architecture & Core Concepts")
display_term_sentinel(
    term="Sentinel Health Co-Pilot", 
    definition="An edge-first health intelligence and action support system. It's designed for resource-limited, high-risk LMIC environments, prioritizing offline functionality, actionable insights for frontline health workers (FHWs), and resilient data flow."
)
display_term_sentinel(
    term="Personal Edge Device (PED)", 
    definition="A ruggedized smartphone, wearable sensor array, or low-power System-on-Chip (SoC) device utilized by FHWs. It runs native applications featuring on-device Edge AI for real-time physiological monitoring, environmental sensing, alert generation, prioritized task management, and Just-In-Time (JIT) guidance protocols, primarily designed for offline operation."
)
display_term_sentinel(
    term="Edge AI / TinyML", 
    definition="Artificial Intelligence (AI) models, often leveraging frameworks like TensorFlow Lite (TFLite) or specialized TinyML libraries, that are optimized to execute directly on PEDs or local Supervisor Hubs. These models operate with minimal computational resources, enabling offline decision support, anomaly detection, and risk stratification at the point of care."
)
display_term_sentinel(
    term="Supervisor Hub (Tier 1)", 
    definition="An optional intermediary device (e.g., tablet, rugged phone) used by a CHW Supervisor or team leader. It locally aggregates data from team members' PEDs via short-range communication (e.g., Bluetooth, Wi-Fi Direct), allowing for localized team oversight, simple dashboard views, and batched data transfer to higher tiers (Facility Nodes)."
)
display_term_sentinel(
    term="Facility Node (Tier 2)", 
    definition="A local server, PC, or robust computing device (e.g., Raspberry Pi, mini-PC) situated at a clinic, health post, or community health center. It aggregates data from Supervisor Hubs or directly from PEDs, can perform more complex local analytics, potentially interfaces with local Electronic Medical Records (EMRs), generates facility-level reports and dashboards (like the Clinic Console), and serves as a crucial staging point for wider data synchronization to regional or cloud systems."
)
display_term_sentinel(
    term="Regional/Cloud Node (Tier 3)", 
    definition="Optional centralized infrastructure, which can be an on-premise regional server or a cloud-based platform. This tier is designed for population-level analytics, epidemiological surveillance, advanced AI model training and refinement using aggregated data, and national-level health reporting. It receives batched data from multiple Facility Nodes."
)
display_term_sentinel(
    term="Lean Data Inputs", 
    definition=f"A core design principle focused on collecting only the minimum viable data points that possess maximum predictive power for Edge AI models and are directly actionable by FHWs. This approach is tailored for constrained LMIC settings. Examples include: age group, chronic condition flag (Yes/No), SpO₂ below a critical threshold (e.g., < {app_config.ALERT_SPO2_CRITICAL_LOW_PCT}%), an observed fatigue flag, or a few key reported symptoms.", 
    related_config_var="ALERT_SPO2_CRITICAL_LOW_PCT" # Example of referencing a specific config var
)
display_term_sentinel(
    term="Action Code / Suggested Action Code", 
    definition="A system-internal alphanumeric code (e.g., 'ACTION_SPO2_MANAGE_URGENT', 'TASK_VISIT_VITALS_URGENT') generated by an alert, an AI model output, or a task. On a PED, this code is mapped (e.g., via `pictogram_map.json` or `escalation_protocols.json`) to display a specific pictogram, trigger Just-In-Time guidance media, initiate an automated communication workflow, or guide the user through a step in a digital protocol."
)
display_term_sentinel(
    term="Opportunistic Sync", 
    definition="A data synchronization strategy where Sentinel devices (PEDs, Hubs, Facility Nodes) transfer data to higher tiers only when a viable, low-cost, and sufficiently stable communication channel becomes available. Examples include Bluetooth peer-to-peer, local Wi-Fi to a hub, brief windows of cellular connectivity, or even physical data transfer via SD card or QR code packets. This is vital for environments with intermittent, unreliable, or expensive internet access."
)

# --- II. Clinical, Epidemiological & Operational Terms ---
st.header("🩺 Clinical, Epidemiological & Operational Terms")
display_term_sentinel(
    term="AI Risk Score (Patient/Worker)", 
    definition=f"A simulated algorithmic score (typically 0-100) predicting an individual's general health risk or likelihood of adverse outcomes. It's derived from a combination of vital signs, reported symptoms, demographic data, and contextual factors. A higher score indicates higher risk (e.g., High Risk generally ≥ {app_config.RISK_SCORE_HIGH_THRESHOLD}).", 
    related_config_var="RISK_SCORE_HIGH_THRESHOLD"
)
display_term_sentinel(
    term="AI Follow-up Priority Score / Task Priority Score", 
    definition=f"A simulated score (0-100) generated by an AI model or rule-set to help prioritize which patients require more urgent follow-up by a CHW/clinic, or which tasks (e.g., home visit, referral tracking) should be attended to first. A high priority is often associated with scores ≥ {app_config.FATIGUE_INDEX_HIGH_THRESHOLD} (this config is also used as a generic high AI priority threshold).", 
    related_config_var="FATIGUE_INDEX_HIGH_THRESHOLD"
)
display_term_sentinel(
    term="Ambient Heat Index (°C)", 
    definition=f"A measure indicating how hot it feels when relative humidity is combined with the actual air temperature. It's a more accurate representation of heat stress risk than air temperature alone. Sentinel uses this for heat stress alerts (e.g., Risk level at {app_config.ALERT_AMBIENT_HEAT_INDEX_RISK_C}°C, Danger level at {app_config.ALERT_AMBIENT_HEAT_INDEX_DANGER_C}°C).", 
    related_config_var="ALERT_AMBIENT_HEAT_INDEX_DANGER_C"
)
display_term_sentinel(
    term="Condition (Key Actionable)", 
    definition=f"Refers to specific health conditions that the Sentinel system is configured to prioritize for monitoring, alert generation, and targeted response protocols. These are typically listed in `app_config.KEY_CONDITIONS_FOR_ACTION` (e.g., '{app_config.KEY_CONDITIONS_FOR_ACTION[0]}', '{app_config.KEY_CONDITIONS_FOR_ACTION[1]}', etc.).", 
    related_config_var="KEY_CONDITIONS_FOR_ACTION"
)
display_term_sentinel(
    term="Encounter (CHW/Clinic)", 
    definition="Any interaction a patient has with the health system or a CHW that is documented within the Sentinel platform. This can include CHW home visits, clinic consultations, responses by a CHW to a PED-generated alert, scheduled follow-up appointments, or remote check-ins facilitated by the system."
)
display_term_sentinel(
    term="Facility Coverage Score (Zonal)", 
    definition=f"A district-level metric (0-100%) designed to reflect the adequacy of health facility access and capacity relative to a zone's population. It can be calculated based on factors like population per clinic, average travel times, or service availability. Low coverage (e.g., < {app_config.DISTRICT_INTERVENTION_FACILITY_COVERAGE_LOW_PCT}%) may trigger a review by District Health Officers.", 
    related_config_var="DISTRICT_INTERVENTION_FACILITY_COVERAGE_LOW_PCT"
)
display_term_sentinel(
    term="Fatigue Index Score (Worker)", 
    definition=f"A simulated score (0-100) indicating a frontline health worker's (e.g., CHW) current level of fatigue. This score is typically derived by an Edge AI model on their PED, using inputs like Heart Rate Variability (HRV), activity patterns from motion sensors, and/or self-reported psychometric data. Configured alert levels: Moderate Fatigue ≥ {app_config.FATIGUE_INDEX_MODERATE_THRESHOLD}, High Fatigue ≥ {app_config.FATIGUE_INDEX_HIGH_THRESHOLD}.", 
    related_config_var="FATIGUE_INDEX_HIGH_THRESHOLD"
)
display_term_sentinel(
    term="HRV (Heart Rate Variability)", 
    definition=f"The physiological phenomenon of variation in the time interval between consecutive heartbeats. It is typically measured in milliseconds (ms), with common metrics being RMSSD (Root Mean Square of Successive Differences) or SDNN (Standard Deviation of NN intervals). Low HRV (e.g., RMSSD < {app_config.STRESS_HRV_LOW_THRESHOLD_MS}ms) can be an indicator of increased physiological stress, fatigue, or reduced adaptability of the autonomic nervous system.",
    related_config_var="STRESS_HRV_LOW_THRESHOLD_MS"
)
display_term_sentinel(
    term="SpO₂ (Peripheral Capillary Oxygen Saturation)", 
    definition=f"An estimate of the amount of oxygen carried by hemoglobin in the blood, expressed as a percentage. Measured non-invasively, typically with a pulse oximeter. Critically Low SpO₂ threshold in Sentinel is generally set around {app_config.ALERT_SPO2_CRITICAL_LOW_PCT}%. A Warning Low SpO₂ level is around {app_config.ALERT_SPO2_WARNING_LOW_PCT}%.", 
    related_config_var="ALERT_SPO2_CRITICAL_LOW_PCT"
)
display_term_sentinel(
    term="TAT (Test Turnaround Time)", 
    definition=f"The total time elapsed from the point of sample collection (or sample registration at the lab/testing point) to when the test result is finalized, validated, and made available to the clinician or patient. Target TATs vary by test type; for critical tests, a general target might be around {app_config.TARGET_TEST_TURNAROUND_DAYS} days, but specific tests have their own targets (see `KEY_TEST_TYPES_FOR_ANALYSIS` in app_config).", 
    related_config_var="TARGET_TEST_TURNAROUND_DAYS"
)


# --- III. Technical & Data Format Terms ---
st.header("💻 Technical, Data & Platform Terms")
display_term_sentinel(
    term="API (Application Programming Interface)", 
    definition="A defined set of rules, protocols, and tools that allows different software applications or components to communicate and exchange data with each other. In Sentinel, APIs are used for tasks like data synchronization between system tiers (e.g., Facility Node to a Regional/Cloud Node) or for integration with external health information systems (e.g., DHIS2, local EMRs)."
)
display_term_sentinel(
    term="CSV (Comma-Separated Values)", 
    definition="A simple text file format where data values in a table are stored as plain text, with each value separated by a comma, and each row on a new line. Widely used in Sentinel for importing raw data sources (e.g., `health_records_expanded.csv`), exporting data for analysis, or for simple reporting outputs."
)
display_term_sentinel(
    term="FHIR (Fast Healthcare Interoperability Resources)", 
    definition="Pronounced 'Fire'. An international standard (from HL7®) describing data formats and elements (known as 'Resources') and an Application Programming Interface (API) for exchanging electronic health records (EHR). FHIR is crucial for achieving interoperability between different health IT systems. Sentinel aims to support FHIR for data exchange at Tier 2 (Facility Nodes) and Tier 3 (Regional/Cloud Nodes) to integrate with broader health ecosystems.",
    related_config_var="FHIR_SERVER_ENDPOINT_LOCAL"
)
display_term_sentinel(
    term="GeoJSON", 
    definition=f"An open standard format, based on JSON (JavaScript Object Notation), designed for encoding a variety of geographic data structures such as points, lines, and polygons, along with their non-spatial attributes. Sentinel uses GeoJSON for representing operational zone boundaries and potentially other spatial features like facility locations. The default Coordinate Reference System (CRS) used is {app_config.DEFAULT_CRS_STANDARD}.",
    related_config_var="ZONE_GEOMETRIES_GEOJSON"
)
display_term_sentinel(
    term="GDF (GeoDataFrame)", 
    definition="A GeoDataFrame is a tabular data structure provided by the Python GeoPandas library. It extends the capabilities of a standard Pandas DataFrame by adding a special 'geometry' column that stores geospatial data (e.g., Points, Lines, Polygons from Shapely library). This allows for spatial operations, analysis, and mapping directly on the data structure."
)
display_term_sentinel(
    term="IoT (Internet of Things)", 
    definition="A network of interconnected physical objects or 'things' (which can include sensors, actuators, software, and other technologies) that collect and exchange data over a network without requiring human-to-human or human-to-computer interaction. In Sentinel, this commonly refers to devices like clinic environmental sensors (monitoring CO2, temperature, PM2.5, noise levels) or potentially patient-worn biometric sensors that are not full PEDs."
)
display_term_sentinel(
    term="JSON (JavaScript Object Notation)", 
    definition="A lightweight, human-readable data-interchange format. It is easy for humans to read and write, and easy for machines to parse and generate. JSON is used extensively within the Sentinel system for various configuration files (e.g., `escalation_protocols.json`, `pictogram_map.json`, `haptic_patterns.json`) and often for data exchange in APIs between system components."
)
display_term_sentinel(
    term="Pictogram", 
    definition="A simple, iconic image or symbol that represents a concept, action, task, piece of information, or object. Pictograms are used extensively in Sentinel Personal Edge Device (PED) User Interfaces (UIs) to enhance clarity and understanding, especially for users with varying literacy levels or in multilingual contexts. The mapping of system codes to pictogram image files is defined in `pictogram_map.json`.",
    related_config_var="EDGE_APP_PICTOGRAM_CONFIG_FILE"
)
display_term_sentinel(
    term="QR Code Packet", 
    definition=f"A method for transferring small amounts of data completely offline by encoding the data into one or more QR (Quick Response) codes. These codes are displayed on one device's screen and then scanned by another device's camera. This is useful for low-bandwidth or no-connectivity scenarios, such as PED-to-Hub or PED-to-PED data exchange. The maximum size for a single QR code data packet in Sentinel is configured via `QR_PACKET_MAX_SIZE_BYTES` (currently {app_config.QR_PACKET_MAX_SIZE_BYTES} bytes).",
    related_config_var="QR_PACKET_MAX_SIZE_BYTES"
)
display_term_sentinel(
    term="SQLite", 
    definition="A C-language library that implements a small, fast, self-contained, high-reliability, full-featured, SQL database engine. SQLite is an embedded SQL database engine, meaning the database engine runs as part of the app. It's commonly used in Sentinel for local data storage on mobile devices (PEDs) and small Supervisor Hubs due to its portability and lack of need for a separate server process.",
    related_config_var="PED_SQLITE_DB_NAME"
)
display_term_sentinel(
    term="TFLite (TensorFlow Lite)", 
    definition="An open-source deep learning framework and a set of tools from Google, designed to help developers run TensorFlow models on mobile, embedded, and IoT devices. TFLite enables on-device machine learning with characteristics like low latency, small binary size, and efficient power consumption. It is a key technology for implementing Edge AI capabilities on Sentinel PEDs (e.g., for models like `vitals_deterioration_v1.tflite`).",
    related_config_var="EDGE_MODEL_VITALS_DETERIORATION"
)

st.divider() # Final divider
st.caption(app_config.APP_FOOTER_TEXT)
logger.info(f"Glossary page for {app_config.APP_NAME} (v{app_config.APP_VERSION}) loaded.")
