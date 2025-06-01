# sentinel_project_root/test/config/app_config.py
# Configuration for "Sentinel Health Co-Pilot" - LMIC Edge-First System

import os
import pandas as pd # Retained for APP_FOOTER_TEXT year calculation

# --- I. Core System & Directory Configuration ---
# BASE_APP_ROOT_DIR assumes this config file is in sentinel_project_root/test/config/
# This makes BASE_APP_ROOT_DIR equivalent to sentinel_project_root/test/
BASE_APP_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ASSETS_DIR = os.path.join(BASE_APP_ROOT_DIR, "assets")
DATA_SOURCES_DIR = os.path.join(BASE_APP_ROOT_DIR, "data_sources")
# FACILITY_NODE_DATA_DIR for aggregated outputs / intermediate data at a facility node
FACILITY_NODE_DATA_DIR = os.path.join(BASE_APP_ROOT_DIR, "facility_node_data")
# PED local data is on the device; this path is illustrative for simulation
LOCAL_DATA_DIR_PED_SIM = os.path.join(BASE_APP_ROOT_DIR, "local_data_ped_sim")

# Paths to primary data sources (for simulation/Tier2-3 loading)
HEALTH_RECORDS_CSV = os.path.join(DATA_SOURCES_DIR, "health_records_expanded.csv")
ZONE_ATTRIBUTES_CSV = os.path.join(DATA_SOURCES_DIR, "zone_attributes.csv")
ZONE_GEOMETRIES_GEOJSON = os.path.join(DATA_SOURCES_DIR, "zone_geometries.geojson")
IOT_CLINIC_ENVIRONMENT_CSV = os.path.join(DATA_SOURCES_DIR, "iot_clinic_environment.csv")

APP_NAME = "Sentinel Health Co-Pilot"
APP_VERSION = "3.1.0" # Version bump after refactoring
APP_LOGO_SMALL = os.path.join(ASSETS_DIR, "sentinel_logo_small.png")
APP_LOGO_LARGE = os.path.join(ASSETS_DIR, "sentinel_logo_large.png")
STYLE_CSS_PATH_WEB = os.path.join(ASSETS_DIR, "style_web_reports.css")

ORGANIZATION_NAME = "LMIC Health Futures Initiative"
APP_FOOTER_TEXT = f"© {pd.Timestamp('now').year} {ORGANIZATION_NAME}. Actionable Intelligence for Resilient Health Systems."
SUPPORT_CONTACT_INFO = "support@lmic-health-futures.org" # Placeholder

# --- II. LMIC-Specific Health & Operational Thresholds ---
ALERT_SPO2_CRITICAL_LOW_PCT = 90
ALERT_SPO2_WARNING_LOW_PCT = 94
ALERT_BODY_TEMP_FEVER_C = 38.0
ALERT_BODY_TEMP_HIGH_FEVER_C = 39.5
ALERT_HR_TACHYCARDIA_ADULT_BPM = 100
ALERT_HR_BRADYCARDIA_ADULT_BPM = 50
HEAT_STRESS_BODY_TEMP_TARGET_C = 37.5 # Target for cooling efforts after heat stress
HEAT_STRESS_RISK_BODY_TEMP_C = 38.5   # Skin/body temp indicating notable heat stress risk

ALERT_AMBIENT_CO2_HIGH_PPM = 1500     # Elevated CO2, potential ventilation issue
ALERT_AMBIENT_CO2_VERY_HIGH_PPM = 2500 # Very high CO2, significant concern
ALERT_AMBIENT_PM25_HIGH_UGM3 = 35     # High PM2.5, WHO interim target 1
ALERT_AMBIENT_PM25_VERY_HIGH_UGM3 = 50 # Very high PM2.5
ALERT_AMBIENT_NOISE_HIGH_DBA = 85     # Sustained exposure risk (e.g., OSHA)
ALERT_AMBIENT_HEAT_INDEX_RISK_C = 32   # Heat Index where caution/risk begins
ALERT_AMBIENT_HEAT_INDEX_DANGER_C = 41 # Heat Index for danger level

FATIGUE_INDEX_MODERATE_THRESHOLD = 60 # Scale 0-100 from Edge AI for worker fatigue
FATIGUE_INDEX_HIGH_THRESHOLD = 80     # Threshold for CHW fatigue or generic high priority AI score
STRESS_HRV_LOW_THRESHOLD_MS = 20      # Example RMSSD/SDNN for HRV-indicated stress

TARGET_CLINIC_WAITING_ROOM_OCCUPANCY_MAX = 10 # Persons, for overcrowding alerts
TARGET_CLINIC_PATIENT_THROUGHPUT_MIN_PER_HOUR = 5 # Persons, operational target

RISK_SCORE_LOW_THRESHOLD = 40      # Patient AI Risk Score: Green
RISK_SCORE_MODERATE_THRESHOLD = 60 # Patient AI Risk Score: Yellow/Amber
RISK_SCORE_HIGH_THRESHOLD = 75     # Patient AI Risk Score: Red/Critical

DISTRICT_ZONE_HIGH_RISK_AVG_SCORE = 70 # DHO level: avg zone AI risk score > this = high risk zone
DISTRICT_INTERVENTION_FACILITY_COVERAGE_LOW_PCT = 60 # DHO level: facility coverage < this = concern
DISTRICT_INTERVENTION_TB_BURDEN_HIGH_ABS = 10 # Absolute TB cases per zone for DHO intervention flag
DISTRICT_INTERVENTION_KEY_DISEASE_PREVALENCE_HIGH_PERCENTILE = 0.80 # Top 20% of zones by prevalence

CRITICAL_SUPPLY_DAYS_REMAINING = 7 # Threshold for critical supply alert
LOW_SUPPLY_DAYS_REMAINING = 14     # Threshold for low supply warning
TARGET_DAILY_STEPS = 8000          # General wellness target reference for patient activity

# --- III. Edge Device (PED) & Application Configuration ---
EDGE_APP_DEFAULT_LANGUAGE = "en"
EDGE_APP_SUPPORTED_LANGUAGES = ["en", "sw", "fr"] # Examples: English, Swahili, French
EDGE_APP_PICTOGRAM_CONFIG_FILE = os.path.join(ASSETS_DIR, "pictogram_map.json")
EDGE_APP_HAPTIC_CONFIG_FILE = os.path.join(ASSETS_DIR, "haptic_patterns.json")
AUDIO_ALERT_FILES_DIR = os.path.join(ASSETS_DIR, "audio_alerts") # e.g., assets/audio_alerts/en/critical_spo2.mp3

EDGE_MODEL_VITALS_DETERIORATION = "vitals_deterioration_v1.tflite"
EDGE_MODEL_FATIGUE_INDEX = "fatigue_index_v1.tflite" # For CHW self-monitoring
EDGE_MODEL_PERSONALIZED_ANOMALY = "anomaly_detection_base.tflite" # For patient baselines
PERSONALIZED_BASELINE_WINDOW_DAYS = 7 # Days of data for personal baseline
EDGE_PROCESSING_INTERVAL_SECONDS = 60 # How often PED AI models might re-evaluate

PED_SQLITE_DB_NAME = "sentinel_ped.db"
PED_MAX_LOCAL_LOG_SIZE_MB = 50
EDGE_SYNC_PROTOCOL_PRIORITY = ["BLUETOOTH_PEER", "WIFI_DIRECT_HUB", "QR_PACKET_SHARE", "SD_CARD_TRANSFER"]
QR_PACKET_MAX_SIZE_BYTES = 256 # Max size for a single QR code data packet
SMS_COMPRESSION_SCHEME = "BASE85_ZLIB" # Example for highly compressed SMS data

# --- IV. Data Input Configuration (Conceptual) ---
# Detailed schemas (e.g., JSON Schema) could be defined for data validation if needed.
# Example: DEMOGRAPHICS_FIELDS_SCHEMA_PATH = os.path.join(ASSETS_DIR, "schemas/demographics_schema.json")

# --- V. Supervisor Hub & Facility Node Configuration ---
HUB_DEVICE_SQLITE_DB_NAME = "sentinel_hub.db"
FACILITY_NODE_DB_TYPE = "POSTGRESQL" # Example for a larger facility node
FHIR_SERVER_ENDPOINT_LOCAL = "http://localhost:8080/fhir" # Example Facility Node FHIR endpoint
DEFAULT_REPORT_INTERVAL_HOURS = 24 # How often batch reports might be generated
CACHE_TTL_SECONDS_WEB_REPORTS = 3600 # 1 hour for Streamlit web report data caching

ESCALATION_PROTOCOL_FILE = os.path.join(ASSETS_DIR, "escalation_protocols.json")

# --- VI. Key Data Semantics & Definitions ---
KEY_TEST_TYPES_FOR_ANALYSIS = { # For analytics, reporting, and string matching
    "Sputum-AFB": {"disease_group": "TB", "target_tat_days": 2, "critical": True, "display_name": "TB Sputum (AFB)"},
    "Sputum-GeneXpert": {"disease_group": "TB", "target_tat_days": 1, "critical": True, "display_name": "TB GeneXpert"},
    "RDT-Malaria": {"disease_group": "Malaria", "target_tat_days": 0.5, "critical": True, "display_name": "Malaria RDT"},
    "HIV-Rapid": {"disease_group": "HIV", "target_tat_days": 0.25, "critical": True, "display_name": "HIV Rapid Test"},
    "HIV-ViralLoad": {"disease_group": "HIV", "target_tat_days": 7, "critical": True, "display_name": "HIV Viral Load"},
    "BP Check": {"disease_group": "Hypertension", "target_tat_days": 0, "critical": False, "display_name": "BP Check"},
    # Add other relevant tests if they are specifically analyzed or reported on.
}
CRITICAL_TESTS_LIST = [k for k, p in KEY_TEST_TYPES_FOR_ANALYSIS.items() if p.get("critical")]

TARGET_TEST_TURNAROUND_DAYS = 2 # General default TAT for non-specific tests
TARGET_OVERALL_TESTS_MEETING_TAT_PCT_FACILITY = 85 # Facility-level target for % tests meeting TAT
TARGET_SAMPLE_REJECTION_RATE_PCT_FACILITY = 5     # Facility-level target for sample rejection
OVERDUE_PENDING_TEST_DAYS_GENERAL_FALLBACK = 7 # For overdue test calculation if specific TAT not found

KEY_CONDITIONS_FOR_ACTION = ['TB', 'Malaria', 'HIV-Positive', 'Pneumonia', 'Severe Dehydration', 'Heat Stroke', 'Sepsis', 'Diarrheal Diseases (Severe)']
KEY_DRUG_SUBSTRINGS_SUPPLY = ['TB-Regimen', 'ACT', 'ARV-Regimen', 'ORS', 'Amoxicillin', 'Paracetamol', 'Penicillin', 'Iron-Folate', 'Insulin']

TARGET_MALARIA_POSITIVITY_RATE = 5.0 # Target % for malaria test positivity (example for dashboards)

# --- VII. Web Dashboarding Configuration (Tiers 2/3) ---
# Default map center (Can be overridden by user or dynamic context)
MAP_DEFAULT_CENTER_LAT = 0.0  # Generic default (e.g., Equator)
MAP_DEFAULT_CENTER_LON = 0.0
MAP_DEFAULT_ZOOM = 2         # Global zoom

WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_VIEW = 7   # Default for views showing recent activity (e.g., 7 days)
WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND = 30 # Default for trend analysis (e.g., 30 days)
WEB_PLOT_DEFAULT_HEIGHT = 400
WEB_PLOT_COMPACT_HEIGHT = 320
WEB_MAP_DEFAULT_HEIGHT = 600

MAPBOX_STYLE_WEB = "carto-positron" # Default open style; ui_helpers will manage actual choice based on token
DEFAULT_CRS_STANDARD = "EPSG:4326" # Standard WGS84 for GeoJSON and general mapping

LOG_LEVEL = os.getenv("SENTINEL_LOG_LEVEL", "INFO").upper() # Allow override by ENV VAR, ensure upper for logging module
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - [%(module)s.%(funcName)s:%(lineno)d] - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# --- VIII. Sentinel System Color Palette (for consistency in UIs) ---
# These are the primary definitions. CSS variables in style_web_reports.css should mirror these.
COLOR_RISK_HIGH = "#D32F2F"         # Strong Red (critical alerts, high risk KPI)
COLOR_RISK_MODERATE = "#FBC02D"     # Amber/Yellow (warnings, moderate risk KPI)
COLOR_RISK_LOW = "#388E3C"          # Green (good/acceptable/low risk KPI)
COLOR_RISK_NEUTRAL = "#757575"      # Grey (neutral/no data KPI)
COLOR_ACTION_PRIMARY = "#1976D2"      # Strong Blue (primary buttons, active elements)
COLOR_ACTION_SECONDARY = "#546E7A"    # Slate Blue/Grey (secondary elements)
COLOR_ACCENT_BRIGHT = "#4D7BF3"    # A brighter blue for specific accents (e.g., h1 underline)
COLOR_POSITIVE_DELTA = "#27AE60"      # Green for positive changes in KPIs
COLOR_NEGATIVE_DELTA = "#C0392B"      # Red for negative changes in KPIs
COLOR_TEXT_DARK = "#343a40"             # Main body text on light backgrounds
COLOR_TEXT_HEADINGS_MAIN = "#1A2557"   # Main H1/H2 headings
COLOR_TEXT_HEADINGS_SUB = "#2C3E50"  # Secondary headings, labels
COLOR_TEXT_MUTED = "#6c757d"         # Lighter grey for captions, details
COLOR_TEXT_LINK_DEFAULT = COLOR_ACTION_PRIMARY # Use primary action color for links for consistency
COLOR_BACKGROUND_PAGE = "#f8f9fa"     # Standard light page background
COLOR_BACKGROUND_CONTENT = "#ffffff"      # Card/component backgrounds
COLOR_BACKGROUND_SUBTLE = "#e9ecef"   # Very light grey for hovers, subtle distinctions
COLOR_BORDER_LIGHT = "#dee2e6"          # Light borders for cards, tables
COLOR_BORDER_MEDIUM = "#ced4da"          # Medium borders for dividers, table headers

# Legacy disease colors for web reports (if needed for continuity or specific charts).
# PEDs would use simplified, high-contrast iconic disease representation.
LEGACY_DISEASE_COLORS_WEB = {
    "TB": "#EF4444", "Malaria": "#F59E0B", "HIV-Positive": "#8B5CF6",
    "Pneumonia": "#3B82F6", "Anemia": "#10B981", "STI": "#EC4899", "Dengue": "#6366F1",
    "Hypertension": "#F97316", "Diabetes": "#0EA5E9", "Wellness Visit": "#84CC16",
    "Other": "#6B7280", "STI-Syphilis": "#c026d3", "STI-Gonorrhea": "#db2777",
    "Heat Stroke": "#FF6347", "Severe Dehydration": "#4682B4", "Sepsis": "#800080",
    "Diarrheal Diseases (Severe)": "#D2691E"
}
# Auto-populate from KEY_CONDITIONS_FOR_ACTION if not explicitly defined, using related keywords if possible
for cond in KEY_CONDITIONS_FOR_ACTION:
    if cond not in LEGACY_DISEASE_COLORS_WEB:
        cond_lower = cond.lower()
        assigned_color = "#A9A9A9" # Default grey
        for disease_keyword, color_val in [("tb", LEGACY_DISEASE_COLORS_WEB.get("TB")),
                                           ("malaria", LEGACY_DISEASE_COLORS_WEB.get("Malaria")),
                                           ("hiv", LEGACY_DISEASE_COLORS_WEB.get("HIV-Positive")),
                                           ("pneumonia", LEGACY_DISEASE_COLORS_WEB.get("Pneumonia")),
                                           ("dehydration", LEGACY_DISEASE_COLORS_WEB.get("Severe Dehydration")),
                                           ("heat", LEGACY_DISEASE_COLORS_WEB.get("Heat Stroke")), # "heat stroke"
                                           ("sepsis", LEGACY_DISEASE_COLORS_WEB.get("Sepsis")),
                                           ("diarrheal", LEGACY_DISEASE_COLORS_WEB.get("Diarrheal Diseases (Severe)"))]:
            if disease_keyword in cond_lower and color_val:
                assigned_color = color_val
                break
        LEGACY_DISEASE_COLORS_WEB[cond] = assigned_color

# --- End of Configuration ---
