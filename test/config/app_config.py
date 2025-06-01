# sentinel_project_root/test/config/app_config.py
# Redesigned for "Sentinel Health Co-Pilot" - LMIC Edge-First System
# (With NameError fix for self-referential variable definition incorporated)

import os
import pandas as pd # Retained for potential use in higher-tier processing

# --- I. Core System & Directory Configuration ---
# BASE_DIR calculation assumes this config file's location relative to project root.
# This makes `BASE_DIR` equivalent to `sentinel_project_root/test/`
# If app_config.py moves to sentinel_project_root/config/, then BASE_DIR calc changes.
BASE_APP_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Should resolve to sentinel_project_root/test/

# ASSETS_DIR: For logos, pre-loaded UI pictograms, offline map tiles, JIT guidance media.
# Assuming 'assets' is a subdirectory within the 'test' app root
ASSETS_DIR = os.path.join(BASE_APP_ROOT_DIR, "assets")

# DATA_SOURCES_DIR: For raw input CSVs/GeoJSONs for simulation or initial loading
# Assuming 'data_sources' is a subdirectory within the 'test' app root for this example
DATA_SOURCES_DIR = os.path.join(BASE_APP_ROOT_DIR, "data_sources")

# FACILITY_NODE_DATA_DIR might be different in a real deployment (e.g., a mounted volume)
# For simulation, placing it within the app structure.
FACILITY_NODE_DATA_DIR = os.path.join(BASE_APP_ROOT_DIR, "facility_node_data") # For aggregated outputs / intermediate data

# PED local data is on the device itself, this path is illustrative for simulation
LOCAL_DATA_DIR_PED_SIM = os.path.join(BASE_APP_ROOT_DIR, "local_data_ped_sim")

# Paths to primary data sources (used by loading functions for simulation/Tier2-3)
# These now point to the DATA_SOURCES_DIR for raw inputs.
HEALTH_RECORDS_CSV = os.path.join(DATA_SOURCES_DIR, "health_records_expanded.csv")
ZONE_ATTRIBUTES_CSV = os.path.join(DATA_SOURCES_DIR, "zone_attributes.csv")
ZONE_GEOMETRIES_GEOJSON = os.path.join(DATA_SOURCES_DIR, "zone_geometries.geojson")
IOT_CLINIC_ENVIRONMENT_CSV = os.path.join(DATA_SOURCES_DIR, "iot_clinic_environment.csv")


APP_NAME = "Sentinel Health Co-Pilot"
APP_VERSION = "3.0.1-alpha" # Incremented after fixes
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
ALERT_HR_TACHYCARDIA_ADULT_BPM = 100 # Example
ALERT_HR_BRADYCARDIA_ADULT_BPM = 50  # Example
HEAT_STRESS_BODY_TEMP_TARGET_C = 37.5
HEAT_STRESS_RISK_BODY_TEMP_C = 38.5

ALERT_AMBIENT_CO2_HIGH_PPM = 1500
ALERT_AMBIENT_CO2_VERY_HIGH_PPM = 2500
ALERT_AMBIENT_PM25_HIGH_UGM3 = 35
ALERT_AMBIENT_PM25_VERY_HIGH_UGM3 = 50
ALERT_AMBIENT_NOISE_HIGH_DBA = 85     # For sustained exposure
ALERT_AMBIENT_HEAT_INDEX_RISK_C = 32   # Heat Index where caution is advised
ALERT_AMBIENT_HEAT_INDEX_DANGER_C = 41 # Heat Index for danger level

FATIGUE_INDEX_MODERATE_THRESHOLD = 60 # Scale 0-100 from Edge AI
FATIGUE_INDEX_HIGH_THRESHOLD = 80     # Threshold used for CHW/AI prio score high examples
STRESS_HRV_LOW_THRESHOLD_MS = 20       # Example for RMSSD or SDNN, highly model specific

TARGET_CLINIC_WAITING_ROOM_OCCUPANCY_MAX = 10 # Persons
TARGET_CLINIC_PATIENT_THROUGHPUT_MIN_PER_HOUR = 5 # Persons

RISK_SCORE_LOW_THRESHOLD = 40          # Green
RISK_SCORE_MODERATE_THRESHOLD = 60     # Yellow
RISK_SCORE_HIGH_THRESHOLD = 75         # Red
DISTRICT_ZONE_HIGH_RISK_AVG_SCORE = 70 # DHO level: avg zone score > this = high risk zone
DISTRICT_INTERVENTION_FACILITY_COVERAGE_LOW_PCT = 60
DISTRICT_INTERVENTION_TB_BURDEN_HIGH_ABS = 10 # Absolute cases per zone for intervention flag
DISTRICT_INTERVENTION_KEY_DISEASE_PREVALENCE_HIGH_PERCENTILE = 0.80 # Top 20% by prevalence

CRITICAL_SUPPLY_DAYS_REMAINING = 7
LOW_SUPPLY_DAYS_REMAINING = 14

# --- III. Edge Device (PED) & Application Configuration ---
EDGE_APP_DEFAULT_LANGUAGE = "en"
EDGE_APP_SUPPORTED_LANGUAGES = ["en", "sw", "fr"] # Swahili, French examples
EDGE_APP_PICTOGRAM_CONFIG_FILE = os.path.join(ASSETS_DIR, "pictogram_map.json")
EDGE_APP_HAPTIC_CONFIG_FILE = os.path.join(ASSETS_DIR, "haptic_patterns.json")
AUDIO_ALERT_FILES_DIR = os.path.join(ASSETS_DIR, "audio_alerts") # e.g., assets/audio_alerts/en/critical_spo2.mp3

EDGE_MODEL_VITALS_DETERIORATION = "vitals_deterioration_v1.tflite"
EDGE_MODEL_FATIGUE_INDEX = "fatigue_index_v1.tflite"
EDGE_MODEL_PERSONALIZED_ANOMALY = "anomaly_detection_base.tflite"
PERSONALIZED_BASELINE_WINDOW_DAYS = 7
EDGE_PROCESSING_INTERVAL_SECONDS = 60

PED_SQLITE_DB_NAME = "sentinel_ped.db"
PED_MAX_LOCAL_LOG_SIZE_MB = 50
EDGE_SYNC_PROTOCOL_PRIORITY = ["BLUETOOTH_PEER", "WIFI_DIRECT_HUB", "QR_PACKET_SHARE", "SD_CARD_TRANSFER"]
QR_PACKET_MAX_SIZE_BYTES = 256
SMS_COMPRESSION_SCHEME = "BASE85_ZLIB" # Example compression

# --- IV. Data Input Configuration (Conceptual) ---
# Detailed schemas would be separate JSON/YAML files if needed.
# DEMOGRAPHICS_FIELDS_SCHEMA_PATH = os.path.join(ASSETS_DIR, "schemas/demographics_schema.json")

# --- V. Supervisor Hub & Facility Node Configuration ---
HUB_DEVICE_SQLITE_DB_NAME = "sentinel_hub.db"
FACILITY_NODE_DB_TYPE = "POSTGRESQL" # Example
FHIR_SERVER_ENDPOINT_LOCAL = "http://localhost:8080/fhir" # Example Facility Node FHIR endpoint
DEFAULT_REPORT_INTERVAL_HOURS = 24
CACHE_TTL_SECONDS_WEB_REPORTS = 3600

ESCALATION_PROTOCOL_FILE = os.path.join(ASSETS_DIR, "escalation_protocols.json")

# --- VI. Key Data Semantics & Definitions ---
KEY_TEST_TYPES_FOR_ANALYSIS = { # For higher-tier analytics and string matching
    "Sputum-AFB": {"disease_group": "TB", "target_tat_days": 2, "critical": True, "display_name": "TB Sputum (AFB)"},
    "Sputum-GeneXpert": {"disease_group": "TB", "target_tat_days": 1, "critical": True, "display_name": "TB GeneXpert"},
    "RDT-Malaria": {"disease_group": "Malaria", "target_tat_days": 0.5, "critical": True, "display_name": "Malaria RDT"},
    "HIV-Rapid": {"disease_group": "HIV", "target_tat_days": 0.25, "critical": True, "display_name": "HIV Rapid Test"},
    "HIV-ViralLoad": {"disease_group": "HIV", "target_tat_days": 7, "critical": True, "display_name": "HIV Viral Load"},
    "BP Check": {"disease_group": "Hypertension", "target_tat_days": 0, "critical": False, "display_name": "BP Check"},
    # Include other tests from the original expanded list if they are still relevant for reporting.
    # For brevity, not all are re-listed here but should be maintained if used.
}
CRITICAL_TESTS_LIST = [k for k, p in KEY_TEST_TYPES_FOR_ANALYSIS.items() if p.get("critical")]

TARGET_TEST_TURNAROUND_DAYS = 2 # General default TAT target for non-specific tests
TARGET_OVERALL_TESTS_MEETING_TAT_PCT_FACILITY = 85 # Facility-level target
TARGET_SAMPLE_REJECTION_RATE_PCT_FACILITY = 5     # Facility-level target
OVERDUE_PENDING_TEST_DAYS_GENERAL_FALLBACK = 7 # For overdue test calculation if specific TAT not found

KEY_CONDITIONS_FOR_ACTION = ['TB', 'Malaria', 'HIV-Positive', 'Pneumonia', 'Severe Dehydration', 'Heat Stroke', 'Sepsis', 'Diarrheal Diseases (Severe)']
KEY_DRUG_SUBSTRINGS_SUPPLY = ['TB-Regimen', 'ACT', 'ARV-Regimen', 'ORS', 'Amoxicillin', 'Paracetamol', 'Penicillin', 'Iron-Folate', 'Insulin'] # Expanded focused list

# Target used in Population Dashboard original KPI, retained if needed
TARGET_MALARIA_POSITIVITY_RATE = 5 # Target % for malaria test positivity

# --- VII. Legacy/Web Dashboarding Configuration (for Tiers 2/3) ---
# Define these first if they are used to set other defaults.
TIJUANA_CENTER_LAT = 32.5149    # Retained for historical context if web maps default to this.
TIJUANA_CENTER_LON = -117.0382   # Can be replaced with a more generic region if not relevant.
TIJUANA_DEFAULT_ZOOM = 10

WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_VIEW = 1
WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND = 30
WEB_PLOT_DEFAULT_HEIGHT = 400
WEB_PLOT_COMPACT_HEIGHT = 320
WEB_MAP_DEFAULT_HEIGHT = 600

# Uses the TIJUANA_ variables defined above in this module.
MAP_DEFAULT_CENTER_LAT = TIJUANA_CENTER_LAT
MAP_DEFAULT_CENTER_LON = TIJUANA_CENTER_LON
MAP_DEFAULT_ZOOM = TIJUANA_DEFAULT_ZOOM

MAPBOX_STYLE_WEB = "carto-positron" # Default open style, actual choice managed by ui_helpers
DEFAULT_CRS_STANDARD = "EPSG:4326"

LOG_LEVEL = "INFO" # Can be overridden by ENV VAR
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - [%(module)s.%(funcName)s:%(lineno)d] - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# --- VIII. Sentinel System Color Palette (for consistency in UIs) ---
COLOR_RISK_HIGH = "#D32F2F"         # Strong Red (for critical alerts, high risk KPI status)
COLOR_RISK_MODERATE = "#FBC02D"     # Strong Yellow/Amber (for warnings, moderate risk KPI status)
COLOR_RISK_LOW = "#388E3C"          # Strong Green (for good/acceptable/low risk KPI status)
COLOR_RISK_NEUTRAL = "#757575"      # Grey (for neutral/no data KPI status)
COLOR_ACTION_PRIMARY = "#1976D2"      # Strong Blue (for primary buttons, active elements in web views)
COLOR_ACTION_SECONDARY = "#546E7A"    # Blue-Grey (for secondary elements, less prominent info)
COLOR_POSITIVE_DELTA = "#27AE60"      # Green for positive changes in KPIs
COLOR_NEGATIVE_DELTA = "#C0392B"      # Red for negative changes in KPIs
COLOR_TEXT_DARK = "#343a40"             # Main body text on light backgrounds
COLOR_HEADINGS_DARK_BLUE = "#1A2557"   # Main H1/H2 headings
COLOR_BACKGROUND_LIGHT_GREY = "#f8f9fa" # Standard light page background
COLOR_BACKGROUND_WHITE = "#ffffff"      # Card/component backgrounds
COLOR_BORDER_LIGHT = "#dee2e6"          # Light borders
COLOR_TEXT_LINK_DEFAULT = "#007bff"     # Default link blue (can be same as ACTION_PRIMARY)

# Legacy disease colors primarily for web reports/dashboards if needed for continuity or specific charts.
# PEDs would use simplified, high-contrast iconic disease representation, not this palette directly.
LEGACY_DISEASE_COLORS_WEB = {
    "TB": "#EF4444", "Malaria": "#F59E0B", "HIV-Positive": "#8B5CF6",
    "Pneumonia": "#3B82F6", "Anemia": "#10B981", "STI": "#EC4899", "Dengue": "#6366F1",
    "Hypertension": "#F97316", "Diabetes": "#0EA5E9", "Wellness Visit": "#84CC16",
    "Other": "#6B7280", "STI-Syphilis": "#c026d3", "STI-Gonorrhea": "#db2777",
    "Heat Stroke": "#FF6347", "Severe Dehydration": "#4682B4", "Sepsis": "#800080",
    "Diarrheal Diseases (Severe)": "#D2691E"
}
# Adding KEY_CONDITIONS_FOR_ACTION to the color map if not already there
for cond in KEY_CONDITIONS_FOR_ACTION:
    if cond not in LEGACY_DISEASE_COLORS_WEB:
        # Assign a fallback color or a specific one if known
        if "TB" in cond: LEGACY_DISEASE_COLORS_WEB[cond] = LEGACY_DISEASE_COLORS_WEB.get("TB", "#A9A9A9")
        elif "Malaria" in cond: LEGACY_DISEASE_COLORS_WEB[cond] = LEGACY_DISEASE_COLORS_WEB.get("Malaria", "#A9A9A9")
        elif "HIV" in cond: LEGACY_DISEASE_COLORS_WEB[cond] = LEGACY_DISEASE_COLORS_WEB.get("HIV-Positive", "#A9A9A9")
        elif "Pneumonia" in cond: LEGACY_DISEASE_COLORS_WEB[cond] = LEGACY_DISEASE_COLORS_WEB.get("Pneumonia", "#A9A9A9")
        elif "Dehydration" in cond: LEGACY_DISEASE_COLORS_WEB[cond] = LEGACY_DISEASE_COLORS_WEB.get("Severe Dehydration", "#4682B4")
        elif "Heat Stroke" in cond: LEGACY_DISEASE_COLORS_WEB[cond] = LEGACY_DISEASE_COLORS_WEB.get("Heat Stroke", "#FF6347")
        elif "Sepsis" in cond: LEGACY_DISEASE_COLORS_WEB[cond] = LEGACY_DISEASE_COLORS_WEB.get("Sepsis", "#800080")
        elif "Diarrheal" in cond: LEGACY_DISEASE_COLORS_WEB[cond] = LEGACY_DISEASE_COLORS_WEB.get("Diarrheal Diseases (Severe)", "#D2691E")
        else: LEGACY_DISEASE_COLORS_WEB[cond] = "#A9A9A9" # Default grey for other action conditions

# --- End of Configuration ---
