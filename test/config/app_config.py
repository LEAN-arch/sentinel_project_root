# sentinel_project_root/test/config/app_config.py
# Configuration for "Sentinel Health Co-Pilot" - LMIC Edge-First System

import os
import logging
from datetime import datetime

# --- Configure Logging ---
LOG_LEVEL = os.getenv("SENTINEL_LOG_LEVEL", "INFO").upper()
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - [%(module)s.%(funcName)s:%(lineno)d] - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)

# --- Path Validation ---
def validate_path(path, description):
    """Validate file or directory path, log warning if missing."""
    if not os.path.exists(path):
        logger.warning(f"{description} not found: {path}")
    return path

# --- I. Core System & Directory Configuration ---
BASE_APP_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ASSETS_DIR = validate_path(os.path.join(BASE_APP_ROOT_DIR, "assets"), "Assets directory")
DATA_SOURCES_DIR = validate_path(os.path.join(BASE_APP_ROOT_DIR, "data_sources"), "Data sources directory")
FACILITY_NODE_DATA_DIR = os.path.join(BASE_APP_ROOT_DIR, "facility_node_data")
LOCAL_DATA_DIR_PED_SIM = os.path.join(BASE_APP_ROOT_DIR, "local_data_ped_sim")

# Data source paths
HEALTH_RECORDS_CSV = os.getenv("HEALTH_RECORDS_CSV", validate_path(os.path.join(DATA_SOURCES_DIR, "health_records_expanded.csv"), "Health records CSV"))
ZONE_ATTRIBUTES_CSV = os.getenv("ZONE_ATTRIBUTES_CSV", validate_path(os.path.join(DATA_SOURCES_DIR, "zone_attributes.csv"), "Zone attributes CSV"))
ZONE_GEOMETRIES_GEOJSON = os.getenv("ZONE_GEOMETRIES_GEOJSON", validate_path(os.path.join(DATA_SOURCES_DIR, "zone_geometries.geojson"), "Zone geometries GeoJSON"))
IOT_CLINIC_ENVIRONMENT_CSV = os.getenv("IOT_CLINIC_ENVIRONMENT_CSV", validate_path(os.path.join(DATA_SOURCES_DIR, "iot_clinic_environment.csv"), "IoT clinic environment CSV"))

APP_NAME = "Sentinel Health Co-Pilot"
APP_VERSION = "3.1.0"
APP_LOGO_SMALL = validate_path(os.path.join(ASSETS_DIR, "sentinel_logo_small.png"), "Small logo")
APP_LOGO_LARGE = validate_path(os.path.join(ASSETS_DIR, "sentinel_logo_large.png"), "Large logo")
STYLE_CSS_PATH = validate_path(os.path.join(ASSETS_DIR, "style_web_reports.css"), "CSS stylesheet")

ORGANIZATION_NAME = "LMIC Health Futures Initiative"
APP_FOOTER_TEXT = f"© {datetime.now().year} {ORGANIZATION_NAME}. Actionable Intelligence for Resilient Health Systems."
SUPPORT_CONTACT_INFO = "support@lmic-health-futures.org"

# --- II. Health & Operational Thresholds ---
ALERT_SPO2_CRITICAL = 90
ALERT_SPO2_WARNING = 94
ALERT_BODY_TEMP_FEV = 38.0
ALERT_BODY_TEMP_HIGH_FEV = 39.5
ALERT_HR_TACHYCARDIA = 100
ALERT_HR_BRADYCARDIA = 50
HEAT_STRESS_BODY_TEMP = 37.5
HEAT_STRESS_RISK_TEMP = 38.5

ALERT_AMBIENT_CO2_HIGH = 1500
ALERT_AMBIENT_CO2_VERY_HIGH = 2500
ALERT_AMBIENT_PM25_HIGH = 35
ALERT_AMBIENT_PM25_VERY_HIGH = 50
ALERT_AMBIENT_NOISE_HIGH = 85
ALERT_AMBIENT_HEAT_INDEX_RISK = 32
ALERT_AMBIENT_HEAT_INDEX_DANGER = 41

FATIGUE_INDEX_MODERATE = 60
FATIGUE_INDEX_HIGH = 80
STRESS_HRV_LOW = 20

CLINIC_WAITING_ROOM_MAX = 10
CLINIC_THROUGHPUT_MIN_HOUR = 5

RISK_SCORE_LOW = 40
RISK_SCORE_MODERATE = 60
RISK_SCORE_HIGH = 75

DISTRICT_ZONE_HIGH_RISK_SCORE = 70
DISTRICT_FACILITY_COVERAGE_LOW = 60
DISTRICT_TB_BURDEN_HIGH = 10
DISTRICT_DISEASE_PREVALENCE_PERCENTILE = 0.80

CRITICAL_SUPPLY_DAYS = 7
LOW_SUPPLY_DAYS = 14
TARGET_DAILY_STEPS = 8000

# --- III. Edge Device Configuration ---
EDGE_DEFAULT_LANGUAGE = "en"
EDGE_SUPPORTED_LANGUAGES = ["en", "sw", "fr"]
EDGE_PICTOGRAM_CONFIG = validate_path(os.path.join(ASSETS_DIR, "pictogram_map.json"), "Pictogram config")
EDGE_HAPTIC_CONFIG = validate_path(os.path.join(ASSETS_DIR, "haptic_patterns.json"), "Haptic config")
AUDIO_ALERTS_DIR = validate_path(os.path.join(ASSETS_DIR, "audio_alerts"), "Audio alerts directory")

EDGE_MODEL_VITALS = "vitals_deterioration_v1.tflite"
EDGE_MODEL_FATIGUE = "fatigue_index_v1.tflite"
EDGE_MODEL_ANOMALY = "anomaly_detection_base.tflite"
BASELINE_WINDOW_DAYS = 7
EDGE_PROCESSING_INTERVAL = 60

PED_SQLITE_DB = "sentinel_ped.db"
PED_MAX_LOG_SIZE_MB = 50
EDGE_SYNC_PROTOCOLS = ["BLUETOOTH_PEER", "WIFI_DIRECT_HUB", "QR_PACKET_SHARE", "SD_CARD_TRANSFER"]
QR_PACKET_MAX_SIZE = 256
SMS_COMPRESSION = "BASE85_ZLIB"

# --- IV. Supervisor Hub & Facility Configuration ---
HUB_SQLITE_DB = "sentinel_hub.db"
FACILITY_DB_TYPE = "POSTGRESQL"
FHIR_ENDPOINT = "http://localhost:8080/fhir"
REPORT_INTERVAL_HOURS = 24
CACHE_TTL_SECONDS = 3600

ESCALATION_PROTOCOLS = validate_path(os.path.join(ASSETS_DIR, "escalation_protocols.json"), "Escalation protocols")

# --- V. Data Semantics ---
TEST_TYPES = {
    "Sputum-AFB": {"disease_group": "TB", "tat_days": 2, "critical": True, "display_name": "TB Sputum (AFB)"},
    "Sputum-GeneXpert": {"disease_group": "TB", "tat_days": 1, "critical": True, "display_name": "TB GeneXpert"},
    "RDT-Malaria": {"disease_group": "Malaria", "tat_days": 0.5, "critical": True, "display_name": "Malaria RDT"},
    "HIV-Rapid": {"disease_group": "HIV", "tat_days": 0.25, "critical": True, "display_name": "HIV Rapid Test"},
    "HIV-ViralLoad": {"disease_group": "HIV", "tat_days": 7, "critical": True, "display_name": "HIV Viral Load"},
    "BP Check": {"disease_group": "Hypertension", "tat_days": 0, "critical": False, "display_name": "BP Check"},
}
CRITICAL_TESTS = [k for k, v in TEST_TYPES.items() if v.get("critical")]

TEST_TURNAROUND_DAYS = 2
TESTS_MEETING_TAT_PCT = 85
SAMPLE_REJECTION_RATE_PCT = 5
OVERDUE_TEST_DAYS = 7

KEY_CONDITIONS = ['TB', 'Malaria', 'HIV-Positive', 'Pneumonia', 'Severe Dehydration', 'Heat Stroke', 'Sepsis', 'Diarrheal Diseases (Severe)']
KEY_DRUGS = ['TB-Regimen', 'ACT', 'ARV-Regimen', 'ORS', 'Amoxicillin', 'Paracetamol', 'Penicillin', 'Iron-Folate', 'Insulin']

MALARIA_POSITIVITY_RATE = 5.0

# --- VI. Web Dashboard Configuration ---
MAP_CENTER_LAT = 0.0
MAP_CENTER_LON = 0.0
MAP_DEFAULT_ZOOM = 2

WEB_DASH_DEFAULT_VIEW_DAYS = 7
WEB_DASH_DEFAULT_TREND_DAYS = 30
WEB_PLOT_HEIGHT = 400
WEB_PLOT_COMPACT_HEIGHT = 320
WEB_MAP_HEIGHT = 600

MAPBOX_STYLE = "carto-positron"
DEFAULT_CRS = "EPSG:4326"

# --- VII. Color Palette ---
COLOR_RISK_HIGH = "#D32F2F"
COLOR_RISK_MODERATE = "#FBC02D"
COLOR_RISK_LOW = "#388E3C"
COLOR_RISK_NEUTRAL = "#757575"
COLOR_ACTION_PRIMARY = "#1976D2"
COLOR_ACTION_SECONDARY = "#546E7A"
COLOR_ACCENT_BRIGHT = "#4D7BF3"
COLOR_POSITIVE_DELTA = "#27AE60"
COLOR_NEGATIVE_DELTA = "#C0392B"
COLOR_TEXT_DARK = "#343a40"
COLOR_TEXT_HEADINGS_MAIN = "#1A2557"
COLOR_TEXT_HEADINGS_SUB = "#2C3E50"
COLOR_TEXT_MUTED = "#6c757d"
COLOR_TEXT_LINK = COLOR_ACTION_PRIMARY
COLOR_BG_PAGE = "#f8f9fa"
COLOR_BG_CONTENT = "#ffffff"
COLOR_BG_SUBTLE = "#e9ecef"
COLOR_BORDER_LIGHT = "#dee2e6"
COLOR_BORDER_MEDIUM = "#ced4da"

DISEASE_COLORS = {
    "TB": "#EF4444",
    "Malaria": "#F59E0B",
    "HIV-Positive": "#8B5CF6",
    "Pneumonia": "#3B82F6",
    "Anemia": "#10B981",
    "STI": "#EC4899",
    "Dengue": "#6366F1",
    "Hypertension": "#F97316",
    "Diabetes": "#0EA5E9",
    "Wellness Visit": "#84CC16",
    "Other": "#6B7280",
    "STI-Syphilis": "#c026d3",
    "STI-Gonorrhea": "#db2777",
    "Heat Stroke": "#FF6347",
    "Severe Dehydration": "#4682B4",
    "Sepsis": "#800080",
    "Diarrheal Diseases (Severe)": "#D2691E"
}

# --- End of Configuration ---
