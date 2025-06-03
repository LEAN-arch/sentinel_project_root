# sentinel_project_root/test/pages/chw_components_sentinel/alert_generator.py
# Processes CHW daily data to generate structured patient alert information for Sentinel.

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional
from datetime import date

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Standardized import block
try:
    from config import app_config
except ImportError as e:
    logger.error(f"Import error: {e}. Ensure config.py is in test/ with __init__.py.")
    raise

def generate_chw_patient_alerts_from_data(
    patient_encounter_data_df: Optional[pd.DataFrame],
    for_date: Any,
    chw_zone_context_str: str,
    max_alerts_to_return: int = 15
) -> List[Dict[str, Any]]:
    """
    Processes CHW daily data to generate a list of structured patient alerts.

    Args:
        patient_encounter_data_df: DataFrame with patient encounter data.
        for_date: The date for which alerts are relevant (used in alert context).
        chw_zone_context_str: Zone context for these alerts.
        max_alerts_to_return: Max number of top alerts for summaries.

    Returns:
        List of dictionaries, each representing an actionable alert, sorted by priority.

    Raises:
        ValueError: If input data is invalid or missing.
    """
    module_log_prefix = "CHWAlertGenerator"

    # Validate app_config attributes
    required_configs = [
        'ALERT_SPO2_CRITICAL_LOW_PCT', 'ALERT_SPO2_WARNING_LOW_PCT',
        'ALERT_BODY_TEMP_HIGH_FEVER_C', 'ALERT_BODY_TEMP_FEVER_C',
        'FATIGUE_INDEX_WARNING_THRESHOLD', 'RISK_SCORE_HIGH_THRESHOLD',
        'KEY_CONDITIONS_FOR_ACTION'
    ]
    for attr in required_configs:
        if not hasattr(app_config, attr):
            logger.error(f"({module_log_prefix}) Missing config: {attr}")
            raise ValueError(f"Missing required configuration: {attr}")

    # Standardize for_date
    try:
        if isinstance(for_date, date):
            for_date_str = for_date.strftime('%Y-%m-%d')
        else:
            for_date_str = pd.to_datetime(for_date, errors='coerce').strftime('%Y-%m-%d')
        if for_date_str == 'NaT':
            raise ValueError
    except Exception as e:
        logger.error(f"({module_log_prefix}) Invalid for_date: {e}")
        raise ValueError("Invalid date format for alert processing")

    logger.info(f"({module_log_prefix}) Generating CHW patient alerts for date: {for_date_str}, zone: {chw_zone_context_str}")

    if not isinstance(patient_encounter_data_df, pd.DataFrame) or patient_encounter_data_df.empty:
        logger.warning(f"({module_log_prefix}) No patient encounter data provided.")
        raise ValueError("No patient encounter data available for alert generation.")

    # Define expected columns and their configurations
    alert_cols_config = {
        'patient_id': {"default": "UnknownPID_Alert", "type": str},
        'encounter_date': {"default": pd.NaT, "type": "datetime"},
        'zone_id': {"default": chw_zone_context_str, "type": str},
        'condition': {"default": "N/A_Alert", "type": str},
        'age': {"default": np.nan, "type": "float"},
        'ai_risk_score': {"default": np.nan, "type": "float"},
        'ai_followup_priority_score': {"default": np.nan, "type": "float"},
        'min_spo2_pct': {"default": np.nan, "type": "float"},
        'vital_signs_temperature_celsius': {"default": np.nan, "type": "float"},
        'max_skin_temp_celsius': {"default": np.nan, "type": "float"},
        'fall_detected_today': {"default": 0, "type": "int"},
        'referral_status': {"default": "Unknown_Alert", "type": str},
        'referral_reason': {"default": "Unknown_Alert", "type": str}
    }
    common_na_strings = ['', 'nan', 'None', 'N/A', '#N/A', 'np.nan', 'NaT', '<NA>', 'null']

    # Prepare DataFrame
    df_alert_src = patient_encounter_data_df.copy()
    for col_name, config in alert_cols_config.items():
        if col_name not in df_alert_src.columns:
            df_alert_src[col_name] = config["default"]
            logger.debug(f"({module_log_prefix}) Added missing column '{col_name}' with default {config['default']}.")
        
        if config["type"] == "datetime":
            df_alert_src[col_name] = pd.to_datetime(df_alert_src[col_name], errors='coerce')
        elif config["type"] == "float":
            df_alert_src[col_name] = pd.to_numeric(df_alert_src[col_name], errors='coerce').fillna(config["default"])
        elif config["type"] == "int":
            df_alert_src[col_name] = pd.to_numeric(df_alert_src[col_name], errors='coerce').fillna(config["default"]).astype(int)
        elif config["type"] == str:
            df_alert_src[col_name] = df_alert_src[col_name].astype(str).replace(common_na_strings, config["default"])

    # Select temperature column
    temp_col_name = 'vital_signs_temperature_celsius' if 'vital_signs_temperature_celsius' in df_alert_src.columns else \
                    'max_skin_temp_celsius' if 'max_skin_temp_celsius' in df_alert_src.columns else None

    # Generate alerts (vectorized where possible)
    alerts_buffer: List[Dict[str, Any]] = []

    # Rule 1: SpO2
    if 'min_spo2_pct' in df_alert_src.columns:
        spo2_critical = df_alert_src['min_spo2_pct'] < app_config.ALERT_SPO2_CRITICAL_LOW_PCT
        alerts_buffer.extend([
            {
                "alert_level": "CRITICAL", "primary_reason": "Critical Low SpO2",
                "brief_details": f"SpO2: {row['min_spo2_pct']:.0f}%", "suggested_action_code": "ACTION_SPO2_MANAGE_URGENT",
                "raw_priority_score": 98 + max(0, app_config.ALERT_SPO2_CRITICAL_LOW_PCT - row['min_spo2_pct']),
                "patient_id": str(row['patient_id']), "context_info": f"Cond: {row['condition']} | Zone: {row['zone_id']} | Date: {row['encounter_date'].strftime('%Y-%m-%d') if pd.notna(row['encounter_date']) else for_date_str}",
                "triggering_value": f"SpO2 {row['min_spo2_pct']:.0f}%", "encounter_date": row['encounter_date'].strftime('%Y-%m-%d') if pd.notna(row['encounter_date']) else for_date_str
            } for _, row in df_alert_src[spo2_critical].iterrows()
        ])

        spo2_warning = (df_alert_src['min_spo2_pct'] < app_config.ALERT_SPO2_WARNING_LOW_PCT) & (~spo2_critical)
        alerts_buffer.extend([
            {
                "alert_level": "WARNING", "primary_reason": "Low SpO2",
                "brief_details": f"SpO2: {row['min_spo2_pct']:.0f}%", "suggested_action_code": "ACTION_SPO2_RECHECK_MONITOR",
                "raw_priority_score": 75 + max(0, app_config.ALERT_SPO2_WARNING_LOW_PCT - row['min_spo2_pct']),
                "patient_id": str(row['patient_id']), "context_info": f"Cond: {row['condition']} | Zone: {row['zone_id']} | Date: {row['encounter_date'].strftime('%Y-%m-%d') if pd.notna(row['encounter_date']) else for_date_str}",
                "triggering_value": f"SpO2 {row['min_spo2_pct']:.0f}%", "encounter_date": row['encounter_date'].strftime('%Y-%m-%d') if pd.notna(row['encounter_date']) else for_date_str
            } for _, row in df_alert_src[spo2_warning].iterrows()
        ])

    # Rule 2: Fever
    if temp_col_name:
        fever_critical = df_alert_src[temp_col_name] >= app_config.ALERT_BODY_TEMP_HIGH_FEVER_C
        alerts_buffer.extend([
            {
                "alert_level": "CRITICAL", "primary_reason": "High Fever",
                "brief_details": f"Temp: {row[temp_col_name]:.1f}°C", "suggested_action_code": "ACTION_FEVER_MANAGE_URGENT",
                "raw_priority_score": 95 + max(0, (row[temp_col_name] - app_config.ALERT_BODY_TEMP_HIGH_FEVER_C) * 2),
                "patient_id": str(row['patient_id']), "context_info": f"Cond: {row['condition']} | Zone: {row['zone_id']} | Date: {row['encounter_date'].strftime('%Y-%m-%d') if pd.notna(row['encounter_date']) else for_date_str}",
                "triggering_value": f"Temp {row[temp_col_name]:.1f}°C", "encounter_date": row['encounter_date'].strftime('%Y-%m-%d') if pd.notna(row['encounter_date']) else for_date_str
            } for _, row in df_alert_src[fever_critical].iterrows()
        ])

        fever_warning = (df_alert_src[temp_col_name] >= app_config.ALERT_BODY_TEMP_FEVER_C) & (~fever_critical)
        alerts_buffer.extend([
            {
                "alert_level": "WARNING", "primary_reason": "Fever Present",
                "brief_details": f"Temp: {row[temp_col_name]:.1f}°C",
                "suggested_action_code": "ACTION_FEVER_MONITOR",
                "raw_priority_score": 70 + max(0, row[temp_col_name] - app_config.ALERT_BODY_TEMP_FEVER_C),
                "patient_id": str(row['patient_id']),
                "context_info": f"Cond: {row['condition']} | Zone: {row['zone_id']} | Date: {row['encounter_date'].strftime('%Y-%m-%d') if pd.notna(row['encounter_date']) else for_date_str}",
                "triggering_value": f"Temp {row['temp_col_name']: row[temp_col_name]:.1f}°C",
                "encounter_date": row['encounter_date'].strftime('%Y-%m-%d') if pd.notna(row['encounter_date']) else for_date_str
            } for _, row in df_alert_src[fever_warning].iterrows()
        ])

    # Rule 3: Fall Detected
    if 'fall_detected' in df_alert_src.columns:
        fall_detected = df_alert_src['fall_detected'] > 0
        alerts_buffer.extend([
            {
                "alert_level": "CRITICAL",
                "primary_reason": "Fall Detected",
                "brief_details": f"Falls: {int(row['fall_detected'])}",
                "suggested_action_code": "ACTION_FALL_ALERT",
                "raw_priority_score": 92,
                "patient_id": str(row['patient_id']),
                "context_info": f"Cond: {row['condition']} | Zone: {row['zone_id']} | Date: {row['encounter_date'].strftime('%Y-%m-%d') if pd.notna(row['encounter_date']) else for_date_str}",
                "triggering_value": "Fall(s) > 0",
                "encounter_date": row['encounter_date'].strftime('%Y-%m-%d') if pd.notna(row['encounter_date']) else for_date_str
            } for _, row in df_alert_src[fall_detected].iterrows()
        ])

    # Rule 4: High AI Follow-up Priority
    if 'ai_followup_priority_score' in df_alert_src.columns:
        high_priority = df_alert_src['ai_followup_priority_score'] >= app_config.FATIGUE_THRESHOLD
        alerts_buffer.extend([
            {
                "alert_level": "HIGH",
                "primary_reason": "High AI Follow-up Score",
                "brief_details": f"AI Prio Score: {row['ai_followup_priority_score']:.0f}",
                "suggested_action_code": "ACTION_AI_REVIEW",
                "raw_priority_score": min(90, row['ai_followup_priority_score']),
                "patient_id": str(row['patient_id']),
                "context_info": f"Cond: {row['condition']} | Zone: {row['zone_id']} | Date: {row['encounter_date'].strftime('%Y-%m-%d') if pd.notna(row['encounter_date']) else for_date_str}",
                "triggering_value": f"AI Prio {row['ai_followup_priority_score']:.0f}",
                "encounter_date": row['encounter_date'].strftime('%Y-%m-%d') if pd.notna(row['encounter_date']) else for_date_str
            } for _, row in df_alert_src[high_priority].iterrows()
        ])

    # Rule 5: High AI Risk Score (INFO level, only if no CRITICAL/WARNING)
    if 'ai_risk_score' in df_alert_src.columns:
        high_risk = df_alert_src['ai_risk_score'] >= app_config.RISK_SCORE_HIGH_THRESHOLD
        for _, row in df_alert_src[high_risk].iterrows():
            patient_id = str(row['patient_id'])
            encounter_date = row['encounter_date'].strftime('%Y-%m-%d') if pd.notna(row['encounter_date']) else for_date_str
            existing_alert = any(a['patient_id'] == patient_id and a['encounter_date'] == encounter_date and a['alert_level'] in ["CRITICAL", "WARNING"] for a in alerts_buffer)
            if not existing_alert:
                alerts_buffer.append({
                    "alert_level": "INFO",
                    "primary_reason": "Elevated AI Risk Score",
                    "brief_details": f"AI Risk Score: {row['ai_risk_score']:.0f}",
                    "suggested_action_code": "ACTION_MONITOR_RISK",
                    "raw_priority_score": min(70, row['ai_risk_score']),
                    "patient_id": patient_id,
                    "context_info": f"Cond: {row['condition']} | Zone: {row['zone_id']} | Date: {encounter_date}",
                    "triggering_value": f"AI Risk {row['ai_risk_score']:.0f}",
                    "encounter_date": encounter_date
                })

    # Rule 6: Pending Critical Referral
    if 'referral_status' in df_alert_src.columns and 'condition' in df_alert_src.columns:
        pending_referral = df_alert_src['referral_status'].str.lower() == 'pending'
        for _, row in df_alert_src[pending_referral].iterrows():
            is_key_condition = any(key_c.lower() in str(row['condition']).lower() for key_c in app_config.KEY_CONDITIONS_FOR_ACTION)
            if is_key_condition:
                alerts_buffer.append({
                    "alert_level": "WARNING",
                    "primary_reason": "Pending Critical Referral",
                    "brief_details": f"For: {row['condition']}",
                    "suggested_action_code": "ACTION_FOLLOWUP_REFERRAL",
                    "raw_priority_score": 80,
                    "patient_id": str(row['patient_id']),
                    "context_info": f"Cond: {row['condition']} | Zone: {row['zone_id']} | Date: {row['encounter_date'].strftime('%Y-%m-%d') if pd.notna(row['encounter_date']) else for_date_str}",
                    "triggering_value": "Pending Critical Referral",
                    "encounter_date": row['encounter_date'].strftime('%Y-%m-%d') if pd.notna(row['encounter_date']) else for_date_str
                })

    # Deduplicate alerts
    if alerts_buffer:
        alerts_dict = {}
        for alert in alerts_buffer:
            key = (alert['patient_id'], alert['encounter_date'])
            if key not in alerts_dict or alert['raw_priority_score'] > alerts_dict[key]['raw_priority_score']:
                alerts_dict[key] = alert
        
        alerts_final = list(alerts_dict.values())
        alerts_final.sort(key=lambda x: x['raw_priority_score'], reverse=True)
        
        logger.info(f"({module_log_prefix}) Generated {len(alerts_final)} unique patient alerts after deduplication.")
        return alerts_final[:max_alerts_to_return]
    
    logger.info(f"({module_log_prefix}) No alerts generated from the provided data.")
    return []
