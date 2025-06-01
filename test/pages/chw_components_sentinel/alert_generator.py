# sentinel_project_root/test/pages/chw_components_sentinel/alert_generator.py
# Processes CHW daily data to generate structured patient alert information for Sentinel.

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional

# Standardized import block
try:
    from config import app_config
except ImportError:
    import sys
    import os
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_for_utils = os.path.abspath(os.path.join(current_script_dir, os.pardir, os.pardir))
    if project_root_for_utils not in sys.path:
        sys.path.insert(0, project_root_for_utils)
    from config import app_config

logger = logging.getLogger(__name__)

def generate_chw_patient_alerts_from_data(
    patient_encounter_data_df: Optional[pd.DataFrame],
    # chw_daily_context_df is not strictly used here if patient_encounter_data_df is comprehensive for alerts
    for_date: Any, # datetime.date or similar, for logging/context
    chw_zone_context_str: str,
    max_alerts_to_return: int = 15
) -> List[Dict[str, Any]]:
    """
    Processes CHW daily data to generate a list of structured patient alerts.

    Args:
        patient_encounter_data_df: DataFrame with patient encounter data.
        for_date: The date for which alerts are relevant.
        chw_zone_context_str: Zone context for these alerts.
        max_alerts_to_return: Max number of top alerts for summaries.

    Returns:
        List of dictionaries, each representing an actionable alert.
    """
    module_log_prefix = "CHWAlertGenerator"
    logger.info(f"({module_log_prefix}) Generating CHW patient alerts for date: {str(for_date)}, zone: {chw_zone_context_str}")
    
    if not isinstance(patient_encounter_data_df, pd.DataFrame) or patient_encounter_data_df.empty:
        logger.info(f"({module_log_prefix}) No patient encounter data provided. No alerts generated.")
        return []

    processed_alerts_list: List[Dict[str, Any]] = []
    df_alert_source = patient_encounter_data_df.copy()

    # Define expected columns and their safe defaults or types for alert generation
    alert_trigger_cols_config = {
        'patient_id': {"default": "UnknownPID", "type": str},
        'encounter_date': {"default": pd.NaT, "type": "datetime"},
        'zone_id': {"default": chw_zone_context_str, "type": str},
        'condition': {"default": "N/A", "type": str},
        'age': {"default": np.nan, "type": float}, # Will be float after pd.to_numeric
        'ai_risk_score': {"default": np.nan, "type": float},
        'ai_followup_priority_score': {"default": np.nan, "type": float},
        'min_spo2_pct': {"default": np.nan, "type": float},
        'vital_signs_temperature_celsius': {"default": np.nan, "type": float},
        'max_skin_temp_celsius': {"default": np.nan, "type": float},
        'fall_detected_today': {"default": 0, "type": int}, # Expect 0 or 1
        'referral_status': {"default": "Unknown", "type": str},
        'referral_reason': {"default": "Unknown", "type": str},
        'medication_adherence_self_report': {"default": "Unknown", "type": str}
    }

    for col_name, config in alert_trigger_cols_config.items():
        if col_name not in df_alert_source.columns:
            df_alert_source[col_name] = config["default"]
            logger.debug(f"({module_log_prefix}) Added missing column '{col_name}' with default '{config['default']}'.")
        
        # Type coercion and filling NaNs for existing columns
        if config["type"] == "datetime":
            df_alert_source[col_name] = pd.to_datetime(df_alert_source[col_name], errors='coerce')
        elif config["type"] == float:
            df_alert_source[col_name] = pd.to_numeric(df_alert_source[col_name], errors='coerce').fillna(config["default"])
        elif config["type"] == int:
            df_alert_source[col_name] = pd.to_numeric(df_alert_source[col_name], errors='coerce').fillna(config["default"]).astype(int)
        elif config["type"] == str:
            df_alert_source[col_name] = df_alert_source[col_name].astype(str).fillna(config["default"])
            # Standardize common NA strings for string columns
            common_na_strings = ['', 'nan', 'None', 'N/A', '#N/A', 'np.nan', 'NaT', '<NA>', 'null']
            df_alert_source[col_name] = df_alert_source[col_name].str.strip().replace(common_na_strings, config["default"], regex=False)


    # Determine the best temperature column to use
    temp_col_for_alerts = None
    if 'vital_signs_temperature_celsius' in df_alert_source.columns and df_alert_source['vital_signs_temperature_celsius'].notna().any():
        temp_col_for_alerts = 'vital_signs_temperature_celsius'
    elif 'max_skin_temp_celsius' in df_alert_source.columns and df_alert_source['max_skin_temp_celsius'].notna().any():
        temp_col_for_alerts = 'max_skin_temp_celsius'
    else:
        logger.debug(f"({module_log_prefix}) No primary temperature column found for fever alerts.")


    # --- Alerting Rules ---
    for _, record in df_alert_source.iterrows():
        current_record_alerts_buffer: List[Dict[str, Any]] = []
        patient_id = str(record.get('patient_id', "UnknownPID"))
        condition_str = str(record.get('condition', "N/A"))
        zone_str = str(record.get('zone_id', chw_zone_context_str))
        
        enc_date = record.get('encounter_date')
        enc_date_str = enc_date.strftime('%Y-%m-%d') if pd.notna(enc_date) else 'N/A'
        context_info = f"Cond: {condition_str} | Zone: {zone_str} | Date: {enc_date_str}"

        # Rule 1: SpO2
        spo2 = record.get('min_spo2_pct')
        if pd.notna(spo2):
            if spo2 < app_config.ALERT_SPO2_CRITICAL_LOW_PCT:
                current_record_alerts_buffer.append({
                    "alert_level": "CRITICAL", "primary_reason": "Critical Low SpO2",
                    "brief_details": f"SpO2: {spo2:.0f}%", "suggested_action_code": "ACTION_SPO2_MANAGE_URGENT",
                    "raw_priority_score": 98 + max(0, app_config.ALERT_SPO2_CRITICAL_LOW_PCT - spo2), # Higher score for lower SpO2
                    "patient_id": patient_id, "context_info": context_info, "triggering_value": f"SpO2 {spo2:.0f}%", "encounter_date": enc_date_str
                })
            elif spo2 < app_config.ALERT_SPO2_WARNING_LOW_PCT:
                 current_record_alerts_buffer.append({
                    "alert_level": "WARNING", "primary_reason": "Low SpO2",
                    "brief_details": f"SpO2: {spo2:.0f}%", "suggested_action_code": "ACTION_SPO2_RECHECK_MONITOR",
                    "raw_priority_score": 75 + max(0, app_config.ALERT_SPO2_WARNING_LOW_PCT - spo2),
                    "patient_id": patient_id, "context_info": context_info, "triggering_value": f"SpO2 {spo2:.0f}%", "encounter_date": enc_date_str
                })

        # Rule 2: Fever
        if temp_col_for_alerts and pd.notna(record.get(temp_col_for_alerts)):
            temp_val = record.get(temp_col_for_alerts)
            if temp_val >= app_config.ALERT_BODY_TEMP_HIGH_FEVER_C:
                current_record_alerts_buffer.append({
                    "alert_level": "CRITICAL", "primary_reason": "High Fever",
                    "brief_details": f"Temp: {temp_val:.1f}°C", "suggested_action_code": "ACTION_FEVER_MANAGE_URGENT",
                    "raw_priority_score": 95 + max(0, (temp_val - app_config.ALERT_BODY_TEMP_HIGH_FEVER_C)*2),
                    "patient_id": patient_id, "context_info": context_info, "triggering_value": f"Temp {temp_val:.1f}°C", "encounter_date": enc_date_str
                })
            elif temp_val >= app_config.ALERT_BODY_TEMP_FEVER_C:
                current_record_alerts_buffer.append({
                    "alert_level": "WARNING", "primary_reason": "Fever Present",
                    "brief_details": f"Temp: {temp_val:.1f}°C", "suggested_action_code": "ACTION_FEVER_MONITOR_SUPPORT",
                    "raw_priority_score": 70 + max(0, temp_val - app_config.ALERT_BODY_TEMP_FEVER_C),
                    "patient_id": patient_id, "context_info": context_info, "triggering_value": f"Temp {temp_val:.1f}°C", "encounter_date": enc_date_str
                })

        # Rule 3: Fall Detected
        if pd.notna(record.get('fall_detected_today')) and int(record.get('fall_detected_today', 0)) > 0:
            current_record_alerts_buffer.append({
                "alert_level": "CRITICAL", "primary_reason": "Fall Detected",
                "brief_details": f"Falls: {int(record.get('fall_detected_today',0))}", "suggested_action_code": "ACTION_FALL_ASSESS_URGENT",
                "raw_priority_score": 92,
                "patient_id": patient_id, "context_info": context_info, "triggering_value": "Fall(s) > 0", "encounter_date": enc_date_str
            })

        # Rule 4: High AI Follow-up Priority Score (Generic high priority)
        ai_followup_score_val = record.get('ai_followup_priority_score')
        if pd.notna(ai_followup_score_val) and ai_followup_score_val >= app_config.FATIGUE_INDEX_HIGH_THRESHOLD:
             current_record_alerts_buffer.append({
                "alert_level": "WARNING", "primary_reason": "High AI Follow-up Priority",
                "brief_details": f"AI Prio. Score: {ai_followup_score_val:.0f}", "suggested_action_code": "ACTION_REVIEW_CASE_AI_PRIO",
                "raw_priority_score": min(90, ai_followup_score_val), # Cap to ensure vitals alerts are higher
                "patient_id": patient_id, "context_info": context_info, "triggering_value": f"AI Prio {ai_followup_score_val:.0f}", "encounter_date": enc_date_str
            })
        
        # Rule 5: High AI Risk Score (INFO level, if no stronger alert)
        ai_risk_score_val = record.get('ai_risk_score')
        if pd.notna(ai_risk_score_val) and ai_risk_score_val >= app_config.RISK_SCORE_HIGH_THRESHOLD:
            # Add as INFO only if no CRITICAL or WARNING alert already exists for this record
            if not any(a['alert_level'] in ["CRITICAL", "WARNING"] for a in current_record_alerts_buffer):
                 current_record_alerts_buffer.append({
                    "alert_level": "INFO", "primary_reason": "Elevated AI Risk Score",
                    "brief_details": f"AI Risk Score: {ai_risk_score_val:.0f}", "suggested_action_code": "ACTION_MONITOR_RISK_PROFILE",
                    "raw_priority_score": min(70, ai_risk_score_val), # Lower than warnings
                    "patient_id": patient_id, "context_info": context_info, "triggering_value": f"AI Risk {ai_risk_score_val:.0f}", "encounter_date": enc_date_str
                })
        
        # Rule 6: Pending Critical Condition Referral
        if str(record.get('referral_status', '')).lower() == 'pending':
            is_critical_referral = any(
                key_cond.lower() in str(record.get('condition','N/A')).lower() for key_cond in app_config.KEY_CONDITIONS_FOR_ACTION
            )
            if is_critical_referral:
                current_record_alerts_buffer.append({
                    "alert_level": "WARNING", "primary_reason": "Pending Critical Referral",
                    "brief_details": f"For: {str(record.get('condition','N/A'))}", "suggested_action_code": "ACTION_FOLLOWUP_REFERRAL_STATUS",
                    "raw_priority_score": 80,
                    "patient_id": patient_id, "context_info": context_info, "triggering_value": "Pending Critical Referral", "encounter_date": enc_date_str
                })

        if current_record_alerts_buffer:
            # Select the single highest priority alert for this specific encounter/record
            top_alert_for_record = max(current_record_alerts_buffer, key=lambda x: x['raw_priority_score'])
            processed_alerts_list.append(top_alert_for_record)

    # De-duplicate alerts: Keep only the single highest priority alert per patient for the day
    if processed_alerts_list:
        final_alerts_df = pd.DataFrame(processed_alerts_list)
        
        # Ensure encounter_date is a date object for daily de-duplication
        # The 'encounter_date' in processed_alerts_list is already a string 'YYYY-MM-DD' or 'N/A'
        # For de-duplication, 'N/A' dates should be treated as unique or handled appropriately.
        # If 'N/A', perhaps use a unique identifier for that day's batch. For now, treat 'N/A' as distinct.
        
        # Sort by priority (descending) then by alert level (Critical > Warning > Info)
        # Map alert levels to sortable integers
        alert_level_sort_map = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}
        final_alerts_df['alert_level_sortable'] = final_alerts_df['alert_level'].map(alert_level_sort_map).fillna(3)
        
        final_alerts_df.sort_values(
            by=["raw_priority_score", "alert_level_sortable"], 
            ascending=[False, True], # Highest score, then most critical level
            inplace=True
        )
        
        # Keep the first (highest priority) alert per patient per 'encounter_date' string
        final_alerts_df.drop_duplicates(subset=["patient_id", "encounter_date"], keep="first", inplace=True)
        final_alerts_df.drop(columns=['alert_level_sortable'], inplace=True) # Clean up sort helper column
        
        logger.info(f"({module_log_prefix}) Generated {len(final_alerts_df)} unique patient alerts after de-duplication.")
        return final_alerts_df.head(max_alerts_to_return).to_dict(orient='records')
        
    logger.info(f"({module_log_prefix}) No alerts generated from the provided data.")
    return []
