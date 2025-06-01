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
    # Assumes this file is in sentinel_project_root/test/pages/chw_components_sentinel/
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_test_root_dir = os.path.abspath(os.path.join(current_script_dir, os.pardir, os.pardir))
    if project_test_root_dir not in sys.path:
        sys.path.insert(0, project_test_root_dir)
    from config import app_config

logger = logging.getLogger(__name__)

def generate_chw_patient_alerts_from_data(
    patient_encounter_data_df: Optional[pd.DataFrame],
    # chw_daily_context_df: Optional[pd.DataFrame] = None, # Context df not actively used if patient_encounter_data_df is comprehensive
    for_date: Any, # datetime.date or similar, for logging/context
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
    """
    module_log_prefix = "CHWAlertGenerator" # Consistent prefix
    logger.info(f"({module_log_prefix}) Generating CHW patient alerts for date: {str(for_date)}, zone: {chw_zone_context_str}")
    
    if not isinstance(patient_encounter_data_df, pd.DataFrame) or patient_encounter_data_df.empty:
        logger.info(f"({module_log_prefix}) No patient encounter data provided. No alerts generated.")
        return []

    alerts_buffer: List[Dict[str, Any]] = [] # Renamed from processed_alerts_list
    df_alert_src = patient_encounter_data_df.copy() # Use a clearer variable name

    # Define expected columns and their safe defaults or types for alert generation
    alert_cols_config = { # Renamed from alert_trigger_cols_config
        'patient_id': {"default": "UnknownPID_Alert", "type": str},
        'encounter_date': {"default": pd.NaT, "type": "datetime"},
        'zone_id': {"default": chw_zone_context_str, "type": str}, # Use context if not in record
        'condition': {"default": "N/A_Alert", "type": str},
        'age': {"default": np.nan, "type": float},
        'ai_risk_score': {"default": np.nan, "type": float},
        'ai_followup_priority_score': {"default": np.nan, "type": float},
        'min_spo2_pct': {"default": np.nan, "type": float},
        'vital_signs_temperature_celsius': {"default": np.nan, "type": float},
        'max_skin_temp_celsius': {"default": np.nan, "type": float},
        'fall_detected_today': {"default": 0, "type": int}, # Expect 0 or 1 typically
        'referral_status': {"default": "Unknown_Alert", "type": str},
        'referral_reason': {"default": "Unknown_Alert", "type": str},
        'medication_adherence_self_report': {"default": "Unknown_Alert", "type": str}
    }
    common_na_strings_alert = ['', 'nan', 'None', 'N/A', '#N/A', 'np.nan', 'NaT', '<NA>', 'null']


    for col_name, config_val in alert_cols_config.items():
        if col_name not in df_alert_src.columns:
            df_alert_src[col_name] = config_val["default"]
            logger.debug(f"({module_log_prefix}) Added missing column '{col_name}' with default '{config_val['default']}'.")
        
        # Type coercion and standardized NaN/empty string handling for existing columns
        if config_val["type"] == "datetime":
            df_alert_src[col_name] = pd.to_datetime(df_alert_src[col_name], errors='coerce')
        elif config_val["type"] == float:
            df_alert_src[col_name] = pd.to_numeric(df_alert_src[col_name], errors='coerce').fillna(config_val["default"])
        elif config_val["type"] == int: # For flags like fall_detected_today
            df_alert_src[col_name] = pd.to_numeric(df_alert_src[col_name], errors='coerce').fillna(config_val["default"]).astype(int)
        elif config_val["type"] == str:
            df_alert_src[col_name] = df_alert_src[col_name].astype(str).fillna(config_val["default"])
            df_alert_src[col_name] = df_alert_src[col_name].str.strip().replace(common_na_strings_alert, config_val["default"], regex=False)


    # Determine the best temperature column to use from the prepared df_alert_src
    temp_col_name = None
    if 'vital_signs_temperature_celsius' in df_alert_src.columns and df_alert_src['vital_signs_temperature_celsius'].notna().any():
        temp_col_name = 'vital_signs_temperature_celsius'
    elif 'max_skin_temp_celsius' in df_alert_src.columns and df_alert_src['max_skin_temp_celsius'].notna().any():
        temp_col_name = 'max_skin_temp_celsius'
    else:
        logger.debug(f"({module_log_prefix}) No primary temperature column with data found for fever alerts.")

    # --- Alerting Rules Iteration ---
    for _, record_item in df_alert_src.iterrows():
        record_alerts_temp_list: List[Dict[str, Any]] = [] # Stores alerts for this specific record
        
        pat_id_alert_val = str(record_item.get('patient_id', alert_cols_config['patient_id']['default']))
        cond_str_alert_val = str(record_item.get('condition', alert_cols_config['condition']['default']))
        zone_str_alert_val = str(record_item.get('zone_id', chw_zone_context_str)) # Fallback to overall context
        
        encounter_dt_alert = record_item.get('encounter_date')
        # Use the string 'YYYY-MM-DD' for encounter_date in alerts for consistent de-duplication key part
        encounter_date_str_alert = encounter_dt_alert.strftime('%Y-%m-%d') if pd.notna(encounter_dt_alert) else str(for_date) # Fallback to processing date
        
        alert_context_info = f"Cond: {cond_str_alert_val} | Zone: {zone_str_alert_val} | Date: {encounter_date_str_alert}"

        # Rule 1: SpO2
        spo2_val_alert = record_item.get('min_spo2_pct')
        if pd.notna(spo2_val_alert): # Already float due to prep
            if spo2_val_alert < app_config.ALERT_SPO2_CRITICAL_LOW_PCT:
                record_alerts_temp_list.append({
                    "alert_level": "CRITICAL", "primary_reason": "Critical Low SpO2",
                    "brief_details": f"SpO2: {spo2_val_alert:.0f}%", "suggested_action_code": "ACTION_SPO2_MANAGE_URGENT",
                    "raw_priority_score": 98 + max(0, app_config.ALERT_SPO2_CRITICAL_LOW_PCT - spo2_val_alert),
                    "patient_id": pat_id_alert_val, "context_info": alert_context_info, "triggering_value": f"SpO2 {spo2_val_alert:.0f}%", "encounter_date": encounter_date_str_alert
                })
            elif spo2_val_alert < app_config.ALERT_SPO2_WARNING_LOW_PCT:
                 record_alerts_temp_list.append({
                    "alert_level": "WARNING", "primary_reason": "Low SpO2",
                    "brief_details": f"SpO2: {spo2_val_alert:.0f}%", "suggested_action_code": "ACTION_SPO2_RECHECK_MONITOR",
                    "raw_priority_score": 75 + max(0, app_config.ALERT_SPO2_WARNING_LOW_PCT - spo2_val_alert),
                    "patient_id": pat_id_alert_val, "context_info": alert_context_info, "triggering_value": f"SpO2 {spo2_val_alert:.0f}%", "encounter_date": encounter_date_str_alert
                })

        # Rule 2: Fever
        if temp_col_name and pd.notna(record_item.get(temp_col_name)):
            temp_val_alert = record_item.get(temp_col_name)
            if temp_val_alert >= app_config.ALERT_BODY_TEMP_HIGH_FEVER_C:
                record_alerts_temp_list.append({
                    "alert_level": "CRITICAL", "primary_reason": "High Fever",
                    "brief_details": f"Temp: {temp_val_alert:.1f}°C", "suggested_action_code": "ACTION_FEVER_MANAGE_URGENT",
                    "raw_priority_score": 95 + max(0, (temp_val_alert - app_config.ALERT_BODY_TEMP_HIGH_FEVER_C) * 2), # Higher deviation, higher score
                    "patient_id": pat_id_alert_val, "context_info": alert_context_info, "triggering_value": f"Temp {temp_val_alert:.1f}°C", "encounter_date": encounter_date_str_alert
                })
            elif temp_val_alert >= app_config.ALERT_BODY_TEMP_FEVER_C:
                record_alerts_temp_list.append({
                    "alert_level": "WARNING", "primary_reason": "Fever Present",
                    "brief_details": f"Temp: {temp_val_alert:.1f}°C", "suggested_action_code": "ACTION_FEVER_MONITOR_SUPPORT",
                    "raw_priority_score": 70 + max(0, temp_val_alert - app_config.ALERT_BODY_TEMP_FEVER_C),
                    "patient_id": pat_id_alert_val, "context_info": alert_context_info, "triggering_value": f"Temp {temp_val_alert:.1f}°C", "encounter_date": encounter_date_str_alert
                })

        # Rule 3: Fall Detected
        if int(record_item.get('fall_detected_today', 0)) > 0: # Already int from prep
            record_alerts_temp_list.append({
                "alert_level": "CRITICAL", "primary_reason": "Fall Detected",
                "brief_details": f"Falls Today: {int(record_item.get('fall_detected_today',0))}", "suggested_action_code": "ACTION_FALL_ASSESS_URGENT",
                "raw_priority_score": 92, # High fixed priority for fall
                "patient_id": pat_id_alert_val, "context_info": alert_context_info, "triggering_value": "Fall(s) > 0", "encounter_date": encounter_date_str_alert
            })

        # Rule 4: High AI Follow-up Priority Score
        ai_followup_score_alert = record_item.get('ai_followup_priority_score')
        if pd.notna(ai_followup_score_alert) and ai_followup_score_alert >= app_config.FATIGUE_INDEX_HIGH_THRESHOLD: # Using high fatigue threshold as generic high prio
             record_alerts_temp_list.append({
                "alert_level": "WARNING", "primary_reason": "High AI Follow-up Priority",
                "brief_details": f"AI Prio. Score: {ai_followup_score_alert:.0f}", "suggested_action_code": "ACTION_REVIEW_CASE_AI_PRIO",
                "raw_priority_score": min(90, ai_followup_score_alert), # Cap to keep below critical vitals alerts
                "patient_id": pat_id_alert_val, "context_info": alert_context_info, "triggering_value": f"AI Prio {ai_followup_score_alert:.0f}", "encounter_date": encounter_date_str_alert
            })
        
        # Rule 5: High AI Risk Score (INFO level, if no stronger CRITICAL/WARNING alert for this record yet)
        ai_risk_score_alert = record_item.get('ai_risk_score')
        if pd.notna(ai_risk_score_alert) and ai_risk_score_alert >= app_config.RISK_SCORE_HIGH_THRESHOLD:
            if not any(alert_r['alert_level'] in ["CRITICAL", "WARNING"] for alert_r in record_alerts_temp_list):
                 record_alerts_temp_list.append({
                    "alert_level": "INFO", "primary_reason": "Elevated AI Risk Score",
                    "brief_details": f"AI Risk Score: {ai_risk_score_alert:.0f}", "suggested_action_code": "ACTION_MONITOR_RISK_PROFILE",
                    "raw_priority_score": min(70, ai_risk_score_alert), # Ensure lower than WARNING level alerts
                    "patient_id": pat_id_alert_val, "context_info": alert_context_info, "triggering_value": f"AI Risk {ai_risk_score_alert:.0f}", "encounter_date": encounter_date_str_alert
                })
        
        # Rule 6: Pending Critical Condition Referral
        if str(record_item.get('referral_status', '')).lower() == 'pending':
            # Check if condition is one of the key actionable conditions
            is_key_action_condition_referral = any(
                key_c.lower() in str(record_item.get('condition','N/A_Alert')).lower() for key_c in app_config.KEY_CONDITIONS_FOR_ACTION
            )
            if is_key_action_condition_referral:
                record_alerts_temp_list.append({
                    "alert_level": "WARNING", "primary_reason": "Pending Critical Referral",
                    "brief_details": f"For: {str(record_item.get('condition','N/A_Alert'))}", "suggested_action_code": "ACTION_FOLLOWUP_REFERRAL_STATUS",
                    "raw_priority_score": 80, # High moderate priority for pending critical referrals
                    "patient_id": pat_id_alert_val, "context_info": alert_context_info, "triggering_value": "Pending Critical Referral", "encounter_date": encounter_date_str_alert
                })

        if record_alerts_temp_list:
            # For this specific record/encounter, select the single alert with the highest raw_priority_score.
            # This is because one encounter might trigger multiple rules, but we want the most severe single alert from it.
            top_alert_this_record = max(record_alerts_temp_list, key=lambda x_alert: x_alert['raw_priority_score'])
            alerts_buffer.append(top_alert_this_record)

    # De-duplicate: Keep only the single highest priority alert PER PATIENT PER DAY from the buffer.
    if alerts_buffer:
        alerts_df_final = pd.DataFrame(alerts_buffer)
        
        # Map alert levels to sortable integers (Critical=0, Warning=1, Info=2)
        alert_level_order_map = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}
        alerts_df_final['alert_level_sort_val'] = alerts_df_final['alert_level'].map(alert_level_order_map).fillna(3) # Unknowns last
        
        # Sort by patient, then date (string 'YYYY-MM-DD'), then priority (desc), then alert level (asc)
        alerts_df_final.sort_values(
            by=["patient_id", "encounter_date", "raw_priority_score", "alert_level_sort_val"], 
            ascending=[True, True, False, True], # Highest score, most critical level for that score
            inplace=True
        )
        
        # Keep the first (highest priority, most critical) alert for each patient-day combination
        alerts_df_final.drop_duplicates(subset=["patient_id", "encounter_date"], keep="first", inplace=True)
        alerts_df_final.drop(columns=['alert_level_sort_val'], inplace=True) # Clean up helper column
        
        # Final sort of the de-duplicated list by overall priority for display
        alerts_df_final.sort_values(by="raw_priority_score", ascending=False, inplace=True)

        logger.info(f"({module_log_prefix}) Generated {len(alerts_df_final)} unique patient alerts after de-duplication.")
        return alerts_df_final.head(max_alerts_to_return).to_dict(orient='records')
        
    logger.info(f"({module_log_prefix}) No alerts generated from the provided data after processing rules.")
    return []
