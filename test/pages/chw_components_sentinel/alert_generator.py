# sentinel_project_root/test/pages/chw_components_sentinel/alert_generator.py
# Part of "Sentinel Health Co-Pilot" - LMIC Edge-First System
# This module processes CHW daily data to generate structured patient alert information.
# It was refactored from the original chw_components/alerts_display.py.
# The logic here would inform native alerts on a PED or provide data for supervisor reports.

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional

# Assuming app_config is accessible via PYTHONPATH
try:
    from config import app_config
except ImportError:
    import sys, os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_test_root = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
    if project_test_root not in sys.path: sys.path.insert(0, project_test_root)
    from config import app_config

logger = logging.getLogger(__name__)

def generate_chw_patient_alerts_from_data(
    patient_encounter_data_df: pd.DataFrame, # Primary source of data that might trigger alerts
    chw_daily_context_df: Optional[pd.DataFrame] = None, # Optional broader daily data for more context
    for_date: Any, # datetime.date or similar, for context
    chw_zone_context_str: str, # e.g., "Zone A" or "All Assigned Zones"
    max_alerts_to_return: int = 15 # Default limit for reports/summaries
) -> List[Dict[str, Any]]:
    """
    Processes CHW daily data to generate a list of structured patient alerts.
    This list can be used by PEDs for native alerting, or by supervisor dashboards for review.
    
    Args:
        patient_encounter_data_df: DataFrame containing patient encounter data that may trigger alerts.
                                   Expected columns could include: patient_id, encounter_date, zone_id,
                                   ai_risk_score, ai_followup_priority_score,
                                   min_spo2_pct, vital_signs_temperature_celsius, max_skin_temp_celsius,
                                   condition, fall_detected_today, referral_status, medication_adherence_self_report.
        chw_daily_context_df: Optional. Broader daily encounters for overall context if `patient_encounter_data_df` is a subset.
        for_date: The date for which these alerts are relevant.
        chw_zone_context_str: The zone(s) this CHW is covering, for contextual information.
        max_alerts_to_return: Max number of top alerts to return (for reporting).

    Returns:
        A list of dictionaries, where each dictionary represents an actionable alert.
        (Structure as defined in prior refactoring step of this logic).
    """
    module_source_context = "CHWAlertGenerator" # For logging
    logger.info(f"({module_source_context}) Generating CHW patient alerts for date: {str(for_date)}, zone: {chw_zone_context_str}")
    
    if patient_encounter_data_df is None or patient_encounter_data_df.empty:
        logger.info(f"({module_source_context}) No patient encounter data provided for alert generation.")
        return []

    processed_alerts_list: List[Dict[str, Any]] = []
    df_alert_source = patient_encounter_data_df.copy()

    # --- Ensure necessary columns exist with safe defaults ---
    # This mirrors the column expectation from when this logic was part of the old alerts_display.py
    # but is now more explicit for this data preparation function.
    alert_trigger_cols_defaults = {
        'patient_id': "UnknownPID", 'encounter_date': pd.NaT, 'zone_id': chw_zone_context_str,
        'condition': "N/A", 'age': np.nan,
        'ai_risk_score': np.nan, 'ai_followup_priority_score': np.nan,
        'min_spo2_pct': np.nan,
        'vital_signs_temperature_celsius': np.nan, 'max_skin_temp_celsius': np.nan,
        'fall_detected_today': 0,
        'referral_status': "Unknown", 'referral_reason': "Unknown",
        'medication_adherence_self_report': "Unknown"
        # Add other relevant columns that your alerting rules might depend on
    }
    for col_name, default_val in alert_trigger_cols_defaults.items():
        if col_name not in df_alert_source.columns:
            df_alert_source[col_name] = default_val
        # Ensure correct dtypes where possible (e.g., fillna for numeric, then astype if certain)
        elif 'date' in col_name: df_alert_source[col_name] = pd.to_datetime(df_alert_source[col_name], errors='coerce')
        elif any(num_keyword in col_name for num_keyword in ['score','pct','temp','age','fall']):
             df_alert_source[col_name] = pd.to_numeric(df_alert_source[col_name], errors='coerce').fillna(default_val if pd.api.types.is_numeric_dtype(type(default_val)) else np.nan)

    # Determine the best temperature column to use from context
    temp_col_for_alerts = None
    context_df_temp = chw_daily_context_df if chw_daily_context_df is not None and not chw_daily_context_df.empty else df_alert_source
    if 'vital_signs_temperature_celsius' in context_df_temp.columns and context_df_temp['vital_signs_temperature_celsius'].notna().any():
        temp_col_for_alerts = 'vital_signs_temperature_celsius'
    elif 'max_skin_temp_celsius' in context_df_temp.columns and context_df_temp['max_skin_temp_celsius'].notna().any():
        temp_col_for_alerts = 'max_skin_temp_celsius'

    # --- Alerting Rules (LMIC context: prioritize life-threatening, easily identifiable issues) ---
    for _, record_data in df_alert_source.iterrows():
        current_record_alerts: List[Dict[str, Any]] = [] # Stores alerts for this specific record/encounter
        patient_id_alert = str(record_data.get('patient_id', 'UnknownPID'))
        condition_alert = str(record_data.get('condition', 'N/A'))
        zone_alert = str(record_data.get('zone_id', chw_zone_context_str))
        encounter_date_alert = pd.to_datetime(record_data.get('encounter_date', pd.NaT)).strftime('%Y-%m-%d') if pd.notna(record_data.get('encounter_date')) else 'N/A'
        
        context_info_alert = f"Cond: {condition_alert} | Zone: {zone_alert} | Enc.Date: {encounter_date_alert}"

        # Rule 1: Critical Low SpO2
        spo2_value = record_data.get('min_spo2_pct')
        if pd.notna(spo2_value):
            if spo2_value < app_config.ALERT_SPO2_CRITICAL_LOW_PCT:
                current_record_alerts.append({
                    "alert_level": "CRITICAL", "primary_reason": "Critical Low SpO2",
                    "brief_details": f"SpO2: {spo2_value:.0f}%",
                    "suggested_action_code": "ACTION_SPO2_MANAGE_URGENT", "raw_priority_score": 98 + (app_config.ALERT_SPO2_CRITICAL_LOW_PCT - spo2_value), # Higher score for lower SpO2
                    "patient_id": patient_id_alert, "context_info": context_info_alert, "triggering_value": f"SpO2 {spo2_value:.0f}%"
                })
            elif spo2_value < app_config.ALERT_SPO2_WARNING_LOW_PCT:
                 current_record_alerts.append({
                    "alert_level": "WARNING", "primary_reason": "Low SpO2",
                    "brief_details": f"SpO2: {spo2_value:.0f}%",
                    "suggested_action_code": "ACTION_SPO2_RECHECK_MONITOR", "raw_priority_score": 75 + (app_config.ALERT_SPO2_WARNING_LOW_PCT - spo2_value),
                    "patient_id": patient_id_alert, "context_info": context_info_alert, "triggering_value": f"SpO2 {spo2_value:.0f}%"
                })

        # Rule 2: High Fever
        if temp_col_for_alerts and pd.notna(record_data.get(temp_col_for_alerts)):
            temp_alert_val = record_data.get(temp_col_for_alerts)
            if temp_alert_val >= app_config.ALERT_BODY_TEMP_HIGH_FEVER_C:
                current_record_alerts.append({
                    "alert_level": "CRITICAL", "primary_reason": "High Fever",
                    "brief_details": f"Temp: {temp_alert_val:.1f}°C",
                    "suggested_action_code": "ACTION_FEVER_MANAGE_URGENT", "raw_priority_score": 95 + (temp_alert_val - app_config.ALERT_BODY_TEMP_HIGH_FEVER_C)*2,
                    "patient_id": patient_id_alert, "context_info": context_info_alert, "triggering_value": f"Temp {temp_alert_val:.1f}°C"
                })
            elif temp_alert_val >= app_config.ALERT_BODY_TEMP_FEVER_C:
                current_record_alerts.append({
                    "alert_level": "WARNING", "primary_reason": "Fever Present",
                    "brief_details": f"Temp: {temp_alert_val:.1f}°C",
                    "suggested_action_code": "ACTION_FEVER_MONITOR_SUPPORT", "raw_priority_score": 70 + (temp_alert_val - app_config.ALERT_BODY_TEMP_FEVER_C),
                    "patient_id": patient_id_alert, "context_info": context_info_alert, "triggering_value": f"Temp {temp_alert_val:.1f}°C"
                })

        # Rule 3: Fall Detected
        if pd.notna(record_data.get('fall_detected_today')) and record_data['fall_detected_today'] > 0:
            current_record_alerts.append({
                "alert_level": "CRITICAL", "primary_reason": "Fall Detected",
                "brief_details": f"Falls Today: {int(record_data['fall_detected_today'])}",
                "suggested_action_code": "ACTION_FALL_ASSESS_URGENT", "raw_priority_score": 92,
                "patient_id": patient_id_alert, "context_info": context_info_alert, "triggering_value": "Fall(s) > 0"
            })

        # Rule 4: High AI Follow-up Priority Score
        ai_followup_score = record_data.get('ai_followup_priority_score')
        if pd.notna(ai_followup_score) and ai_followup_score >= app_config.FATIGUE_INDEX_HIGH_THRESHOLD: # Using high fatigue as proxy for high prio
             current_record_alerts.append({
                "alert_level": "WARNING", "primary_reason": "High AI Follow-up Prio.",
                "brief_details": f"AI Prio. Score: {ai_followup_score:.0f}",
                "suggested_action_code": "ACTION_REVIEW_CASE_AI_PRIO", "raw_priority_score": ai_followup_score,
                "patient_id": patient_id_alert, "context_info": context_info_alert, "triggering_value": f"AI Prio {ai_followup_score:.0f}"
            })
        
        # Rule 5: High AI Risk Score (if not already covered by specific vital or AI Follow-up Prio above for CRITICAL/WARNING)
        ai_general_risk_score = record_data.get('ai_risk_score')
        if pd.notna(ai_general_risk_score) and ai_general_risk_score >= app_config.RISK_SCORE_HIGH_THRESHOLD:
            is_already_crit_warn = any(a['alert_level'] in ["CRITICAL", "WARNING"] for a in current_record_alerts)
            if not is_already_crit_warn: # Only add as INFO if no stronger alert exists for this record yet
                 current_record_alerts.append({
                    "alert_level": "INFO", "primary_reason": "Elevated AI Risk Score",
                    "brief_details": f"AI Risk Score: {ai_general_risk_score:.0f}",
                    "suggested_action_code": "ACTION_MONITOR_RISK_PROFILE", "raw_priority_score": ai_general_risk_score,
                    "patient_id": patient_id_alert, "context_info": context_info_alert, "triggering_value": f"AI Risk {ai_general_risk_score:.0f}"
                })
        
        # Rule 6: Pending Critical Condition Referral (generates a WARNING level alert)
        if str(record_data.get('referral_status', '')).lower() == 'pending':
            is_critical_cond_referral = any(
                cond_key.lower() in str(record_data.get('condition','')).lower() for cond_key in app_config.KEY_CONDITIONS_FOR_ACTION
            )
            if is_critical_cond_referral:
                current_record_alerts.append({
                    "alert_level": "WARNING", "primary_reason": f"Pending Critical Referral",
                    "brief_details": f"For: {record_data.get('condition','N/A')}",
                    "suggested_action_code": "ACTION_FOLLOWUP_REFERRAL_STATUS", "raw_priority_score": 80, # High moderate priority
                    "patient_id": patient_id_alert, "context_info": context_info_alert, "triggering_value": "Pending Critical Ref."
                })

        if current_record_alerts:
            # Take the single alert with the highest priority for this specific encounter/record
            # (as one encounter might trigger multiple rules).
            # A PED could show multiple contributing factors.
            top_alert_for_this_record = max(current_record_alerts, key=lambda x: x['raw_priority_score'])
            processed_alerts_list.append(top_alert_for_this_record)

    # De-duplicate: If a patient has multiple encounters/records in `df_alert_source` that generate alerts,
    # ensure we only return the single highest priority alert *for that patient on that day*.
    if processed_alerts_list:
        final_alerts_df = pd.DataFrame(processed_alerts_list)
        final_alerts_df['encounter_date_obj'] = pd.to_datetime(final_alerts_df['encounter_date'], errors='coerce').dt.date
        
        # Sort by priority descending, then by a more specific trigger if possible, then by patient to make drop_duplicates deterministic
        final_alerts_df.sort_values(by=["raw_priority_score", "alert_level"], ascending=[False, True], inplace=True) # Critical first if scores equal
        
        # Keep only the highest priority alert per patient per day
        # A supervisor might want to see all alerts, but for concise list this is often done.
        final_alerts_df.drop_duplicates(subset=["patient_id", "encounter_date_obj"], keep="first", inplace=True)
        
        return final_alerts_df.head(max_alerts_to_return).to_dict(orient='records')
        
    return []
