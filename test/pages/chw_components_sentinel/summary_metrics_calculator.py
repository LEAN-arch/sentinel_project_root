# sentinel_project_root/test/pages/chw_components_sentinel/summary_metrics_calculator.py
# Part of "Sentinel Health Co-Pilot" - LMIC Edge-First System
# This module calculates key summary metrics for a CHW's daily activity.
# It was refactored from the original chw_components/kpi_snapshots.py.
# The output is a structured dictionary for supervisor reports or data sync.

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional

# Assuming app_config is accessible via PYTHONPATH (e.g., if 'test' dir is on path)
# Or, if this module is always called from a script where app_config is already imported,
# it might not need its own direct import. However, for standalone testability, direct import is good.
# Adjust import path if needed based on actual project structure from execution root.
try:
    from config import app_config
except ImportError:
    # Fallback for environments where 'test' is not directly on path but 'test.config' might be
    # This can happen if running tests from sentinel_project_root/
    # Or if this file is imported from a script located higher up.
    import sys
    import os
    # Assuming this file is in sentinel_project_root/test/pages/chw_components_sentinel/
    # Go up three levels to reach sentinel_project_root/test/
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_test_root = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
    if project_test_root not in sys.path:
        sys.path.insert(0, project_test_root)
    from config import app_config


logger = logging.getLogger(__name__)

def calculate_chw_daily_summary_metrics(
    chw_daily_kpi_input_data: Optional[Dict[str, Any]], # Pre-calculated aggregates if available
    chw_daily_encounter_df: Optional[pd.DataFrame], # Raw daily encounter DataFrame for detailed calculations
    for_date: Any # datetime.date or similar, for context
) -> Dict[str, Any]:
    """
    Calculates and returns a dictionary of key CHW daily summary metrics.
    Designed to be called by a CHW Supervisor View or a data aggregation process.

    Args:
        chw_daily_kpi_input_data: Optional dictionary possibly containing pre-aggregated
            values (e.g., from a daily automated summary job).
            Expected keys if provided: 'visits_today', 'avg_patient_risk_visited_today', etc.
        chw_daily_encounter_df: The raw DataFrame of a specific CHW's (or team's)
            encounters for the specified day. Required if chw_daily_kpi_input_data is minimal.
            Expected columns: 'patient_id', 'ai_followup_priority_score', 'ai_risk_score',
                              'min_spo2_pct', 'vital_signs_temperature_celsius' (or 'max_skin_temp_celsius'),
                              'avg_daily_steps', 'fall_detected_today'.
        for_date: The date these metrics apply to.

    Returns:
        Dict[str, Any]: A dictionary of calculated metrics.
    """
    logger.info(f"Calculating CHW daily summary metrics for date: {str(for_date)}")

    metrics_summary = {
        "date_of_activity": str(for_date),
        "visits_count": 0,
        "high_ai_prio_followups_count": 0,
        "avg_risk_of_visited_patients": np.nan,
        "fever_cases_identified_count": 0,
        "critical_spo2_cases_identified_count": 0,
        "avg_steps_of_visited_patients": np.nan,
        "fall_events_among_visited_count": 0,
        "pending_critical_referrals_generated_today_count": 0, # Added
        "worker_self_fatigue_level_code": "NOT_ASSESSED" # Placeholder
    }

    # Use pre-calculated input if provided and valid, otherwise derive from encounter_df
    if chw_daily_kpi_input_data:
        metrics_summary["visits_count"] = int(chw_daily_kpi_input_data.get('visits_today', 0))
        metrics_summary["avg_risk_of_visited_patients"] = float(chw_daily_kpi_input_data.get('avg_patient_risk_visited_today', np.nan))
        metrics_summary["fever_cases_identified_count"] = int(chw_daily_kpi_input_data.get('patients_high_fever_today', 0)) # Align with output of get_chw_summary
        metrics_summary["critical_spo2_cases_identified_count"] = int(chw_daily_kpi_input_data.get('patients_critical_spo2_today', 0)) # Align
        metrics_summary["avg_steps_of_visited_patients"] = float(chw_daily_kpi_input_data.get('avg_patient_steps_visited_today', np.nan))
        metrics_summary["fall_events_among_visited_count"] = int(chw_daily_kpi_input_data.get('patients_fall_detected_today', 0))
        metrics_summary["pending_critical_referrals_generated_today_count"] = int(chw_daily_kpi_input_data.get('pending_critical_condition_referrals', 0)) # Align

    # Refine or calculate metrics if chw_daily_encounter_df is available
    if chw_daily_encounter_df is not None and not chw_daily_encounter_df.empty:
        df_enc = chw_daily_encounter_df.copy() # Work on a copy

        # Ensure essential columns exist in df_enc for calculations, fill with safe defaults if not
        essential_cols_with_defaults = {
            'patient_id': "UnknownPatient", 'ai_followup_priority_score': np.nan,
            'ai_risk_score': np.nan, 'min_spo2_pct': np.nan,
            'vital_signs_temperature_celsius': np.nan, 'max_skin_temp_celsius': np.nan,
            'avg_daily_steps': np.nan, 'fall_detected_today': 0,
            'condition': "UnknownCondition", 'referral_status': "Unknown", 'referral_reason': "Unknown"
        }
        for col, default_val in essential_cols_with_defaults.items():
            if col not in df_enc.columns:
                df_enc[col] = default_val
            elif df_enc[col].isnull().all() and default_val is not np.nan and default_val != "UnknownPatient" and default_val != "UnknownCondition" : #If column exists but all null, fill only if not string defaults
                 df_enc[col] = df_enc[col].fillna(default_val)


        if 'patient_id' in df_enc:
            metrics_summary["visits_count"] = df_enc['patient_id'].nunique() # Always recalculate if df provided for accuracy

        if df_enc['ai_followup_priority_score'].notna().any():
            metrics_summary["high_ai_prio_followups_count"] = df_enc[
                df_enc['ai_followup_priority_score'] >= app_config.FATIGUE_INDEX_HIGH_THRESHOLD # Sentinel config
            ]['patient_id'].nunique()

        if df_enc['ai_risk_score'].notna().any():
            unique_patient_risks = df_enc.drop_duplicates(subset=['patient_id'])['ai_risk_score']
            if unique_patient_risks.notna().any():
                metrics_summary["avg_risk_of_visited_patients"] = unique_patient_risks.mean()

        temp_col_to_use = next((tc for tc in ['vital_signs_temperature_celsius', 'max_skin_temp_celsius'] if tc in df_enc and df_enc[tc].notna().any()), None)
        if temp_col_to_use:
            metrics_summary["fever_cases_identified_count"] = df_enc[
                df_enc[temp_col_to_use] >= app_config.ALERT_BODY_TEMP_FEVER_C # General fever for this count
            ]['patient_id'].nunique()

        if df_enc['min_spo2_pct'].notna().any():
            metrics_summary["critical_spo2_cases_identified_count"] = df_enc[
                df_enc['min_spo2_pct'] < app_config.ALERT_SPO2_CRITICAL_LOW_PCT
            ]['patient_id'].nunique()
        
        if df_enc['avg_daily_steps'].notna().any():
            unique_patient_steps = df_enc.drop_duplicates(subset=['patient_id'])['avg_daily_steps']
            if unique_patient_steps.notna().any():
                metrics_summary["avg_steps_of_visited_patients"] = unique_patient_steps.mean()

        if df_enc['fall_detected_today'].notna().any():
            metrics_summary["fall_events_among_visited_count"] = df_enc[df_enc['fall_detected_today'] > 0]['patient_id'].nunique()
        
        # Calculate pending critical referrals generated from today's encounters
        crit_conds_set = set(app_config.KEY_CONDITIONS_FOR_ACTION)
        urgent_keywords_ref = ['urgent', 'emergency', 'critical', 'severe'] # Keywords indicating urgent referral reason
        df_enc['is_critical_referral_generated_today'] = df_enc.apply(
            lambda r: (str(r.get('referral_status','Unknown')).lower() == 'pending' and
                       (any(ck.lower() in str(r.get('condition','')).lower() for ck in crit_conds_set) or
                        any(uk.lower() in str(r.get('referral_reason','')).lower() for uk in urgent_keywords_ref))
                      ), axis=1
        )
        metrics_summary["pending_critical_referrals_generated_today_count"] = df_enc[
            df_enc['is_critical_referral_generated_today']
        ]['patient_id'].nunique()


    # Round numeric metrics for cleaner output
    for key in ["avg_risk_of_visited_patients", "avg_steps_of_visited_patients"]:
        if pd.notna(metrics_summary.get(key)):
            metrics_summary[key] = round(metrics_summary[key], 1 if key == "avg_risk_of_visited_patients" else 0)
            
    # Placeholder logic for worker's self-fatigue level (would come from worker's PED data in reality)
    # This assumes `chw_daily_kpi_input_data` might have this field if a daily summary was created for the CHW themselves.
    worker_fatigue_score_from_input = chw_daily_kpi_input_data.get('worker_self_fatigue_index_today', np.nan) if chw_daily_kpi_input_data else np.nan
    if pd.notna(worker_fatigue_score_from_input):
        if worker_fatigue_score_from_input >= app_config.FATIGUE_INDEX_HIGH_THRESHOLD:
            metrics_summary["worker_self_fatigue_level_code"] = "HIGH"
        elif worker_fatigue_score_from_input >= app_config.FATIGUE_INDEX_MODERATE_THRESHOLD:
            metrics_summary["worker_self_fatigue_level_code"] = "MODERATE"
        else:
            metrics_summary["worker_self_fatigue_level_code"] = "LOW"
    
    logger.info(f"CHW daily summary metrics calculated: {metrics_summary}")
    return metrics_summary
