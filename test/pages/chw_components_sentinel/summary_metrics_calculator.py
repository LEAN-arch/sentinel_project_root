# sentinel_project_root/test/pages/chw_components_sentinel/summary_metrics_calculator.py
# Calculates key summary metrics for a CHW's daily activity for Sentinel.

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional

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

def calculate_chw_daily_summary_metrics(
    chw_daily_kpi_input_data: Optional[Dict[str, Any]], # Pre-calculated aggregates
    chw_daily_encounter_df: Optional[pd.DataFrame],    # Raw daily encounters for detailed calcs
    for_date: Any # datetime.date or similar, for context
) -> Dict[str, Any]:
    """
    Calculates and returns a dictionary of key CHW daily summary metrics.

    Args:
        chw_daily_kpi_input_data: Optional dict with pre-aggregated values.
        chw_daily_encounter_df: Raw DataFrame of CHW's encounters for the day.
        for_date: The date these metrics apply to.

    Returns:
        Dict[str, Any]: A dictionary of calculated metrics.
    """
    module_log_prefix = "CHWDailySummaryMetrics"
    logger.info(f"({module_log_prefix}) Calculating CHW daily summary metrics for date: {str(for_date)}")

    metrics_summary: Dict[str, Any] = {
        "date_of_activity": str(for_date),
        "visits_count": 0,
        "high_ai_prio_followups_count": 0,
        "avg_risk_of_visited_patients": np.nan,
        "fever_cases_identified_count": 0, # Patients with fever (>= ALERT_BODY_TEMP_FEVER_C)
        "high_fever_cases_identified_count": 0, # Patients with high fever (>= ALERT_BODY_TEMP_HIGH_FEVER_C) - added for distinct count
        "critical_spo2_cases_identified_count": 0,
        "avg_steps_of_visited_patients": np.nan,
        "fall_events_among_visited_count": 0,
        "pending_critical_referrals_generated_today_count": 0,
        "worker_self_fatigue_level_code": "NOT_ASSESSED", # e.g., LOW, MODERATE, HIGH
        "worker_self_fatigue_index_today": np.nan # Actual score if available
    }

    # Use pre-calculated input if provided (can be overridden/refined by encounter_df)
    if chw_daily_kpi_input_data: # Ensure it's not None
        metrics_summary["visits_count"] = int(chw_daily_kpi_input_data.get('visits_today', 0))
        metrics_summary["avg_risk_of_visited_patients"] = float(chw_daily_kpi_input_data.get('avg_patient_risk_visited_today', np.nan))
        # Assuming get_chw_summary from core_data might use 'patients_high_fever_today' for any fever
        metrics_summary["fever_cases_identified_count"] = int(chw_daily_kpi_input_data.get('patients_fever_today', # A more general key
                                                                chw_daily_kpi_input_data.get('patients_high_fever_today', 0))) 
        metrics_summary["high_fever_cases_identified_count"] = int(chw_daily_kpi_input_data.get('patients_high_fever_today', 0))
        metrics_summary["critical_spo2_cases_identified_count"] = int(chw_daily_kpi_input_data.get('patients_critical_spo2_today', 0))
        metrics_summary["avg_steps_of_visited_patients"] = float(chw_daily_kpi_input_data.get('avg_patient_steps_visited_today', np.nan))
        metrics_summary["fall_events_among_visited_count"] = int(chw_daily_kpi_input_data.get('patients_fall_detected_today', 0))
        metrics_summary["pending_critical_referrals_generated_today_count"] = int(chw_daily_kpi_input_data.get('pending_critical_condition_referrals', 0))
        metrics_summary["worker_self_fatigue_index_today"] = float(chw_daily_kpi_input_data.get('worker_self_fatigue_index_today', np.nan))


    # Refine or calculate metrics if chw_daily_encounter_df is available
    if isinstance(chw_daily_encounter_df, pd.DataFrame) and not chw_daily_encounter_df.empty:
        df_enc = chw_daily_encounter_df.copy()

        # Ensure essential columns exist, adding with safe defaults if not
        # These defaults help prevent errors in subsequent calculations.
        essential_cols_config_summary = {
            'patient_id': {"default": "UnknownPID_Summary", "type": str},
            'encounter_type': {"default": "UnknownType", "type": str}, # For filtering worker self-checks
            'ai_followup_priority_score': {"default": np.nan, "type": float},
            'ai_risk_score': {"default": np.nan, "type": float},
            'min_spo2_pct': {"default": np.nan, "type": float},
            'vital_signs_temperature_celsius': {"default": np.nan, "type": float},
            'max_skin_temp_celsius': {"default": np.nan, "type": float},
            'avg_daily_steps': {"default": np.nan, "type": float},
            'fall_detected_today': {"default": 0, "type": int},
            'condition': {"default": "UnknownCondition_Summary", "type": str},
            'referral_status': {"default": "Unknown_Summary", "type": str},
            'referral_reason': {"default": "Unknown_Summary", "type": str}
        }
        common_na_strings_summary = ['', 'nan', 'None', 'N/A', '#N/A', 'np.nan', 'NaT', '<NA>', 'null']

        for col, config in essential_cols_config_summary.items():
            if col not in df_enc.columns:
                df_enc[col] = config["default"]
            # Coercion and filling for existing columns
            if config["type"] == float:
                df_enc[col] = pd.to_numeric(df_enc[col], errors='coerce').fillna(config["default"])
            elif config["type"] == int:
                 df_enc[col] = pd.to_numeric(df_enc[col], errors='coerce').fillna(config["default"]).astype(int)
            elif config["type"] == str:
                df_enc[col] = df_enc[col].astype(str).str.strip().replace(common_na_strings_summary, config["default"], regex=False)
                df_enc[col] = df_enc[col].fillna(config["default"])
        
        # Exclude worker self-checks from patient-related metrics
        patient_enc_df = df_enc[~df_enc['encounter_type'].str.contains("WORKER_SELF", case=False, na=False)]

        if not patient_enc_df.empty:
            if 'patient_id' in patient_enc_df.columns: # Should always be true after prep
                metrics_summary["visits_count"] = patient_enc_df['patient_id'].nunique()

            if patient_enc_df['ai_followup_priority_score'].notna().any():
                metrics_summary["high_ai_prio_followups_count"] = patient_enc_df[
                    patient_enc_df['ai_followup_priority_score'] >= app_config.FATIGUE_INDEX_HIGH_THRESHOLD
                ]['patient_id'].nunique()

            if patient_enc_df['ai_risk_score'].notna().any():
                unique_patient_risk_scores = patient_enc_df.drop_duplicates(subset=['patient_id'])['ai_risk_score']
                if unique_patient_risk_scores.notna().any():
                    metrics_summary["avg_risk_of_visited_patients"] = unique_patient_risk_scores.mean()

            temp_col_to_use = next((tc for tc in ['vital_signs_temperature_celsius', 'max_skin_temp_celsius'] 
                                    if tc in patient_enc_df.columns and patient_enc_df[tc].notna().any()), None)
            if temp_col_to_use:
                metrics_summary["fever_cases_identified_count"] = patient_enc_df[
                    patient_enc_df[temp_col_to_use] >= app_config.ALERT_BODY_TEMP_FEVER_C
                ]['patient_id'].nunique()
                metrics_summary["high_fever_cases_identified_count"] = patient_enc_df[
                    patient_enc_df[temp_col_to_use] >= app_config.ALERT_BODY_TEMP_HIGH_FEVER_C
                ]['patient_id'].nunique()


            if patient_enc_df['min_spo2_pct'].notna().any():
                metrics_summary["critical_spo2_cases_identified_count"] = patient_enc_df[
                    patient_enc_df['min_spo2_pct'] < app_config.ALERT_SPO2_CRITICAL_LOW_PCT
                ]['patient_id'].nunique()
            
            if patient_enc_df['avg_daily_steps'].notna().any():
                unique_patient_steps = patient_enc_df.drop_duplicates(subset=['patient_id'])['avg_daily_steps']
                if unique_patient_steps.notna().any():
                    metrics_summary["avg_steps_of_visited_patients"] = unique_patient_steps.mean()

            if patient_enc_df['fall_detected_today'].notna().any(): # Already int from prep
                metrics_summary["fall_events_among_visited_count"] = patient_enc_df[
                    patient_enc_df['fall_detected_today'] > 0
                ]['patient_id'].nunique()
            
            crit_conds_set = set(app_config.KEY_CONDITIONS_FOR_ACTION)
            urgent_keywords = {'urgent', 'emergency', 'critical', 'severe'}
            
            def _is_pending_critical_referral_summary(row_data):
                if str(row_data.get('referral_status','')).lower() != 'pending': return False
                cond_str_lower = str(row_data.get('condition','')).lower()
                reason_str_lower = str(row_data.get('referral_reason','')).lower()
                is_key_cond = any(ck.lower() in cond_str_lower for ck in crit_conds_set)
                is_urgent_kw = any(uk in reason_str_lower for uk in urgent_keywords)
                return is_key_cond or is_urgent_kw
            
            patient_enc_df['is_crit_ref_today'] = patient_enc_df.apply(_is_pending_critical_referral_summary, axis=1)
            metrics_summary["pending_critical_referrals_generated_today_count"] = patient_enc_df[
                patient_enc_df['is_crit_ref_today']
            ]['patient_id'].nunique()
        else:
            logger.info(f"({module_log_prefix}) No patient-specific encounters in provided daily data to refine metrics.")


    # Update worker self-fatigue level code based on the index (either from input or derived if worker data was in encounter_df)
    worker_fatigue_score = metrics_summary.get("worker_self_fatigue_index_today", np.nan) # Get from potentially updated summary
    
    # If worker_self_fatigue_index_today is still NaN, try to get it from WORKER_SELF_CHECK in original df_enc
    if pd.isna(worker_fatigue_score) and isinstance(chw_daily_encounter_df, pd.DataFrame) and not chw_daily_encounter_df.empty:
        # Re-check original df_enc before it was filtered to patient_enc_df
        worker_self_checks_in_df_enc = chw_daily_encounter_df[
            chw_daily_encounter_df.get('encounter_type', pd.Series(dtype=str)).str.contains("WORKER_SELF_CHECK", case=False, na=False)
        ]
        if not worker_self_checks_in_df_enc.empty:
            # Assume 'ai_followup_priority_score' from WORKER_SELF_CHECK is the fatigue index, or a dedicated column
            fatigue_score_col_candidate = 'ai_followup_priority_score' # Example
            if fatigue_score_col_candidate in worker_self_checks_in_df_enc.columns and \
               worker_self_checks_in_df_enc[fatigue_score_col_candidate].notna().any():
                worker_fatigue_score = worker_self_checks_in_df_enc[fatigue_score_col_candidate].max()
                metrics_summary["worker_self_fatigue_index_today"] = worker_fatigue_score # Update the score as well


    if pd.notna(worker_fatigue_score):
        if worker_fatigue_score >= app_config.FATIGUE_INDEX_HIGH_THRESHOLD:
            metrics_summary["worker_self_fatigue_level_code"] = "HIGH"
        elif worker_fatigue_score >= app_config.FATIGUE_INDEX_MODERATE_THRESHOLD:
            metrics_summary["worker_self_fatigue_level_code"] = "MODERATE"
        else:
            metrics_summary["worker_self_fatigue_level_code"] = "LOW"
            
    # Round numeric metrics for cleaner output
    for key_round in ["avg_risk_of_visited_patients", "avg_steps_of_visited_patients", "worker_self_fatigue_index_today"]:
        if pd.notna(metrics_summary.get(key_round)):
            metrics_summary[key_round] = round(metrics_summary[key_round], 1 if "risk" in key_round or "index" in key_round else 0)
            
    logger.info(f"({module_log_prefix}) CHW daily summary metrics calculated: Visits={metrics_summary['visits_count']}, HighPrioFollowups={metrics_summary['high_ai_prio_followups_count']}")
    return metrics_summary
