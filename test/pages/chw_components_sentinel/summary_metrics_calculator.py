# sentinel_project_root/test/pages/chw_components_sentinel/summary_metrics_calculator.py
# Calculates key summary metrics for a CHW's daily activity for Sentinel.

import pandas as pd
import geopandas as gpd
import numpy as np
import logging
from typing import Dict, Any, Optional

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

def calculate_chw_daily_summary_metrics(
    chw_daily_kpi_input_data: Optional[Dict[str, Any]], # Pre-calculated aggregates, if any
    chw_daily_encounter_df: Optional[pd.DataFrame],    # Raw daily encounters for detailed calculations
    for_date: Any # datetime.date or similar, for context in the output dict
) -> Dict[str, Any]:
    """
    Calculates and returns a dictionary of key CHW daily summary metrics.

    Args:
        chw_daily_kpi_input_data: Optional dict with pre-aggregated values (e.g., worker_self_fatigue_index_today).
        chw_daily_encounter_df: Raw DataFrame of CHW's encounters for the day.
                                Used to calculate/refine most patient-related metrics.
        for_date: The date these metrics apply to.

    Returns:
        Dict[str, Any]: A dictionary of calculated metrics.
    """
    module_log_prefix = "CHWDailySummaryMetrics" # Consistent prefix
    logger.info(f"({module_log_prefix}) Calculating CHW daily summary metrics for date: {str(for_date)}")

    # Initialize with all expected keys and default values
    metrics_summary: Dict[str, Any] = {
        "date_of_activity": str(for_date),
        "visits_count": 0,
        "high_ai_prio_followups_count": 0,
        "avg_risk_of_visited_patients": np.nan,
        "fever_cases_identified_count": 0,      # Patients with temp >= ALERT_BODY_TEMP_FEVER_C
        "high_fever_cases_identified_count": 0, # Patients with temp >= ALERT_BODY_TEMP_HIGH_FEVER_C
        "critical_spo2_cases_identified_count": 0,
        "avg_steps_of_visited_patients": np.nan,
        "fall_events_among_visited_count": 0,
        "pending_critical_referrals_generated_today_count": 0,
        "worker_self_fatigue_level_code": "NOT_ASSESSED", # e.g., LOW, MODERATE, HIGH
        "worker_self_fatigue_index_today": np.nan # Actual score if available
    }

    # Populate from pre-calculated input data first (can be overridden by encounter_df)
    if isinstance(chw_daily_kpi_input_data, dict): # Check if it's a dict
        metrics_summary["visits_count"] = int(chw_daily_kpi_input_data.get('visits_today', metrics_summary["visits_count"]))
        metrics_summary["avg_risk_of_visited_patients"] = float(chw_daily_kpi_input_data.get('avg_patient_risk_visited_today', metrics_summary["avg_risk_of_visited_patients"]))
        metrics_summary["fever_cases_identified_count"] = int(chw_daily_kpi_input_data.get('patients_fever_today', metrics_summary["fever_cases_identified_count"]))
        metrics_summary["high_fever_cases_identified_count"] = int(chw_daily_kpi_input_data.get('patients_high_fever_today', metrics_summary["high_fever_cases_identified_count"]))
        metrics_summary["critical_spo2_cases_identified_count"] = int(chw_daily_kpi_input_data.get('patients_critical_spo2_today', metrics_summary["critical_spo2_cases_identified_count"]))
        metrics_summary["avg_steps_of_visited_patients"] = float(chw_daily_kpi_input_data.get('avg_patient_steps_visited_today', metrics_summary["avg_steps_of_visited_patients"]))
        metrics_summary["fall_events_among_visited_count"] = int(chw_daily_kpi_input_data.get('patients_fall_detected_today', metrics_summary["fall_events_among_visited_count"]))
        metrics_summary["pending_critical_referrals_generated_today_count"] = int(chw_daily_kpi_input_data.get('pending_critical_condition_referrals', metrics_summary["pending_critical_referrals_generated_today_count"]))
        metrics_summary["worker_self_fatigue_index_today"] = float(chw_daily_kpi_input_data.get('worker_self_fatigue_index_today', metrics_summary["worker_self_fatigue_index_today"]))

    # Refine or calculate metrics using the raw daily encounter DataFrame
    if isinstance(chw_daily_encounter_df, pd.DataFrame) and not chw_daily_encounter_df.empty:
        df_enc_src = chw_daily_encounter_df.copy() # Work on a copy

        # Define essential columns and their configurations for safe processing
        essential_cols_for_summary = {
            'patient_id': {"default": "UnknownPID_SumCalc", "type": str},
            'encounter_type': {"default": "UnknownType_SumCalc", "type": str},
            'ai_followup_priority_score': {"default": np.nan, "type": float},
            'ai_risk_score': {"default": np.nan, "type": float},
            'min_spo2_pct': {"default": np.nan, "type": float},
            'vital_signs_temperature_celsius': {"default": np.nan, "type": float},
            'max_skin_temp_celsius': {"default": np.nan, "type": float},
            'avg_daily_steps': {"default": np.nan, "type": float},
            'fall_detected_today': {"default": 0, "type": int},
            'condition': {"default": "UnknownCondition_SumCalc", "type": str},
            'referral_status': {"default": "Unknown_SumCalc", "type": str},
            'referral_reason': {"default": "Unknown_SumCalc", "type": str}
        }
        common_na_strings_sum_calc = ['', 'nan', 'None', 'N/A', '#N/A', 'np.nan', 'NaT', '<NA>', 'null']

        for col, config in essential_cols_for_summary.items():
            if col not in df_enc_src.columns:
                df_enc_src[col] = config["default"]
            # Robust type coercion and NA filling
            if config["type"] == float:
                df_enc_src[col] = pd.to_numeric(df_enc_src[col], errors='coerce').fillna(config["default"])
            elif config["type"] == int:
                 df_enc_src[col] = pd.to_numeric(df_enc_src[col], errors='coerce').fillna(config["default"]).astype(int)
            elif config["type"] == str:
                df_enc_src[col] = df_enc_src[col].astype(str).str.strip().replace(common_na_strings_sum_calc, config["default"], regex=False)
                df_enc_src[col] = df_enc_src[col].fillna(config["default"])
        
        # Exclude worker self-checks from patient-related metrics
        patient_enc_records_df = df_enc_src[
            ~df_enc_src['encounter_type'].str.contains("WORKER_SELF", case=False, na=False)
        ]

        if not patient_enc_records_df.empty:
            # Always recalculate visit count from actual patient encounters if df provided
            metrics_summary["visits_count"] = patient_enc_records_df['patient_id'].nunique()

            if patient_enc_records_df['ai_followup_priority_score'].notna().any():
                metrics_summary["high_ai_prio_followups_count"] = patient_enc_records_df[
                    patient_enc_records_df['ai_followup_priority_score'] >= app_config.FATIGUE_INDEX_HIGH_THRESHOLD # Generic high prio
                ]['patient_id'].nunique()

            if patient_enc_records_df['ai_risk_score'].notna().any():
                # Average risk score per unique patient visited
                unique_patient_risk_series = patient_enc_records_df.drop_duplicates(subset=['patient_id'])['ai_risk_score']
                if unique_patient_risk_series.notna().any():
                    metrics_summary["avg_risk_of_visited_patients"] = unique_patient_risk_series.mean()

            temp_col_name_summary = next((tc for tc in ['vital_signs_temperature_celsius', 'max_skin_temp_celsius'] 
                                          if tc in patient_enc_records_df.columns and patient_enc_records_df[tc].notna().any()), None)
            if temp_col_name_summary:
                metrics_summary["fever_cases_identified_count"] = patient_enc_records_df[
                    patient_enc_records_df[temp_col_name_summary] >= app_config.ALERT_BODY_TEMP_FEVER_C
                ]['patient_id'].nunique()
                metrics_summary["high_fever_cases_identified_count"] = patient_enc_records_df[ # Distinct count for high fever
                    patient_enc_records_df[temp_col_name_summary] >= app_config.ALERT_BODY_TEMP_HIGH_FEVER_C
                ]['patient_id'].nunique()

            if patient_enc_records_df['min_spo2_pct'].notna().any():
                metrics_summary["critical_spo2_cases_identified_count"] = patient_enc_records_df[
                    patient_enc_records_df['min_spo2_pct'] < app_config.ALERT_SPO2_CRITICAL_LOW_PCT
                ]['patient_id'].nunique()
            
            if patient_enc_records_df['avg_daily_steps'].notna().any():
                unique_patient_steps_series = patient_enc_records_df.drop_duplicates(subset=['patient_id'])['avg_daily_steps']
                if unique_patient_steps_series.notna().any():
                    metrics_summary["avg_steps_of_visited_patients"] = unique_patient_steps_series.mean()

            if patient_enc_records_df['fall_detected_today'].notna().any(): # Already int from prep
                metrics_summary["fall_events_among_visited_count"] = patient_enc_records_df[
                    patient_enc_records_df['fall_detected_today'] > 0
                ]['patient_id'].nunique()
            
            # Pending critical referrals generated from today's patient encounters
            crit_conds_set_for_summary = set(app_config.KEY_CONDITIONS_FOR_ACTION)
            urgent_keywords_for_summary = {'urgent', 'emergency', 'critical', 'severe'} # Use set for efficiency
            
            def _is_pending_critical_referral_today(row_data_ref):
                if str(row_data_ref.get('referral_status','')).lower() != 'pending': return False
                cond_str_lower_ref = str(row_data_ref.get('condition','')).lower()
                reason_str_lower_ref = str(row_data_ref.get('referral_reason','')).lower()
                is_key_cond_ref = any(ck_ref.lower() in cond_str_lower_ref for ck_ref in crit_conds_set_for_summary)
                is_urgent_kw_ref = any(uk_ref in reason_str_lower_ref for uk_ref in urgent_keywords_for_summary)
                return is_key_cond_ref or is_urgent_kw_ref
            
            # Apply on the patient encounters DataFrame
            patient_enc_records_df['is_crit_ref_generated_today'] = patient_enc_records_df.apply(_is_pending_critical_referral_today, axis=1)
            metrics_summary["pending_critical_referrals_generated_today_count"] = patient_enc_records_df[
                patient_enc_records_df['is_crit_ref_generated_today']
            ]['patient_id'].nunique()
        else:
            logger.info(f"({module_log_prefix}) No patient-specific encounters in provided daily data to refine metrics after filtering self-checks.")

        # If worker_self_fatigue_index_today was not in kpi_input_data, try to derive from encounter_df
        if pd.isna(metrics_summary.get("worker_self_fatigue_index_today")):
            worker_self_check_recs_from_df_enc = df_enc_src[ # Use original df_enc_src for this
                df_enc_src['encounter_type'].str.contains("WORKER_SELF_CHECK", case=False, na=False)
            ]
            if not worker_self_check_recs_from_df_enc.empty:
                # Assume 'ai_followup_priority_score' or 'rapid_psychometric_distress_score' from WORKER_SELF_CHECK is the fatigue index
                fatigue_score_col_candidate_self = 'ai_followup_priority_score' # Primary candidate
                if fatigue_score_col_candidate_self not in worker_self_check_recs_from_df_enc.columns or \
                   worker_self_check_recs_from_df_enc[fatigue_score_col_candidate_self].isnull().all():
                    fatigue_score_col_candidate_self = 'rapid_psychometric_distress_score' # Fallback candidate
                
                if fatigue_score_col_candidate_self in worker_self_check_recs_from_df_enc.columns and \
                   worker_self_check_recs_from_df_enc[fatigue_score_col_candidate_self].notna().any():
                    # Take the max fatigue score reported by the worker for the day from these records
                    derived_fatigue_score = worker_self_check_recs_from_df_enc[fatigue_score_col_candidate_self].max()
                    metrics_summary["worker_self_fatigue_index_today"] = float(derived_fatigue_score)


    # Update worker_self_fatigue_level_code based on the final worker_self_fatigue_index_today
    final_worker_fatigue_score = metrics_summary.get("worker_self_fatigue_index_today")
    if pd.notna(final_worker_fatigue_score):
        if final_worker_fatigue_score >= app_config.FATIGUE_INDEX_HIGH_THRESHOLD:
            metrics_summary["worker_self_fatigue_level_code"] = "HIGH"
        elif final_worker_fatigue_score >= app_config.FATIGUE_INDEX_MODERATE_THRESHOLD:
            metrics_summary["worker_self_fatigue_level_code"] = "MODERATE"
        else:
            metrics_summary["worker_self_fatigue_level_code"] = "LOW"
            
    # Round numeric metrics for cleaner display output
    for key_to_round in ["avg_risk_of_visited_patients", "avg_steps_of_visited_patients", "worker_self_fatigue_index_today"]:
        if pd.notna(metrics_summary.get(key_to_round)):
            # Round scores/indices to 1 decimal, steps to 0
            decimals = 1 if "risk" in key_to_round or "index" in key_to_round else 0
            metrics_summary[key_to_round] = round(metrics_summary[key_to_round], decimals)
            
    logger.info(f"({module_log_prefix}) CHW daily summary metrics calculated: Visits={metrics_summary['visits_count']}, AvgRisk={metrics_summary['avg_risk_of_visited_patients']:.1f if pd.notna(metrics_summary['avg_risk_of_visited_patients']) else 'N/A'}, FatigueCode={metrics_summary['worker_self_fatigue_level_code']}")
    return metrics_summary
