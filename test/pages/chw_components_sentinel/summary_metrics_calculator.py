# sentinel_project_root/test/pages/chw_components_sentinel/summary_metrics_calculator.py
# Calculates key summary metrics for a CHW's daily activity for Sentinel.

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional
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

def calculate_chw_daily_summary_metrics(
    chw_daily_kpi_input_data: Optional[Dict[str, Any]],
    chw_daily_encounter_df: Optional[pd.DataFrame],
    for_date: Any
) -> Dict[str, Any]:
    """
    Calculates and returns a dictionary of key CHW daily summary metrics.

    Args:
        chw_daily_kpi_input_data: Optional dict with pre-aggregated values.
        chw_daily_encounter_df: Raw DataFrame of CHW's encounters for the day.
        for_date: The date these metrics apply to.

    Returns:
        Dict[str, Any]: A dictionary of calculated metrics.

    Raises:
        ValueError: If no valid input data is provided.
    """
    module_log_prefix = "CHWDailySummaryMetrics"

    # Validate app_config attributes
    required_configs = [
        'FATIGUE_INDEX_HIGH_THRESHOLD', 'FATIGUE_INDEX_MODERATE_THRESHOLD',
        'ALERT_BODY_TEMP_FEVER_C', 'ALERT_BODY_TEMP_HIGH_FEVER_C',
        'ALERT_SPO2_CRITICAL_LOW_PCT', 'KEY_CONDITIONS_FOR_ACTION'
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
        raise ValueError("Invalid date format for summary metrics")

    logger.info(f"({module_log_prefix}) Calculating CHW daily summary metrics for date: {for_date_str}")

    # Initialize metrics
    metrics_summary: Dict[str, Any] = {
        "date_of_activity": for_date_str,
        "visits_count": 0,
        "high_ai_prio_followups_count": 0,
        "avg_risk_of_visited_patients": np.nan,
        "fever_cases_identified_count": 0,
        "high_fever_cases_identified_count": 0,
        "critical_spo2_cases_identified_count": 0,
        "avg_steps_of_visited_patients": np.nan,
        "fall_events_among_visited_count": 0,
        "pending_critical_referrals_generated_today_count": 0,
        "worker_self_fatigue_level_code": "NOT_ASSESSED",
        "worker_self_fatigue_index_today": np.nan
    }

    # Populate from KPI input data
    if isinstance(chw_daily_kpi_input_data, dict):
        try:
            metrics_summary["visits_count"] = int(chw_daily_kpi_input_data.get('visits_today', 0))
            metrics_summary["avg_risk_of_visited_patients"] = float(chw_daily_kpi_input_data.get('avg_patient_risk_visited_today', np.nan))
            metrics_summary["fever_cases_identified_count"] = int(chw_daily_kpi_input_data.get('patients_fever_today', 0))
            metrics_summary["high_fever_cases_identified_count"] = int(chw_daily_kpi_input_data.get('patients_high_fever_today', 0))
            metrics_summary["critical_spo2_cases_identified_count"] = int(chw_daily_kpi_input_data.get('patients_critical_spo2_today', 0))
            metrics_summary["avg_steps_of_visited_patients"] = float(chw_daily_kpi_input_data.get('avg_patient_steps_visited_today', np.nan))
            metrics_summary["fall_events_among_visited_count"] = int(chw_daily_kpi_input_data.get('patients_fall_detected_today', 0))
            metrics_summary["pending_critical_referrals_generated_today_count"] = int(chw_daily_kpi_input_data.get('pending_critical_condition_referrals', 0))
            metrics_summary["worker_self_fatigue_index_today"] = float(chw_daily_kpi_input_data.get('worker_self_fatigue_index_today', np.nan))
        except (ValueError, TypeError) as e:
            logger.warning(f"({module_log_prefix}) Invalid KPI data: {e}")

    # Refine metrics using encounter DataFrame
    if isinstance(chw_daily_encounter_df, pd.DataFrame) and not chw_daily_encounter_df.empty:
        # Define essential columns
        essential_cols_for_summary = {
            'patient_id': {"default": "UnknownPID_SumCalc", "type": str},
            'encounter_type': {"default": "UnknownType_SumCalc", "type": str},
            'ai_followup_priority_score': {"default": np.nan, "type": "float"},
            'ai_risk_score': {"default": np.nan, "type": "float"},
            'min_spo2_pct': {"default": np.nan, "type": "float"},
            'vital_signs_temperature_celsius': {"default": np.nan, "type": "float"},
            'max_skin_temp_celsius': {"default": np.nan, "type": "float"},
            'avg_daily_steps': {"default": np.nan, "type": "float"},
            'fall_detected_today': {"default": 0, "type": "int"},
            'condition': {"default": "UnknownCondition_SumCalc", "type": str},
            'referral_status': {"default": "Unknown_SumCalc", "type": str},
            'referral_reason': {"default": "Unknown_SumCalc", "type": str}
        }
        common_na_strings = ['', 'nan', 'None', 'N/A', '#N/A', 'np.nan', 'NaT', '<NA>', 'null']

        # Select necessary columns
        required_cols = list(essential_cols_for_summary.keys())
        df_enc_src = chw_daily_encounter_df[required_cols].copy() if all(col in chw_daily_encounter_df.columns for col in required_cols) else chw_daily_encounter_df.copy()

        for col, config in essential_cols_for_summary.items():
            if col not in df_enc_src.columns:
                df_enc_src[col] = config["default"]
            if config["type"] == "float":
                df_enc_src[col] = pd.to_numeric(df_enc_src[col], errors='coerce').fillna(config["default"])
            elif config["type"] == "int":
                df_enc_src[col] = pd.to_numeric(df_enc_src[col], errors='coerce').fillna(config["default"]).astype(int)
            elif config["type"] == str:
                df_enc_src[col] = df_enc_src[col].astype(str).replace(common_na_strings, config["default"])

        # Exclude worker self-checks
        patient_enc_records_df = df_enc_src[~df_enc_src['encounter_type'].str.contains("WORKER_SELF", case=False, na=False)]

        if not patient_enc_records_df.empty:
            metrics_summary["visits_count"] = patient_enc_records_df['patient_id'].nunique()

            if 'ai_followup_priority_score' in patient_enc_records_df.columns:
                metrics_summary["high_ai_prio_followups_count"] = patient_enc_records_df[
                    patient_enc_records_df['ai_followup_priority_score'] >= app_config.FATIGUE_INDEX_HIGH_THRESHOLD
                ]['patient_id'].nunique()

            if 'ai_risk_score' in patient_enc_records_df.columns:
                unique_risk = patient_enc_records_df.drop_duplicates(subset=['patient_id'])['ai_risk_score']
                if unique_risk.notna().any():
                    metrics_summary["avg_risk_of_visited_patients"] = unique_risk.mean()

            temp_col_name = 'vital_signs_temperature_celsius' if 'vital_signs_temperature_celsius' in patient_enc_records_df.columns else \
                            'max_skin_temp_celsius' if 'max_skin_temp_celsius' in patient_enc_records_df.columns else None
            if temp_col_name:
                metrics_summary["fever_cases_identified_count"] = patient_enc_records_df[
                    patient_enc_records_df[temp_col_name] >= app_config.ALERT_BODY_TEMP_FEVER_C
                ]['patient_id'].nunique()
                metrics_summary["high_fever_cases_identified_count"] = patient_enc_records_df[
                    patient_enc_records_df[temp_col_name] >= app_config.ALERT_BODY_TEMP_HIGH_FEVER_C
                ]['patient_id'].nunique()

            if 'min_spo2_pct' in patient_enc_records_df.columns:
                metrics_summary["critical_spo2_cases_identified_count"] = patient_enc_records_df[
                    patient_enc_records_df['min_spo2_pct'] < app_config.ALERT_SPO2_CRITICAL_LOW_PCT
                ]['patient_id'].nunique()

            if 'avg_daily_steps' in patient_enc_records_df.columns:
                unique_steps = patient_enc_records_df.drop_duplicates(subset=['patient_id'])['avg_daily_steps']
                if unique_steps.notna().any():
                    metrics_summary["avg_steps_of_visited_patients"] = unique_steps.mean()

            if 'fall_detected_today' in patient_enc_records_df.columns:
                metrics_summary["fall_events_among_visited_count"] = patient_enc_records_df[
                    patient_enc_records_df['fall_detected_today'] > 0
                ]['patient_id'].nunique()

            # Vectorized critical referrals
            crit_conds_set = set(app_config.KEY_CONDITIONS_FOR_ACTION)
            urgent_keywords = {'urgent', 'emergency', 'critical', 'severe'}
            crit_cond_mask = patient_enc_records_df['condition'].str.lower().apply(
                lambda x: any(ck.lower() in x for ck in crit_conds_set)
            )
            urgent_mask = patient_enc_records_df['referral_reason'].str.lower().apply(
                lambda x: any(uk in x for uk in urgent_keywords)
            )
            is_crit_ref = (patient_enc_records_df['referral_status'].str.lower() == 'pending') & (crit_cond_mask | urgent_mask)
            metrics_summary["pending_critical_referrals_generated_today_count"] = patient_enc_records_df[is_crit_ref]['patient_id'].nunique()
        else:
            logger.info(f"({module_log_prefix}) No patient-specific encounters found.")

        # Derive worker fatigue from encounter data
        if pd.isna(metrics_summary["worker_self_fatigue_index_today"]):
            worker_self_checks = df_enc_src[df_enc_src['encounter_type'].str.contains("WORKER_SELF_CHECK", case=False, na=False)]
            if not worker_self_checks.empty:
                fatigue_col = 'ai_followup_priority_score' if 'ai_followup_priority_score' in worker_self_checks.columns and worker_self_checks['ai_followup_priority_score'].notna().any() else \
                              'rapid_psychometric_distress_score' if 'rapid_psychometric_distress_score' in worker_self_checks.columns and worker_self_checks['rapid_psychometric_distress_score'].notna().any() else None
                if fatigue_col:
                    metrics_summary["worker_self_fatigue_index_today"] = float(worker_self_checks[fatigue_col].max())
                else:
                    logger.warning(f"({module_log_prefix}) No valid fatigue score in worker self-checks.")

    # Update fatigue level code
    fatigue_score = metrics_summary["worker_self_fatigue_index_today"]
    if pd.notna(fatigue_score):
        if fatigue_score >= app_config.FATIGUE_INDEX_HIGH_THRESHOLD:
            metrics_summary["worker_self_fatigue_level_code"] = "HIGH"
        elif fatigue_score >= app_config.FATIGUE_INDEX_MODERATE_THRESHOLD:
            metrics_summary["worker_self_fatigue_level_code"] = "MODERATE"
        else:
            metrics_summary["worker_self_fatigue_level_code"] = "LOW"

    # Round numeric metrics
    round_keys = {
        "avg_risk_of_visited_patients": 1,
        "avg_steps_of_visited_patients": 0,
        "worker_self_fatigue_index_today": 1
    }
    for key, decimals in round_keys.items():
        if pd.notna(metrics_summary.get(key)):
            metrics_summary[key] = round(metrics_summary[key], decimals)

    logger.info(
        f"({module_log_prefix}) CHW daily summary metrics calculated: "
        f"Visits={metrics_summary['visits_count']}, "
        f"AvgRisk={metrics_summary['avg_risk_of_visited_patients']:.1f if pd.notna(metrics_summary['avg_risk_of_visited_patients']) else 'N/A'}, "
        f"FatigueCode={metrics_summary['worker_self_fatigue_level_code']}"
    )
    return metrics_summary
