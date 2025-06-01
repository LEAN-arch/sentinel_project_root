# sentinel_project_root/test/pages/clinic_components_sentinel/patient_focus_data_preparer.py
# Prepares data for clinic patient load and flagged patient cases for Sentinel.

import pandas as pd
import numpy as np # For np.nan if used, though less direct here
import logging
from typing import Dict, Any, Optional, List

# Standardized import block
try:
    from config import app_config
    from utils.core_data_processing import get_patient_alerts_for_clinic # Central utility for flagged patients
except ImportError:
    import sys
    import os
    # Assumes this file is in sentinel_project_root/test/pages/clinic_components_sentinel/
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_test_root_dir = os.path.abspath(os.path.join(current_script_dir, os.pardir, os.pardir))
    if project_test_root_dir not in sys.path:
        sys.path.insert(0, project_test_root_dir)
    from config import app_config
    from utils.core_data_processing import get_patient_alerts_for_clinic

logger = logging.getLogger(__name__)

def prepare_clinic_patient_focus_data(
    filtered_health_df_clinic_period: Optional[pd.DataFrame], # Health data for the clinic and specific period
    reporting_period_str: str,
    patient_load_time_aggregation: str = 'W-Mon', # Default to weekly load for clinic overview (e.g., 'D', 'W-Mon')
) -> Dict[str, Any]:
    """
    Prepares data for patient load analysis and a list of flagged patient cases for clinical review.

    Args:
        filtered_health_df_clinic_period: DataFrame of health records for the clinic and selected period.
                                          Expected columns: 'encounter_date', 'condition', 'patient_id',
                                          plus any columns needed by get_patient_alerts_for_clinic.
        reporting_period_str: String describing the reporting period (for context).
        patient_load_time_aggregation: Aggregation period for patient load ('D' for daily, 'W-Mon' for weekly).

    Returns:
        Dict[str, Any]: A dictionary containing:
            "reporting_period": str,
            "patient_load_by_key_condition_df": pd.DataFrame (period_start_date, condition, unique_patients_count),
            "flagged_patients_for_review_df": pd.DataFrame (from get_patient_alerts_for_clinic),
            "processing_notes": List[str]
    """
    module_log_prefix = "ClinicPatientFocusPreparer" # Consistent prefix
    logger.info(f"({module_log_prefix}) Preparing patient focus data for period: {reporting_period_str}, Load Agg: {patient_load_time_aggregation}")

    # Initialize output structure with defaults, especially for DataFrames
    expected_flagged_cols_init = ['patient_id', 'encounter_date', 'condition', 'Alert Reason', 'Priority Score', 'ai_risk_score', 'age', 'gender', 'zone_id', 'referred_to_facility_id']
    patient_focus_output: Dict[str, Any] = {
        "reporting_period": reporting_period_str,
        "patient_load_by_key_condition_df": pd.DataFrame(columns=['period_start_date', 'condition', 'unique_patients_count']),
        "flagged_patients_for_review_df": pd.DataFrame(columns=expected_flagged_cols_init),
        "processing_notes": []
    }

    if not isinstance(filtered_health_df_clinic_period, pd.DataFrame) or filtered_health_df_clinic_period.empty:
        note = "No health data provided for patient focus data preparation. Output will be empty."
        logger.warning(f"({module_log_prefix}) {note}")
        patient_focus_output["processing_notes"].append(note)
        return patient_focus_output

    df_focus_src = filtered_health_df_clinic_period.copy() # Work on a copy

    # Ensure essential columns for this module's direct logic are present and correctly typed
    essential_cols_for_focus = {
        'encounter_date': {"default": pd.NaT, "type": "datetime"},
        'patient_id': {"default": "UnknownPID_FocusPrep", "type": str},
        'condition': {"default": "UnknownCondition_FocusPrep", "type": str}
    }
    common_na_strings_focus_prep = ['', 'nan', 'None', 'N/A', '#N/A', 'np.nan', 'NaT', '<NA>', 'null']

    for col, config_item in essential_cols_for_focus.items():
        if col not in df_focus_src.columns:
            df_focus_src[col] = config_item["default"]
            note = f"Essential column '{col}' missing for patient focus prep, added with default '{config_item['default']}'."
            logger.debug(f"({module_log_prefix}) {note}") # Debug as loader should handle this mostly
            # patient_focus_output["processing_notes"].append(note) # Not critical enough for user note if loader handles
        
        if config_item["type"] == "datetime":
            df_focus_src[col] = pd.to_datetime(df_focus_src[col], errors='coerce')
        elif config_item["type"] == str:
             df_focus_src[col] = df_focus_src[col].astype(str).str.strip().replace(common_na_strings_focus_prep, config_item["default"], regex=False)
             df_focus_src[col] = df_focus_src[col].fillna(config_item["default"])

    # Drop rows if critical identifiers are still missing/NaT after cleaning
    df_focus_src.dropna(subset=['encounter_date', 'patient_id'], inplace=True)
    if df_focus_src.empty:
        note = "No valid records with encounter_date & patient_id after cleaning for patient focus data."
        logger.warning(f"({module_log_prefix}) {note}")
        patient_focus_output["processing_notes"].append(note)
        return patient_focus_output # Return initialized empty structures

    # 1. Patient Load by Key Condition
    key_conditions_list_load = app_config.KEY_CONDITIONS_FOR_ACTION
    
    if 'condition' in df_focus_src.columns and key_conditions_list_load:
        # Build a regex pattern for more robust matching of key conditions (e.g., whole word or common variations)
        # This example is simple contains, but could be enhanced.
        condition_regex_pattern = '|'.join([f"(?:^|[^a-zA-Z]){cond}(?:$|[^a-zA-Z])" for cond in key_conditions_list_load])
        
        df_for_load_aggregation = df_focus_src[
            df_focus_src['condition'].str.contains(condition_regex_pattern, case=False, na=False) &
            (~df_focus_src['patient_id'].isin(["UnknownPID_FocusPrep"])) # Exclude generic placeholder PIDs
        ].copy() # Use .copy() if further modifications are made to this slice

        if not df_for_load_aggregation.empty:
            # Group by specified time period and condition, counting unique patients
            df_load_summary = df_for_load_aggregation.groupby(
                [pd.Grouper(key='encounter_date', freq=patient_load_time_aggregation, label='left', closed='left'), 'condition']
            )['patient_id'].nunique().reset_index() # nunique for unique patients
            
            df_load_summary.rename(
                columns={'encounter_date': 'period_start_date', 'patient_id': 'unique_patients_count'},
                inplace=True
            )
            patient_focus_output["patient_load_by_key_condition_df"] = df_load_summary
            if df_load_summary.empty:
                patient_focus_output["processing_notes"].append("No patient load data aggregated for key conditions in the period (empty after grouping).")
        else:
            patient_focus_output["processing_notes"].append("No encounters found matching key conditions for patient load analysis in the period.")
    else:
        patient_focus_output["processing_notes"].append("'condition' column missing or no KEY_CONDITIONS_FOR_ACTION in config, skipping patient load by condition.")


    # 2. Flagged Patient Cases for Clinical Review
    # This relies on the robust get_patient_alerts_for_clinic from core_data_processing.
    # That function handles AI scores, critical vitals, and other rules internally.
    df_flagged_patients = get_patient_alerts_for_clinic(
        health_df_period=df_focus_src, # Pass the period-filtered, cleaned DataFrame
        risk_threshold_moderate=app_config.RISK_SCORE_MODERATE_THRESHOLD, # Sourced from app_config
        source_context=f"{module_log_prefix}/FlaggedPatientCasesReview"
    )

    if isinstance(df_flagged_patients, pd.DataFrame) and not df_flagged_patients.empty:
        patient_focus_output["flagged_patients_for_review_df"] = df_flagged_patients
        logger.info(f"({module_log_prefix}) Identified {len(df_flagged_patients)} patient cases flagged for clinical review.")
    else: # Handles both None and empty DataFrame return from get_patient_alerts_for_clinic
        note = "No specific patient cases were flagged for clinical review in this period based on current criteria."
        logger.info(f"({module_log_prefix}) {note}")
        patient_focus_output["processing_notes"].append(note)
        # Ensure flagged_patients_for_review_df is an empty DataFrame with expected schema if none found
        # The schema should match the output of get_patient_alerts_for_clinic
        # patient_focus_output["flagged_patients_for_review_df"] remains as initialized (empty DF with schema)

    logger.info(f"({module_log_prefix}) Clinic patient focus data preparation complete. Notes: {len(patient_focus_output['processing_notes'])}")
    return patient_focus_output
