# sentinel_project_root/test/pages/clinic_components_sentinel/patient_focus_data_preparer.py
# Prepares data for clinic patient load and flagged patient cases for Sentinel.

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List

# Standardized import block
try:
    from config import app_config
    from utils.core_data_processing import get_patient_alerts_for_clinic # This is the key utility now
except ImportError:
    import sys
    import os
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_for_utils = os.path.abspath(os.path.join(current_script_dir, os.pardir, os.pardir))
    if project_root_for_utils not in sys.path:
        sys.path.insert(0, project_root_for_utils)
    from config import app_config
    from utils.core_data_processing import get_patient_alerts_for_clinic

logger = logging.getLogger(__name__)

def prepare_clinic_patient_focus_data(
    filtered_health_df_clinic_period: Optional[pd.DataFrame],
    reporting_period_str: str,
    patient_load_time_aggregation: str = 'W-Mon', # Default to weekly load for clinic overview
) -> Dict[str, Any]:
    """
    Prepares data for patient load analysis and a list of flagged patient cases for review.

    Args:
        filtered_health_df_clinic_period: DataFrame of health records for clinic and period.
        reporting_period_str: String describing the reporting period.
        patient_load_time_aggregation: Aggregation period for patient load ('D', 'W-Mon').

    Returns:
        Dictionary containing patient load DataFrame, flagged patients DataFrame, and notes.
    """
    module_log_prefix = "ClinicPatientFocusPreparer"
    logger.info(f"({module_log_prefix}) Preparing patient focus data for period: {reporting_period_str}, Agg: {patient_load_time_aggregation}")

    patient_focus_output: Dict[str, Any] = {
        "reporting_period": reporting_period_str,
        "patient_load_by_key_condition_df": None, # DF: period_start_date, condition, unique_patients_count
        "flagged_patients_for_review_df": None,   # DF from get_patient_alerts_for_clinic
        "processing_notes": []
    }

    if not isinstance(filtered_health_df_clinic_period, pd.DataFrame) or filtered_health_df_clinic_period.empty:
        note = "No health data provided for patient focus data preparation."
        logger.warning(f"({module_log_prefix}) {note}")
        patient_focus_output["processing_notes"].append(note)
        # Initialize with empty DFs for consistent return structure
        patient_focus_output["patient_load_by_key_condition_df"] = pd.DataFrame(columns=['period_start_date', 'condition', 'unique_patients_count'])
        patient_focus_output["flagged_patients_for_review_df"] = pd.DataFrame(columns=['patient_id', 'encounter_date', 'Alert Reason', 'Priority Score'])
        return patient_focus_output

    df_focus_source = filtered_health_df_clinic_period.copy()

    # Ensure essential columns for this module's logic are present
    essential_cols = {
        'encounter_date': {"default": pd.NaT, "type": "datetime"},
        'patient_id': {"default": "UnknownPID_Focus", "type": str},
        'condition': {"default": "UnknownCondition_Focus", "type": str}
    }
    common_na_strings_focus = ['', 'nan', 'None', 'N/A', '#N/A', 'np.nan', 'NaT', '<NA>', 'null']

    for col, config in essential_cols.items():
        if col not in df_focus_source.columns:
            df_focus_source[col] = config["default"]
            note = f"Essential column '{col}' missing for patient focus, added with default."
            logger.warning(f"({module_log_prefix}) {note}")
            patient_focus_output["processing_notes"].append(note)
        
        if config["type"] == "datetime":
            df_focus_source[col] = pd.to_datetime(df_focus_source[col], errors='coerce')
        elif config["type"] == str: # Ensure string columns are cleaned
             df_focus_source[col] = df_focus_source[col].astype(str).str.strip().replace(common_na_strings_focus, config["default"], regex=False)
             df_focus_source[col] = df_focus_source[col].fillna(config["default"])

    # Drop rows if critical identifiers are still missing after cleaning
    df_focus_source.dropna(subset=['encounter_date', 'patient_id'], inplace=True)
    if df_focus_source.empty:
        note = "No valid records after cleaning encounter_date/patient_id for patient focus data."
        logger.warning(f"({module_log_prefix}) {note}")
        patient_focus_output["processing_notes"].append(note)
        patient_focus_output["patient_load_by_key_condition_df"] = pd.DataFrame(columns=['period_start_date', 'condition', 'unique_patients_count'])
        patient_focus_output["flagged_patients_for_review_df"] = pd.DataFrame(columns=['patient_id', 'encounter_date', 'Alert Reason', 'Priority Score'])
        return patient_focus_output

    # 1. Patient Load by Key Condition
    key_conditions_for_load = app_config.KEY_CONDITIONS_FOR_ACTION
    
    if 'condition' in df_focus_source.columns and key_conditions_for_load:
        # Filter for encounters matching key conditions and valid patient IDs
        # Use a regex pattern for str.contains for more flexible matching of conditions
        condition_pattern = '|'.join([f"(?:^|[^a-zA-Z]){cond}(?:$|[^a-zA-Z])" for cond in key_conditions_for_load]) # Whole word match-ish
        
        df_load_analysis = df_focus_source[
            df_focus_source['condition'].str.contains(condition_pattern, case=False, na=False) &
            (~df_focus_source['patient_id'].isin(["UnknownPID_Focus"])) # Exclude generic PIDs
        ].copy()

        if not df_load_analysis.empty:
            load_grouped = df_load_analysis.groupby(
                [pd.Grouper(key='encounter_date', freq=patient_load_time_aggregation, label='left', closed='left'), 'condition']
            )['patient_id'].nunique().reset_index()
            
            load_grouped.rename(
                columns={'encounter_date': 'period_start_date', 'patient_id': 'unique_patients_count'},
                inplace=True
            )
            patient_focus_output["patient_load_by_key_condition_df"] = load_grouped
            if load_grouped.empty:
                patient_focus_output["processing_notes"].append("No patient load data aggregated for key conditions in the period (empty after group).")
        else:
            patient_focus_output["processing_notes"].append("No encounters found matching key conditions for patient load analysis.")
            patient_focus_output["patient_load_by_key_condition_df"] = pd.DataFrame(columns=['period_start_date', 'condition', 'unique_patients_count'])
    else:
        patient_focus_output["processing_notes"].append("'condition' column missing or no KEY_CONDITIONS_FOR_ACTION in config for patient load analysis.")
        patient_focus_output["patient_load_by_key_condition_df"] = pd.DataFrame(columns=['period_start_date', 'condition', 'unique_patients_count'])


    # 2. Flagged Patient Cases for Clinical Review
    # This now fully relies on the get_patient_alerts_for_clinic utility.
    # That utility should handle all complex logic of AI scores, vitals, etc.
    flagged_patients_df = get_patient_alerts_for_clinic(
        health_df_period=df_focus_source, # Pass the cleaned, period-filtered DataFrame
        risk_threshold_moderate=app_config.RISK_SCORE_MODERATE_THRESHOLD,
        source_context=f"{module_log_prefix}/FlaggedPatientCases"
    )

    if isinstance(flagged_patients_df, pd.DataFrame) and not flagged_patients_df.empty:
        patient_focus_output["flagged_patients_for_review_df"] = flagged_patients_df
        logger.info(f"({module_log_prefix}) {len(flagged_patients_df)} patient cases flagged for clinical review.")
    else:
        note = "No specific patient cases flagged for clinical review in this period based on criteria."
        logger.info(f"({module_log_prefix}) {note}")
        patient_focus_output["processing_notes"].append(note)
        # Ensure consistent return type: an empty DataFrame with expected columns
        expected_flagged_cols = ['patient_id', 'encounter_date', 'condition', 'Alert Reason', 'Priority Score', 'ai_risk_score', 'age', 'gender', 'zone_id', 'referred_to_facility_id']
        patient_focus_output["flagged_patients_for_review_df"] = pd.DataFrame(columns=expected_flagged_cols)

    logger.info(f"({module_log_prefix}) Clinic patient focus data preparation complete. Notes: {len(patient_focus_output['processing_notes'])}")
    return patient_focus_output
