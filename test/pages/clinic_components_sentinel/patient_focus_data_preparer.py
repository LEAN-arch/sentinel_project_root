# sentinel_project_root/test/pages/clinic_components_sentinel/patient_focus_data_preparer.py
# Part of "Sentinel Health Co-Pilot" - LMIC Edge-First System
# This module prepares data related to clinic patient load and identifies
# flagged patient cases for clinical review.
# Refactored from the original clinic_components/patient_focus_tab.py.
# Output is structured data for display on the Clinic Management Console (Tier 2).

import pandas as pd
import numpy as np # Not heavily used here but often a pandas companion
import logging
from typing import Dict, Any, Optional, List

# Assuming app_config and core_data_processing utilities are accessible
try:
    from config import app_config
    from utils.core_data_processing import get_patient_alerts_for_clinic # Key utility for flagged patients
except ImportError:
    import sys, os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_test_root = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
    if project_test_root not in sys.path: sys.path.insert(0, project_test_root)
    from config import app_config
    from utils.core_data_processing import get_patient_alerts_for_clinic

logger = logging.getLogger(__name__)

def prepare_clinic_patient_focus_data(
    filtered_health_df_clinic_period: Optional[pd.DataFrame], # Health data for the clinic and period
    reporting_period_str: str,
    patient_load_time_aggregation: str = 'D', # 'D' for daily, 'W-Mon' for weekly patient load
    # Thresholds for patient flagging can be sourced from app_config inside get_patient_alerts_for_clinic
) -> Dict[str, Any]:
    """
    Prepares data for patient load analysis and a list of flagged patient cases for review.

    Args:
        filtered_health_df_clinic_period: DataFrame of health records for the clinic and selected period.
                                          Expected columns: 'encounter_date', 'condition', 'patient_id',
                                          'ai_risk_score', etc. (as needed by get_patient_alerts_for_clinic).
        reporting_period_str: String describing the reporting period.
        patient_load_time_aggregation: Aggregation period for patient load ('D', 'W-Mon').

    Returns:
        Dict[str, Any]: A dictionary containing:
            "reporting_period": str,
            "patient_load_by_key_condition_df": pd.DataFrame (period_start_date, condition, unique_patients_count),
            "flagged_patients_for_review_df": pd.DataFrame (from get_patient_alerts_for_clinic),
            "processing_notes": List[str]
    """
    module_source_context = "ClinicPatientFocusPreparer"
    logger.info(f"({module_source_context}) Preparing patient focus data for period: {reporting_period_str}")

    patient_focus_data_output: Dict[str, Any] = {
        "reporting_period": reporting_period_str,
        "patient_load_by_key_condition_df": None,
        "flagged_patients_for_review_df": None,
        "processing_notes": []
    }

    if filtered_health_df_clinic_period is None or filtered_health_df_clinic_period.empty:
        note = "No health data provided for patient focus data preparation."
        logger.warning(f"({module_source_context}) {note}")
        patient_focus_data_output["processing_notes"].append(note)
        return patient_focus_data_output

    df_clinic_focus = filtered_health_df_clinic_period.copy()

    # Ensure essential columns are present
    essential_cols_focus = {'encounter_date': pd.NaT, 'patient_id': "UnknownPID", 'condition': "UnknownCondition"}
    for col, default_val in essential_cols_focus.items():
        if col not in df_clinic_focus.columns:
            df_clinic_focus[col] = default_val
            note = f"Essential column '{col}' missing for patient focus; added with default."
            logger.warning(f"({module_source_context}) {note}")
            patient_focus_data_output["processing_notes"].append(note)
        elif 'date' in col:
            df_clinic_focus[col] = pd.to_datetime(df_clinic_focus[col], errors='coerce')
    
    df_clinic_focus.dropna(subset=['encounter_date', 'patient_id'], inplace=True) # Need these for any analysis

    if df_clinic_focus.empty:
        note = "No valid records after cleaning encounter_date/patient_id for patient focus data."
        logger.warning(f"({module_source_context}) {note}")
        patient_focus_data_output["processing_notes"].append(note)
        return patient_focus_data_output


    # 1. Patient Load by Key Condition
    #    Uses KEY_CONDITIONS_FOR_ACTION from app_config for relevance.
    conditions_for_load_chart = app_config.KEY_CONDITIONS_FOR_ACTION
    
    if 'condition' in df_clinic_focus.columns:
        # Filter for encounters matching key conditions and valid (non-generic) patient IDs
        df_for_load_calc = df_clinic_focus[
            df_clinic_focus['condition'].str.contains('|'.join(conditions_for_load_chart), case=False, na=False) &
            (~df_clinic_focus['patient_id'].astype(str).str.lower().isin(['unknown', 'n/a', '']))
        ].copy()

        if not df_for_load_calc.empty:
            # Group by specified time period and condition, counting unique patients
            load_summary_df = df_for_load_calc.groupby(
                [pd.Grouper(key='encounter_date', freq=patient_load_time_aggregation), 'condition']
            )['patient_id'].nunique().reset_index()
            
            load_summary_df.rename(
                columns={'encounter_date': 'period_start_date', 'patient_id': 'unique_patients_count'},
                inplace=True
            )
            patient_focus_data_output["patient_load_by_key_condition_df"] = load_summary_df
            if load_summary_df.empty:
                patient_focus_data_output["processing_notes"].append("No patient load data aggregated for key conditions in the period.")
        else:
            patient_focus_data_output["processing_notes"].append("No encounters found for key conditions for patient load analysis.")
    else:
        patient_focus_data_output["processing_notes"].append("'condition' column missing for patient load analysis.")


    # 2. Flagged Patient Cases for Clinical Review
    #    Utilizes the robust get_patient_alerts_for_clinic from core_data_processing.
    #    This function already incorporates AI scores, critical vitals, and other rules.
    flagged_patients_output_df = get_patient_alerts_for_clinic(
        health_df_period=df_clinic_focus, # Use the period-filtered, cleaned DataFrame
        risk_threshold_moderate=app_config.RISK_SCORE_MODERATE_THRESHOLD, # From Sentinel config
        source_context=f"{module_source_context}/FlaggedCases"
    ) # This function is expected to return a DataFrame

    if flagged_patients_output_df is not None and not flagged_patients_output_df.empty:
        patient_focus_data_output["flagged_patients_for_review_df"] = flagged_patients_output_df
        logger.info(f"({module_source_context}) Generated {len(flagged_patients_output_df)} flagged patient cases for review.")
    else:
        note = "No specific patient cases flagged for clinical review in this period based on current criteria."
        logger.info(f"({module_source_context}) {note}")
        patient_focus_data_output["processing_notes"].append(note)
        # Initialize with empty DF if none found, ensures consistent output key
        # The columns should match the output of get_patient_alerts_for_clinic
        # Example key columns (actual list depends on get_patient_alerts_for_clinic refactor):
        expected_flagged_cols = ['patient_id', 'encounter_date', 'condition', 'Alert Reason', 'Priority Score', 'ai_risk_score']
        patient_focus_data_output["flagged_patients_for_review_df"] = pd.DataFrame(columns=expected_flagged_cols)


    logger.info(f"({module_source_context}) Clinic patient focus data preparation complete. Notes: {len(patient_focus_data_output['processing_notes'])}")
    return patient_focus_data_output
