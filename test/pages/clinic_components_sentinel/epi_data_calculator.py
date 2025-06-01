# sentinel_project_root/test/pages/clinic_components_sentinel/epi_data_calculator.py
# Calculates clinic-level epidemiological data for Sentinel Health Co-Pilot.

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List

# Standardized import block
try:
    from config import app_config
    from utils.core_data_processing import get_trend_data
except ImportError:
    import sys
    import os
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_for_utils = os.path.abspath(os.path.join(current_script_dir, os.pardir, os.pardir))
    if project_root_for_utils not in sys.path:
        sys.path.insert(0, project_root_for_utils)
    from config import app_config
    from utils.core_data_processing import get_trend_data

logger = logging.getLogger(__name__)

def calculate_clinic_epi_data(
    filtered_health_df_clinic_period: Optional[pd.DataFrame],
    reporting_period_str: str,
    selected_condition_for_demographics: str = "All Conditions (Aggregated)",
    top_n_symptoms_to_trend: int = 5
) -> Dict[str, Any]:
    """
    Calculates various epidemiological data sets for a clinic over a specified period.
    """
    module_log_prefix = "ClinicEpiDataCalculator"
    logger.info(f"({module_log_prefix}) Calculating clinic epi data. Period: {reporting_period_str}, Demo Cond: {selected_condition_for_demographics}")

    epi_data_output: Dict[str, Any] = {
        "reporting_period": reporting_period_str,
        "symptom_trends_weekly_top_n_df": None, # DF: week_start_date, symptom, count
        "key_test_positivity_trends": {},       # Dict: {test_display_name: pd.Series(pos_rate_pct)}
        "demographics_by_condition_data": None, # Dict: {age_df, gender_df, condition_analyzed}
        "referral_funnel_summary_df": None,     # DF: Stage, Count
        "calculation_notes": []
    }

    if not isinstance(filtered_health_df_clinic_period, pd.DataFrame) or filtered_health_df_clinic_period.empty:
        msg = "No health data provided for clinic epidemiological analysis."
        logger.warning(f"({module_log_prefix}) {msg}")
        epi_data_output["calculation_notes"].append(msg)
        return epi_data_output

    df_clinic_source = filtered_health_df_clinic_period.copy()

    # Critical: Ensure 'encounter_date' exists and is datetime
    if 'encounter_date' not in df_clinic_source.columns:
        msg = "'encounter_date' column missing, critical for epi calculations."
        logger.error(f"({module_log_prefix}) {msg}")
        epi_data_output["calculation_notes"].append(msg); return epi_data_output
    df_clinic_source['encounter_date'] = pd.to_datetime(df_clinic_source['encounter_date'], errors='coerce')
    df_clinic_source.dropna(subset=['encounter_date'], inplace=True)
    if df_clinic_source.empty:
        msg = "No valid encounter dates after cleaning for epi calculations."
        logger.warning(f"({module_log_prefix}) {msg}")
        epi_data_output["calculation_notes"].append(msg); return epi_data_output

    # Ensure other essential columns exist with safe defaults
    essential_cols_epi_calc = {
        'patient_id': {"default": "UnknownPID_Epi", "type": str},
        'patient_reported_symptoms': {"default": "", "type": str},
        'condition': {"default": "UnknownCondition_Epi", "type": str},
        'test_type': {"default": "UnknownTest_Epi", "type": str},
        'test_result': {"default": "UnknownResult_Epi", "type": str},
        'age': {"default": np.nan, "type": float},
        'gender': {"default": "Unknown", "type": str},
        'referral_status': {"default": "Unknown", "type": str},
        'referral_outcome': {"default": "Unknown", "type": str}, # Added for referral funnel
        'encounter_id': {"default": "UnknownEncounterID_Epi", "type": str}
    }
    common_na_strings_epi = ['', 'nan', 'None', 'N/A', '#N/A', 'np.nan', 'NaT', '<NA>', 'null']
    for col, config in essential_cols_epi_calc.items():
        if col not in df_clinic_source.columns:
            df_clinic_source[col] = config["default"]
        if config["type"] == float:
            df_clinic_source[col] = pd.to_numeric(df_clinic_source[col], errors='coerce').fillna(config["default"])
        elif config["type"] == str:
            df_clinic_source[col] = df_clinic_source[col].astype(str).str.strip().replace(common_na_strings_epi, config["default"], regex=False)
            df_clinic_source[col] = df_clinic_source[col].fillna(config["default"])


    # 1. Symptom Trends (Weekly Top N Symptoms)
    if df_clinic_source['patient_reported_symptoms'].notna().any() and \
       df_clinic_source['patient_reported_symptoms'].str.strip().astype(bool).any(): # Check if any non-empty strings
        
        symptoms_for_trend_df = df_clinic_source[['encounter_date', 'patient_reported_symptoms']].copy()
        # Robustly exclude common non-informative symptom entries
        non_informative_symptoms_lower = ["unknown", "n/a", "none", "", " ", "no symptoms", "asymptomatic"]
        symptoms_for_trend_df.dropna(subset=['patient_reported_symptoms'], inplace=True)
        symptoms_for_trend_df = symptoms_for_trend_df[
            ~symptoms_for_trend_df['patient_reported_symptoms'].str.lower().isin(non_informative_symptoms_lower)
        ]

        if not symptoms_for_trend_df.empty:
            # Explode multi-symptom strings, standardize, and remove blanks
            symptoms_exploded = symptoms_for_trend_df.assign(
                symptom=symptoms_for_trend_df['patient_reported_symptoms'].str.split(';')
            ).explode('symptom')
            symptoms_exploded['symptom'] = symptoms_exploded['symptom'].str.strip().str.title() # Standardize case
            symptoms_exploded.dropna(subset=['symptom'], inplace=True)
            symptoms_exploded = symptoms_exploded[symptoms_exploded['symptom'] != '']

            if not symptoms_exploded.empty:
                top_symptoms_list = symptoms_exploded['symptom'].value_counts().nlargest(top_n_symptoms_to_trend).index.tolist()
                df_top_symptoms_trend = symptoms_exploded[symptoms_exploded['symptom'].isin(top_symptoms_list)]

                if not df_top_symptoms_trend.empty:
                    weekly_symptom_counts = df_top_symptoms_trend.groupby(
                        [pd.Grouper(key='encounter_date', freq='W-Mon', label='left', closed='left'), 'symptom'] # Ensure consistent week start
                    ).size().reset_index(name='count')
                    weekly_symptom_counts.rename(columns={'encounter_date': 'week_start_date'}, inplace=True)
                    epi_data_output["symptom_trends_weekly_top_n_df"] = weekly_symptom_counts
                else: epi_data_output["calculation_notes"].append(f"Not enough data for top {top_n_symptoms_to_trend} symptoms trend.")
            else: epi_data_output["calculation_notes"].append("No valid individual symptoms after cleaning for trend.")
        else: epi_data_output["calculation_notes"].append("No actionable patient-reported symptoms data found for trends.")
    else: epi_data_output["calculation_notes"].append("'patient_reported_symptoms' column missing or empty, skipping symptom trends.")

    # 2. Test Positivity Rate Trends
    # Using KEY_TEST_TYPES_FOR_ANALYSIS from app_config to identify relevant tests
    positivity_trends_map = {}
    conclusive_results_filter = ~df_clinic_source.get('test_result', pd.Series(dtype=str)).isin(
        ["Pending", "Rejected Sample", "Unknown", "Indeterminate", "N/A", "", "UnknownResult_Epi"] # Include default
    )
    tests_for_pos_calc_df = df_clinic_source[conclusive_results_filter].copy()

    for test_key_orig, test_props_cfg in app_config.KEY_TEST_TYPES_FOR_ANALYSIS.items():
        test_display_name_cfg = test_props_cfg.get("display_name", test_key_orig)
        
        # Filter for the specific original test key
        specific_test_data_df = tests_for_pos_calc_df[tests_for_pos_calc_df['test_type'] == test_key_orig]
        if not specific_test_data_df.empty:
            specific_test_data_df['is_positive'] = (specific_test_data_df['test_result'] == 'Positive')
            
            weekly_pos_rate_series = get_trend_data(
                df=specific_test_data_df, value_col='is_positive', date_col='encounter_date',
                period='W-Mon', agg_func='mean', # Mean of boolean (0/1) gives proportion
                source_context=f"{module_log_prefix}/Positivity/{test_display_name_cfg}"
            )
            if isinstance(weekly_pos_rate_series, pd.Series) and not weekly_pos_rate_series.empty:
                positivity_trends_map[test_display_name_cfg] = (weekly_pos_rate_series * 100).round(1) # Convert to percentage
            else: epi_data_output["calculation_notes"].append(f"No aggregated weekly positivity data for {test_display_name_cfg}.")
        else: epi_data_output["calculation_notes"].append(f"No conclusive test data for {test_display_name_cfg} in period.")
    epi_data_output["key_test_positivity_trends"] = positivity_trends_map

    # 3. Demographic Breakdown for Selected Condition
    demo_output_for_cond = {"age_distribution_df": None, "gender_distribution_df": None, "condition_analyzed": selected_condition_for_demographics}
    df_for_demographics = df_clinic_source.copy()
    if selected_condition_for_demographics != "All Conditions (Aggregated)":
        # Use str.contains for partial matching if selected_condition is a general term
        df_for_demographics = df_clinic_source[
            df_clinic_source['condition'].str.contains(selected_condition_for_demographics, case=False, na=False)
        ]

    if not df_for_demographics.empty and 'patient_id' in df_for_demographics.columns:
        unique_patients_for_demog_df = df_for_demographics.drop_duplicates(subset=['patient_id'])
        if not unique_patients_for_demog_df.empty:
            # Age breakdown
            if 'age' in unique_patients_for_demog_df.columns and unique_patients_for_demog_df['age'].notna().any():
                age_bins_list = [0, 5, 15, 25, 35, 50, 65, np.inf] # More granular bins
                age_labels_list = ['0-4', '5-14', '15-24', '25-34', '35-49', '50-64', '65+']
                temp_age_df = unique_patients_for_demog_df.copy() # Avoid SettingWithCopyWarning
                temp_age_df['age_group'] = pd.cut(temp_age_df['age'], bins=age_bins_list, labels=age_labels_list, right=False)
                age_dist_df = temp_age_df['age_group'].value_counts().sort_index().reset_index()
                age_dist_df.columns = ['Age Group', 'Patient Count']
                demo_output_for_cond["age_distribution_df"] = age_dist_df
            
            # Gender breakdown (focus on Male/Female, group others)
            if 'gender' in unique_patients_for_demog_df.columns and unique_patients_for_demog_df['gender'].notna().any():
                temp_gender_df = unique_patients_for_demog_df.copy()
                # Normalize gender: map variations to Male, Female, or Other/Unknown
                gender_map = lambda g: "Male" if str(g).lower() in ['m', 'male'] else \
                                     "Female" if str(g).lower() in ['f', 'female'] else "Other/Unknown"
                temp_gender_df['gender_normalized'] = temp_gender_df['gender'].apply(gender_map)
                gender_dist_df = temp_gender_df['gender_normalized'].value_counts().reset_index()
                gender_dist_df.columns = ['Gender', 'Patient Count']
                demo_output_for_cond["gender_distribution_df"] = gender_dist_df
            
            epi_data_output["demographics_by_condition_data"] = demo_output_for_cond
        else: epi_data_output["calculation_notes"].append(f"No unique patients for condition '{selected_condition_for_demographics}' for demographics.")
    elif 'patient_id' not in df_clinic_source.columns:
        epi_data_output["calculation_notes"].append("'patient_id' missing for demographic breakdown.")
    else: epi_data_output["calculation_notes"].append(f"No patient data found for condition '{selected_condition_for_demographics}'.")


    # 4. Referral Funnel Analysis (Simplified Summary)
    if 'referral_status' in df_clinic_source.columns and 'encounter_id' in df_clinic_source.columns:
        # Consider referrals that were at least initiated or have a definitive status
        actionable_referral_statuses_lower = ['pending', 'completed', 'initiated', 'service provided', 'attended', 'missed appointment', 'declined', 'admitted']
        df_referrals_funnel = df_clinic_source[
            df_clinic_source.get('referral_status', pd.Series(dtype=str)).str.lower().isin(actionable_referral_statuses_lower)
        ].copy()

        if not df_referrals_funnel.empty:
            # Count unique referral processes (e.g., by encounter_id or a dedicated referral_id if available)
            # Using encounter_id assumes one primary referral per encounter for this summary
            total_referrals_in_period = df_referrals_funnel['encounter_id'].nunique()
            
            positively_concluded_outcomes_lower = ['completed', 'service provided', 'attended consult', 'attended followup', 'attended', 'admitted']
            # Check 'referral_outcome' for positive conclusions
            num_positively_concluded = 0
            if 'referral_outcome' in df_referrals_funnel.columns:
                num_positively_concluded = df_referrals_funnel[
                    df_referrals_funnel.get('referral_outcome', pd.Series(dtype=str)).str.lower().isin(positively_concluded_outcomes_lower)
                ]['encounter_id'].nunique()
            
            num_still_pending = df_referrals_funnel[
                df_referrals_funnel['referral_status'].str.lower() == 'pending'
            ]['encounter_id'].nunique()
            
            funnel_data = pd.DataFrame([
                {'Stage': 'Referrals Made/Active (Period)', 'Count': total_referrals_in_period},
                {'Stage': 'Concluded Positively (Outcome Recorded)', 'Count': num_positively_concluded},
                {'Stage': 'Still Pending (Status "Pending")', 'Count': num_still_pending},
            ])
            epi_data_output["referral_funnel_summary_df"] = funnel_data[funnel_data['Count'] > 0].reset_index(drop=True)
        else: epi_data_output["calculation_notes"].append("No actionable referral records found for funnel analysis.")
    else: epi_data_output["calculation_notes"].append("Referral status/encounter ID data missing for referral funnel.")
    
    logger.info(f"({module_log_prefix}) Clinic epi data calculation finished. Notes: {len(epi_data_output['calculation_notes'])}")
    return epi_data_output
