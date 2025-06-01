# sentinel_project_root/test/pages/clinic_components_sentinel/epi_data_calculator.py
# Calculates clinic-level epidemiological data for Sentinel Health Co-Pilot.

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List

# Standardized import block
try:
    from config import app_config
    from utils.core_data_processing import get_trend_data # For time-series trends
except ImportError:
    import sys
    import os
    # Assumes this file is in sentinel_project_root/test/pages/clinic_components_sentinel/
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_test_root_dir = os.path.abspath(os.path.join(current_script_dir, os.pardir, os.pardir))
    if project_test_root_dir not in sys.path:
        sys.path.insert(0, project_test_root_dir)
    from config import app_config
    from utils.core_data_processing import get_trend_data

logger = logging.getLogger(__name__)

def calculate_clinic_epi_data(
    filtered_health_df_clinic_period: Optional[pd.DataFrame],
    reporting_period_str: str,
    selected_condition_for_demographics: str = "All Conditions (Aggregated)", # Default for demographics tab
    top_n_symptoms_to_trend: int = 7 # Slightly increased default for more visibility
) -> Dict[str, Any]:
    """
    Calculates various epidemiological data sets for a clinic over a specified period.
    """
    module_log_prefix = "ClinicEpiDataCalc" # Shortened prefix
    logger.info(f"({module_log_prefix}) Calculating clinic epi data. Period: {reporting_period_str}, Demo Cond: {selected_condition_for_demographics}")

    epi_data_output: Dict[str, Any] = {
        "reporting_period": reporting_period_str,
        "symptom_trends_weekly_top_n_df": None, # DF: week_start_date, symptom, count
        "key_test_positivity_trends": {},       # Dict: {test_display_name: pd.Series(pos_rate_pct)}
        "demographics_by_condition_data": None, # Dict: {age_df, gender_df, condition_analyzed}
        "referral_funnel_summary_df": None,     # DF: Stage, Count
        "calculation_notes": []                 # List to store notes on data availability/issues
    }

    if not isinstance(filtered_health_df_clinic_period, pd.DataFrame) or filtered_health_df_clinic_period.empty:
        msg = "No health data provided for clinic epidemiological analysis. Calculations skipped."
        logger.warning(f"({module_log_prefix}) {msg}")
        epi_data_output["calculation_notes"].append(msg)
        return epi_data_output

    df_epi_src = filtered_health_df_clinic_period.copy() # Work on a copy

    # Critical: Ensure 'encounter_date' exists and is valid datetime
    if 'encounter_date' not in df_epi_src.columns:
        msg = "'encounter_date' column missing, critical for epidemiological calculations."
        logger.error(f"({module_log_prefix}) {msg}")
        epi_data_output["calculation_notes"].append(msg); return epi_data_output
        
    df_epi_src['encounter_date'] = pd.to_datetime(df_epi_src['encounter_date'], errors='coerce')
    df_epi_src.dropna(subset=['encounter_date'], inplace=True) # Remove rows where date conversion failed
    if df_epi_src.empty:
        msg = "No valid encounter dates found in data after cleaning for epi calculations."
        logger.warning(f"({module_log_prefix}) {msg}")
        epi_data_output["calculation_notes"].append(msg); return epi_data_output

    # Ensure other essential columns exist, adding with safe defaults if not
    essential_cols_config_epi_calc = { # Renamed for clarity
        'patient_id': {"default": "UnknownPID_EpiCalc", "type": str},
        'patient_reported_symptoms': {"default": "", "type": str}, # Default to empty string for string ops
        'condition': {"default": "UnknownCondition_EpiCalc", "type": str},
        'test_type': {"default": "UnknownTest_EpiCalc", "type": str},
        'test_result': {"default": "UnknownResult_EpiCalc", "type": str},
        'age': {"default": np.nan, "type": float},
        'gender': {"default": "Unknown", "type": str},
        'referral_status': {"default": "Unknown", "type": str},
        'referral_outcome': {"default": "Unknown", "type": str}, # Added for referral funnel
        'encounter_id': {"default": "UnknownEncID_EpiCalc", "type": str}
    }
    common_na_strings_epi_calc = ['', 'nan', 'None', 'N/A', '#N/A', 'np.nan', 'NaT', '<NA>', 'null']

    for col, config_item in essential_cols_config_epi_calc.items():
        if col not in df_epi_src.columns:
            df_epi_src[col] = config_item["default"]
        # Type coercion and standardized NA handling
        if config_item["type"] == float:
            df_epi_src[col] = pd.to_numeric(df_epi_src[col], errors='coerce').fillna(config_item["default"])
        elif config_item["type"] == str:
            df_epi_src[col] = df_epi_src[col].astype(str).str.strip().replace(common_na_strings_epi_calc, config_item["default"], regex=False)
            df_epi_src[col] = df_epi_src[col].fillna(config_item["default"])


    # 1. Symptom Trends (Weekly Top N Symptoms)
    if 'patient_reported_symptoms' in df_epi_src.columns and \
       df_epi_src['patient_reported_symptoms'].str.strip().astype(bool).any(): # Check if any non-empty strings
        
        df_symptoms_for_trend_analysis = df_epi_src[['encounter_date', 'patient_reported_symptoms']].copy()
        non_informative_symptoms_list = ["unknown", "n/a", "none", "", " ", "no symptoms", "asymptomatic", "well"] # Added "well"
        df_symptoms_for_trend_analysis.dropna(subset=['patient_reported_symptoms'], inplace=True)
        df_symptoms_for_trend_analysis = df_symptoms_for_trend_analysis[
            ~df_symptoms_for_trend_analysis['patient_reported_symptoms'].str.lower().isin(non_informative_symptoms_list)
        ]

        if not df_symptoms_for_trend_analysis.empty:
            symptoms_exploded_df = df_symptoms_for_trend_analysis.assign(
                symptom=df_symptoms_for_trend_analysis['patient_reported_symptoms'].str.split(';') # Assumes semicolon delimiter
            ).explode('symptom')
            symptoms_exploded_df['symptom'] = symptoms_exploded_df['symptom'].str.strip().str.title() # Standardize case
            symptoms_exploded_df.dropna(subset=['symptom'], inplace=True)
            symptoms_exploded_df = symptoms_exploded_df[symptoms_exploded_df['symptom'] != ''] # Remove empty strings after split

            if not symptoms_exploded_df.empty:
                list_top_symptoms = symptoms_exploded_df['symptom'].value_counts().nlargest(top_n_symptoms_to_trend).index.tolist()
                df_top_symptoms_for_trend = symptoms_exploded_df[symptoms_exploded_df['symptom'].isin(list_top_symptoms)]

                if not df_top_symptoms_for_trend.empty:
                    df_weekly_symptom_counts = df_top_symptoms_for_trend.groupby(
                        [pd.Grouper(key='encounter_date', freq='W-Mon', label='left', closed='left'), 'symptom'] # Consistent week start
                    ).size().reset_index(name='count')
                    df_weekly_symptom_counts.rename(columns={'encounter_date': 'week_start_date'}, inplace=True)
                    epi_data_output["symptom_trends_weekly_top_n_df"] = df_weekly_symptom_counts
                else: epi_data_output["calculation_notes"].append(f"Not enough diverse symptom data to trend top {top_n_symptoms_to_trend} symptoms.")
            else: epi_data_output["calculation_notes"].append("No valid individual symptoms after cleaning/exploding for trend analysis.")
        else: epi_data_output["calculation_notes"].append("No actionable patient-reported symptoms data found for trends after filtering non-informative entries.")
    else: epi_data_output["calculation_notes"].append("'patient_reported_symptoms' column missing or empty, skipping symptom trends.")

    # 2. Test Positivity Rate Trends
    positivity_trends_result_map = {}
    # Define conclusive results (excluding pending, rejected, etc.)
    conclusive_results_mask_epi = ~df_epi_src.get('test_result', pd.Series(dtype=str)).isin(
        ["Pending", "Rejected Sample", "Unknown", "Indeterminate", "N/A", "", "UnknownResult_EpiCalc"] 
    )
    df_conclusive_tests_epi = df_epi_src[conclusive_results_mask_epi].copy()

    for test_original_key, test_config_props in app_config.KEY_TEST_TYPES_FOR_ANALYSIS.items():
        test_display_name_for_output = test_config_props.get("display_name", test_original_key)
        
        df_specific_test_data = df_conclusive_tests_epi[df_conclusive_tests_epi['test_type'] == test_original_key]
        if not df_specific_test_data.empty:
            df_specific_test_data['is_positive_result_flag'] = (df_specific_test_data['test_result'] == 'Positive')
            
            series_weekly_pos_rate = get_trend_data(
                df=df_specific_test_data, value_col='is_positive_result_flag', date_col='encounter_date',
                period='W-Mon', agg_func='mean', # Mean of boolean (0/1) gives proportion
                source_context=f"{module_log_prefix}/PositivityTrend/{test_display_name_for_output}"
            )
            if isinstance(series_weekly_pos_rate, pd.Series) and not series_weekly_pos_rate.empty:
                positivity_trends_result_map[test_display_name_for_output] = (series_weekly_pos_rate * 100).round(1) # Store as percentage
            else: epi_data_output["calculation_notes"].append(f"No aggregated weekly positivity trend data for {test_display_name_for_output}.")
        else: epi_data_output["calculation_notes"].append(f"No conclusive test data found for {test_display_name_for_output} in the period.")
    epi_data_output["key_test_positivity_trends"] = positivity_trends_result_map

    # 3. Demographic Breakdown for Selected Condition
    demographics_result_dict = {"age_distribution_df": None, "gender_distribution_df": None, "condition_analyzed": selected_condition_for_demographics}
    df_for_demog_src = df_epi_src.copy() # Use the fully prepped df_epi_src
    if selected_condition_for_demographics != "All Conditions (Aggregated)":
        # Use str.contains for broader matching if selected_condition_for_demographics is a general term
        df_for_demog_src = df_epi_src[
            df_epi_src['condition'].str.contains(selected_condition_for_demographics, case=False, na=False)
        ]

    if not df_for_demog_src.empty and 'patient_id' in df_for_demog_src.columns:
        df_unique_patients_for_demog = df_for_demog_src.drop_duplicates(subset=['patient_id'])
        if not df_unique_patients_for_demog.empty:
            # Age breakdown
            if 'age' in df_unique_patients_for_demog.columns and df_unique_patients_for_demog['age'].notna().any():
                age_bins_list_demog = [0, 5, 15, 25, 35, 50, 65, np.inf] # Standard age bins
                age_labels_list_demog = ['0-4', '5-14', '15-24', '25-34', '35-49', '50-64', '65+']
                temp_age_df_demog = df_unique_patients_for_demog.copy() 
                temp_age_df_demog['age_group_display'] = pd.cut(temp_age_df_demog['age'], bins=age_bins_list_demog, labels=age_labels_list_demog, right=False)
                df_age_dist = temp_age_df_demog['age_group_display'].value_counts().sort_index().reset_index()
                df_age_dist.columns = ['Age Group', 'Patient Count']
                demographics_result_dict["age_distribution_df"] = df_age_dist
            
            # Gender breakdown
            if 'gender' in df_unique_patients_for_demog.columns and df_unique_patients_for_demog['gender'].notna().any():
                temp_gender_df_demog = df_unique_patients_for_demog.copy()
                gender_map_func_demog = lambda g_str: "Male" if str(g_str).lower() in ['m', 'male'] else \
                                                     "Female" if str(g_str).lower() in ['f', 'female'] else "Other/Unknown"
                temp_gender_df_demog['gender_normalized_display'] = temp_gender_df_demog['gender'].apply(gender_map_func_demog)
                df_gender_dist = temp_gender_df_demog[
                    temp_gender_df_demog['gender_normalized_display'].isin(["Male", "Female"]) # Focus on Male/Female
                ]['gender_normalized_display'].value_counts().reset_index()
                df_gender_dist.columns = ['Gender', 'Patient Count']
                demographics_result_dict["gender_distribution_df"] = df_gender_dist
            
            epi_data_output["demographics_by_condition_data"] = demographics_result_dict
        else: epi_data_output["calculation_notes"].append(f"No unique patients found for condition '{selected_condition_for_demographics}' for demographic breakdown.")
    elif 'patient_id' not in df_epi_src.columns: # Should have been caught by prep if critical
        epi_data_output["calculation_notes"].append("'patient_id' column missing for demographic breakdown.")
    else: epi_data_output["calculation_notes"].append(f"No patient data found for condition '{selected_condition_for_demographics}' after initial filtering.")

    # 4. Referral Funnel Analysis
    if 'referral_status' in df_epi_src.columns and 'encounter_id' in df_epi_src.columns:
        actionable_referral_statuses_list_lower = ['pending', 'completed', 'initiated', 'service provided', 'attended', 'missed appointment', 'declined', 'admitted']
        df_referrals_for_funnel_analysis = df_epi_src[
            df_epi_src.get('referral_status', pd.Series(dtype=str)).str.lower().isin(actionable_referral_statuses_list_lower)
        ].copy()

        if not df_referrals_for_funnel_analysis.empty:
            # Count unique referral processes (e.g., by encounter_id)
            total_active_referrals = df_referrals_for_funnel_analysis['encounter_id'].nunique()
            
            positively_concluded_outcomes_list_lower = ['completed', 'service provided', 'attended consult', 'attended followup', 'attended', 'admitted']
            count_positively_concluded = 0
            if 'referral_outcome' in df_referrals_for_funnel_analysis.columns: # Check if outcome column exists
                count_positively_concluded = df_referrals_for_funnel_analysis[
                    df_referrals_for_funnel_analysis.get('referral_outcome', pd.Series(dtype=str)).str.lower().isin(positively_concluded_outcomes_list_lower)
                ]['encounter_id'].nunique()
            
            count_still_pending = df_referrals_for_funnel_analysis[
                df_referrals_for_funnel_analysis['referral_status'].str.lower() == 'pending'
            ]['encounter_id'].nunique()
            
            funnel_summary_data = pd.DataFrame([
                {'Stage': 'Referrals Made/Active (in Period)', 'Count': total_active_referrals},
                {'Stage': 'Concluded Positively (Outcome Known)', 'Count': count_positively_concluded},
                {'Stage': 'Still Pending (Status "Pending")', 'Count': count_still_pending},
            ])
            # Only show stages with non-zero counts for cleaner display
            epi_data_output["referral_funnel_summary_df"] = funnel_summary_data[funnel_summary_data['Count'] > 0].reset_index(drop=True)
        else: epi_data_output["calculation_notes"].append("No actionable referral records found for funnel analysis in the period.")
    else: epi_data_output["calculation_notes"].append("Referral status or encounter ID data missing for referral funnel analysis.")
    
    logger.info(f"({module_log_prefix}) Clinic epi data calculation finished. Notes recorded: {len(epi_data_output['calculation_notes'])}")
    return epi_data_output
