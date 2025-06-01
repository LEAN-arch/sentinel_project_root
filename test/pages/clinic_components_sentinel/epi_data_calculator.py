# sentinel_project_root/test/pages/clinic_components_sentinel/epi_data_calculator.py
# Part of "Sentinel Health Co-Pilot" - LMIC Edge-First System
# This module calculates clinic-level epidemiological data including symptom trends,
# test positivity trends, demographic breakdowns for conditions, and referral patterns.
# Refactored from the original clinic_components/epi_module.py.
# Output is structured data for display on the Clinic Management Console (Tier 2).

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple

# Assuming app_config and core_data_processing.get_trend_data are accessible
try:
    from config import app_config
    from utils.core_data_processing import get_trend_data
except ImportError:
    import sys, os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_test_root = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
    if project_test_root not in sys.path: sys.path.insert(0, project_test_root)
    from config import app_config
    from utils.core_data_processing import get_trend_data

logger = logging.getLogger(__name__)

def calculate_clinic_epi_data(
    filtered_health_df_clinic_period: Optional[pd.DataFrame],
    reporting_period_str: str,
    selected_condition_for_demographics: str = "All Conditions (Aggregated)", # Default for demographics
    top_n_symptoms_to_trend: int = 5 # For symptom trends
) -> Dict[str, Any]:
    """
    Calculates various epidemiological data sets for a clinic over a specified period.

    Args:
        filtered_health_df_clinic_period: DataFrame of health records filtered for the
                                          clinic and the desired reporting period.
                                          Expected columns include: 'encounter_date', 'patient_id',
                                          'patient_reported_symptoms', 'condition', 'test_type',
                                          'test_result', 'age', 'gender', 'referral_status',
                                          'referral_outcome', 'encounter_id'.
        reporting_period_str: A string describing the reporting period for context.
        selected_condition_for_demographics: The condition for which to generate detailed
                                             demographic breakdowns.
        top_n_symptoms_to_trend: How many of the most frequent symptoms to generate trends for.


    Returns:
        Dict[str, Any]: A dictionary containing structured epidemiological data sets.
    """
    module_source_context = "ClinicEpiDataCalculator"
    logger.info(f"({module_source_context}) Calculating clinic epi data. Period: {reporting_period_str}, DemoCond: {selected_condition_for_demographics}")

    epi_data_sets: Dict[str, Any] = {
        "reporting_period": reporting_period_str,
        "symptom_trends_weekly_top_n_df": None,       # DF: week_start_date, symptom, count
        "key_test_positivity_trends": {},             # Dict: {test_display_name: pd.Series(pos_rate)}
        "demographics_by_condition_data": None,       # Dict: {age_df: pd.DataFrame, gender_df: pd.DataFrame}
        "referral_funnel_summary_df": None,           # DF: Stage, Count
        "calculation_notes": []                       # List to store notes on data availability/issues
    }

    if filtered_health_df_clinic_period is None or filtered_health_df_clinic_period.empty:
        msg = "No health data provided for clinic epidemiological analysis."
        logger.warning(f"({module_source_context}) {msg}")
        epi_data_sets["calculation_notes"].append(msg)
        return epi_data_sets

    df_clinic_epi = filtered_health_df_clinic_period.copy() # Work on a copy

    # Ensure essential columns exist and 'encounter_date' is datetime
    if 'encounter_date' not in df_clinic_epi.columns:
        msg = "'encounter_date' column is missing, which is critical for most epidemiological calculations."
        logger.error(f"({module_source_context}) {msg}")
        epi_data_sets["calculation_notes"].append(msg)
        return epi_data_sets # Cannot proceed with most calculations
    df_clinic_epi['encounter_date'] = pd.to_datetime(df_clinic_epi['encounter_date'], errors='coerce')
    df_clinic_epi.dropna(subset=['encounter_date'], inplace=True)
    
    # Basic check for other important columns, log if missing but proceed where possible
    min_cols_for_calc = ['patient_id', 'patient_reported_symptoms', 'condition', 'test_type', 'test_result', 'age', 'gender', 'referral_status', 'encounter_id']
    for col in min_cols_for_calc:
        if col not in df_clinic_epi.columns:
            df_clinic_epi[col] = np.nan if col in ['age'] else "Unknown" # Add as empty/default
            note = f"Column '{col}' missing from input data; related calculations may be affected."
            logger.warning(f"({module_source_context}) {note}")
            epi_data_sets["calculation_notes"].append(note)


    # 1. Symptom Trends (Weekly Top N Symptoms)
    if 'patient_reported_symptoms' in df_clinic_epi.columns and df_clinic_epi['patient_reported_symptoms'].notna().any():
        symptoms_df_epi = df_clinic_epi[['encounter_date', 'patient_reported_symptoms']].copy()
        symptoms_df_epi.dropna(subset=['patient_reported_symptoms'], inplace=True)
        # Exclude common non-informative/blank entries more robustly
        non_info_symptoms = ["unknown", "n/a", "none", "", " ", "nan"]
        symptoms_df_epi = symptoms_df_epi[~symptoms_df_epi['patient_reported_symptoms'].str.lower().isin(non_info_symptoms)]

        if not symptoms_df_epi.empty:
            symptoms_exploded_epi = symptoms_df_epi.assign(
                symptom=symptoms_df_epi['patient_reported_symptoms'].str.split(';')
            ).explode('symptom')
            symptoms_exploded_epi['symptom'] = symptoms_exploded_epi['symptom'].str.strip().str.title() # Standardize
            symptoms_exploded_epi.dropna(subset=['symptom'], inplace=True)
            symptoms_exploded_epi = symptoms_exploded_epi[symptoms_exploded_epi['symptom'] != ''] # Remove empty strings

            if not symptoms_exploded_epi.empty:
                overall_top_symptoms = symptoms_exploded_epi['symptom'].value_counts().nlargest(top_n_symptoms_to_trend).index.tolist()
                symptoms_for_trend_calc = symptoms_exploded_epi[symptoms_exploded_epi['symptom'].isin(overall_top_symptoms)]

                if not symptoms_for_trend_calc.empty:
                    # Using Grouper for weekly counts per symptom
                    weekly_symptom_counts_df = symptoms_for_trend_calc.groupby(
                        [pd.Grouper(key='encounter_date', freq='W-Mon'), 'symptom']
                    ).size().reset_index(name='count')
                    weekly_symptom_counts_df.rename(columns={'encounter_date': 'week_start_date'}, inplace=True)
                    epi_data_sets["symptom_trends_weekly_top_n_df"] = weekly_symptom_counts_df
                else: epi_data_sets["calculation_notes"].append(f"Not enough distinct symptom data to trend top {top_n_symptoms_to_trend} symptoms.")
            else: epi_data_sets["calculation_notes"].append("No valid symptom data found after cleaning for trend analysis.")
        else: epi_data_sets["calculation_notes"].append("No actionable patient reported symptoms data found.")
    else: epi_data_sets["calculation_notes"].append("'patient_reported_symptoms' column missing for symptom trend analysis.")

    # 2. Test Positivity Rate Trends (Key Tests from app_config)
    # Focus on tests relevant for clinic-level epi surveillance, e.g., Malaria, HIV, TB
    key_tests_for_positivity = {
        "RDT-Malaria": {"display_name": app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get("RDT-Malaria",{}).get("display_name","Malaria RDT"),
                          "target_rate": app_config.TARGET_MALARIA_POSITIVITY_RATE},
        # Add other tests if configured and relevant e.g., HIV-Rapid
        "HIV-Rapid": {"display_name": app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get("HIV-Rapid",{}).get("display_name","HIV Rapid"),
                        "target_rate": 2.0} # Example target
    }
    calculated_positivity_trends = {}
    if 'test_type' in df_clinic_epi.columns and 'test_result' in df_clinic_epi.columns:
        for test_orig_key, test_info in key_tests_for_positivity.items():
            specific_test_df = df_clinic_epi[
                (df_clinic_epi['test_type'] == test_orig_key) &
                (~df_clinic_epi.get('test_result', pd.Series(dtype=str)).isin(
                    ['Pending', 'Rejected Sample', 'Unknown', 'Indeterminate', 'N/A', '', 'nan']))
            ].copy()
            if not specific_test_df.empty:
                specific_test_df['is_positive_result'] = (specific_test_df['test_result'] == 'Positive')
                weekly_pos_rate = get_trend_data(
                    df=specific_test_df, value_col='is_positive_result', date_col='encounter_date',
                    period='W-Mon', agg_func='mean', # Mean of booleans gives proportion
                    source_context=f"{module_source_context}/Positivity/{test_info['display_name']}"
                )
                if weekly_pos_rate is not None and not weekly_pos_rate.empty:
                    calculated_positivity_trends[test_info['display_name']] = (weekly_pos_rate * 100) # Convert to percentage
                else: epi_data_sets["calculation_notes"].append(f"No aggregated weekly positivity data for {test_info['display_name']}.")
            else: epi_data_sets["calculation_notes"].append(f"No conclusive test data for {test_info['display_name']} in period.")
        epi_data_sets["key_test_positivity_trends"] = calculated_positivity_trends
    else: epi_data_sets["calculation_notes"].append("'test_type' or 'test_result' columns missing for positivity trends.")

    # 3. Demographic Breakdown for Selected Condition
    if 'condition' in df_clinic_epi.columns and 'patient_id' in df_clinic_epi.columns:
        demo_data_for_cond = {"age_distribution_df": None, "gender_distribution_df": None, "condition_analyzed": selected_condition_for_demographics}
        df_for_demog_analysis = df_clinic_epi.copy()
        if selected_condition_for_demographics != "All Conditions (Aggregated)":
            df_for_demog_analysis = df_clinic_epi[df_clinic_epi['condition'].str.contains(selected_condition_for_demographics, case=False, na=False)]

        if not df_for_demog_analysis.empty:
            # Use unique patients for demographic breakdown
            unique_patients_demog = df_for_demog_analysis.drop_duplicates(subset=['patient_id'])
            if not unique_patients_demog.empty:
                # Age breakdown
                if 'age' in unique_patients_demog.columns and unique_patients_demog['age'].notna().any():
                    age_bins = [0, 5, 18, 35, 50, 65, np.inf]; age_labels = ['0-4', '5-17', '18-34', '35-49', '50-64', '65+']
                    temp_age_unique_df = unique_patients_demog.copy()
                    temp_age_unique_df.loc[:, 'age_group_display_val'] = pd.cut(temp_age_unique_df['age'], bins=age_bins, labels=age_labels, right=False)
                    age_df_result = temp_age_unique_df['age_group_display_val'].value_counts().sort_index().reset_index(); age_df_result.columns = ['Age Group', 'Patient Count']
                    demo_data_for_cond["age_distribution_df"] = age_df_result
                # Gender breakdown
                if 'gender' in unique_patients_demog.columns and unique_patients_demog['gender'].notna().any():
                    temp_gender_unique_df = unique_patients_demog.copy()
                    temp_gender_unique_df['gender_clean_val'] = temp_gender_unique_df['gender'].fillna("Unknown").astype(str).str.strip().replace(['','nan','None'],"Unknown")
                    gender_df_result = temp_gender_unique_df[temp_gender_unique_df['gender_clean_val'] != "Unknown"]['gender_clean_val'].value_counts().reset_index(); gender_df_result.columns = ['Gender', 'Patient Count']
                    demo_data_for_cond["gender_distribution_df"] = gender_df_result
                epi_data_sets["demographics_by_condition_data"] = demo_data_for_cond
            else: epi_data_sets["calculation_notes"].append(f"No unique patients for condition '{selected_condition_for_demographics}' for demographics.")
        else: epi_data_sets["calculation_notes"].append(f"No patient data found for condition '{selected_condition_for_demographics}'.")
    else: epi_data_sets["calculation_notes"].append("'condition' or 'patient_id' missing for demographic breakdown.")

    # 4. Referral Funnel Analysis (Simplified Summary)
    if 'referral_status' in df_clinic_epi.columns and 'encounter_id' in df_clinic_epi.columns:
        # Focus on known, actionable referral statuses
        actionable_ref_statuses = ['pending', 'completed', 'initiated', 'service provided', 'attended', 'missed appointment', 'declined']
        referral_df_for_funnel = df_clinic_epi[df_clinic_epi.get('referral_status',pd.Series(dtype=str)).str.lower().isin(actionable_ref_statuses)].copy()
        if not referral_df_for_funnel.empty:
            total_initiated_refs = referral_df_for_funnel['encounter_id'].nunique() # Assuming one referral process per encounter ID that has a referral status
            
            completed_outcomes_funnel = ['completed', 'service provided', 'attended consult', 'attended followup', 'attended']
            refs_positively_concluded = 0
            if 'referral_outcome' in referral_df_for_funnel.columns:
                refs_positively_concluded = referral_df_for_funnel[referral_df_for_funnel.get('referral_outcome',pd.Series(dtype=str)).str.lower().isin(completed_outcomes_funnel)]['encounter_id'].nunique()
            
            refs_still_pending_status = referral_df_for_funnel[referral_df_for_funnel['referral_status'].str.lower() == 'pending']['encounter_id'].nunique()
            
            funnel_stages_data = pd.DataFrame([
                {'Stage': 'Referrals Initiated (Period)', 'Count': total_initiated_refs},
                {'Stage': 'Concluded Positively (Outcome Recorded)', 'Count': refs_positively_concluded},
                {'Stage': 'Still Pending (Status "Pending")', 'Count': refs_still_pending_status},
            ])
            epi_data_sets["referral_funnel_summary_df"] = funnel_stages_data[funnel_stages_data['Count'] > 0] # Only show stages with counts
        else: epi_data_sets["calculation_notes"].append("No actionable referral records found for funnel analysis.")
    else: epi_data_sets["calculation_notes"].append("'referral_status' or 'encounter_id' data missing for referral funnel.")
    
    logger.info(f"({module_source_context}) Clinic epi data calculation finished. Notes: {len(epi_data_sets['calculation_notes'])}")
    return epi_data_sets
