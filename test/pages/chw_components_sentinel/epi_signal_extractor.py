# sentinel_project_root/test/pages/chw_components_sentinel/epi_signal_extractor.py
# Extracts epidemiological signals from CHW daily data for Sentinel Health Co-Pilot.

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List

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

def extract_chw_local_epi_signals(
    chw_daily_encounter_df: Optional[pd.DataFrame],
    pre_calculated_chw_kpis: Optional[Dict[str, Any]] = None,
    for_date: Any, # datetime.date or similar
    chw_zone_context: str,
    max_symptom_clusters_to_report: int = 3
) -> Dict[str, Any]:
    """
    Extracts epidemiological signals and task-related counts from a CHW's daily data.

    Args:
        chw_daily_encounter_df: DataFrame of CHW's encounters for the day.
        pre_calculated_chw_kpis: Optional pre-calculated CHW daily summary metrics.
        for_date: The date for which these signals are relevant.
        chw_zone_context: Zone context for these signals.
        max_symptom_clusters_to_report: Max symptom clusters to report.

    Returns:
        Dictionary containing key epi signals and counts.
    """
    module_log_prefix = "CHWEpiSignalExtractor"
    logger.info(f"({module_log_prefix}) Extracting CHW local epi signals for date: {str(for_date)}, context: {chw_zone_context}")

    epi_signals_output: Dict[str, Any] = {
        "date_of_activity": str(for_date),
        "operational_context": chw_zone_context,
        "symptomatic_patients_key_conditions_count": 0,
        "symptom_keywords_for_monitoring": "",
        "newly_identified_malaria_patients_count": 0,
        "newly_identified_tb_patients_count": 0,
        "pending_tb_contact_tracing_tasks_count": 0,
        "demographics_of_high_ai_risk_patients_today": {
            "total_high_risk_patients_count": 0,
            "age_group_distribution": {},
            "gender_distribution": {}
        },
        "detected_symptom_clusters": [] # List of {"symptoms_pattern": "fever;cough", "patient_count": 3, "location_hint": "Zone X"}
    }

    if not isinstance(chw_daily_encounter_df, pd.DataFrame) or chw_daily_encounter_df.empty:
        logger.warning(f"({module_log_prefix}) No daily encounter data provided.")
        # Try to populate TB contacts from pre-calculated if available, even if encounters are empty
        if pre_calculated_chw_kpis:
            epi_signals_output["pending_tb_contact_tracing_tasks_count"] = int(
                pre_calculated_chw_kpis.get('pending_tb_contact_tracing_tasks_count', # Prefer specific
                                             pre_calculated_chw_kpis.get('pending_critical_condition_referrals', 0)) # Fallback
            )
        return epi_signals_output

    df_enc = chw_daily_encounter_df.copy() # Work on a copy

    # Ensure essential columns exist and have appropriate types/defaults
    essential_cols_config = {
        'patient_id': {"default": "UnknownPID", "type": str},
        'condition': {"default": "UnknownCondition", "type": str},
        'patient_reported_symptoms': {"default": "", "type": str},
        'ai_risk_score': {"default": np.nan, "type": float},
        'age': {"default": np.nan, "type": float},
        'gender': {"default": "Unknown", "type": str},
        'referral_reason': {"default": "", "type": str},
        'referral_status': {"default": "Unknown", "type": str}
    }
    common_na_strings = ['', 'nan', 'None', 'N/A', '#N/A', 'np.nan', 'NaT', '<NA>', 'null']

    for col, config in essential_cols_config.items():
        if col not in df_enc.columns:
            df_enc[col] = config["default"]
            logger.debug(f"({module_log_prefix}) Added missing column '{col}' with default '{config['default']}'.")
        
        if config["type"] == float:
            df_enc[col] = pd.to_numeric(df_enc[col], errors='coerce').fillna(config["default"])
        elif config["type"] == str:
            df_enc[col] = df_enc[col].astype(str).str.strip().replace(common_na_strings, config["default"], regex=False)
            df_enc[col] = df_enc[col].fillna(config["default"])


    # 1. Symptomatic Patients with Key Conditions
    key_symptomatic_conditions = list(set(app_config.KEY_CONDITIONS_FOR_ACTION) & \
                                      {"TB", "Pneumonia", "Malaria", "Dengue", "Sepsis", "Diarrheal Diseases (Severe)", "Heat Stroke"})
    
    symptom_keywords = "fever|cough|chills|headache|ache|pain|diarrhea|vomit|rash|breathless|fatigue|dizzy|nausea" # Expanded slightly
    epi_signals_output["symptom_keywords_for_monitoring"] = symptom_keywords.replace("|", ", ")

    if 'patient_reported_symptoms' in df_enc.columns and 'condition' in df_enc.columns:
        symptoms_present_mask = df_enc['patient_reported_symptoms'].str.contains(symptom_keywords, case=False, na=False)
        key_condition_mask = df_enc['condition'].apply(
            lambda x_cond: any(key_c.lower() in str(x_cond).lower() for key_c in key_symptomatic_conditions)
        )
        symptomatic_key_condition_patients_df = df_enc[key_condition_mask & symptoms_present_mask]
        if 'patient_id' in symptomatic_key_condition_patients_df.columns:
             epi_signals_output["symptomatic_patients_key_conditions_count"] = symptomatic_key_condition_patients_df['patient_id'].nunique()

    # 2. Specific Disease Counts (Newly identified today by this CHW/Team)
    if 'condition' in df_enc.columns and 'patient_id' in df_enc.columns:
        epi_signals_output["newly_identified_malaria_patients_count"] = df_enc[df_enc['condition'].str.contains("Malaria", case=False, na=False)]['patient_id'].nunique()
        epi_signals_output["newly_identified_tb_patients_count"] = df_enc[df_enc['condition'].str.contains("TB", case=False, na=False)]['patient_id'].nunique()

    # 3. Pending TB Contact Tracing Tasks
    if pre_calculated_chw_kpis and 'pending_tb_contact_tracing_tasks_count' in pre_calculated_chw_kpis:
        epi_signals_output["pending_tb_contact_tracing_tasks_count"] = int(pre_calculated_chw_kpis.get('pending_tb_contact_tracing_tasks_count', 0))
    elif pre_calculated_chw_kpis and 'pending_critical_condition_referrals' in pre_calculated_chw_kpis and \
         "TB" in app_config.KEY_CONDITIONS_FOR_ACTION and isinstance(pre_calculated_chw_kpis.get('pending_critical_condition_referrals'), (int, float)):
         # Fallback if TB contacts are rolled into general critical referrals count
         epi_signals_output["pending_tb_contact_tracing_tasks_count"] = int(pre_calculated_chw_kpis.get('pending_critical_condition_referrals',0))
    elif 'condition' in df_enc.columns and 'referral_status' in df_enc.columns and 'referral_reason' in df_enc.columns: # Basic derivation if no pre-calc
        epi_signals_output["pending_tb_contact_tracing_tasks_count"] = df_enc[
            df_enc['condition'].str.contains("TB", case=False, na=False) &
            df_enc['referral_reason'].str.contains("Contact Trac", case=False, na=False) & # Assuming "Contact Tracing" or similar
            (df_enc['referral_status'].str.lower() == 'pending')
        ]['patient_id'].nunique()

    # 4. Demographics of High AI Risk Patients encountered today
    if 'ai_risk_score' in df_enc.columns and 'patient_id' in df_enc.columns:
        high_risk_patients_df = df_enc[
            df_enc['ai_risk_score'] >= app_config.RISK_SCORE_HIGH_THRESHOLD
        ].drop_duplicates(subset=['patient_id']) # Unique patients
        
        if not high_risk_patients_df.empty:
            demo_stats_dict = epi_signals_output["demographics_of_high_ai_risk_patients_today"]
            demo_stats_dict["total_high_risk_patients_count"] = len(high_risk_patients_df)
            
            if 'age' in high_risk_patients_df.columns and high_risk_patients_df['age'].notna().any():
                age_bins = [0, 5, 15, 50, np.inf] # Under-5, Child/Adol, Adult, Elderly
                age_labels = ['0-4 yrs', '5-14 yrs', '15-49 yrs', '50+ yrs']
                # Use .copy() to avoid SettingWithCopyWarning if high_risk_patients_df is a slice
                temp_age_df = high_risk_patients_df.copy() 
                temp_age_df['age_group_summary'] = pd.cut(
                    temp_age_df['age'], bins=age_bins, labels=age_labels, right=False # right=False means [bin_start, bin_end)
                )
                demo_stats_dict["age_group_distribution"] = temp_age_df['age_group_summary'].value_counts().sort_index().to_dict()
            
            if 'gender' in high_risk_patients_df.columns and high_risk_patients_df['gender'].notna().any():
                # Clean gender values before counting
                gender_cleaned_series = high_risk_patients_df['gender'].replace(common_na_strings + ["Other"], "Unknown", regex=False)
                gender_counts = gender_cleaned_series[gender_cleaned_series.isin(["Male", "Female"])].value_counts().to_dict() # Only Male/Female for this summary
                demo_stats_dict["gender_distribution"] = gender_counts

    # 5. Basic Symptom Cluster Detection
    if 'patient_reported_symptoms' in df_enc.columns and df_enc['patient_reported_symptoms'].notna().any():
        symptoms_series = df_enc['patient_reported_symptoms'].str.lower() # Already string and filled by prep
        
        # Define symptom patterns of interest (can be expanded via app_config)
        symptom_patterns_config = {
            "Fever & Diarrhea": (symptoms_series.str.contains('fever', na=False) & symptoms_series.str.contains('diarrhea', na=False)),
            "Fever & Cough": (symptoms_series.str.contains('fever', na=False) & symptoms_series.str.contains('cough', na=False)),
            "Fever & Rash": (symptoms_series.str.contains('fever', na=False) & symptoms_series.str.contains('rash', na=False)),
            "Breathlessness & Chest Pain": (symptoms_series.str.contains('breathless|short of breath', na=False) & symptoms_series.str.contains('chest pain', na=False))
        }
        cluster_min_count_threshold = 2 # Alert if >= X unique patients show this pattern in CHW's daily encounters
        
        detected_clusters_list = []
        for pattern_name, pattern_mask in symptom_patterns_config.items():
            if pattern_mask.any(): # Check if any True values in mask before trying to count
                unique_patients_with_pattern_count = df_enc[pattern_mask]['patient_id'].nunique()
                if unique_patients_with_pattern_count >= cluster_min_count_threshold:
                    detected_clusters_list.append({
                        "symptoms_pattern": pattern_name,
                        "patient_count": int(unique_patients_with_pattern_count),
                        "location_hint": chw_zone_context # Location is CHW's operational area for the day
                    })
        
        if detected_clusters_list:
            epi_signals_output["detected_symptom_clusters"] = sorted(
                detected_clusters_list, key=lambda x: x['patient_count'], reverse=True
            )[:max_symptom_clusters_to_report]

    logger.info(f"({module_log_prefix}) CHW local epi signals extracted successfully. Clusters: {len(epi_signals_output['detected_symptom_clusters'])}")
    return epi_signals_output
