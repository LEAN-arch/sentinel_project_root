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
    # Assumes this file is in sentinel_project_root/test/pages/chw_components_sentinel/
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_test_root_dir = os.path.abspath(os.path.join(current_script_dir, os.pardir, os.pardir))
    if project_test_root_dir not in sys.path:
        sys.path.insert(0, project_test_root_dir)
    from config import app_config

logger = logging.getLogger(__name__)

def extract_chw_local_epi_signals(
    chw_daily_encounter_df: Optional[pd.DataFrame],
    pre_calculated_chw_kpis: Optional[Dict[str, Any]] = None,
    for_date: Any, # datetime.date or similar
    chw_zone_context: str, # e.g., "Zone A" or "CHW01 - All Zones"
    max_symptom_clusters_to_report: int = 3
) -> Dict[str, Any]:
    """
    Extracts epidemiological signals and task-related counts from a CHW's daily data.

    Args:
        chw_daily_encounter_df: DataFrame of CHW's encounters for the day.
        pre_calculated_chw_kpis: Optional pre-calculated CHW daily summary metrics.
                                 Can provide 'pending_tb_contact_tracing_tasks_count'.
        for_date: The date for which these signals are relevant.
        chw_zone_context: Zone context for these signals (for logging/display).
        max_symptom_clusters_to_report: Max symptom clusters to report in output.

    Returns:
        Dictionary containing key epi signals and counts.
    """
    module_log_prefix = "CHWEpiSignalExtractor" # Consistent prefix
    logger.info(f"({module_log_prefix}) Extracting CHW local epi signals for date: {str(for_date)}, context: {chw_zone_context}")

    # Initialize the output structure with defaults
    epi_signals_output: Dict[str, Any] = {
        "date_of_activity": str(for_date),
        "operational_context": chw_zone_context,
        "symptomatic_patients_key_conditions_count": 0,
        "symptom_keywords_for_monitoring": "", # Will be populated
        "newly_identified_malaria_patients_count": 0,
        "newly_identified_tb_patients_count": 0,
        "pending_tb_contact_tracing_tasks_count": 0,
        "demographics_of_high_ai_risk_patients_today": {
            "total_high_risk_patients_count": 0,
            "age_group_distribution": {}, # e.g., {"0-4 yrs": 2, "50+ yrs": 1}
            "gender_distribution": {}     # e.g., {"Male": 1, "Female": 2}
        },
        "detected_symptom_clusters": [] # List of dicts
    }

    if not isinstance(chw_daily_encounter_df, pd.DataFrame) or chw_daily_encounter_df.empty:
        logger.warning(f"({module_log_prefix}) No daily encounter data provided to extract epi signals.")
        # Attempt to populate TB contacts from pre_calculated_chw_kpis if available
        if pre_calculated_chw_kpis and isinstance(pre_calculated_chw_kpis, dict):
            # Prefer specific key, fallback to a more general one if it implies TB contacts
            epi_signals_output["pending_tb_contact_tracing_tasks_count"] = int(
                pre_calculated_chw_kpis.get('pending_tb_contact_tracing_tasks_count',
                                             pre_calculated_chw_kpis.get('pending_critical_condition_referrals', 0))
            )
        return epi_signals_output

    df_epi_src = chw_daily_encounter_df.copy() # Work on a copy

    # Ensure essential columns exist and have appropriate types/defaults for robust processing
    essential_cols_config_epi = { # Renamed for clarity
        'patient_id': {"default": "UnknownPID_EpiExt", "type": str},
        'condition': {"default": "UnknownCondition_EpiExt", "type": str},
        'patient_reported_symptoms': {"default": "", "type": str}, # Default to empty string for str ops
        'ai_risk_score': {"default": np.nan, "type": float},
        'age': {"default": np.nan, "type": float},
        'gender': {"default": "Unknown", "type": str},
        'referral_reason': {"default": "", "type": str},
        'referral_status': {"default": "Unknown", "type": str}
    }
    common_na_strings_epi_ext = ['', 'nan', 'None', 'N/A', '#N/A', 'np.nan', 'NaT', '<NA>', 'null']

    for col, config_item in essential_cols_config_epi.items():
        if col not in df_epi_src.columns:
            df_epi_src[col] = config_item["default"]
        # Type coercion and standardized NA handling
        if config_item["type"] == float:
            df_epi_src[col] = pd.to_numeric(df_epi_src[col], errors='coerce').fillna(config_item["default"])
        elif config_item["type"] == str:
            df_epi_src[col] = df_epi_src[col].astype(str).str.strip().replace(common_na_strings_epi_ext, config_item["default"], regex=False)
            df_epi_src[col] = df_epi_src[col].fillna(config_item["default"]) # Final fill for any remaining NaNs


    # 1. Symptomatic Patients with Key Conditions
    # KEY_CONDITIONS_FOR_ACTION that are typically acutely symptomatic
    key_symptomatic_conditions_for_epi = list(set(app_config.KEY_CONDITIONS_FOR_ACTION) & \
                                              {"TB", "Pneumonia", "Malaria", "Dengue", "Sepsis", "Diarrheal Diseases (Severe)", "Heat Stroke"})
    
    general_symptom_keywords_regex = "fever|cough|chills|headache|ache|pain|diarrhea|vomit|rash|breathless|fatigue|dizzy|nausea" # Expanded
    epi_signals_output["symptom_keywords_for_monitoring"] = general_symptom_keywords_regex.replace("|", ", ")

    if 'patient_reported_symptoms' in df_epi_src.columns and 'condition' in df_epi_src.columns:
        # Ensure series are string type for .str operations
        symptoms_series_for_epi = df_epi_src['patient_reported_symptoms'].astype(str)
        condition_series_for_epi = df_epi_src['condition'].astype(str)

        symptoms_present_mask_epi = symptoms_series_for_epi.str.contains(general_symptom_keywords_regex, case=False, na=False)
        key_condition_mask_epi = condition_series_for_epi.apply(
            lambda x_condition_str: any(key_c.lower() in x_condition_str.lower() for key_c in key_symptomatic_conditions_for_epi)
        )
        symptomatic_key_cond_df = df_epi_src[key_condition_mask_epi & symptoms_present_mask_epi]
        
        if 'patient_id' in symptomatic_key_cond_df.columns: # Should exist due to prep
             epi_signals_output["symptomatic_patients_key_conditions_count"] = symptomatic_key_cond_df['patient_id'].nunique()

    # 2. Specific Disease Counts (Newly identified today by this CHW/Team)
    if 'condition' in df_epi_src.columns and 'patient_id' in df_epi_src.columns:
        condition_series_lc_epi = df_epi_src['condition'].str.lower() # For case-insensitive matching
        epi_signals_output["newly_identified_malaria_patients_count"] = df_epi_src[condition_series_lc_epi.str.contains("malaria", na=False)]['patient_id'].nunique()
        epi_signals_output["newly_identified_tb_patients_count"] = df_epi_src[condition_series_lc_epi.str.contains("tb", na=False)]['patient_id'].nunique()
        # Can add more for other KEY_CONDITIONS_FOR_ACTION if needed

    # 3. Pending TB Contact Tracing Tasks
    # Prioritize pre_calculated_chw_kpis if available and specific
    if pre_calculated_chw_kpis and isinstance(pre_calculated_chw_kpis, dict) and \
       'pending_tb_contact_tracing_tasks_count' in pre_calculated_chw_kpis:
        epi_signals_output["pending_tb_contact_tracing_tasks_count"] = int(pre_calculated_chw_kpis.get('pending_tb_contact_tracing_tasks_count', 0))
    elif pre_calculated_chw_kpis and isinstance(pre_calculated_chw_kpis, dict) and \
         'pending_critical_condition_referrals' in pre_calculated_chw_kpis and \
         "TB" in app_config.KEY_CONDITIONS_FOR_ACTION and \
         isinstance(pre_calculated_chw_kpis.get('pending_critical_condition_referrals'), (int, float)):
         # Fallback if TB contacts are rolled into general critical referrals count
         epi_signals_output["pending_tb_contact_tracing_tasks_count"] = int(pre_calculated_chw_kpis.get('pending_critical_condition_referrals',0))
    elif all(col in df_epi_src.columns for col in ['condition', 'referral_status', 'referral_reason', 'patient_id']): # Basic derivation from encounters
        epi_signals_output["pending_tb_contact_tracing_tasks_count"] = df_epi_src[
            df_epi_src['condition'].str.contains("TB", case=False, na=False) &
            df_epi_src['referral_reason'].str.contains("Contact Trac|Trace Contact", case=False, na=False) & # More flexible matching
            (df_epi_src['referral_status'].str.lower() == 'pending')
        ]['patient_id'].nunique()

    # 4. Demographics of High AI Risk Patients encountered today
    if 'ai_risk_score' in df_epi_src.columns and 'patient_id' in df_epi_src.columns:
        df_high_risk_unique_patients = df_epi_src[
            df_epi_src['ai_risk_score'] >= app_config.RISK_SCORE_HIGH_THRESHOLD
        ].drop_duplicates(subset=['patient_id']) # Get unique patients with high risk scores
        
        if not df_high_risk_unique_patients.empty:
            demo_stats = epi_signals_output["demographics_of_high_ai_risk_patients_today"]
            demo_stats["total_high_risk_patients_count"] = len(df_high_risk_unique_patients)
            
            if 'age' in df_high_risk_unique_patients.columns and df_high_risk_unique_patients['age'].notna().any():
                age_bins_epi = [0, 5, 15, 50, np.inf] 
                age_labels_epi = ['0-4 yrs', '5-14 yrs', '15-49 yrs', '50+ yrs']
                # Use .copy() to avoid SettingWithCopyWarning if df_high_risk_unique_patients is a slice
                temp_hr_age_analysis_df = df_high_risk_unique_patients.copy() 
                temp_hr_age_analysis_df['age_group_summary_val'] = pd.cut( # Use different temp col name
                    temp_hr_age_analysis_df['age'], bins=age_bins_epi, labels=age_labels_epi, right=False
                )
                demo_stats["age_group_distribution"] = temp_hr_age_analysis_df['age_group_summary_val'].value_counts().sort_index().to_dict()
            
            if 'gender' in df_high_risk_unique_patients.columns and df_high_risk_unique_patients['gender'].notna().any():
                temp_hr_gender_analysis_df = df_high_risk_unique_patients.copy()
                # Normalize gender more robustly for counting
                gender_map_func = lambda g_val: "Male" if str(g_val).lower() in ['m', 'male'] else \
                                         "Female" if str(g_val).lower() in ['f', 'female'] else "Other/Unknown"
                temp_hr_gender_analysis_df['gender_normalized_val'] = temp_hr_gender_analysis_df['gender'].apply(gender_map_func)
                gender_counts_dict = temp_hr_gender_analysis_df[
                    temp_hr_gender_analysis_df['gender_normalized_val'].isin(["Male", "Female"]) # Focus on Male/Female for this summary
                ]['gender_normalized_val'].value_counts().to_dict()
                demo_stats["gender_distribution"] = gender_counts_dict

    # 5. Basic Symptom Cluster Detection
    if 'patient_reported_symptoms' in df_epi_src.columns and df_epi_src['patient_reported_symptoms'].str.strip().astype(bool).any():
        symptoms_lc_series = df_epi_src['patient_reported_symptoms'].str.lower() # Already string and filled by prep
        
        symptom_patterns_to_monitor = { # Can be expanded or moved to app_config
            "Fever & Diarrhea": (symptoms_lc_series.str.contains('fever', na=False) & symptoms_lc_series.str.contains('diarrhea', na=False)),
            "Fever & Cough": (symptoms_lc_series.str.contains('fever', na=False) & symptoms_lc_series.str.contains('cough', na=False)),
            "Fever & Rash": (symptoms_lc_series.str.contains('fever', na=False) & symptoms_lc_series.str.contains('rash', na=False)),
            "Breathlessness & Chest Pain": (symptoms_lc_series.str.contains('breathless|short of breath|difficulty breathing', na=False) & symptoms_lc_series.str.contains('chest pain|chest discomfort', na=False))
        }
        min_patients_for_cluster_alert = 2 # Alert if >= X unique patients show pattern in CHW's daily encounters for this zone/context

        detected_clusters_buffer = []
        for pattern_display_name, pattern_eval_mask in symptom_patterns_to_monitor.items():
            if pattern_eval_mask.any(): # Optimization: only count if mask has True values
                # Count unique patients matching this pattern
                num_unique_patients_with_pattern = df_epi_src[pattern_eval_mask]['patient_id'].nunique()
                if num_unique_patients_with_pattern >= min_patients_for_cluster_alert:
                    detected_clusters_buffer.append({
                        "symptoms_pattern": pattern_display_name,
                        "patient_count": int(num_unique_patients_with_pattern),
                        "location_hint": chw_zone_context # Location is CHW's operational area/zone for the day
                    })
        
        if detected_clusters_buffer: # Sort by patient_count descending if clusters are found
            epi_signals_output["detected_symptom_clusters"] = sorted(
                detected_clusters_buffer, key=lambda x_cluster: x_cluster['patient_count'], reverse=True
            )[:max_symptom_clusters_to_report]

    num_clusters_found = len(epi_signals_output.get('detected_symptom_clusters', []))
    logger.info(f"({module_log_prefix}) CHW local epi signals extracted successfully. Clusters found: {num_clusters_found}")
    return epi_signals_output
