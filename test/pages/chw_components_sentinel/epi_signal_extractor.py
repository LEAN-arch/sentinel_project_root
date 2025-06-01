# sentinel_project_root/test/pages/chw_components_sentinel/epi_signal_extractor.py
# Part of "Sentinel Health Co-Pilot" - LMIC Edge-First System
# This module processes CHW daily encounter data to extract potential
# epidemiological signals and key task-related counts.
# Refactored from the original chw_components/epi_watch.py.
# Output is a structured dictionary for supervisor reports or Tier 2 aggregation.

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List

# Assuming app_config is accessible via PYTHONPATH
try:
    from config import app_config
except ImportError:
    import sys, os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_test_root = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
    if project_test_root not in sys.path: sys.path.insert(0, project_test_root)
    from config import app_config

logger = logging.getLogger(__name__)

def extract_chw_local_epi_signals(
    chw_daily_encounter_df: Optional[pd.DataFrame], # CHW's encounters for the day
    # `pre_calculated_chw_kpis` might contain daily roll-ups (e.g., from get_chw_summary).
    # Key 'tb_contacts_to_trace_today' would be useful.
    pre_calculated_chw_kpis: Optional[Dict[str, Any]] = None,
    for_date: Any, # datetime.date or similar, for context
    chw_zone_context: str, # e.g., "Zone A" or "All Assigned Zones for CHW01"
    max_symptom_clusters_to_report: int = 3 # Limit for reporting
) -> Dict[str, Any]:
    """
    Extracts epidemiological signals and task-related counts from a CHW's daily data.

    Args:
        chw_daily_encounter_df: DataFrame of the CHW's encounters for the specified day.
                               Expected cols: 'patient_id', 'condition', 'patient_reported_symptoms',
                                              'ai_risk_score', 'age', 'gender', 'referral_reason', 'referral_status'.
        pre_calculated_chw_kpis: Optional pre-calculated CHW daily summary metrics.
        for_date: The date for which these signals are relevant.
        chw_zone_context: The zone(s) context for these signals.
        max_symptom_clusters_to_report: Max number of symptom clusters to report.

    Returns:
        Dict[str, Any]: A dictionary containing key epi signals and counts.
    """
    module_source_context = "CHWEpiSignalExtractor"
    logger.info(f"({module_source_context}) Extracting CHW local epi signals for date: {str(for_date)}, context: {chw_zone_context}")

    # Initialize the output structure
    epi_signals_output: Dict[str, Any] = {
        "date_of_activity": str(for_date),
        "operational_context": chw_zone_context,
        "symptomatic_patients_key_conditions_count": 0, # Unique patients with key conditions + symptoms
        "symptom_keywords_for_monitoring": "",          # For context on what was looked for
        # Counts for specific, highly relevant conditions (from KEY_CONDITIONS_FOR_ACTION)
        "newly_identified_malaria_patients_count": 0,
        "newly_identified_tb_patients_count": 0, # Patients with TB condition reported today
        "pending_tb_contact_tracing_tasks_count": 0,
        "demographics_of_high_ai_risk_patients_today": { # Simplified summary for supervisor
            "total_high_risk_patients_count": 0,
            "age_group_distribution": {}, # e.g., {"0-4 yrs": 2, "50+ yrs": 1}
            "gender_distribution": {}     # e.g., {"Male": 1, "Female": 2}
        },
        "detected_symptom_clusters": [] # List of {"symptoms_pattern": "fever;cough", "patient_count": 3}
    }

    if chw_daily_encounter_df is None or chw_daily_encounter_df.empty:
        logger.warning(f"({module_source_context}) No daily encounter data provided.")
        # Try to populate TB contacts from pre-calculated if available
        if pre_calculated_chw_kpis:
            epi_signals_output["pending_tb_contact_tracing_tasks_count"] = int(pre_calculated_chw_kpis.get('pending_critical_condition_referrals', 0)) # Assuming this kpi name if tb contacts rolled into it
        return epi_signals_output

    df_enc_epi = chw_daily_encounter_df.copy() # Work on a copy

    # Ensure necessary columns exist for safety, filling with neutral defaults
    essential_cols_epi = {
        'patient_id': "UnknownPID", 'condition': "UnknownCond", 'patient_reported_symptoms': "",
        'ai_risk_score': np.nan, 'age': np.nan, 'gender': "Unknown",
        'referral_reason': "", 'referral_status': ""
    }
    for col, default in essential_cols_epi.items():
        if col not in df_enc_epi.columns: df_enc_epi[col] = default
        elif 'age' in col or 'score' in col: df_enc_epi[col] = pd.to_numeric(df_enc_epi[col], errors='coerce').fillna(default)
        else: df_enc_epi[col] = df_enc_epi[col].fillna(default)


    # 1. Symptomatic Patients with Key Conditions
    # Focus on conditions from app_config.KEY_CONDITIONS_FOR_ACTION that are typically acutely symptomatic
    key_symptomatic_conditions_list = list(set(app_config.KEY_CONDITIONS_FOR_ACTION) & \
                                      {"TB", "Pneumonia", "Malaria", "Dengue", "Sepsis", "Diarrheal Diseases (Severe)", "Heat Stroke"})
    
    # General symptom keywords (can be expanded in app_config if needed)
    general_symptom_keywords = "fever|cough|chills|headache|ache|pain|diarrhea|vomit|rash|breathless|fatigue" # Broader net
    epi_signals_output["symptom_keywords_for_monitoring"] = general_symptom_keywords.replace("|", ", ")

    if 'patient_reported_symptoms' in df_enc_epi.columns and 'condition' in df_enc_epi.columns:
        symptoms_series_epi = df_enc_epi['patient_reported_symptoms'].astype(str).fillna('')
        symptomatic_patients_df = df_enc_epi[
            df_enc_epi['condition'].apply(lambda x_cond: any(key_c.lower() in str(x_cond).lower() for key_c in key_symptomatic_conditions_list)) &
            (symptoms_series_epi.str.contains(general_symptom_keywords, case=False, na=False))
        ]
        if 'patient_id' in symptomatic_patients_df.columns:
             epi_signals_output["symptomatic_patients_key_conditions_count"] = symptomatic_patients_df['patient_id'].nunique()


    # 2. Specific Disease Counts (Newly identified today by this CHW/Team)
    if 'condition' in df_enc_epi.columns and 'patient_id' in df_enc_epi.columns:
        epi_signals_output["newly_identified_malaria_patients_count"] = df_enc_epi[df_enc_epi['condition'].str.contains("Malaria", case=False, na=False)]['patient_id'].nunique()
        epi_signals_output["newly_identified_tb_patients_count"] = df_enc_epi[df_enc_epi['condition'].str.contains("TB", case=False, na=False)]['patient_id'].nunique()
        # Can add more for other KEY_CONDITIONS_FOR_ACTION


    # 3. Pending TB Contact Tracing Tasks (derived from today's new TB cases needing contact tracing)
    # If pre_calculated_chw_kpis already has this from a more complex source (like the Task Processor output), use it.
    if pre_calculated_chw_kpis and 'pending_tb_contact_tracing_tasks_count' in pre_calculated_chw_kpis: # if a dedicated field exists
        epi_signals_output["pending_tb_contact_tracing_tasks_count"] = int(pre_calculated_chw_kpis.get('pending_tb_contact_tracing_tasks_count', 0))
    elif pre_calculated_chw_kpis and 'pending_critical_condition_referrals' in pre_calculated_chw_kpis and isinstance(pre_calculated_chw_kpis.get('pending_critical_condition_referrals'), int):
        # Fallback if TB contacts are rolled into general critical referrals count
        # This is less specific, better to have a dedicated count if possible.
         epi_signals_output["pending_tb_contact_tracing_tasks_count"] = int(pre_calculated_chw_kpis.get('pending_critical_condition_referrals',0)) if "TB" in app_config.KEY_CONDITIONS_FOR_ACTION else 0
    elif 'condition' in df_enc_epi and 'referral_status' in df_enc_epi and 'referral_reason' in df_enc_epi : # Basic derivation
        epi_signals_output["pending_tb_contact_tracing_tasks_count"] = df_enc_epi[
            df_enc_epi['condition'].str.contains("TB", case=False, na=False) &
            df_enc_epi['referral_reason'].str.contains("Contact Trac", case=False, na=False) &
            (df_enc_epi['referral_status'].astype(str).str.lower() == 'pending')
        ]['patient_id'].nunique()


    # 4. Demographics of High AI Risk Patients encountered today (for supervisor context)
    if 'ai_risk_score' in df_enc_epi.columns and 'patient_id' in df_enc_epi.columns:
        # Get unique patients with high risk scores
        high_risk_patients_today_unique_df = df_enc_epi[
            df_enc_epi['ai_risk_score'] >= app_config.RISK_SCORE_HIGH_THRESHOLD
        ].drop_duplicates(subset=['patient_id'])
        
        if not high_risk_patients_today_unique_df.empty:
            demo_stats = epi_signals_output["demographics_of_high_ai_risk_patients_today"]
            demo_stats["total_high_risk_patients_count"] = len(high_risk_patients_today_unique_df)
            
            if 'age' in high_risk_patients_today_unique_df.columns and high_risk_patients_today_unique_df['age'].notna().any():
                # Use broader LMIC-relevant age bins for this summary
                age_bins_for_epi_summary = [0, 5, 15, 50, np.inf] # e.g., Under-5, Child/Adol, Adult, Elderly
                age_labels_for_epi_summary = ['0-4 yrs', '5-14 yrs', '15-49 yrs', '50+ yrs']
                
                temp_hr_age_df = high_risk_patients_today_unique_df.copy() # Avoid SettingWithCopy
                temp_hr_age_df.loc[:, 'age_group_summary'] = pd.cut(
                    temp_hr_age_df['age'], bins=age_bins_for_epi_summary, labels=age_labels_for_epi_summary, right=False
                )
                demo_stats["age_group_distribution"] = temp_hr_age_df['age_group_summary'].value_counts().sort_index().to_dict()
            
            if 'gender' in high_risk_patients_today_unique_df.columns and high_risk_patients_today_unique_df['gender'].notna().any():
                temp_hr_gender_df = high_risk_patients_today_unique_df.copy()
                temp_hr_gender_df['gender_clean'] = temp_hr_gender_df['gender'].fillna("Unknown").astype(str).str.strip().replace(["","nan","None"], "Unknown")
                gender_counts = temp_hr_gender_df[temp_hr_gender_df['gender_clean'] != 'Unknown']['gender_clean'].value_counts().to_dict()
                demo_stats["gender_distribution"] = gender_counts

    # 5. Basic Symptom Cluster Detection (count co-occurrence)
    #    PEDs could do more advanced local proximity checks; this is for summary reporting from aggregated daily data.
    if 'patient_reported_symptoms' in df_enc_epi.columns and df_enc_epi['patient_reported_symptoms'].notna().any():
        symptoms_lower_series = df_enc_epi['patient_reported_symptoms'].astype(str).str.lower().fillna("")
        # Define some patterns of interest for LMICs (can be expanded in app_config)
        symptom_patterns_to_check = {
            "Fever & Diarrhea": (symptoms_lower_series.str.contains('fever', na=False) & symptoms_lower_series.str.contains('diarrhea', na=False)),
            "Fever & Cough": (symptoms_lower_series.str.contains('fever', na=False) & symptoms_lower_series.str.contains('cough', na=False)),
            "Fever & Rash": (symptoms_lower_series.str.contains('fever', na=False) & symptoms_lower_series.str.contains('rash', na=False))
            # Could add: "Unexplained Bleeding", "Jaundice & Fever", etc.
        }
        cluster_alert_threshold = 2 # Alert if >= X patients show this pattern (context dependent)
        
        detected_clusters = []
        for pattern_name, pattern_mask in symptom_patterns_to_check.items():
            count = df_enc_epi[pattern_mask]['patient_id'].nunique() # Unique patients with this pattern
            if count >= cluster_alert_threshold:
                detected_clusters.append({
                    "symptoms_pattern": pattern_name,
                    "patient_count": int(count),
                    "location_hint": chw_zone_context # Supervisor sees aggregate for their area
                })
        if detected_clusters: # Sort by count descending if clusters are found
            epi_signals_output["detected_symptom_clusters"] = sorted(detected_clusters, key=lambda x: x['patient_count'], reverse=True)[:max_symptom_clusters_to_report]

    logger.info(f"({module_source_context}) CHW local epi signals extracted successfully.")
    return epi_signals_output
