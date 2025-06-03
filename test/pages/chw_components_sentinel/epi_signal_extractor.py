# sentinel_project_root/test/pages/chw_components_sentinel/epi_signal_extractor.py
# Extracts epidemiological signals from CHW daily data for Sentinel Health Co-Pilot.

import pandas as pd
import numpy as np
import logging
import re
from typing import Dict, Any, Optional, List
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

def extract_chw_local_epi_signals(
    chw_daily_encounter_df: Optional[pd.DataFrame],
    pre_calculated_chw_kpis: Optional[Dict[str, Any]] = None,
    for_date: Any,
    chw_zone_context: str,
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

    Raises:
        ValueError: If input data or date is invalid.
    """
    module_log_prefix = "CHWEpiSignalExtractor"

    # Validate app_config attributes
    required_configs = ['KEY_CONDITIONS_FOR_ACTION', 'RISK_SCORE_HIGH_THRESHOLD']
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
        raise ValueError("Invalid date format for epi signal extraction")

    logger.info(f"({module_log_prefix}) Extracting CHW local epi signals for date: {for_date_str}, context: {chw_zone_context}")

    # Initialize output structure
    epi_signals_output: Dict[str, Any] = {
        "date_of_activity": for_date_str,
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
        "detected_symptom_clusters": []
    }

    if not isinstance(chw_daily_encounter_df, pd.DataFrame) or chw_daily_encounter_df.empty:
        logger.warning(f"({module_log_prefix}) No daily encounter data provided.")
        if pre_calculated_chw_kpis and isinstance(pre_calculated_chw_kpis, dict):
            epi_signals_output["pending_tb_contact_tracing_tasks_count"] = int(
                pre_calculated_chw_kpis.get('pending_tb_contact_tracing_tasks_count', 
                    pre_calculated_chw_kpis.get('pending_critical_condition_referrals', 0))
            )
        return epi_signals_output

    # Define essential columns
    essential_cols_config_epi = {
        'patient_id': {"default": "UnknownPID_EpiExt", "type": str},
        'condition': {"default": "UnknownCondition_EpiExt", "type": str},
        'patient_reported_symptoms': {"default": "", "type": str},
        'ai_risk_score': {"default": np.nan, "type": "float"},
        'age': {"default": np.nan, "type": "float"},
        'gender': {"default": "Unknown", "type": str},
        'referral_reason': {"default": "", "type": str},
        'referral_status': {"default": "Unknown", "type": str}
    }
    common_na_strings = ['', 'nan', 'None', 'N/A', '#N/A', 'np.nan', 'NaT', '<NA>', 'null']

    # Prepare DataFrame with only necessary columns
    required_cols = list(essential_cols_config_epi.keys())
    df_epi_src = chw_daily_encounter_df[required_cols].copy() if all(col in chw_daily_encounter_df.columns for col in required_cols) else chw_daily_encounter_df.copy()

    for col, config in essential_cols_config_epi.items():
        if col not in df_epi_src.columns:
            df_epi_src[col] = config["default"]
        if config["type"] == "float":
            df_epi_src[col] = pd.to_numeric(df_epi_src[col], errors='coerce').fillna(config["default"])
        elif config["type"] == str:
            df_epi_src[col] = df_epi_src[col].astype(str).replace(common_na_strings, config["default"])

    # 1. Symptomatic Patients with Key Conditions
    key_symptomatic_conditions = list(set(app_config.KEY_CONDITIONS_FOR_ACTION) & 
                                     {"TB", "Pneumonia", "Malaria", "Dengue", "Sepsis", "Diarrheal Diseases (Severe)", "Heat Stroke"})
    
    general_symptom_keywords_regex = re.compile(
        r"fever|cough|chills|headache|ache|pain|diarrhea|vomit|rash|breathless|short of breath|fatigue|dizzy|nausea",
        re.IGNORECASE
    )
    epi_signals_output["symptom_keywords_for_monitoring"] = general_symptom_keywords_regex.pattern.replace("|", ", ")

    if 'patient_reported_symptoms' in df_epi_src.columns and 'condition' in df_epi_src.columns:
        symptoms_series = df_epi_src['patient_reported_symptoms'].astype(str)
        condition_series = df_epi_src['condition'].astype(str)

        symptoms_present = symptoms_series.str.contains(general_symptom_keywords_regex, na=False)
        key_condition_mask = condition_series.apply(
            lambda x: any(key_c.lower() in x.lower() for key_c in key_symptomatic_conditions)
        )
        symptomatic_key_cond_df = df_epi_src[symptoms_present & key_condition_mask]
        epi_signals_output["symptomatic_patients_key_conditions_count"] = symptomatic_key_cond_df['patient_id'].nunique()

    # 2. Specific Disease Counts
    if 'condition' in df_epi_src.columns:
        condition_series_lc = df_epi_src['condition'].str.lower()
        epi_signals_output["newly_identified_malaria_patients_count"] = df_epi_src[condition_series_lc.str.contains("malaria", na=False)]['patient_id'].nunique()
        epi_signals_output["newly_identified_tb_patients_count"] = df_epi_src[condition_series_lc.str.contains("tb", na=False)]['patient_id'].nunique()

    # 3. Pending TB Contact Tracing Tasks
    if pre_calculated_chw_kpis and isinstance(pre_calculated_chw_kpis, dict):
        epi_signals_output["pending_tb_contact_tracing_tasks_count"] = int(
            pre_calculated_chw_kpis.get('pending_tb_contact_tracing_tasks_count', 
                pre_calculated_chw_kpis.get('pending_critical_condition_referrals', 0))
        )
    elif all(col in df_epi_src.columns for col in ['condition', 'referral_status', 'referral_reason']):
        epi_signals_output["pending_tb_contact_tracing_tasks_count"] = df_epi_src[
            df_epi_src['condition'].str.contains("TB", case=False, na=False) &
            df_epi_src['referral_reason'].str.contains("Contact Trac|Trace Contact", case=False, na=False) &
            (df_epi_src['referral_status'].str.lower() == 'pending')
        ]['patient_id'].nunique()

    # 4. Demographics of High AI Risk Patients
    if 'ai_risk_score' in df_epi_src.columns:
        df_high_risk = df_epi_src[df_epi_src['ai_risk_score'] >= app_config.RISK_SCORE_HIGH_THRESHOLD].drop_duplicates(subset=['patient_id'])
        if not df_high_risk.empty:
            demo_stats = epi_signals_output["demographics_of_high_ai_risk_patients_today"]
            demo_stats["total_high_risk_patients_count"] = len(df_high_risk)
            
            if 'age' in df_high_risk.columns and df_high_risk['age'].notna().any():
                age_bins = [0, 5, 15, 50, np.inf]
                age_labels = ['0-4 yrs', '5-14 yrs', '15-49 yrs', '50+ yrs']
                df_high_risk['age_group'] = pd.cut(df_high_risk['age'], bins=age_bins, labels=age_labels, right=False)
                demo_stats["age_group_distribution"] = df_high_risk['age_group'].value_counts().sort_index().to_dict()
            
            if 'gender' in df_high_risk.columns and df_high_risk['gender'].notna().any():
                gender_map = lambda g: "Male" if str(g).lower() in ['m', 'male'] else \
                              "Female" if str(g).lower() in ['f', 'female'] else "Other/Unknown"
                df_high_risk['gender_normalized'] = df_high_risk['gender'].apply(gender_map)
                gender_counts = df_high_risk[df_high_risk['gender_normalized'].isin(["Male", "Female"])]['gender_normalized'].value_counts().to_dict()
                demo_stats["gender_distribution"] = gender_counts

    # 5. Symptom Cluster Detection
    if 'patient_reported_symptoms' in df_epi_src.columns and df_epi_src['patient_reported_symptoms'].str.strip().astype(bool).any():
        symptoms_series = df_epi_src['patient_reported_symptoms'].str.lower()
        symptom_patterns = getattr(app_config, 'SYMPTOM_CLUSTERS', {
            "Fever & Diarrhea": symptoms_series.str.contains('fever', na=False) & symptoms_series.str.contains('diarrhea', na=False),
            "Fever & Cough": symptoms_series.str.contains('fever', na=False) & symptoms_series.str.contains('cough', na=False),
            "Fever & Rash": symptoms_series.str.contains('fever', na=False) & symptoms_series.str.contains('rash', na=False),
            "Breathlessness & Chest Pain": symptoms_series.str.contains('breathless|short of breath|difficulty breathing', na=False) & 
                                          symptoms_series.str.contains('chest pain|chest discomfort', na=False)
        })
        min_patients_for_cluster = 2

        detected_clusters = []
        for pattern_name, mask in symptom_patterns.items():
            if mask.any():
                patient_count = df_epi_src[mask]['patient_id'].nunique()
                if patient_count >= min_patients_for_cluster:
                    detected_clusters.append({
                        "symptoms_pattern": pattern_name,
                        "patient_count": int(patient_count),
                        "location_hint": chw_zone_context
                    })
        
        epi_signals_output["detected_symptom_clusters"] = sorted(
            detected_clusters, key=lambda x: x['patient_count'], reverse=True
        )[:max_symptom_clusters_to_report]

    num_clusters = len(epi_signals_output["detected_symptom_clusters"])
    logger.info(f"({module_log_prefix}) CHW local epi signals extracted. Clusters found: {num_clusters}")
    return epi_signals_output
