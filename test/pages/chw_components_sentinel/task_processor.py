# sentinel_project_root/test/pages/chw_components_sentinel/task_processor.py
# Part of "Sentinel Health Co-Pilot" - LMIC Edge-First System
# This module processes CHW data (especially alerts and AI scores) to generate
# a prioritized list of tasks for the CHW.
# Refactored from the original chw_components/tasks_display.py.
# The output is a structured list of task objects for PEDs or supervisor review.

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime # For generating task IDs or default due dates

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

def generate_chw_prioritized_tasks(
    source_patient_data_df: pd.DataFrame, # Data containing potential task triggers (AI scores, alerts)
    # chw_daily_encounter_context_df: Optional[pd.DataFrame] = None, # Less critical here if source_patient_data_df is comprehensive
    for_date: Any, # Typically datetime.date, for context and default due dates
    chw_id_context: Optional[str] = "TeamDefaultCHW", # For assigning task if not inherent in data
    zone_context_str: Optional[str] = "GeneralArea",
    max_tasks_to_return_for_summary: int = 20 # Limit for reporting to supervisor
) -> List[Dict[str, Any]]:
    """
    Generates a prioritized list of CHW tasks based on input patient data (alerts, AI scores).

    Args:
        source_patient_data_df: DataFrame with patient data. Expected columns may include:
                                 patient_id, encounter_date, zone_id, condition, age,
                                 ai_risk_score, ai_followup_priority_score,
                                 alert_reason_primary (from alert_generator),
                                 min_spo2_pct, vital_signs_temperature_celsius/max_skin_temp_celsius,
                                 fall_detected_today, referral_status.
        for_date: The date for which these tasks are relevant (used for due dates, IDs).
        chw_id_context: Optional CHW ID for whom tasks are being generated or assigned.
        zone_context_str: General zone context if not available per patient.
        max_tasks_to_return_for_summary: Max tasks if this list is for a summary.

    Returns:
        List[Dict[str, Any]]: A list of task dictionaries, sorted by priority.
    """
    module_source_context = "CHWTaskProcessor"
    logger.info(f"({module_source_context}) Generating CHW tasks for date: {str(for_date)}, CHW: {chw_id_context}, Zone: {zone_context_str}")

    if source_patient_data_df is None or source_patient_data_df.empty:
        logger.info(f"({module_source_context}) No input patient data for task generation.")
        return []

    generated_tasks_list: List[Dict[str, Any]] = []
    df_task_input_source = source_patient_data_df.copy()

    # --- Ensure necessary columns for task generation exist ---
    task_gen_cols_defaults = {
        'patient_id': "UnknownPID", 'encounter_date': pd.NaT, 'zone_id': zone_context_str,
        'condition': "N/A", 'age': np.nan, 'chw_id': chw_id_context, # Assign CHW context if not per record
        'ai_risk_score': np.nan, 'ai_followup_priority_score': np.nan,
        'alert_reason_primary': "", # Assumes this comes from previous alert generation step
        'min_spo2_pct': np.nan, 'vital_signs_temperature_celsius': np.nan, 'max_skin_temp_celsius': np.nan,
        'fall_detected_today': 0, 'referral_status': "Unknown",
        'medication_adherence_self_report': "Unknown"
        # 'task_status' column might exist if updating existing tasks, default to PENDING for new
    }
    for col, default in task_gen_cols_defaults.items():
        if col not in df_task_input_source.columns:
            df_task_input_source[col] = default
        elif col in ['encounter_date']: df_task_input_source[col] = pd.to_datetime(df_task_input_source[col], errors='coerce')
        # Other type coercions/fills as needed

    # Prefer AI Follow-up Priority Score if available, then AI Risk Score for sorting input
    primary_sort_col = 'ai_followup_priority_score'
    secondary_sort_col = 'ai_risk_score'
    
    sort_cols_task_input = []
    if primary_sort_col in df_task_input_source.columns and df_task_input_source[primary_sort_col].notna().any():
        sort_cols_task_input.append(primary_sort_col)
    if secondary_sort_col in df_task_input_source.columns and df_task_input_source[secondary_sort_col].notna().any():
        if not sort_cols_task_input or sort_cols_task_input[0] != secondary_sort_col:
            sort_cols_task_input.append(secondary_sort_col)
            
    if sort_cols_task_input:
        df_sorted_for_tasks = df_task_input_source.sort_values(by=sort_cols_task_input, ascending=[False]*len(sort_cols_task_input))
    else:
        df_sorted_for_tasks = df_task_input_source
        logger.warning(f"({module_source_context}) No AI priority/risk scores for initial task input sorting.")
    
    # Determine best temperature column name based on availability in data
    temp_col_name_for_context = next((tc for tc in ['vital_signs_temperature_celsius', 'max_skin_temp_celsius'] if tc in df_sorted_for_tasks.columns and df_sorted_for_tasks[tc].notna().any()), None)

    # --- Task Generation Rules ---
    for index, record in df_sorted_for_tasks.iterrows():
        pat_id = str(record.get('patient_id', 'UnknownPID'))
        # Use encounter date if present, else 'for_date' (processing date) for due date & ID
        record_date = pd.to_datetime(record.get('encounter_date', for_date), errors='coerce').date() if pd.notna(record.get('encounter_date', for_date)) else pd.Timestamp(for_date).date()
        record_date_str = record_date.strftime('%Y%m%d')

        # Base priority on AI Follow-up Score, fallback to AI Risk, then default low
        base_priority_score = float(record.get('ai_followup_priority_score', record.get('ai_risk_score', 30.0))) # Default to 30 if no scores

        # Default task, can be overridden by specific rules
        task_desc_generated = f"Routine Checkup/Follow-up for {pat_id}"
        task_type_code_generated = "TASK_VISIT_ROUTINE" # General routine task
        task_priority_generated = base_priority_score

        # Alert-driven tasks (using 'alert_reason_primary' if available from prior alert generation step)
        alert_reason_source = str(record.get('alert_reason_primary', '')).lower()
        
        if "critical low spo2" in alert_reason_source or (pd.notna(record.get('min_spo2_pct')) and record.get('min_spo2_pct') < app_config.ALERT_SPO2_CRITICAL_LOW_PCT):
            task_desc_generated = "URGENT: Assess Critical Low SpO2"
            task_type_code_generated = "TASK_VISIT_VITALS_URGENT"
            task_priority_generated = max(base_priority_score, 98)
        elif "high fever" in alert_reason_source or \
             (temp_col_name_for_context and pd.notna(record.get(temp_col_name_for_context)) and record.get(temp_col_name_for_context) >= app_config.ALERT_BODY_TEMP_HIGH_FEVER_C):
            task_desc_generated = "URGENT: Assess High Fever"
            task_type_code_generated = "TASK_VISIT_VITALS_URGENT"
            task_priority_generated = max(base_priority_score, 95)
        elif "fall detected" in alert_reason_source or (pd.notna(record.get('fall_detected_today')) and record['fall_detected_today'] > 0):
            task_desc_generated = "Assess Patient After Fall Detection"
            task_type_code_generated = "TASK_VISIT_FALL_ASSESS"
            task_priority_generated = max(base_priority_score, 92)
        elif "pending critical referral" in alert_reason_source or \
             (str(record.get('referral_status', '')).lower() == 'pending' and any(ck.lower() in str(record.get('condition','')).lower() for ck in app_config.KEY_CONDITIONS_FOR_ACTION) ):
            task_desc_generated = f"Follow-up: Critical Referral for {record.get('condition', 'N/A')}"
            task_type_code_generated = "TASK_VISIT_REFERRAL_TRACK"
            task_priority_generated = max(base_priority_score, 88)
        elif "high ai follow-up prio" in alert_reason_source or base_priority_score >= app_config.FATIGUE_INDEX_HIGH_THRESHOLD : # FATIGUE_INDEX_HIGH_THRESHOLD as high priority general
            task_desc_generated = "Priority Follow-up: High AI Score"
            task_type_code_generated = "TASK_VISIT_FOLLOWUP_AI"
            # Priority already set by base_priority_score
        elif "poor medication adherence" in alert_reason_source or str(record.get('medication_adherence_self_report','Unknown')).lower() == 'poor':
            task_desc_generated = "Counseling: Medication Adherence"
            task_type_code_generated = "TASK_VISIT_ADHERENCE_SUPPORT"
            task_priority_generated = max(base_priority_score, 75)

        # Other task generation logic could be:
        # - Scheduled routine visits based on patient type (e.g., antenatal, chronic disease mgt)
        # - TB DOTS observation tasks
        # - Supply delivery tasks for chronic patients
        
        # Build context string for the task
        task_context_parts = []
        if pd.notna(record.get('condition')) and str(record.get('condition','N/A')).lower() not in ['unknown', 'n/a']: task_context_parts.append(f"Cond: {record.get('condition')}")
        if pd.notna(record.get('age')): task_context_parts.append(f"Age: {record.get('age'):.0f}")
        if pd.notna(record.get('min_spo2_pct')): task_context_parts.append(f"Last SpO2: {record.get('min_spo2_pct'):.0f}%")
        if temp_col_name_for_context and pd.notna(record.get(temp_col_name_for_context)): task_context_parts.append(f"Last Temp: {record.get(temp_col_name_for_context):.1f}°C")
        if pd.notna(record.get('ai_risk_score')): task_context_parts.append(f"AI Risk: {record.get('ai_risk_score'):.0f}")

        task_id_final = f"TSK_{pat_id}_{record_date_str}_{task_type_code_generated.split('_')[-1]}" # Generate unique-ish ID

        task_object = {
            "task_id": task_id_final,
            "patient_id": pat_id,
            "assigned_chw_id": str(record.get('chw_id', chw_id_context)),
            "zone_id": str(record.get('zone_id', zone_context_str)),
            "task_type_code": task_type_code_generated, # For PED pictogram mapping
            "task_description": task_desc_generated,
            "priority_score": round(task_priority_generated, 1),
            "due_date": record_date.isoformat(), # Due date is today for tasks generated from today's events
            "status": "PENDING", # Default status for new tasks
            "key_patient_context": " | ".join(task_context_parts) if task_context_parts else "General Check",
            "alert_source_info": alert_reason_source if alert_reason_source else "Routine or Score-based"
        }
        generated_tasks_list.append(task_object)

    # Final sort and de-duplication of tasks for a patient for the day
    if generated_tasks_list:
        final_tasks_df = pd.DataFrame(generated_tasks_list)
        final_tasks_df.sort_values(by="priority_score", ascending=False, inplace=True)
        # De-duplicate: if multiple rules generate similar high-priority tasks for the same patient today,
        # take the one with the absolute highest priority.
        # (E.g., "Assess Critical SpO2" might override "Follow-up High AI Prio" if SpO2 is the driver)
        final_tasks_df.drop_duplicates(subset=["patient_id", "due_date"], keep="first", inplace=True) # One top task per patient per day from this batch
        
        return final_tasks_df.head(max_tasks_to_return_for_summary).to_dict(orient='records')

    return []
