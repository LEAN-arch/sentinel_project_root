# sentinel_project_root/test/pages/chw_components_sentinel/task_processor.py
# Processes CHW data to generate a prioritized list of tasks for Sentinel.

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, date # Added date for type hinting

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

def generate_chw_prioritized_tasks(
    source_patient_data_df: Optional[pd.DataFrame],
    for_date: Any, # Typically datetime.date or compatible string
    chw_id_context: Optional[str] = "TeamDefaultCHW",
    zone_context_str: Optional[str] = "GeneralArea",
    max_tasks_to_return_for_summary: int = 20
) -> List[Dict[str, Any]]:
    """
    Generates a prioritized list of CHW tasks based on input patient data.

    Args:
        source_patient_data_df: DataFrame with patient data for task generation.
        for_date: The date for which tasks are relevant (used for due dates, IDs).
        chw_id_context: Optional CHW ID for task assignment.
        zone_context_str: General zone context.
        max_tasks_to_return_for_summary: Max tasks for summary views.

    Returns:
        List of task dictionaries, sorted by priority.
    """
    module_log_prefix = "CHWTaskProcessor"
    logger.info(f"({module_log_prefix}) Generating CHW tasks for date: {str(for_date)}, CHW: {chw_id_context}, Zone: {zone_context_str}")

    if not isinstance(source_patient_data_df, pd.DataFrame) or source_patient_data_df.empty:
        logger.info(f"({module_log_prefix}) No input patient data provided. No tasks generated.")
        return []

    generated_tasks_list: List[Dict[str, Any]] = []
    df_task_source = source_patient_data_df.copy()

    # Define expected columns and their defaults/types for task generation
    task_gen_cols_config = {
        'patient_id': {"default": "UnknownPID_Task", "type": str},
        'encounter_date': {"default": pd.NaT, "type": "datetime"},
        'zone_id': {"default": zone_context_str, "type": str},
        'condition': {"default": "N/A", "type": str},
        'age': {"default": np.nan, "type": float},
        'chw_id': {"default": chw_id_context, "type": str},
        'ai_risk_score': {"default": np.nan, "type": float},
        'ai_followup_priority_score': {"default": np.nan, "type": float},
        'alert_reason_primary': {"default": "", "type": str}, # From alert_generator.py
        'min_spo2_pct': {"default": np.nan, "type": float},
        'vital_signs_temperature_celsius': {"default": np.nan, "type": float},
        'max_skin_temp_celsius': {"default": np.nan, "type": float},
        'fall_detected_today': {"default": 0, "type": int},
        'referral_status': {"default": "Unknown", "type": str},
        'medication_adherence_self_report': {"default": "Unknown", "type": str}
    }
    common_na_strings_task = ['', 'nan', 'None', 'N/A', '#N/A', 'np.nan', 'NaT', '<NA>', 'null']

    for col, config in task_gen_cols_config.items():
        if col not in df_task_source.columns:
            df_task_source[col] = config["default"]
        
        if config["type"] == "datetime":
            df_task_source[col] = pd.to_datetime(df_task_source[col], errors='coerce')
        elif config["type"] == float:
            df_task_source[col] = pd.to_numeric(df_task_source[col], errors='coerce').fillna(config["default"])
        elif config["type"] == int:
            df_task_source[col] = pd.to_numeric(df_task_source[col], errors='coerce').fillna(config["default"]).astype(int)
        elif config["type"] == str:
            df_task_source[col] = df_task_source[col].astype(str).str.strip().replace(common_na_strings_task, config["default"], regex=False)
            df_task_source[col] = df_task_source[col].fillna(config["default"])

    # Sort input data to process higher risk/priority patients first (influences de-duplication)
    sort_by_cols = []
    if 'ai_followup_priority_score' in df_task_source.columns and df_task_source['ai_followup_priority_score'].notna().any():
        sort_by_cols.append('ai_followup_priority_score')
    if 'ai_risk_score' in df_task_source.columns and df_task_source['ai_risk_score'].notna().any():
        if not sort_by_cols or sort_by_cols[-1] != 'ai_risk_score': # Avoid duplicate if same as primary
            sort_by_cols.append('ai_risk_score')
            
    if sort_by_cols:
        df_sorted_source = df_task_source.sort_values(by=sort_by_cols, ascending=[False]*len(sort_by_cols))
    else:
        df_sorted_source = df_task_source
        logger.debug(f"({module_log_prefix}) No AI priority/risk scores for initial task input sorting.")
    
    temp_col_for_context = next((tc for tc in ['vital_signs_temperature_celsius', 'max_skin_temp_celsius'] 
                                if tc in df_sorted_source.columns and df_sorted_source[tc].notna().any()), None)

    # Standardize for_date to datetime.date for consistent due_date calculation
    try:
        processing_date_obj = pd.to_datetime(for_date, errors='raise').date()
    except Exception:
        logger.error(f"({module_log_prefix}) Invalid 'for_date' for task processing: {for_date}. Using today.")
        processing_date_obj = date.today()


    # --- Task Generation Rules ---
    for _, record in df_sorted_source.iterrows():
        pat_id_task = str(record.get('patient_id', "UnknownPID_Task"))
        
        # Use encounter_date for task if available and valid, otherwise use the general processing_date_obj
        record_specific_date = record.get('encounter_date')
        task_relevant_date = record_specific_date.date() if pd.notna(record_specific_date) else processing_date_obj
        task_relevant_date_str_id = task_relevant_date.strftime('%Y%m%d')

        # Base priority: use ai_followup_priority_score if available, else ai_risk_score, else a low default
        base_prio = 30.0 # Low default
        if pd.notna(record.get('ai_followup_priority_score')):
            base_prio = float(record.get('ai_followup_priority_score', base_prio))
        elif pd.notna(record.get('ai_risk_score')):
            base_prio = float(record.get('ai_risk_score', base_prio))

        task_desc = f"Routine Checkup/Follow-up for {pat_id_task}"
        task_type = "TASK_VISIT_ROUTINE"
        task_prio = base_prio
        alert_source_str = str(record.get('alert_reason_primary', '')).lower() # From previous alert gen step

        # Rule-based task overrides and priority boosts
        if "critical low spo2" in alert_source_str or \
           (pd.notna(record.get('min_spo2_pct')) and record.get('min_spo2_pct') < app_config.ALERT_SPO2_CRITICAL_LOW_PCT):
            task_desc = f"URGENT: Assess Critical Low SpO2 for {pat_id_task}"
            task_type = "TASK_VISIT_VITALS_URGENT"
            task_prio = max(base_prio, 98.0)
        elif "high fever" in alert_source_str or \
             (temp_col_for_context and pd.notna(record.get(temp_col_for_context)) and record.get(temp_col_for_context) >= app_config.ALERT_BODY_TEMP_HIGH_FEVER_C):
            task_desc = f"URGENT: Assess High Fever for {pat_id_task}"
            task_type = "TASK_VISIT_VITALS_URGENT"
            task_prio = max(base_prio, 95.0)
        elif "fall detected" in alert_source_str or \
             (pd.notna(record.get('fall_detected_today')) and int(record.get('fall_detected_today',0)) > 0):
            task_desc = f"Assess Patient {pat_id_task} After Fall Detection"
            task_type = "TASK_VISIT_FALL_ASSESS"
            task_prio = max(base_prio, 92.0)
        elif "pending critical referral" in alert_source_str or \
             (str(record.get('referral_status', '')).lower() == 'pending' and \
              any(ck.lower() in str(record.get('condition','')).lower() for ck in app_config.KEY_CONDITIONS_FOR_ACTION) ):
            task_desc = f"Follow-up: Critical Referral for {pat_id_task} ({record.get('condition', 'N/A')})"
            task_type = "TASK_VISIT_REFERRAL_TRACK"
            task_prio = max(base_prio, 88.0)
        elif "high ai follow-up prio" in alert_source_str or base_prio >= app_config.FATIGUE_INDEX_HIGH_THRESHOLD :
            task_desc = f"Priority Follow-up (High AI Score) for {pat_id_task}"
            task_type = "TASK_VISIT_FOLLOWUP_AI"
            # Priority already set by base_prio
        elif str(record.get('medication_adherence_self_report','Unknown')).lower() == 'poor':
            task_desc = f"Counseling: Medication Adherence for {pat_id_task}"
            task_type = "TASK_VISIT_ADHERENCE_SUPPORT"
            task_prio = max(base_prio, 75.0)
        
        # Context string for the task display
        context_parts = []
        if pd.notna(record.get('condition')) and str(record.get('condition','N/A')).lower() not in ['unknown', 'n/a', 'unknowncondition_task']: context_parts.append(f"Cond: {record.get('condition')}")
        if pd.notna(record.get('age')): context_parts.append(f"Age: {record.get('age'):.0f}")
        if pd.notna(record.get('min_spo2_pct')): context_parts.append(f"Last SpO2: {record.get('min_spo2_pct'):.0f}%")
        if temp_col_for_context and pd.notna(record.get(temp_col_for_context)): context_parts.append(f"Last Temp: {record.get(temp_col_for_context):.1f}°C")
        if pd.notna(record.get('ai_risk_score')): context_parts.append(f"AI Risk: {record.get('ai_risk_score'):.0f}")

        task_id_gen = f"TSK_{pat_id_task}_{task_relevant_date_str_id}_{task_type.split('_')[-1]}"

        task_obj_generated = {
            "task_id": task_id_gen,
            "patient_id": pat_id_task,
            "assigned_chw_id": str(record.get('chw_id', chw_id_context)),
            "zone_id": str(record.get('zone_id', zone_context_str)),
            "task_type_code": task_type, # For PED pictogram mapping
            "task_description": task_desc,
            "priority_score": round(task_prio, 1),
            "due_date": task_relevant_date.isoformat(), # Task due on the day of event/processing
            "status": "PENDING", # Default for new tasks
            "key_patient_context": " | ".join(context_parts) if context_parts else "General Check",
            "alert_source_info": alert_source_str if alert_source_str else ("AI Score Based" if base_prio > 30 else "Routine")
        }
        generated_tasks_list.append(task_obj_generated)

    if generated_tasks_list:
        final_tasks_output_df = pd.DataFrame(generated_tasks_list)
        final_tasks_output_df.sort_values(by="priority_score", ascending=False, inplace=True)
        # De-duplicate: Keep the single highest priority task per patient per due_date (day of relevance)
        final_tasks_output_df.drop_duplicates(subset=["patient_id", "due_date"], keep="first", inplace=True)
        
        logger.info(f"({module_log_prefix}) Generated {len(final_tasks_output_df)} unique prioritized tasks.")
        return final_tasks_output_df.head(max_tasks_to_return_for_summary).to_dict(orient='records')

    logger.info(f"({module_log_prefix}) No tasks generated from the provided data.")
    return []
