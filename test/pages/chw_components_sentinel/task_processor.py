# sentinel_project_root/test/pages/chw_components_sentinel/task_processor.py
# Processes CHW data to generate a prioritized list of tasks for Sentinel.

import pandas as pd
import geopandas as gpd
import numpy as np
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, date # For date handling

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

def generate_chw_prioritized_tasks(
    source_patient_data_df: Optional[pd.DataFrame], # Dataframe containing potential task triggers
    for_date: Any, # Typically datetime.date or compatible string, for context and default due dates
    chw_id_context: Optional[str] = "TeamDefaultCHW", # For assigning task if not inherent in data
    zone_context_str: Optional[str] = "GeneralArea",   # General zone context if not in record
    max_tasks_to_return_for_summary: int = 20 # Limit for reporting to supervisor
) -> List[Dict[str, Any]]:
    """
    Generates a prioritized list of CHW tasks based on input patient data (alerts, AI scores).

    Args:
        source_patient_data_df: DataFrame with patient data.
        for_date: The date for which these tasks are relevant.
        chw_id_context: Optional CHW ID for task assignment.
        zone_context_str: General zone context.
        max_tasks_to_return_for_summary: Max tasks if this list is for a summary view.

    Returns:
        List of task dictionaries, sorted by priority.
    """
    module_log_prefix = "CHWTaskProcessor" # Consistent prefix
    logger.info(f"({module_log_prefix}) Generating CHW tasks for date: {str(for_date)}, CHW: {chw_id_context}, Zone: {zone_context_str}")

    if not isinstance(source_patient_data_df, pd.DataFrame) or source_patient_data_df.empty:
        logger.info(f"({module_log_prefix}) No input patient data provided. No tasks generated.")
        return []

    tasks_buffer: List[Dict[str, Any]] = [] # Renamed from generated_tasks_list
    df_task_src = source_patient_data_df.copy() # Renamed from df_task_input_source

    # Define expected columns, their defaults, and types for robust task generation
    task_cols_config = { # Renamed from task_gen_cols_defaults
        'patient_id': {"default": "UnknownPID_TaskProc", "type": str},
        'encounter_date': {"default": pd.NaT, "type": "datetime"},
        'zone_id': {"default": zone_context_str, "type": str},
        'condition': {"default": "N/A_TaskProc", "type": str},
        'age': {"default": np.nan, "type": float},
        'chw_id': {"default": chw_id_context, "type": str}, # Assign CHW context if not per record
        'ai_risk_score': {"default": np.nan, "type": float},
        'ai_followup_priority_score': {"default": np.nan, "type": float},
        'alert_reason_primary': {"default": "", "type": str}, # Assumes this comes from alert_generator.py
        'min_spo2_pct': {"default": np.nan, "type": float},
        'vital_signs_temperature_celsius': {"default": np.nan, "type": float},
        'max_skin_temp_celsius': {"default": np.nan, "type": float},
        'fall_detected_today': {"default": 0, "type": int},
        'referral_status': {"default": "Unknown_TaskProc", "type": str},
        'medication_adherence_self_report': {"default": "Unknown_TaskProc", "type": str}
    }
    common_na_strings_task_proc = ['', 'nan', 'None', 'N/A', '#N/A', 'np.nan', 'NaT', '<NA>', 'null']

    for col, config_val in task_cols_config.items():
        if col not in df_task_src.columns:
            df_task_src[col] = config_val["default"]
        
        if config_val["type"] == "datetime":
            df_task_src[col] = pd.to_datetime(df_task_src[col], errors='coerce')
        elif config_val["type"] == float:
            df_task_src[col] = pd.to_numeric(df_task_src[col], errors='coerce').fillna(config_val["default"])
        elif config_val["type"] == int: # For flags
            df_task_src[col] = pd.to_numeric(df_task_src[col], errors='coerce').fillna(config_val["default"]).astype(int)
        elif config_val["type"] == str:
            df_task_src[col] = df_task_src[col].astype(str).str.strip().replace(common_na_strings_task_proc, config_val["default"], regex=False)
            df_task_src[col] = df_task_src[col].fillna(config_val["default"])


    # Sort input data to process higher risk/priority patients first (influences de-duplication outcome)
    sort_columns_task = []
    if 'ai_followup_priority_score' in df_task_src.columns and df_task_src['ai_followup_priority_score'].notna().any():
        sort_columns_task.append('ai_followup_priority_score')
    if 'ai_risk_score' in df_task_src.columns and df_task_src['ai_risk_score'].notna().any():
        if not sort_columns_task or sort_columns_task[-1] != 'ai_risk_score': # Avoid duplicate if same as primary
            sort_columns_task.append('ai_risk_score')
            
    if sort_columns_task:
        df_sorted_for_task_gen = df_task_src.sort_values(by=sort_columns_task, ascending=[False]*len(sort_columns_task))
    else:
        df_sorted_for_task_gen = df_task_src
        logger.debug(f"({module_log_prefix}) No AI priority/risk scores available for initial task input sorting.")
    
    temp_col_for_task_context = next((tc for tc in ['vital_signs_temperature_celsius', 'max_skin_temp_celsius'] 
                                      if tc in df_sorted_for_task_gen.columns and df_sorted_for_task_gen[tc].notna().any()), None)

    # Standardize `for_date` to a datetime.date object for consistent due date calculation
    try:
        task_processing_date = pd.to_datetime(for_date, errors='raise').date()
    except Exception:
        logger.error(f"({module_log_prefix}) Invalid 'for_date' for task processing: {for_date}. Defaulting to today.")
        task_processing_date = date.today()

    # --- Task Generation Rules Iteration ---
    for _, record_data in df_sorted_for_task_gen.iterrows():
        patient_id_val = str(record_data.get('patient_id', task_cols_config['patient_id']['default']))
        
        # Use encounter_date for task if available and valid, otherwise use the general task_processing_date
        record_encounter_date = record_data.get('encounter_date')
        task_date_for_record = record_encounter_date.date() if pd.notna(record_encounter_date) else task_processing_date
        task_date_str_for_id = task_date_for_record.strftime('%Y%m%d') # For unique task ID

        # Base priority: use ai_followup_priority_score if available, else ai_risk_score, else a low default
        base_priority_task = 30.0 # Low default priority
        if pd.notna(record_data.get('ai_followup_priority_score')):
            base_priority_task = float(record_data.get('ai_followup_priority_score', base_priority_task))
        elif pd.notna(record_data.get('ai_risk_score')):
            base_priority_task = float(record_data.get('ai_risk_score', base_priority_task))

        # Default task, can be overridden by specific rules below
        generated_task_description = f"Routine Checkup/Follow-up for {patient_id_val}"
        generated_task_type_code = "TASK_VISIT_ROUTINE"
        generated_task_priority = base_priority_task
        alert_reason_from_source = str(record_data.get('alert_reason_primary', '')).lower() # From alert_generator output

        # Apply rules to potentially override default task and boost priority
        if "critical low spo2" in alert_reason_from_source or \
           (pd.notna(record_data.get('min_spo2_pct')) and record_data.get('min_spo2_pct') < app_config.ALERT_SPO2_CRITICAL_LOW_PCT):
            generated_task_description = f"URGENT: Assess Critical Low SpO2 for {patient_id_val}"
            generated_task_type_code = "TASK_VISIT_VITALS_URGENT"
            generated_task_priority = max(base_priority_task, 98.0) # High priority
        elif "high fever" in alert_reason_from_source or \
             (temp_col_for_task_context and pd.notna(record_data.get(temp_col_for_task_context)) and \
              record_data.get(temp_col_for_task_context) >= app_config.ALERT_BODY_TEMP_HIGH_FEVER_C):
            generated_task_description = f"URGENT: Assess High Fever for {patient_id_val}"
            generated_task_type_code = "TASK_VISIT_VITALS_URGENT"
            generated_task_priority = max(base_priority_task, 95.0)
        elif "fall detected" in alert_reason_from_source or \
             (pd.notna(record_data.get('fall_detected_today')) and int(record_data.get('fall_detected_today',0)) > 0):
            generated_task_description = f"Assess Patient {patient_id_val} After Fall Detection"
            generated_task_type_code = "TASK_VISIT_FALL_ASSESS"
            generated_task_priority = max(base_priority_task, 92.0)
        elif "pending critical referral" in alert_reason_from_source or \
             (str(record_data.get('referral_status', '')).lower() == 'pending' and \
              any(ck.lower() in str(record_data.get('condition','')).lower() for ck in app_config.KEY_CONDITIONS_FOR_ACTION) ):
            generated_task_description = f"Follow-up: Critical Referral for {patient_id_val} ({record_data.get('condition', 'N/A')})"
            generated_task_type_code = "TASK_VISIT_REFERRAL_TRACK"
            generated_task_priority = max(base_priority_task, 88.0)
        elif "high ai follow-up prio" in alert_reason_from_source or base_priority_task >= app_config.FATIGUE_INDEX_HIGH_THRESHOLD :
            generated_task_description = f"Priority Follow-up (High AI Score) for {patient_id_val}"
            generated_task_type_code = "TASK_VISIT_FOLLOWUP_AI"
            # Priority is already set by base_priority_task if it's high
        elif str(record_data.get('medication_adherence_self_report','Unknown')).lower() == 'poor':
            generated_task_description = f"Support: Medication Adherence for {patient_id_val}" # Changed from Counseling for broader scope
            generated_task_type_code = "TASK_VISIT_ADHERENCE_SUPPORT"
            generated_task_priority = max(base_priority_task, 75.0)
        
        # Build context string for the task display
        task_context_info_parts = []
        if pd.notna(record_data.get('condition')) and str(record_data.get('condition','N/A_TaskProc')).lower() not in ['unknown', 'n/a', 'n/a_taskproc', 'unknowncondition_taskproc']:
            task_context_info_parts.append(f"Cond: {record_data.get('condition')}")
        if pd.notna(record_data.get('age')): task_context_info_parts.append(f"Age: {record_data.get('age'):.0f}")
        if pd.notna(record_data.get('min_spo2_pct')): task_context_info_parts.append(f"Last SpO2: {record_data.get('min_spo2_pct'):.0f}%")
        if temp_col_for_task_context and pd.notna(record_data.get(temp_col_for_task_context)): task_context_info_parts.append(f"Last Temp: {record_data.get(temp_col_for_task_context):.1f}°C")
        if pd.notna(record_data.get('ai_risk_score')): task_context_info_parts.append(f"AI Risk: {record_data.get('ai_risk_score'):.0f}")

        # Generate a somewhat unique task ID
        task_id_final_val = f"TSK_{patient_id_val}_{task_date_str_for_id}_{generated_task_type_code.split('_')[-1]}"

        task_item_object = {
            "task_id": task_id_final_val,
            "patient_id": patient_id_val,
            "assigned_chw_id": str(record_data.get('chw_id', chw_id_context)), # Use record's CHW ID if available
            "zone_id": str(record_data.get('zone_id', zone_context_str)),     # Use record's Zone ID if available
            "task_type_code": generated_task_type_code,
            "task_description": generated_task_description,
            "priority_score": round(generated_task_priority, 1),
            "due_date": task_date_for_record.isoformat(), # Task due on the day of event/processing
            "status": "PENDING", # Default status for newly generated tasks
            "key_patient_context": " | ".join(task_context_info_parts) if task_context_info_parts else "General Check Required",
            "alert_source_info": alert_reason_from_source if alert_reason_from_source else ("AI Score Based" if base_priority_task > 40 else "Routine Schedule") # More descriptive source
        }
        tasks_buffer.append(task_item_object)

    # Final sort and de-duplication of tasks for a patient for the day
    if tasks_buffer:
        final_tasks_df_for_output = pd.DataFrame(tasks_buffer)
        final_tasks_df_for_output.sort_values(by="priority_score", ascending=False, inplace=True)
        
        # De-duplicate: Keep the single highest priority task per patient per due_date (day of relevance)
        # This ensures that if multiple rules generate tasks for the same patient on the same day, only the most pressing one is kept.
        final_tasks_df_for_output.drop_duplicates(subset=["patient_id", "due_date"], keep="first", inplace=True)
        
        logger.info(f"({module_log_prefix}) Generated {len(final_tasks_df_for_output)} unique prioritized tasks for the period/context.")
        return final_tasks_df_for_output.head(max_tasks_to_return_for_summary).to_dict(orient='records')

    logger.info(f"({module_log_prefix}) No tasks generated from the provided data after processing rules.")
    return []
