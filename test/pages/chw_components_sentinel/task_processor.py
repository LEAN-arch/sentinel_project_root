# sentinel_project_root/test/pages/chw_components_sentinel/task_processor.py
# Processes CHW data to generate a prioritized list of tasks for Sentinel.

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional
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

def generate_chw_prioritized_tasks(
    source_patient_data_df: Optional[pd.DataFrame],
    for_date: Any,
    chw_id_context: Optional[str] = "TeamDefaultCHW",
    zone_context_str: Optional[str] = "GeneralArea",
    max_tasks_to_return_for_summary: int = 20
) -> List[Dict[str, Any]]:
    """
    Generates a prioritized list of CHW tasks based on input patient data.

    Args:
        source_patient_data_df: DataFrame with patient data.
        for_date: The date for which these tasks are relevant.
        chw_id_context: Optional CHW ID for task assignment.
        zone_context_str: General zone context.
        max_tasks_to_return_for_summary: Max tasks for summary view.

    Returns:
        List of task dictionaries, sorted by priority.

    Raises:
        ValueError: If input data or date is invalid.
    """
    module_log_prefix = "CHWTaskProcessor"

    # Validate app_config attributes
    required_configs = [
        'ALERT_SPO2_CRITICAL_LOW_PCT', 'ALERT_BODY_TEMP_HIGH_FEVER_C',
        'FATIGUE_INDEX_HIGH_THRESHOLD', 'KEY_CONDITIONS_FOR_ACTION'
    ]
    for attr in required_configs:
        if not hasattr(app_config, attr):
            logger.error(f"({module_log_prefix}) Missing config: {attr}")
            raise ValueError(f"Missing required configuration: {attr}")

    # Standardize for_date
    try:
        task_processing_date = for_date if isinstance(for_date, date) else pd.to_datetime(for_date, errors='coerce').date()
        if not task_processing_date:
            raise ValueError
    except (ValueError, TypeError):
        logger.warning(f"({module_log_prefix}) Invalid for_date: {for_date}. Using today.")
        task_processing_date = date.today()

    logger.info(f"({module_log_prefix}) Generating CHW tasks for date: {task_processing_date}, CHW: {chw_id_context}, Zone: {zone_context_str}")

    if not isinstance(source_patient_data_df, pd.DataFrame) or source_patient_data_df.empty:
        logger.warning(f"({module_log_prefix}) No valid patient data provided.")
        raise ValueError("No patient data available for task generation.")

    # Define expected columns
    task_cols_config = {
        'patient_id': {"default": "UnknownPID_TaskProc", "type": str},
        'encounter_date': {"default": pd.NaT, "type": "datetime"},
        'zone_id': {"default": zone_context_str, "type": str},
        'condition': {"default": "N/A_TaskProc", "type": str},
        'age': {"default": np.nan, "type": "float"},
        'chw_id': {"default": chw_id_context, "type": str},
        'ai_risk_score': {"default": np.nan, "type": "float"},
        'ai_followup_priority_score': {"default": np.nan, "type": "float"},
        'alert_reason_primary': {"default": "", "type": str},
        'min_spo2_pct': {"default": np.nan, "type": "float"},
        'vital_signs_temperature_celsius': {"default": np.nan, "type": "float"},
        'max_skin_temp_celsius': {"default": np.nan, "type": "float"},
        'fall_detected_today': {"default": 0, "type": "int"},
        'referral_status': {"default": "Unknown_TaskProc", "type": str},
        'medication_adherence_self_report': {"default": "Unknown_TaskProc", "type": str}
    }
    common_na_strings = ['', 'nan', 'None', 'N/A', '#N/A', 'np.nan', 'NaT', '<NA>', 'null']

    # Prepare DataFrame
    required_cols = list(task_cols_config.keys())
    df_task_src = source_patient_data_df[required_cols].copy() if all(col in source_patient_data_df.columns for col in required_cols) else source_patient_data_df.copy()

    for col, config in task_cols_config.items():
        if col not in df_task_src.columns:
            df_task_src[col] = config["default"]
        if config["type"] == "datetime":
            df_task_src[col] = pd.to_datetime(df_task_src[col], errors='coerce')
        elif config["type"] == "float":
            df_task_src[col] = pd.to_numeric(df_task_src[col], errors='coerce').fillna(config["default"])
        elif config["type"] == "int":
            df_task_src[col] = pd.to_numeric(df_task_src[col], errors='coerce').fillna(config["default"]).astype(int)
        elif config["type"] == str:
            df_task_src[col] = df_task_src[col].astype(str).replace(common_na_strings, config["default"])

    # Select temperature column
    temp_col_name = 'vital_signs_temperature_celsius' if 'vital_signs_temperature_celsius' in df_task_src.columns else \
                    'max_skin_temp_celsius' if 'max_skin_temp_celsius' in df_task_src.columns else None

    # Sort DataFrame
    sort_columns = ['ai_followup_priority_score', 'ai_risk_score']
    sort_columns = [col for col in sort_columns if col in df_task_src.columns and df_task_src[col].notna().any()]
    df_filtered = df_task_src.sort_values(by=sort_columns, ascending=[False]*len(sort_columns)) if sort_columns else df_task_src

    # Generate tasks
    tasks = []
    for _, row in df_filtered.iterrows():
        patient_id = str(row['patient_id'])
        encounter_date = row['encounter_date'].date() if pd.notna(row['encounter_date']) else task_processing_date
        task_date_str = encounter_date.strftime('%Y%m%d')
        base_priority = float(row['ai_followup_priority_score']) if pd.notna(row['ai_followup_priority_score']) else \
                        float(row['ai_risk_score']) if pd.notna(row['ai_risk_score']) else 30.0
        alert_reason = str(row['alert_reason_primary']).lower()

        # Default task
        task_desc = f"Routine Checkup/Follow-up for {patient_id}"
        task_type = "TASK_VISIT_ROUTINE"
        priority = base_priority

        # Task rules
        if "critical low spo2" in alert_reason or (pd.notna(row['min_spo2_pct']) and row['min_spo2_pct'] < app_config.ALERT_SPO2_CRITICAL_LOW_PCT):
            task_desc = f"URGENT: Assess Critical Low SpO2 for {patient_id}"
            task_type = "TASK_VISIT_VITALS_URGENT"
            priority = max(base_priority, 98.0)
        elif "high fever" in alert_reason or (temp_col_name and pd.notna(row[temp_col_name]) and row[temp_col_name] >= app_config.ALERT_BODY_TEMP_HIGH_FEVER_C):
            task_desc = f"URGENT: Assess High Fever for {patient_id}"
            task_type = "TASK_VISIT_VITALS_URGENT"
            priority = max(base_priority, 95.0)
        elif "fall detected" in alert_reason or (pd.notna(row['fall_detected_today']) and row['fall_detected_today'] > 0):
            task_desc = f"Assess Patient {patient_id} After Fall Detection"
            task_type = "TASK_VISIT_FALL_ASSESS"
            priority = max(base_priority, 92.0)
        elif "pending critical referral" in alert_reason or (
            row['referral_status'].lower() == 'pending' and 
            any(ck.lower() in str(row['condition']).lower() for ck in app_config.KEY_CONDITIONS_FOR_ACTION)
        ):
            task_desc = f"Follow-up: Critical Referral for {patient_id} ({row['condition']})"
            task_type = "TASK_VISIT_REFERRAL_TRACK"
            priority = max(base_priority, 88.0)
        elif "high ai follow-up prio" in alert_reason or base_priority >= app_config.FATIGUE_INDEX_HIGH_THRESHOLD:
            task_desc = f"Priority Follow-up (High AI Score) for {patient_id}"
            task_type = "TASK_VISIT_FOLLOWUP_AI"
        elif row['medication_adherence_self_report'].lower() == 'poor':
            task_desc = f"Support: Medication Adherence for {patient_id}"
            task_type = "TASK_VISIT_ADHERENCE_SUPPORT"
            priority = max(base_priority, 75.0)

        # Build context
        context_parts = []
        if row['condition'].lower() not in ['unknown', 'n/a', 'n/a_taskproc']:
            context_parts.append(f"Cond: {row['condition']}")
        if pd.notna(row['age']):
            context_parts.append(f"Age: {row['age']:.0f}")
        if pd.notna(row['min_spo2_pct']):
            context_parts.append(f"Last SpO2: {row['min_spo2_pct']:.0f}%")
        if temp_col_name and pd.notna(row[temp_col_name]):
            context_parts.append(f"Last Temp: {row[temp_col_name]:.1f}°C")
        if pd.notna(row['ai_risk_score']):
            context_parts.append(f"AI Risk: {row['ai_risk_score']:.0f}")

        task = {
            "task_id": f"TSK_{patient_id}_{task_date_str}_{task_type.split('_')[-1]}_{len(tasks)}",
            "patient_id": patient_id,
            "assigned_chw_id": str(row['chw_id']),
            "zone_id": str(row['zone_id']),
            "task_type_code": task_type,
            "task_description": task_desc,
            "priority_score": round(priority, 1),
            "due_date": encounter_date.isoformat(),
            "status": "PENDING",
            "key_patient_context": " | ".join(context_parts) if context_parts else "General Check Required",
            "alert_source_info": alert_reason if alert_reason else ("AI Score Based" if base_priority > 40 else "Routine Schedule")
        }
        tasks.append(task)

    # Deduplicate tasks
    if tasks:
        tasks_dict = {}
        for task in tasks:
            key = (task['patient_id'], task['due_date'])
            if key not in tasks_dict or task['priority_score'] > tasks_dict[key]['priority_score']:
                tasks_dict[key] = task
        
        final_tasks = sorted(tasks_dict.values(), key=lambda x: x['priority_score'], reverse=True)
        logger.info(f"({module_log_prefix}) Generated {len(final_tasks)} unique prioritized tasks.")
        return final_tasks[:max_tasks_to_return_for_summary]
    
    logger.info(f"({module_log_prefix}) No tasks generated.")
    return []
