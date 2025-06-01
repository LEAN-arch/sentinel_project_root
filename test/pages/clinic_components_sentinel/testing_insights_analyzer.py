# sentinel_project_root/test/pages/clinic_components_sentinel/testing_insights_analyzer.py
# Prepares detailed data for laboratory testing performance and trends for Sentinel.

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List

# Standardized import block
try:
    from config import app_config
    from utils.core_data_processing import get_trend_data
except ImportError:
    import sys
    import os
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_for_utils = os.path.abspath(os.path.join(current_script_dir, os.pardir, os.pardir))
    if project_root_for_utils not in sys.path:
        sys.path.insert(0, project_root_for_utils)
    from config import app_config
    from utils.core_data_processing import get_trend_data

logger = logging.getLogger(__name__)

def prepare_clinic_testing_insights_data(
    filtered_health_df_clinic_period: Optional[pd.DataFrame],
    clinic_service_kpis_summary: Optional[Dict[str, Any]], 
    reporting_period_str: str,
    selected_test_group_display_name: str = "All Critical Tests Summary" 
) -> Dict[str, Any]:
    """
    Prepares structured data for detailed testing insights.
    """
    module_log_prefix = "ClinicTestInsightsAnalyzer" # Shortened
    logger.info(f"({module_log_prefix}) Preparing testing insights. Focus: '{selected_test_group_display_name}', Period: {reporting_period_str}")

    insights_output: Dict[str, Any] = {
        "reporting_period": reporting_period_str,
        "selected_focus_area": selected_test_group_display_name,
        "all_critical_tests_summary_table_df": None,
        "focused_test_group_kpis": None,
        "focused_test_group_tat_trend": None,
        "focused_test_group_volume_trend_df": None,
        "overdue_pending_tests_list_df": None,
        "sample_rejection_reasons_summary_df": None,
        "top_rejected_samples_examples_df": None,
        "processing_notes": []
    }

    if not isinstance(filtered_health_df_clinic_period, pd.DataFrame) or filtered_health_df_clinic_period.empty:
        note = "No health data provided for testing insights analysis."
        logger.warning(f"({module_log_prefix}) {note}")
        insights_output["processing_notes"].append(note)
        return insights_output # Cannot proceed without raw data
    
    if not isinstance(clinic_service_kpis_summary, dict) or \
       "test_summary_details" not in clinic_service_kpis_summary or \
       not isinstance(clinic_service_kpis_summary["test_summary_details"], dict):
        note = "Clinic service KPI summary or 'test_summary_details' is missing/invalid; aggregated metrics will be affected."
        logger.warning(f"({module_log_prefix}) {note}")
        insights_output["processing_notes"].append(note)
        # Allow to proceed for overdue/rejection analysis from raw data if summary is flawed
        clinic_service_kpis_summary = {"test_summary_details": {}} # Use empty for safety

    df_tests_source = filtered_health_df_clinic_period.copy()
    
    # Ensure essential columns for calculations exist with proper types/defaults
    test_cols_config = {
        'test_type': {"default": "UnknownTest_Insight", "type": str},
        'test_result': {"default": "UnknownResult_Insight", "type": str},
        'sample_status': {"default": "UnknownStatus_Insight", "type": str},
        'encounter_date': {"default": pd.NaT, "type": "datetime"}, # Primary date for trends
        'test_turnaround_days': {"default": np.nan, "type": float},
        'patient_id': {"default": "UnknownPID_Insight", "type": str},
        'sample_collection_date': {"default": pd.NaT, "type": "datetime"},
        'sample_registered_lab_date': {"default": pd.NaT, "type": "datetime"},
        'rejection_reason': {"default": "UnknownReason_Insight", "type": str}
    }
    common_na_strings_insights = ['', 'nan', 'None', 'N/A', '#N/A', 'np.nan', 'NaT', '<NA>', 'null']

    for col, config in test_cols_config.items():
        if col not in df_tests_source.columns:
            df_tests_source[col] = config["default"]
        if config["type"] == "datetime":
            df_tests_source[col] = pd.to_datetime(df_tests_source[col], errors='coerce')
        elif config["type"] == float:
            df_tests_source[col] = pd.to_numeric(df_tests_source[col], errors='coerce').fillna(config["default"])
        elif config["type"] == str:
            df_tests_source[col] = df_tests_source[col].astype(str).str.strip().replace(common_na_strings_insights, config["default"], regex=False)
            df_tests_source[col] = df_tests_source[col].fillna(config["default"])


    # --- A. Data for Selected Focus Area (All Critical or Specific Test Group) ---
    test_summary_map = clinic_service_kpis_summary.get("test_summary_details", {})

    if selected_test_group_display_name == "All Critical Tests Summary":
        crit_tests_summary_list = []
        if test_summary_map:
            for display_name, stats in test_summary_map.items():
                # Find original key to check 'critical' flag in app_config
                original_key = next((k for k, v_cfg in app_config.KEY_TEST_TYPES_FOR_ANALYSIS.items() 
                                     if v_cfg.get("display_name") == display_name), None)
                if original_key and app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get(original_key, {}).get("critical"):
                    crit_tests_summary_list.append({
                        "Test Group (Critical)": display_name,
                        "Positivity (%)": stats.get("positive_rate_perc", np.nan),
                        "Avg. TAT (Days)": stats.get("avg_tat_days", np.nan),
                        "% Met TAT Target": stats.get("perc_met_tat_target", np.nan),
                        "Pending (Patients)": stats.get("pending_count_patients", 0),
                        "Rejected (Patients)": stats.get("rejected_count_patients", 0),
                        "Total Conclusive Tests": stats.get("total_conclusive_tests", 0)
                    })
            if crit_tests_summary_list:
                insights_output["all_critical_tests_summary_table_df"] = pd.DataFrame(crit_tests_summary_list)
            else: insights_output["processing_notes"].append("No data for critical tests in summary or none configured as critical.")
        else: insights_output["processing_notes"].append("Detailed test statistics map missing for 'All Critical Tests' summary.")

    elif selected_test_group_display_name in test_summary_map: # Specific test group selected
        selected_stats = test_summary_map[selected_test_group_display_name]
        insights_output["focused_test_group_kpis"] = {
            "Positivity Rate (%)": selected_stats.get("positive_rate_perc", np.nan),
            "Avg. TAT (Days)": selected_stats.get("avg_tat_days", np.nan),
            "% Met TAT Target": selected_stats.get("perc_met_tat_target", np.nan),
            "Pending Tests (Patients)": selected_stats.get("pending_count_patients", 0),
            "Rejected Samples (Patients)": selected_stats.get("rejected_count_patients", 0),
            "Total Conclusive Tests": selected_stats.get("total_conclusive_tests",0)
        }
        
        # Trends for the specific selected test group (using raw data)
        original_key_focus = next((k for k, v_cfg in app_config.KEY_TEST_TYPES_FOR_ANALYSIS.items() 
                                   if v_cfg.get("display_name") == selected_test_group_display_name), None)
        if original_key_focus:
            # Handle cases where a "display name" might represent multiple actual test_type keys
            raw_test_keys_for_group = app_config.KEY_TEST_TYPES_FOR_ANALYSIS[original_key_focus].get("types_in_group", [original_key_focus])
            if isinstance(raw_test_keys_for_group, str): raw_test_keys_for_group = [raw_test_keys_for_group]

            # TAT Trend (Daily average)
            df_focus_tat = df_tests_source[
                (df_tests_source['test_type'].isin(raw_test_keys_for_group)) &
                (df_tests_source['test_turnaround_days'].notna()) &
                (~df_tests_source.get('test_result', pd.Series(dtype=str)).isin(
                    ['Pending','Unknown','Rejected Sample','Indeterminate', "UnknownResult_Insight"])) # Conclusive results
            ].copy()
            if not df_focus_tat.empty:
                tat_trend = get_trend_data(df_focus_tat, 'test_turnaround_days', date_col='encounter_date', 
                                           period='D', agg_func='mean', 
                                           source_context=f"{module_log_prefix}/TATTrend/{selected_test_group_display_name}")
                insights_output["focused_test_group_tat_trend"] = tat_trend if isinstance(tat_trend, pd.Series) and not tat_trend.empty else None
            
            # Volume Trend (Daily: Conclusive vs. Pending)
            df_focus_vol = df_tests_source[df_tests_source['test_type'].isin(raw_test_keys_for_group)].copy()
            if not df_focus_vol.empty and 'patient_id' in df_focus_vol.columns:
                conclusive_results_mask_vol = ~df_focus_vol.get('test_result', pd.Series(dtype=str)).isin(
                                                ['Pending','Unknown','Rejected Sample','Indeterminate', "UnknownResult_Insight"])
                
                conclusive_vol_trend = get_trend_data(df_focus_vol[conclusive_results_mask_vol], 'patient_id', 
                                                      date_col='encounter_date', period='D', agg_func='count').rename("Conclusive Tests")
                pending_vol_trend = get_trend_data(df_focus_vol[df_focus_vol.get('test_result',pd.Series(dtype=str)) == 'Pending'], 
                                                   'patient_id', date_col='encounter_date', period='D', agg_func='count').rename("Pending Tests")
                
                volume_trends_list = [s for s in [conclusive_vol_trend, pending_vol_trend] if isinstance(s, pd.Series) and not s.empty]
                if volume_trends_list:
                    insights_output["focused_test_group_volume_trend_df"] = pd.concat(volume_trends_list, axis=1).fillna(0).reset_index()
        else:
            insights_output["processing_notes"].append(f"Could not find original config for '{selected_test_group_display_name}' to generate trends.")
    else: # Neither "All Critical" nor a known specific group
        insights_output["processing_notes"].append(f"No detailed stats or config found for selected focus: '{selected_test_group_display_name}'.")

    # --- B. Overdue Pending Tests (from raw period data) ---
    # Prioritize sample_collection_date, then registered_date, then encounter_date for pending calculation
    date_col_overdue = 'encounter_date' # Default
    if 'sample_collection_date' in df_tests_source.columns and df_tests_source['sample_collection_date'].notna().any():
        date_col_overdue = 'sample_collection_date'
    elif 'sample_registered_lab_date' in df_tests_source.columns and df_tests_source['sample_registered_lab_date'].notna().any():
        date_col_overdue = 'sample_registered_lab_date'
    
    df_pending_tests = df_tests_source[
        (df_tests_source.get('test_result', pd.Series(dtype=str)) == 'Pending') & 
        (df_tests_source[date_col_overdue].notna()) # Must have a date to calculate pending days
    ].copy()

    if not df_pending_tests.empty:
        # Ensure the date column is datetime
        df_pending_tests[date_col_overdue] = pd.to_datetime(df_pending_tests[date_col_overdue], errors='coerce')
        df_pending_tests.dropna(subset=[date_col_overdue], inplace=True)

        if not df_pending_tests.empty:
            current_date_normalized = pd.Timestamp('now').normalize() # Today at midnight
            df_pending_tests['days_pending'] = (current_date_normalized - df_pending_tests[date_col_overdue]).dt.days
            
            def get_overdue_threshold(test_type_str: str) -> int:
                test_config = app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get(test_type_str)
                buffer = 2 # Allowable buffer beyond target TAT
                return (test_config['target_tat_days'] + buffer) if test_config and 'target_tat_days' in test_config \
                       else (app_config.OVERDUE_PENDING_TEST_DAYS_GENERAL_FALLBACK + buffer)
            
            df_pending_tests['overdue_after_days'] = df_pending_tests['test_type'].apply(get_overdue_threshold)
            df_overdue = df_pending_tests[df_pending_tests['days_pending'] > df_pending_tests['overdue_after_days']]
            
            if not df_overdue.empty:
                cols_to_show_overdue = ['patient_id', 'test_type', date_col_overdue, 'days_pending', 'overdue_after_days']
                df_overdue_display = df_overdue.rename(columns={date_col_overdue:"Sample/Registered Date"})
                # Filter for existing columns before selecting
                display_cols_final = [col for col in ['patient_id', 'test_type', "Sample/Registered Date", 'days_pending', 'overdue_after_days'] if col in df_overdue_display.columns]
                insights_output["overdue_pending_tests_list_df"] = df_overdue_display[display_cols_final].sort_values('days_pending', ascending=False)
            else: insights_output["processing_notes"].append("No tests currently pending longer than their TAT + buffer.")
        else: insights_output["processing_notes"].append("No valid pending tests with dates for overdue calculation after cleaning.")
    else: insights_output["processing_notes"].append("No pending tests found in the period for overdue status evaluation.")

    # --- C. Sample Rejection Analysis (From raw period data) ---
    if 'sample_status' in df_tests_source.columns and 'rejection_reason' in df_tests_source.columns:
        df_rejected_samples = df_tests_source[df_tests_source.get('sample_status', pd.Series(dtype=str)) == 'Rejected'].copy()
        if not df_rejected_samples.empty:
            df_rejected_samples['rejection_reason_clean'] = df_rejected_samples['rejection_reason'].fillna('Unknown Reason').astype(str).str.strip()
            df_rejected_samples.loc[df_rejected_samples['rejection_reason_clean'].isin(common_na_strings_insights + ["UnknownReason_Insight"]), 'rejection_reason_clean'] = 'Unknown Reason'
            
            rejection_counts = df_rejected_samples['rejection_reason_clean'].value_counts().reset_index()
            rejection_counts.columns = ['Rejection Reason', 'Count']
            insights_output["sample_rejection_reasons_summary_df"] = rejection_counts

            # Example rejected samples list
            cols_for_rejected_examples = ['patient_id', 'test_type', 'encounter_date', 'rejection_reason_clean']
            if 'sample_collection_date' in df_rejected_samples.columns: 
                cols_for_rejected_examples.insert(2, 'sample_collection_date') # Add if available
            insights_output["top_rejected_samples_examples_df"] = df_rejected_samples[
                [col for col in cols_for_rejected_examples if col in df_rejected_samples.columns] # Select only existing cols
            ].head(15)
        else: insights_output["processing_notes"].append("No rejected samples recorded in this period for analysis.")
    else: insights_output["processing_notes"].append("Sample status or rejection reason data columns missing for rejection analysis.")
    
    logger.info(f"({module_log_prefix}) Clinic testing insights data preparation finished. Notes: {len(insights_output['processing_notes'])}")
    return insights_output
