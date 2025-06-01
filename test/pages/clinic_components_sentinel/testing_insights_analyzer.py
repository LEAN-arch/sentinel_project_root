# sentinel_project_root/test/pages/clinic_components_sentinel/testing_insights_analyzer.py
# Prepares detailed data for laboratory testing performance and trends for Sentinel.

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List

# Standardized import block
try:
    from config import app_config
    from utils.core_data_processing import get_trend_data # For TAT/Volume trends
except ImportError:
    import sys
    import os
    # Assumes this file is in sentinel_project_root/test/pages/clinic_components_sentinel/
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_test_root_dir = os.path.abspath(os.path.join(current_script_dir, os.pardir, os.pardir))
    if project_test_root_dir not in sys.path:
        sys.path.insert(0, project_test_root_dir)
    from config import app_config
    from utils.core_data_processing import get_trend_data

logger = logging.getLogger(__name__)

def prepare_clinic_testing_insights_data(
    filtered_health_df_clinic_period: Optional[pd.DataFrame],
    clinic_service_kpis_summary: Optional[Dict[str, Any]], # Contains 'test_summary_details'
    reporting_period_str: str,
    selected_test_group_display_name: str = "All Critical Tests Summary" # Default focus
) -> Dict[str, Any]:
    """
    Prepares structured data for detailed testing insights, including performance metrics,
    trends, overdue tests, and sample rejection analysis.
    """
    module_log_prefix = "ClinicTestInsightsAnalyzer" # Consistent prefix
    logger.info(f"({module_log_prefix}) Preparing testing insights. Focus: '{selected_test_group_display_name}', Period: {reporting_period_str}")

    # Initialize output structure with defaults, especially for DataFrames
    insights_output: Dict[str, Any] = {
        "reporting_period": reporting_period_str,
        "selected_focus_area": selected_test_group_display_name,
        "all_critical_tests_summary_table_df": pd.DataFrame(), # Empty DF default
        "focused_test_group_kpis": None,           # Dict of KPIs for a specific selected group
        "focused_test_group_tat_trend": None,      # pd.Series for TAT trend
        "focused_test_group_volume_trend_df": None,  # pd.DataFrame for Volume (Conclusive, Pending)
        "overdue_pending_tests_list_df": pd.DataFrame(),       # Empty DF default
        "sample_rejection_reasons_summary_df": pd.DataFrame(), # Empty DF default
        "top_rejected_samples_examples_df": pd.DataFrame(),    # Empty DF default
        "processing_notes": []
    }

    if not isinstance(filtered_health_df_clinic_period, pd.DataFrame) or filtered_health_df_clinic_period.empty:
        note = "No health data provided for testing insights analysis. All outputs will be empty/default."
        logger.warning(f"({module_log_prefix}) {note}")
        insights_output["processing_notes"].append(note)
        return insights_output # Cannot proceed without raw data
    
    # Validate clinic_service_kpis_summary and its critical sub-dictionary
    if not isinstance(clinic_service_kpis_summary, dict) or \
       "test_summary_details" not in clinic_service_kpis_summary or \
       not isinstance(clinic_service_kpis_summary.get("test_summary_details"), dict):
        note = "Clinic service KPI summary or its 'test_summary_details' is missing/invalid. Aggregated metrics for test groups will be unavailable."
        logger.warning(f"({module_log_prefix}) {note}")
        insights_output["processing_notes"].append(note)
        # Allow to proceed for overdue/rejection analysis from raw data if summary is flawed.
        # Ensure test_summary_details exists as an empty dict for safe access later.
        if not isinstance(clinic_service_kpis_summary, dict): clinic_service_kpis_summary = {}
        if not isinstance(clinic_service_kpis_summary.get("test_summary_details"), dict):
            clinic_service_kpis_summary["test_summary_details"] = {}
            
    df_tests_src = filtered_health_df_clinic_period.copy() # Work on a copy
    
    # Ensure essential columns for calculations exist with proper types/defaults
    test_analysis_cols_config = { # Renamed for clarity
        'test_type': {"default": "UnknownTest_InsightPrep", "type": str},
        'test_result': {"default": "UnknownResult_InsightPrep", "type": str},
        'sample_status': {"default": "UnknownStatus_InsightPrep", "type": str},
        'encounter_date': {"default": pd.NaT, "type": "datetime"}, # Primary date for trends
        'test_turnaround_days': {"default": np.nan, "type": float},
        'patient_id': {"default": "UnknownPID_InsightPrep", "type": str},
        'sample_collection_date': {"default": pd.NaT, "type": "datetime"},
        'sample_registered_lab_date': {"default": pd.NaT, "type": "datetime"},
        'rejection_reason': {"default": "UnknownReason_InsightPrep", "type": str}
    }
    common_na_strings_insights_prep = ['', 'nan', 'None', 'N/A', '#N/A', 'np.nan', 'NaT', '<NA>', 'null']

    for col, config_item in test_analysis_cols_config.items():
        if col not in df_tests_src.columns:
            df_tests_src[col] = config_item["default"]
        # Type coercion and standardized NA handling for existing columns
        if config_item["type"] == "datetime":
            df_tests_src[col] = pd.to_datetime(df_tests_src[col], errors='coerce')
        elif config_item["type"] == float:
            df_tests_src[col] = pd.to_numeric(df_tests_src[col], errors='coerce').fillna(config_item["default"])
        elif config_item["type"] == str:
            df_tests_src[col] = df_tests_src[col].astype(str).str.strip().replace(common_na_strings_insights_prep, config_item["default"], regex=False)
            df_tests_src[col] = df_tests_src[col].fillna(config_item["default"])


    # --- A. Data for Selected Focus Area (All Critical or Specific Test Group) ---
    test_summary_map_from_kpis = clinic_service_kpis_summary.get("test_summary_details", {})

    if selected_test_group_display_name == "All Critical Tests Summary":
        critical_tests_summary_list_for_df = []
        if test_summary_map_from_kpis: # Only proceed if detailed stats map is available
            for disp_name_iter, stats_data_iter in test_summary_map_from_kpis.items():
                # Find original key to check 'critical' flag in app_config.KEY_TEST_TYPES_FOR_ANALYSIS
                original_test_key_iter = next((k_orig for k_orig, v_cfg_iter in app_config.KEY_TEST_TYPES_FOR_ANALYSIS.items() 
                                               if v_cfg_iter.get("display_name") == disp_name_iter), None)
                if original_test_key_iter and app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get(original_test_key_iter, {}).get("critical"):
                    critical_tests_summary_list_for_df.append({
                        "Test Group (Critical)": disp_name_iter, # User-friendly display name
                        "Positivity (%)": stats_data_iter.get("positive_rate_perc", np.nan),
                        "Avg. TAT (Days)": stats_data_iter.get("avg_tat_days", np.nan),
                        "% Met TAT Target": stats_data_iter.get("perc_met_tat_target", np.nan),
                        "Pending (Patients)": stats_data_iter.get("pending_count_patients", 0),
                        "Rejected (Patients)": stats_data_iter.get("rejected_count_patients", 0),
                        "Total Conclusive Tests": stats_data_iter.get("total_conclusive_tests", 0)
                    })
            if critical_tests_summary_list_for_df:
                insights_output["all_critical_tests_summary_table_df"] = pd.DataFrame(critical_tests_summary_list_for_df)
            else: insights_output["processing_notes"].append("No data for critical tests found in summary or none configured as critical.")
        else: insights_output["processing_notes"].append("Detailed test statistics map ('test_summary_details') missing for 'All Critical Tests' summary.")

    elif selected_test_group_display_name in test_summary_map_from_kpis: # A specific test group is selected
        stats_for_selected_group = test_summary_map_from_kpis[selected_test_group_display_name]
        insights_output["focused_test_group_kpis"] = { # Store as a dict for easy access by UI
            "Positivity Rate (%)": stats_for_selected_group.get("positive_rate_perc", np.nan),
            "Avg. TAT (Days)": stats_for_selected_group.get("avg_tat_days", np.nan),
            "% Met TAT Target": stats_for_selected_group.get("perc_met_tat_target", np.nan),
            "Pending Tests (Patients)": stats_for_selected_group.get("pending_count_patients", 0),
            "Rejected Samples (Patients)": stats_for_selected_group.get("rejected_count_patients", 0),
            "Total Conclusive Tests": stats_for_selected_group.get("total_conclusive_tests",0)
        }
        
        # For trends, we need to use the raw `df_tests_src` filtered by original test keys
        original_key_for_focused_group = next((k_orig_focus for k_orig_focus, v_cfg_focus in app_config.KEY_TEST_TYPES_FOR_ANALYSIS.items() 
                                               if v_cfg_focus.get("display_name") == selected_test_group_display_name), None)
        if original_key_for_focused_group:
            test_config_for_focused_group = app_config.KEY_TEST_TYPES_FOR_ANALYSIS[original_key_for_focused_group]
            # Handle cases where a "display name" might represent multiple actual test_type keys (e.g., "TB Microscopy" might include AFB Smear, Culture)
            actual_raw_test_keys_for_group = test_config_for_focused_group.get("types_in_group", [original_key_for_focused_group]) 
            if isinstance(actual_raw_test_keys_for_group, str): actual_raw_test_keys_for_group = [actual_raw_test_keys_for_group]

            # TAT Trend for selected group (Daily average TAT)
            if 'test_turnaround_days' in df_tests_src.columns and 'encounter_date' in df_tests_src.columns:
                df_focused_group_tat_src = df_tests_src[
                    (df_tests_src['test_type'].isin(actual_raw_test_keys_for_group)) &
                    (df_tests_src['test_turnaround_days'].notna()) & # Must have TAT value
                    (~df_tests_src.get('test_result', pd.Series(dtype=str)).isin( # Conclusive results only
                        ['Pending','Unknown','Rejected Sample','Indeterminate', "UnknownResult_InsightPrep"]))
                ].copy()
                if not df_focused_group_tat_src.empty:
                    series_tat_trend_for_group = get_trend_data(df_focused_group_tat_src, 'test_turnaround_days', 
                                                                date_col='encounter_date', period='D', agg_func='mean', 
                                                                source_context=f"{module_log_prefix}/TATTrend/{selected_test_group_display_name}")
                    insights_output["focused_test_group_tat_trend"] = series_tat_trend_for_group if isinstance(series_tat_trend_for_group, pd.Series) and not series_tat_trend_for_group.empty else None
            
            # Volume Trend (Daily: Conclusive vs. Pending) for selected group
            if 'patient_id' in df_tests_src.columns and 'encounter_date' in df_tests_src.columns: # patient_id for count/nunique
                df_focused_group_vol_src = df_tests_src[df_tests_src['test_type'].isin(actual_raw_test_keys_for_group)].copy()
                if not df_focused_group_vol_src.empty:
                    conclusive_results_mask_for_vol = ~df_focused_group_vol_src.get('test_result', pd.Series(dtype=str)).isin(
                                                        ['Pending','Unknown','Rejected Sample','Indeterminate', "UnknownResult_InsightPrep"])
                    
                    series_conclusive_vol = get_trend_data(df_focused_group_vol_src[conclusive_results_mask_for_vol], 'patient_id', 
                                                           date_col='encounter_date', period='D', agg_func='count').rename("Conclusive Tests")
                    series_pending_vol = get_trend_data(df_focused_group_vol_src[df_focused_group_vol_src.get('test_result', pd.Series(dtype=str)) == 'Pending'], 
                                                        'patient_id', date_col='encounter_date', period='D', agg_func='count').rename("Pending Tests")
                    
                    volume_trends_to_concat = [s for s in [series_conclusive_vol, series_pending_vol] if isinstance(s, pd.Series) and not s.empty]
                    if volume_trends_to_concat:
                        insights_output["focused_test_group_volume_trend_df"] = pd.concat(volume_trends_to_concat, axis=1).fillna(0).reset_index() # Ensure date index is a column
        else: # Original key not found for selected display name
            insights_output["processing_notes"].append(f"Could not find original configuration key for test group '{selected_test_group_display_name}' to generate trends.")
    else: # Selected display name not in the summary map
        insights_output["processing_notes"].append(f"No detailed aggregated stats found in summary for selected test group: '{selected_test_group_display_name}'.")


    # --- B. Overdue Pending Tests (Calculated from raw period data, df_tests_src) ---
    # Prioritize sample_collection_date, then registered_date, then encounter_date for pending calculation
    date_col_for_overdue = 'encounter_date' # Default fallback
    if 'sample_collection_date' in df_tests_src.columns and df_tests_src['sample_collection_date'].notna().any():
        date_col_for_overdue = 'sample_collection_date'
    elif 'sample_registered_lab_date' in df_tests_src.columns and df_tests_src['sample_registered_lab_date'].notna().any(): # Check after sample_collection_date
        date_col_for_overdue = 'sample_registered_lab_date'
    
    df_pending_for_overdue_calc = df_tests_src[
        (df_tests_src.get('test_result', pd.Series(dtype=str)) == 'Pending') & 
        (df_tests_src[date_col_for_overdue].notna()) # Must have a valid date to calculate pending days
    ].copy()

    if not df_pending_for_overdue_calc.empty:
        # Ensure the chosen date column is datetime (should be from prep, but double check)
        df_pending_for_overdue_calc[date_col_for_overdue] = pd.to_datetime(df_pending_for_overdue_calc[date_col_for_overdue], errors='coerce')
        df_pending_for_overdue_calc.dropna(subset=[date_col_for_overdue], inplace=True) # Remove if date conversion failed

        if not df_pending_for_overdue_calc.empty:
            current_date_for_pending_calc = pd.Timestamp('now').normalize() # Today at midnight for consistent "days pending"
            df_pending_for_overdue_calc['days_pending'] = (current_date_for_pending_calc - df_pending_for_overdue_calc[date_col_for_overdue]).dt.days
            
            # Helper to get specific TAT target + buffer for overdue threshold
            def get_overdue_threshold_days(test_type_val: str) -> int:
                test_configuration = app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get(test_type_val)
                buffer_days_overdue = 2 # Allowable buffer beyond target TAT before flagging as "overdue"
                target_tat = app_config.TARGET_TEST_TURNAROUND_DAYS # General default
                if test_configuration and 'target_tat_days' in test_configuration and pd.notna(test_config['target_tat_days']):
                    target_tat = test_config['target_tat_days']
                return int(target_tat + buffer_days_overdue) if pd.notna(target_tat) else \
                       int(app_config.OVERDUE_PENDING_TEST_DAYS_GENERAL_FALLBACK + buffer_days_overdue) # Fallback
            
            df_pending_for_overdue_calc['overdue_if_exceeds_days'] = df_pending_for_overdue_calc['test_type'].apply(get_overdue_threshold_days)
            df_final_overdue_tests = df_pending_for_overdue_calc[
                df_pending_for_overdue_calc['days_pending'] > df_pending_for_overdue_calc['overdue_if_exceeds_days']
            ]
            
            if not df_final_overdue_tests.empty:
                # Select and rename columns for display
                cols_to_display_overdue = ['patient_id', 'test_type', date_col_for_overdue, 'days_pending', 'overdue_if_exceeds_days']
                df_overdue_display_final = df_final_overdue_tests.rename(columns={date_col_for_overdue:"Sample/Registered Date"})
                # Ensure only existing columns are selected
                final_display_cols_overdue = [col for col in ['patient_id', 'test_type', "Sample/Registered Date", 'days_pending', 'overdue_if_exceeds_days'] 
                                              if col in df_overdue_display_final.columns]
                insights_output["overdue_pending_tests_list_df"] = df_overdue_display_final[final_display_cols_overdue].sort_values('days_pending', ascending=False)
            else: insights_output["processing_notes"].append("No tests found pending longer than their target TAT + buffer within the period.")
        else: insights_output["processing_notes"].append("No valid pending tests with dates for overdue calculation after data cleaning.")
    else: insights_output["processing_notes"].append("No pending tests found in the specified period for overdue status evaluation.")

    # --- C. Sample Rejection Analysis (From raw period data, df_tests_src) ---
    if 'sample_status' in df_tests_src.columns and 'rejection_reason' in df_tests_src.columns:
        df_rejected_samples_src = df_tests_src[df_tests_src.get('sample_status', pd.Series(dtype=str)) == 'Rejected'].copy()
        if not df_rejected_samples_src.empty:
            # Clean rejection reasons: fillna, strip, and map common NAs to "Unknown Reason"
            df_rejected_samples_src['rejection_reason_clean'] = df_rejected_samples_src['rejection_reason'].astype(str).str.strip()
            df_rejected_samples_src.loc[df_rejected_samples_src['rejection_reason_clean'].isin(common_na_strings_insights_prep + ["UnknownReason_InsightPrep"]), 'rejection_reason_clean'] = 'Unknown Reason'
            
            df_rejection_counts = df_rejected_samples_src['rejection_reason_clean'].value_counts().reset_index()
            df_rejection_counts.columns = ['Rejection Reason', 'Count']
            insights_output["sample_rejection_reasons_summary_df"] = df_rejection_counts

            # Provide a list of example rejected samples for review
            cols_for_rejected_examples_list = ['patient_id', 'test_type', 'encounter_date', 'rejection_reason_clean']
            # Add sample_collection_date if available for more context
            if 'sample_collection_date' in df_rejected_samples_src.columns: 
                cols_for_rejected_examples_list.insert(2, 'sample_collection_date') 
            
            insights_output["top_rejected_samples_examples_df"] = df_rejected_samples_src[
                [col for col in cols_for_rejected_examples_list if col in df_rejected_samples_src.columns] # Select only existing cols
            ].head(15) # Show top N examples
        else: insights_output["processing_notes"].append("No rejected samples recorded in this period for rejection analysis.")
    else: insights_output["processing_notes"].append("Sample status or rejection reason data columns missing for rejection analysis.")
    
    logger.info(f"({module_log_prefix}) Clinic testing insights data preparation finished. Notes: {len(insights_output['processing_notes'])}")
    return insights_output
