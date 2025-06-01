# sentinel_project_root/test/pages/clinic_components_sentinel/testing_insights_analyzer.py
# Part of "Sentinel Health Co-Pilot" - LMIC Edge-First System
# This module prepares detailed data for laboratory testing performance and trends at a clinic.
# It was refactored from the original clinic_components/testing_insights_tab.py.
# Output is a structured dictionary for display on the Clinic Management Console (Tier 2).

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Union

# Assuming app_config and core utilities are accessible
try:
    from config import app_config
    from utils.core_data_processing import get_trend_data # For TAT/Volume trends
except ImportError:
    import sys, os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_test_root = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
    if project_test_root not in sys.path: sys.path.insert(0, project_test_root)
    from config import app_config
    from utils.core_data_processing import get_trend_data

logger = logging.getLogger(__name__)

def prepare_clinic_testing_insights_data(
    filtered_health_df_clinic_period: Optional[pd.DataFrame], # Health data for clinic and period
    # clinic_service_kpis_summary should contain 'test_summary_details' from get_clinic_summary
    clinic_service_kpis_summary: Optional[Dict[str, Any]], 
    reporting_period_str: str,
    # The UI (Clinic Console page) will provide this based on user selection or default
    selected_test_group_display_name: str = "All Critical Tests Summary" 
) -> Dict[str, Any]:
    """
    Prepares structured data for detailed testing insights, including performance metrics,
    trends, overdue tests, and rejection analysis.

    Args:
        filtered_health_df_clinic_period: DataFrame of health records for the clinic and period.
                                          Expected columns: 'test_type', 'test_result', 
                                          'test_turnaround_days', 'encounter_date', 'patient_id',
                                          'sample_collection_date', 'sample_registered_lab_date',
                                          'sample_status', 'rejection_reason'.
        clinic_service_kpis_summary: Summary dict from get_clinic_summary (core_data_processing),
                                     must contain 'test_summary_details' for aggregated metrics.
        reporting_period_str: String describing the reporting period for context.
        selected_test_group_display_name: The test group (display name from KEY_TEST_TYPES_FOR_ANALYSIS)
                                         or "All Critical Tests Summary" to focus on for detailed
                                         metrics and trends.

    Returns:
        Dict[str, Any]: A dictionary containing structured testing insights data.
    """
    module_source_context = "ClinicTestingInsightsAnalyzer"
    logger.info(f"({module_source_context}) Preparing testing insights. Focus: {selected_test_group_display_name}, Period: {reporting_period_str}")

    testing_insights_data_output: Dict[str, Any] = {
        "reporting_period": reporting_period_str,
        "selected_focus_area": selected_test_group_display_name,
        "all_critical_tests_summary_table_df": None, # DataFrame for "All Critical Tests" view
        "focused_test_group_kpis": None,           # Dict of KPIs for a specific selected group
        "focused_test_group_tat_trend": None,      # pd.Series for TAT trend
        "focused_test_group_volume_trend_df": None,  # pd.DataFrame for Volume (Conclusive, Pending)
        "overdue_pending_tests_list_df": None,       # DataFrame of overdue tests
        "sample_rejection_reasons_summary_df": None, # DataFrame: Rejection Reason, Count
        "top_rejected_samples_examples_df": None,    # DataFrame: List of example rejected samples
        "processing_notes": []
    }

    if filtered_health_df_clinic_period is None or filtered_health_df_clinic_period.empty:
        note = "No health data provided for testing insights analysis."
        logger.warning(f"({module_source_context}) {note}")
        testing_insights_data_output["processing_notes"].append(note)
        return testing_insights_data_output
    
    if not clinic_service_kpis_summary or "test_summary_details" not in clinic_service_kpis_summary:
        note = "Clinic service KPI summary or 'test_summary_details' missing; aggregated metrics will be unavailable."
        logger.warning(f"({module_source_context}) {note}")
        testing_insights_data_output["processing_notes"].append(note)
        # We can still proceed with overdue and rejection analysis from raw data.
    
    df_clinic_tests_source = filtered_health_df_clinic_period.copy() # Work on a copy
    # Ensure essential columns for calculations exist
    test_analysis_cols = {
        'test_type': "UnknownTest", 'test_result': "UnknownResult", 'sample_status': "UnknownStatus",
        'encounter_date': pd.NaT, 'test_turnaround_days': np.nan, 'patient_id': "UnknownPID",
        'sample_collection_date': pd.NaT, 'sample_registered_lab_date': pd.NaT,
        'rejection_reason': "UnknownReason"
    }
    for col, default in test_analysis_cols.items():
        if col not in df_clinic_tests_source.columns: df_clinic_tests_source[col] = default
        elif 'date' in col: df_clinic_tests_source[col] = pd.to_datetime(df_clinic_tests_source[col], errors='coerce')
        elif 'days' in col : df_clinic_tests_source[col] = pd.to_numeric(df_clinic_tests_source[col], errors='coerce').fillna(default)

    # --- A. Data for Selected Focus Area (All Critical or Specific Test Group) ---
    test_summary_details_map = clinic_service_kpis_summary.get("test_summary_details", {}) if clinic_service_kpis_summary else {}

    if selected_test_group_display_name == "All Critical Tests Summary":
        critical_tests_summary_for_table = []
        if test_summary_details_map: # Only if detailed stats are available
            for group_display_name_iter, stats_iter in test_summary_details_map.items():
                original_key_iter = next((k for k, v_cfg in app_config.KEY_TEST_TYPES_FOR_ANALYSIS.items() if v_cfg.get("display_name") == group_display_name_iter), None)
                if original_key_iter and app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get(original_key_iter, {}).get("critical"):
                    critical_tests_summary_for_table.append({
                        "Test Group (Critical)": group_display_name_iter,
                        "Positivity (%)": stats_iter.get("positive_rate_perc", 0.0), # From updated get_clinic_summary
                        "Avg. TAT (Days)": stats_iter.get("avg_tat_days", np.nan),
                        "% Met TAT Target": stats_iter.get("perc_met_tat_target", 0.0),
                        "Pending (Patients)": stats_iter.get("pending_count_patients", 0),
                        "Rejected (Patients)": stats_iter.get("rejected_count_patients", 0),
                        "Total Conclusive Tests": stats_iter.get("total_conclusive_tests", 0)
                    })
            if critical_tests_summary_for_table:
                testing_insights_data_output["all_critical_tests_summary_table_df"] = pd.DataFrame(critical_tests_summary_for_table)
            else: testing_insights_data_output["processing_notes"].append("No data for critical tests found in summary or none configured as critical.")
        else: testing_insights_data_output["processing_notes"].append("Detailed test statistics map missing for 'All Critical Tests' summary.")

    elif selected_test_group_display_name in test_summary_details_map:
        stats_for_selected = test_summary_details_map[selected_test_group_display_name]
        testing_insights_data_output["focused_test_group_kpis"] = { # Store as a dict for easy access
            "Positivity Rate (%)": stats_for_selected.get("positive_rate_perc", np.nan),
            "Avg. TAT (Days)": stats_for_selected.get("avg_tat_days", np.nan),
            "% Met TAT Target": stats_for_selected.get("perc_met_tat_target", 0.0),
            "Pending Tests (Patients)": stats_for_selected.get("pending_count_patients", 0),
            "Rejected Samples (Patients)": stats_for_selected.get("rejected_count_patients", 0),
            "Total Conclusive Tests": stats_for_selected.get("total_conclusive_tests",0)
        }
        
        # For trends, we need to use the raw `df_clinic_tests_source` filtered by original test keys
        orig_key_for_focus_group = next((k for k,v_cfg in app_config.KEY_TEST_TYPES_FOR_ANALYSIS.items() if v_cfg.get("display_name") == selected_test_group_display_name), None)
        if orig_key_for_focus_group:
            test_config_for_focus = app_config.KEY_TEST_TYPES_FOR_ANALYSIS[orig_key_for_focus_group]
            actual_test_keys_in_data = test_config_for_focus.get("types_in_group", [orig_key_for_focus_group]) # Handles grouped tests
            if isinstance(actual_test_keys_in_data, str): actual_test_keys_in_data = [actual_test_keys_in_data]

            # TAT Trend for selected group
            if 'test_turnaround_days' in df_clinic_tests_source.columns and 'encounter_date' in df_clinic_tests_source.columns:
                tat_trend_src_df = df_clinic_tests_source[
                    (df_clinic_tests_source['test_type'].isin(actual_test_keys_in_data)) &
                    (df_clinic_tests_source['test_turnaround_days'].notna()) &
                    (~df_clinic_tests_source.get('test_result', pd.Series(dtype=str)).isin(['Pending','Unknown','Rejected Sample','Indeterminate','Unknown']))
                ].copy()
                if not tat_trend_src_df.empty:
                    tat_trend_series_data = get_trend_data(tat_trend_src_df, 'test_turnaround_days', date_col='encounter_date', period='D', agg_func='mean', source_context=f"{module_source_context}/TATTrend/{selected_test_group_display_name}")
                    testing_insights_data_output["focused_test_group_tat_trend"] = tat_trend_series_data if tat_trend_series_data is not None and not tat_trend_series_data.empty else None
            
            # Volume Trend (Conclusive vs. Pending) for selected group
            if 'patient_id' in df_clinic_tests_source.columns and 'encounter_date' in df_clinic_tests_source.columns:
                vol_src_df = df_clinic_tests_source[df_clinic_tests_source['test_type'].isin(actual_test_keys_in_data)].copy()
                if not vol_src_df.empty:
                    # Count argument `count` implies non-NA count for patient_id (or any other column basically)
                    conclusive_vol = get_trend_data(vol_src_df[~vol_src_df.get('test_result', pd.Series(dtype=str)).isin(['Pending','Unknown','Rejected Sample','Indeterminate','Unknown'])], 'patient_id', date_col='encounter_date', period='D', agg_func='count').rename("Conclusive Tests")
                    pending_vol = get_trend_data(vol_src_df[vol_src_df.get('test_result', pd.Series(dtype=str)) == 'Pending'], 'patient_id', date_col='encounter_date', period='D', agg_func='count').rename("Pending Tests")
                    if (conclusive_vol is not None and not conclusive_vol.empty) or (pending_vol is not None and not pending_vol.empty):
                        testing_insights_data_output["focused_test_group_volume_trend_df"] = pd.concat([conclusive_vol, pending_vol], axis=1).fillna(0).reset_index()
        else:
            testing_insights_data_output["processing_notes"].append(f"Could not find original configuration key for '{selected_test_group_display_name}' to generate trends.")
    else:
        testing_insights_data_output["processing_notes"].append(f"No detailed aggregated stats found for selected test group: '{selected_test_group_display_name}'.")


    # --- B. Overdue Pending Tests (Calculated from raw period data) ---
    date_col_for_overdue_calc = 'encounter_date' # Fallback
    if 'sample_collection_date' in df_clinic_tests_source.columns and df_clinic_tests_source['sample_collection_date'].notna().any(): date_col_for_overdue_calc = 'sample_collection_date'
    elif 'sample_registered_lab_date' in df_clinic_tests_source.columns and df_clinic_tests_source['sample_registered_lab_date'].notna().any(): date_col_for_overdue_calc = 'sample_registered_lab_date'
    
    overdue_calc_df = df_clinic_tests_source[
        (df_clinic_tests_source.get('test_result', pd.Series(dtype=str)) == 'Pending') & (df_clinic_tests_source[date_col_for_overdue_calc].notna())
    ].copy()
    if not overdue_calc_df.empty:
        overdue_calc_df[date_col_for_overdue_calc] = pd.to_datetime(overdue_calc_df[date_col_for_overdue_calc], errors='coerce') # Ensure datetime
        overdue_calc_df.dropna(subset=[date_col_for_overdue_calc], inplace=True)
        if not overdue_calc_df.empty:
            overdue_calc_df['days_pending_calculated'] = (pd.Timestamp('now').normalize() - overdue_calc_df[date_col_for_overdue_calc]).dt.days # Use 'now' for current pending days
            def get_specific_overdue_days_threshold(test_type_val: str) -> int:
                test_cfg = app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get(test_type_val)
                buffer_days = 2 # Allowable buffer beyond target TAT
                return (test_cfg['target_tat_days'] + buffer_days) if test_cfg and 'target_tat_days' in test_cfg else (app_config.OVERDUE_PENDING_TEST_DAYS_GENERAL_FALLBACK + buffer_days)
            overdue_calc_df['overdue_if_exceeds_days'] = overdue_calc_df['test_type'].apply(get_specific_overdue_days_threshold)
            final_overdue_tests_df = overdue_calc_df[overdue_calc_df['days_pending_calculated'] > overdue_calc_df['overdue_if_exceeds_days']]
            if not final_overdue_tests_df.empty:
                cols_to_show_overdue_list = ['patient_id', 'test_type', date_col_for_overdue_calc, 'days_pending_calculated', 'overdue_if_exceeds_days']
                # Rename date_col_for_overdue_calc for display friendliness
                final_overdue_tests_df_renamed = final_overdue_tests_df.rename(columns={date_col_for_overdue_calc:"Sample/Registered Date"})
                testing_insights_data_output["overdue_pending_tests_list_df"] = final_overdue_tests_df_renamed[[col for col in cols_to_show_overdue_list if col in final_overdue_tests_df_renamed.columns or col == "Sample/Registered Date"]].sort_values('days_pending_calculated', ascending=False) # Keep desired columns
            else: testing_insights_data_output["processing_notes"].append("No tests found pending longer than their target TAT + buffer.")
        else: testing_insights_data_output["processing_notes"].append("No valid pending tests with dates for overdue calculation.")
    else: testing_insights_data_output["processing_notes"].append("No pending tests found in the period for overdue status evaluation.")


    # --- C. Sample Rejection Analysis (From raw period data) ---
    if 'sample_status' in df_clinic_tests_source.columns and 'rejection_reason' in df_clinic_tests_source.columns:
        rejected_df = df_clinic_tests_source[df_clinic_tests_source.get('sample_status', pd.Series(dtype=str)) == 'Rejected'].copy()
        if not rejected_df.empty:
            rejected_df['rejection_reason_clean'] = rejected_df['rejection_reason'].fillna('Unknown Reason').astype(str).str.strip().replace(['','nan','None'], 'Unknown Reason')
            rejection_counts_df = rejected_df['rejection_reason_clean'].value_counts().reset_index(); rejection_counts_df.columns = ['Rejection Reason', 'Count']
            testing_insights_data_output["sample_rejection_reasons_summary_df"] = rejection_counts_df

            cols_for_rejected_list_tab = ['patient_id', 'test_type', 'encounter_date', 'rejection_reason_clean']
            if 'sample_collection_date' in rejected_df.columns : cols_for_rejected_list_tab.insert(2,'sample_collection_date')
            testing_insights_data_output["top_rejected_samples_examples_df"] = rejected_df[cols_for_rejected_list_tab].head(15) # Show top examples
        else: testing_insights_data_output["processing_notes"].append("No rejected samples recorded in this period for rejection analysis.")
    else: testing_insights_data_output["processing_notes"].append("Sample status or rejection reason data columns missing for rejection analysis.")
    
    logger.info(f"({module_source_context}) Clinic testing insights data preparation finished. Notes: {len(testing_insights_data_output['processing_notes'])}")
    return testing_insights_data_output
