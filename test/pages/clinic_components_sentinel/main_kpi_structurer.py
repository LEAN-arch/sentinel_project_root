# sentinel_project_root/test/pages/clinic_components_sentinel/main_kpi_structurer.py
# Structures key clinic performance and disease-specific KPIs for Sentinel.

import pandas as pd # Not directly used, but numpy often comes with it
import numpy as np
import logging
from typing import Dict, Any, List, Optional

# Standardized import block
try:
    from config import app_config
except ImportError:
    import sys
    import os
    # Assumes this file is in sentinel_project_root/test/pages/clinic_components_sentinel/
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_test_root_dir = os.path.abspath(os.path.join(current_script_dir, os.pardir, os.pardir))
    if project_test_root_dir not in sys.path:
        sys.path.insert(0, project_test_root_dir)
    from config import app_config

logger = logging.getLogger(__name__)

def structure_main_clinic_kpis_data(
    clinic_service_kpis_summary: Optional[Dict[str, Any]], # Expected from core_data_processing.get_clinic_summary
    reporting_period_str: str # For context if needed in help_text, not directly used in value calc
) -> List[Dict[str, Any]]:
    """
    Structures main clinic performance KPIs from a summary dictionary into a list
    of dictionaries, each representing a KPI ready for display (e.g., via render_web_kpi_card).

    Args:
        clinic_service_kpis_summary: A dictionary containing pre-calculated clinic service KPIs.
        reporting_period_str: String describing the reporting period (for context).

    Returns:
        List of KPI dictionaries for main clinic performance.
    """
    module_log_prefix = "ClinicMainKPIStructurer" # Consistent prefix
    logger.info(f"({module_log_prefix}) Structuring main clinic KPIs for period: {reporting_period_str}")
    structured_main_kpis: List[Dict[str, Any]] = [] # Renamed for clarity

    if not isinstance(clinic_service_kpis_summary, dict) or not clinic_service_kpis_summary:
        logger.warning(f"({module_log_prefix}) No clinic service KPI summary data provided. Returning empty list.")
        return structured_main_kpis

    # 1. Overall Average Test Turnaround Time (TAT) for Conclusive Tests
    # Value from get_clinic_summary: 'overall_avg_test_turnaround_conclusive_days'
    avg_overall_tat = clinic_service_kpis_summary.get('overall_avg_test_turnaround_conclusive_days', np.nan)
    overall_tat_status = "NO_DATA"
    # Use general TAT target as a benchmark. Could be more nuanced with specific overall target.
    general_tat_target_days = app_config.TARGET_TEST_TURNAROUND_DAYS 
    if pd.notna(avg_overall_tat):
        if avg_overall_tat > (general_tat_target_days + 1.5): # Significantly over target
            overall_tat_status = "HIGH_CONCERN"
        elif avg_overall_tat > general_tat_target_days: # Moderately over
            overall_tat_status = "MODERATE_CONCERN"
        else: # Meeting or better than general target
            overall_tat_status = "ACCEPTABLE" # Or "GOOD_PERFORMANCE"
    structured_main_kpis.append({
        "metric_code": "AVG_TAT_ALL_CONCLUSIVE_TESTS", "title": "Overall Avg. TAT (Conclusive)",
        "value_str": f"{avg_overall_tat:.1f}" if pd.notna(avg_overall_tat) else "N/A", "units": "days",
        "icon": "⏱️", "status_level": overall_tat_status,
        "help_text": f"Average Turnaround Time for all conclusive diagnostic tests. Target reference: ~{general_tat_target_days} days."
    })

    # 2. Percentage of CRITICAL Tests Meeting TAT Target
    # Value from get_clinic_summary: 'perc_critical_tests_tat_met'
    percent_critical_tat_met = clinic_service_kpis_summary.get('perc_critical_tests_tat_met', np.nan)
    critical_tat_status = "NO_DATA"
    target_percent_tat_met = app_config.TARGET_OVERALL_TESTS_MEETING_TAT_PCT_FACILITY
    if pd.notna(percent_critical_tat_met):
        if percent_critical_tat_met >= target_percent_tat_met:
            critical_tat_status = "GOOD_PERFORMANCE"
        elif percent_critical_tat_met >= target_percent_tat_met * 0.75: # 75-99.9% of target
            critical_tat_status = "MODERATE_CONCERN"
        else: # Below 75% of target
            critical_tat_status = "HIGH_CONCERN"
    structured_main_kpis.append({
        "metric_code": "PERC_CRITICAL_TESTS_TAT_MET", "title": "% Critical Tests TAT Met",
        "value_str": f"{percent_critical_tat_met:.1f}" if pd.notna(percent_critical_tat_met) else "N/A", "units": "%",
        "icon": "🎯", "status_level": critical_tat_status,
        "help_text": f"Percentage of critical diagnostic tests meeting their defined TAT targets. Target: ≥{target_percent_tat_met}%."
    })

    # 3. Total Pending Critical Tests (by unique patients)
    # Value from get_clinic_summary: 'total_pending_critical_tests_patients'
    num_pending_critical_patients = clinic_service_kpis_summary.get('total_pending_critical_tests_patients', 0)
    pending_critical_status = "NO_DATA"
    if pd.notna(num_pending_critical_patients): # Ensure it's not NaN before int conversion
        count_pending = int(num_pending_critical_patients)
        if count_pending == 0: pending_critical_status = "GOOD_PERFORMANCE" # Ideal state
        elif count_pending <= 3: pending_critical_status = "ACCEPTABLE" # Small, manageable backlog
        elif count_pending <= 10: pending_critical_status = "MODERATE_CONCERN"
        else: pending_critical_status = "HIGH_CONCERN" # Significant backlog
    structured_main_kpis.append({
        "metric_code": "COUNT_PENDING_CRITICAL_TESTS_PATIENTS", "title": "Pending Critical Tests (Patients)",
        "value_str": str(int(num_pending_critical_patients)) if pd.notna(num_pending_critical_patients) else "N/A", "units": "patients",
        "icon": "⏳", "status_level": pending_critical_status,
        "help_text": "Number of unique patients with critical test results still pending. Target: 0."
    })

    # 4. Sample Rejection Rate (%)
    # Value from get_clinic_summary: 'sample_rejection_rate_perc'
    rejection_rate_percent = clinic_service_kpis_summary.get('sample_rejection_rate_perc', np.nan)
    rejection_rate_status = "NO_DATA"
    target_rejection_percent_facility = app_config.TARGET_SAMPLE_REJECTION_RATE_PCT_FACILITY
    if pd.notna(rejection_rate_percent):
        if rejection_rate_percent > target_rejection_percent_facility * 1.75: # Significantly above target (e.g., >8.75% if target 5%)
            rejection_rate_status = "HIGH_CONCERN"
        elif rejection_rate_percent > target_rejection_percent_facility: # Moderately above
            rejection_rate_status = "MODERATE_CONCERN"
        else: # Meeting or better than target
            rejection_rate_status = "GOOD_PERFORMANCE"
    structured_main_kpis.append({
        "metric_code": "PERC_SAMPLE_REJECTION_RATE", "title": "Sample Rejection Rate",
        "value_str": f"{rejection_rate_percent:.1f}" if pd.notna(rejection_rate_percent) else "N/A", "units":"%",
        "icon": "🚫", "status_level": rejection_rate_status,
        "help_text": f"Overall rate of laboratory samples rejected. Target: < {target_rejection_percent_facility}%."
    })
    
    logger.info(f"({module_log_prefix}) Structured {len(structured_main_kpis)} main clinic KPIs.")
    return structured_main_kpis


def structure_disease_specific_kpis_data(
    clinic_service_kpis_summary: Optional[Dict[str, Any]],
    reporting_period_str: str # For context if needed
) -> List[Dict[str, Any]]:
    """
    Structures disease-specific KPIs (e.g., test positivity) and key drug stockouts.
    """
    module_log_prefix = "ClinicDiseaseSupplyKPIStructurer"
    logger.info(f"({module_log_prefix}) Structuring disease-specific & supply KPIs for period: {reporting_period_str}")
    structured_disease_supply_kpis: List[Dict[str, Any]] = [] # Renamed for clarity

    if not isinstance(clinic_service_kpis_summary, dict) or not clinic_service_kpis_summary:
        logger.warning(f"({module_log_prefix}) No clinic service KPI summary data provided. Returning empty list.")
        return structured_disease_supply_kpis

    test_summary_details_map = clinic_service_kpis_summary.get("test_summary_details")
    if not isinstance(test_summary_details_map, dict):
        logger.warning(f"({module_log_prefix}) 'test_summary_details' missing or not a dict in KPI summary. Cannot structure test positivity KPIs.")
        test_summary_details_map = {} # Use empty dict to avoid errors below

    # Define which tests to highlight for positivity rates on the console dashboard.
    # Keys are exact display names as produced by get_clinic_summary (derived from app_config.KEY_TEST_TYPES_FOR_ANALYSIS).
    # Values include properties for structuring the KPI card.
    tests_for_positivity_display = {
        app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get("Sputum-GeneXpert", {}).get("display_name", "TB GeneXpert"):
            {"icon": "🫁", "target_max_pos_perc": 15.0, "metric_code": "POS_RATE_TB_GENEXPERT", "disease_label": "TB"},
        app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get("RDT-Malaria", {}).get("display_name", "Malaria RDT"):
            {"icon": "🦟", "target_max_pos_perc": app_config.TARGET_MALARIA_POSITIVITY_RATE, "metric_code": "POS_RATE_MALARIA_RDT", "disease_label": "Malaria"},
        app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get("HIV-Rapid", {}).get("display_name", "HIV Rapid Test"):
            {"icon": "🩸", "target_max_pos_perc": 5.0, "metric_code": "POS_RATE_HIV_RAPID", "disease_label": "HIV"} # Example target
        # Add other key tests for positivity monitoring as needed
    }

    for test_disp_name_kpi, kpi_props in tests_for_positivity_display.items():
        stats_for_this_test = test_summary_details_map.get(test_disp_name_kpi, {}) # Get stats for this display name
        positivity_rate_val = stats_for_this_test.get("positive_rate_perc", np.nan) # Key from get_clinic_summary
        
        pos_rate_status_level = "NO_DATA"
        target_max_positivity = kpi_props.get("target_max_pos_perc", 10.0) # General target if not specified for test
        if pd.notna(positivity_rate_val):
            if positivity_rate_val > target_max_positivity * 1.5: pos_rate_status_level = "HIGH_CONCERN" # e.g., >50% over target
            elif positivity_rate_val > target_max_positivity : pos_rate_status_level = "MODERATE_CONCERN"
            else: pos_rate_status_level = "ACCEPTABLE"
        
        structured_disease_supply_kpis.append({
            "metric_code": kpi_props["metric_code"], 
            "title": f"{kpi_props['disease_label']} Positivity ({test_disp_name_kpi})", # Clearer title
            "value_str": f"{positivity_rate_val:.1f}" if pd.notna(positivity_rate_val) else "N/A", "units":"%",
            "icon": kpi_props["icon"], "status_level": pos_rate_status_level,
            "help_text": f"Positivity rate for {test_disp_name_kpi}. Target reference: < {target_max_positivity}% (varies by context)."
        })

    # Key Drug Stockouts Count
    # Value from get_clinic_summary: 'key_drug_stockouts_count'
    num_key_drug_stockouts = clinic_service_kpis_summary.get('key_drug_stockouts_count', 0) # Default to 0 if missing
    stockout_status_level = "NO_DATA"
    if pd.notna(num_key_drug_stockouts): # Ensure it's not NaN before int conversion
        count_stockouts = int(num_key_drug_stockouts)
        if count_stockouts == 0: stockout_status_level = "GOOD_PERFORMANCE"
        elif count_stockouts <= 2: stockout_status_level = "MODERATE_CONCERN" # 1-2 key drugs low/out
        else: stockout_status_level = "HIGH_CONCERN" # More than 2 key drugs
    
    structured_disease_supply_kpis.append({
        "metric_code": "COUNT_KEY_DRUG_STOCKOUTS", "title": "Key Drug Stockouts",
        "value_str": str(int(num_key_drug_stockouts)) if pd.notna(num_key_drug_stockouts) else "N/A", "units":"items",
        "icon": "💊", "status_level": stockout_status_level,
        "help_text": f"Number of key drugs/supplies with less than {app_config.CRITICAL_SUPPLY_DAYS_REMAINING} days of stock remaining. Target: 0."
    })
    
    logger.info(f"({module_log_prefix}) Structured {len(structured_disease_supply_kpis)} disease-specific & supply KPIs.")
    return structured_disease_supply_kpis
