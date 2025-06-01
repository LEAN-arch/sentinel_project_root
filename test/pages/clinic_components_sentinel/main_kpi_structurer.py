# sentinel_project_root/test/pages/clinic_components_sentinel/main_kpi_structurer.py
# Structures key clinic performance and disease-specific KPIs for Sentinel.

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional

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

def structure_main_clinic_kpis_data(
    clinic_service_kpis_summary: Optional[Dict[str, Any]],
    reporting_period_str: str
) -> List[Dict[str, Any]]:
    """
    Structures main clinic performance KPIs from a summary dictionary.

    Args:
        clinic_service_kpis_summary: Dictionary from core_data_processing.get_clinic_summary.
        reporting_period_str: String describing the reporting period.

    Returns:
        List of KPI dictionaries for main performance.
    """
    module_log_prefix = "ClinicMainKPIStructurer"
    logger.info(f"({module_log_prefix}) Structuring main clinic KPIs for period: {reporting_period_str}")
    main_kpis_list: List[Dict[str, Any]] = []

    if not clinic_service_kpis_summary:
        logger.warning(f"({module_log_prefix}) No clinic service KPI summary data provided. Returning empty list.")
        return main_kpis_list

    # 1. Overall Average Test Turnaround Time (TAT) for Conclusive Tests
    overall_tat_days = clinic_service_kpis_summary.get('overall_avg_test_turnaround_conclusive_days', np.nan)
    tat_status = "NO_DATA"
    # Using general TAT target as a reference for overall performance
    general_target_tat = app_config.TARGET_TEST_TURNAROUND_DAYS 
    if pd.notna(overall_tat_days):
        if overall_tat_days > (general_target_tat + 1.5):  # More than 1.5 days over general target
            tat_status = "HIGH_CONCERN"
        elif overall_tat_days > general_target_tat:
            tat_status = "MODERATE_CONCERN"
        else:
            tat_status = "ACCEPTABLE"
    main_kpis_list.append({
        "metric_code": "AVG_TAT_ALL_CONCLUSIVE", "title": "Overall Avg. TAT (Conclusive)",
        "value_str": f"{overall_tat_days:.1f}" if pd.notna(overall_tat_days) else "N/A", "units": "days",
        "icon": "⏱️", "status_level": tat_status,
        "help_text": f"Average Turnaround Time for all conclusive tests. Target reference: ~{general_target_tat} days."
    })

    # 2. Percentage of CRITICAL Tests Meeting TAT
    perc_critical_tat_met = clinic_service_kpis_summary.get('perc_critical_tests_tat_met', np.nan)
    crit_tat_status = "NO_DATA"
    target_crit_tat_met_pct = app_config.TARGET_OVERALL_TESTS_MEETING_TAT_PCT_FACILITY
    if pd.notna(perc_critical_tat_met):
        if perc_critical_tat_met >= target_crit_tat_met_pct:
            crit_tat_status = "GOOD_PERFORMANCE"
        elif perc_critical_tat_met >= target_crit_tat_met_pct * 0.8: # e.g., 80-99.9% of target
            crit_tat_status = "MODERATE_CONCERN"
        else: # Below 80% of target
            crit_tat_status = "HIGH_CONCERN"
    main_kpis_list.append({
        "metric_code": "PERC_CRIT_TESTS_TAT_MET", "title": "% Critical Tests TAT Met",
        "value_str": f"{perc_critical_tat_met:.1f}" if pd.notna(perc_critical_tat_met) else "N/A", "units": "%",
        "icon": "🎯", "status_level": crit_tat_status,
        "help_text": f"Percentage of critical tests meeting defined TAT. Target: ≥{target_crit_tat_met_pct}%."
    })

    # 3. Total Pending Critical Tests (by unique patients)
    pending_critical_patients_count = clinic_service_kpis_summary.get('total_pending_critical_tests_patients', 0)
    pending_status = "NO_DATA" # Default if count is NaN, though get_clinic_summary should give 0
    if pd.notna(pending_critical_patients_count):
        count_val = int(pending_critical_patients_count)
        if count_val == 0: pending_status = "GOOD_PERFORMANCE"
        elif count_val <= 3: pending_status = "ACCEPTABLE" # Small, manageable backlog
        elif count_val <= 10: pending_status = "MODERATE_CONCERN"
        else: pending_status = "HIGH_CONCERN" # Significant backlog
    main_kpis_list.append({
        "metric_code": "PENDING_CRITICAL_TESTS_PATIENTS", "title": "Pending Critical Tests (Patients)",
        "value_str": str(int(pending_critical_patients_count)) if pd.notna(pending_critical_patients_count) else "N/A", "units": "patients",
        "icon": "⏳", "status_level": pending_status,
        "help_text": "Unique patients with critical test results still pending. Aim for zero."
    })

    # 4. Sample Rejection Rate
    sample_rejection_rate = clinic_service_kpis_summary.get('sample_rejection_rate_perc', np.nan)
    rejection_status = "NO_DATA"
    target_rejection_pct = app_config.TARGET_SAMPLE_REJECTION_RATE_PCT_FACILITY
    if pd.notna(sample_rejection_rate):
        if sample_rejection_rate > target_rejection_pct * 1.5: # e.g., >7.5% if target is 5%
            rejection_status = "HIGH_CONCERN"
        elif sample_rejection_rate > target_rejection_pct:
            rejection_status = "MODERATE_CONCERN"
        else:
            rejection_status = "GOOD_PERFORMANCE"
    main_kpis_list.append({
        "metric_code": "SAMPLE_REJECTION_RATE", "title": "Sample Rejection Rate",
        "value_str": f"{sample_rejection_rate:.1f}" if pd.notna(sample_rejection_rate) else "N/A", "units":"%",
        "icon": "🚫", "status_level": rejection_status,
        "help_text": f"Overall rate of lab samples rejected. Target: < {target_rejection_pct}%."
    })
    
    logger.info(f"({module_log_prefix}) Structured {len(main_kpis_list)} main clinic KPIs.")
    return main_kpis_list


def structure_disease_specific_kpis_data(
    clinic_service_kpis_summary: Optional[Dict[str, Any]],
    reporting_period_str: str
) -> List[Dict[str, Any]]:
    """
    Structures disease-specific KPIs (test positivity) and key drug stockouts.

    Args:
        clinic_service_kpis_summary: Dictionary from get_clinic_summary.
        reporting_period_str: String describing the reporting period.

    Returns:
        List of structured KPI dictionaries for diseases/supply.
    """
    module_log_prefix = "ClinicDiseaseSupplyKPIStructurer" # Renamed for clarity
    logger.info(f"({module_log_prefix}) Structuring disease-specific & supply KPIs for period: {reporting_period_str}")
    disease_supply_kpis_list: List[Dict[str, Any]] = []

    if not clinic_service_kpis_summary:
        logger.warning(f"({module_log_prefix}) No clinic service KPI summary data provided. Returning empty list.")
        return disease_supply_kpis_list

    test_summary_details = clinic_service_kpis_summary.get("test_summary_details")
    if not isinstance(test_summary_details, dict): # Check if it's a dictionary
        logger.warning(f"({module_log_prefix}) 'test_summary_details' missing or not a dict in KPI summary. Cannot structure test positivity KPIs.")
        test_summary_details = {} # Use empty dict to avoid errors below

    # Define which tests to highlight for positivity rates on the console.
    # Keys are display names, matching keys in `test_summary_details` from `get_clinic_summary`.
    # `target_max_pos_perc` can be specific per test or a general heuristic.
    # `metric_code` should be unique.
    tests_to_display_positivity = {
        # TB GeneXpert (using display name from app_config)
        app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get("Sputum-GeneXpert", {}).get("display_name", "TB GeneXpert"):
            {"icon": "🫁", "target_max_pos_perc": 15.0, "metric_code": "POS_TB_GENEXPERT", "disease_short": "TB"},
        # Malaria RDT
        app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get("RDT-Malaria", {}).get("display_name", "Malaria RDT"):
            {"icon": "🦟", "target_max_pos_perc": app_config.TARGET_MALARIA_POSITIVITY_RATE, "metric_code": "POS_MALARIA_RDT", "disease_short": "Malaria"},
        # HIV Rapid Test
        app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get("HIV-Rapid", {}).get("display_name", "HIV Rapid Test"):
            {"icon": "🩸", "target_max_pos_perc": 5.0, "metric_code": "POS_HIV_RAPID", "disease_short": "HIV"} # Example general target
    }

    for test_display_name, props in tests_to_display_positivity.items():
        test_stats = test_summary_details.get(test_display_name, {}) # Get stats for this display name
        positivity_rate = test_stats.get("positive_rate_perc", np.nan)
        
        status_level_pos = "NO_DATA"
        target_max = props.get("target_max_pos_perc", 10.0) # Default target if not specified
        if pd.notna(positivity_rate):
            if positivity_rate > target_max * 1.5: status_level_pos = "HIGH_CONCERN"
            elif positivity_rate > target_max: status_level_pos = "MODERATE_CONCERN"
            else: status_level_pos = "ACCEPTABLE"
        
        disease_supply_kpis_list.append({
            "metric_code": props["metric_code"], 
            "title": f"{props['disease_short']} Positivity ({test_display_name})",
            "value_str": f"{positivity_rate:.1f}" if pd.notna(positivity_rate) else "N/A", "units":"%",
            "icon": props["icon"], "status_level": status_level_pos,
            "help_text": f"Positivity rate for {test_display_name}. Target ref: < {target_max}% (context-dependent)."
        })

    # Key Drug Stockouts Count
    key_drug_stockouts = clinic_service_kpis_summary.get('key_drug_stockouts_count', 0) # Should be an int
    stockout_status = "NO_DATA"
    if pd.notna(key_drug_stockouts): # Ensure it's not NaN before int conversion
        count_val = int(key_drug_stockouts)
        if count_val == 0: stockout_status = "GOOD_PERFORMANCE"
        elif count_val <= 2: stockout_status = "MODERATE_CONCERN" # 1-2 key drugs low/out
        else: stockout_status = "HIGH_CONCERN" # More than 2 key drugs
    
    disease_supply_kpis_list.append({
        "metric_code": "KEY_DRUG_STOCKOUT_COUNT", "title": "Key Drug Stockouts",
        "value_str": str(int(key_drug_stockouts)) if pd.notna(key_drug_stockouts) else "N/A", "units":"items",
        "icon": "💊", "status_level": stockout_status,
        "help_text": f"Key drugs/supplies with < {app_config.CRITICAL_SUPPLY_DAYS_REMAINING} days of stock. Target: 0."
    })
    
    logger.info(f"({module_log_prefix}) Structured {len(disease_supply_kpis_list)} disease-specific & supply KPIs.")
    return disease_supply_kpis_list
