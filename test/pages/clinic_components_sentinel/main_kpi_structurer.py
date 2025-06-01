# sentinel_project_root/test/pages/clinic_components_sentinel/main_kpi_structurer.py
# Part of "Sentinel Health Co-Pilot" - LMIC Edge-First System
# This module structures key clinic performance and disease-specific KPIs
# based on summarized clinic service data.
# Refactored from the original clinic_components/kpi_display.py.
# The output is lists of structured KPI dictionaries for web reports/dashboards.

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional

# Assuming app_config is accessible
try:
    from config import app_config
except ImportError:
    import sys, os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_test_root = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
    if project_test_root not in sys.path: sys.path.insert(0, project_test_root)
    from config import app_config

logger = logging.getLogger(__name__)

def structure_main_clinic_kpis_data(
    clinic_service_kpis_summary: Dict[str, Any], # Expected from core_data_processing.get_clinic_summary
    reporting_period_str: str
) -> List[Dict[str, Any]]:
    """
    Structures main clinic performance KPIs from a summary dictionary into a list
    of dictionaries, each representing a KPI ready for display or reporting.

    Args:
        clinic_service_kpis_summary: A dictionary containing pre-calculated clinic service KPIs.
            Expected keys from updated get_clinic_summary:
            'overall_avg_test_turnaround_conclusive_days', 'perc_critical_tests_tat_met',
            'total_pending_critical_tests_patients', 'sample_rejection_rate_perc'.
        reporting_period_str: String describing the reporting period for context.

    Returns:
        List[Dict[str, Any]]: A list of KPI dictionaries for main performance.
    """
    module_source_context = "ClinicMainKPIStructurer"
    logger.info(f"({module_source_context}) Structuring main clinic KPIs for period: {reporting_period_str}")
    main_kpis_structured: List[Dict[str, Any]] = []

    if not clinic_service_kpis_summary:
        logger.warning(f"({module_source_context}) No clinic service KPI summary data provided.")
        return main_kpis_structured

    # 1. Overall Average Test Turnaround Time (TAT) for Conclusive Tests
    overall_tat = clinic_service_kpis_summary.get('overall_avg_test_turnaround_conclusive_days', np.nan)
    tat_stat = "NO_DATA"
    # General TAT target from app_config (TARGET_TEST_TURNAROUND_DAYS might be too aggressive for "overall")
    # Let's use a slightly more lenient check for "overall" or make it configurable.
    # For now, we use original logic that compared against TARGET_TEST_TURNAROUND_DAYS + buffer.
    general_target_tat_overall = app_config.TARGET_TEST_TURNAROUND_DAYS # Or a new specific config for overall
    if pd.notna(overall_tat):
        if overall_tat > (general_target_tat_overall + 1): tat_stat = "HIGH_CONCERN" # More than 1 day over general target
        elif overall_tat > general_target_tat_overall: tat_stat = "MODERATE_CONCERN"
        else: tat_stat = "ACCEPTABLE"
    main_kpis_structured.append({
        "metric_code": "AVG_TAT_ALL_CONCLUSIVE", "title": "Overall Avg. TAT (Conclusive)",
        "value_str": f"{overall_tat:.1f}" if pd.notna(overall_tat) else "N/A", "units": "days",
        "icon": "⏱️", "status_level": tat_stat,
        "help_text": f"Average Turnaround Time for all conclusive diagnostic tests. Target ref: ~{general_target_tat_overall} days."
    })

    # 2. Percentage of CRITICAL Tests Meeting TAT
    perc_crit_tat_met = clinic_service_kpis_summary.get('perc_critical_tests_tat_met', np.nan) # Allow NaN if no critical tests done
    crit_tat_stat = "NO_DATA"
    target_crit_tat_pct = app_config.TARGET_OVERALL_TESTS_MEETING_TAT_PCT_FACILITY
    if pd.notna(perc_crit_tat_met):
        if perc_crit_tat_met >= target_crit_tat_pct: crit_tat_stat = "GOOD_PERFORMANCE"
        elif perc_crit_tat_met >= target_crit_tat_pct * 0.8: crit_tat_stat = "MODERATE_CONCERN" # Between 80-100% of target
        else: crit_tat_stat = "HIGH_CONCERN"
    main_kpis_structured.append({
        "metric_code": "PERC_CRIT_TESTS_TAT_MET", "title": "% Critical Tests TAT Met",
        "value_str": f"{perc_crit_tat_met:.1f}" if pd.notna(perc_crit_tat_met) else "N/A", "units": "%",
        "icon": "🎯", "status_level": crit_tat_stat,
        "help_text": f"Critical diagnostic tests meeting defined TAT targets. Target: ≥{target_crit_tat_pct}%."
    })

    # 3. Total Pending Critical Tests (by unique patients)
    pending_crit_patients = clinic_service_kpis_summary.get('total_pending_critical_tests_patients', 0)
    pending_stat = "ACCEPTABLE" if pd.notna(pending_crit_patients) and pending_crit_patients == 0 else "NO_DATA"
    if pd.notna(pending_crit_patients) and pending_crit_patients > 0:
        if pending_crit_patients > 10: pending_stat = "HIGH_CONCERN"       # Example: High if >10 patients affected
        elif pending_crit_patients > 3: pending_stat = "MODERATE_CONCERN" # Moderate if >3
        else: pending_stat = "ACCEPTABLE" # Manageable low number
    main_kpis_structured.append({
        "metric_code": "PENDING_CRITICAL_TESTS_PATIENTS", "title": "Pending Critical Tests (Patients)",
        "value_str": str(int(pending_crit_patients)) if pd.notna(pending_crit_patients) else "N/A", "units": "patients",
        "icon": "⏳", "status_level": pending_stat,
        "help_text": "Unique patients with critical test results still pending. Aim for minimal backlog."
    })

    # 4. Sample Rejection Rate
    rejection_rate = clinic_service_kpis_summary.get('sample_rejection_rate_perc', np.nan)
    rejection_stat = "NO_DATA"
    target_reject_rate = app_config.TARGET_SAMPLE_REJECTION_RATE_PCT_FACILITY
    if pd.notna(rejection_rate):
        if rejection_rate > target_reject_rate * 1.5: rejection_stat = "HIGH_CONCERN" # e.g., >7.5% if target is 5%
        elif rejection_rate > target_reject_rate: rejection_stat = "MODERATE_CONCERN"
        else: rejection_stat = "GOOD_PERFORMANCE"
    main_kpis_structured.append({
        "metric_code": "SAMPLE_REJECTION_RATE", "title": "Sample Rejection Rate",
        "value_str": f"{rejection_rate:.1f}" if pd.notna(rejection_rate) else "N/A", "units":"%",
        "icon": "🚫", "status_level": rejection_stat,
        "help_text": f"Overall rate of lab samples rejected. Target: < {target_reject_rate}%."
    })
    
    logger.info(f"({module_source_context}) Structured main clinic KPIs: {len(main_kpis_structured)} items.")
    return main_kpis_structured


def structure_disease_specific_kpis_data(
    clinic_service_kpis_summary: Dict[str, Any],
    reporting_period_str: str
) -> List[Dict[str, Any]]:
    """
    Structures disease-specific KPIs (like test positivity) and key drug stockouts
    into a list of dictionaries for reporting at the clinic management level.

    Args:
        clinic_service_kpis_summary: A dictionary from get_clinic_summary, must contain
                                     'test_summary_details' and 'key_drug_stockouts_count'.
        reporting_period_str: String describing the reporting period.

    Returns:
        List[Dict[str, Any]]: List of structured KPI dictionaries for diseases/supply.
    """
    module_source_context = "ClinicDiseaseKPIStructurer"
    logger.info(f"({module_source_context}) Structuring disease-specific and supply KPIs for period: {reporting_period_str}")
    disease_kpis_structured: List[Dict[str, Any]] = []

    if not clinic_service_kpis_summary:
        logger.warning(f"({module_source_context}) No clinic service KPI summary data provided.")
        return disease_kpis_structured

    test_details = clinic_service_kpis_summary.get("test_summary_details", {})
    if not test_details:
        logger.warning(f"({module_source_context}) 'test_summary_details' missing from KPI summary.")
    
    # Define which tests to highlight for positivity rates on the console
    # These use display names as keys, matching the keys in test_summary_details
    # Values include icons and context-specific target positivity rate guidance.
    tests_for_positivity_kpis = {
        app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get("Sputum-GeneXpert", {}).get("display_name", "TB GeneXpert"):
            {"icon": "🫁", "target_max_pos_perc": 15.0, "metric_code": "POS_TB_GENEXPERT", "disease": "TB"},
        app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get("RDT-Malaria", {}).get("display_name", "Malaria RDT"):
            {"icon": "🦟", "target_max_pos_perc": app_config.TARGET_MALARIA_POSITIVITY_RATE, "metric_code": "POS_MALARIA_RDT", "disease": "Malaria"},
        app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get("HIV-Rapid", {}).get("display_name", "HIV Rapid Test"):
            {"icon": "🩸", "target_max_pos_perc": 5.0, "metric_code": "POS_HIV_RAPID", "disease": "HIV"} # Example national target might be lower
        # Add other important local tests like Dengue, Syphilis if in test_summary_details
    }

    for test_disp_name, props in tests_for_positivity_kpis.items():
        stats_for_test = test_details.get(test_disp_name, {})
        pos_rate_val = stats_for_test.get("positive_rate_perc", np.nan) # Key from get_clinic_summary
        
        status_pos_rate = "NO_DATA"
        if pd.notna(pos_rate_val):
            target_max = props.get("target_max_pos_perc", 10.0) # General target if not specified for test
            if pos_rate_val > target_max * 1.5: status_pos_rate = "HIGH_CONCERN" # e.g., >50% over target
            elif pos_rate_val > target_max : status_pos_rate = "MODERATE_CONCERN"
            else: status_pos_rate = "ACCEPTABLE"
        disease_kpis_structured.append({
            "metric_code": props["metric_code"], "title": f"{props['disease']} Positivity ({test_disp_name})",
            "value_str": f"{pos_rate_val:.1f}" if pd.notna(pos_rate_val) else "N/A", "units":"%",
            "icon": props["icon"], "status_level": status_pos_rate,
            "help_text": f"Positivity rate for {test_disp_name}. Target generally < {props.get('target_max_pos_perc', 10)}% (context-dependent)."
        })

    # Key Drug Stockouts Count
    stockouts_val = clinic_service_kpis_summary.get('key_drug_stockouts_count', 0) # Key from get_clinic_summary
    stockout_stat = "GOOD_PERFORMANCE" # Default, assumes 0 stockouts
    if pd.notna(stockouts_val) and stockouts_val > 0:
        stockout_stat = "HIGH_CONCERN" if stockouts_val > 2 else "MODERATE_CONCERN" # e.g. >2 key drugs = high
    
    disease_kpis_structured.append({
        "metric_code": "KEY_DRUG_STOCKOUT_COUNT", "title": "Key Drug Stockouts",
        "value_str": str(int(stockouts_val)) if pd.notna(stockouts_val) else "N/A", "units":"items",
        "icon": "💊", "status_level": stockout_stat,
        "help_text": f"Count of key drugs/supplies with less than {app_config.CRITICAL_SUPPLY_DAYS_REMAINING} days of stock remaining. Target: 0."
    })
    
    logger.info(f"({module_source_context}) Structured disease-specific & supply KPIs: {len(disease_kpis_structured)} items.")
    return disease_kpis_structured
