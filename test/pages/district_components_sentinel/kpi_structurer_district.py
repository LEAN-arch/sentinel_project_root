# sentinel_project_root/test/pages/district_components_sentinel/kpi_structurer_district.py
# Structures district-wide KPIs for Sentinel Health Co-Pilot DHO dashboards.

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
    # Assumes this file is in sentinel_project_root/test/pages/district_components_sentinel/
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_test_root_dir = os.path.abspath(os.path.join(current_script_dir, os.pardir, os.pardir))
    if project_test_root_dir not in sys.path:
        sys.path.insert(0, project_test_root_dir)
    from config import app_config

logger = logging.getLogger(__name__)

def structure_district_kpis_data(
    district_overall_kpis_summary: Optional[Dict[str, Any]],
    district_enriched_gdf: Optional[pd.DataFrame] = None, # GeoDataFrame expected, using pd.DataFrame for broader typing
    reporting_period_str: str = "Latest Aggregated Data" # Default value for reporting period context
) -> List[Dict[str, Any]]:
    """
    Structures district-wide KPIs from a summary dictionary into a list of KPI dictionaries,
    suitable for display with `render_web_kpi_card`.

    Args:
        district_overall_kpis_summary: Dict from get_district_summary_kpis (core_data_processing).
        district_enriched_gdf: Optional. The GDF used to calculate summary, for context like total zone count.
        reporting_period_str: String describing the reporting period (for context).

    Returns:
        List of structured KPI dictionaries.
    """
    module_log_prefix = "DistrictKPIStructurer" # Consistent prefix
    logger.info(f"({module_log_prefix}) Structuring district KPIs for period: {reporting_period_str}")
    structured_district_kpis_list: List[Dict[str, Any]] = [] # Renamed for clarity

    if not isinstance(district_overall_kpis_summary, dict) or not district_overall_kpis_summary:
        logger.warning(f"({module_log_prefix}) No district overall KPI summary data provided. Returning empty list.")
        return structured_district_kpis_list

    # Determine total zones for percentage calculations if GDF is provided
    num_total_zones = 0
    if isinstance(district_enriched_gdf, pd.DataFrame) and not district_enriched_gdf.empty:
        if 'zone_id' in district_enriched_gdf.columns:
            num_total_zones = district_enriched_gdf['zone_id'].nunique()
        else: # Fallback if 'zone_id' somehow missing but GDF exists
            num_total_zones = len(district_enriched_gdf)
            logger.debug(f"({module_log_prefix}) 'zone_id' column not found in GDF, using GDF length ({num_total_zones}) for total zone count.")
    
    # --- KPI Definition and Structuring Logic ---

    # 1. Avg. Population AI Risk Score (Weighted)
    avg_population_ai_risk = district_overall_kpis_summary.get('population_weighted_avg_ai_risk_score', np.nan)
    pop_ai_risk_status_level = "NO_DATA"
    if pd.notna(avg_population_ai_risk):
        if avg_population_ai_risk >= app_config.RISK_SCORE_HIGH_THRESHOLD: pop_ai_risk_status_level = "HIGH_RISK"
        elif avg_population_ai_risk >= app_config.RISK_SCORE_MODERATE_THRESHOLD: pop_ai_risk_status_level = "MODERATE_RISK"
        else: pop_ai_risk_status_level = "ACCEPTABLE" # Low risk is considered acceptable
    structured_district_kpis_list.append({
        "metric_code": "DISTRICT_AVG_POPULATION_AI_RISK", "title": "Avg. Population AI Risk", 
        "value_str": f"{avg_population_ai_risk:.1f}" if pd.notna(avg_population_ai_risk) else "N/A", "units": "score", 
        "icon": "🎯", "status_level": pop_ai_risk_status_level, 
        "help_text": "Population-weighted average AI risk score across all zones in the district."
    })

    # 2. Facility Coverage Score (District Average, Weighted)
    avg_district_facility_coverage = district_overall_kpis_summary.get('district_avg_facility_coverage_score', np.nan)
    facility_coverage_status_level = "NO_DATA"
    if pd.notna(avg_district_facility_coverage):
        # Example thresholds for status; these could be configurable
        if avg_district_facility_coverage >= 80: facility_coverage_status_level = "GOOD_PERFORMANCE" 
        elif avg_district_facility_coverage >= app_config.DISTRICT_INTERVENTION_FACILITY_COVERAGE_LOW_PCT: facility_coverage_status_level = "MODERATE_CONCERN"
        else: facility_coverage_status_level = "HIGH_CONCERN"
    structured_district_kpis_list.append({
        "metric_code": "DISTRICT_AVG_FACILITY_COVERAGE", "title": "Facility Coverage Score (Avg %)", 
        "value_str": f"{avg_district_facility_coverage:.1f}" if pd.notna(avg_district_facility_coverage) else "N/A", "units": "%", 
        "icon": "🏥", "status_level": facility_coverage_status_level, 
        "help_text": f"Population-weighted average facility coverage score. Target > {app_config.DISTRICT_INTERVENTION_FACILITY_COVERAGE_LOW_PCT}%."
    })

    # 3. High AI Risk Zones (Count and Percentage)
    count_high_risk_zones = district_overall_kpis_summary.get('zones_meeting_high_risk_criteria_count', 0)
    percent_hr_zones_display_str = "N/A"
    high_risk_zones_status_level = "ACCEPTABLE" # Default if no high-risk zones
    
    if pd.notna(count_high_risk_zones):
        num_high_risk_zones_int = int(count_high_risk_zones)
        if num_total_zones > 0:
            percent_val_hr_zones = (num_high_risk_zones_int / num_total_zones) * 100
            percent_hr_zones_display_str = f"{percent_val_hr_zones:.0f}%"
            if percent_val_hr_zones > 30: high_risk_zones_status_level = "HIGH_CONCERN" # e.g., >30% of zones are high risk
            elif num_high_risk_zones_int > 0: high_risk_zones_status_level = "MODERATE_CONCERN" # Any high-risk zone is a concern
        elif num_high_risk_zones_int > 0 : # Have count, but no total_zones to calculate percentage
             percent_hr_zones_display_str = "(% unavailable)" # Indicate percentage cannot be calculated
             high_risk_zones_status_level = "MODERATE_CONCERN" # Still a concern if any such zones exist
        # If num_high_risk_zones_int is 0, status remains ACCEPTABLE

    structured_district_kpis_list.append({
        "metric_code": "DISTRICT_COUNT_HIGH_RISK_ZONES", "title": "High AI Risk Zones", 
        "value_str": f"{int(count_high_risk_zones) if pd.notna(count_high_risk_zones) else '0'} ({percent_hr_zones_display_str})", "units": "zones", 
        "icon": "⚠️", "status_level": high_risk_zones_status_level, 
        "help_text": f"Number (and percentage) of zones with average AI risk score ≥ {app_config.DISTRICT_ZONE_HIGH_RISK_AVG_SCORE}."
    })

    # 4. Overall Key Disease Prevalence per 1,000 Population
    district_prevalence_per_1k = district_overall_kpis_summary.get('district_overall_key_disease_prevalence_per_1000', np.nan)
    prevalence_status_level = "NO_DATA"
    if pd.notna(district_prevalence_per_1k): # Example thresholds, highly context-dependent and should be configurable
        if district_prevalence_per_1k > 50: prevalence_status_level = "HIGH_CONCERN" # e.g., >5% effective prevalence for combined key diseases
        elif district_prevalence_per_1k > 20: prevalence_status_level = "MODERATE_CONCERN"
        else: prevalence_status_level = "ACCEPTABLE"
    structured_district_kpis_list.append({
        "metric_code": "DISTRICT_KEY_DISEASE_PREVALENCE", "title": "Key Disease Prevalence", 
        "value_str": f"{district_prevalence_per_1k:.1f}" if pd.notna(district_prevalence_per_1k) else "N/A", "units": "/1k pop", 
        "icon": "📈", "status_level": prevalence_status_level, 
        "help_text": "Combined prevalence of specified key infectious diseases per 1,000 population."
    })

    # 5. Dynamically add KPIs for total active cases of each key condition
    default_condition_icon = "🌡️" # Generic health icon
    # Simplified icon map based on keywords in condition name
    condition_icon_keyword_map = {"TB": "🫁", "Malaria": "🦟", "HIV": "🩸", "Pneumonia": "💨", "Sepsis": "☣️", "Dehydration": "💧", "Heat": "☀️"}

    for condition_key_from_config in app_config.KEY_CONDITIONS_FOR_ACTION:
        # Construct the metric key name as used in get_district_summary_kpis output
        metric_key_name_summary = f"district_total_active_{condition_key_from_config.lower().replace(' ', '_').replace('-', '_').replace('(severe)','')}_cases"
        total_active_cases_condition = district_overall_kpis_summary.get(metric_key_name_summary, 0) # Default to 0 if key not found
        
        condition_burden_status_level = "ACCEPTABLE" # Default if low or zero cases
        if pd.notna(total_active_cases_condition):
            num_cases_condition = int(total_active_cases_condition)
            # Example thresholds for district-wide active cases of a specific key condition
            # These could be made more specific per condition in app_config if needed
            if num_cases_condition > app_config.DISTRICT_INTERVENTION_TB_BURDEN_HIGH_ABS * 2: # e.g., >20 for TB-like burden
                condition_burden_status_level = "HIGH_CONCERN" 
            elif num_cases_condition > app_config.DISTRICT_INTERVENTION_TB_BURDEN_HIGH_ABS * 0.5: # e.g., >5
                condition_burden_status_level = "MODERATE_CONCERN"
        
        # Attempt to find a representative icon based on keywords
        icon_for_this_condition = default_condition_icon
        condition_key_lower_search = condition_key_from_config.lower()
        for keyword, icon_char in condition_icon_keyword_map.items():
            if keyword.lower() in condition_key_lower_search:
                icon_for_this_condition = icon_char
                break
        
        display_name_for_condition_kpi = condition_key_from_config.replace("(Severe)", "").strip() # Cleaner title
        
        # Sanitize condition_key_from_config for metric_code
        metric_code_cond_part = condition_key_from_config.upper().replace(' ','_').replace('(SEVERE)','').replace('-','_').replace('/','_')

        structured_district_kpis_list.append({
            "metric_code": f"DIST_TOTAL_{metric_code_cond_part}_CASES", 
            "title": f"Total Active {display_name_for_condition_kpi} Cases", 
            "value_str": str(int(total_active_cases_condition)) if pd.notna(total_active_cases_condition) else "N/A", 
            "units": "cases", 
            "icon": icon_for_this_condition, "status_level": condition_burden_status_level, 
            "help_text": f"Total active {display_name_for_condition_kpi} cases identified across the district."
        })

    # 6. District Avg Patient Daily Steps (Population Weighted Wellness Proxy)
    avg_district_daily_steps = district_overall_kpis_summary.get('district_population_weighted_avg_steps', np.nan)
    steps_status_level = "NO_DATA"
    if pd.notna(avg_district_daily_steps):
        if avg_district_daily_steps >= app_config.TARGET_DAILY_STEPS * 0.8: steps_status_level = "GOOD_PERFORMANCE" # >=80% of target
        elif avg_district_daily_steps >= app_config.TARGET_DAILY_STEPS * 0.5: steps_status_level = "MODERATE_CONCERN" # 50-79%
        else: steps_status_level = "HIGH_CONCERN" # <50%
    structured_district_kpis_list.append({
        "metric_code": "DISTRICT_AVG_PATIENT_STEPS", "title": "Avg. Patient Steps (Pop. Wt.)", 
        "value_str": f"{avg_district_daily_steps:,.0f}" if pd.notna(avg_district_daily_steps) else "N/A", "units": "steps/day", 
        "icon": "👣", "status_level": steps_status_level, 
        "help_text": f"Population-weighted average daily steps from patient data. Target reference: {app_config.TARGET_DAILY_STEPS:,.0f} steps."
    })
    
    # 7. District Avg Clinic CO2 Levels
    avg_district_clinic_co2 = district_overall_kpis_summary.get('district_avg_clinic_co2_ppm', np.nan)
    clinic_co2_status_level = "NO_DATA"
    if pd.notna(avg_district_clinic_co2):
        if avg_district_clinic_co2 > app_config.ALERT_AMBIENT_CO2_VERY_HIGH_PPM : clinic_co2_status_level = "HIGH_RISK"
        elif avg_district_clinic_co2 > app_config.ALERT_AMBIENT_CO2_HIGH_PPM: clinic_co2_status_level = "MODERATE_RISK"
        else: clinic_co2_status_level = "ACCEPTABLE"
    structured_district_kpis_list.append({
        "metric_code": "DISTRICT_AVG_CLINIC_CO2", "title": "Avg. Clinic CO2 (District)", 
        "value_str": f"{avg_district_clinic_co2:.0f}" if pd.notna(avg_district_clinic_co2) else "N/A", "units": "ppm", 
        "icon": "💨", "status_level": clinic_co2_status_level, 
        "help_text": f"District average of zonal mean clinic CO2 levels. Aim for < {app_config.ALERT_AMBIENT_CO2_HIGH_PPM}ppm for good ventilation."
    })
    
    logger.info(f"({module_log_prefix}) Structured {len(structured_district_kpis_list)} district-level KPIs.")
    return structured_district_kpis_list
