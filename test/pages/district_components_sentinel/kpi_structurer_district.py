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
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_for_utils = os.path.abspath(os.path.join(current_script_dir, os.pardir, os.pardir))
    if project_root_for_utils not in sys.path:
        sys.path.insert(0, project_root_for_utils)
    from config import app_config

logger = logging.getLogger(__name__)

def structure_district_kpis_data(
    district_overall_kpis_summary: Optional[Dict[str, Any]],
    district_enriched_gdf: Optional[pd.DataFrame] = None, # GeoDataFrame expected
    reporting_period_str: str = "Latest Aggregated Data" # Default value
) -> List[Dict[str, Any]]:
    """
    Structures district-wide KPIs from a summary dictionary into a list of KPI dictionaries.
    """
    module_log_prefix = "DistrictKPIStructurer"
    logger.info(f"({module_log_prefix}) Structuring district KPIs for period: {reporting_period_str}")
    district_kpis_list: List[Dict[str, Any]] = []

    if not district_overall_kpis_summary:
        logger.warning(f"({module_log_prefix}) No district overall KPI summary data provided. Returning empty list.")
        return district_kpis_list

    total_zones_count = 0
    if isinstance(district_enriched_gdf, pd.DataFrame) and not district_enriched_gdf.empty:
        if 'zone_id' in district_enriched_gdf.columns:
            total_zones_count = district_enriched_gdf['zone_id'].nunique()
        else: # Fallback if zone_id somehow missing but GDF exists
            total_zones_count = len(district_enriched_gdf)
            logger.debug(f"({module_log_prefix}) 'zone_id' not in GDF, using GDF length for total_zones_count.")
    
    # --- KPI Definition and Structuring ---

    # 1. Avg. Population AI Risk
    avg_pop_ai_risk = district_overall_kpis_summary.get('population_weighted_avg_ai_risk_score', np.nan)
    pop_ai_risk_status = "NO_DATA"
    if pd.notna(avg_pop_ai_risk):
        if avg_pop_ai_risk >= app_config.RISK_SCORE_HIGH_THRESHOLD: pop_ai_risk_status = "HIGH_RISK"
        elif avg_pop_ai_risk >= app_config.RISK_SCORE_MODERATE_THRESHOLD: pop_ai_risk_status = "MODERATE_RISK"
        else: pop_ai_risk_status = "ACCEPTABLE" # Low risk is acceptable
    district_kpis_list.append({
        "metric_code": "DIST_AVG_POP_AI_RISK", "title": "Avg. Population AI Risk", 
        "value_str": f"{avg_pop_ai_risk:.1f}" if pd.notna(avg_pop_ai_risk) else "N/A", "units": "score", 
        "icon": "🎯", "status_level": pop_ai_risk_status, 
        "help_text": "Population-weighted average AI risk score across all zones in the district."
    })

    # 2. Facility Coverage Score (Average)
    avg_facility_coverage = district_overall_kpis_summary.get('district_avg_facility_coverage_score', np.nan)
    facility_coverage_status = "NO_DATA"
    if pd.notna(avg_facility_coverage):
        if avg_facility_coverage >= 80: facility_coverage_status = "GOOD_PERFORMANCE" # Example: 80%+ is good
        elif avg_facility_coverage >= app_config.DISTRICT_INTERVENTION_FACILITY_COVERAGE_LOW_PCT: facility_coverage_status = "MODERATE_CONCERN"
        else: facility_coverage_status = "HIGH_CONCERN"
    district_kpis_list.append({
        "metric_code": "DIST_AVG_FACILITY_COVERAGE", "title": "Facility Coverage Score (Avg %)", 
        "value_str": f"{avg_facility_coverage:.1f}" if pd.notna(avg_facility_coverage) else "N/A", "units": "%", 
        "icon": "🏥", "status_level": facility_coverage_status, 
        "help_text": f"Population-weighted average facility coverage score. Target > {app_config.DISTRICT_INTERVENTION_FACILITY_COVERAGE_LOW_PCT}%."
    })

    # 3. High AI Risk Zones Count & Percentage
    num_high_risk_zones = district_overall_kpis_summary.get('zones_meeting_high_risk_criteria_count', 0)
    percent_high_risk_zones_str = "N/A"
    high_risk_zones_status = "ACCEPTABLE" # Default if no high risk zones
    if pd.notna(num_high_risk_zones):
        num_hr_zones_int = int(num_high_risk_zones)
        if total_zones_count > 0:
            percent_val = (num_hr_zones_int / total_zones_count) * 100
            percent_high_risk_zones_str = f"{percent_val:.0f}%"
            if percent_val > 25: high_risk_zones_status = "HIGH_CONCERN" # >25% of zones are high risk
            elif num_hr_zones_int > 0: high_risk_zones_status = "MODERATE_CONCERN" # Any high risk zone is a concern
        elif num_hr_zones_int > 0 : # Have count, but no total_zones to calculate percentage
             percent_high_risk_zones_str = "(% unavailable)"
             high_risk_zones_status = "MODERATE_CONCERN" # Still a concern if any exist

    district_kpis_list.append({
        "metric_code": "DIST_COUNT_HIGH_RISK_ZONES", "title": "High AI Risk Zones", 
        "value_str": f"{int(num_high_risk_zones) if pd.notna(num_high_risk_zones) else '0'} ({percent_high_risk_zones_str})", "units": "zones", 
        "icon": "⚠️", "status_level": high_risk_zones_status, 
        "help_text": f"Number (and percentage) of zones with avg. AI risk score ≥ {app_config.DISTRICT_ZONE_HIGH_RISK_AVG_SCORE}."
    })

    # 4. Overall Key Disease Prevalence per 1,000 Population
    overall_prevalence_per_1k = district_overall_kpis_summary.get('district_overall_key_disease_prevalence_per_1000', np.nan)
    prevalence_status = "NO_DATA"
    if pd.notna(overall_prevalence_per_1k): # Example thresholds, highly context-dependent
        if overall_prevalence_per_1k > 50: prevalence_status = "HIGH_CONCERN" # e.g., >5% effective prevalence
        elif overall_prevalence_per_1k > 20: prevalence_status = "MODERATE_CONCERN"
        else: prevalence_status = "ACCEPTABLE"
    district_kpis_list.append({
        "metric_code": "DIST_KEY_DISEASE_PREVALENCE", "title": "Key Disease Prevalence", 
        "value_str": f"{overall_prevalence_per_1k:.1f}" if pd.notna(overall_prevalence_per_1k) else "N/A", "units": "/1k pop", 
        "icon": "📈", "status_level": prevalence_status, 
        "help_text": "Combined prevalence of specified key infectious diseases per 1,000 population."
    })

    # 5. Dynamically add KPIs for total active cases of each key condition
    default_cond_icon = "🌡️"
    condition_icons = {"TB": "🫁", "Malaria": "🦟", "HIV": "🩸", "Pneumonia": "💨"} # Simplified map

    for condition_key_cfg in app_config.KEY_CONDITIONS_FOR_ACTION:
        # Construct the column name as expected from get_district_summary_kpis
        metric_key_in_summary = f"district_total_active_{condition_key_cfg.lower().replace(' ', '_').replace('-', '_').replace('(severe)','')}_cases"
        total_cases_for_condition = district_overall_kpis_summary.get(metric_key_in_summary, 0)
        
        condition_burden_status = "ACCEPTABLE" # Default if low or zero
        if pd.notna(total_cases_for_condition):
            cases_count = int(total_cases_for_condition)
            # Generic thresholds for district-wide active cases of a key condition
            if cases_count > 50 : condition_burden_status = "HIGH_CONCERN"
            elif cases_count > 10: condition_burden_status = "MODERATE_CONCERN"
        
        # Attempt to find a representative icon
        icon_for_cond = default_cond_icon
        for iconic_disease, i_val in condition_icons.items():
            if iconic_disease.lower() in condition_key_cfg.lower():
                icon_for_cond = i_val
                break
        
        display_name_for_cond_kpi = condition_key_cfg.replace("(Severe)", "").strip()
        district_kpis_list.append({
            "metric_code": f"DIST_TOTAL_{condition_key_cfg.upper().replace(' ','_').replace('(SEVERE)','').replace('-','_')}_CASES", 
            "title": f"Total Active {display_name_for_cond_kpi} Cases", 
            "value_str": str(int(total_cases_for_condition)) if pd.notna(total_cases_for_condition) else "N/A", "units": "cases", 
            "icon": icon_for_cond, "status_level": condition_burden_status, 
            "help_text": f"Total active {display_name_for_cond_kpi} cases identified across the district."
        })

    # 6. District Avg Patient Daily Steps (Wellness Proxy)
    avg_district_steps = district_overall_kpis_summary.get('district_population_weighted_avg_steps', np.nan)
    steps_status = "NO_DATA"
    if pd.notna(avg_district_steps):
        if avg_district_steps >= app_config.TARGET_DAILY_STEPS * 0.75: steps_status = "GOOD_PERFORMANCE" # >=75% of target
        elif avg_district_steps >= app_config.TARGET_DAILY_STEPS * 0.50: steps_status = "MODERATE_CONCERN" # 50-74%
        else: steps_status = "HIGH_CONCERN" # <50%
    district_kpis_list.append({
        "metric_code": "DIST_AVG_PATIENT_STEPS", "title": "Avg. Patient Steps (Pop. Wt.)", 
        "value_str": f"{avg_district_steps:,.0f}" if pd.notna(avg_district_steps) else "N/A", "units": "steps/day", 
        "icon": "👣", "status_level": steps_status, 
        "help_text": f"Population-weighted average daily steps from patient data. Target approx. {app_config.TARGET_DAILY_STEPS:,.0f}."
    })
    
    # 7. District Avg Clinic CO2 Levels
    avg_clinic_co2_district = district_overall_kpis_summary.get('district_avg_clinic_co2_ppm', np.nan)
    clinic_co2_status = "NO_DATA"
    if pd.notna(avg_clinic_co2_district):
        if avg_clinic_co2_district > app_config.ALERT_AMBIENT_CO2_VERY_HIGH_PPM : clinic_co2_status = "HIGH_RISK"
        elif avg_clinic_co2_district > app_config.ALERT_AMBIENT_CO2_HIGH_PPM: clinic_co2_status = "MODERATE_RISK"
        else: clinic_co2_status = "ACCEPTABLE"
    district_kpis_list.append({
        "metric_code": "DIST_AVG_CLINIC_CO2", "title": "Avg. Clinic CO2 (District)", 
        "value_str": f"{avg_clinic_co2_district:.0f}" if pd.notna(avg_clinic_co2_district) else "N/A", "units": "ppm", 
        "icon": "💨", "status_level": clinic_co2_status, 
        "help_text": f"District average of zonal mean clinic CO2 levels. Aim for < {app_config.ALERT_AMBIENT_CO2_HIGH_PPM}ppm."
    })
    
    logger.info(f"({module_log_prefix}) Structured {len(district_kpis_list)} district-level KPIs.")
    return district_kpis_list
