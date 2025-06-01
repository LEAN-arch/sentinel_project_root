# sentinel_project_root/test/pages/district_components_sentinel/kpi_structurer_district.py
# Part of "Sentinel Health Co-Pilot" - LMIC Edge-First System
# This module structures district-wide Key Performance Indicators (KPIs)
# based on aggregated zonal data. Output for DHO web dashboards/reports.

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional

try:
    from config import app_config
except ImportError:
    import sys, os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_test_root = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
    if project_test_root not in sys.path: sys.path.insert(0, project_test_root)
    from config import app_config

logger = logging.getLogger(__name__)

def structure_district_kpis_data(
    district_overall_kpis_summary: Optional[Dict[str, Any]],
    district_enriched_gdf: Optional[pd.DataFrame] = None,
    reporting_period_str: Optional[str] = "Latest Aggregated Data"
) -> List[Dict[str, Any]]:
    """
    Structures district-wide KPIs from a summary dictionary into a list of KPI dictionaries.
    (Full implementation as previously provided in File 30 response)
    """
    module_source_context = "DistrictKPIStructurer"
    logger.info(f"({module_source_context}) Structuring district KPIs for period: {reporting_period_str}")
    district_kpis_structured: List[Dict[str, Any]] = []

    if not district_overall_kpis_summary:
        logger.warning(f"({module_source_context}) No district overall KPI summary data provided.")
        return district_kpis_structured

    total_zones = 0
    if district_enriched_gdf is not None and not district_enriched_gdf.empty and 'zone_id' in district_enriched_gdf.columns:
        total_zones = district_enriched_gdf['zone_id'].nunique()
    elif district_enriched_gdf is not None and not district_enriched_gdf.empty : total_zones = len(district_enriched_gdf)


    # --- Define KPI transformation logic here ---
    # Avg. Population AI Risk
    avg_pop_risk = district_overall_kpis_summary.get('population_weighted_avg_ai_risk_score', np.nan)
    pop_risk_stat = "NO_DATA"
    if pd.notna(avg_pop_risk):
        if avg_pop_risk >= app_config.RISK_SCORE_HIGH_THRESHOLD: pop_risk_stat = "HIGH_RISK"
        elif avg_pop_risk >= app_config.RISK_SCORE_MODERATE_THRESHOLD: pop_risk_stat = "MODERATE_RISK"
        else: pop_risk_stat = "ACCEPTABLE" # Renamed from LOW_RISK for better general meaning
    district_kpis_structured.append({"metric_code": "DIST_AVG_POP_RISK", "title": "Avg. Population AI Risk", "value_str": f"{avg_pop_risk:.1f}" if pd.notna(avg_pop_risk) else "N/A", "units": "score", "icon": "🎯", "status_level": pop_risk_stat, "help_text": "Population-weighted avg. AI risk across zones."})

    # Facility Coverage Score
    facility_cov = district_overall_kpis_summary.get('district_avg_facility_coverage_score', np.nan)
    facility_stat = "NO_DATA"
    if pd.notna(facility_cov):
        if facility_cov >= 75: facility_stat = "GOOD_PERFORMANCE" # Example target for good coverage
        elif facility_cov >= app_config.DISTRICT_INTERVENTION_FACILITY_COVERAGE_LOW_PCT: facility_stat = "MODERATE_CONCERN"
        else: facility_stat = "HIGH_CONCERN"
    district_kpis_structured.append({"metric_code": "DIST_FACILITY_COVERAGE", "title": "Facility Coverage Score (Avg)", "value_str": f"{facility_cov:.1f}" if pd.notna(facility_cov) else "N/A", "units": "%", "icon": "🏥", "status_level": facility_stat, "help_text": "Population-weighted avg. facility coverage score."})

    # High AI Risk Zones
    high_risk_zone_count = district_overall_kpis_summary.get('zones_meeting_high_risk_criteria_count', 0)
    perc_hr_zones_str = "N/A"; hr_zone_stat = "ACCEPTABLE"
    if total_zones > 0 and pd.notna(high_risk_zone_count):
        perc_val_hr = (high_risk_zone_count / total_zones) * 100
        perc_hr_zones_str = f"{perc_val_hr:.0f}%"
        if perc_val_hr > 25: hr_zone_stat = "HIGH_CONCERN" # More than 25% of zones are high risk
        elif high_risk_zone_count > 0: hr_zone_stat = "MODERATE_CONCERN" # Any high risk zone is a concern
    elif pd.notna(high_risk_zone_count) and high_risk_zone_count > 0: perc_hr_zones_str = "(% N/A)" # Have count, no total zones
    district_kpis_structured.append({"metric_code": "DIST_HIGH_RISK_ZONES", "title": "High AI Risk Zones", "value_str": f"{int(high_risk_zone_count) if pd.notna(high_risk_zone_count) else '0'} ({perc_hr_zones_str})", "units": "zones", "icon": "⚠️", "status_level": hr_zone_stat, "help_text": f"Zones with avg. AI risk score ≥ {app_config.DISTRICT_ZONE_HIGH_RISK_AVG_SCORE}."})

    # Overall Key Disease Prevalence
    dist_prev = district_overall_kpis_summary.get('district_overall_key_disease_prevalence_per_1000', np.nan)
    prev_stat = "NO_DATA"
    if pd.notna(dist_prev):
        if dist_prev > 50: prev_stat = "HIGH_CONCERN" # e.g., >5% population with key disease
        elif dist_prev > 20: prev_stat = "MODERATE_CONCERN"
        else: prev_stat = "ACCEPTABLE"
    district_kpis_structured.append({"metric_code": "DIST_KEY_PREVALENCE", "title": "Key Disease Prevalence", "value_str": f"{dist_prev:.1f}" if pd.notna(dist_prev) else "N/A", "units": "/1k pop", "icon": "📈", "status_level": prev_stat, "help_text": "Combined prevalence of key infectious diseases per 1,000 pop."})

    # Dynamic Key Disease Burdens (using keys from get_district_summary_kpis output)
    for cond_key_kpi_dist in app_config.KEY_CONDITIONS_FOR_ACTION:
        metric_name_in_summary = f"district_total_active_{cond_key_kpi_dist.lower().replace(' ', '_').replace('-', '_').replace('(severe)','')}_cases"
        total_cases_cond = district_overall_kpis_summary.get(metric_name_in_summary, 0)
        cond_burden_stat = "ACCEPTABLE"
        # Generic thresholding for any key condition (can be made more specific)
        if pd.notna(total_cases_cond):
             # Simple thresholds based on absolute numbers, e.g. for a district
            if total_cases_cond > 50 : cond_burden_stat = "HIGH_CONCERN" # More than 50 active cases of this key condition district-wide
            elif total_cases_cond > 10: cond_burden_stat = "MODERATE_CONCERN"
        
        # Find display name & icon for the condition if available in KEY_TEST_TYPES (imperfect mapping, better to have separate condition config)
        # This part is a bit heuristic if a condition name doesn't map perfectly to a test config's disease_group.
        cond_icon = "🌡️" # Default icon
        cond_display_name_for_kpi = cond_key_kpi.replace("(Severe)", "").strip()
        # Attempt to find a more specific icon based on disease group matching condition name
        for test_cfg in app_config.KEY_TEST_TYPES_FOR_ANALYSIS.values():
            if cond_key_kpi.lower() in test_cfg.get("disease_group", "").lower():
                if "TB" in cond_key_kpi: cond_icon = "🫁"
                elif "Malaria" in cond_key_kpi: cond_icon = "🦟"
                elif "HIV" in cond_key_kpi: cond_icon = "🩸"
                elif "Pneumonia" in cond_key_kpi: cond_icon = "🫁" # Could share
                break
        
        district_kpis_structured.append({"metric_code": f"DIST_TOTAL_{cond_key_kpi.upper().replace(' ','_')}_CASES", "title": f"Total Active {cond_display_name_for_kpi} Cases", "value_str": str(int(total_cases_cond)) if pd.notna(total_cases_cond) else "N/A", "units": "cases", "icon": cond_icon, "status_level": cond_burden_stat, "help_text": f"Total active {cond_display_name_for_kpi} cases across the district."})

    # District Avg Patient Steps (Wellness Proxy)
    avg_steps = district_overall_kpis_summary.get('district_population_weighted_avg_steps', np.nan)
    steps_stat = "NO_DATA"
    if pd.notna(avg_steps):
        if avg_steps >= app_config.TARGET_DAILY_STEPS * 0.8: steps_stat = "GOOD_PERFORMANCE" # >= 80% of target
        elif avg_steps >= app_config.TARGET_DAILY_STEPS * 0.5: steps_stat = "MODERATE_CONCERN" # 50-80%
        else: steps_stat = "HIGH_CONCERN"
    district_kpis_structured.append({"metric_code": "DIST_AVG_STEPS", "title": "Avg. Patient Steps (Pop. Wt.)", "value_str": f"{avg_steps:,.0f}" if pd.notna(avg_steps) else "N/A", "units": "steps", "icon": "👣", "status_level": steps_stat, "help_text": f"Pop-weighted avg daily steps from patient data. Target ref: {app_config.TARGET_DAILY_STEPS:,.0f}."})
    
    # District Avg Clinic CO2
    avg_co2 = district_overall_kpis_summary.get('district_avg_clinic_co2_ppm', np.nan)
    co2_stat = "NO_DATA"
    if pd.notna(avg_co2):
        if avg_co2 > app_config.ALERT_AMBIENT_CO2_VERY_HIGH_PPM : co2_stat = "HIGH_RISK"
        elif avg_co2 > app_config.ALERT_AMBIENT_CO2_HIGH_PPM: co2_stat = "MODERATE_RISK"
        else: co2_stat = "ACCEPTABLE"
    district_kpis_structured.append({"metric_code": "DIST_AVG_CLINIC_CO2", "title": "Avg. Clinic CO2 (District)", "value_str": f"{avg_co2:.0f}" if pd.notna(avg_co2) else "N/A", "units": "ppm", "icon": "💨", "status_level": co2_stat, "help_text": f"District avg of zonal mean clinic CO2 levels. Aim < {app_config.ALERT_AMBIENT_CO2_HIGH_PPM}ppm."})
    
    logger.info(f"({module_source_context}) Structured {len(district_kpis_structured)} district KPIs.")
    return district_kpis_structured
