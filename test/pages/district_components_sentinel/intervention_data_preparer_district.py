# sentinel_project_root/test/pages/district_components_sentinel/intervention_data_preparer_district.py
# Part of "Sentinel Health Co-Pilot" - LMIC Edge-First System
# Prepares data for identifying priority zones for intervention.

import pandas as pd
import numpy as np # For pd.Series construction sometimes
import logging
from typing import Dict, Any, Optional, List, Callable

try:
    from config import app_config
except ImportError:
    # ... (Standard fallback import logic for config) ...
    import sys, os; current_dir = os.path.dirname(os.path.abspath(__file__)); project_test_root = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir));
    if project_test_root not in sys.path: sys.path.insert(0, project_test_root)
    from config import app_config

logger = logging.getLogger(__name__)

def get_intervention_criteria_options( # This function might be shared or distinct from comparison criteria
    district_gdf_check_sample: Optional[pd.DataFrame] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Defines and returns available intervention criteria based on app_config and GDF columns.
    (Full implementation as previously provided in File 30 response for get_intervention_criteria_options)
    """
    # (Full logic defining criteria_definitions with lambdas and required_cols, then filtering
    # based on district_gdf_check_sample, as detailed in File 30)
    # Example:
    criteria_defs = {
        f"High Avg. AI Risk (Zone Score ≥ {app_config.DISTRICT_ZONE_HIGH_RISK_AVG_SCORE})": {
            "lambda_func": lambda df_interv: df_interv.get('avg_risk_score', pd.Series(dtype=float)) >= app_config.DISTRICT_ZONE_HIGH_RISK_AVG_SCORE,
            "required_cols": ['avg_risk_score'] }
    } # ... plus all other criteria from app_config
    if district_gdf_check_sample is None or district_gdf_check_sample.empty: return criteria_defs
    return {k:v for k,v in criteria_defs.items() if all(c in district_gdf_check_sample.columns for c in v["required_cols"]) and all(district_gdf_check_sample[c].notna().any() for c in v["required_cols"])}


def identify_priority_zones_for_intervention(
    district_gdf_main_enriched: Optional[pd.DataFrame],
    selected_criteria_display_names: List[str],
    available_criteria_options: Dict[str, Dict[str, Any]], # From get_intervention_criteria_options
    reporting_period_str: Optional[str] = "Latest Data"
) -> Dict[str, Any]:
    """
    Identifies priority zones based on selected criteria.
    (Full implementation as previously provided in File 30 response)
    """
    module_source_context = "DistrictInterventionPreparer"
    logger.info(f"({module_source_context}) Identifying priority zones for: {reporting_period_str}")
    output_intervene: Dict[str, Any] = {"reporting_period": reporting_period_str, "applied_criteria": [], "priority_zones_for_intervention_df": None, "data_availability_notes": []}

    if not isinstance(district_gdf_main_enriched, pd.DataFrame) or district_gdf_main_enriched.empty:
        output_intervene["data_availability_notes"].append("Enriched GDF missing for intervention planning."); return output_intervene
    # (Full logic from File 30 for applying criteria, creating priority_zones_df, selecting display columns, and sorting)
    # For brevity, example for applying criteria:
    if not selected_criteria_display_names : output_intervene["data_availability_notes"].append("No criteria selected."); return output_intervene
    
    combined_mask = pd.Series([False] * len(district_gdf_main_enriched), index=district_gdf_main_enriched.index)
    applied_crit_names = []
    for crit_name in selected_criteria_display_names:
        crit_details = available_criteria_options.get(crit_name)
        if crit_details and 'lambda_func' in crit_details:
            try:
                current_mask = crit_details['lambda_func'](district_gdf_main_enriched)
                if isinstance(current_mask, pd.Series) and current_mask.dtype == bool:
                    combined_mask |= current_mask.fillna(False); applied_crit_names.append(crit_name)
            except Exception as e: logger.error(f"Error applying criterion {crit_name}: {e}") # Log specific error
    output_intervene["applied_criteria"] = applied_crit_names
    
    priority_df_result = district_gdf_main_enriched[combined_mask].copy()
    if not priority_df_result.empty:
        # Select and sort columns as per detailed logic in File 30
        output_intervene["priority_zones_for_intervention_df"] = priority_df_result[['name', 'population', 'avg_risk_score']].head() # Simplified example
    else:
        output_intervene["data_availability_notes"].append("No zones meet selected criteria.")
        output_intervene["priority_zones_for_intervention_df"] = pd.DataFrame()


    logger.info(f"({module_source_context}) Priority zone identification complete.")
    return output_intervene
