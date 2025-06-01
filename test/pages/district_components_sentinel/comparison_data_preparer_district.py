# sentinel_project_root/test/pages/district_components_sentinel/comparison_data_preparer_district.py
# Prepares data for DHO zonal comparative analysis for Sentinel Health Co-Pilot.

import pandas as pd
import numpy as np # For np.nan
import logging
from typing import Dict, Any, Optional, List

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

def get_comparison_criteria_options_district(
    district_gdf_sample: Optional[pd.DataFrame] = None
) -> Dict[str, Dict[str, str]]:
    """
    Defines metrics available for zonal comparison, checking against GDF sample if provided.
    Returns: {Display Name: {"col": actual_col_name, "format_str": "{:.1f}", "colorscale_hint": "Viridis"}}
    """
    # Define all potential comparison metrics with their properties
    # These column names must match those produced by enrich_zone_geodata_with_health_aggregates
    all_comparison_metrics = {
        "Avg. AI Risk Score (Zone)": {"col": "avg_risk_score", "format_str": "{:.1f}", "colorscale_hint": "OrRd_r"},
        "Key Disease Prevalence (/1k pop)": {"col": "prevalence_per_1000", "format_str": "{:.1f}", "colorscale_hint": "YlOrRd_r"},
        "Facility Coverage Score (%)": {"col": "facility_coverage_score", "format_str": "{:.1f}%", "colorscale_hint": "Greens_r"}, # Higher is better
        "Population (Total)": {"col": "population", "format_str": "{:,.0f}", "colorscale_hint": "Blues"},
        "CHW Density (/10k pop)": {"col": "chw_density_per_10k", "format_str": "{:.2f}", "colorscale_hint": "Greens_r"}, # Placeholder, needs chw_count_zone
        "Avg. Clinic CO2 (ppm)": {"col": "zone_avg_co2", "format_str": "{:.0f}", "colorscale_hint": "Oranges_r"},
        "Population Density (per sqkm)": {"col": "population_density", "format_str": "{:.1f}", "colorscale_hint": "Plasma"},
        "Avg. Critical Test TAT (days)": {"col": "avg_test_turnaround_critical", "format_str": "{:.1f}", "colorscale_hint": "Reds_r"},
        "% Critical Tests TAT Met": {"col": "perc_critical_tests_tat_met", "format_str": "{:.1f}%", "colorscale_hint": "Greens_r"},
        "Total Patient Encounters": {"col": "total_patient_encounters", "format_str": "{:,.0f}", "colorscale_hint": "Blues"},
    }
    # Dynamically add active cases for key conditions
    for cond_key in app_config.KEY_CONDITIONS_FOR_ACTION:
        col_name = f"active_{cond_key.lower().replace(' ', '_').replace('-', '_').replace('(severe)','')}_cases"
        display_cond_name = cond_key.replace("(Severe)", "").strip()
        all_comparison_metrics[f"Active {display_cond_name} Cases"] = {"col": col_name, "format_str": "{:.0f}", "colorscale_hint": "Purples_r"}

    if not isinstance(district_gdf_sample, pd.DataFrame) or district_gdf_sample.empty:
        logger.debug("No GDF sample provided to get_comparison_criteria_options_district, returning all defined metrics.")
        return all_comparison_metrics

    # Filter metrics based on column existence and non-null data in the sample
    available_metrics = {}
    for display_name, details in all_comparison_metrics.items():
        col = details["col"]
        if col in district_gdf_sample.columns and district_gdf_sample[col].notna().any():
            available_metrics[display_name] = details
        else:
            logger.debug(f"Metric '{display_name}' (column '{col}') excluded: column missing or all NaN in GDF sample.")
            
    return available_metrics


def prepare_zonal_comparison_data(
    district_gdf_main_enriched: Optional[pd.DataFrame], # Actually a GeoDataFrame, but pandas DataFrame for typing simplicity here
    reporting_period_str: str = "Latest Aggregated Data" # Default value for reporting period
) -> Dict[str, Any]:
    """
    Prepares data for zonal comparative analysis tables and provides metric configuration.
    """
    module_log_prefix = "DistrictComparisonPreparer"
    logger.info(f"({module_log_prefix}) Preparing zonal comparison data for: {reporting_period_str}")
    
    output_data: Dict[str, Any] = {
        "reporting_period": reporting_period_str,
        "comparison_metrics_config": {}, # Will store {display_name: {col, format_str, ...}}
        "zonal_comparison_table_df": None, # DataFrame for table display
        "data_availability_notes": []
    }
    
    if not isinstance(district_gdf_main_enriched, pd.DataFrame) or district_gdf_main_enriched.empty:
        note = "Enriched District GeoDataFrame is missing or empty. Cannot prepare comparison data."
        logger.warning(f"({module_log_prefix}) {note}")
        output_data["data_availability_notes"].append(note)
        return output_data

    # Get available metrics configuration based on the provided GDF
    # Pass a sample of the GDF (e.g., head) to check for column existence and data
    available_metrics_for_comparison = get_comparison_criteria_options_district(district_gdf_main_enriched.head())
    
    if not available_metrics_for_comparison:
        note = "No valid metrics found for zonal comparison based on the provided GeoDataFrame."
        logger.warning(f"({module_log_prefix}) {note}")
        output_data["data_availability_notes"].append(note)
        return output_data
        
    output_data["comparison_metrics_config"] = available_metrics_for_comparison
    
    # Determine the zone identifier column (prefer 'name', fallback to 'zone_id')
    zone_identifier_col = 'name'
    if 'name' not in district_gdf_main_enriched.columns or district_gdf_main_enriched['name'].isnull().all():
        if 'zone_id' in district_gdf_main_enriched.columns:
            zone_identifier_col = 'zone_id'
        else: # Should not happen if GDF is from load_zone_data
            note = "Critical error: Neither 'name' nor 'zone_id' found in GeoDataFrame for comparison table indexing."
            logger.error(f"({module_log_prefix}) {note}")
            output_data["data_availability_notes"].append(note)
            return output_data
            
    # Select columns for the comparison table: zone identifier + all available metric columns
    columns_for_table = [zone_identifier_col] + [details['col'] for details in available_metrics_for_comparison.values()]
    # Ensure all selected columns actually exist in the GDF to prevent KeyErrors
    actual_columns_in_gdf = [col for col in columns_for_table if col in district_gdf_main_enriched.columns]
    
    if len(actual_columns_in_gdf) <= 1: # Only zone identifier, no metrics
        note = "No metric columns available in GDF for comparison table after filtering."
        logger.warning(f"({module_log_prefix}) {note}")
        output_data["data_availability_notes"].append(note)
        output_data["zonal_comparison_table_df"] = pd.DataFrame(columns=[zone_identifier_col]) # Empty table with ID col
        return output_data

    comparison_df = district_gdf_main_enriched[actual_columns_in_gdf].copy()
    
    # Set the zone identifier as index, but also keep it as a regular column for flexibility
    # (e.g., if a plotting function prefers it as a column)
    if zone_identifier_col in comparison_df.columns:
        comparison_df = comparison_df.set_index(zone_identifier_col, drop=False)
    
    output_data["zonal_comparison_table_df"] = comparison_df
    
    logger.info(f"({module_log_prefix}) Zonal comparison data prepared with {len(comparison_df)} zones and {len(actual_columns_in_gdf)-1} metrics.")
    return output_data
