# sentinel_project_root/test/pages/district_components_sentinel/comparison_data_preparer_district.py
# Prepares data for DHO zonal comparative analysis for Sentinel Health Co-Pilot.

import pandas as pd
import numpy as np # For np.nan if still needed
import logging
from typing import Dict, Any, Optional, List

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

def get_comparison_criteria_options_district(
    district_gdf_sample_check: Optional[pd.DataFrame] = None # Use a more descriptive name
) -> Dict[str, Dict[str, str]]: # Return type: {Display Name: {"col": col_name, "format_str": "fmt", "colorscale_hint": "scale"}}
    """
    Defines metrics available for zonal comparison tables/charts, checking against GDF sample columns.
    The column names ('col') must match those produced by `enrich_zone_geodata_with_health_aggregates`.
    """
    # Define all potential comparison metrics with their properties for display and potential charting
    all_potential_comparison_metrics = {
        "Avg. AI Risk Score (Zone)": {"col": "avg_risk_score", "format_str": "{:.1f}", "colorscale_hint": "OrRd_r"},
        "Key Disease Prevalence (/1k pop)": {"col": "prevalence_per_1000", "format_str": "{:.1f}", "colorscale_hint": "YlOrRd_r"},
        "Facility Coverage Score (%)": {"col": "facility_coverage_score", "format_str": "{:.1f}%", "colorscale_hint": "Greens_r"}, # Higher is better
        "Population (Total)": {"col": "population", "format_str": "{:,.0f}", "colorscale_hint": "Blues"},
        "CHW Density (/10k pop)": {"col": "chw_density_per_10k", "format_str": "{:.2f}", "colorscale_hint": "Greens_r"}, # This column needs to be in GDF
        "Avg. Clinic CO2 (ppm)": {"col": "zone_avg_co2", "format_str": "{:.0f}", "colorscale_hint": "Oranges_r"}, # Higher is worse
        "Population Density (per sqkm)": {"col": "population_density", "format_str": "{:.1f}", "colorscale_hint": "Plasma_r"},
        "Avg. Critical Test TAT (days)": {"col": "avg_test_turnaround_critical", "format_str": "{:.1f}", "colorscale_hint": "Reds_r"}, # Higher is worse
        "% Critical Tests TAT Met": {"col": "perc_critical_tests_tat_met", "format_str": "{:.1f}%", "colorscale_hint": "Greens_r"}, # Higher is better
        "Total Patient Encounters (Zone)": {"col": "total_patient_encounters", "format_str": "{:,.0f}", "colorscale_hint": "Purples"},
        "Avg. Patient Daily Steps (Zone)": {"col": "avg_daily_steps_zone", "format_str": "{:,.0f}", "colorscale_hint": "BuGn_r"} # Higher is better
    }
    # Dynamically add active cases for key conditions from app_config
    for condition_key_name in app_config.KEY_CONDITIONS_FOR_ACTION:
        # Construct column name exactly as created in enrich_zone_geodata_with_health_aggregates
        col_name_for_condition = f"active_{condition_key_name.lower().replace(' ', '_').replace('-', '_').replace('(severe)','')}_cases"
        display_condition_label = condition_key_name.replace("(Severe)", "").strip() # Cleaner label
        all_potential_comparison_metrics[f"Active {display_condition_label} Cases (Zone)"] = {
            "col": col_name_for_condition, 
            "format_str": "{:.0f}", 
            "colorscale_hint": "Reds_r" # Default for disease burden, higher is worse
        }

    if not isinstance(district_gdf_sample_check, pd.DataFrame) or district_gdf_sample_check.empty:
        logger.debug("No GDF sample provided to get_comparison_criteria_options_district, returning all defined potential metrics.")
        return all_potential_comparison_metrics

    # Filter metrics: only include if the required column exists in the GDF sample and has some non-null data
    available_metrics_for_config = {}
    for display_name_metric, metric_details in all_potential_comparison_metrics.items():
        column_name_metric = metric_details["col"]
        if column_name_metric in district_gdf_sample_check.columns and \
           district_gdf_sample_check[column_name_metric].notna().any(): # Check if at least one non-NaN value exists
            available_metrics_for_config[display_name_metric] = metric_details
        else:
            logger.debug(f"Comparison metric '{display_name_metric}' (column '{column_name_metric}') excluded: column missing or all NaN in GDF sample.")
            
    return available_metrics_for_config


def prepare_zonal_comparison_data(
    district_gdf_main_enriched: Optional[pd.DataFrame], # GeoDataFrame expected, using pd.DataFrame for broader type hint
    reporting_period_str: str = "Latest Aggregated Data"
) -> Dict[str, Any]:
    """
    Prepares data for zonal comparative analysis tables and provides metric configuration.
    The input district_gdf_main_enriched should be the output of enrich_zone_geodata_with_health_aggregates.
    """
    module_log_prefix = "DistrictComparisonPreparer" # Consistent prefix
    logger.info(f"({module_log_prefix}) Preparing zonal comparison data for reporting period: {reporting_period_str}")
    
    output_comparison_data: Dict[str, Any] = {
        "reporting_period": reporting_period_str,
        "comparison_metrics_config": {}, # Stores {display_name: {col, format_str, ...}} for available metrics
        "zonal_comparison_table_df": None, # DataFrame for table display (Zone as index, metrics as columns)
        "data_availability_notes": []
    }
    
    if not isinstance(district_gdf_main_enriched, pd.DataFrame) or district_gdf_main_enriched.empty: # Using pd.DataFrame for check as GDF is subclass
        note = "Enriched District GeoDataFrame is missing or empty. Cannot prepare zonal comparison data."
        logger.warning(f"({module_log_prefix}) {note}")
        output_comparison_data["data_availability_notes"].append(note)
        output_comparison_data["zonal_comparison_table_df"] = pd.DataFrame() # Ensure empty DF for consistency
        return output_comparison_data

    # Get available metrics configuration based on the columns present in the provided GDF
    # Pass a sample of the GDF (e.g., head) to check for column existence and non-null data
    available_metrics_config = get_comparison_criteria_options_district(district_gdf_main_enriched.head(2)) # Pass small sample
    
    if not available_metrics_config:
        note = "No valid metrics found for zonal comparison based on the columns and data in the provided GeoDataFrame."
        logger.warning(f"({module_log_prefix}) {note}")
        output_comparison_data["data_availability_notes"].append(note)
        output_comparison_data["zonal_comparison_table_df"] = pd.DataFrame()
        return output_comparison_data
        
    output_comparison_data["comparison_metrics_config"] = available_metrics_config
    
    # Determine the zone identifier column (prefer 'name' for display, fallback to 'zone_id')
    zone_id_display_col = 'name' # User-friendly name for display
    if 'name' not in district_gdf_main_enriched.columns or district_gdf_main_enriched['name'].isnull().all():
        if 'zone_id' in district_gdf_main_enriched.columns:
            zone_id_display_col = 'zone_id' # Fallback to zone_id if 'name' is unusable
        else: 
            note = "Critical error: Neither 'name' nor 'zone_id' found in GeoDataFrame. Cannot create comparison table."
            logger.error(f"({module_log_prefix}) {note}")
            output_comparison_data["data_availability_notes"].append(note)
            output_comparison_data["zonal_comparison_table_df"] = pd.DataFrame()
            return output_comparison_data
            
    # Select columns for the comparison table: zone identifier + all available metric columns
    # Ensure 'zone_id' is also included if it's different from zone_id_display_col and needed for joins later, though not primary display.
    columns_for_comparison_table = [zone_id_display_col] + [details['col'] for details in available_metrics_config.values()]
    if 'zone_id' in district_gdf_main_enriched.columns and 'zone_id' not in columns_for_comparison_table:
        columns_for_comparison_table.append('zone_id') # Ensure zone_id is there if not the display col

    # Ensure all selected columns actually exist in the GDF to prevent KeyErrors during selection
    final_columns_for_table = [col for col in list(set(columns_for_comparison_table)) if col in district_gdf_main_enriched.columns] # Use set for unique, then list
    
    if len(final_columns_for_table) <= 1 and zone_id_display_col not in final_columns_for_table : # Only zone identifier, no actual metrics, or identifier missing
        note = "No metric columns available in GDF for comparison table after filtering based on config and GDF content."
        logger.warning(f"({module_log_prefix}) {note}")
        output_comparison_data["data_availability_notes"].append(note)
        output_comparison_data["zonal_comparison_table_df"] = pd.DataFrame(columns=[zone_id_display_col] if zone_id_display_col in district_gdf_main_enriched.columns else [])
        return output_comparison_data

    df_comparison_table = district_gdf_main_enriched[final_columns_for_table].copy()
    
    # Set the zone identifier as index for the table, but also keep it as a regular column for flexibility
    if zone_id_display_col in df_comparison_table.columns:
        # Using .set_index but keeping the column via .copy() or by re-adding if needed,
        # or ensure the plotting functions can use index or a column. For direct df display, index is good.
        df_comparison_table = df_comparison_table.set_index(zone_id_display_col, drop=False) 
        df_comparison_table.index.name = "Zone" # Set index name for clarity
    
    output_comparison_data["zonal_comparison_table_df"] = df_comparison_table
    
    num_metrics_in_table = len(available_metrics_config)
    logger.info(f"({module_log_prefix}) Zonal comparison data prepared with {len(df_comparison_table)} zones and {num_metrics_in_table} metrics.")
    return output_comparison_data
