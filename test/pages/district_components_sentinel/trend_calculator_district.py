# sentinel_project_root/test/pages/district_components_sentinel/trend_calculator_district.py
# Calculates district-wide health & environmental trend data for Sentinel DHO dashboards.

import pandas as pd
import numpy as np # Not directly used, but often with pandas
import logging
from typing import Dict, Any, Optional, Union, Callable # Added Callable

# Standardized import block
try:
    from config import app_config
    from utils.core_data_processing import get_trend_data
except ImportError:
    import sys
    import os
    # Assumes this file is in sentinel_project_root/test/pages/district_components_sentinel/
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_test_root_dir = os.path.abspath(os.path.join(current_script_dir, os.pardir, os.pardir))
    if project_test_root_dir not in sys.path:
        sys.path.insert(0, project_test_root_dir)
    from config import app_config
    from utils.core_data_processing import get_trend_data

logger = logging.getLogger(__name__)

def calculate_district_trends_data(
    filtered_health_for_trends: Optional[pd.DataFrame], # Health data already filtered for the trend period
    filtered_iot_for_trends: Optional[pd.DataFrame],    # IoT data already filtered for the trend period
    trend_start_date: Any, # Primarily for logging/context, as data is pre-filtered
    trend_end_date: Any,   # Primarily for logging/context
    reporting_period_str: str, # For the 'reporting_period' key in the output dict
    disease_agg_period: str = 'W-Mon', # Default: Weekly (Monday start) for disease incidence
    general_agg_period: str = 'D'      # Default: Daily for other general trends (AI risk, steps, CO2)
) -> Dict[str, Any]:
    """
    Calculates district-wide health and environmental trends using provided, period-filtered data.

    Args:
        filtered_health_for_trends: DataFrame of health records for the trend period.
        filtered_iot_for_trends: DataFrame of IoT environmental data for the trend period.
        trend_start_date, trend_end_date: For context/logging; data should be pre-filtered.
        reporting_period_str: String for the 'reporting_period' key in output.
        disease_agg_period: Aggregation period for disease incidence (e.g., 'W-Mon').
        general_agg_period: Aggregation period for other trends (e.g., 'D').

    Returns:
        Dict[str, Any]: Dictionary containing trend Series and processing notes.
    """
    module_log_prefix = "DistrictTrendCalculator" # Consistent prefix
    logger.info(f"({module_log_prefix}) Calculating district trends for period: {reporting_period_str} (Data from {str(trend_start_date)} to {str(trend_end_date)})")
    
    # Initialize output structure
    output_trends_dict: Dict[str, Any] = { # Renamed for clarity
        "reporting_period": reporting_period_str,
        "disease_incidence_trends": {}, # Dict: {condition_display_name: pd.Series of new case counts}
        "avg_patient_ai_risk_trend": None, # pd.Series
        "avg_patient_daily_steps_trend": None, # pd.Series
        "avg_clinic_co2_trend": None, # pd.Series (district-wide average of clinic means)
        "data_availability_notes": []
    }
    
    # Check if any data is available to process
    no_health_data_for_trends = not isinstance(filtered_health_for_trends, pd.DataFrame) or filtered_health_for_trends.empty
    no_iot_data_for_trends = not isinstance(filtered_iot_for_trends, pd.DataFrame) or filtered_iot_for_trends.empty

    if no_health_data_for_trends and no_iot_data_for_trends:
        note = "No health or IoT data provided for the selected trend period. Cannot calculate any trends."
        logger.warning(f"({module_log_prefix}) {note}")
        output_trends_dict["data_availability_notes"].append(note)
        return output_trends_dict

    # --- 1. Disease Incidence Trends (e.g., New Cases per Week/Month) ---
    if not no_health_data_for_trends and \
       'condition' in filtered_health_for_trends.columns and \
       'patient_id' in filtered_health_for_trends.columns and \
       'encounter_date' in filtered_health_for_trends.columns: # encounter_date is crucial
        
        disease_trends_map_buffer = {}
        for condition_name_config_key in app_config.KEY_CONDITIONS_FOR_ACTION:
            # Use a clean display name for keys in the output map
            display_name_for_condition_trend = condition_name_config_key.replace("(Severe)", "").strip()
            
            # Filter for records matching this specific condition (case-insensitive)
            # Ensure 'condition' column is string type for .str accessor
            condition_mask_for_trend_calc = filtered_health_for_trends['condition'].astype(str).str.contains(condition_name_config_key, case=False, na=False)
            df_this_condition_for_trend = filtered_health_for_trends[condition_mask_for_trend_calc]
            
            if not df_this_condition_for_trend.empty:
                # Assumption: For incidence trend, each unique patient_id within a period for a condition
                # represents a "newly identified/active case" within that aggregated period.
                # More complex logic might be needed for true epidemiological incidence (first-time diagnosis).
                series_incidence_trend = get_trend_data(
                    df=df_this_condition_for_trend,
                    value_col='patient_id', # We count unique patients as new/active cases in the period
                    date_col='encounter_date', # This column should be datetime from core_data_processing
                    period=disease_agg_period,
                    agg_func='nunique', # Number of unique patients (newly identified/active in period)
                    source_context=f"{module_log_prefix}/IncidenceTrend/{display_name_for_condition_trend}"
                )
                if isinstance(series_incidence_trend, pd.Series) and not series_incidence_trend.empty:
                    disease_trends_map_buffer[display_name_for_condition_trend] = series_incidence_trend
                else:
                    output_trends_dict["data_availability_notes"].append(f"No trend data generated for '{display_name_for_condition_trend}' incidence (empty series).")
            else: # No records for this specific condition in the filtered period
                 output_trends_dict["data_availability_notes"].append(f"No records found for condition '{display_name_for_condition_trend}' in the trend period.")
        output_trends_dict["disease_incidence_trends"] = disease_trends_map_buffer
    elif not no_health_data_for_trends: # Health data exists, but missing critical columns
        missing_cols_incid = [c for c in ['condition','patient_id','encounter_date'] if c not in filtered_health_for_trends.columns]
        output_trends_dict["data_availability_notes"].append(f"Health data missing critical columns for disease incidence trends: {missing_cols_incid}.")


    # --- 2. Average Patient AI Risk Score Trend ---
    if not no_health_data_for_trends and \
       'ai_risk_score' in filtered_health_for_trends.columns and \
       filtered_health_for_trends['ai_risk_score'].notna().any() and \
       'encounter_date' in filtered_health_for_trends.columns:
        
        series_ai_risk_trend = get_trend_data(
            df=filtered_health_for_trends, value_col='ai_risk_score', date_col='encounter_date',
            period=general_agg_period, agg_func='mean', # Average AI risk score over the period
            source_context=f"{module_log_prefix}/AIRiskScoreTrend"
        )
        if isinstance(series_ai_risk_trend, pd.Series) and not series_ai_risk_trend.empty:
            output_trends_dict["avg_patient_ai_risk_trend"] = series_ai_risk_trend
        else:
            output_trends_dict["data_availability_notes"].append("Could not generate AI risk score trend (empty series).")
    elif not no_health_data_for_trends:
        output_trends_dict["data_availability_notes"].append("AI risk score data ('ai_risk_score' or 'encounter_date') missing or all NaN for trend calculation.")

    # --- 3. Average Patient Daily Steps Trend ---
    if not no_health_data_for_trends and \
       'avg_daily_steps' in filtered_health_for_trends.columns and \
       filtered_health_for_trends['avg_daily_steps'].notna().any() and \
       'encounter_date' in filtered_health_for_trends.columns:

        series_daily_steps_trend = get_trend_data(
            df=filtered_health_for_trends, value_col='avg_daily_steps', date_col='encounter_date',
            period=general_agg_period, agg_func='mean',
            source_context=f"{module_log_prefix}/AvgDailyStepsTrend"
        )
        if isinstance(series_daily_steps_trend, pd.Series) and not series_daily_steps_trend.empty:
            output_trends_dict["avg_patient_daily_steps_trend"] = series_daily_steps_trend
        else:
            output_trends_dict["data_availability_notes"].append("Could not generate average daily steps trend (empty series).")
    elif not no_health_data_for_trends:
        output_trends_dict["data_availability_notes"].append("Average daily steps data ('avg_daily_steps' or 'encounter_date') missing or all NaN for trend calculation.")

    # --- 4. Average Clinic CO2 Levels Trend (District-wide average of clinic means from IoT data) ---
    if not no_iot_data_for_trends and \
       'avg_co2_ppm' in filtered_iot_for_trends.columns and \
       filtered_iot_for_trends['avg_co2_ppm'].notna().any() and \
       'timestamp' in filtered_iot_for_trends.columns: # IoT data uses 'timestamp'

        series_avg_co2_trend = get_trend_data(
            df=filtered_iot_for_trends, value_col='avg_co2_ppm', date_col='timestamp',
            period=general_agg_period, agg_func='mean',
            source_context=f"{module_log_prefix}/AvgClinicCO2Trend"
        )
        if isinstance(series_avg_co2_trend, pd.Series) and not series_avg_co2_trend.empty:
            output_trends_dict["avg_clinic_co2_trend"] = series_avg_co2_trend
        else:
            output_trends_dict["data_availability_notes"].append("Could not generate average clinic CO2 trend (empty series).")
    elif not no_iot_data_for_trends:
        output_trends_dict["data_availability_notes"].append("Clinic CO2 data ('avg_co2_ppm' or 'timestamp' in IoT data) missing or all NaN for trend calculation.")

    num_generated_trends = sum(1 for val_trend in output_trends_dict.values() if isinstance(val_trend, pd.Series) and not val_trend.empty) + \
                           len(output_trends_dict.get("disease_incidence_trends", {})) # Count disease trends separately
    logger.info(f"({module_log_prefix}) District trends calculation complete. Generated {num_generated_trends} distinct trend series. Notes: {len(output_trends_dict['data_availability_notes'])}")
    return output_trends_dict
