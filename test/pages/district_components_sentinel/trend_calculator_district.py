# sentinel_project_root/test/pages/district_components_sentinel/trend_calculator_district.py
# Calculates district-wide health & environmental trend data for Sentinel DHO dashboards.

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Union, Callable # Added Callable

# Standardized import block
try:
    from config import app_config
    from utils.core_data_processing import get_trend_data
except ImportError:
    import sys
    import os
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_for_utils = os.path.abspath(os.path.join(current_script_dir, os.pardir, os.pardir))
    if project_root_for_utils not in sys.path:
        sys.path.insert(0, project_root_for_utils)
    from config import app_config
    from utils.core_data_processing import get_trend_data

logger = logging.getLogger(__name__)

def calculate_district_trends_data(
    filtered_health_for_trends: Optional[pd.DataFrame],
    filtered_iot_for_trends: Optional[pd.DataFrame],
    trend_start_date: Any, # For logging/context, get_trend_data filters internally
    trend_end_date: Any,   # For logging/context
    reporting_period_str: str, # For the 'reporting_period' key in output
    disease_agg_period: str = 'W-Mon', # Default: Weekly for disease incidence
    general_agg_period: str = 'D'      # Default: Daily for other general trends
) -> Dict[str, Any]:
    """
    Calculates district-wide health and environmental trends using provided filtered data.
    The input DataFrames should already be filtered for the desired trend period.
    """
    module_log_prefix = "DistrictTrendCalculator"
    logger.info(f"({module_log_prefix}) Calculating district trends for period: {reporting_period_str}")
    
    output_trends: Dict[str, Any] = {
        "reporting_period": reporting_period_str,
        "disease_incidence_trends": {}, # Dict: {condition_display_name: pd.Series}
        "avg_patient_ai_risk_trend": None, # pd.Series
        "avg_patient_daily_steps_trend": None, # pd.Series
        "avg_clinic_co2_trend": None, # pd.Series
        "data_availability_notes": []
    }
    
    no_health_data = not isinstance(filtered_health_for_trends, pd.DataFrame) or filtered_health_for_trends.empty
    no_iot_data = not isinstance(filtered_iot_for_trends, pd.DataFrame) or filtered_iot_for_trends.empty

    if no_health_data and no_iot_data:
        note = "No health or IoT data provided for the trend period. Cannot calculate trends."
        logger.warning(f"({module_log_prefix}) {note}")
        output_trends["data_availability_notes"].append(note)
        return output_trends

    # --- 1. Disease Incidence Trends (e.g., New Cases per Week) ---
    if not no_health_data and 'condition' in filtered_health_for_trends.columns and 'patient_id' in filtered_health_for_trends.columns:
        disease_trends_map = {}
        for condition_name_cfg in app_config.KEY_CONDITIONS_FOR_ACTION:
            display_name_cond = condition_name_cfg.replace("(Severe)", "").strip()
            
            # Filter for this specific condition (case-insensitive)
            condition_mask_trend = filtered_health_for_trends['condition'].str.contains(condition_name_cfg, case=False, na=False)
            df_condition_specific_trend = filtered_health_for_trends[condition_mask_trend]
            
            if not df_condition_specific_trend.empty:
                # Assuming 'encounter_date' marks new identification for this trend period.
                # Count unique patients per period for this condition.
                incidence_trend_series = get_trend_data(
                    df=df_condition_specific_trend,
                    value_col='patient_id', # Count unique patients
                    date_col='encounter_date', # Assumed to be datetime from loader
                    period=disease_agg_period,
                    agg_func='nunique', # Number of unique patients (new cases in period)
                    source_context=f"{module_log_prefix}/Incidence/{display_name_cond}"
                )
                if isinstance(incidence_trend_series, pd.Series) and not incidence_trend_series.empty:
                    disease_trends_map[display_name_cond] = incidence_trend_series
                else:
                    output_trends["data_availability_notes"].append(f"No trend data generated for {display_name_cond} incidence.")
            else:
                 output_trends["data_availability_notes"].append(f"No records found for condition '{display_name_cond}' in the period.")
        output_trends["disease_incidence_trends"] = disease_trends_map
    elif not no_health_data:
        output_trends["data_availability_notes"].append("Health data missing 'condition' or 'patient_id' for disease incidence trends.")


    # --- 2. Average Patient AI Risk Score Trend ---
    if not no_health_data and 'ai_risk_score' in filtered_health_for_trends.columns and \
       filtered_health_for_trends['ai_risk_score'].notna().any():
        ai_risk_trend = get_trend_data(
            df=filtered_health_for_trends, value_col='ai_risk_score', date_col='encounter_date',
            period=general_agg_period, agg_func='mean',
            source_context=f"{module_log_prefix}/AIRiskTrend"
        )
        if isinstance(ai_risk_trend, pd.Series) and not ai_risk_trend.empty:
            output_trends["avg_patient_ai_risk_trend"] = ai_risk_trend
        else:
            output_trends["data_availability_notes"].append("Could not generate AI risk score trend.")
    elif not no_health_data:
        output_trends["data_availability_notes"].append("AI risk score data missing or all NaN for trend.")

    # --- 3. Average Patient Daily Steps Trend ---
    if not no_health_data and 'avg_daily_steps' in filtered_health_for_trends.columns and \
       filtered_health_for_trends['avg_daily_steps'].notna().any():
        daily_steps_trend = get_trend_data(
            df=filtered_health_for_trends, value_col='avg_daily_steps', date_col='encounter_date',
            period=general_agg_period, agg_func='mean',
            source_context=f"{module_log_prefix}/DailyStepsTrend"
        )
        if isinstance(daily_steps_trend, pd.Series) and not daily_steps_trend.empty:
            output_trends["avg_patient_daily_steps_trend"] = daily_steps_trend
        else:
            output_trends["data_availability_notes"].append("Could not generate average daily steps trend.")
    elif not no_health_data:
        output_trends["data_availability_notes"].append("Average daily steps data missing or all NaN for trend.")

    # --- 4. Average Clinic CO2 Levels Trend (District-wide average of clinic means) ---
    if not no_iot_data and 'avg_co2_ppm' in filtered_iot_for_trends.columns and \
       filtered_iot_for_trends['avg_co2_ppm'].notna().any():
        # This assumes filtered_iot_for_trends contains data from multiple clinics/rooms across the district.
        # The trend will be the average of all reported avg_co2_ppm values per period.
        avg_co2_trend = get_trend_data(
            df=filtered_iot_for_trends, value_col='avg_co2_ppm', date_col='timestamp', # IoT uses 'timestamp'
            period=general_agg_period, agg_func='mean',
            source_context=f"{module_log_prefix}/ClinicCO2Trend"
        )
        if isinstance(avg_co2_trend, pd.Series) and not avg_co2_trend.empty:
            output_trends["avg_clinic_co2_trend"] = avg_co2_trend
        else:
            output_trends["data_availability_notes"].append("Could not generate average clinic CO2 trend.")
    elif not no_iot_data:
        output_trends["data_availability_notes"].append("Clinic CO2 data (avg_co2_ppm) missing or all NaN for trend.")

    num_trends_generated = sum(1 for val in output_trends.values() if isinstance(val, pd.Series) and not val.empty) + \
                           len(output_trends.get("disease_incidence_trends", {}))
    logger.info(f"({module_log_prefix}) District trends calculation complete. Generated {num_trends_generated} trend series. Notes: {len(output_trends['data_availability_notes'])}")
    return output_trends
