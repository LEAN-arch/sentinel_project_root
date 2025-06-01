# sentinel_project_root/test/pages/clinic_components_sentinel/environment_detail_preparer.py
# Part of "Sentinel Health Co-Pilot" - LMIC Edge-First System
# This module prepares detailed environmental data from clinic IoT sensors,
# including current alerts, trends, and latest room readings.
# Refactored from the original clinic_components/environment_details_tab.py.
# Output is structured data for display on the Clinic Management Console (Tier 2).

import pandas as pd
import numpy as np # For np.nan if needed, though less directly here
import logging
from typing import Dict, Any, Optional, List

# Assuming app_config and core_data_processing utilities are accessible
try:
    from config import app_config
    from utils.core_data_processing import get_trend_data, get_clinic_environmental_summary
except ImportError:
    import sys, os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_test_root = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
    if project_test_root not in sys.path: sys.path.insert(0, project_test_root)
    from config import app_config
    from utils.core_data_processing import get_trend_data, get_clinic_environmental_summary

logger = logging.getLogger(__name__)

def prepare_clinic_environment_details_data(
    filtered_iot_df_clinic_period: Optional[pd.DataFrame],
    iot_data_source_exists: bool, # True if the main IoT data source file/connection is generally available
    reporting_period_str: str
) -> Dict[str, Any]:
    """
    Prepares structured data for detailed environmental trends and latest room readings.

    Args:
        filtered_iot_df_clinic_period: DataFrame containing processed IoT sensor readings
                                       for the clinic over the specified reporting period.
        iot_data_source_exists: Flag indicating if the IoT data source is generally expected to be available.
        reporting_period_str: A string describing the reporting period for context.

    Returns:
        Dict[str, Any]: A dictionary containing structured environmental detail data.
    """
    module_source_context = "ClinicEnvDetailPreparer"
    logger.info(f"({module_source_context}) Preparing clinic environment details data for period: {reporting_period_str}")

    env_details_data: Dict[str, Any] = {
        "reporting_period": reporting_period_str,
        "current_environmental_alerts_list": [], # List of dicts: {alert_type, message, level}
        "hourly_avg_co2_trend": None,            # pd.Series: Index=datetime, Value=avg_co2
        "hourly_avg_occupancy_trend": None,      # pd.Series: Index=datetime, Value=avg_occupancy
        "latest_room_sensor_readings_df": None,  # pd.DataFrame: Cols based on available IoT data
        "processing_notes": []                   # For issues like missing data for a specific part
    }

    if filtered_iot_df_clinic_period is None or filtered_iot_df_clinic_period.empty:
        if iot_data_source_exists:
            note = f"No clinic environmental IoT data found for the period '{reporting_period_str}' to prepare details."
            logger.info(f"({module_source_context}) {note}")
            env_details_data["processing_notes"].append(note)
        else:
            note = "IoT data source appears generally unavailable; environmental monitoring details cannot be prepared."
            logger.warning(f"({module_source_context}) {note}")
            env_details_data["processing_notes"].append(note)
        return env_details_data

    df_iot_current_period = filtered_iot_df_clinic_period.copy()
    if 'timestamp' not in df_iot_current_period.columns or \
       not pd.api.types.is_datetime64_any_dtype(df_iot_current_period['timestamp']):
        note = "Timestamp data missing or invalid in provided IoT records. Cannot process environment details."
        logger.error(f"({module_source_context}) {note}")
        env_details_data["processing_notes"].append(note)
        return env_details_data # Critical for most operations here
    
    df_iot_current_period['timestamp'] = pd.to_datetime(df_iot_current_period['timestamp'], errors='coerce')
    df_iot_current_period_valid_ts = df_iot_current_period.dropna(subset=['timestamp'])

    if df_iot_current_period_valid_ts.empty:
        note = "No IoT records with valid timestamps in the provided period for environment details."
        logger.info(f"({module_source_context}) {note}")
        env_details_data["processing_notes"].append(note)
        return env_details_data


    # 1. Generate Current Environmental Alerts Summary (from latest readings in period)
    #    Uses `get_clinic_environmental_summary` from core_data_processing for consistency.
    env_summary_for_alerts_list = get_clinic_environmental_summary(
        df_iot_current_period_valid_ts, # Use data with valid timestamps
        source_context=f"{module_source_context}/CurrentAlerts"
    )
    
    alerts_list_output = []
    # Key names from get_clinic_environmental_summary are like 'rooms_co2_very_high_alert_latest_count'
    if env_summary_for_alerts_list.get('rooms_co2_very_high_alert_latest_count', 0) > 0:
        alerts_list_output.append({
            "alert_type": "CO2 Contamination", # More descriptive for manager
            "message": f"{env_summary_for_alerts_list['rooms_co2_very_high_alert_latest_count']} area(s) with CO2 > {app_config.ALERT_AMBIENT_CO2_VERY_HIGH_PPM}ppm.",
            "level": "HIGH_RISK", "icon": "💨"
        })
    if env_summary_for_alerts_list.get('rooms_pm25_very_high_alert_latest_count', 0) > 0:
        alerts_list_output.append({
            "alert_type": "Air Quality PM2.5",
            "message": f"{env_summary_for_alerts_list['rooms_pm25_very_high_alert_latest_count']} area(s) with PM2.5 > {app_config.ALERT_AMBIENT_PM25_VERY_HIGH_UGM3}µg/m³.",
            "level": "HIGH_RISK", "icon": "🌫️"
        })
    if env_summary_for_alerts_list.get('rooms_noise_high_alert_latest_count', 0) > 0:
        alerts_list_output.append({
            "alert_type": "High Noise Level",
            "message": f"{env_summary_for_alerts_list['rooms_noise_high_alert_latest_count']} area(s) with Noise > {app_config.ALERT_AMBIENT_NOISE_HIGH_DBA}dB.",
            "level": "HIGH_CONCERN", "icon": "🔊" # Using "concern" as it's about comfort/communication more than immediate life threat
        })
    if env_summary_for_alerts_list.get('waiting_room_high_occupancy_alert_latest_flag', False):
        alerts_list_output.append({
            "alert_type": "Waiting Area Overcrowding",
            "message": f"High Occupancy: At least one waiting area > {app_config.TARGET_CLINIC_WAITING_ROOM_OCCUPANCY_MAX} persons.",
            "level": "HIGH_CONCERN", "icon": "👨‍👩‍👧‍👦"
        })
    
    if not alerts_list_output: # If no specific critical alerts from the summary
        alerts_list_output.append({"alert_type": "General Environment", "message": "No critical environmental alerts identified from latest readings in period.", "level": "ACCEPTABLE", "icon":"✅"})
    env_details_data["current_environmental_alerts_list"] = alerts_list_output


    # 2. Hourly Trends for Key Environmental Metrics (using df_iot_current_period_valid_ts)
    # CO2 Trend
    if 'avg_co2_ppm' in df_iot_current_period_valid_ts.columns and df_iot_current_period_valid_ts['avg_co2_ppm'].notna().any():
        co2_trend_series = get_trend_data(
            df_iot_current_period_valid_ts, value_col='avg_co2_ppm',
            date_col='timestamp', period='H', agg_func='mean', # Hourly average
            source_context=f"{module_source_context}/CO2Trend"
        )
        if co2_trend_series is not None and not co2_trend_series.empty:
            env_details_data["hourly_avg_co2_trend"] = co2_trend_series.rename("avg_co2_ppm_hourly")
        else: env_details_data["processing_notes"].append("Could not generate hourly CO2 trend for the period.")
    else: env_details_data["processing_notes"].append("CO2 data ('avg_co2_ppm') column missing or all NaN for trend calculation.")

    # Waiting Room Occupancy Trend
    if 'waiting_room_occupancy' in df_iot_current_period_valid_ts.columns and df_iot_current_period_valid_ts['waiting_room_occupancy'].notna().any():
        # Filter for records that might reasonably represent waiting room to avoid averaging unrelated rooms
        # This assumes room_name might contain 'waiting' or similar. A more robust system would use room type flags.
        waiting_room_data_for_trend = df_iot_current_period_valid_ts[
            df_iot_current_period_valid_ts.get('room_name', pd.Series(dtype=str)).str.contains('Waiting', case=False, na=False)
        ]
        if not waiting_room_data_for_trend.empty:
            occupancy_trend_series = get_trend_data(
                waiting_room_data_for_trend, value_col='waiting_room_occupancy',
                date_col='timestamp', period='H', agg_func='mean', # Hourly average
                source_context=f"{module_source_context}/OccupancyTrend"
            )
            if occupancy_trend_series is not None and not occupancy_trend_series.empty:
                env_details_data["hourly_avg_occupancy_trend"] = occupancy_trend_series.rename("avg_occupancy_hourly")
            else: env_details_data["processing_notes"].append("Could not generate hourly waiting room occupancy trend.")
        else: env_details_data["processing_notes"].append("No data points specifically for 'waiting_room_occupancy' in waiting areas found for trend.")
    else: env_details_data["processing_notes"].append("Waiting room occupancy data column missing or all NaN for trend.")


    # 3. Latest Sensor Readings by Room (from end of selected period)
    cols_for_latest_readings = [ # Prioritized list of columns for the room summary table
        'clinic_id', 'room_name', 'timestamp', # Identifying info
        'avg_co2_ppm', 'avg_pm25', 'avg_temp_celsius', 'avg_humidity_rh', 'avg_noise_db', # Core env
        'waiting_room_occupancy', # Specific to some rooms
        'patient_throughput_per_hour', 'sanitizer_dispenses_per_hour' # Operational proxies if available
    ]
    actual_cols_available_for_latest = [col for col in cols_for_latest_readings if col in df_iot_current_period_valid_ts.columns]

    if all(c in actual_cols_available_for_latest for c in ['clinic_id', 'room_name']): # timestamp already validated
        # Get the absolute latest reading for each unique room in the period_valid_ts DataFrame
        latest_readings_per_room_df = df_iot_current_period_valid_ts.sort_values('timestamp').drop_duplicates(
            subset=['clinic_id', 'room_name'], keep='last'
        )
        if not latest_readings_per_room_df.empty:
            env_details_data["latest_room_sensor_readings_df"] = latest_readings_per_room_df[actual_cols_available_for_latest].reset_index(drop=True)
        else:
            env_details_data["processing_notes"].append("No distinct room sensor readings found for the latest point in this period after initial filtering.")
    else:
        missing_key_cols_latest = [c for c in ['clinic_id','room_name'] if c not in actual_cols_available_for_latest]
        env_details_data["processing_notes"].append(f"Essential columns for latest room readings missing: {missing_key_cols_latest}.")
        
    logger.info(f"({module_source_context}) Clinic environment details data preparation finished. Notes: {len(env_details_data['processing_notes'])}")
    return env_details_data
