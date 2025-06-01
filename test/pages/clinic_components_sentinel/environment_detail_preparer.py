# sentinel_project_root/test/pages/clinic_components_sentinel/environment_detail_preparer.py
# Prepares detailed environmental data from clinic IoT sensors for Sentinel.

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List

# Standardized import block
try:
    from config import app_config
    from utils.core_data_processing import get_trend_data, get_clinic_environmental_summary
except ImportError:
    import sys
    import os
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_for_utils = os.path.abspath(os.path.join(current_script_dir, os.pardir, os.pardir))
    if project_root_for_utils not in sys.path:
        sys.path.insert(0, project_root_for_utils)
    from config import app_config
    from utils.core_data_processing import get_trend_data, get_clinic_environmental_summary

logger = logging.getLogger(__name__)

def prepare_clinic_environment_details_data(
    filtered_iot_df_clinic_period: Optional[pd.DataFrame],
    iot_data_source_exists: bool, # General availability of IoT source
    reporting_period_str: str
) -> Dict[str, Any]:
    """
    Prepares structured data for detailed environmental trends and latest room readings.

    Args:
        filtered_iot_df_clinic_period: DataFrame of IoT readings for the clinic and period.
        iot_data_source_exists: Flag indicating if IoT data source is generally available.
        reporting_period_str: String describing the reporting period.

    Returns:
        Dictionary containing structured environmental detail data.
    """
    module_log_prefix = "ClinicEnvDetailPreparer"
    logger.info(f"({module_log_prefix}) Preparing clinic environment details for period: {reporting_period_str}")

    env_details_output: Dict[str, Any] = {
        "reporting_period": reporting_period_str,
        "current_environmental_alerts_list": [], # List of dicts: {alert_type, message, level, icon}
        "hourly_avg_co2_trend": None,            # pd.Series: Index=datetime, Value=avg_co2
        "hourly_avg_occupancy_trend": None,      # pd.Series: Index=datetime, Value=avg_occupancy (waiting rooms)
        "latest_room_sensor_readings_df": None,  # pd.DataFrame: Latest readings by room
        "processing_notes": []                   # For issues or context
    }

    if not isinstance(filtered_iot_df_clinic_period, pd.DataFrame) or filtered_iot_df_clinic_period.empty:
        note_msg = ""
        if iot_data_source_exists:
            note_msg = f"No clinic environmental IoT data found for the period '{reporting_period_str}' to prepare details."
            logger.info(f"({module_log_prefix}) {note_msg}")
        else:
            note_msg = "IoT data source appears generally unavailable; environmental monitoring details cannot be prepared."
            logger.warning(f"({module_log_prefix}) {note_msg}")
        env_details_output["processing_notes"].append(note_msg)
        return env_details_output

    df_iot_period = filtered_iot_df_clinic_period.copy()

    if 'timestamp' not in df_iot_period.columns:
        note_msg = "Critical 'timestamp' column missing from IoT data. Cannot process environment details."
        logger.error(f"({module_log_prefix}) {note_msg}")
        env_details_output["processing_notes"].append(note_msg)
        return env_details_output
        
    df_iot_period['timestamp'] = pd.to_datetime(df_iot_period['timestamp'], errors='coerce')
    df_iot_period.dropna(subset=['timestamp'], inplace=True) # Remove rows where timestamp conversion failed

    if df_iot_period.empty:
        note_msg = "No IoT records with valid timestamps in the provided period for environment details."
        logger.info(f"({module_log_prefix}) {note_msg}")
        env_details_output["processing_notes"].append(note_msg)
        return env_details_output

    # 1. Current Environmental Alerts Summary (from latest readings in period)
    #    Uses get_clinic_environmental_summary from core_data_processing for consistent alert logic.
    env_summary_latest = get_clinic_environmental_summary(
        df_iot_period, # Pass the period data; summary func will find latest internally
        source_context=f"{module_log_prefix}/CurrentAlertsFromSummary"
    )
    
    current_alerts = []
    if env_summary_latest.get('rooms_co2_very_high_alert_latest_count', 0) > 0:
        current_alerts.append({
            "alert_type": "CO2 Contamination",
            "message": f"{env_summary_latest['rooms_co2_very_high_alert_latest_count']} area(s) with CO2 > {app_config.ALERT_AMBIENT_CO2_VERY_HIGH_PPM}ppm.",
            "level": "HIGH_RISK", "icon": "💨"
        })
    if env_summary_latest.get('rooms_pm25_very_high_alert_latest_count', 0) > 0:
        current_alerts.append({
            "alert_type": "Air Quality PM2.5",
            "message": f"{env_summary_latest['rooms_pm25_very_high_alert_latest_count']} area(s) with PM2.5 > {app_config.ALERT_AMBIENT_PM25_VERY_HIGH_UGM3}µg/m³.",
            "level": "HIGH_RISK", "icon": "🌫️"
        })
    if env_summary_latest.get('rooms_noise_high_alert_latest_count', 0) > 0:
        current_alerts.append({
            "alert_type": "High Noise Level",
            "message": f"{env_summary_latest['rooms_noise_high_alert_latest_count']} area(s) with Noise > {app_config.ALERT_AMBIENT_NOISE_HIGH_DBA}dB.",
            "level": "HIGH_CONCERN", "icon": "🔊"
        })
    if env_summary_latest.get('waiting_room_high_occupancy_alert_latest_flag', False):
        current_alerts.append({
            "alert_type": "Waiting Area Overcrowding",
            "message": f"High Occupancy: At least one waiting area > {app_config.TARGET_CLINIC_WAITING_ROOM_OCCUPANCY_MAX} persons.",
            "level": "HIGH_CONCERN", "icon": "👨‍👩‍👧‍👦"
        })
    
    if not current_alerts: # If no specific critical alerts
        current_alerts.append({"alert_type": "General Environment", "message": "No critical environmental alerts identified from latest readings in period.", "level": "ACCEPTABLE", "icon":"✅"})
    env_details_output["current_environmental_alerts_list"] = current_alerts

    # 2. Hourly Trends for Key Environmental Metrics
    # CO2 Trend (Overall average for the clinic)
    if 'avg_co2_ppm' in df_iot_period.columns and df_iot_period['avg_co2_ppm'].notna().any():
        co2_trend = get_trend_data(
            df_iot_period, value_col='avg_co2_ppm', date_col='timestamp', 
            period='H', agg_func='mean', # Hourly average
            source_context=f"{module_log_prefix}/HourlyCO2Trend"
        )
        if isinstance(co2_trend, pd.Series) and not co2_trend.empty:
            env_details_output["hourly_avg_co2_trend"] = co2_trend.rename("avg_co2_ppm_hourly")
        else: env_details_output["processing_notes"].append("Could not generate hourly CO2 trend.")
    else: env_details_output["processing_notes"].append("CO2 data ('avg_co2_ppm') missing or all NaN for trend.")

    # Waiting Room Occupancy Trend
    if 'waiting_room_occupancy' in df_iot_period.columns and df_iot_period['waiting_room_occupancy'].notna().any():
        # Filter for records likely representing waiting rooms
        waiting_room_iot_df = df_iot_period[
            df_iot_period.get('room_name', pd.Series(dtype=str)).str.contains('Waiting', case=False, na=False)
        ]
        if not waiting_room_iot_df.empty:
            occupancy_trend = get_trend_data(
                waiting_room_iot_df, value_col='waiting_room_occupancy', date_col='timestamp', 
                period='H', agg_func='mean',
                source_context=f"{module_log_prefix}/HourlyOccupancyTrend"
            )
            if isinstance(occupancy_trend, pd.Series) and not occupancy_trend.empty:
                env_details_output["hourly_avg_occupancy_trend"] = occupancy_trend.rename("avg_occupancy_hourly")
            else: env_details_output["processing_notes"].append("Could not generate hourly waiting room occupancy trend.")
        else: env_details_output["processing_notes"].append("No data points identified for 'waiting_room_occupancy' in waiting areas for trend.")
    else: env_details_output["processing_notes"].append("Waiting room occupancy data column missing or all NaN for trend.")

    # 3. Latest Sensor Readings by Room
    desired_cols_for_latest = [
        'clinic_id', 'room_name', 'timestamp', 
        'avg_co2_ppm', 'max_co2_ppm', 'avg_pm25', 'voc_index', 
        'avg_temp_celsius', 'avg_humidity_rh', 'avg_noise_db',
        'waiting_room_occupancy', 'patient_throughput_per_hour', 'sanitizer_dispenses_per_hour'
    ]
    # Filter to only columns that actually exist in the DataFrame
    available_cols_for_latest = [col for col in desired_cols_for_latest if col in df_iot_period.columns]

    if 'clinic_id' in available_cols_for_latest and 'room_name' in available_cols_for_latest:
        # Get the absolute latest reading for each unique room in the period
        latest_readings_df = df_iot_period.sort_values('timestamp', ascending=True).drop_duplicates(
            subset=['clinic_id', 'room_name'], keep='last' # Keep the last (most recent)
        )
        if not latest_readings_df.empty:
            env_details_output["latest_room_sensor_readings_df"] = latest_readings_df[available_cols_for_latest].reset_index(drop=True)
        else:
            env_details_output["processing_notes"].append("No distinct room sensor readings found for the latest point in this period.")
    else:
        missing_keys = [key for key in ['clinic_id', 'room_name'] if key not in available_cols_for_latest]
        env_details_output["processing_notes"].append(f"Essential columns for latest room readings missing: {missing_keys}.")
        
    logger.info(f"({module_log_prefix}) Clinic environment details preparation finished. Notes: {len(env_details_output['processing_notes'])}")
    return env_details_output
