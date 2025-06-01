# sentinel_project_root/test/pages/clinic_components_sentinel/environment_detail_preparer.py
# Prepares detailed environmental data from clinic IoT sensors for Sentinel.

import pandas as pd
import numpy as np # For np.nan if still needed, less direct use now
import logging
from typing import Dict, Any, Optional, List

# Standardized import block
try:
    from config import app_config
    from utils.core_data_processing import get_trend_data, get_clinic_environmental_summary
except ImportError:
    import sys
    import os
    # Assumes this file is in sentinel_project_root/test/pages/clinic_components_sentinel/
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_test_root_dir = os.path.abspath(os.path.join(current_script_dir, os.pardir, os.pardir))
    if project_test_root_dir not in sys.path:
        sys.path.insert(0, project_test_root_dir)
    from config import app_config
    from utils.core_data_processing import get_trend_data, get_clinic_environmental_summary

logger = logging.getLogger(__name__)

def prepare_clinic_environment_details_data(
    filtered_iot_df_clinic_period: Optional[pd.DataFrame],
    iot_data_source_exists: bool, # General availability of IoT source file/connection
    reporting_period_str: str
) -> Dict[str, Any]:
    """
    Prepares structured data for detailed environmental trends and latest room readings.

    Args:
        filtered_iot_df_clinic_period: DataFrame of IoT readings for the clinic and period.
        iot_data_source_exists: Flag indicating if IoT data source is generally available.
        reporting_period_str: String describing the reporting period for context.

    Returns:
        Dictionary containing structured environmental detail data.
    """
    module_log_prefix = "ClinicEnvDetailPreparer" # Consistent prefix
    logger.info(f"({module_log_prefix}) Preparing clinic environment details for period: {reporting_period_str}")

    env_details_output: Dict[str, Any] = {
        "reporting_period": reporting_period_str,
        "current_environmental_alerts_list": [], # List of dicts: {alert_type, message, level, icon}
        "hourly_avg_co2_trend": None,            # pd.Series: Index=datetime, Value=avg_co2
        "hourly_avg_occupancy_trend": None,      # pd.Series: Index=datetime, Value=avg_occupancy (waiting rooms)
        "latest_room_sensor_readings_df": None,  # pd.DataFrame: Latest readings by room
        "processing_notes": []                   # For issues or contextual info during processing
    }

    if not isinstance(filtered_iot_df_clinic_period, pd.DataFrame) or filtered_iot_df_clinic_period.empty:
        note_to_add = ""
        if iot_data_source_exists: # Source exists, but no data for this period/filter
            note_to_add = f"No clinic environmental IoT data found for the period '{reporting_period_str}' to prepare details."
            logger.info(f"({module_log_prefix}) {note_to_add}")
        else: # Source itself is likely missing
            note_to_add = "IoT data source appears generally unavailable; environmental monitoring details cannot be prepared."
            logger.warning(f"({module_log_prefix}) {note_to_add}")
        env_details_output["processing_notes"].append(note_to_add)
        return env_details_output

    df_iot_current_period_src = filtered_iot_df_clinic_period.copy() # Work on a copy

    # Critical: Ensure 'timestamp' column exists and is valid datetime
    if 'timestamp' not in df_iot_current_period_src.columns:
        critical_note = "Critical 'timestamp' column missing from IoT data. Cannot process environment details."
        logger.error(f"({module_log_prefix}) {critical_note}")
        env_details_output["processing_notes"].append(critical_note)
        return env_details_output
        
    df_iot_current_period_src['timestamp'] = pd.to_datetime(df_iot_current_period_src['timestamp'], errors='coerce')
    df_iot_current_period_src.dropna(subset=['timestamp'], inplace=True) # Remove rows where timestamp conversion failed

    if df_iot_current_period_src.empty:
        empty_note = "No IoT records with valid timestamps found in the provided period for environment details."
        logger.info(f"({module_log_prefix}) {empty_note}")
        env_details_output["processing_notes"].append(empty_note)
        return env_details_output

    # 1. Current Environmental Alerts Summary (derived from latest readings in period)
    #    get_clinic_environmental_summary is expected to use the latest reading per room within the df_iot_current_period_src.
    env_summary_from_core = get_clinic_environmental_summary(
        df_iot_current_period_src, 
        source_context=f"{module_log_prefix}/LatestAlertsSummary"
    )
    
    current_alerts_list_buffer = []
    # Structure alerts based on the summary from core_data_processing
    if env_summary_from_core.get('rooms_co2_very_high_alert_latest_count', 0) > 0:
        current_alerts_list_buffer.append({
            "alert_type": "High CO2 Levels", # User-friendly type
            "message": f"{env_summary_from_core['rooms_co2_very_high_alert_latest_count']} area(s) with CO2 > {app_config.ALERT_AMBIENT_CO2_VERY_HIGH_PPM}ppm (Very High).",
            "level": "HIGH_RISK", "icon": "💨"
        })
    if env_summary_from_core.get('rooms_pm25_very_high_alert_latest_count', 0) > 0:
        current_alerts_list_buffer.append({
            "alert_type": "Poor Air Quality (PM2.5)",
            "message": f"{env_summary_from_core['rooms_pm25_very_high_alert_latest_count']} area(s) with PM2.5 > {app_config.ALERT_AMBIENT_PM25_VERY_HIGH_UGM3}µg/m³ (Very High).",
            "level": "HIGH_RISK", "icon": "🌫️"
        })
    if env_summary_from_core.get('rooms_noise_high_alert_latest_count', 0) > 0:
        current_alerts_list_buffer.append({
            "alert_type": "High Noise Levels",
            "message": f"{env_summary_from_core['rooms_noise_high_alert_latest_count']} area(s) with Noise > {app_config.ALERT_AMBIENT_NOISE_HIGH_DBA}dBA (Sustained).",
            "level": "MODERATE_CONCERN", "icon": "🔊" # Noise usually less critical than CO2/PM2.5 unless extreme
        })
    if env_summary_from_core.get('waiting_room_high_occupancy_alert_latest_flag', False):
        current_alerts_list_buffer.append({
            "alert_type": "Waiting Area Overcrowding",
            "message": f"High Occupancy: At least one waiting area exceeded {app_config.TARGET_CLINIC_WAITING_ROOM_OCCUPANCY_MAX} persons.",
            "level": "MODERATE_CONCERN", "icon": "👨‍👩‍👧‍👦"
        })
    
    if not current_alerts_list_buffer: # If no specific critical/warning alerts
        current_alerts_list_buffer.append({"alert_type": "Environmental Check", "message": "No significant environmental alerts identified from latest readings in this period.", "level": "ACCEPTABLE", "icon":"✅"})
    env_details_output["current_environmental_alerts_list"] = current_alerts_list_buffer

    # 2. Hourly Trends for Key Environmental Metrics using df_iot_current_period_src
    # CO2 Trend (Overall average for the clinic if multiple rooms/sensors)
    if 'avg_co2_ppm' in df_iot_current_period_src.columns and df_iot_current_period_src['avg_co2_ppm'].notna().any():
        co2_hourly_trend = get_trend_data(
            df_iot_current_period_src, value_col='avg_co2_ppm', date_col='timestamp', 
            period='H', agg_func='mean', # Hourly average
            source_context=f"{module_log_prefix}/CO2HourlyTrend"
        )
        if isinstance(co2_hourly_trend, pd.Series) and not co2_hourly_trend.empty:
            env_details_output["hourly_avg_co2_trend"] = co2_hourly_trend.rename("avg_co2_ppm_hourly")
        else: env_details_output["processing_notes"].append("Could not generate hourly CO2 trend for the period.")
    else: env_details_output["processing_notes"].append("CO2 data ('avg_co2_ppm') column missing or all NaN for trend calculation.")

    # Waiting Room Occupancy Trend (Average occupancy in designated waiting areas)
    if 'waiting_room_occupancy' in df_iot_current_period_src.columns and df_iot_current_period_src['waiting_room_occupancy'].notna().any():
        # Filter for records that likely represent waiting rooms
        df_waiting_areas_iot = df_iot_current_period_src[
            # Ensure 'room_name' access is safe (it should be str after loader)
            df_iot_current_period_src.get('room_name', pd.Series(dtype=str)).str.contains('Waiting', case=False, na=False)
        ]
        if not df_waiting_areas_iot.empty:
            occupancy_hourly_trend = get_trend_data(
                df_waiting_areas_iot, value_col='waiting_room_occupancy', date_col='timestamp', 
                period='H', agg_func='mean',
                source_context=f"{module_log_prefix}/WaitingOccupancyHourlyTrend"
            )
            if isinstance(occupancy_hourly_trend, pd.Series) and not occupancy_hourly_trend.empty:
                env_details_output["hourly_avg_occupancy_trend"] = occupancy_hourly_trend.rename("avg_waiting_occupancy_hourly")
            else: env_details_output["processing_notes"].append("Could not generate hourly waiting room occupancy trend.")
        else: env_details_output["processing_notes"].append("No data points specifically identified as 'waiting_room_occupancy' in waiting areas found for trend.")
    else: env_details_output["processing_notes"].append("Waiting room occupancy data column ('waiting_room_occupancy') missing or all NaN for trend.")

    # 3. Latest Sensor Readings by Room (from end of selected period)
    desired_cols_for_latest_table = [
        'clinic_id', 'room_name', 'timestamp', 
        'avg_co2_ppm', 'max_co2_ppm', 'avg_pm25', 'voc_index', 
        'avg_temp_celsius', 'avg_humidity_rh', 'avg_noise_db',
        'waiting_room_occupancy', 'patient_throughput_per_hour', 'sanitizer_dispenses_per_hour'
    ]
    # Filter to only columns that actually exist in the DataFrame to avoid KeyErrors
    available_cols_for_latest_table = [col for col in desired_cols_for_latest_table if col in df_iot_current_period_src.columns]

    # Need 'clinic_id' and 'room_name' to identify unique rooms
    if 'clinic_id' in available_cols_for_latest_table and 'room_name' in available_cols_for_latest_table:
        # Get the absolute latest (last) reading for each unique room within the period
        df_latest_readings_by_room = df_iot_current_period_src.sort_values('timestamp', ascending=True).drop_duplicates(
            subset=['clinic_id', 'room_name'], keep='last'
        )
        if not df_latest_readings_by_room.empty:
            env_details_output["latest_room_sensor_readings_df"] = df_latest_readings_by_room[available_cols_for_latest_table].reset_index(drop=True)
        else:
            env_details_output["processing_notes"].append("No distinct room sensor readings found for the latest point in this period after initial data filtering.")
    else:
        missing_key_cols_for_latest = [key_col for key_col in ['clinic_id','room_name'] if key_col not in available_cols_for_latest_table]
        env_details_output["processing_notes"].append(f"Essential columns for identifying latest room readings are missing: {missing_key_cols_for_latest}.")
        
    logger.info(f"({module_log_prefix}) Clinic environment details preparation finished. Notes recorded: {len(env_details_output['processing_notes'])}")
    return env_details_output
