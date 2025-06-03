# sentinel_project_root/test/pages/chw_components_sentinel/activity_trend_calculator.py
# Calculates CHW activity trend data for Sentinel Health Co-Pilot.

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Union
from datetime import date as date_type, datetime
import streamlit as st

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Standardized import block
try:
    from config import app_config
    from utils.core_data_processing import get_trend_data
except ImportError as e:
    logger.error(f"Import error: {e}. Ensure config and utils modules are in test/ with __init__.py.")
    raise

def calculate_chw_activity_trends(
    chw_historical_health_df: Optional[pd.DataFrame],
    trend_start_date_input: Any,
    trend_end_date_input: Any,
    zone_filter: Optional[str] = None,
    time_period_aggregation: str = 'D',
    source_context_log_prefix: str = "CHWActivityTrends"
) -> Dict[str, Optional[pd.Series]]:
    """
    Calculates CHW activity trends (visits, high-priority follow-ups) over a period.

    Args:
        chw_historical_health_df: DataFrame with historical health records for CHW/team.
                                  Expected columns: 'encounter_date', 'patient_id', 
                                  'ai_followup_priority_score', 'zone_id' (if zone_filter).
        trend_start_date_input: Start date for trend analysis.
        trend_end_date_input: End date for trend analysis.
        zone_filter: Optional. Filter for a specific zone.
        time_period_aggregation: Aggregation period ('D', 'W-Mon').
        source_context_log_prefix: Prefix for log messages.

    Returns:
        Dict[str, Optional[pd.Series]]: Trends data (index=date, value=metric).
                                        Returns None for a metric if data is insufficient.
    """
    trends_output: Dict[str, Optional[pd.Series]] = {
        "patient_visits_trend": None,
        "high_priority_followups_trend": None
    }

    if not isinstance(chw_historical_health_df, pd.DataFrame) or chw_historical_health_df.empty:
        logger.warning(f"({source_context_log_prefix}) No historical health data provided. Skipping trend calculation.")
        return trends_output
    
    # Validate aggregation period
    valid_periods = ['D', 'W-Mon']
    if time_period_aggregation not in valid_periods:
        logger.error(f"({source_context_log_prefix}) Invalid time_period_aggregation: {time_period_aggregation}. Must be one of {valid_periods}.")
        return trends_output

    # Standardize input dates
    try:
        if isinstance(trend_start_date_input, date_type):
            start_date = trend_start_date_input
        else:
            start_date = pd.to_datetime(trend_start_date_input, errors='coerce').date()
        
        if isinstance(trend_end_date_input, date_type):
            end_date = trend_end_date_input
        else:
            end_date = pd.to_datetime(trend_end_date_input, errors='coerce').date()

        if pd.isna(start_date) or pd.isna(end_date):
            raise ValueError("Invalid date format for trend period after coercion.")
        if start_date > end_date:
            logger.error(f"({source_context_log_prefix}) Trend period error: Start date ({start_date}) is after end date ({end_date}).")
            return trends_output
    except Exception as e:
        logger.error(f"({source_context_log_prefix}) Invalid date inputs: {e}")
        return trends_output
        
    logger.info(
        f"({source_context_log_prefix}) Calculating CHW activity trends: "
        f"{start_date} to {end_date}, Zone: {zone_filter or 'All'}, Period Agg: {time_period_aggregation}"
    )
    
    # Select only necessary columns to optimize memory
    required_columns = ['encounter_date', 'patient_id', 'ai_followup_priority_score']
    if zone_filter:
        required_columns.append('zone_id')
    df_source_copy = chw_historical_health_df[required_columns].copy() if all(col in chw_historical_health_df.columns for col in required_columns) else chw_historical_health_df.copy()

    # Validate and prepare 'encounter_date'
    if 'encounter_date' not in df_source_copy.columns:
        logger.error(f"({source_context_log_prefix}) Critical column 'encounter_date' missing. Cannot calculate trends.")
        return trends_output
        
    df_source_copy['encounter_date'] = pd.to_datetime(df_source_copy['encounter_date'], errors='coerce')
    df_source_copy.dropna(subset=['encounter_date'], inplace=True)
    
    if df_source_copy.empty:
        logger.info(f"({source_context_log_prefix}) No records with valid encounter dates after cleaning.")
        return trends_output

    # Filter by trend date range
    df_period_for_analysis = df_source_copy[
        (df_source_copy['encounter_date'].dt.date >= start_date) &
        (df_source_copy['encounter_date'].dt.date <= end_date)
    ]

    # Apply zone filter
    if zone_filter:
        if 'zone_id' in df_period_for_analysis.columns:
            df_period_for_analysis = df_period_for_analysis[df_period_for_analysis['zone_id'] == zone_filter]
        else:
            logger.warning(f"({source_context_log_prefix}) 'zone_id' column missing, cannot apply zone filter '{zone_filter}'. Using all data.")
            st.warning(f"Zone filter '{zone_filter}' ignored: 'zone_id' column missing.")

    if df_period_for_analysis.empty:
        logger.info(f"({source_context_log_prefix}) No CHW data found for the specified trend period/zone.")
        return trends_output

    # 1. Trend for Patient Visits
    if 'patient_id' in df_period_for_analysis.columns:
        try:
            visits_trend_series = get_trend_data(
                df=df_period_for_analysis,
                value_col='patient_id',
                date_col='encounter_date',
                period=time_period_aggregation,
                agg_func='nunique',
                source_context=f"{source_context_log_prefix}/PatientVisitsTrend"
            )
            if isinstance(visits_trend_series, pd.Series) and not visits_trend_series.empty:
                trends_output["patient_visits_trend"] = visits_trend_series.rename("unique_patient_visits_count")
        except Exception as e:
            logger.error(f"({source_context_log_prefix}) Error in get_trend_data for visits: {e}")
    else:
        logger.warning(f"({source_context_log_prefix}) 'patient_id' column missing, cannot calculate patient visits trend.")

    # 2. Trend for High Priority Follow-ups
    required_cols_for_high_prio = ['ai_followup_priority_score', 'patient_id']
    if all(col in df_period_for_analysis.columns for col in required_cols_for_high_prio):
        if df_period_for_analysis['ai_followup_priority_score'].notna().any():
            df_high_prio_followups = df_period_for_analysis[
                pd.to_numeric(df_period_for_analysis['ai_followup_priority_score'], errors='coerce') >= app_config.FATIGUE_INDEX_HIGH_THRESHOLD
            ]
            
            if not df_high_prio_followups.empty:
                try:
                    high_prio_trend_series = get_trend_data(
                        df=df_high_prio_followups,
                        value_col='patient_id',
                        date_col='encounter_date',
                        period=time_period_aggregation,
                        agg_func='nunique',
                        source_context=f"{source_context_log_prefix}/HighPrioFollowupsTrend"
                    )
                    if isinstance(high_prio_trend_series, pd.Series) and not high_prio_trend_series.empty:
                        trends_output["high_priority_followups_trend"] = high_prio_trend_series.rename("high_priority_followups_count")
                except Exception as e:
                    logger.error(f"({source_context_log_prefix}) Error in get_trend_data for high priority follow-ups: {e}")
            else:
                logger.info(f"({source_context_log_prefix}) No encounters met high AI follow-up priority criteria.")
        else:
            logger.info(f"({source_context_log_prefix}) 'ai_followup_priority_score' has no valid data.")
    else:
        missing_cols_str = ", ".join([col for col in required_cols_for_high_prio if col not in df_period_for_analysis.columns])
        logger.warning(f"({source_context_log_prefix}) Missing columns for high priority follow-ups: [{missing_cols_str}].")

    calculated_trends_count = sum(1 for trend_data_val in trends_output.values() if isinstance(trend_data_val, pd.Series) and not trend_data_val.empty)
    logger.info(f"({source_context_log_prefix}) CHW activity trends calculation complete. {calculated_trends_count} trend(s) generated.")
    return trends_output
