# sentinel_project_root/test/pages/chw_components_sentinel/activity_trend_calculator.py
# Calculates CHW activity trend data for Sentinel Health Co-Pilot.

import pandas as pd
import numpy as np # Often a companion to pandas, though not heavily used here
import logging
from typing import Dict, Any, Optional, Union, cast # Added cast for type checking if needed
from datetime import date as date_type, datetime # For type hinting and conversion

# Standardized import block
try:
    from config import app_config
    from utils.core_data_processing import get_trend_data
except ImportError:
    import sys
    import os
    # Assumes this file is in sentinel_project_root/test/pages/chw_components_sentinel/
    # Navigate three levels up to sentinel_project_root/test/
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_for_utils = os.path.abspath(os.path.join(current_script_dir, os.pardir, os.pardir))
    if project_root_for_utils not in sys.path:
        sys.path.insert(0, project_root_for_utils)
    from config import app_config
    from utils.core_data_processing import get_trend_data

logger = logging.getLogger(__name__)

def calculate_chw_activity_trends(
    chw_historical_health_df: Optional[pd.DataFrame],
    trend_start_date_input: Any, # Can be date, datetime, or string
    trend_end_date_input: Any,   # Can be date, datetime, or string
    zone_filter: Optional[str] = None,
    time_period_aggregation: str = 'D', # 'D' for daily, 'W-Mon' for weekly
    source_context_log_prefix: str = "CHWActivityTrends"
) -> Dict[str, Optional[pd.Series]]: # Return type hint corrected to pd.Series
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
    """
    trends_output: Dict[str, Optional[pd.Series]] = {
        "patient_visits_trend": None,
        "high_priority_followups_trend": None
    }

    if not isinstance(chw_historical_health_df, pd.DataFrame) or chw_historical_health_df.empty:
        logger.warning(f"({source_context_log_prefix}) No historical health data provided. Skipping trend calculation.")
        return trends_output
    
    try:
        # Standardize input dates to datetime.date objects
        start_date = pd.to_datetime(trend_start_date_input, errors='coerce').date()
        end_date = pd.to_datetime(trend_end_date_input, errors='coerce').date()
        if pd.isna(start_date) or pd.isna(end_date):
            raise ValueError("Invalid date format for trend period after coercion.")
        if start_date > end_date:
            logger.error(f"({source_context_log_prefix}) Trend period error: Start date ({start_date}) is after end date ({end_date}).")
            return trends_output
    except Exception as e:
        logger.error(f"({source_context_log_prefix}) Invalid date inputs for trend period: {e}")
        return trends_output
        
    logger.info(
        f"({source_context_log_prefix}) Calculating CHW activity trends: "
        f"{start_date.isoformat()} to {end_date.isoformat()}, "
        f"Zone: {zone_filter or 'All'}, Period Agg: {time_period_aggregation}"
    )
    
    df_source = chw_historical_health_df.copy()

    # Validate and prepare 'encounter_date'
    if 'encounter_date' not in df_source.columns:
        logger.error(f"({source_context_log_prefix}) Critical column 'encounter_date' missing. Cannot calculate trends.")
        return trends_output
    df_source['encounter_date'] = pd.to_datetime(df_source['encounter_date'], errors='coerce')
    df_source.dropna(subset=['encounter_date'], inplace=True)
    if df_source.empty:
        logger.info(f"({source_context_log_prefix}) No records with valid encounter dates after cleaning.")
        return trends_output

    # Filter by the overall trend date range
    df_period_base = df_source[
        (df_source['encounter_date'].dt.date >= start_date) &
        (df_source['encounter_date'].dt.date <= end_date)
    ]

    # Apply zone filter if specified
    if zone_filter:
        if 'zone_id' in df_period_base.columns:
            df_period_filtered = df_period_base[df_period_base['zone_id'] == zone_filter]
        else:
            logger.warning(f"({source_context_log_prefix}) 'zone_id' column missing, cannot apply zone filter '{zone_filter}'. Using all data for period.")
            df_period_filtered = df_period_base
    else:
        df_period_filtered = df_period_base

    if df_period_filtered.empty:
        logger.info(f"({source_context_log_prefix}) No CHW data found for the specified trend period/zone combination.")
        return trends_output

    # 1. Trend for Patient Visits (Unique Patients per period)
    if 'patient_id' in df_period_filtered.columns:
        visits_trend = get_trend_data(
            df=df_period_filtered,
            value_col='patient_id',
            date_col='encounter_date', # Already ensured this is datetime
            period=time_period_aggregation,
            agg_func='nunique',
            source_context=f"{source_context_log_prefix}/PatientVisits"
        )
        if isinstance(visits_trend, pd.Series) and not visits_trend.empty:
            trends_output["patient_visits_trend"] = visits_trend.rename("unique_patient_visits_count")
    else:
        logger.warning(f"({source_context_log_prefix}) 'patient_id' column missing, cannot calculate patient visits trend.")

    # 2. Trend for High Priority Follow-ups
    required_cols_for_prio_trend = ['ai_followup_priority_score', 'patient_id']
    if all(col in df_period_filtered.columns for col in required_cols_for_prio_trend):
        if df_period_filtered['ai_followup_priority_score'].notna().any():
            high_prio_encounters_df = df_period_filtered[
                df_period_filtered['ai_followup_priority_score'] >= app_config.FATIGUE_INDEX_HIGH_THRESHOLD # Using this as generic high prio
            ]
            if not high_prio_encounters_df.empty:
                high_prio_trend = get_trend_data(
                    df=high_prio_encounters_df,
                    value_col='patient_id', # Count unique patients flagged
                    date_col='encounter_date',
                    period=time_period_aggregation,
                    agg_func='nunique',
                    source_context=f"{source_context_log_prefix}/HighPrioFollowups"
                )
                if isinstance(high_prio_trend, pd.Series) and not high_prio_trend.empty:
                    trends_output["high_priority_followups_trend"] = high_prio_trend.rename("high_priority_followups_count")
            else:
                logger.info(f"({source_context_log_prefix}) No encounters met high AI follow-up priority criteria for trend calculation.")
        else:
            logger.info(f"({source_context_log_prefix}) 'ai_followup_priority_score' column has no valid data for high priority trend.")
    else:
        missing_prio_cols = [col for col in required_cols_for_prio_trend if col not in df_period_filtered.columns]
        logger.warning(f"({source_context_log_prefix}) Missing columns for high priority follow-ups trend: {missing_prio_cols}.")

    calculated_trends = sum(1 for data in trends_output.values() if data is not None)
    logger.info(f"({source_context_log_prefix}) CHW activity trends calculation complete. {calculated_trends} trend(s) generated.")
    return trends_output
