# sentinel_project_root/test/pages/chw_components_sentinel/activity_trend_calculator.py
# Part of "Sentinel Health Co-Pilot" - LMIC Edge-First System
# This module calculates CHW activity trend data over a specified period.
# Refactored from the original chw_components/trends_display.py.
# The output is structured data for supervisor reports or Tier 1/2 dashboards.

import pandas as pd
import numpy as np # Not directly used but often a companion to pandas
import logging
from typing import Dict, Any, Optional, Union # Union for DataFrame or Series

# Assuming app_config and core_data_processing.get_trend_data are accessible
try:
    from config import app_config
    from utils.core_data_processing import get_trend_data
except ImportError:
    import sys, os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_test_root = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
    if project_test_root not in sys.path: sys.path.insert(0, project_test_root)
    from config import app_config
    from utils.core_data_processing import get_trend_data


logger = logging.getLogger(__name__)

def calculate_chw_activity_trends(
    chw_historical_health_df: Optional[pd.DataFrame], # Historical data for a CHW or CHW team
    trend_start_date: Any, # datetime.date object or compatible string
    trend_end_date: Any,   # datetime.date object or compatible string
    zone_filter: Optional[str] = None, # e.g., "ZoneA" or None for all zones in df_chw_hist
    time_period_aggregation: str = 'D', # 'D' for daily, 'W-Mon' for weekly starting Monday
    source_context_log_prefix: str = "CHWActivityTrends" # For logging context
) -> Dict[str, Optional[Union[pd.DataFrame, pd.Series]]]:
    """
    Calculates CHW activity trends (e.g., visits, high-priority tasks) over a specified period.

    Args:
        chw_historical_health_df: DataFrame containing historical health records relevant
                                  to the CHW or team being analyzed. Expected columns include:
                                  'encounter_date', 'patient_id', 'ai_followup_priority_score',
                                  'zone_id' (if zone_filter is used).
        trend_start_date: The start date for the trend analysis.
        trend_end_date: The end date for the trend analysis.
        zone_filter: Optional. If provided, filters data for a specific zone before calculating trends.
        time_period_aggregation: Aggregation period for trends (e.g., 'D' for daily, 'W-Mon' for weekly).
        source_context_log_prefix: Prefix for log messages from this function call.

    Returns:
        Dict[str, Optional[Union[pd.DataFrame, pd.Series]]]:
            A dictionary where keys are trend metric names (e.g., "patient_visits_trend",
            "high_priority_followups_trend") and values are Pandas DataFrames or Series
            containing the trend data (index=date, value=metric).
            Returns None for a metric if data is insufficient or an error occurs.
    """
    logger.info(
        f"({source_context_log_prefix}) Calculating CHW activity trends: "
        f"{str(trend_start_date)} to {str(trend_end_date)}, "
        f"Zone: {zone_filter or 'All'}, Period Agg: {time_period_aggregation}"
    )

    trends_data_output: Dict[str, Optional[Union[pd.DataFrame, pd.Series]]] = {
        "patient_visits_trend": None,           # Series: Index=Date, Values=Count of unique patients
        "high_priority_followups_trend": None   # Series: Index=Date, Values=Count of unique high-prio patients/tasks
        # Future trends could include: "avg_risk_score_of_visited_patients_trend"
    }

    if chw_historical_health_df is None or chw_historical_health_df.empty:
        logger.warning(f"({source_context_log_prefix}) No historical health data provided for trend calculation.")
        return trends_data_output
    
    try:
        start_dt = pd.to_datetime(trend_start_date).date()
        end_dt = pd.to_datetime(trend_end_date).date()
        if start_dt > end_dt:
            logger.error(f"({source_context_log_prefix}) Trend period error: Start date ({start_dt}) is after end date ({end_dt}).")
            return trends_data_output
    except Exception as e_date:
        logger.error(f"({source_context_log_prefix}) Invalid date format for trend period: {e_date}")
        return trends_data_output
        
    df_trends_src = chw_historical_health_df.copy() # Work on a copy

    # Ensure 'encounter_date' is a datetime type for filtering and get_trend_data utility
    if 'encounter_date' not in df_trends_src.columns:
        logger.error(f"({source_context_log_prefix}) Missing 'encounter_date' column, essential for trend analysis.")
        return trends_data_output
    
    df_trends_src['encounter_date'] = pd.to_datetime(df_trends_src['encounter_date'], errors='coerce')
    df_trends_src.dropna(subset=['encounter_date'], inplace=True) # Remove rows where date conversion failed
    
    if df_trends_src.empty:
        logger.info(f"({source_context_log_prefix}) No valid encounter dates after cleaning for trend analysis.")
        return trends_data_output

    # Filter by the selected trend date range
    date_mask_for_trends = (df_trends_src['encounter_date'].dt.date >= start_dt) & \
                           (df_trends_src['encounter_date'].dt.date <= end_dt)
    df_period_for_trends = df_trends_src[date_mask_for_trends]

    # Apply zone filter if specified
    if zone_filter and 'zone_id' in df_period_for_trends.columns:
        df_period_for_trends = df_period_for_trends[df_period_for_trends['zone_id'] == zone_filter]

    if df_period_for_trends.empty:
        logger.info(f"({source_context_log_prefix}) No CHW data found for the specified trend period/zone combination.")
        return trends_data_output

    # 1. Trend for Patient Visits (Unique Patients per period)
    if 'patient_id' in df_period_for_trends.columns:
        visits_trend_series_data = get_trend_data(
            df=df_period_for_trends,
            value_col='patient_id',
            date_col='encounter_date', # Already ensured this is datetime
            period=time_period_aggregation,
            agg_func='nunique', # Counts unique patients per aggregation period
            source_context=f"{source_context_log_prefix}/VisitsTrend"
        )
        if visits_trend_series_data is not None and not visits_trend_series_data.empty:
            trends_data_output["patient_visits_trend"] = visits_trend_series_data.rename("unique_patient_visits_count")
    else:
        logger.warning(f"({source_context_log_prefix}) 'patient_id' column missing, cannot calculate visits trend.")


    # 2. Trend for High Priority Follow-ups (Unique Patients with high AI Follow-up Score)
    # Assumes 'ai_followup_priority_score' is present from PED sync / upstream AI processing.
    if 'ai_followup_priority_score' in df_period_for_trends.columns and \
       'patient_id' in df_period_for_trends.columns and \
       df_period_for_trends['ai_followup_priority_score'].notna().any():
        
        # Filter encounters that are high priority
        high_priority_followups_df = df_period_for_trends[
            df_period_for_trends['ai_followup_priority_score'] >= app_config.FATIGUE_INDEX_HIGH_THRESHOLD # Sentinel config
        ]
        
        if not high_priority_followups_df.empty:
            high_prio_trend_series_data = get_trend_data(
                df=high_priority_followups_df,
                value_col='patient_id', # Count unique patients flagged for high priority follow-up
                date_col='encounter_date',
                period=time_period_aggregation,
                agg_func='nunique',
                source_context=f"{source_context_log_prefix}/HighPrioTrend"
            )
            if high_prio_trend_series_data is not None and not high_prio_trend_series_data.empty:
                trends_data_output["high_priority_followups_trend"] = high_prio_trend_series_data.rename("high_priority_followups_count")
        else:
            logger.info(f"({source_context_log_prefix}) No encounters met high AI follow-up priority criteria for trend calculation.")
    else:
        logger.info(f"({source_context_log_prefix}) 'ai_followup_priority_score' or 'patient_id' missing or all NaN, cannot calculate high priority follow-ups trend.")

    calculated_trends_count = sum(1 for trend_data in trends_data_output.values() if trend_data is not None and not trend_data.empty)
    logger.info(f"({source_context_log_prefix}) CHW activity trends calculation complete. {calculated_trends_count} trend(s) generated.")
    return trends_data_output
