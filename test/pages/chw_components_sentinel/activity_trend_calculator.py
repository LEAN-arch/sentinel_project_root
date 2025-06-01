# sentinel_project_root/test/pages/chw_components_sentinel/activity_trend_calculator.py
# Calculates CHW activity trend data for Sentinel Health Co-Pilot.

import pandas as pd
import numpy as np # Often a companion to pandas
import logging
from typing import Dict, Any, Optional, Union # Union for return type hint
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
    # To import from test/config and test/utils, we need 'test/' on the path.
    # pages -> chw_components_sentinel -> (up to pages) -> (up to test)
    project_test_root_dir = os.path.abspath(os.path.join(current_script_dir, os.pardir, os.pardir))
    if project_test_root_dir not in sys.path:
        sys.path.insert(0, project_test_root_dir)
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
                                        Returns None for a metric if data is insufficient.
    """
    trends_output: Dict[str, Optional[pd.Series]] = {
        "patient_visits_trend": None,
        "high_priority_followups_trend": None
        # Future trends could include: "avg_risk_score_of_visited_patients_trend"
    }

    if not isinstance(chw_historical_health_df, pd.DataFrame) or chw_historical_health_df.empty:
        logger.warning(f"({source_context_log_prefix}) No historical health data provided. Skipping trend calculation.")
        return trends_output
    
    try:
        # Standardize input dates to datetime.date objects for comparison
        start_date = pd.to_datetime(trend_start_date_input, errors='coerce').date()
        end_date = pd.to_datetime(trend_end_date_input, errors='coerce').date()
        if pd.isna(start_date) or pd.isna(end_date): # Check if coercion failed
            raise ValueError("Invalid date format for trend period after coercion to date object.")
        if start_date > end_date:
            logger.error(f"({source_context_log_prefix}) Trend period error: Start date ({start_date.isoformat()}) is after end date ({end_date.isoformat()}).")
            return trends_output
    except Exception as e: # Catches errors from to_datetime or attribute access on NaT
        logger.error(f"({source_context_log_prefix}) Invalid date inputs for trend period: {e}")
        return trends_output
        
    logger.info(
        f"({source_context_log_prefix}) Calculating CHW activity trends: "
        f"{start_date.isoformat()} to {end_date.isoformat()}, "
        f"Zone: {zone_filter or 'All'}, Period Agg: {time_period_aggregation}"
    )
    
    df_source_copy = chw_historical_health_df.copy() # Work on a copy

    # Validate and prepare 'encounter_date'
    if 'encounter_date' not in df_source_copy.columns:
        logger.error(f"({source_context_log_prefix}) Critical column 'encounter_date' missing from input DataFrame. Cannot calculate trends.")
        return trends_output
        
    df_source_copy['encounter_date'] = pd.to_datetime(df_source_copy['encounter_date'], errors='coerce')
    df_source_copy.dropna(subset=['encounter_date'], inplace=True) # Remove rows where date conversion failed
    
    if df_source_copy.empty:
        logger.info(f"({source_context_log_prefix}) No records with valid encounter dates after cleaning for trend analysis.")
        return trends_output

    # Filter by the overall trend date range using datetime.date objects
    df_period_for_analysis = df_source_copy[
        (df_source_copy['encounter_date'].dt.date >= start_date) &
        (df_source_copy['encounter_date'].dt.date <= end_date)
    ]

    # Apply zone filter if specified
    if zone_filter:
        if 'zone_id' in df_period_for_analysis.columns:
            df_period_for_analysis = df_period_for_analysis[df_period_for_analysis['zone_id'] == zone_filter]
        else:
            logger.warning(f"({source_context_log_prefix}) 'zone_id' column missing, cannot apply zone filter '{zone_filter}'. Using all data for the period.")
            # df_period_for_analysis remains as is (already period-filtered)

    if df_period_for_analysis.empty:
        logger.info(f"({source_context_log_prefix}) No CHW data found for the specified trend period/zone combination.")
        return trends_output

    # 1. Trend for Patient Visits (Unique Patients per period)
    if 'patient_id' in df_period_for_analysis.columns:
        visits_trend_series = get_trend_data(
            df=df_period_for_analysis,
            value_col='patient_id',
            date_col='encounter_date', # Already ensured this is datetime
            period=time_period_aggregation,
            agg_func='nunique', # Counts unique patients per aggregation period
            source_context=f"{source_context_log_prefix}/PatientVisitsTrend"
        )
        if isinstance(visits_trend_series, pd.Series) and not visits_trend_series.empty:
            trends_output["patient_visits_trend"] = visits_trend_series.rename("unique_patient_visits_count")
    else:
        logger.warning(f"({source_context_log_prefix}) 'patient_id' column missing, cannot calculate patient visits trend.")


    # 2. Trend for High Priority Follow-ups (Unique Patients with high AI Follow-up Score)
    required_cols_for_high_prio = ['ai_followup_priority_score', 'patient_id']
    if all(col in df_period_for_analysis.columns for col in required_cols_for_high_prio):
        if df_period_for_analysis['ai_followup_priority_score'].notna().any(): # Check if there's any non-NaN data to process
            # Filter encounters that are high priority
            df_high_prio_followups = df_period_for_analysis[
                # Ensure score is numeric before comparison
                pd.to_numeric(df_period_for_analysis['ai_followup_priority_score'], errors='coerce') >= app_config.FATIGUE_INDEX_HIGH_THRESHOLD 
            ]
            
            if not df_high_prio_followups.empty:
                high_prio_trend_series = get_trend_data(
                    df=df_high_prio_followups,
                    value_col='patient_id', # Count unique patients flagged for high priority follow-up
                    date_col='encounter_date',
                    period=time_period_aggregation,
                    agg_func='nunique',
                    source_context=f"{source_context_log_prefix}/HighPrioFollowupsTrend"
                )
                if isinstance(high_prio_trend_series, pd.Series) and not high_prio_trend_series.empty:
                    trends_output["high_priority_followups_trend"] = high_prio_trend_series.rename("high_priority_followups_count")
            else:
                logger.info(f"({source_context_log_prefix}) No encounters met high AI follow-up priority criteria for trend calculation.")
        else:
            logger.info(f"({source_context_log_prefix}) 'ai_followup_priority_score' column has no valid (non-NaN) data for high priority trend.")
    else:
        missing_cols_str = ", ".join([col for col in required_cols_for_high_prio if col not in df_period_for_analysis.columns])
        logger.warning(f"({source_context_log_prefix}) Missing columns for high priority follow-ups trend: [{missing_cols_str}].")

    calculated_trends_count = sum(1 for trend_data_val in trends_output.values() if isinstance(trend_data_val, pd.Series) and not trend_data_val.empty)
    logger.info(f"({source_context_log_prefix}) CHW activity trends calculation complete. {calculated_trends_count} trend(s) generated.")
    return trends_output
