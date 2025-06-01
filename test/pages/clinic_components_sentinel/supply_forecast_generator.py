# sentinel_project_root/test/pages/clinic_components_sentinel/supply_forecast_generator.py
# Part of "Sentinel Health Co-Pilot" - LMIC Edge-First System
# This module calculates and prepares supply forecast data for medical items at a clinic.
# It can orchestrate calls to either a simple linear forecast model or an AI-simulated model.
# Refactored from the original clinic_components/supply_chain_tab.py.
# Output is structured data for display on the Clinic Management Console (Tier 2).

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List

# Assuming app_config and core utilities are accessible
try:
    from config import app_config
    from utils.core_data_processing import get_supply_forecast_data # Simple linear model
    from utils.ai_analytics_engine import SupplyForecastingModel    # AI-simulated model
except ImportError:
    import sys, os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_test_root = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
    if project_test_root not in sys.path: sys.path.insert(0, project_test_root)
    from config import app_config
    from utils.core_data_processing import get_supply_forecast_data
    from utils.ai_analytics_engine import SupplyForecastingModel

logger = logging.getLogger(__name__)

def prepare_clinic_supply_forecast_data(
    clinic_historical_health_df: Optional[pd.DataFrame], # Full historical data for consumption rates and latest stock status
    reporting_period_str: str, # Contextual string
    forecast_days_out: int = 30,
    use_ai_forecast_model: bool = False,
    items_to_forecast: Optional[List[str]] = None # Specific items to forecast, else defaults to key drugs
) -> Dict[str, Any]:
    """
    Prepares supply forecast data using either a simple linear or an AI-simulated model.

    Args:
        clinic_historical_health_df: DataFrame of health records containing item usage,
                                     stock levels, and consumption rates. Required cols:
                                     'item', 'encounter_date', 'item_stock_agg_zone', 'consumption_rate_per_day'.
        reporting_period_str: String describing the current reporting context.
        forecast_days_out: Number of days into the future to forecast.
        use_ai_forecast_model: Boolean flag to select the AI simulation model.
        items_to_forecast: Optional list of specific item names. If None, forecasts for
                           items matching KEY_DRUG_SUBSTRINGS_SUPPLY from app_config.

    Returns:
        Dict[str, Any]: Contains the forecast model used, the detailed forecast DataFrame,
                        a summary list per item, and processing notes.
    """
    module_source_context = "ClinicSupplyForecastGenerator"
    model_type_str = "AI-Simulated" if use_ai_forecast_model else "Simple Linear"
    logger.info(f"({module_source_context}) Preparing clinic supply forecast. Model: {model_type_str}, Horizon: {forecast_days_out} days.")

    forecast_output_data: Dict[str, Any] = {
        "reporting_context": reporting_period_str, # Clarified key name
        "forecast_model_type_used": model_type_str,
        "forecast_detail_df": None,      # Main DataFrame with daily/periodic forecast per item
        "forecast_items_overview_list": [], # List of dicts, one per item (current_stock, est_stockout)
        "data_processing_notes": []      # For any issues or contextual info
    }

    required_hist_cols = ['item', 'encounter_date', 'item_stock_agg_zone', 'consumption_rate_per_day']
    if clinic_historical_health_df is None or clinic_historical_health_df.empty or \
       not all(col in clinic_historical_health_df.columns for col in required_hist_cols):
        msg = f"Historical health data is missing required columns for supply forecasts. Needed: {required_hist_cols}"
        logger.error(f"({module_source_context}) {msg}")
        forecast_output_data["data_processing_notes"].append(msg)
        return forecast_output_data

    df_supply_hist_source = clinic_historical_health_df.copy()
    # Ensure encounter_date is datetime and other key columns have safe defaults/types
    df_supply_hist_source['encounter_date'] = pd.to_datetime(df_supply_hist_source['encounter_date'], errors='coerce')
    df_supply_hist_source.dropna(subset=['encounter_date', 'item'], inplace=True) # Item and date are critical
    df_supply_hist_source['item_stock_agg_zone'] = pd.to_numeric(df_supply_hist_source.get('item_stock_agg_zone'), errors='coerce').fillna(0)
    df_supply_hist_source['consumption_rate_per_day'] = pd.to_numeric(df_supply_hist_source.get('consumption_rate_per_day'), errors='coerce').fillna(0.0001) # Avoid zero for rate


    # Determine the list of items to forecast
    final_items_to_forecast_list: List[str]
    if items_to_forecast:
        final_items_to_forecast_list = items_to_forecast
    elif app_config.KEY_DRUG_SUBSTRINGS_SUPPLY:
        unique_items_in_data = df_supply_hist_source['item'].dropna().unique()
        final_items_to_forecast_list = [
            item_name for item_name in unique_items_in_data
            if any(drug_sub.lower() in str(item_name).lower() for drug_sub in app_config.KEY_DRUG_SUBSTRINGS_SUPPLY)
        ]
        if not final_items_to_forecast_list and len(unique_items_in_data) > 0:
            final_items_to_forecast_list = unique_items_in_data[:min(5, len(unique_items_in_data))].tolist() # Fallback
            forecast_output_data["data_processing_notes"].append(
                "No key drug substrings matched items in data; forecasting for first few available items."
            )
        elif not final_items_to_forecast_list: # Still no items
             forecast_output_data["data_processing_notes"].append("No items found in historical data to forecast.")
             return forecast_output_data
    else: # No specific list, no key substrings in config, fallback to a few unique items
        unique_items_in_data = df_supply_hist_source['item'].dropna().unique()
        final_items_to_forecast_list = unique_items_in_data[:min(5, len(unique_items_in_data))].tolist()
        if not final_items_to_forecast_list:
            forecast_output_data["data_processing_notes"].append("No items found in historical data to forecast.")
            return forecast_output_data

    if not final_items_to_forecast_list: # Final check if list is empty
        forecast_output_data["data_processing_notes"].append("No items determined for forecasting.")
        return forecast_output_data


    # --- Generate Forecast based on Selected Model ---
    generated_forecast_df: Optional[pd.DataFrame] = None

    if use_ai_forecast_model:
        logger.info(f"({module_source_context}) Using AI-Simulated Supply Forecasting Model for: {final_items_to_forecast_list}")
        ai_supply_forecaster = SupplyForecastingModel()
        
        # Prepare the input DataFrame for the AI model: item, current_stock, avg_daily_consumption_historical, last_stock_update_date
        latest_status_per_item_df = df_supply_hist_source.sort_values('encounter_date').drop_duplicates(subset=['item'], keep='last')
        input_for_ai_model_df = latest_status_per_item_df.rename(columns={
            'item_stock_agg_zone': 'current_stock',
            'consumption_rate_per_day': 'avg_daily_consumption_historical',
            'encounter_date': 'last_stock_update_date'
        })[['item', 'current_stock', 'avg_daily_consumption_historical', 'last_stock_update_date']]
        
        # Filter this prepared df for the items specifically selected for forecasting
        input_for_ai_model_df_filtered = input_for_ai_model_df[input_for_ai_model_df['item'].isin(final_items_to_forecast_list)]

        if not input_for_ai_model_df_filtered.empty:
            generated_forecast_df = ai_supply_forecaster.forecast_supply_levels_advanced(
                current_supply_levels_df=input_for_ai_model_df_filtered,
                forecast_days_out=forecast_days_out
                # item_filter_list is already applied to input_for_ai_model_df_filtered
            )
            if generated_forecast_df is not None and not generated_forecast_df.empty:
                generated_forecast_df.rename(columns={'forecast_date':'date', 'estimated_stockout_date_ai':'estimated_stockout_date'}, inplace=True, errors='ignore')
        else:
            forecast_output_data["data_processing_notes"].append(f"No current status data found for AI forecasting the selected items: {final_items_to_forecast_list}")
    
    else: # Use simple linear forecast
        logger.info(f"({module_source_context}) Using Simple Linear Supply Forecasting for: {final_items_to_forecast_list}")
        generated_forecast_df = get_supply_forecast_data(
            health_df=df_supply_hist_source, # Uses the full historical DF it knows how to process
            forecast_days_out=forecast_days_out,
            item_filter_list=final_items_to_forecast_list, # Pass specific items to filter within the function
            source_context=f"{module_source_context}/LinearForecast"
        )
        if generated_forecast_df is not None and not generated_forecast_df.empty:
            # Align column names for consistency in output, if different
            rename_map_linear = {
                'estimated_stockout_date_linear':'estimated_stockout_date',
                'current_stock_at_forecast_start': 'initial_stock_at_forecast_start', # For summary
                'base_consumption_rate_per_day': 'initial_consumption_rate_per_day'   # For summary
            }
            generated_forecast_df.rename(columns=rename_map_linear, inplace=True, errors='ignore')


    if generated_forecast_df is not None and not generated_forecast_df.empty:
        forecast_output_data["forecast_detail_df"] = generated_forecast_df.sort_values(by=['item', 'date']).reset_index(drop=True)
        
        # Create the summary list
        item_overview_list = []
        # Group by item and take first row after sorting (or use drop_duplicates on item for latest stockout date if already sorted)
        summary_src_df = generated_forecast_df.sort_values(['item', 'date']).drop_duplicates(subset=['item'], keep='first')

        for _, item_summary_row in summary_src_df.iterrows():
            current_stock_val = item_summary_row.get('initial_stock_at_forecast_start', item_summary_row.get('current_stock', 0)) # Handle different source columns
            base_cons_val = item_summary_row.get('initial_consumption_rate_per_day', item_summary_row.get('consumption_rate', 0.0001))
            if pd.isna(base_cons_val) or base_cons_val <= 0 : base_cons_val = 0.0001 # Avoid div by zero

            initial_dos_val = item_summary_row.get('initial_days_supply_at_forecast_start', current_stock_val / base_cons_val)
            est_stockout_dt_val = pd.to_datetime(item_summary_row.get('estimated_stockout_date', pd.NaT), errors='coerce')

            item_overview_list.append({
                "item_name": item_summary_row['item'],
                "current_stock_on_hand": current_stock_val,
                "avg_daily_consumption_used": base_cons_val,
                "initial_days_of_supply_est": round(initial_dos_val,1) if np.isfinite(initial_dos_val) else "Adequate (>Forecast)",
                "estimated_stockout_date": est_stockout_dt_val.strftime('%Y-%m-%d') if pd.notna(est_stockout_dt_val) else "Beyond Forecast"
            })
        forecast_output_data["forecast_items_overview_list"] = item_overview_list
        
    elif not forecast_output_data["data_processing_notes"]: # If forecast DF is empty and no specific notes added yet
        forecast_output_data["data_processing_notes"].append("Supply forecast could not be generated with the selected model and data.")

    logger.info(f"({module_source_context}) Clinic supply forecast data preparation complete. Items forecasted: {len(forecast_output_data.get('forecast_items_overview_list',[]))}")
    return forecast_output_data
