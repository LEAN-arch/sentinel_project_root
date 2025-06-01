# sentinel_project_root/test/pages/clinic_components_sentinel/supply_forecast_generator.py
# Prepares supply forecast data for medical items at a clinic for Sentinel.

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List

# Standardized import block
try:
    from config import app_config
    from utils.core_data_processing import get_supply_forecast_data # Simple linear model
    from utils.ai_analytics_engine import SupplyForecastingModel    # AI-simulated model
except ImportError:
    import sys
    import os
    # Assumes this file is in sentinel_project_root/test/pages/clinic_components_sentinel/
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_test_root_dir = os.path.abspath(os.path.join(current_script_dir, os.pardir, os.pardir))
    if project_test_root_dir not in sys.path:
        sys.path.insert(0, project_test_root_dir)
    from config import app_config
    from utils.core_data_processing import get_supply_forecast_data # For simple linear forecast
    from utils.ai_analytics_engine import SupplyForecastingModel    # For AI-simulated forecast

logger = logging.getLogger(__name__)

def prepare_clinic_supply_forecast_data(
    clinic_historical_health_df: Optional[pd.DataFrame], # Full historical data needed for rates and latest stock
    reporting_period_str: str, # Contextual string for the reporting period (not directly used in calc)
    forecast_days_out: int = 30, # Default forecast horizon
    use_ai_forecast_model: bool = False, # Flag to switch between models
    items_to_forecast: Optional[List[str]] = None # Specific items to forecast, else defaults to key drugs
) -> Dict[str, Any]:
    """
    Prepares supply forecast data using either a simple linear or an AI-simulated model.

    Args:
        clinic_historical_health_df: DataFrame of health records containing item usage,
                                     stock levels ('item_stock_agg_zone'), and 
                                     consumption rates ('consumption_rate_per_day').
        reporting_period_str: String describing the current reporting context.
        forecast_days_out: Number of days into the future to forecast.
        use_ai_forecast_model: Boolean flag to select the AI simulation model.
        items_to_forecast: Optional list of specific item names. If None, forecasts for
                           items matching KEY_DRUG_SUBSTRINGS_SUPPLY from app_config.

    Returns:
        Dict[str, Any]: Contains the forecast model type used, the detailed forecast DataFrame,
                        a summary list per item (overview), and processing notes.
    """
    module_log_prefix = "ClinicSupplyForecastGen" # Consistent prefix
    model_type_desc = "AI-Simulated" if use_ai_forecast_model else "Simple Linear"
    logger.info(f"({module_log_prefix}) Preparing clinic supply forecast. Model: {model_type_desc}, Horizon: {forecast_days_out} days, Items: {'Specific List Provided' if items_to_forecast else 'Auto-detect from Config/Data'}.")

    # Initialize output structure
    forecast_output_dict: Dict[str, Any] = {
        "reporting_context": reporting_period_str,
        "forecast_model_type_used": model_type_desc,
        "forecast_detail_df": None,      # Main DataFrame with daily/periodic forecast per item
        "forecast_items_overview_list": [], # List of dicts, one per item (current_stock, est_stockout, etc.)
        "data_processing_notes": []      # For any issues or contextual information during processing
    }

    # Validate input DataFrame
    required_historical_cols = ['item', 'encounter_date', 'item_stock_agg_zone', 'consumption_rate_per_day']
    if not isinstance(clinic_historical_health_df, pd.DataFrame) or \
       clinic_historical_health_df.empty or \
       not all(col in clinic_historical_health_df.columns for col in required_historical_cols):
        
        missing_cols_str = [col for col in required_historical_cols if col not in (clinic_historical_health_df.columns if isinstance(clinic_historical_health_df, pd.DataFrame) else [])]
        error_message = f"Historical health data is insufficient for supply forecasts. Required columns: {required_historical_cols}. Missing or DataFrame empty. Missing cols identified: {missing_cols_str if missing_cols_str else 'None, but DF may be empty'}."
        logger.error(f"({module_log_prefix}) {error_message}")
        forecast_output_dict["data_processing_notes"].append(error_message)
        return forecast_output_dict

    df_supply_hist = clinic_historical_health_df.copy() # Work on a copy
    # Data Cleaning and Preparation
    df_supply_hist['encounter_date'] = pd.to_datetime(df_supply_hist['encounter_date'], errors='coerce')
    df_supply_hist.dropna(subset=['encounter_date', 'item'], inplace=True) # Item name and date are critical
    
    df_supply_hist['item_stock_agg_zone'] = pd.to_numeric(df_supply_hist.get('item_stock_agg_zone'), errors='coerce').fillna(0.0)
    # Ensure consumption rate is a small positive number to avoid division by zero if it's used as a divisor.
    df_supply_hist['consumption_rate_per_day'] = pd.to_numeric(df_supply_hist.get('consumption_rate_per_day'), errors='coerce').fillna(1e-6) # Small default
    df_supply_hist.loc[df_supply_hist['consumption_rate_per_day'] <= 0, 'consumption_rate_per_day'] = 1e-6 # Replace 0 or negative with small positive

    if df_supply_hist.empty: # Check after cleaning dates
        msg = "No valid historical records remaining after cleaning for supply forecast."
        logger.warning(f"({module_log_prefix}) {msg}")
        forecast_output_dict["data_processing_notes"].append(msg); return forecast_output_dict

    # Determine the list of items to forecast
    final_item_list_to_forecast: List[str]
    if items_to_forecast and isinstance(items_to_forecast, list) and len(items_to_forecast) > 0:
        final_item_list_to_forecast = list(set(items_to_forecast)) # Ensure unique
        logger.debug(f"({module_log_prefix}) Using provided specific item list: {final_item_list_to_forecast}")
    elif app_config.KEY_DRUG_SUBSTRINGS_SUPPLY:
        all_unique_items_in_data = df_supply_hist['item'].dropna().unique()
        final_item_list_to_forecast = [
            item_name for item_name in all_unique_items_in_data
            if any(drug_substr.lower() in str(item_name).lower() for drug_substr in app_config.KEY_DRUG_SUBSTRINGS_SUPPLY)
        ]
        if not final_item_list_to_forecast and len(all_unique_items_in_data) > 0:
            # Fallback: if no key drugs match, take a sample of available items
            num_fallback_items = min(5, len(all_unique_items_in_data))
            final_item_list_to_forecast = np.random.choice(all_unique_items_in_data, num_fallback_items, replace=False).tolist() if len(all_unique_items_in_data) > num_fallback_items else all_unique_items_in_data.tolist()
            note = f"No key drug substrings matched items in data; forecasting for {len(final_item_list_to_forecast)} sample available items: {final_item_list_to_forecast}"
            logger.info(f"({module_log_prefix}) {note}")
            forecast_output_dict["data_processing_notes"].append(note)
        elif not final_item_list_to_forecast: # Still no items (e.g. unique_items_in_data_source was empty)
             msg = "No items found in historical data matching key drug substrings to forecast (and no fallback items available)."
             logger.warning(f"({module_log_prefix}) {msg}")
             forecast_output_dict["data_processing_notes"].append(msg); return forecast_output_dict
    else: # No specific list, no key substrings in config, fallback to a sample of unique items
        all_unique_items_in_data = df_supply_hist['item'].dropna().unique()
        num_fallback_items = min(5, len(all_unique_items_in_data))
        final_item_list_to_forecast = np.random.choice(all_unique_items_in_data, num_fallback_items, replace=False).tolist() if len(all_unique_items_in_data) > num_fallback_items else all_unique_items_in_data.tolist()
        if not final_item_list_to_forecast:
            msg = "No items available in historical data to forecast (no specific list provided, no key drugs in config, and no unique items found in data)."
            logger.warning(f"({module_log_prefix}) {msg}")
            forecast_output_dict["data_processing_notes"].append(msg); return forecast_output_dict
        note = f"No specific items or key drug config; forecasting for {len(final_item_list_to_forecast)} sample available items: {final_item_list_to_forecast}"
        logger.info(f"({module_log_prefix}) {note}")
        forecast_output_dict["data_processing_notes"].append(note)
    
    if not final_item_list_to_forecast: # Should be caught above, but final safety check
        forecast_output_dict["data_processing_notes"].append("No items were ultimately determined for forecasting after all checks."); return forecast_output_dict
    logger.info(f"({module_log_prefix}) Final items selected for forecast: {final_item_list_to_forecast}")


    # --- Generate Forecast based on Selected Model ---
    df_forecast_results: Optional[pd.DataFrame] = None # Initialize

    if use_ai_forecast_model:
        logger.info(f"({module_log_prefix}) Initiating AI-Simulated Supply Forecasting Model for: {final_item_list_to_forecast}")
        ai_supply_model_instance = SupplyForecastingModel() # Instantiate AI model
        
        # AI model expects latest status: item, current_stock, avg_daily_consumption_historical, last_stock_update_date
        df_latest_item_status_for_ai = df_supply_hist.sort_values('encounter_date').drop_duplicates(subset=['item'], keep='last')
        # Filter this latest status DF for the items specifically selected for forecasting
        df_latest_item_status_for_ai = df_latest_item_status_for_ai[df_latest_item_status_for_ai['item'].isin(final_item_list_to_forecast)]
        
        df_input_for_ai_model = df_latest_item_status_for_ai.rename(columns={
            'item_stock_agg_zone': 'current_stock',
            'consumption_rate_per_day': 'avg_daily_consumption_historical',
            'encounter_date': 'last_stock_update_date'
        })[['item', 'current_stock', 'avg_daily_consumption_historical', 'last_stock_update_date']] # Select required columns

        if not df_input_for_ai_model.empty:
            df_forecast_results = ai_supply_model_instance.forecast_supply_levels_advanced(
                current_supply_levels_df=df_input_for_ai_model,
                forecast_days_out=forecast_days_out
            )
            if isinstance(df_forecast_results, pd.DataFrame) and not df_forecast_results.empty:
                # Standardize column names from AI model output for consistency downstream
                df_forecast_results.rename(columns={
                    'forecast_date': 'date', # AI model might output 'forecast_date'
                    'estimated_stockout_date_ai': 'estimated_stockout_date' # Standardize stockout date column name
                }, inplace=True, errors='ignore') # errors='ignore' is safer if columns might vary
        else:
            msg = f"No current status data found for AI forecasting the selected items: {final_item_list_to_forecast}"
            logger.warning(f"({module_log_prefix}) {msg}")
            forecast_output_dict["data_processing_notes"].append(msg)
    
    else: # Use simple linear forecast (from core_data_processing)
        logger.info(f"({module_log_prefix}) Initiating Simple Linear Supply Forecasting for: {final_item_list_to_forecast}")
        df_forecast_results = get_supply_forecast_data(
            health_df=df_supply_hist, # This function uses the historical DF and filters internally
            forecast_days_out=forecast_days_out,
            item_filter_list=final_item_list_to_forecast, # Pass specific items to filter within the function
            source_context=f"{module_log_prefix}/SimpleLinearForecast"
        )
        if isinstance(df_forecast_results, pd.DataFrame) and not df_forecast_results.empty:
            # Standardize column names from linear model output if they differ
            rename_map_linear_forecast = {
                'estimated_stockout_date_linear':'estimated_stockout_date',
                # 'initial_days_supply_at_forecast_start' - name is likely fine
                # 'initial_stock_at_forecast_start' - name is likely fine
                # 'base_consumption_rate_per_day' - name is likely fine
            }
            df_forecast_results.rename(columns=rename_map_linear_forecast, inplace=True, errors='ignore')

    # Process and summarize the generated forecast DataFrame
    if isinstance(df_forecast_results, pd.DataFrame) and not df_forecast_results.empty:
        forecast_output_dict["forecast_detail_df"] = df_forecast_results.sort_values(by=['item', 'date']).reset_index(drop=True)
        
        item_overview_summary_list = []
        # For summary, group by item and take first row after sorting (or use drop_duplicates on item for latest stockout date)
        # The first day of forecast for each item should contain initial stock & consumption rate used.
        df_summary_source_for_overview = df_forecast_results.sort_values(['item', 'date']).drop_duplicates(subset=['item'], keep='first')

        for _, item_summary_data_row in df_summary_source_for_overview.iterrows():
            item_name_val = item_summary_data_row['item']
            # Try to get initial stock from specific columns, fallback if names vary by model
            initial_stock_val = item_summary_data_row.get('initial_stock_at_forecast_start', item_summary_data_row.get('current_stock', 0.0))
            base_consumption_val = item_summary_data_row.get('base_consumption_rate_per_day', item_summary_data_row.get('predicted_daily_consumption', 1e-6)) # If AI, predicted on day 1
            if pd.isna(base_consumption_val) or base_consumption_val <= 0 : base_consumption_val = 1e-6 # Final check for DivByZero

            initial_dos_val = item_summary_data_row.get('initial_days_supply_at_forecast_start', (initial_stock_val / base_consumption_val if base_consumption_val > 0 else np.inf))
            
            estimated_stockout_dt_val = pd.to_datetime(item_summary_data_row.get('estimated_stockout_date', pd.NaT), errors='coerce')

            item_overview_summary_list.append({
                "item_name": item_name_val,
                "current_stock_on_hand": float(initial_stock_val),
                "avg_daily_consumption_used": float(base_consumption_val), # The rate effectively used for this item's forecast start
                "initial_days_of_supply_est": round(initial_dos_val,1) if np.isfinite(initial_dos_val) else "Adequate (>Forecast Period)",
                "estimated_stockout_date": estimated_stockout_dt_val.strftime('%Y-%m-%d') if pd.notna(estimated_stockout_dt_val) else "Beyond Forecast Period"
            })
        forecast_output_dict["forecast_items_overview_list"] = item_overview_summary_list
        
    elif not forecast_output_dict["data_processing_notes"]: # If forecast DF is empty and no specific notes were added yet
        forecast_output_dict["data_processing_notes"].append("Supply forecast could not be generated with the selected model and available data.")

    num_items_in_overview = len(forecast_output_dict.get('forecast_items_overview_list',[]))
    logger.info(f"({module_log_prefix}) Clinic supply forecast data preparation complete. Items in overview: {num_items_in_overview}")
    return forecast_output_dict
