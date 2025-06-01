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
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_for_utils = os.path.abspath(os.path.join(current_script_dir, os.pardir, os.pardir))
    if project_root_for_utils not in sys.path:
        sys.path.insert(0, project_root_for_utils)
    from config import app_config
    from utils.core_data_processing import get_supply_forecast_data
    from utils.ai_analytics_engine import SupplyForecastingModel

logger = logging.getLogger(__name__)

def prepare_clinic_supply_forecast_data(
    clinic_historical_health_df: Optional[pd.DataFrame],
    reporting_period_str: str,
    forecast_days_out: int = 30,
    use_ai_forecast_model: bool = False,
    items_to_forecast: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Prepares supply forecast data using either a simple linear or an AI-simulated model.
    """
    module_log_prefix = "ClinicSupplyForecastGen" # Shortened prefix
    model_type = "AI-Simulated" if use_ai_forecast_model else "Simple Linear"
    logger.info(f"({module_log_prefix}) Preparing clinic supply forecast. Model: {model_type}, Horizon: {forecast_days_out} days, Items: {'Specific List' if items_to_forecast else 'Auto-detect'}.")

    forecast_output: Dict[str, Any] = {
        "reporting_context": reporting_period_str,
        "forecast_model_type_used": model_type,
        "forecast_detail_df": None,      # DataFrame with daily/periodic forecast per item
        "forecast_items_overview_list": [], # List of dicts: item summary (stock, stockout date)
        "data_processing_notes": []
    }

    required_cols_hist = ['item', 'encounter_date', 'item_stock_agg_zone', 'consumption_rate_per_day']
    if not isinstance(clinic_historical_health_df, pd.DataFrame) or \
       clinic_historical_health_df.empty or \
       not all(col in clinic_historical_health_df.columns for col in required_cols_hist):
        missing = [col for col in required_cols_hist if col not in (clinic_historical_health_df.columns if isinstance(clinic_historical_health_df, pd.DataFrame) else [])]
        msg = f"Historical health data is insufficient for supply forecasts. Missing or empty. Required: {required_cols_hist}. Missing: {missing if missing else 'None, but df may be empty'}."
        logger.error(f"({module_log_prefix}) {msg}")
        forecast_output["data_processing_notes"].append(msg)
        return forecast_output

    df_supply_source = clinic_historical_health_df.copy()
    df_supply_source['encounter_date'] = pd.to_datetime(df_supply_source['encounter_date'], errors='coerce')
    df_supply_source.dropna(subset=['encounter_date', 'item'], inplace=True) # Item name and date are critical
    df_supply_source['item_stock_agg_zone'] = pd.to_numeric(df_supply_source.get('item_stock_agg_zone'), errors='coerce').fillna(0.0)
    # Ensure consumption rate is a small positive number, not zero, to avoid DivByZero
    df_supply_source['consumption_rate_per_day'] = pd.to_numeric(df_supply_source.get('consumption_rate_per_day'), errors='coerce').fillna(1e-6)
    df_supply_source.loc[df_supply_source['consumption_rate_per_day'] <= 0, 'consumption_rate_per_day'] = 1e-6

    # Determine the list of items to forecast
    item_list_for_forecast: List[str]
    if items_to_forecast and isinstance(items_to_forecast, list) and len(items_to_forecast) > 0:
        item_list_for_forecast = items_to_forecast
        logger.debug(f"({module_log_prefix}) Using provided item list: {item_list_for_forecast}")
    elif app_config.KEY_DRUG_SUBSTRINGS_SUPPLY:
        unique_items_in_data_source = df_supply_source['item'].dropna().unique()
        item_list_for_forecast = [
            item for item in unique_items_in_data_source
            if any(drug_sub.lower() in str(item).lower() for drug_sub in app_config.KEY_DRUG_SUBSTRINGS_SUPPLY)
        ]
        if not item_list_for_forecast and len(unique_items_in_data_source) > 0:
            # Fallback to a few items if no key drugs match
            item_list_for_forecast = unique_items_in_data_source[:min(5, len(unique_items_in_data_source))].tolist()
            forecast_output["data_processing_notes"].append(
                f"No key drug substrings matched items in data; forecasting for first {len(item_list_for_forecast)} available items."
            )
        elif not item_list_for_forecast: # Still no items
             msg = "No items found in historical data matching key drug substrings to forecast."
             logger.warning(f"({module_log_prefix}) {msg}")
             forecast_output["data_processing_notes"].append(msg); return forecast_output
    else: # No specific list, no key substrings, fallback to a few unique items
        unique_items_in_data_source = df_supply_source['item'].dropna().unique()
        item_list_for_forecast = unique_items_in_data_source[:min(5, len(unique_items_in_data_source))].tolist()
        if not item_list_for_forecast:
            msg = "No items available in historical data to forecast (no specific list, no key drugs config, no unique items)."
            logger.warning(f"({module_log_prefix}) {msg}")
            forecast_output["data_processing_notes"].append(msg); return forecast_output

    logger.info(f"({module_log_prefix}) Items selected for forecast: {item_list_for_forecast}")
    if not item_list_for_forecast: # Final check
        forecast_output["data_processing_notes"].append("No items were ultimately determined for forecasting."); return forecast_output

    # --- Generate Forecast ---
    forecast_df_generated: Optional[pd.DataFrame] = None
    if use_ai_forecast_model:
        logger.info(f"({module_log_prefix}) Using AI-Simulated Supply Forecasting Model.")
        ai_forecaster = SupplyForecastingModel()
        # AI model expects latest status: item, current_stock, avg_daily_consumption_historical, last_stock_update_date
        latest_item_data_for_ai = df_supply_source.sort_values('encounter_date').drop_duplicates(subset=['item'], keep='last')
        latest_item_data_for_ai = latest_item_data_for_ai[latest_item_data_for_ai['item'].isin(item_list_for_forecast)] # Filter for selected items
        
        input_df_for_ai = latest_item_data_for_ai.rename(columns={
            'item_stock_agg_zone': 'current_stock',
            'consumption_rate_per_day': 'avg_daily_consumption_historical',
            'encounter_date': 'last_stock_update_date'
        })[['item', 'current_stock', 'avg_daily_consumption_historical', 'last_stock_update_date']]

        if not input_df_for_ai.empty:
            forecast_df_generated = ai_forecaster.forecast_supply_levels_advanced(
                current_supply_levels_df=input_df_for_ai,
                forecast_days_out=forecast_days_out
                # Item filtering already applied to input_df_for_ai
            )
            if isinstance(forecast_df_generated, pd.DataFrame) and not forecast_df_generated.empty:
                # Standardize column names from AI model output if necessary
                forecast_df_generated.rename(columns={
                    'forecast_date': 'date', # AI model might output 'forecast_date'
                    'estimated_stockout_date_ai': 'estimated_stockout_date'
                }, inplace=True, errors='ignore') # errors='ignore' is safer
        else:
            forecast_output["data_processing_notes"].append(f"No current status data found for AI forecasting the selected items: {item_list_for_forecast}")
    
    else: # Use simple linear forecast
        logger.info(f"({module_log_prefix}) Using Simple Linear Supply Forecasting.")
        forecast_df_generated = get_supply_forecast_data(
            health_df=df_supply_source, # Uses the historical DF, filters internally
            forecast_days_out=forecast_days_out,
            item_filter_list=item_list_for_forecast,
            source_context=f"{module_log_prefix}/LinearForecast"
        )
        if isinstance(forecast_df_generated, pd.DataFrame) and not forecast_df_generated.empty:
            # Standardize column names from linear model output
            rename_map_linear_fc = {
                'estimated_stockout_date_linear':'estimated_stockout_date',
                # 'initial_days_supply_at_forecast_start' is already good
                # 'initial_stock_at_forecast_start' is already good
                # 'base_consumption_rate_per_day' is already good
            }
            forecast_df_generated.rename(columns=rename_map_linear_fc, inplace=True, errors='ignore')

    # Process and summarize the generated forecast
    if isinstance(forecast_df_generated, pd.DataFrame) and not forecast_df_generated.empty:
        forecast_output["forecast_detail_df"] = forecast_df_generated.sort_values(by=['item', 'date']).reset_index(drop=True)
        
        overview_list = []
        # Ensure summary uses consistent source columns for initial state, regardless of model
        # Use drop_duplicates on item after sorting by date to get the first forecast day's data for each item
        # This first day should contain the initial stock and consumption rate used.
        summary_source_df_for_overview = forecast_df_generated.sort_values(['item', 'date']).drop_duplicates(subset=['item'], keep='first')

        for _, row_item_sum in summary_source_df_for_overview.iterrows():
            item_name_sum = row_item_sum['item']
            # Try to get initial stock from specific columns, fallback to current_stock if AI model used that
            initial_stock = row_item_sum.get('initial_stock_at_forecast_start', row_item_sum.get('current_stock', 0.0))
            # Try specific consumption rate, fallback to predicted for first day if AI used that
            base_consumption = row_item_sum.get('base_consumption_rate_per_day', row_item_sum.get('predicted_daily_consumption', 1e-6))
            if pd.isna(base_consumption) or base_consumption <= 0 : base_consumption = 1e-6 # Avoid DivByZero

            # Initial Days of Supply (DOS) - use a consistent column if available or calculate
            initial_dos = row_item_sum.get('initial_days_supply_at_forecast_start', (initial_stock / base_consumption if base_consumption > 0 else np.inf))
            
            stockout_dt = pd.to_datetime(row_item_sum.get('estimated_stockout_date', pd.NaT), errors='coerce')

            overview_list.append({
                "item_name": item_name_sum,
                "current_stock_on_hand": float(initial_stock),
                "avg_daily_consumption_used": float(base_consumption), # The rate used for this item's forecast start
                "initial_days_of_supply_est": round(initial_dos,1) if np.isfinite(initial_dos) else "Adequate (>Forecast Period)",
                "estimated_stockout_date": stockout_dt.strftime('%Y-%m-%d') if pd.notna(stockout_dt) else "Beyond Forecast Period"
            })
        forecast_output["forecast_items_overview_list"] = overview_list
        
    elif not forecast_output["data_processing_notes"]: # If forecast DF is empty and no specific notes added yet
        forecast_output["data_processing_notes"].append("Supply forecast could not be generated with the selected model and data.")

    num_items_forecasted = len(forecast_output.get('forecast_items_overview_list',[]))
    logger.info(f"({module_log_prefix}) Clinic supply forecast data preparation complete. Items forecasted: {num_items_forecasted}")
    return forecast_output
