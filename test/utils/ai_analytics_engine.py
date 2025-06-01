# sentinel_project_root/test/utils/ai_analytics_engine.py
# Simulates core AI/Analytics logic for the Sentinel Health Co-Pilot System.

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Callable # Ensured Callable is here if needed
import logging
from config import app_config

logger = logging.getLogger(__name__)

class RiskPredictionModel:
    """
    SIMULATES a pre-trained patient/worker risk prediction model.
    Uses rule-based logic with weights for core features.
    Actual Edge model would be an optimized version (e.g., TFLite).
    """
    def __init__(self):
        self.base_risk_factors = {
            'age': {'weight': 0.5, 'threshold_low': 18, 'factor_low': -3, 'threshold_high': 60, 'factor_high': 10, 'threshold_very_high': 75, 'factor_very_high': 15},
            'min_spo2_pct': {'weight': 2.5, 'threshold_low': app_config.ALERT_SPO2_CRITICAL_LOW_PCT, 'factor_low': 30, 'mid_threshold_low': app_config.ALERT_SPO2_WARNING_LOW_PCT, 'factor_mid_low': 15},
            'vital_signs_temperature_celsius': {'weight': 2.0, 'threshold_high': app_config.ALERT_BODY_TEMP_FEVER_C, 'factor_high': 15, 'super_high_threshold': app_config.ALERT_BODY_TEMP_HIGH_FEVER_C, 'factor_super_high': 25},
            'max_skin_temp_celsius': {'weight': 1.8, 'threshold_high': app_config.HEAT_STRESS_RISK_BODY_TEMP_C, 'factor_high': 10, 'super_high_threshold': app_config.ALERT_BODY_TEMP_HIGH_FEVER_C - 0.5, 'factor_super_high': 20},
            'stress_level_score': {'weight': 0.8, 'threshold_high': app_config.FATIGUE_INDEX_MODERATE_THRESHOLD, 'factor_high': 8, 'super_high_threshold': app_config.FATIGUE_INDEX_HIGH_THRESHOLD, 'factor_super_high': 12},
            'hrv_rmssd_ms': {'weight': 1.2, 'threshold_low': app_config.STRESS_HRV_LOW_THRESHOLD_MS, 'factor_low': 15},
            'tb_contact_traced': {'weight': 1.0, 'is_flag': True, 'flag_value': 1, 'factor_true': 12}, # Assumes 1 means is a contact
            'fall_detected_today': {'weight': 2.2, 'is_flag': True, 'flag_value': 1, 'factor_true': 25}, # Assumes 1 means fall detected
            'ambient_heat_index_c': {'weight': 0.7, 'threshold_high': app_config.ALERT_AMBIENT_HEAT_INDEX_RISK_C, 'factor_high': 8, 'super_high_threshold': app_config.ALERT_AMBIENT_HEAT_INDEX_DANGER_C, 'factor_super_high': 15},
            'ppe_compliant_flag': {'weight': 1.0, 'is_flag': True, 'flag_value': 0, 'factor_true': 10} # Risk if flag is 0 (non-compliant)
        }
        self.condition_base_scores = {cond: 25 for cond in app_config.KEY_CONDITIONS_FOR_ACTION}
        self.condition_base_scores.update({
            "Sepsis": 45, "Severe Dehydration": 40, "Heat Stroke": 42, "TB": 30,
            "HIV-Positive": 22, "Pneumonia": 35, "Malaria": 20,
            "Wellness Visit": -15, "Follow-up Health": -8, "Minor Cold": -5 # Negative for wellness/minor
        })
        self.CHRONIC_CONDITION_FLAG_RISK_POINTS = 18 # Increased from 15
        logger.info("Simulated RiskPredictionModel initialized with Sentinel factors.")

    def _get_condition_base_score(self, condition_str: Optional[str]) -> float:
        if pd.isna(condition_str) or not isinstance(condition_str, str) or \
           condition_str.strip().lower() in ["unknown", "none", "n/a", ""]:
            return 0.0
        
        max_score_found = 0.0
        condition_input_lower = condition_str.lower()
        
        # Check for multi-conditions separated by common delimiters like ';' or ','
        # and evaluate each part, taking the max score.
        potential_conditions = [c.strip() for delim in [';', ','] for c in condition_input_lower.split(delim)]
        if not any(delim in condition_input_lower for delim in [';', ',']): # If no delimiters, treat as single
            potential_conditions = [condition_input_lower]

        for part_cond_lower in potential_conditions:
            if not part_cond_lower: continue # Skip empty strings after split

            # Attempt exact match first for performance on single conditions
            exact_match_score = self.condition_base_scores.get(part_cond_lower.title()) # Check title case
            if exact_match_score is not None:
                 max_score_found = max(max_score_found, exact_match_score)
                 continue # Found for this part_cond_lower

            # If no exact match, check for partial matches (e.g., "TB" in "Suspected TB")
            for known_cond, score_val in self.condition_base_scores.items():
                if known_cond.lower() in part_cond_lower:
                    max_score_found = max(max_score_found, score_val)
        
        return max_score_found

    def predict_risk_score(self, features: pd.Series) -> float:
        if not isinstance(features, pd.Series):
            logger.error("RiskPredictionModel.predict_risk_score expects a pandas Series.")
            return 0.0 # Or raise error
            
        calculated_risk = self._get_condition_base_score(features.get('condition'))

        # Handle boolean-like flags robustly (could be int 0/1 or string '0'/'1', 'yes'/'no')
        if str(features.get('chronic_condition_flag', '0')).lower() in ['1', 'yes', 'true']:
            calculated_risk += self.CHRONIC_CONDITION_FLAG_RISK_POINTS

        for feature_key, params in self.base_risk_factors.items():
            feature_value = features.get(feature_key)
            if pd.notna(feature_value): # Ensure value is not NaN before processing
                value = feature_value 
                weight = params.get('weight', 1.0)

                if params.get('is_flag'):
                    expected_flag_val = params.get('flag_value', 1) # Default true flag is 1
                    # Robust check for flag values (e.g., 1, '1', True)
                    if str(value).lower() == str(expected_flag_val).lower() or \
                       (isinstance(value, bool) and value == bool(expected_flag_val)):
                        calculated_risk += params.get('factor_true', 0) * weight
                else: # Threshold-based logic
                    # Ensure value is numeric for comparisons
                    try:
                        numeric_value = float(value)
                    except (ValueError, TypeError):
                        logger.debug(f"Could not convert feature '{feature_key}' value '{value}' to float for risk scoring.")
                        continue # Skip this feature if not convertible

                    if 'super_high_threshold' in params and numeric_value >= params['super_high_threshold']:
                        calculated_risk += params.get('factor_super_high', 0) * weight
                    elif 'threshold_very_high' in params and numeric_value >= params['threshold_very_high']: # For age
                         calculated_risk += params.get('factor_very_high', 0) * weight
                    elif 'threshold_high' in params and numeric_value >= params['threshold_high']:
                        calculated_risk += params.get('factor_high', 0) * weight
                    
                    if 'threshold_low' in params and numeric_value < params['threshold_low']:
                        calculated_risk += params.get('factor_low', 0) * weight
                    elif 'mid_threshold_low' in params and numeric_value < params['mid_threshold_low']:
                        calculated_risk += params.get('factor_mid_low', 0) * weight
        
        adherence = str(features.get('medication_adherence_self_report', "Unknown")).lower()
        if adherence == 'poor': calculated_risk += 12
        elif adherence == 'fair': calculated_risk += 6
        
        psych_distress_score = features.get('rapid_psychometric_distress_score')
        if pd.notna(psych_distress_score):
            try:
                calculated_risk += float(psych_distress_score) * 1.5 # Max 15 points if score is 10
            except (ValueError, TypeError):
                logger.debug(f"Could not convert rapid_psychometric_distress_score '{psych_distress_score}' to float.")

        if str(features.get('signs_of_fatigue_observed_flag', '0')).lower() in ['1', 'true']:
            calculated_risk += 10

        return float(np.clip(calculated_risk, 0, 100))

    def predict_bulk_risk_scores(self, data_df: pd.DataFrame) -> pd.Series:
        if not isinstance(data_df, pd.DataFrame) or data_df.empty:
            return pd.Series(dtype='float64')
        
        temp_df = data_df.copy()
        # Ensure all features the model expects are present, adding with sensible defaults if not
        all_expected_model_features = list(self.base_risk_factors.keys()) + [
            'condition', 'chronic_condition_flag', 'medication_adherence_self_report',
            'rapid_psychometric_distress_score', 'signs_of_fatigue_observed_flag'
        ]
        for feature_name in all_expected_model_features:
            if feature_name not in temp_df.columns:
                if any(flag_part in feature_name for flag_part in ['_flag', '_traced', '_detected']):
                    default_val = 0 # Flags default to 'false' or 'no' state
                    if feature_name == 'ppe_compliant_flag': default_val = 1 # Default to compliant for PPE
                elif any(num_part in feature_name for num_part in ['_score', 'age', '_ms', '_pct', '_celsius', '_index_c']):
                    default_val = np.nan # Numeric scores/vitals default to NaN
                else: # String columns like 'condition', 'medication_adherence_self_report'
                    default_val = "Unknown"
                temp_df[feature_name] = default_val
                logger.debug(f"Added missing feature '{feature_name}' with default '{default_val}' for bulk risk prediction.")
            
            # Ensure correct dtypes for features that will be used in numeric comparisons
            if any(num_part in feature_name for num_part in ['_score', 'age', '_ms', '_pct', '_celsius', '_index_c']):
                 if feature_name in temp_df.columns: # Check again as it might have been added
                    temp_df[feature_name] = pd.to_numeric(temp_df[feature_name], errors='coerce')

        return temp_df.apply(lambda row: self.predict_risk_score(row), axis=1)


class FollowUpPrioritizer:
    """ Simulates logic to prioritize patients for follow-up or tasks. """
    def __init__(self):
        self.priority_weights = {
            'base_ai_risk_score_contribution_pct': 0.40, # Increased impact of AI risk
            'critical_vital_alert_points': 40,        
            'pending_urgent_task_points': 30,         
            'acute_condition_severity_points': 25,    
            'contextual_hazard_points': 15,           
            'task_overdue_factor_per_day': 1.0,       # Increased impact
            'poor_adherence_points': 15,              # Increased impact
            'observed_fatigue_points': 12             # For CHW-observed patient fatigue
        }
        logger.info("Simulated FollowUpPrioritizer initialized with Sentinel weights.")

    def _has_active_critical_vitals_alert(self, features: pd.Series) -> bool:
        if pd.notna(features.get('min_spo2_pct')) and \
           float(features.get('min_spo2_pct', 100)) < app_config.ALERT_SPO2_CRITICAL_LOW_PCT: return True
        
        temp_val = features.get('vital_signs_temperature_celsius', features.get('max_skin_temp_celsius'))
        if pd.notna(temp_val) and float(temp_val) >= app_config.ALERT_BODY_TEMP_HIGH_FEVER_C: return True
        
        if pd.notna(features.get('fall_detected_today')) and \
           str(features.get('fall_detected_today', '0')).lower() in ['1', 'true']: return True
        return False

    def _is_pending_urgent_task(self, features: pd.Series) -> bool:
        if str(features.get('referral_status', '')).lower() == 'pending':
            condition_str_lower = str(features.get('condition', '')).lower()
            if any(key_cond.lower() in condition_str_lower for key_cond in app_config.KEY_CONDITIONS_FOR_ACTION):
                return True
        # Assuming 'worker_task_priority' could be a field indicating externally set urgency
        if str(features.get('worker_task_priority', 'Normal')).lower() == 'urgent':
            return True
        return False
        
    def _has_acute_condition_severity(self, features: pd.Series) -> bool:
        condition_str_lower = str(features.get('condition','')).lower()
        if "pneumonia" in condition_str_lower and \
           pd.notna(features.get('min_spo2_pct')) and \
           float(features.get('min_spo2_pct', 100)) < app_config.ALERT_SPO2_WARNING_LOW_PCT:
            return True
        if any(crit_cond.lower() in condition_str_lower for crit_cond in ["sepsis", "severe dehydration", "heat stroke"]):
            return True
        return False

    def _contextual_hazard_present(self, features: pd.Series) -> bool:
        if pd.notna(features.get('ambient_heat_index_c')) and \
           float(features.get('ambient_heat_index_c', 0)) >= app_config.ALERT_AMBIENT_HEAT_INDEX_DANGER_C:
            return True
        # Could add other environmental checks like CO2 if relevant data is in `features`
        # e.g., if features.get('ambient_co2_ppm') >= app_config.ALERT_AMBIENT_CO2_VERY_HIGH_PPM
        return False

    def calculate_priority_score(self, features: pd.Series, days_task_overdue: int = 0) -> float:
        if not isinstance(features, pd.Series):
            logger.error("FollowUpPrioritizer.calculate_priority_score expects a pandas Series.")
            return 0.0
        priority_score = 0.0
        
        ai_risk_val = features.get('ai_risk_score', 0.0) # Default to 0 if missing
        if pd.notna(ai_risk_val):
            try:
                priority_score += float(ai_risk_val) * self.priority_weights['base_ai_risk_score_contribution_pct']
            except (ValueError, TypeError): pass # Ignore if not convertible
        
        if self._has_active_critical_vitals_alert(features):
            priority_score += self.priority_weights['critical_vital_alert_points']
        if self._is_pending_urgent_task(features):
            priority_score += self.priority_weights['pending_urgent_task_points']
        if self._has_acute_condition_severity(features):
            priority_score += self.priority_weights['acute_condition_severity_points']
        if self._contextual_hazard_present(features):
            priority_score += self.priority_weights['contextual_hazard_points']
            
        if str(features.get('medication_adherence_self_report', "Unknown")).lower() == 'poor':
            priority_score += self.priority_weights['poor_adherence_points']
        
        if str(features.get('signs_of_fatigue_observed_flag', '0')).lower() in ['1', 'true']:
            priority_score += self.priority_weights['observed_fatigue_points']
            
        priority_score += min(int(days_task_overdue), 60) * self.priority_weights['task_overdue_factor_per_day'] # Cap overdue impact
        
        return float(np.clip(priority_score, 0, 100))

    def generate_followup_priorities(self, data_df: pd.DataFrame) -> pd.Series:
        if not isinstance(data_df, pd.DataFrame) or data_df.empty:
            return pd.Series(dtype='float64')
        
        temp_df = data_df.copy()
        # Ensure 'days_task_overdue' exists and is int
        if 'days_task_overdue' not in temp_df.columns:
            temp_df['days_task_overdue'] = 0
        else:
            temp_df['days_task_overdue'] = pd.to_numeric(temp_df['days_task_overdue'], errors='coerce').fillna(0).astype(int)
        
        # Ensure all features needed by helper methods are present, adding with neutral defaults if not
        # This is critical to prevent errors if data_df is sparse.
        features_to_check_for_prio = [
            'min_spo2_pct', 'vital_signs_temperature_celsius', 'max_skin_temp_celsius', 
            'fall_detected_today', 'referral_status', 'condition', 'worker_task_priority', 
            'ambient_heat_index_c', 'medication_adherence_self_report', 
            'signs_of_fatigue_observed_flag', 'ai_risk_score'
        ]
        for feature_name_prio in features_to_check_for_prio:
            if feature_name_prio not in temp_df.columns:
                default_val_prio = np.nan if any(x in feature_name_prio for x in ['pct', 'temp', 'score', 'index']) else \
                                   0 if any(x in feature_name_prio for x in ['flag', 'today']) else \
                                   "Unknown"
                temp_df[feature_name_prio] = default_val_prio
                logger.debug(f"Added missing feature '{feature_name_prio}' with default '{default_val_prio}' for priority calculation.")
            # Coerce numeric types
            if any(x in feature_name_prio for x in ['pct', 'temp', 'score', 'index']):
                temp_df[feature_name_prio] = pd.to_numeric(temp_df[feature_name_prio], errors='coerce')


        return temp_df.apply(
            lambda row: self.calculate_priority_score(row, row.get('days_task_overdue', 0)),
            axis=1
        )


class SupplyForecastingModel:
    """ Simulates an AI-driven supply forecasting model with seasonality and trend. """
    def __init__(self):
        self.item_params: Dict[str, Dict[str, Any]] = {}
        base_coeffs = np.array([1.0] * 12) # Flat seasonality baseline
        
        for item_key_substring in app_config.KEY_DRUG_SUBSTRINGS_SUPPLY:
            # Use hash for deterministic pseudo-randomness based on item name
            # Ensures different items get different (but consistent across runs) params
            seed_val = abs(hash(item_key_substring)) % (2**32 - 1)
            rng_item = np.random.RandomState(seed_val)
            
            self.item_params[item_key_substring] = {
                "coeffs": (base_coeffs * rng_item.uniform(0.85, 1.15, 12)).round(3).tolist(), # Monthly seasonality factors
                "trend": rng_item.uniform(0.0001, 0.003),  # Small positive daily trend component
                "noise_std": rng_item.uniform(0.02, 0.08) # Reduced noise for more stable forecasts
            }
        
        # Example: Specific override for "ACT" (Artemisinin-based Combination Therapy) if it's a key drug
        act_key = next((s_act for s_act in app_config.KEY_DRUG_SUBSTRINGS_SUPPLY if "act" in s_act.lower()), None)
        if act_key and act_key in self.item_params: # Check if ACT key exists in our generated params
            self.item_params[act_key]["coeffs"] = [0.7,0.7,0.8,1.0,1.3,1.5,1.6,1.4,1.1,0.9,0.8,0.7] # Malaria seasonality
            self.item_params[act_key]["trend"] = 0.0015 # Slightly different trend for ACT

        logger.info(f"Simulated AI SupplyForecastingModel initialized with params for {len(self.item_params)} key drug groups.")

    def _get_item_params(self, item_name: str) -> Dict[str, Any]:
        item_name_lower = str(item_name).lower() # Ensure item_name is string
        for key_substring, params in self.item_params.items():
            if key_substring.lower() in item_name_lower:
                return params
        # Generic fallback if no specific substring match
        logger.debug(f"No specific params for item '{item_name}', using generic fallback supply forecast params.")
        return {"coeffs": [1.0]*12, "trend": 0.0001, "noise_std": 0.05} 

    def _predict_daily_consumption_ai(self, base_avg_daily_consumption: float, item_name: str, forecast_date: pd.Timestamp, days_since_forecast_start: int) -> float:
        if pd.isna(base_avg_daily_consumption) or base_avg_daily_consumption <= 1e-6: # Use a small epsilon
            return 1e-6 # Avoid division by zero, ensure minimal positive consumption if base is zero/NaN
        
        params = self._get_item_params(item_name)
        
        month_index = forecast_date.month - 1 # .month is 1-indexed for Timestamp
        seasonality_factor = params["coeffs"][month_index]
        
        # Trend effect: (1 + daily_trend_rate) ^ number_of_days
        trend_effect = (1 + params["trend"]) ** days_since_forecast_start
        
        # Add some random noise for simulation. For testing, this could be seeded or disabled.
        # In a real model, this would represent model uncertainty or unexplained variance.
        noise_factor = np.random.normal(1.0, params["noise_std"]) # Centered around 1
        
        predicted_consumption = base_avg_daily_consumption * seasonality_factor * trend_effect * noise_factor
        return max(1e-6, predicted_consumption) # Ensure consumption is always a small positive number

    def forecast_supply_levels_advanced(
            self, current_supply_levels_df: pd.DataFrame, 
            forecast_days_out: int = 30, # Default to 30 days
            item_filter_list: Optional[List[str]] = None
        ) -> pd.DataFrame:
        logger.info(f"AI-simulated supply forecast initiated. Horizon: {forecast_days_out} days. Input items: {len(current_supply_levels_df) if current_supply_levels_df is not None else 0}")
        
        output_cols_supply_fc = ['item', 'forecast_date', 'forecasted_stock_level', 
                                 'forecasted_days_of_supply', 'predicted_daily_consumption', 
                                 'estimated_stockout_date_ai']

        if not isinstance(current_supply_levels_df, pd.DataFrame) or current_supply_levels_df.empty:
            logger.warning("AI Supply Forecast: Input DataFrame of current supply levels is empty.")
            return pd.DataFrame(columns=output_cols_supply_fc)

        required_cols_supply_fc_input = ['item', 'current_stock', 'avg_daily_consumption_historical', 'last_stock_update_date']
        missing_cols = [col for col in required_cols_supply_fc_input if col not in current_supply_levels_df.columns]
        if missing_cols:
            logger.error(f"AI Supply Forecast: Input DataFrame missing required columns: {missing_cols}")
            return pd.DataFrame(columns=output_cols_supply_fc)

        df_to_process = current_supply_levels_df.copy()
        if item_filter_list:
            df_to_process = df_to_process[df_to_process['item'].isin(item_filter_list)]
        
        if df_to_process.empty:
            logger.info("AI Supply Forecast: No items to forecast after applying item_filter_list.")
            return pd.DataFrame(columns=output_cols_supply_fc)
        
        df_to_process['last_stock_update_date'] = pd.to_datetime(df_to_process['last_stock_update_date'], errors='coerce')
        # Ensure critical columns for calculations are numeric and filled
        df_to_process['current_stock'] = pd.to_numeric(df_to_process['current_stock'], errors='coerce').fillna(0.0)
        df_to_process['avg_daily_consumption_historical'] = pd.to_numeric(df_to_process['avg_daily_consumption_historical'], errors='coerce').fillna(1e-6) # Min positive consumption
        df_to_process.dropna(subset=['last_stock_update_date', 'item'], inplace=True)

        if df_to_process.empty: # After cleaning, check again
            logger.warning("AI Supply Forecast: No valid items left after cleaning dates/numerics.")
            return pd.DataFrame(columns=output_cols_supply_fc)

        all_forecast_records_list = []
        for _, item_row_data in df_to_process.iterrows():
            item_name = str(item_row_data['item'])
            stock_start = max(0.0, float(item_row_data['current_stock'])) # Ensure non-negative
            base_daily_cons = max(1e-6, float(item_row_data['avg_daily_consumption_historical'])) # Ensure positive
            last_update_dt = item_row_data['last_stock_update_date']

            running_stock_level = stock_start
            estimated_stockout_for_item = pd.NaT # Initialize as Not a Time
            
            item_daily_forecasts_buffer = [] # Buffer to hold this item's forecasts before finalizing stockout date

            for day_offset_val in range(forecast_days_out):
                current_fc_dt = last_update_dt + pd.Timedelta(days=day_offset_val + 1)
                predicted_use_today = self._predict_daily_consumption_ai(base_daily_cons, item_name, current_fc_dt, day_offset_val + 1)
                
                stock_before_today_use = running_stock_level
                running_stock_level = max(0, running_stock_level - predicted_use_today)
                days_supply_at_eod = (running_stock_level / predicted_use_today) if predicted_use_today > 1e-7 else \
                                     (np.inf if running_stock_level > 0 else 0)

                # Estimate stockout date if it occurs
                if pd.isna(estimated_stockout_for_item) and stock_before_today_use > 0 and running_stock_level <= 0:
                    # Estimate fractional day of stockout more precisely
                    fraction_of_day_at_stockout = (stock_before_today_use / predicted_use_today) if predicted_use_today > 1e-7 else 0.0
                    estimated_stockout_for_item = last_update_dt + pd.Timedelta(days=day_offset_val + fraction_of_day_at_stockout)
                
                item_daily_forecasts_buffer.append({
                    'item': item_name, 'forecast_date': current_fc_dt, 
                    'forecasted_stock_level': running_stock_level, 
                    'forecasted_days_of_supply': days_supply_at_eod, 
                    'predicted_daily_consumption': predicted_use_today,
                    'estimated_stockout_date_ai': estimated_stockout_for_item # Will be NaT for many initial records
                })
            
            # If no stockout within forecast_days_out, estimate based on average predicted consumption
            if pd.isna(estimated_stockout_for_item) and stock_start > 0 and item_daily_forecasts_buffer:
                avg_pred_consumption_over_period = pd.Series([d['predicted_daily_consumption'] for d in item_daily_forecasts_buffer]).mean()
                if avg_pred_consumption_over_period > 1e-7:
                    days_to_stockout_from_start = stock_start / avg_pred_consumption_over_period
                    final_estimated_stockout_dt = last_update_dt + pd.to_timedelta(days_to_stockout_from_start, unit='D')
                    # Update all buffered records for this item with this final stockout date
                    for record in item_daily_forecasts_buffer:
                        if pd.isna(record['estimated_stockout_date_ai']): # Only update if not already set
                             record['estimated_stockout_date_ai'] = final_estimated_stockout_dt
            
            all_forecast_records_list.extend(item_daily_forecasts_buffer)
        
        if not all_forecast_records_list:
            return pd.DataFrame(columns=output_cols_supply_fc)
        
        final_forecast_df = pd.DataFrame(all_forecast_records_list)
        # Ensure estimated_stockout_date_ai is datetime
        if 'estimated_stockout_date_ai' in final_forecast_df.columns:
             final_forecast_df['estimated_stockout_date_ai'] = pd.to_datetime(final_forecast_df['estimated_stockout_date_ai'], errors='coerce')
        
        logger.info(f"AI-Simulated supply forecast complete. Generated {len(final_forecast_df)} daily records for {df_to_process['item'].nunique()} items.")
        return final_forecast_df


# --- Central AI Application Function ---
def apply_ai_models(
    health_df: pd.DataFrame, # Input health records
    current_supply_status_df: Optional[pd.DataFrame] = None, # Optional: For supply forecasting
    source_context: str = "AIModelOrchestrator" # More generic context name
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    logger.info(f"({source_context}) Applying AI models to health data (rows: {len(health_df) if health_df is not None else 0}).")
    
    if not isinstance(health_df, pd.DataFrame):
        logger.error(f"({source_context}) Input health_df is not a DataFrame (type: {type(health_df)}). Cannot apply AI models.")
        # Return empty DataFrame with expected AI columns, and None for supply forecast
        return pd.DataFrame(columns=['ai_risk_score', 'ai_followup_priority_score']), None

    if health_df.empty:
        logger.warning(f"({source_context}) Input health_df to apply_ai_models is empty. Returning empty enriched DataFrame.")
        # Create a DataFrame with expected AI columns but no data
        base_cols = health_df.columns.tolist() # Get columns from the empty input if any
        ai_cols_to_add = ['ai_risk_score', 'ai_followup_priority_score']
        # Ensure no duplicate columns if input already had them (though unlikely if truly empty)
        final_empty_cols = list(set(base_cols + ai_cols_to_add))
        return pd.DataFrame(columns=final_empty_cols), None

    df_enriched = health_df.copy() # Work on a copy to avoid modifying original

    # 1. Risk Scoring
    risk_model_instance = RiskPredictionModel()
    df_enriched['ai_risk_score'] = risk_model_instance.predict_bulk_risk_scores(df_enriched)
    logger.debug(f"({source_context}) Applied AI Risk Scoring. Example scores: {df_enriched['ai_risk_score'].head(2).tolist() if not df_enriched.empty else 'N/A'}")

    # 2. Follow-up/Task Prioritization
    followup_prioritizer_instance = FollowUpPrioritizer()
    # Ensure 'days_task_overdue' exists for the prioritizer model; default to 0 if not present
    if 'days_task_overdue' not in df_enriched.columns: 
        df_enriched['days_task_overdue'] = 0 
    else: # If column exists, ensure it's integer and fill NaNs with 0
        df_enriched['days_task_overdue'] = pd.to_numeric(df_enriched['days_task_overdue'], errors='coerce').fillna(0).astype(int)
    
    df_enriched['ai_followup_priority_score'] = followup_prioritizer_instance.generate_followup_priorities(df_enriched)
    logger.debug(f"({source_context}) Applied AI Follow-up Prioritization. Example scores: {df_enriched['ai_followup_priority_score'].head(2).tolist() if not df_enriched.empty else 'N/A'}")
    
    # 3. AI-Simulated Supply Forecasting (Optional, if current_supply_status_df is provided)
    generated_supply_forecast_df: Optional[pd.DataFrame] = None # Initialize
    if isinstance(current_supply_status_df, pd.DataFrame) and not current_supply_status_df.empty:
        # Check for required columns in current_supply_status_df
        required_supply_input_cols = ['item', 'current_stock', 'avg_daily_consumption_historical', 'last_stock_update_date']
        if all(col in current_supply_status_df.columns for col in required_supply_input_cols):
            supply_forecaster_ai_instance = SupplyForecastingModel()
            generated_supply_forecast_df = supply_forecaster_ai_instance.forecast_supply_levels_advanced(
                current_supply_levels_df=current_supply_status_df,
                # Forecast for a meaningful period, e.g., twice the "low supply" warning threshold
                forecast_days_out=app_config.LOW_SUPPLY_DAYS_REMAINING * 2 
            )
            rec_count_forecast = len(generated_supply_forecast_df) if generated_supply_forecast_df is not None else 0
            logger.debug(f"({source_context}) Generated AI-simulated supply forecast: {rec_count_forecast} daily records.")
        else:
            missing_supply_cols_str = ", ".join([col for col in required_supply_input_cols if col not in current_supply_status_df.columns])
            logger.warning(f"({source_context}) 'current_supply_status_df' provided but missing required columns ({missing_supply_cols_str}) for AI supply forecast. Skipping.")
    else:
        logger.debug(f"({source_context}) No 'current_supply_status_df' provided or it's empty; AI supply forecast skipped.")

    logger.info(f"({source_context}) AI model applications complete. Enriched Health DataFrame shape: {df_enriched.shape}")
    return df_enriched, generated_supply_forecast_df
