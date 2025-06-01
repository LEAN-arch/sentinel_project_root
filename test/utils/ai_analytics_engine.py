# sentinel_project_root/test/utils/ai_analytics_engine.py
# Redesigned for "Sentinel Health Co-Pilot" - LMIC Edge-First System

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import logging
from config import app_config # Uses the fully refactored app_config

logger = logging.getLogger(__name__)

class RiskPredictionModel:
    """
    SIMULATES a pre-trained patient/worker risk prediction model, adaptable for Edge deployment.
    Uses rule-based logic with weights and factors for core features.
    """
    def __init__(self):
        self.base_risk_factors = {
            'age': {'weight': 0.5, 'threshold_high': 60, 'factor_high': 10, 'threshold_low': 18, 'factor_low': -3, 'threshold_very_high': 75, 'factor_very_high': 15}, # Added very_high for older age
            'min_spo2_pct': {'weight': 2.5, 'threshold_low': app_config.ALERT_SPO2_CRITICAL_LOW_PCT, 'factor_low': 30, 'mid_threshold_low': app_config.ALERT_SPO2_WARNING_LOW_PCT, 'factor_mid_low': 15},
            'vital_signs_temperature_celsius': {'weight': 2.0, 'threshold_high': app_config.ALERT_BODY_TEMP_FEVER_C, 'factor_high': 15, 'super_high_threshold': app_config.ALERT_BODY_TEMP_HIGH_FEVER_C, 'factor_super_high': 25},
            'max_skin_temp_celsius': {'weight': 1.8, 'threshold_high': app_config.HEAT_STRESS_RISK_BODY_TEMP_C, 'factor_high': 10, 'super_high_threshold': app_config.ALERT_BODY_TEMP_HIGH_FEVER_C - 0.5, 'factor_super_high': 20}, # Skin temp related to heat stress
            'stress_level_score': {'weight': 0.8, 'threshold_high': app_config.FATIGUE_INDEX_MODERATE_THRESHOLD, 'factor_high': 8, 'super_high_threshold': app_config.FATIGUE_INDEX_HIGH_THRESHOLD, 'factor_super_high': 12}, # Used for worker fatigue primarily
            'hrv_rmssd_ms': {'weight': 1.2, 'threshold_low': app_config.STRESS_HRV_LOW_THRESHOLD_MS, 'factor_low': 15}, # Low HRV indicates stress
            'tb_contact_traced': {'weight': 1.0, 'is_flag': True, 'flag_value': 1, 'factor_true': 12}, # Being a TB contact
            'fall_detected_today': {'weight': 2.2, 'is_flag': True, 'flag_value': 1, 'factor_true': 25}, # Increased weight for falls
            'ambient_heat_index_c': {'weight': 0.7, 'threshold_high': app_config.ALERT_AMBIENT_HEAT_INDEX_RISK_C, 'factor_high': 8, 'super_high_threshold': app_config.ALERT_AMBIENT_HEAT_INDEX_DANGER_C, 'factor_super_high': 15},
            'ppe_compliant_flag': {'weight': 1.0, 'is_flag': True, 'flag_value': 0, 'factor_true': 10} # Risk if flag is 0 (non-compliant)
        }
        # Base scores for conditions
        self.condition_base_scores = {cond: 25 for cond in app_config.KEY_CONDITIONS_FOR_ACTION} # Default for key conditions
        self.condition_base_scores.update({ # Specific overrides
            "Sepsis": 45, "Severe Dehydration": 40, "Heat Stroke": 42, "TB": 30, # Higher for severe conditions
            "HIV-Positive": 22, "Pneumonia": 35, "Malaria": 20, # Adjusted Pneumonia
            "Wellness Visit": -15, "Follow-up Health": -8, "Minor Cold": -5 # Negative for wellness/minor
        })
        self.CHRONIC_CONDITION_FLAG_RISK_POINTS = 18 # Slightly increased
        logger.info("Simulated RiskPredictionModel initialized.")

    def _get_condition_base_score(self, condition_str: Optional[str]) -> float:
        if pd.isna(condition_str) or not isinstance(condition_str, str) or condition_str.strip().lower() in ["unknown", "none", "n/a", ""]:
            return 0.0
        
        base_score = 0.0
        max_score_found = 0.0 # To handle multiple conditions by taking the highest base
        condition_input_lower = condition_str.lower()

        # Exact matches first, then partial
        for known_cond, score_val in self.condition_base_scores.items():
            if known_cond.lower() == condition_input_lower:
                max_score_found = max(max_score_found, score_val)
                break # Found exact, no need to check partials for this specific known_cond
        else: # If no exact match found, check for partials
            for known_cond, score_val in self.condition_base_scores.items():
                if known_cond.lower() in condition_input_lower:
                    max_score_found = max(max_score_found, score_val)
        
        return max_score_found if max_score_found != 0 else 0.0 # Return 0 if no relevant condition found


    def predict_risk_score(self, features: pd.Series) -> float:
        if not isinstance(features, pd.Series):
            logger.error("RiskPredictionModel.predict_risk_score expects a pandas Series.")
            return 0.0
            
        calculated_risk = self._get_condition_base_score(features.get('condition'))

        if features.get('chronic_condition_flag') == 1 or str(features.get('chronic_condition_flag', '0')).lower() in ['1', 'yes', 'true']:
            calculated_risk += self.CHRONIC_CONDITION_FLAG_RISK_POINTS

        for feature_key, params in self.base_risk_factors.items():
            feature_value = features.get(feature_key)
            if pd.notna(feature_value):
                value = feature_value # Already checked for NaT etc if date
                weight = params.get('weight', 1.0)

                if params.get('is_flag'):
                    expected_flag_val = params.get('flag_value', 1) # Default true flag is 1
                    if value == expected_flag_val:
                        calculated_risk += params.get('factor_true', 0) * weight
                else: # Threshold-based
                    if 'super_high_threshold' in params and value >= params['super_high_threshold']:
                        calculated_risk += params.get('factor_super_high', 0) * weight
                    elif 'threshold_high' in params and value >= params['threshold_high']:
                        calculated_risk += params.get('factor_high', 0) * weight
                    # Age has special very_high for >75
                    elif 'threshold_very_high' in params and value >= params['threshold_very_high']:
                         calculated_risk += params.get('factor_very_high', 0) * weight

                    if 'threshold_low' in params and value < params['threshold_low']: # e.g. SpO2, HRV
                        calculated_risk += params.get('factor_low', 0) * weight
                    elif 'mid_threshold_low' in params and value < params['mid_threshold_low']: # e.g. SpO2 warning
                        calculated_risk += params.get('factor_mid_low', 0) * weight
        
        adherence = str(features.get('medication_adherence_self_report', "Unknown")).lower()
        if adherence == 'poor': calculated_risk += 12 # Increased slightly
        elif adherence == 'fair': calculated_risk += 6
        
        psych_distress = features.get('rapid_psychometric_distress_score')
        if pd.notna(psych_distress):
            calculated_risk += float(psych_distress) * 1.5 # Max 15 points if score is 10

        if features.get('signs_of_fatigue_observed_flag') == 1:
            calculated_risk += 10

        return float(np.clip(calculated_risk, 0, 100))

    def predict_bulk_risk_scores(self, data_df: pd.DataFrame) -> pd.Series:
        if not isinstance(data_df, pd.DataFrame) or data_df.empty:
            return pd.Series(dtype='float64')
        
        temp_df = data_df.copy()
        # Ensure all features the model expects are present, adding with defaults if not
        all_expected_features = list(self.base_risk_factors.keys()) + [
            'condition', 'chronic_condition_flag', 'medication_adherence_self_report',
            'rapid_psychometric_distress_score', 'signs_of_fatigue_observed_flag'
        ]
        for feature_name in all_expected_features:
            if feature_name not in temp_df.columns:
                # Determine appropriate default based on expected type or typical non-risky value
                if any(flag_part in feature_name for flag_part in ['_flag', '_today', 'compliant_flag']):
                    default_val = 0 if feature_name != 'ppe_compliant_flag' else 1 # Default to compliant for PPE
                elif any(num_part in feature_name for num_part in ['_score', 'age', 'hrv', 'spo2', 'temp', 'heat_index']):
                    default_val = np.nan # Let model handle NaNs if appropriate or fill later
                else: # Typically string columns like 'condition'
                    default_val = "Unknown"
                temp_df[feature_name] = default_val
            
            # Coerce types for safety if column was just added or to ensure consistency
            if any(num_part in feature_name for num_part in ['_score', 'age', 'hrv', 'spo2', 'temp', 'heat_index']):
                temp_df[feature_name] = pd.to_numeric(temp_df[feature_name], errors='coerce')
        
        return temp_df.apply(lambda row: self.predict_risk_score(row), axis=1)


class FollowUpPrioritizer:
    def __init__(self):
        self.priority_weights = {
            'base_ai_risk_score_contribution_pct': 0.40, # Increased weight of AI risk
            'critical_vital_alert_points': 40,      # Increased
            'pending_urgent_task_points': 30,       # Increased  
            'acute_condition_severity_points': 25,  # Increased
            'contextual_hazard_points': 15,         
            'task_overdue_factor_per_day': 1.0,     # Increased impact of overdue tasks
            'poor_adherence_points': 15,            # Increased
            'observed_fatigue_points': 12           # For CHW observed patient fatigue
        }
        logger.info("Simulated FollowUpPrioritizer initialized.")

    def _has_active_critical_vitals_alert(self, features: pd.Series) -> bool:
        if pd.notna(features.get('min_spo2_pct')) and features['min_spo2_pct'] < app_config.ALERT_SPO2_CRITICAL_LOW_PCT: return True
        temp_val = features.get('vital_signs_temperature_celsius', features.get('max_skin_temp_celsius'))
        if pd.notna(temp_val) and temp_val >= app_config.ALERT_BODY_TEMP_HIGH_FEVER_C: return True
        # Ensure fall_detected_today is treated as a flag (0 or 1)
        if pd.notna(features.get('fall_detected_today')) and int(features.get('fall_detected_today', 0)) > 0: return True
        return False

    def _is_pending_urgent_task(self, features: pd.Series) -> bool:
        # Check for pending referrals for key conditions
        if str(features.get('referral_status', '')).lower() == 'pending':
            condition_str = str(features.get('condition', '')).lower()
            if any(key_cond.lower() in condition_str for key_cond in app_config.KEY_CONDITIONS_FOR_ACTION):
                return True
        # Add check for explicit 'worker_task_priority' if such a field exists (e.g., manually set by supervisor)
        if str(features.get('worker_task_priority', 'Normal')).lower() == 'urgent':
            return True
        return False
        
    def _has_acute_condition_severity(self, features: pd.Series) -> bool:
        condition_str = str(features.get('condition','')).lower()
        # Example: Pneumonia with concerning SpO2 (not necessarily critical, but warning)
        if "pneumonia" in condition_str and \
           pd.notna(features.get('min_spo2_pct')) and \
           features.get('min_spo2_pct') < app_config.ALERT_SPO2_WARNING_LOW_PCT:
            return True
        # Conditions that are inherently severe
        if any(crit_cond.lower() in condition_str for crit_cond in ["sepsis", "severe dehydration", "heat stroke"]):
            return True
        return False

    def _contextual_hazard_present(self, features: pd.Series) -> bool:
        if pd.notna(features.get('ambient_heat_index_c')) and features['ambient_heat_index_c'] >= app_config.ALERT_AMBIENT_HEAT_INDEX_DANGER_C: return True
        # Add other environmental hazards if data available (e.g., CO2, PM2.5 from PED sensors if applicable)
        # if pd.notna(features.get('ambient_co2_ppm')) and features['ambient_co2_ppm'] >= app_config.ALERT_AMBIENT_CO2_VERY_HIGH_PPM: return True
        return False

    def calculate_priority_score(self, features: pd.Series, days_task_overdue: int = 0) -> float:
        if not isinstance(features, pd.Series): return 0.0
        priority_score = 0.0
        
        ai_risk_val = features.get('ai_risk_score', 0.0)
        if pd.notna(ai_risk_val):
            priority_score += float(ai_risk_val) * self.priority_weights['base_ai_risk_score_contribution_pct']
        
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
        
        # Check for 'signs_of_fatigue_observed_flag' from input features
        if pd.notna(features.get('signs_of_fatigue_observed_flag')) and int(features.get('signs_of_fatigue_observed_flag',0)) == 1:
            priority_score += self.priority_weights['observed_fatigue_points']
            
        priority_score += min(days_task_overdue, 60) * self.priority_weights['task_overdue_factor_per_day'] # Cap overdue impact
        
        return float(np.clip(priority_score, 0, 100))

    def generate_followup_priorities(self, data_df: pd.DataFrame) -> pd.Series:
        if not isinstance(data_df, pd.DataFrame) or data_df.empty:
            return pd.Series(dtype='float64')
        
        temp_df = data_df.copy()
        if 'days_task_overdue' not in temp_df.columns:
            temp_df['days_task_overdue'] = 0
        else:
            temp_df['days_task_overdue'] = pd.to_numeric(temp_df['days_task_overdue'], errors='coerce').fillna(0).astype(int)
        
        # Ensure all features needed by helper methods are present
        expected_prio_features = ['min_spo2_pct', 'vital_signs_temperature_celsius', 'max_skin_temp_celsius', 
                                  'fall_detected_today', 'referral_status', 'condition', 
                                  'worker_task_priority', 'ambient_heat_index_c', 
                                  'medication_adherence_self_report', 'signs_of_fatigue_observed_flag', 'ai_risk_score']
        for feat in expected_prio_features:
            if feat not in temp_df.columns:
                # Add with a neutral/default that won't trigger points if missing
                temp_df[feat] = np.nan if 'pct' in feat or 'temp' in feat or 'score' in feat or 'index' in feat else \
                                0 if 'flag' in feat or 'today' in feat else \
                                "Unknown"
        
        return temp_df.apply(
            lambda row: self.calculate_priority_score(row, row.get('days_task_overdue', 0)),
            axis=1
        )


class SupplyForecastingModel:
    """ Simulates an AI-driven supply forecasting model with seasonality and trend. """
    def __init__(self):
        self.item_params: Dict[str, Dict[str, Any]] = {}
        # Default base coefficients (flat seasonality)
        base_coeffs = np.array([1.0] * 12) 
        
        # Initialize parameters for key drugs from app_config
        for item_key_substring in app_config.KEY_DRUG_SUBSTRINGS_SUPPLY:
            # Use hash for deterministic pseudo-randomness based on item name
            seed_val = abs(hash(item_key_substring)) % (2**32 - 1)
            rng = np.random.RandomState(seed_val) # Each item gets its own RNG stream
            
            # Generate somewhat plausible seasonality and trend
            self.item_params[item_key_substring] = {
                "coeffs": (base_coeffs * rng.uniform(0.8, 1.2, 12)).round(3).tolist(), # Monthly seasonality factors
                "trend": rng.uniform(0.0005, 0.003), # Small positive daily trend component
                "noise_std": rng.uniform(0.03, 0.08) # Reduced noise for more stable forecasts
            }
        
        # Example: Specific override for "ACT" (Artemisinin-based Combination Therapy) if it's a key drug
        act_key = next((s for s in app_config.KEY_DRUG_SUBSTRINGS_SUPPLY if "act" in s.lower()), None)
        if act_key and act_key in self.item_params:
            # Simulate malaria seasonality (e.g., higher in certain months)
            self.item_params[act_key]["coeffs"] = [0.7,0.7,0.8,1.0,1.3,1.5,1.6,1.4,1.1,0.9,0.8,0.7] # Example seasonality
            self.item_params[act_key]["trend"] = 0.0015 # Modest trend for ACT

        logger.info(f"Simulated AI SupplyForecastingModel initialized with params for {len(self.item_params)} key drug groups.")

    def _get_item_params(self, item_name: str) -> Dict[str, Any]:
        item_name_lower = item_name.lower()
        for key_substring, params in self.item_params.items():
            if key_substring.lower() in item_name_lower:
                return params
        # Generic fallback if no specific substring match
        logger.debug(f"No specific params for '{item_name}', using generic fallback supply forecast params.")
        return {"coeffs": [1.0]*12, "trend": 0.0001, "noise_std": 0.05}

    def _predict_daily_consumption_ai(self, base_avg_daily_consumption: float, item_name: str, forecast_date: pd.Timestamp, days_since_forecast_start: int) -> float:
        if pd.isna(base_avg_daily_consumption) or base_avg_daily_consumption <= 0:
            return 0.00001 # Avoid division by zero, ensure minimal consumption
        
        params = self._get_item_params(item_name)
        
        # Monthly seasonality (forecast_date.month is 1-indexed)
        month_index = forecast_date.month - 1
        seasonality_factor = params["coeffs"][month_index]
        
        # Daily trend component (assuming 'trend' in params is a small daily multiplier effect)
        trend_effect = (1 + params["trend"]) ** days_since_forecast_start
        
        # Random noise for simulation (can be seeded for testing if needed)
        # For actual deployment, this would be part of model uncertainty.
        noise_factor = np.random.normal(1.0, params["noise_std"])
        
        predicted_consumption = base_avg_daily_consumption * seasonality_factor * trend_effect * noise_factor
        return max(0.00001, predicted_consumption) # Ensure non-negative consumption

    def forecast_supply_levels_advanced(self, current_supply_levels_df: pd.DataFrame, forecast_days_out: int = 30, item_filter_list: Optional[List[str]] = None) -> pd.DataFrame:
        output_cols = ['item', 'forecast_date', 'forecasted_stock_level', 
                       'forecasted_days_of_supply', 'predicted_daily_consumption', 
                       'estimated_stockout_date_ai']

        if not isinstance(current_supply_levels_df, pd.DataFrame) or current_supply_levels_df.empty:
            logger.warning("AI Supply Forecast: Input DataFrame is empty.")
            return pd.DataFrame(columns=output_cols)

        required_input_cols = ['item', 'current_stock', 'avg_daily_consumption_historical', 'last_stock_update_date']
        missing_cols = [col for col in required_input_cols if col not in current_supply_levels_df.columns]
        if missing_cols:
            logger.error(f"AI Supply Forecast: Input DataFrame missing required columns: {missing_cols}")
            return pd.DataFrame(columns=output_cols)

        df_proc = current_supply_levels_df.copy()
        if item_filter_list:
            df_proc = df_proc[df_proc['item'].isin(item_filter_list)]
        if df_proc.empty:
            logger.info("AI Supply Forecast: No items to forecast after filtering.")
            return pd.DataFrame(columns=output_cols)
        
        df_proc['last_stock_update_date'] = pd.to_datetime(df_proc['last_stock_update_date'], errors='coerce')
        df_proc.dropna(subset=['last_stock_update_date', 'item'], inplace=True) # Critical for forecast start

        all_forecasts = []
        for _, row in df_proc.iterrows():
            item_name = str(row['item'])
            current_stock = float(row.get('current_stock', 0.0))
            base_consumption = float(row.get('avg_daily_consumption_historical', 0.00001))
            last_update_date = row['last_stock_update_date']

            if pd.isna(current_stock) or current_stock < 0: current_stock = 0.0
            if pd.isna(base_consumption) or base_consumption <= 0: base_consumption = 0.00001

            running_stock = current_stock
            stockout_date_item = pd.NaT
            
            item_daily_forecasts = []
            for day_offset in range(forecast_days_out):
                fc_date = last_update_date + pd.Timedelta(days=day_offset + 1)
                pred_consumption = self._predict_daily_consumption_ai(base_consumption, item_name, fc_date, day_offset + 1)
                
                stock_before_use = running_stock
                running_stock = max(0, running_stock - pred_consumption)
                days_supply_eod = (running_stock / pred_consumption) if pred_consumption > 1e-6 else (np.inf if running_stock > 0 else 0)

                if pd.isna(stockout_date_item) and stock_before_use > 0 and running_stock <= 0:
                    # Estimate fractional day of stockout
                    fraction_day = (stock_before_use / pred_consumption) if pred_consumption > 1e-6 else 0.0
                    stockout_date_item = last_update_date + pd.Timedelta(days=day_offset + fraction_day)

                item_daily_forecasts.append({
                    'item': item_name, 'forecast_date': fc_date, 
                    'forecasted_stock_level': running_stock, 
                    'forecasted_days_of_supply': days_supply_eod, 
                    'predicted_daily_consumption': pred_consumption,
                    'estimated_stockout_date_ai': stockout_date_item # Will be NaT until stockout
                })
            
            # If still no stockout date after forecast period, assign the one calculated at the end
            if pd.isna(stockout_date_item) and current_stock > 0 and item_daily_forecasts:
                # Re-estimate based on average predicted consumption over the forecast window if still no stockout
                avg_pred_consumption_period = pd.Series([d['predicted_daily_consumption'] for d in item_daily_forecasts]).mean()
                if avg_pred_consumption_period > 1e-6:
                    days_to_so_from_start = current_stock / avg_pred_consumption_period
                    final_est_so_date = last_update_date + pd.to_timedelta(days_to_so_from_start, unit='D')
                    for record in item_daily_forecasts:
                        if pd.isna(record['estimated_stockout_date_ai']):
                             record['estimated_stockout_date_ai'] = final_est_so_date # Assign to all records for this item if it was NaT
            
            all_forecasts.extend(item_daily_forecasts)
        
        if not all_forecasts: return pd.DataFrame(columns=output_cols)
        
        final_df = pd.DataFrame(all_forecasts)
        # Ensure stockout date is datetime
        if 'estimated_stockout_date_ai' in final_df.columns:
            final_df['estimated_stockout_date_ai'] = pd.to_datetime(final_df['estimated_stockout_date_ai'], errors='coerce')
        return final_df


# --- Central AI Application Function ---
def apply_ai_models(
    health_df: pd.DataFrame,
    current_supply_status_df: Optional[pd.DataFrame] = None, 
    source_context: str = "AIModelApplication" # More generic context
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    logger.info(f"({source_context}) Applying AI models to health data (rows: {len(health_df) if health_df is not None else 0}).")
    
    if not isinstance(health_df, pd.DataFrame): # Robust check
        logger.error(f"({source_context}) Input health_df to apply_ai_models is not a DataFrame. Type: {type(health_df)}")
        # Define expected schema for empty df for consistent returns
        base_cols = ['ai_risk_score', 'ai_followup_priority_score']
        return pd.DataFrame(columns=base_cols), None

    if health_df.empty:
        logger.warning(f"({source_context}) Input health_df is empty. Returning empty enriched DataFrame.")
        # Return DataFrame with expected AI columns but no data
        base_cols = health_df.columns.tolist() if hasattr(health_df, 'columns') else []
        ai_cols_to_add = ['ai_risk_score', 'ai_followup_priority_score']
        final_cols = list(set(base_cols + ai_cols_to_add))
        return pd.DataFrame(columns=final_cols), None

    df_enriched = health_df.copy() # Work on a copy

    # 1. Risk Scoring
    risk_model = RiskPredictionModel()
    df_enriched['ai_risk_score'] = risk_model.predict_bulk_risk_scores(df_enriched)
    logger.debug(f"({source_context}) Applied AI Risk Scoring.")

    # 2. Follow-up/Task Prioritization
    priority_model = FollowUpPrioritizer()
    # Ensure 'days_task_overdue' exists for the prioritizer model
    if 'days_task_overdue' not in df_enriched.columns:
        df_enriched['days_task_overdue'] = 0 # Default to 0 if not present
    else: # Ensure it's numeric and filled
        df_enriched['days_task_overdue'] = pd.to_numeric(df_enriched['days_task_overdue'], errors='coerce').fillna(0).astype(int)
    df_enriched['ai_followup_priority_score'] = priority_model.generate_followup_priorities(df_enriched)
    logger.debug(f"({source_context}) Applied AI Follow-up/Task Prioritization.")
    
    # 3. AI-Simulated Supply Forecasting (Optional)
    supply_forecast_df: Optional[pd.DataFrame] = None
    if isinstance(current_supply_status_df, pd.DataFrame) and not current_supply_status_df.empty:
        required_supply_cols = ['item', 'current_stock', 'avg_daily_consumption_historical', 'last_stock_update_date']
        if all(col in current_supply_status_df.columns for col in required_supply_cols):
            supply_model_ai = SupplyForecastingModel()
            supply_forecast_df = supply_model_ai.forecast_supply_levels_advanced(
                current_supply_levels_df=current_supply_status_df,
                forecast_days_out=app_config.LOW_SUPPLY_DAYS_REMAINING * 2 # Forecast for twice the "low supply" period
            )
            logger.debug(f"({source_context}) Generated AI-simulated supply forecast (records: {len(supply_forecast_df) if supply_forecast_df is not None else 0}).")
        else:
            logger.warning(f"({source_context}) 'current_supply_status_df' missing required columns for AI supply forecast. Skipped.")
    else:
        logger.debug(f"({source_context}) No 'current_supply_status_df' provided or empty; AI supply forecast skipped.")

    logger.info(f"({source_context}) AI model applications complete. Enriched DF shape: {df_enriched.shape}")
    return df_enriched, supply_forecast_df
