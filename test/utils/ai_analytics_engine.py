# sentinel_project_root/test/utils/ai_analytics_engine.py
# Redesigned for "Sentinel Health Co-Pilot" - LMIC Edge-First System
# This module simulates the core AI/Analytics logic that would run primarily on:
#   - Personal Edge Devices (PEDs) using lightweight models (e.g., TinyML, TensorFlow Lite).
#   - Supervisor Hubs for team-level aggregation.
#   - Facility Nodes/Cloud for more complex model training and population analytics.
# The Python classes here serve as reference implementations, for backend simulation,
# and for generating training data or baseline logic for edge models.

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple # Ensure all used types from typing are here
import logging
from config import app_config # Uses the fully refactored app_config

logger = logging.getLogger(__name__)

class RiskPredictionModel:
    """
    SIMULATES a pre-trained patient/worker risk prediction model, adaptable for Edge deployment.
    Uses rule-based logic with weights and factors for core features.
    Actual Edge model would be optimized (e.g., quantized TFLite model derived from this logic or richer data).
    """
    def __init__(self):
        # Factors based on new app_config thresholds and lean data inputs
        self.base_risk_factors = {
            'age': { # Demographics
                'weight': 0.5, 'threshold_high': app_config.RISK_SCORE_MODERATE_THRESHOLD, 'factor_high': 10, # Older age (contextualized)
                'threshold_low': 18, 'factor_low': -2 # Very young might have specific vulnerabilities (not neg risk)
            },
            'min_spo2_pct': { # Sensor stream
                'weight': 2.5, # Increased weight for critical vital
                'threshold_low': app_config.ALERT_SPO2_CRITICAL_LOW_PCT, 'factor_low': 30,
                'mid_threshold_low': app_config.ALERT_SPO2_WARNING_LOW_PCT, 'factor_mid_low': 15
            },
            'vital_signs_temperature_celsius': { # Sensor stream / manual
                'weight': 2.0,
                'threshold_high': app_config.ALERT_BODY_TEMP_FEVER_C, 'factor_high': 15,
                'super_high_threshold': app_config.ALERT_BODY_TEMP_HIGH_FEVER_C, 'factor_super_high': 25
            },
            'max_skin_temp_celsius': { # Alternative for temp if body temp not available
                'weight': 1.8,
                'threshold_high': app_config.HEAT_STRESS_RISK_BODY_TEMP_C, 'factor_high': 10,
                'super_high_threshold': app_config.ALERT_BODY_TEMP_HIGH_FEVER_C - 0.5, 'factor_super_high': 20 # Skin temp slightly less direct
            },
            'stress_level_score': { # Psychometric input or derived from HRV
                'weight': 0.8, 'threshold_high': app_config.FATIGUE_INDEX_MODERATE_THRESHOLD, 'factor_high': 10,
                'super_high_threshold': app_config.FATIGUE_INDEX_HIGH_THRESHOLD, 'factor_super_high': 15
            },
            'hrv_rmssd_ms': { # Sensor stream (if available and processed into a numeric index like RMSSD)
                'weight': 1.2, 'threshold_low': app_config.STRESS_HRV_LOW_THRESHOLD_MS, 'factor_low': 15 # Low HRV indicates stress/fatigue
            },
            'tb_contact_traced': { # Contextual / CHW Input (Less for worker self-risk)
                'weight': 1.0, 'is_flag': True, 'flag_value': 1, 'factor_true': 12 # Example: for patient risk
            },
            'fall_detected_today': { # Sensor stream
                'weight': 2.0, 'is_flag': True, 'flag_value': 1, 'factor_true': 20
            },
            'ambient_heat_index_c': { # Ambient sensor input
                 'weight': 0.7, 'threshold_high': app_config.ALERT_AMBIENT_HEAT_INDEX_RISK_C, 'factor_high': 8,
                 'super_high_threshold': app_config.ALERT_AMBIENT_HEAT_INDEX_DANGER_C, 'factor_super_high': 15
            },
            'ppe_compliant_flag': { # Job Context / Input. Risk if 0 (non-compliant).
                'weight': 1.0, 'is_flag': True, 'flag_value': 0, 'factor_true': 10 
            }
        }
        # More focused, actionable conditions from the new config
        self.condition_base_scores = {cond: 25 for cond in app_config.KEY_CONDITIONS_FOR_ACTION} # Assign a base score
        self.condition_base_scores.update({ # Refine scores for specific conditions if needed
            "Sepsis": 40, "Severe Dehydration": 35, "Heat Stroke": 38,
            "TB": 30, "HIV-Positive": 22, "Pneumonia": 28, "Malaria": 20,
            "Wellness Visit": -10, "Follow-up Health": -5 # For patient context
        })
        # Chronic condition flag contribution
        self.CHRONIC_CONDITION_FLAG_RISK_POINTS = 15

        logger.info("Simulated RiskPredictionModel (Edge Optimized Logic) initialized.")

    def _get_condition_base_score(self, condition_str: Optional[str]) -> float:
        if pd.isna(condition_str) or str(condition_str).lower() in ["unknown", "none", "n/a", ""]: return 0.0
        base_score = 0.0
        condition_input_lower = str(condition_str).lower()
        for known_cond, score_val in self.condition_base_scores.items():
            if known_cond.lower() in condition_input_lower: # Direct match or partial for flexibility
                base_score = max(base_score, score_val) 
        return base_score

    def predict_risk_score(self, features: pd.Series) -> float:
        """Predicts risk score for a single entity (patient or worker) based on features."""
        calculated_risk = 0.0
        
        calculated_risk += self._get_condition_base_score(features.get('condition'))

        if features.get('chronic_condition_flag') == 1 or str(features.get('chronic_condition_flag')).lower() == 'yes':
            calculated_risk += self.CHRONIC_CONDITION_FLAG_RISK_POINTS

        for feature_key, params in self.base_risk_factors.items():
            if feature_key in features and pd.notna(features[feature_key]):
                value = features[feature_key]; weight = params.get('weight', 1.0)
                if params.get('is_flag'):
                    if feature_key == 'ppe_compliant_flag' and value == params.get('flag_value', 0): # is_flag & value is 0
                        calculated_risk += params.get('factor_true', 0) * weight
                    elif feature_key != 'ppe_compliant_flag' and value == params.get('flag_value', 1): # Standard flags are 1 for true
                        calculated_risk += params.get('factor_true', 0) * weight
                else: 
                    if 'super_high_threshold' in params and value >= params['super_high_threshold']: calculated_risk += params.get('factor_super_high', 0) * weight
                    elif 'threshold_high' in params and value >= params['threshold_high']: calculated_risk += params.get('factor_high', 0) * weight
                    
                    if 'threshold_low' in params and value < params['threshold_low']: calculated_risk += params.get('factor_low', 0) * weight
                    elif 'mid_threshold_low' in params and value < params['mid_threshold_low']: calculated_risk += params.get('factor_mid_low', 0) * weight
        
        adherence = features.get('medication_adherence_self_report', "Unknown")
        if str(adherence).lower() == 'poor': calculated_risk += 10
        elif str(adherence).lower() == 'fair': calculated_risk += 5
        
        if pd.notna(features.get('rapid_psychometric_distress_score')):
            calculated_risk += features.get('rapid_psychometric_distress_score',0.0) * 1.5
        if pd.notna(features.get('signs_of_fatigue_observed_flag')) and features.get('signs_of_fatigue_observed_flag') == 1 :
            calculated_risk += 10 

        return float(np.clip(calculated_risk, 0, 100))

    def predict_bulk_risk_scores(self, data_df: pd.DataFrame) -> pd.Series:
        if data_df.empty: return pd.Series(dtype='float64')
        
        temp_df_for_pred = data_df.copy()
        all_expected_model_features = list(self.base_risk_factors.keys()) + [
            'condition', 'chronic_condition_flag', 'medication_adherence_self_report',
            'rapid_psychometric_distress_score', 'signs_of_fatigue_observed_flag'
        ]
        for feature_name_check in all_expected_model_features:
            if feature_name_check not in temp_df_for_pred.columns:
                # Assign default based on likely type/role
                if any(flag_part in feature_name_check for flag_part in ['_flag', '_today', 'compliant_flag']): default_val_feat = 0
                elif any(score_part in feature_name_check for score_part in ['_score', 'age', 'hrv_rmssd_ms', 'min_spo2_pct', 'temperature', 'heat_index']): default_val_feat = np.nan
                else: default_val_feat = "Unknown" # For string-like (condition, adherence)
                temp_df_for_pred[feature_name_check] = default_val_feat
                # Ensure numeric columns are numeric after adding default (if default was NaN)
                if default_val_feat is np.nan and any(num_part in feature_name_check for num_part in ['score', 'age', 'hrv', 'spo2', 'temp', 'heat_index']):
                     temp_df_for_pred[feature_name_check] = pd.to_numeric(temp_df_for_pred[feature_name_check], errors='coerce')

        return temp_df_for_pred.apply(lambda row_data: self.predict_risk_score(row_data), axis=1)


class FollowUpPrioritizer:
    def __init__(self):
        self.priority_weights = {
            'base_ai_risk_score_contribution_pct': 0.35, 
            'critical_vital_alert_points': 35,        
            'pending_urgent_task_points': 25,         
            'acute_condition_severity_points': 20,    
            'contextual_hazard_points': 15,           
            'task_overdue_factor_per_day': 0.5,       
            'poor_adherence_points': 10,
            'observed_fatigue_points': 12
        }
        logger.info("Simulated FollowUpPrioritizer (Edge Optimized Logic) initialized.")

    def _has_active_critical_vitals_alert(self, features: pd.Series) -> bool:
        if pd.notna(features.get('min_spo2_pct')) and features['min_spo2_pct'] < app_config.ALERT_SPO2_CRITICAL_LOW_PCT: return True
        temp_val_prio = features.get('vital_signs_temperature_celsius', features.get('max_skin_temp_celsius', np.nan))
        if pd.notna(temp_val_prio) and temp_val_prio >= app_config.ALERT_BODY_TEMP_HIGH_FEVER_C: return True
        if pd.notna(features.get('fall_detected_today')) and features['fall_detected_today'] > 0: return True
        return False

    def _is_pending_urgent_task(self, features: pd.Series) -> bool:
        if str(features.get('referral_status', 'Unknown')).lower() == 'pending':
            if any(cond_key.lower() in str(features.get('condition', '')).lower() for cond_key in app_config.KEY_CONDITIONS_FOR_ACTION):
                return True
        if str(features.get('worker_task_priority', 'Normal')).lower() == 'urgent': return True
        return False
        
    def _has_acute_condition_severity(self, features: pd.Series) -> bool:
        condition_str_prio = str(features.get('condition','')).lower()
        if ("pneumonia" in condition_str_prio and pd.notna(features.get('min_spo2_pct')) and features.get('min_spo2_pct') < app_config.ALERT_SPO2_WARNING_LOW_PCT): return True
        if any(crit_cond_prio.lower() in condition_str_prio for crit_cond_prio in ["sepsis", "severe dehydration", "heat stroke"]): return True
        return False

    def _contextual_hazard_present(self, features: pd.Series) -> bool:
        if pd.notna(features.get('ambient_heat_index_c')) and features['ambient_heat_index_c'] >= app_config.ALERT_AMBIENT_HEAT_INDEX_DANGER_C: return True
        if pd.notna(features.get('ambient_co2_ppm')) and features['ambient_co2_ppm'] >= app_config.ALERT_AMBIENT_CO2_VERY_HIGH_PPM: return True
        if pd.notna(features.get('ambient_pm25_ugm3')) and features['ambient_pm25_ugm3'] >= app_config.ALERT_AMBIENT_PM25_VERY_HIGH_UGM3: return True
        return False

    def calculate_priority_score(self, features: pd.Series, days_task_overdue: int = 0) -> float:
        priority_score = 0.0
        ai_risk_prio = features.get('ai_risk_score', 0.0) # Ensure it has a default
        if pd.notna(ai_risk_prio): priority_score += ai_risk_prio * self.priority_weights['base_ai_risk_score_contribution_pct']
        if self._has_active_critical_vitals_alert(features): priority_score += self.priority_weights['critical_vital_alert_points']
        if self._is_pending_urgent_task(features): priority_score += self.priority_weights['pending_urgent_task_points']
        if self._has_acute_condition_severity(features): priority_score += self.priority_weights['acute_condition_severity_points']
        if self._contextual_hazard_present(features): priority_score += self.priority_weights['contextual_hazard_points']
        if str(features.get('medication_adherence_self_report', 'Unknown')).lower() == 'poor': priority_score += self.priority_weights['poor_adherence_points']
        if features.get('signs_of_fatigue_observed_flag') == 1 : priority_score += self.priority_weights['observed_fatigue_points']
        priority_score += min(days_task_overdue, 60) * self.priority_weights['task_overdue_factor_per_day'] # Cap overdue bonus at 30 points for 60 days
        return float(np.clip(priority_score, 0, 100))

    def generate_followup_priorities(self, data_df: pd.DataFrame) -> pd.Series:
        if data_df.empty: return pd.Series(dtype='float64')
        temp_df_followup_prio = data_df.copy()
        if 'days_task_overdue' not in temp_df_followup_prio.columns: temp_df_followup_prio['days_task_overdue'] = 0 # Default
        else: temp_df_followup_prio['days_task_overdue'] = pd.to_numeric(temp_df_followup_prio['days_task_overdue'], errors='coerce').fillna(0)
        return temp_df_followup_prio.apply(lambda row_data: self.calculate_priority_score(row_data, int(row_data.get('days_task_overdue',0))), axis=1)

class SupplyForecastingModel:
    def __init__(self):
        self.item_params: Dict[str, Dict[str, Any]] = {}
        base_coeffs = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        for item_key in app_config.KEY_DRUG_SUBSTRINGS_SUPPLY:
            # More deterministic random seed for consistent testability if params generated here
            # Though ideally these params are loaded from a config or trained.
            # For simulation: slightly vary each item's base parameters.
            # Use hash of item_key for some deterministic pseudo-randomness based on name.
            seed_val = abs(hash(item_key)) % (2**32 -1)
            rng = np.random.RandomState(seed_val)

            self.item_params[item_key] = {
                "coeffs": (base_coeffs * rng.uniform(0.85, 1.15, 12)).round(3).tolist(),
                "trend": rng.uniform(0.0001, 0.005), # Smaller, more realistic daily trend component
                "noise_std": rng.uniform(0.02, 0.10) # Reduced noise for more stable sim
            }
        # Example explicit override for a key item (ACT Tablets for malaria season)
        if "ACT Tablets" in self.item_params:
            self.item_params["ACT Tablets"]["coeffs"] = [0.6,0.6,0.7,0.9,1.2,1.4,1.5,1.3,1.0,0.8,0.7,0.6]
            self.item_params["ACT Tablets"]["trend"] = 0.002
        logger.info(f"Simulated AI SupplyForecastingModel initialized with {len(self.item_params)} item parameter sets.")

    def _get_item_params(self, item_name: str) -> Dict[str, Any]:
        for key_substring_param in self.item_params.keys():
            if key_substring_param.lower() in item_name.lower():
                return self.item_params[key_substring_param]
        # Generic fallback parameters if no specific substring match is found
        return {"coeffs": [1.0]*12, "trend": 0.0001, "noise_std": 0.05}

    def _predict_daily_consumption_ai(self, base_avg_daily_consumption: float, item_name: str, forecast_date: pd.Timestamp, days_since_forecast_start: int) -> float:
        if pd.isna(base_avg_daily_consumption) or base_avg_daily_consumption <= 0: return 0.00001 # Avoid zero/negative
        
        item_fc_params = self._get_item_params(item_name)
        monthly_seasonality_factor = item_fc_params["coeffs"][forecast_date.month - 1] # forecast_date.month is 1-12
        # Trend calculated daily, compounded. trend is per 30 days, so divide by 30 for daily.
        daily_trend_factor = item_fc_params["trend"] / 30.0
        trend_effect = (1 + daily_trend_factor)**days_since_forecast_start
        
        # Seed random noise with date and item for more deterministic "randomness" in tests, if desired
        # For simulation, a true random component is fine.
        # random_noise_seed = abs(hash(item_name + forecast_date.strftime('%Y%m%d'))) % (2**32 -1)
        # noise_rng = np.random.RandomState(random_noise_seed)
        # random_noise_factor = noise_rng.normal(1.0, item_fc_params["noise_std"])
        random_noise_factor = np.random.normal(1.0, item_fc_params["noise_std"])

        predicted_consumption = base_avg_daily_consumption * monthly_seasonality_factor * trend_effect * random_noise_factor
        return max(0.00001, predicted_consumption) # Ensure consumption is not zero or negative for calculations

    def forecast_supply_levels_advanced(
        self, current_supply_levels_df: pd.DataFrame,
        forecast_days_out: int = 7, # Default to shorter, field-relevant horizon
        item_filter_list: Optional[List[str]] = None
    ) -> pd.DataFrame:
        logger.info(f"AI-simulated supply forecast. Horizon: {forecast_days_out} days. Input items: {len(current_supply_levels_df) if current_supply_levels_df is not None else 0}")
        output_cols_fc_ai = ['item', 'forecast_date', 'forecasted_stock_level', 'forecasted_days_of_supply', 'predicted_daily_consumption', 'estimated_stockout_date_ai']
        
        if not isinstance(current_supply_levels_df, pd.DataFrame) or current_supply_levels_df.empty:
            logger.warning("AI Supply Forecast: Input DataFrame of current supply levels is empty.")
            return pd.DataFrame(columns=output_cols_fc_ai)
        
        required_input_cols_fc = ['item', 'current_stock', 'avg_daily_consumption_historical', 'last_stock_update_date']
        if not all(col in current_supply_levels_df.columns for col in required_input_cols_fc):
            missing_cols = [col for col in required_input_cols_fc if col not in current_supply_levels_df.columns]
            logger.error(f"AI Supply Forecast: Input DataFrame missing required columns: {missing_cols}")
            return pd.DataFrame(columns=output_cols_fc_ai)

        df_to_process_fc = current_supply_levels_df.copy()
        if item_filter_list: df_to_process_fc = df_to_process_fc[df_to_process_fc['item'].isin(item_filter_list)]
        if df_to_process_fc.empty: logger.info("AI Supply Forecast: No items to forecast after filtering."); return pd.DataFrame(columns=output_cols_fc_ai)

        df_to_process_fc['last_stock_update_date'] = pd.to_datetime(df_to_process_fc['last_stock_update_date'], errors='coerce')
        df_to_process_fc.dropna(subset=['last_stock_update_date', 'item'], inplace=True) # Need valid date and item

        all_item_forecasts_list = []
        for _, item_data_row in df_to_process_fc.iterrows():
            item_name_fc = str(item_data_row['item'])
            current_stock_fc = float(item_data_row.get('current_stock', 0.0))
            base_daily_cons_fc = float(item_data_row.get('avg_daily_consumption_historical', 0.00001)) # Base historical rate
            last_update_date_fc = item_data_row['last_stock_update_date']

            if pd.isna(current_stock_fc) or current_stock_fc < 0: current_stock_fc = 0.0
            if pd.isna(base_daily_cons_fc) or base_daily_cons_fc <= 0: base_daily_cons_fc = 0.00001 # Min consumption to avoid DivByZero

            running_stock_level_fc = current_stock_fc
            estimated_stockout_date_for_item = pd.NaT
            
            forecasts_for_this_item = [] # To calculate avg predicted use for final stockout estimate if needed

            for day_offset_val in range(forecast_days_out):
                fc_date_curr = last_update_date_fc + pd.Timedelta(days=day_offset_val + 1)
                predicted_use_today_fc = self._predict_daily_consumption_ai(base_daily_cons_fc, item_name_fc, fc_date_curr, day_offset_val + 1)
                
                stock_before_today_use = running_stock_level_fc
                running_stock_level_fc = max(0, running_stock_level_fc - predicted_use_today_fc)
                days_of_supply_at_eod = (running_stock_level_fc / predicted_use_today_fc) if predicted_use_today_fc > 0.000001 else (np.inf if running_stock_level_fc > 0 else 0)

                if pd.isna(estimated_stockout_date_for_item) and stock_before_today_use > 0 and running_stock_level_fc <= 0: # Just stocked out
                    fraction_of_day_at_stockout = (stock_before_today_use / predicted_use_today_fc) if predicted_use_today_fc > 0.000001 else 0.0
                    estimated_stockout_date_for_item = last_update_date_fc + pd.Timedelta(days=day_offset_val + fraction_of_day_at_stockout)
                
                current_day_fc_record = {
                    'item': item_name_fc, 'forecast_date': fc_date_curr,
                    'forecasted_stock_level': running_stock_level_fc, 'forecasted_days_of_supply': days_of_supply_at_eod,
                    'predicted_daily_consumption': predicted_use_today_fc, 
                    'estimated_stockout_date_ai': estimated_stockout_date_for_item # Will be NaT until determined
                }
                all_item_forecasts_list.append(current_day_fc_record)
                forecasts_for_this_item.append(current_day_fc_record)
            
            # If not stocked out within forecast_days_out, estimate final stockout date
            if pd.isna(estimated_stockout_date_for_item) and current_stock_fc > 0 and forecasts_for_this_item:
                avg_predicted_use_in_fc_period = pd.Series([f_rec['predicted_daily_consumption'] for f_rec in forecasts_for_this_item]).mean()
                if avg_predicted_use_in_fc_period > 0.000001:
                    days_to_final_stockout_est = current_stock_fc / avg_predicted_use_in_fc_period
                    final_est_stockout_date = last_update_date_fc + pd.to_timedelta(days_to_final_stockout_est, unit='D')
                    # Update all NaT stockout dates for this item's forecast records
                    for fc_record_update in all_item_forecasts_list:
                        if fc_record_update['item'] == item_name_fc and pd.isna(fc_record_update['estimated_stockout_date_ai']):
                            fc_record_update['estimated_stockout_date_ai'] = final_est_stockout_date
        
        if not all_item_forecasts_list: return pd.DataFrame(columns=output_cols_fc_ai)
        final_forecast_df_ai = pd.DataFrame(all_item_forecasts_list)
        if 'estimated_stockout_date_ai' in final_forecast_df_ai.columns:
             final_forecast_df_ai['estimated_stockout_date_ai'] = pd.to_datetime(final_forecast_df_ai['estimated_stockout_date_ai'], errors='coerce')
        return final_forecast_df_ai


# --- Central AI Application Function ---
def apply_ai_models(
    health_df: pd.DataFrame,
    current_supply_status_df: Optional[pd.DataFrame] = None, # DataFrame with: item, current_stock, avg_daily_consumption_historical, last_stock_update_date
    source_context: str = "FacilityNode/BatchAI"
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    logger.info(f"({source_context}) Applying AI models to health data (rows: {len(health_df) if health_df is not None else 0}).")
    # Ensure essential columns for AI models, otherwise, the models themselves will add defaults.
    # This function orchestrates their application.
    if health_df is None or health_df.empty:
        logger.warning(f"({source_context}) Input health_df to apply_ai_models is empty. Returning empty/None.")
        # Determine expected columns for empty DataFrame based on what AI models add + original.
        cols_orig = list(health_df.columns) if health_df is not None and health_df.columns.any() else []
        cols_ai_added = ['ai_risk_score', 'ai_followup_priority_score']
        final_empty_cols = list(set(cols_orig + cols_ai_added)) # Unique list
        return pd.DataFrame(columns=final_empty_cols), None

    df_enriched_main = health_df.copy()

    # 1. Risk Scoring
    risk_predictor = RiskPredictionModel()
    df_enriched_main['ai_risk_score'] = risk_predictor.predict_bulk_risk_scores(df_enriched_main)
    logger.info(f"({source_context}) Applied AI Risk Scoring. Column 'ai_risk_score' added/updated.")

    # 2. Follow-up/Task Prioritization
    followup_prioritizer = FollowUpPrioritizer()
    # The prioritizer expects 'days_task_overdue'. If not present, its internal logic defaults to 0.
    # It also uses 'ai_risk_score' calculated above.
    if 'days_task_overdue' not in df_enriched_main.columns: # Ensure column exists if downstream processing assumes it
        logger.debug(f"({source_context}) 'days_task_overdue' not in input for prioritization, will default to 0 internally.")
        df_enriched_main['days_task_overdue'] = 0 # Make explicit default for clarity if other logic depends on it
    df_enriched_main['ai_followup_priority_score'] = followup_prioritizer.generate_followup_priorities(df_enriched_main)
    logger.info(f"({source_context}) Applied AI Follow-up/Task Prioritization. Column 'ai_followup_priority_score' added/updated.")
    
    # 3. AI-Simulated Supply Forecasting (Optional, if input data for it is provided)
    generated_supply_forecast_df: Optional[pd.DataFrame] = None
    if current_supply_status_df is not None and not current_supply_status_df.empty:
        supply_forecaster_ai = SupplyForecastingModel()
        generated_supply_forecast_df = supply_forecaster_ai.forecast_supply_levels_advanced(
            current_supply_levels_df=current_supply_status_df,
            # Using a configuration value for default forecast horizon, e.g., a common review period
            forecast_days_out=app_config.LOW_SUPPLY_DAYS_REMAINING # Example: forecast for 2 weeks (14 days)
        )
        rec_count_supply = len(generated_supply_forecast_df) if generated_supply_forecast_df is not None else 0
        logger.info(f"({source_context}) Generated AI-simulated supply forecast. {rec_count_supply} forecast records created.")
    else:
        logger.info(f"({source_context}) No 'current_supply_status_df' provided or it's empty; AI supply forecast skipped.")

    logger.info(f"({source_context}) All AI model applications complete. Final enriched DataFrame shape: {df_enriched_main.shape}")
    return df_enriched_main, generated_supply_forecast_df
