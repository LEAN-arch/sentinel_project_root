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
from typing import Dict, Any, Optional, List, Tuple # ENSURED Tuple and others are here
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
            'age': { 'weight': 0.5, 'threshold_high': app_config.RISK_SCORE_MODERATE_THRESHOLD, 'factor_high': 10, 'threshold_low': 18, 'factor_low': -2 },
            'min_spo2_pct': { 'weight': 2.5, 'threshold_low': app_config.ALERT_SPO2_CRITICAL_LOW_PCT, 'factor_low': 30, 'mid_threshold_low': app_config.ALERT_SPO2_WARNING_LOW_PCT, 'factor_mid_low': 15 },
            'vital_signs_temperature_celsius': { 'weight': 2.0, 'threshold_high': app_config.ALERT_BODY_TEMP_FEVER_C, 'factor_high': 15, 'super_high_threshold': app_config.ALERT_BODY_TEMP_HIGH_FEVER_C, 'factor_super_high': 25 },
            'max_skin_temp_celsius': { 'weight': 1.8, 'threshold_high': app_config.HEAT_STRESS_RISK_BODY_TEMP_C, 'factor_high': 10, 'super_high_threshold': app_config.ALERT_BODY_TEMP_HIGH_FEVER_C - 0.5, 'factor_super_high': 20 },
            'stress_level_score': { 'weight': 0.8, 'threshold_high': app_config.FATIGUE_INDEX_MODERATE_THRESHOLD, 'factor_high': 10, 'super_high_threshold': app_config.FATIGUE_INDEX_HIGH_THRESHOLD, 'factor_super_high': 15 },
            'hrv_rmssd_ms': { 'weight': 1.2, 'threshold_low': app_config.STRESS_HRV_LOW_THRESHOLD_MS, 'factor_low': 15 },
            'tb_contact_traced': { 'weight': 1.0, 'is_flag': True, 'flag_value': 1, 'factor_true': 12 },
            'fall_detected_today': { 'weight': 2.0, 'is_flag': True, 'flag_value': 1, 'factor_true': 20 },
            'ambient_heat_index_c': { 'weight': 0.7, 'threshold_high': app_config.ALERT_AMBIENT_HEAT_INDEX_RISK_C, 'factor_high': 8, 'super_high_threshold': app_config.ALERT_AMBIENT_HEAT_INDEX_DANGER_C, 'factor_super_high': 15 },
            'ppe_compliant_flag': { 'weight': 1.0, 'is_flag': True, 'flag_value': 0, 'factor_true': 10 } # Risk if flag is 0 (non-compliant)
        }
        self.condition_base_scores = {cond: 25 for cond in app_config.KEY_CONDITIONS_FOR_ACTION}
        self.condition_base_scores.update({
            "Sepsis": 40, "Severe Dehydration": 35, "Heat Stroke": 38, "TB": 30,
            "HIV-Positive": 22, "Pneumonia": 28, "Malaria": 20,
            "Wellness Visit": -10, "Follow-up Health": -5
        })
        self.CHRONIC_CONDITION_FLAG_RISK_POINTS = 15
        logger.info("Simulated RiskPredictionModel (Edge Optimized Logic) initialized.")

    def _get_condition_base_score(self, condition_str: Optional[str]) -> float:
        if pd.isna(condition_str) or str(condition_str).lower() in ["unknown", "none", "n/a", ""]: return 0.0
        base_score = 0.0
        condition_input_lower = str(condition_str).lower()
        for known_cond, score_val in self.condition_base_scores.items():
            if known_cond.lower() in condition_input_lower:
                base_score = max(base_score, score_val)
        return base_score

    def predict_risk_score(self, features: pd.Series) -> float:
        calculated_risk = self._get_condition_base_score(features.get('condition'))

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
                default_val_feat = 0 if any(flag_part in feature_name_check for flag_part in ['_flag', '_today', 'compliant_flag']) else \
                              np.nan if any(score_part in feature_name_check for score_part in ['_score', 'age', 'hrv_rmssd_ms', 'min_spo2_pct', 'temperature_celsius', 'max_skin_temp_celsius', 'heat_index_c']) else \
                              "Unknown"
                temp_df_for_pred[feature_name_check] = default_val_feat
                if default_val_feat is np.nan and any(num_part in feature_name_check for num_part in ['score', 'age', 'hrv', 'spo2', 'temp', 'heat_index']): # Ensure correct dtype for numeric defaults if column was created
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
        ai_risk_prio = features.get('ai_risk_score', 0.0) 
        if pd.notna(ai_risk_prio): priority_score += ai_risk_prio * self.priority_weights['base_ai_risk_score_contribution_pct']
        if self._has_active_critical_vitals_alert(features): priority_score += self.priority_weights['critical_vital_alert_points']
        if self._is_pending_urgent_task(features): priority_score += self.priority_weights['pending_urgent_task_points']
        if self._has_acute_condition_severity(features): priority_score += self.priority_weights['acute_condition_severity_points']
        if self._contextual_hazard_present(features): priority_score += self.priority_weights['contextual_hazard_points']
        if str(features.get('medication_adherence_self_report', 'Unknown')).lower() == 'poor': priority_score += self.priority_weights['poor_adherence_points']
        if features.get('signs_of_fatigue_observed_flag') == 1 : priority_score += self.priority_weights['observed_fatigue_points']
        priority_score += min(days_task_overdue, 60) * self.priority_weights['task_overdue_factor_per_day'] 
        return float(np.clip(priority_score, 0, 100))

    def generate_followup_priorities(self, data_df: pd.DataFrame) -> pd.Series:
        if data_df.empty: return pd.Series(dtype='float64')
        temp_df_followup_prio = data_df.copy()
        if 'days_task_overdue' not in temp_df_followup_prio.columns: temp_df_followup_prio['days_task_overdue'] = 0
        else: temp_df_followup_prio['days_task_overdue'] = pd.to_numeric(temp_df_followup_prio['days_task_overdue'], errors='coerce').fillna(0).astype(int) # Ensure int
        return temp_df_followup_prio.apply(lambda row_data: self.calculate_priority_score(row_data, row_data.get('days_task_overdue',0)), axis=1)


class SupplyForecastingModel:
    def __init__(self):
        self.item_params: Dict[str, Dict[str, Any]] = {}
        base_coeffs = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        for item_key_supply_fc in app_config.KEY_DRUG_SUBSTRINGS_SUPPLY: # Iterate through configured key drugs
            # Use hash of item_key for some deterministic pseudo-randomness based on name for simulation
            seed_val_fc = abs(hash(item_key_supply_fc)) % (2**32 -1)
            rng_fc = np.random.RandomState(seed_val_fc)
            self.item_params[item_key_supply_fc] = {
                "coeffs": (base_coeffs * rng_fc.uniform(0.85, 1.15, 12)).round(3).tolist(), # Monthly seasonality
                "trend": rng_fc.uniform(0.0001, 0.005),  # Smaller, more realistic daily trend
                "noise_std": rng_fc.uniform(0.02, 0.10) # Reduced noise for more stable simulation
            }
        # Specific override for ACT Tablets example
        if "ACT Tablets" in self.item_params:
            self.item_params["ACT Tablets"]["coeffs"] = [0.6,0.6,0.7,0.9,1.2,1.4,1.5,1.3,1.0,0.8,0.7,0.6] # Malaria seasonality
            self.item_params["ACT Tablets"]["trend"] = 0.002 # Slightly higher trend for example
        logger.info(f"Simulated AI SupplyForecastingModel initialized with {len(self.item_params)} item parameter sets for key drugs.")

    def _get_item_params(self, item_name: str) -> Dict[str, Any]:
        for key_substring_param_fc in self.item_params.keys(): # Check against defined item_params
            if key_substring_param_fc.lower() in item_name.lower():
                return self.item_params[key_substring_param_fc]
        # Generic fallback if no specific substring match
        return {"coeffs": [1.0]*12, "trend": 0.0001, "noise_std": 0.05} 

    def _predict_daily_consumption_ai(self, base_avg_daily_consumption: float, item_name: str, forecast_date: pd.Timestamp, days_since_forecast_start: int) -> float:
        if pd.isna(base_avg_daily_consumption) or base_avg_daily_consumption <= 0: return 0.00001 # Minimum consumption
        params_item_curr_fc = self._get_item_params(item_name)
        monthly_seasonality_factor_fc = params_item_curr_fc["coeffs"][forecast_date.month - 1] # .month is 1-12
        daily_trend_component_fc = params_item_curr_fc["trend"] / 30.0 # Assuming trend is per 30 days
        trend_effect_factor_fc = (1 + daily_trend_component_fc)**days_since_forecast_start
        # For simulation, simple random noise. For testing, one might want to seed this.
        random_noise_factor_fc = np.random.normal(1.0, params_item_curr_fc["noise_std"])
        predicted_consumption_final_fc = base_avg_daily_consumption * monthly_seasonality_factor_fc * trend_effect_factor_fc * random_noise_factor_fc
        return max(0.00001, predicted_consumption_final_fc)

    def forecast_supply_levels_advanced(self, current_supply_levels_df: pd.DataFrame, forecast_days_out: int = 7, item_filter_list: Optional[List[str]] = None) -> pd.DataFrame:
        logger.info(f"AI-simulated supply forecast initiated. Horizon: {forecast_days_out} days. Input items: {len(current_supply_levels_df) if current_supply_levels_df is not None else 0}")
        output_cols_supply_fc = ['item', 'forecast_date', 'forecasted_stock_level', 'forecasted_days_of_supply', 'predicted_daily_consumption', 'estimated_stockout_date_ai']
        if not isinstance(current_supply_levels_df, pd.DataFrame) or current_supply_levels_df.empty:
            logger.warning("AI Supply Forecast: Input DataFrame of current supply levels is empty.")
            return pd.DataFrame(columns=output_cols_supply_fc)
        required_cols_supply_fc_input = ['item', 'current_stock', 'avg_daily_consumption_historical', 'last_stock_update_date']
        if not all(col in current_supply_levels_df.columns for col in required_cols_supply_fc_input):
            missing_cols_supply = [col for col in required_cols_supply_fc_input if col not in current_supply_levels_df.columns]
            logger.error(f"AI Supply Forecast: Input DataFrame missing required columns: {missing_cols_supply}")
            return pd.DataFrame(columns=output_cols_supply_fc)

        df_to_process_fc_ai = current_supply_levels_df.copy()
        if item_filter_list: df_to_process_fc_ai = df_to_process_fc_ai[df_to_process_fc_ai['item'].isin(item_filter_list)]
        if df_to_process_fc_ai.empty: logger.info("AI Supply Forecast: No items to forecast after applying item_filter_list."); return pd.DataFrame(columns=output_cols_supply_fc)
        
        df_to_process_fc_ai['last_stock_update_date'] = pd.to_datetime(df_to_process_fc_ai['last_stock_update_date'], errors='coerce')
        df_to_process_fc_ai.dropna(subset=['last_stock_update_date', 'item'], inplace=True) # Critical for forecast start

        all_forecast_records_ai_list = []
        for _, item_row_data_fc_ai in df_to_process_fc_ai.iterrows():
            item_name_curr_fc_ai = str(item_row_data_fc_ai['item'])
            stock_start_curr_fc_ai = float(item_row_data_fc_ai.get('current_stock', 0.0))
            base_daily_cons_curr_fc_ai = float(item_row_data_fc_ai.get('avg_daily_consumption_historical', 0.00001))
            last_update_dt_curr_fc_ai = item_row_data_fc_ai['last_stock_update_date']

            if pd.isna(stock_start_curr_fc_ai) or stock_start_curr_fc_ai < 0: stock_start_curr_fc_ai = 0.0
            if pd.isna(base_daily_cons_curr_fc_ai) or base_daily_cons_curr_fc_ai <=0: base_daily_cons_curr_fc_ai = 0.00001

            running_stock_level_curr_fc = stock_start_curr_fc_ai
            estimated_stockout_date_for_item_ai = pd.NaT
            forecasts_for_this_item_ai_list = []

            for day_offset_curr_fc_val in range(forecast_days_out):
                current_fc_dt_val = last_update_dt_curr_fc_ai + pd.Timedelta(days=day_offset_curr_fc_val + 1)
                predicted_use_today_fc_val = self._predict_daily_consumption_ai(base_daily_cons_curr_fc_ai, item_name_curr_fc_ai, current_fc_dt_val, day_offset_curr_fc_val + 1)
                stock_before_today_use_fc = running_stock_level_curr_fc
                running_stock_level_curr_fc = max(0, running_stock_level_curr_fc - predicted_use_today_fc_val)
                days_supply_at_eod_fc = (running_stock_level_curr_fc / predicted_use_today_fc_val) if predicted_use_today_fc_val > 0.000001 else (np.inf if running_stock_level_curr_fc > 0 else 0)

                if pd.isna(estimated_stockout_date_for_item_ai) and stock_before_today_use_fc > 0 and running_stock_level_curr_fc <= 0:
                    fraction_day_at_stockout_fc = (stock_before_today_use_fc / predicted_use_today_fc_val) if predicted_use_today_fc_val > 0.000001 else 0.0
                    estimated_stockout_date_for_item_ai = last_update_dt_curr_fc_ai + pd.Timedelta(days=day_offset_curr_fc_val + fraction_day_at_stockout_fc)
                
                current_day_fc_record_ai = {'item':item_name_curr_fc_ai, 'forecast_date':current_fc_dt_val, 'forecasted_stock_level':running_stock_level_curr_fc, 'forecasted_days_of_supply':days_supply_at_eod_fc, 'predicted_daily_consumption':predicted_use_today_fc_val, 'estimated_stockout_date_ai':estimated_stockout_date_for_item_ai}
                all_forecast_records_ai_list.append(current_day_fc_record_ai)
                forecasts_for_this_item_ai_list.append(current_day_fc_record_ai)
            
            if pd.isna(estimated_stockout_date_for_item_ai) and stock_start_curr_fc_ai > 0 and forecasts_for_this_item_ai_list:
                avg_pred_use_fc_period_ai = pd.Series([f_rec_prd_ai['predicted_daily_consumption'] for f_rec_prd_ai in forecasts_for_this_item_ai_list]).mean()
                if avg_pred_use_fc_period_ai > 0.000001:
                    days_to_final_so_est_ai = stock_start_curr_fc_ai / avg_pred_use_fc_period_ai
                    final_est_stockout_date_item_ai = last_update_dt_curr_fc_ai + pd.to_timedelta(days_to_final_so_est_ai, unit='D')
                    for rec_upd_ai in all_forecast_records_ai_list:
                        if rec_upd_ai['item'] == item_name_curr_fc_ai and pd.isna(rec_upd_ai['estimated_stockout_date_ai']):
                            rec_upd_ai['estimated_stockout_date_ai'] = final_est_stockout_date_item_ai
        
        if not all_forecast_records_ai_list: return pd.DataFrame(columns=output_cols_fc_final)
        final_forecast_result_df_ai = pd.DataFrame(all_forecast_records_ai_list)
        if 'estimated_stockout_date_ai' in final_forecast_result_df_ai.columns:
             final_forecast_result_df_ai['estimated_stockout_date_ai'] = pd.to_datetime(final_forecast_result_df_ai['estimated_stockout_date_ai'], errors='coerce')
        return final_forecast_result_df_ai


# --- Central AI Application Function ---
def apply_ai_models(
    health_df: pd.DataFrame,
    current_supply_status_df: Optional[pd.DataFrame] = None, 
    source_context: str = "FacilityNode/BatchAI"
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]: # Ensure Tuple and Optional are imported from typing
    logger.info(f"({source_context}) Applying AI models to health data (rows: {len(health_df) if health_df is not None else 0}).")
    
    # Create expected column list for return schema if health_df is empty
    expected_cols_output_if_empty = ['ai_risk_score', 'ai_followup_priority_score']
    if health_df is not None and hasattr(health_df, 'columns') and health_df.columns.any():
        expected_cols_output_if_empty = list(set(health_df.columns.tolist() + expected_cols_output_if_empty))
    
    if health_df is None or health_df.empty:
        logger.warning(f"({source_context}) Input health_df to apply_ai_models is empty. Returning empty/None.")
        return pd.DataFrame(columns=expected_cols_output_if_empty), None

    df_enriched_main_ai = health_df.copy()

    # 1. Risk Scoring
    risk_predictor_instance = RiskPredictionModel()
    df_enriched_main_ai['ai_risk_score'] = risk_predictor_instance.predict_bulk_risk_scores(df_enriched_main_ai)
    logger.info(f"({source_context}) Applied AI Risk Scoring.")

    # 2. Follow-up/Task Prioritization
    followup_prioritizer_instance = FollowUpPrioritizer()
    if 'days_task_overdue' not in df_enriched_main_ai.columns: 
        df_enriched_main_ai['days_task_overdue'] = 0 # Ensure column exists and default to 0
    else: # If column exists, ensure it's integer and fill NaNs
        df_enriched_main_ai['days_task_overdue'] = pd.to_numeric(df_enriched_main_ai['days_task_overdue'], errors='coerce').fillna(0).astype(int)
    df_enriched_main_ai['ai_followup_priority_score'] = followup_prioritizer_instance.generate_followup_priorities(df_enriched_main_ai)
    logger.info(f"({source_context}) Applied AI Follow-up/Task Prioritization.")
    
    # 3. AI-Simulated Supply Forecasting (Optional)
    generated_supply_fc_df: Optional[pd.DataFrame] = None # Ensure it's initialized
    if current_supply_status_df is not None and not current_supply_status_df.empty:
        # Ensure required columns for supply status df are present before calling model
        req_supply_cols = ['item', 'current_stock', 'avg_daily_consumption_historical', 'last_stock_update_date']
        if all(col in current_supply_status_df.columns for col in req_supply_cols):
            supply_forecaster_instance_ai = SupplyForecastingModel()
            generated_supply_fc_df = supply_forecaster_instance_ai.forecast_supply_levels_advanced(
                current_supply_levels_df=current_supply_status_df,
                forecast_days_out=app_config.LOW_SUPPLY_DAYS_REMAINING # Example: 2 weeks
            )
            rec_count_fc = len(generated_supply_fc_df) if generated_supply_fc_df is not None else 0
            logger.info(f"({source_context}) Generated AI-simulated supply forecast: {rec_count_fc} records.")
        else:
            logger.warning(f"({source_context}) 'current_supply_status_df' missing required columns for AI supply forecast. Skipping.")
    else:
        logger.info(f"({source_context}) No 'current_supply_status_df' provided or it's empty; AI supply forecast skipped.")

    logger.info(f"({source_context}) AI model applications complete. Enriched DataFrame shape: {df_enriched_main_ai.shape}")
    return df_enriched_main_ai, generated_supply_fc_df
