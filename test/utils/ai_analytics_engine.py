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
from typing import Dict, Any, Optional, List, Tuple
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
        self.base_risk_factors = {
            'age': { 'weight': 0.5, 'threshold_high': app_config.RISK_SCORE_MODERATE_THRESHOLD, 'factor_high': 10, 'threshold_low': 18, 'factor_low': -2 },
            'min_spo2_pct': { 'weight': 2.5, 'threshold_low': app_config.ALERT_SPO2_CRITICAL_LOW_PCT, 'factor_low': 30, 'mid_threshold_low': app_config.ALERT_SPO2_WARNING_LOW_PCT, 'factor_mid_low': 15 },
            'vital_signs_temperature_celsius': { 'weight': 2.0, 'threshold_high': app_config.ALERT_BODY_TEMP_FEVER_C, 'factor_high': 15, 'super_high_threshold': app_config.ALERT_BODY_TEMP_HIGH_FEVER_C, 'factor_super_high': 25 },
            'max_skin_temp_celsius': { 'weight': 1.8, 'threshold_high': app_config.HEAT_STRESS_RISK_BODY_TEMP_C, 'factor_high': 10, 'super_high_threshold': app_config.ALERT_BODY_TEMP_HIGH_FEVER_C - 0.5, 'factor_super_high': 20 },
            'stress_level_score': { 'weight': 0.8, 'threshold_high': app_config.FATIGUE_INDEX_MODERATE_THRESHOLD, 'factor_high': 10, 'super_high_threshold': app_config.FATIGUE_INDEX_HIGH_THRESHOLD, 'factor_super_high': 15 },
            'avg_hrv_rmssd_ms': { 'weight': 1.2, 'threshold_low': app_config.STRESS_HRV_LOW_THRESHOLD_MS, 'factor_low': 15 }, # Name changed for clarity
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
        if pd.isna(condition_str) or str(condition_str).lower() in ["unknown", "none", "n/a"]: return 0.0
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
                    # Special handling for ppe_compliant_flag (risk if 0, not if 1)
                    if feature_key == 'ppe_compliant_flag' and value == params.get('flag_value', 0): # flag_value 0 is non-compliant
                        calculated_risk += params.get('factor_true', 0) * weight
                    elif feature_key != 'ppe_compliant_flag' and value == params.get('flag_value', 1): # Standard flags
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
             # Assume score 0-10, higher means more distress, scaled contribution
            calculated_risk += features.get('rapid_psychometric_distress_score', 0) * 1.5
        if pd.notna(features.get('signs_of_fatigue_observed_flag')) and features.get('signs_of_fatigue_observed_flag') == 1 :
            calculated_risk += 10 # Fixed points if fatigue signs observed

        return float(np.clip(calculated_risk, 0, 100))

    def predict_bulk_risk_scores(self, data_df: pd.DataFrame) -> pd.Series:
        if data_df.empty: return pd.Series(dtype='float64')
        temp_df_for_pred = data_df.copy()
        all_expected_features = list(self.base_risk_factors.keys()) + [
            'condition', 'chronic_condition_flag', 'medication_adherence_self_report',
            'rapid_psychometric_distress_score', 'signs_of_fatigue_observed_flag'
        ]
        for feature_name in all_expected_features:
            if feature_name not in temp_df_for_pred.columns:
                default_val = 0 if any(f in feature_name for f in ['_flag', '_score','fall_detected_today', 'ppe_compliant']) else \
                              np.nan if any(f in feature_name for f in ['spo2','temp','hrv','heat_index','age','rmssd']) else \
                              "Unknown"
                temp_df_for_pred[feature_name] = default_val
        return temp_df_for_pred.apply(lambda row: self.predict_risk_score(row), axis=1)

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
        temp_val = features.get('vital_signs_temperature_celsius', features.get('max_skin_temp_celsius', np.nan))
        if pd.notna(temp_val) and temp_val >= app_config.ALERT_BODY_TEMP_HIGH_FEVER_C: return True
        if pd.notna(features.get('fall_detected_today')) and features['fall_detected_today'] > 0: return True
        return False

    def _is_pending_urgent_task(self, features: pd.Series) -> bool:
        if str(features.get('referral_status', 'Unknown')).lower() == 'pending':
            if any(cond_key.lower() in str(features.get('condition', '')).lower() for cond_key in app_config.KEY_CONDITIONS_FOR_ACTION):
                return True
        if str(features.get('worker_task_priority', 'Normal')).lower() == 'urgent': return True # Worker specific tasks
        return False
        
    def _has_acute_condition_severity(self, features: pd.Series) -> bool:
        condition_str = str(features.get('condition','')).lower()
        if ("pneumonia" in condition_str and pd.notna(features.get('min_spo2_pct')) and features.get('min_spo2_pct') < app_config.ALERT_SPO2_WARNING_LOW_PCT): return True
        if any(crit_cond.lower() in condition_str for crit_cond in ["sepsis", "severe dehydration", "heat stroke"]): return True
        return False

    def _contextual_hazard_present(self, features: pd.Series) -> bool:
        if pd.notna(features.get('ambient_heat_index_c')) and features['ambient_heat_index_c'] >= app_config.ALERT_AMBIENT_HEAT_INDEX_DANGER_C: return True
        if pd.notna(features.get('ambient_co2_ppm')) and features['ambient_co2_ppm'] >= app_config.ALERT_AMBIENT_CO2_VERY_HIGH_PPM: return True
        if pd.notna(features.get('ambient_pm25_ugm3')) and features['ambient_pm25_ugm3'] >= app_config.ALERT_AMBIENT_PM25_VERY_HIGH_UGM3: return True
        return False

    def calculate_priority_score(self, features: pd.Series, days_task_overdue: int = 0) -> float:
        priority_score = 0.0
        ai_risk = features.get('ai_risk_score', 0.0)
        if pd.notna(ai_risk): priority_score += ai_risk * self.priority_weights['base_ai_risk_score_contribution_pct']
        if self._has_active_critical_vitals_alert(features): priority_score += self.priority_weights['critical_vital_alert_points']
        if self._is_pending_urgent_task(features): priority_score += self.priority_weights['pending_urgent_task_points']
        if self._has_acute_condition_severity(features): priority_score += self.priority_weights['acute_condition_severity_points']
        if self._contextual_hazard_present(features): priority_score += self.priority_weights['contextual_hazard_points']
        if str(features.get('medication_adherence_self_report', 'Unknown')).lower() == 'poor': priority_score += self.priority_weights['poor_adherence_points']
        if features.get('signs_of_fatigue_observed_flag') == 1 : priority_score += self.priority_weights['observed_fatigue_points']
        priority_score += min(days_task_overdue, 60) * self.priority_weights['task_overdue_factor_per_day'] # Cap overdue bonus e.g. at 30 points
        return float(np.clip(priority_score, 0, 100))

    def generate_followup_priorities(self, data_df: pd.DataFrame) -> pd.Series:
        if data_df.empty: return pd.Series(dtype='float64')
        temp_df_prio = data_df.copy()
        if 'days_task_overdue' not in temp_df_prio.columns: temp_df_prio['days_task_overdue'] = 0
        return temp_df_prio.apply(lambda row: self.calculate_priority_score(row, row.get('days_task_overdue',0)), axis=1)

class SupplyForecastingModel:
    def __init__(self):
        self.item_params = {}
        for item_key_config in app_config.KEY_DRUG_SUBSTRINGS_SUPPLY:
            self.item_params[item_key_config] = {
                "coeffs": np.random.uniform(0.7, 1.3, 12).tolist(), # Monthly seasonality
                "trend": np.random.uniform(0.0005, 0.01),           # Slight positive trend per 30 days
                "noise_std": np.random.uniform(0.03, 0.12)
            }
        if "ACT Tablets" in self.item_params: self.item_params["ACT Tablets"]["coeffs"] = [0.6,0.6,0.7,0.9,1.2,1.4,1.5,1.3,1.0,0.8,0.7,0.6] # Malaria seasonality
        logger.info("Simulated AI SupplyForecastingModel initialized (dynamic params for key drugs).")

    def _get_item_params(self, item_name: str) -> Dict:
        for key_substr in self.item_params.keys():
            if key_substr.lower() in item_name.lower(): return self.item_params[key_substr]
        return {"coeffs": [1.0]*12, "trend": 0.0005, "noise_std": 0.1} # Generic fallback

    def _predict_daily_consumption_ai(self, base_avg_daily_consumption: float, item_name: str, forecast_date: pd.Timestamp, days_since_forecast_start: int) -> float:
        if pd.isna(base_avg_daily_consumption) or base_avg_daily_consumption <= 0: return 0.0001
        params_item = self._get_item_params(item_name)
        seasonal_adj = params_item["coeffs"][forecast_date.month - 1]
        trend_adj = (1 + params_item["trend"] / 30)**days_since_forecast_start
        noise_adj = np.random.normal(1.0, params_item["noise_std"])
        predicted_cons = base_avg_daily_consumption * seasonal_adj * trend_adj * noise_adj
        return max(0.0001, predicted_cons)

    def forecast_supply_levels_advanced(self, current_supply_levels_df: pd.DataFrame, forecast_days_out: int = 7, item_filter_list: Optional[List[str]] = None) -> pd.DataFrame:
        logger.info(f"AI-simulated supply forecast. Horizon: {forecast_days_out} days. Items: {item_filter_list or 'All in input'}")
        output_cols_fc = ['item', 'forecast_date', 'forecasted_stock_level', 'forecasted_days_of_supply', 'predicted_daily_consumption', 'estimated_stockout_date_ai']
        if not isinstance(current_supply_levels_df, pd.DataFrame) or current_supply_levels_df.empty: return pd.DataFrame(columns=output_cols_fc)
        req_cols_fc = ['item', 'current_stock', 'avg_daily_consumption_historical', 'last_stock_update_date']
        if not all(col in current_supply_levels_df.columns for col in req_cols_fc): logger.error(f"AI Supply Forecast input DF missing cols: {req_cols_fc}"); return pd.DataFrame(columns=output_cols_fc)

        df_to_fc = current_supply_levels_df.copy()
        if item_filter_list: df_to_fc = df_to_fc[df_to_fc['item'].isin(item_filter_list)]
        if df_to_fc.empty: return pd.DataFrame(columns=output_cols_fc)
        df_to_fc['last_stock_update_date'] = pd.to_datetime(df_to_fc['last_stock_update_date'], errors='coerce')
        df_to_fc.dropna(subset=['last_stock_update_date', 'item'], inplace=True)

        all_fc_list = []
        for _, item_row_fc in df_to_fc.iterrows():
            item_name_fc_curr = item_row_fc['item']
            stock_start_fc_curr = float(item_row_fc.get('current_stock', 0.0))
            base_cons_fc_curr = float(item_row_fc.get('avg_daily_consumption_historical', 0.0001))
            last_date_fc_curr = item_row_fc['last_stock_update_date']
            if pd.isna(stock_start_fc_curr) or stock_start_fc_curr < 0: stock_start_fc_curr = 0.0
            if pd.isna(base_cons_fc_curr) or base_cons_fc_curr <= 0: base_cons_fc_curr = 0.0001

            running_stock = stock_start_fc_curr; est_stockout_dt_curr = pd.NaT
            item_daily_forecasts = [] # Store forecasts for this item to later average consumption for stockout estimate
            for day_offset_fc in range(forecast_days_out):
                current_fc_dt = last_date_fc_curr + pd.Timedelta(days=day_offset_fc + 1)
                daily_pred_use_fc = self._predict_daily_consumption_ai(base_cons_fc_curr, item_name_fc_curr, current_fc_dt, day_offset_fc + 1)
                stock_before_use_today = running_stock
                running_stock = max(0, running_stock - daily_pred_use_fc)
                days_supply_fc = (running_stock / daily_pred_use_fc) if daily_pred_use_fc > 0.00001 else (np.inf if running_stock > 0 else 0)
                if pd.isna(est_stockout_dt_curr) and stock_before_use_today > 0 and running_stock <= 0:
                    fraction_day = (stock_before_use_today / daily_pred_use_fc) if daily_pred_use_fc > 0.00001 else 0.0
                    est_stockout_dt_curr = last_date_fc_curr + pd.Timedelta(days=day_offset_fc + fraction_day)
                
                current_day_record = {
                    'item': item_name_fc_curr, 'forecast_date': current_fc_dt,
                    'forecasted_stock_level': running_stock, 'forecasted_days_of_supply': days_supply_fc,
                    'predicted_daily_consumption': daily_pred_use_fc, 'estimated_stockout_date_ai': est_stockout_dt_curr
                }
                all_fc_list.append(current_day_record)
                item_daily_forecasts.append(current_day_record)
            
            # If not stocked out in period, do a final estimate beyond horizon for this item's block
            if pd.isna(est_stockout_dt_curr) and stock_start_fc_curr > 0 and item_daily_forecasts:
                avg_pred_cons_fc_period = pd.Series([f['predicted_daily_consumption'] for f in item_daily_forecasts]).mean()
                if avg_pred_cons_fc_period > 0.00001:
                    days_to_final_so = stock_start_fc_curr / avg_pred_cons_fc_period
                    final_est_so_dt_for_item = last_date_fc_curr + pd.to_timedelta(days_to_final_so, unit='D')
                    for rec in all_fc_list: # Update previous records for this item if their stockout was NaT
                        if rec['item'] == item_name_fc_curr and pd.isna(rec['estimated_stockout_date_ai']):
                            rec['estimated_stockout_date_ai'] = final_est_so_dt_for_item
        
        if not all_fc_list: return pd.DataFrame(columns=output_cols_fc)
        forecast_df_result = pd.DataFrame(all_fc_list)
        forecast_df_result['estimated_stockout_date_ai'] = pd.to_datetime(forecast_df_result['estimated_stockout_date_ai'], errors='coerce')
        return forecast_df_result

# --- Central AI Application Function ---
def apply_ai_models(
    health_df: pd.DataFrame,
    current_supply_status_df: Optional[pd.DataFrame] = None,
    source_context: str = "FacilityNode/BatchAI"
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    logger.info(f"({source_context}) Applying AI models to health data (rows: {len(health_df) if health_df is not None else 0}).")
    if health_df is None or health_df.empty:
        logger.warning(f"({source_context}) Input health_df to apply_ai_models is empty. Returning empty health_df, None for supply.")
        return pd.DataFrame(columns=health_df.columns if health_df is not None else []), None

    df_enriched_ai = health_df.copy()
    risk_model_ai = RiskPredictionModel()
    df_enriched_ai['ai_risk_score'] = risk_model_ai.predict_bulk_risk_scores(df_enriched_ai)
    logger.info(f"({source_context}) Applied AI risk scoring.")

    priority_model_ai = FollowUpPrioritizer()
    # generate_followup_priorities expects 'days_task_overdue'. If not present, it defaults to 0 internally.
    # A more sophisticated pipeline might calculate this based on task creation dates vs. current date.
    if 'days_task_overdue' not in df_enriched_ai.columns: df_enriched_ai['days_task_overdue'] = 0
    df_enriched_ai['ai_followup_priority_score'] = priority_model_ai.generate_followup_priorities(df_enriched_ai)
    logger.info(f"({source_context}) Applied AI follow-up/task prioritization.")
    
    supply_fc_df_output: Optional[pd.DataFrame] = None
    if current_supply_status_df is not None and not current_supply_status_df.empty:
        supply_model_ai_instance = SupplyForecastingModel()
        supply_fc_df_output = supply_model_ai_instance.forecast_supply_levels_advanced(
            current_supply_status_df,
            # Using a short forecast horizon default relevant for Sentinel; this could be configured
            forecast_days_out=app_config.LOW_SUPPLY_DAYS_REMAINING # e.g., 2 weeks horizon
        )
        logger.info(f"({source_context}) Generated AI-simulated supply forecast with {len(supply_fc_df_output) if supply_fc_df_output is not None else 0} records.")
    else:
        logger.info(f"({source_context}) No current_supply_status_df provided; AI supply forecast skipped.")

    logger.info(f"({source_context}) AI model application complete. Enriched DF shape: {df_enriched_ai.shape}")
    return df_enriched_ai, supply_fc_df_output
