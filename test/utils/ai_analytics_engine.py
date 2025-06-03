```python
# sentinel_project_root/test/utils/ai_analytics_engine.py
# Simulates core AI/Analytics logic for the Sentinel Health Co-Pilot System.

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import logging
from config import app_config
from .protocol_executor import execute_escalation_protocol

logger = logging.getLogger(__name__)

# Fallback thresholds
ALERT_SPO2_CRITICAL = getattr(app_config, 'ALERT_SPO2_CRITICAL', 90)
ALERT_SPO2_WARNING = getattr(app_config, 'ALERT_SPO2_WARNING', 94)
ALERT_BODY_TEMP_FEV = getattr(app_config, 'ALERT_BODY_TEMP_FEV', 38.0)
HEAT_STRESS_RISK_BODY_TEMP_C = getattr(app_config, 'HEAT_STRESS_RISK_BODY_TEMP_C', 37.5)
FATIGUE_INDEX_MODERATE = getattr(app_config, 'FATIGUE_INDEX_MODERATE', 50)
FATIGUE_INDEX_HIGH = getattr(app_config, 'FATIGUE_INDEX_HIGH', 75)
STRESS_HRV_LOW_MS = getattr(app_config, 'STRESS_HRV_LOW_THRESHOLD_MS', 30)
ALERT_AMBIENT_HEAT_RISK_C = getattr(app_config, 'ALERT_AMBIENT_HEAT_INDEX_RISK_C', 32)
ALERT_AMBIENT_HEAT_DANGER_C = getattr(app_config, 'ALERT_AMBIENT_HEAT_INDEX_DANGER_C', 40)
KEY_CONDITIONS_ACTION = getattr(app_config, 'KEY_CONDITIONS', ['Malaria', 'TB', 'HIV'])
LOW_SUPPLY_DAYS = getattr(app_config, 'CRITICAL_SUPPLY_DAYS', 7)
AGE_THRESHOLD_LOW = getattr(app_config, 'AGE_THRESHOLD_LOW', 18)
AGE_THRESHOLD_HIGH = getattr(app_config, 'AGE_THRESHOLD_HIGH', 60)
AGE_THRESHOLD_VERY_HIGH = getattr(app_config, 'AGE_THRESHOLD_VERY_HIGH', 75)

class RiskPredictionModel:
    """
    Simulates a pre-trained patient/worker risk prediction model.
    Uses rule-based logic with weights for core features.
    """
    def __init__(self):
        self.base_risk_factors = {
            'age': {'weight': 0.5, 'threshold_low': AGE_THRESHOLD_LOW, 'factor_low': -3, 'threshold_high': AGE_THRESHOLD_HIGH, 'factor_high': 10, 'threshold_very_high': AGE_THRESHOLD_VERY_HIGH, 'factor_very_high': 15},
            'min_spo2_pct': {'weight': 2.5, 'threshold_low': ALERT_SPO2_CRITICAL, 'factor_low': 30, 'mid_threshold_low': ALERT_SPO2_WARNING, 'factor_mid_low': 15},
            'vital_signs_temperature_celsius': {'weight': 2.0, 'threshold_high': ALERT_BODY_TEMP_FEV, 'factor_high': 15, 'super_high_threshold': ALERT_BODY_TEMP_FEV + 1.5, 'factor_super_high': 25},
            'max_skin_temp_celsius': {'weight': 1.8, 'threshold_high': HEAT_STRESS_RISK_BODY_TEMP_C, 'factor_high': 10, 'super_high_threshold': ALERT_BODY_TEMP_FEV - 0.5, 'factor_super_high': 20},
            'stress_level_score': {'weight': 0.8, 'threshold_high': FATIGUE_INDEX_MODERATE, 'factor_high': 8, 'super_high_threshold': FATIGUE_INDEX_HIGH, 'factor_super_high': 12},
            'hrv_rmssd_ms': {'weight': 1.2, 'threshold_low': STRESS_HRV_LOW_MS, 'factor_low': 15},
            'tb_contact_traced': {'weight': 1.0, 'is_flag': True, 'flag_value': 1, 'factor_true': 12},
            'fall_detected_today': {'weight': 2.2, 'is_flag': True, 'flag_value': 1, 'factor_true': 25},
            'ambient_heat_index_c': {'weight': 0.7, 'threshold_high': ALERT_AMBIENT_HEAT_RISK_C, 'factor_high': 8, 'super_high_threshold': ALERT_AMBIENT_HEAT_DANGER_C, 'factor_super_high': 15},
            'ppe_compliant_flag': {'weight': 1.0, 'is_flag': True, 'flag_value': 0, 'factor_true': 10}
        }
        self.condition_base_scores = {cond: 25 for cond in KEY_CONDITIONS_ACTION}
        self.condition_base_scores.update({
            "Sepsis": 45, "Severe Dehydration": 40, "Heat Stroke": 42, "TB": 30,
            "HIV-Positive": 22, "Pneumonia": 35, "Malaria": 20,
            "Wellness Visit": -15, "Follow-up Health": -8, "Minor Cold": -5
        })
        self.CHRONIC_CONDITION_FLAG_RISK_POINTS = 18
        logger.info("RiskPredictionModel initialized.")

    def _get_condition_base_score(self, condition_str: Optional[str]) -> float:
        if pd.isna(condition_str) or not isinstance(condition_str, str) or condition_str.strip().lower() in ["unknown", "none", "n/a", ""]:
            return 0.0
        max_score = 0.0
        condition_lower = condition_str.lower()
        potential_conditions = [c.strip() for delim in [';', ','] for c in condition_lower.split(delim)]
        if not any(delim in condition_lower for delim in [';', ',']):
            potential_conditions = [condition_lower]
        for part_cond in potential_conditions:
            if not part_cond:
                continue
            exact_match_score = self.condition_base_scores.get(part_cond.title())
            if exact_match_score is not None:
                max_score = max(max_score, exact_match_score)
                continue
            for known_cond, score in self.condition_base_scores.items():
                if known_cond.lower() in part_cond:
                    max_score = max(max_score, score)
        return max_score

    def predict_risk_score(self, features: pd.Series) -> float:
        if not isinstance(features, pd.Series):
            logger.error("predict_risk_score expects a pandas Series.")
            return 0.0
        features = features.copy()
        for key in self.base_risk_factors:
            if key not in features:
                features[key] = 0 if 'flag' in key else np.nan
        calculated_risk = self._get_condition_base_score(features.get('condition'))
        if str(features.get('chronic_condition_flag', '0')).lower() in ['1', 'yes', 'true']:
            calculated_risk += self.CHRONIC_CONDITION_FLAG_RISK_POINTS
        for feature_key, params in self.base_risk_factors.items():
            value = features.get(feature_key)
            if pd.notna(value):
                weight = params.get('weight', 1.0)
                if params.get('is_flag'):
                    if str(value).lower() == str(params.get('flag_value', 1)).lower():
                        calculated_risk += params.get('factor_true', 0) * weight
                        if feature_key == 'fall_detected_today':
                            execute_escalation_protocol("PATIENT_FALL_DETECTED", features.to_dict())
                else:
                    try:
                        num_val = float(value)
                        if 'super_high_threshold' in params and num_val >= params['super_high_threshold']:
                            calculated_risk += params['factor_super_high'] * weight
                        elif 'threshold_very_high' in params and num_val >= params['threshold_very_high']:
                            calculated_risk += params['factor_very_high'] * weight
                        elif 'threshold_high' in params and num_val >= params['threshold_high']:
                            calculated_risk += params['factor_high'] * weight
                        if 'threshold_low' in params and num_val < params['threshold_low']:
                            calculated_risk += params['factor_low'] * weight
                            if feature_key == 'min_spo2_pct' and num_val < ALERT_SPO2_CRITICAL:
                                execute_escalation_protocol("PATIENT_CRITICAL_SPO2_LOW", features.to_dict())
                        elif 'mid_threshold_low' in params and num_val < params['mid_threshold_low']:
                            calculated_risk += params['factor_mid_low'] * weight
                    except (ValueError, TypeError):
                        pass
        adherence = str(features.get('medication_adherence_self_report', 'Unknown')).lower()
        if adherence == 'poor':
            calculated_risk += 12
        elif adherence == 'fair':
            calculated_risk += 6
        psych_score = features.get('rapid_psychometric_distress_score')
        if pd.notna(psych_score):
            try:
                calculated_risk += float(psych_score) * 1.5
            except (ValueError, TypeError):
                pass
        if str(features.get('signs_of_fatigue_observed_flag', '0')).lower() in ['1', 'true']:
            calculated_risk += 10
        return float(np.clip(calculated_risk, 0, 100))

    def predict_bulk_risk_scores(self, data_df: pd.DataFrame) -> pd.Series:
        if not isinstance(data_df, pd.DataFrame) or data_df.empty:
            return pd.Series(dtype='float64')
        risk_scores = np.zeros(len(data_df))
        temp_df = data_df.copy()
        for feature_key, params in self.base_risk_factors.items():
            if feature_key not in temp_df.columns:
                temp_df[feature_key] = 0 if 'flag' in feature_key else np.nan
            values = pd.to_numeric(temp_df[feature_key], errors='coerce')
            weight = params.get('weight', 1.0)
            if params.get('is_flag'):
                mask = values == params.get('flag_value', 1)
                risk_scores += mask * params.get('factor_true', 0) * weight
                if feature_key == 'fall_detected_today':
                    for idx in temp_df[mask].index:
                        execute_escalation_protocol("PATIENT_FALL_DETECTED", temp_df.loc[idx].to_dict())
            else:
                if 'threshold_low' in params:
                    mask = values < params['threshold_low']
                    risk_scores += mask * params.get('factor_low', 0) * weight
                    if feature_key == 'min_spo2_pct':
                        critical_mask = values < ALERT_SPO2_CRITICAL
                        for idx in temp_df[critical_mask].index:
                            execute_escalation_protocol("PATIENT_CRITICAL_SPO2_LOW", temp_df.loc[idx].to_dict())
                if 'mid_threshold_low' in params:
                    mask = values < params['mid_threshold_low']
                    risk_scores += mask * params.get('factor_mid_low', 0) * weight
                if 'threshold_high' in params:
                    mask = values >= params['threshold_high']
                    risk_scores += mask * params.get('factor_high', 0) * weight
                if 'threshold_very_high' in params:
                    mask = values >= params['threshold_very_high']
                    risk_scores += mask * params.get('factor_very_high', 0) * weight
                if 'super_high_threshold' in params:
                    mask = values >= params['super_high_threshold']
                    risk_scores += mask * params.get('factor_super_high', 0) * weight
        condition_scores = temp_df['condition'].apply(self._get_condition_base_score)
        risk_scores += condition_scores
        chronic_mask = temp_df.get('chronic_condition_flag', pd.Series(['0'] * len(temp_df))).astype(str).str.lower().isin(['1', 'yes', 'true'])
        risk_scores += chronic_mask * self.CHRONIC_CONDITION_FLAG_RISK_POINTS
        adherence = temp_df.get('medication_adherence_self_report', pd.Series(['Unknown'] * len(temp_df))).astype(str).str.lower()
        risk_scores += (adherence == 'poor') * 12 + (adherence == 'fair') * 6
        psych_scores = pd.to_numeric(temp_df.get('rapid_psychometric_distress_score', pd.Series([0] * len(temp_df))), errors='coerce').fillna(0)
        risk_scores += psych_scores * 1.5
        fatigue_mask = temp_df.get('signs_of_fatigue_observed_flag', pd.Series(['0'] * len(temp_df))).astype(str).str.lower().isin(['1', 'true'])
        risk_scores += fatigue_mask * 10
        return pd.Series(np.clip(risk_scores, 0, 100), index=temp_df.index)

class FollowUpPrioritizer:
    """Simulates logic to prioritize patients for follow-up or tasks."""
    def __init__(self):
        self.priority_weights = {
            'base_ai_risk_score_contribution_pct': 0.4,
            'critical_vital_alert_points': 40,
            'pending_urgent_task_points': 30,
            'acute_condition_severity_points': 25,
            'contextual_hazard_points': 15,
            'task_overdue_factor_per_day': 1.0,
            'poor_adherence_points': 15,
            'observed_fatigue_points': 12
        }
        logger.info("FollowUpPrioritizer initialized.")

    def _has_active_critical_vitals_alert(self, features: pd.Series) -> bool:
        if pd.notna(features.get('min_spo2_pct')) and float(features.get('min_spo2_pct', 100)) < ALERT_SPO2_CRITICAL:
            execute_escalation_protocol("PATIENT_CRITICAL_SPO2_LOW", features.to_dict())
            return True
        temp_val = features.get('vital_signs_temperature_celsius', features.get('max_skin_temp_celsius'))
        if pd.notna(temp_val) and float(temp_val) >= ALERT_BODY_TEMP_FEV + 1.5:
            return True
        if pd.notna(features.get('fall_detected_today')) and str(features.get('fall_detected_today', '0')).lower() in ['1', 'true']:
            execute_escalation_protocol("PATIENT_FALL_DETECTED", features.to_dict())
            return True
        return False

    def _is_pending_urgent_task(self, features: pd.Series) -> bool:
        if str(features.get('referral_status', '')).lower() == 'pending':
            condition_lower = str(features.get('condition', '')).lower()
            if any(kc.lower() in condition_lower for kc in KEY_CONDITIONS_ACTION):
                return True
        if str(features.get('worker_task_priority', 'Normal')).lower() == 'urgent':
            return True
        return False

    def _has_acute_condition_severity(self, features: pd.Series) -> bool:
        condition_lower = str(features.get('condition', '')).lower()
        if "pneumonia" in condition_lower and pd.notna(features.get('min_spo2_pct')) and \
           float(features.get('min_spo2_pct', 100)) < ALERT_SPO2_WARNING:
            return True
        if any(crit.lower() in condition_lower for crit in ["sepsis", "severe dehydration", "heat stroke"]):
            return True
        return False

    def _contextual_hazard_present(self, features: pd.Series) -> bool:
        if pd.notna(features.get('ambient_heat_index_c')) and \
           float(features.get('ambient_heat_index_c', 0)) >= ALERT_AMBIENT_HEAT_DANGER_C:
            return True
        return False

    def calculate_priority_score(self, features: pd.Series, days_overdue: int = 0) -> float:
        if not isinstance(features, pd.Series):
            logger.error("calculate_priority_score expects a pandas Series.")
            return 0.0
        priority_score = 0.0
        ai_risk = pd.to_numeric(features.get('ai_risk_score', 0.0), errors='coerce').fillna(0.0)
        priority_score += ai_risk * self.priority_weights['base_ai_risk_score_contribution_pct']
        if self._has_active_critical_vitals_alert(features):
            priority_score += self.priority_weights['critical_vital_alert_points']
        if self._is_pending_urgent_task(features):
            priority_score += self.priority_weights['pending_urgent_task_points']
        if self._has_acute_condition_severity(features):
            priority_score += self.priority_weights['acute_condition_severity_points']
        if self._contextual_hazard_present(features):
            priority_score += self.priority_weights['contextual_hazard_points']
        if str(features.get('medication_adherence_self_report', 'Unknown')).lower() == 'poor':
            priority_score += self.priority_weights['poor_adherence_points']
        if str(features.get('signs_of_fatigue_observed_flag', '0')).lower() in ['1', 'true']:
            priority_score += self.priority_weights['observed_fatigue_points']
        priority_score += min(int(days_overdue), 60) * self.priority_weights['task_overdue_factor_per_day']
        return float(np.clip(priority_score, 0, 100))

    def generate_followup_priorities(self, data_df: pd.DataFrame) -> pd.Series:
        if not isinstance(data_df, pd.DataFrame) or data_df.empty:
            return pd.Series(dtype='float64')
        temp_df = data_df.copy()
        if 'days_task_overdue' not in temp_df.columns:
            temp_df['days_task_overdue'] = 0
        else:
            temp_df['days_task_overdue'] = pd.to_numeric(temp_df['days_task_overdue'], errors='coerce').fillna(0).astype(int)
        features_to_check = [
            'min_spo2_pct', 'vital_signs_temperature_celsius', 'max_skin_temp_celsius',
            'fall_detected_today', 'referral_status', 'condition', 'worker_task_priority',
            'ambient_heat_index_c', 'medication_adherence_self_report', 'signs_of_fatigue_observed_flag', 'ai_risk_score'
        ]
        for feature in features_to_check:
            if feature not in temp_df.columns:
                temp_df[feature] = np.nan if feature in ['min_spo2_pct', 'vital_signs_temperature_celsius', 'max_skin_temp_celsius', 'ambient_heat_index_c', 'ai_risk_score'] else \
                    0 if feature in ['fall_detected_today', 'signs_of_fatigue_observed_flag'] else 'Unknown'
            if feature in ['min_spo2_pct', 'vital_signs_temperature_celsius', 'max_skin_temp_celsius', 'ambient_heat_index_c', 'ai_risk_score']:
                temp_df[feature] = pd.to_numeric(temp_df[feature], errors='coerce')
        return temp_df.apply(
            lambda row: self.calculate_priority_score(row, row.get('days_task_overdue', 0)),
            axis=1
        )

class SupplyForecastingModel:
    """Simulates an AI-driven supply forecasting model with seasonality and trend."""
    def __init__(self):
        np.random.seed(getattr(app_config, 'RANDOM_SEED', 42))
        self.item_params: Dict[str, Dict[str, Any]] = {}
        base_coeffs = np.array([1.0] * 12)
        for item_key in getattr(app_config, 'KEY_DRUGS', []):
            seed_val = abs(hash(item_key)) % (2**32 - 1)
            rng_item = np.random.RandomState(seed_val)
            self.item_params[item_key] = {
                "coeffs": (base_coeffs * rng_item.uniform(0.85, 1.15, 12)).round(3).tolist(),
                "trend": rng_item.uniform(0.0001, 0.003),
                "noise_std": rng_item.uniform(0.02, 0.08)
            }
        act_key = next((k for k in self.item_params if "act" in k.lower()), None)
        if act_key:
            self.item_params[act_key]["coeffs"] = [0.7, 0.7, 0.8, 1.0, 1.3, 1.5, 1.6, 1.4, 1.1, 0.9, 0.8, 0.7]
            self.item_params[act_key]["trend"] = 0.0015
        logger.info(f"SupplyForecastingModel initialized with params for {len(self.item_params)} drugs.")

    def _get_item_params(self, item_name: str) -> Dict[str, Any]:
        item_name_lower = str(item_name).lower()
        for key, params in self.item_params.items():
            if key.lower() in item_name_lower:
                return params
        logger.debug(f"No params for item '{item_name}', using fallback.")
        return {"coeffs": [1.0]*12, "trend": 0.0001, "noise_std": 0.05}

    def _predict_daily_consumption_ai(self, base_avg_daily_consumption: float, item_name: str, forecast_date: pd.Timestamp, days_since_forecast_start: int) -> float:
        if pd.isna(base_avg_daily_consumption) or base_avg_daily_consumption <= 1e-6:
            return 1e-6
        params = self._get_item_params(item_name)
        month_index = forecast_date.month - 1
        seasonality_factor = params["coeffs"][month_index]
        trend_effect = (1 + params["trend"]) ** days_since_forecast_start
        noise_factor = np.random.normal(1.0, params["noise_std"])
        predicted_consumption = base_avg_daily_consumption * seasonality_factor * trend_effect * noise_factor
        return max(1e-6, predicted_consumption)

    def forecast_supply_levels_advanced(
            self, current_supply_levels_df: pd.DataFrame,
            forecast_days_out: int = 30,
            item_filter_list: Optional[List[str]] = None
    ) -> pd.DataFrame:
        logger.info(f"Supply forecast initiated. Horizon: {forecast_days_out} days.")
        output_cols = ['item', 'forecast_date', 'forecasted_stock_level',
                       'forecasted_days_of_supply', 'predicted_daily_consumption',
                       'estimated_stockout_date_ai']
        if not isinstance(current_supply_levels_df, pd.DataFrame) or current_supply_levels_df.empty:
            return pd.DataFrame(columns=output_cols)
        required_cols = ['item', 'current_stock', 'avg_daily_consumption_historical', 'last_stock_update_date']
        missing_cols = [col for col in required_cols if col not in current_supply_levels_df.columns]
        if missing_cols:
            logger.error(f"Missing columns: {missing_cols}")
            return pd.DataFrame(columns=output_cols)
        df = current_supply_levels_df.copy()
        if item_filter_list:
            df = df[df['item'].isin(item_filter_list)]
        df['last_stock_update_date'] = pd.to_datetime(df['last_stock_update_date'], errors='coerce', format='%Y-%m-%d')
        df['current_stock'] = pd.to_numeric(df['current_stock'], errors='coerce').fillna(0.0)
        df['avg_daily_consumption_historical'] = pd.to_numeric(df['avg_daily_consumption_historical'], errors='coerce').fillna(1e-6)
        df.dropna(subset=['last_stock_update_date', 'item'], inplace=True)
        if df.empty:
            return pd.DataFrame(columns=output_cols)
        records = []
        for _, row in df.iterrows():
            item = row['item']
            stock = max(0.0, float(row['current_stock']))
            base_daily_cons = max(1e-6, float(row['avg_daily_consumption_historical']))
            last_update = row['last_stock_update_date']
            running_stock = stock
            stockout_date = pd.NaT
            item_forecasts = []
            for day in range(forecast_days_out):
                fc_date = last_update + pd.Timedelta(days=day + 1)
                predicted_cons = self._predict_daily_consumption_ai(base_daily_cons, item, fc_date, day + 1)
                stock_before = running_stock
                running_stock = max(0, running_stock - predicted_cons)
                days_supply = running_stock / predicted_cons if predicted_cons > 1e-7 else np.inf
                if pd.isna(stockout_date) and stock_before > 0 and running_stock <= 0:
                    fraction = stock_before / predicted_cons if predicted_cons > 1e-7 else 0.0
                    stockout_date = last_update + pd.Timedelta(days=day + fraction)
                item_forecasts.append({
                    'item': item, 'forecast_date': fc_date,
                    'forecasted_stock_level': running_stock,
                    'forecasted_days_of_supply': days_supply,
                    'predicted_daily_consumption': predicted_cons,
                    'estimated_stockout_date_ai': stockout_date
                })
            if pd.isna(stockout_date) and stock > 0 and item_forecasts:
                avg_cons = pd.Series([d['predicted_daily_consumption'] for d in item_forecasts]).mean()
                if avg_cons > 1e-7:
                    days_to_stockout = stock / avg_cons
                    stockout_date = last_update + pd.to_timedelta(days_to_stockout, unit='D')
                    for record in item_forecasts:
                        if pd.isna(record['estimated_stockout_date_ai']):
                            record['estimated_stockout_date_ai'] = stockout_date
            records.extend(item_forecasts)
        if not records:
            return pd.DataFrame(columns=output_cols)
        final_df = pd.DataFrame(records)
        final_df['estimated_stockout_date_ai'] = pd.to_datetime(final_df['estimated_stockout_date_ai'], errors='coerce')
        logger.info(f"Supply forecast complete: {len(final_df)} records for {df['item'].nunique()} items.")
        return final_df

def apply_ai_models(
    health_df: pd.DataFrame,
    current_supply_status_df: Optional[pd.DataFrame] = None,
    source_context: str = "AIModelOrchestrator"
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    logger.info(f"({source_context}) Applying AI models to health data (rows: {len(health_df) if health_df is not None else 0}).")
    if not isinstance(health_df, pd.DataFrame):
        logger.error(f"({source_context}) Input health_df is not a DataFrame.")
        return pd.DataFrame(columns=['ai_risk_score', 'ai_followup_priority_score']), None
    if health_df.empty:
        logger.warning(f"({source_context}) Input health_df is empty.")
        base_cols = health_df.columns.tolist()
        ai_cols = ['ai_risk_score', 'ai_followup_priority_score']
        return pd.DataFrame(columns=list(set(base_cols + ai_cols))), None
    required_cols = ['patient_id', 'condition', 'min_spo2_pct']
    missing_cols = [col for col in required_cols if col not in health_df.columns]
    if missing_cols:
        logger.warning(f"({source_context}) Missing columns: {missing_cols}")
    df_enriched = health_df.copy()
    risk_model = RiskPredictionModel()
    df_enriched['ai_risk_score'] = risk_model.predict_bulk_risk_scores(df_enriched)
    if len(df_enriched) < 100:
        logger.debug(f"({source_context}) Risk scores: {df_enriched['ai_risk_score'].head(2).tolist()}")
    prioritizer = FollowUpPrioritizer()
    if 'days_task_overdue' not in df_enriched.columns:
        df_enriched['days_task_overdue'] = 0
    else:
        df_enriched['days_task_overdue'] = pd.to_numeric(df_enriched['days_task_overdue'], errors='coerce').fillna(0).astype(int)
    df_enriched['ai_followup_priority_score'] = prioritizer.generate_followup_priorities(df_enriched)
    if len(df_enriched) < 100:
        logger.debug(f"({source_context}) Priority scores: {df_enriched['ai_followup_priority_score'].head(2).tolist()}")
    supply_forecast_df = None
    if isinstance(current_supply_status_df, pd.DataFrame) and not current_supply_status_df.empty:
        required_supply_cols = ['item', 'current_stock', 'avg_daily_consumption_historical', 'last_stock_update_date']
        if all(col in current_supply_status_df.columns for col in required_supply_cols):
            forecaster = SupplyForecastingModel()
            supply_forecast_df = forecaster.forecast_supply_levels_advanced(
                current_supply_levels_df=current_supply_status_df,
                forecast_days_out=LOW_SUPPLY_DAYS * 2
            )
            logger.debug(f"({source_context}) Supply forecast: {len(supply_forecast_df)} records.")
        else:
            logger.warning(f"({source_context}) Missing supply columns: {[col for col in required_supply_cols if col not in current_supply_status_df.columns]}")
    logger.info(f"({source_context}) AI models applied. Enriched shape: {df_enriched.shape}")
    return df_enriched, supply_forecast_df
```
