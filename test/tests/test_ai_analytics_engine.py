# sentinel_project_root/test/tests/test_ai_analytics_engine.py
# Pytest tests for the refactored AI simulation logic in utils.ai_analytics_engine.py
# Aligned with "Sentinel Health Co-Pilot" redesign.

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Classes and functions to be tested from the refactored module
from utils.ai_analytics_engine import (
    RiskPredictionModel,
    FollowUpPrioritizer,
    SupplyForecastingModel,
    apply_ai_models
)
# Import app_config to use its thresholds, keys, and lean data field names in tests
from config import app_config # The NEW, redesigned app_config

# Fixtures are automatically sourced by pytest from conftest.py in the same directory or parent.
# We expect: sample_health_records_df_main_sentinel.

# --- Tests for RiskPredictionModel ---

@pytest.fixture(scope="module") # Use module scope for fixture efficiency
def risk_model_instance():
    """Provides a single instance of RiskPredictionModel for tests in this module."""
    return RiskPredictionModel()

def test_risk_model_condition_base_score_sentinel(risk_model_instance):
    # Test KEY_CONDITIONS_FOR_ACTION
    if app_config.KEY_CONDITIONS_FOR_ACTION:
        for condition in app_config.KEY_CONDITIONS_FOR_ACTION:
            expected_score = risk_model_instance.condition_base_scores.get(condition, 0) # Get from model's internal dict
            assert risk_model_instance._get_condition_base_score(condition) == expected_score, f"Score mismatch for key condition: {condition}"
            assert risk_model_instance._get_condition_base_score(condition.lower()) == expected_score, f"Lowercase score mismatch: {condition}"
    
    assert risk_model_instance._get_condition_base_score("Pneumonia;Severe Dehydration") >= risk_model_instance.condition_base_scores.get("Severe Dehydration", 0) # Should pick up highest or combine based on model logic
    assert risk_model_instance._get_condition_base_score("NonExistentCondition") == 0.0
    assert risk_model_instance._get_condition_base_score(None) == 0.0
    assert risk_model_instance._get_condition_base_score("Wellness Visit") < 0 # Should be negative based on model setup


def test_risk_model_predict_score_with_sentinel_factors(risk_model_instance):
    # Base case: Wellness Visit, good vitals, no adverse flags
    base_features = pd.Series({
        'condition': 'Wellness Visit', 'age': 30, 'chronic_condition_flag': 0,
        'min_spo2_pct': 98, 'vital_signs_temperature_celsius': 37.0,
        'fall_detected_today': 0, 'ambient_heat_index_c': 28,
        'ppe_compliant_flag': 1, 'signs_of_fatigue_observed_flag': 0,
        'rapid_psychometric_distress_score': 1, 'hrv_rmssd_ms': 50,
        'medication_adherence_self_report': 'Good'
    })
    base_score = risk_model_instance.predict_risk_score(base_features.copy())
    assert 0 <= base_score <= 100

    # Test CRITICAL SpO2 impact (should significantly increase score)
    features_low_spo2 = base_features.copy()
    features_low_spo2['min_spo2_pct'] = app_config.ALERT_SPO2_CRITICAL_LOW_PCT - 1
    score_low_spo2 = risk_model_instance.predict_risk_score(features_low_spo2)
    assert score_low_spo2 > base_score + (risk_model_instance.base_risk_factors['min_spo2_pct']['factor_low'] * risk_model_instance.base_risk_factors['min_spo2_pct']['weight']) - 10 # Account for interactions & wellness visit negative base

    # Test HIGH Fever impact
    features_high_fever = base_features.copy()
    features_high_fever['vital_signs_temperature_celsius'] = app_config.ALERT_BODY_TEMP_HIGH_FEVER_C + 0.1
    score_high_fever = risk_model_instance.predict_risk_score(features_high_fever)
    assert score_high_fever > base_score + (risk_model_instance.base_risk_factors['vital_signs_temperature_celsius']['factor_super_high'] * risk_model_instance.base_risk_factors['vital_signs_temperature_celsius']['weight']) - 10

    # Test Fall Detected
    features_fall = base_features.copy(); features_fall['fall_detected_today'] = 1
    score_fall = risk_model_instance.predict_risk_score(features_fall)
    assert score_fall > base_score # Specific factor applied

    # Test Chronic Condition Flag
    features_chronic = base_features.copy(); features_chronic['chronic_condition_flag'] = 1
    score_chronic = risk_model_instance.predict_risk_score(features_chronic)
    assert score_chronic > base_score + risk_model_instance.CHRONIC_CONDITION_FLAG_RISK_POINTS - 5 # Almost direct addition

    # Test PPE Non-Compliant Flag (Risk if 0)
    features_no_ppe = base_features.copy(); features_no_ppe['ppe_compliant_flag'] = 0
    score_no_ppe = risk_model_instance.predict_risk_score(features_no_ppe)
    assert score_no_ppe > base_score # Non-compliance adds risk

    # Test High Ambient Heat Index
    features_high_heat = base_features.copy(); features_high_heat['ambient_heat_index_c'] = app_config.ALERT_AMBIENT_HEAT_INDEX_DANGER_C + 1
    score_high_heat = risk_model_instance.predict_risk_score(features_high_heat)
    assert score_high_heat > base_score # Heat exposure adds risk

def test_risk_model_bulk_predict(risk_model_instance, sample_health_records_df_main_sentinel):
    if sample_health_records_df_main_sentinel.empty: pytest.skip("Sample health records for bulk risk test are empty.")
    
    # sample_health_records_df_main_sentinel fixture already includes AI scores from apply_ai_models
    # So, to test predict_bulk_risk_scores in isolation, we might need a version of the fixture *before* AI scores are added,
    # OR, we drop them before passing to this specific method.
    # For now, let's assume it's okay to run it on data that might already have them (it will overwrite).
    
    df_to_score = sample_health_records_df_main_sentinel.copy()
    # Drop existing AI scores if we want to test their fresh calculation by this model instance
    if 'ai_risk_score' in df_to_score.columns: df_to_score = df_to_score.drop(columns=['ai_risk_score'])

    risk_scores_series = risk_model_instance.predict_bulk_risk_scores(df_to_score)
    assert isinstance(risk_scores_series, pd.Series)
    assert len(risk_scores_series) == len(df_to_score)
    assert risk_scores_series.notna().all(), "Risk scores should not be NaN after clipping."
    assert risk_scores_series.min() >= 0 and risk_scores_series.max() <= 100, "Risk scores out of 0-100 bounds."


# --- Tests for FollowUpPrioritizer ---
@pytest.fixture(scope="module")
def priority_model_instance():
    return FollowUpPrioritizer()

def test_priority_model_helper_logics(priority_model_instance):
    assert priority_model_instance._has_active_critical_vitals_alert(pd.Series({'min_spo2_pct': app_config.ALERT_SPO2_CRITICAL_LOW_PCT - 1})) is True
    assert priority_model_instance._is_pending_urgent_task(pd.Series({'referral_status': 'Pending', 'condition': app_config.KEY_CONDITIONS_FOR_ACTION[0]})) is True
    assert priority_model_instance._has_acute_condition_severity(pd.Series({'condition': 'Pneumonia', 'min_spo2_pct': app_config.ALERT_SPO2_WARNING_LOW_PCT -1 })) is True
    assert priority_model_instance._contextual_hazard_present(pd.Series({'ambient_heat_index_c': app_config.ALERT_AMBIENT_HEAT_INDEX_DANGER_C + 1})) is True
    assert priority_model_instance._has_active_critical_vitals_alert(pd.Series({'min_spo2_pct': 98})) is False # Healthy


def test_priority_model_calculate_score_components(priority_model_instance):
    base_features = pd.Series({'ai_risk_score': 30.0}) # Low base AI risk
    score_base_only = priority_model_instance.calculate_priority_score(base_features.copy())
    
    # Critical vitals add significant points
    features_crit = base_features.copy(); features_crit['fall_detected_today'] = 1
    score_crit = priority_model_instance.calculate_priority_score(features_crit.copy())
    assert score_crit >= score_base_only + priority_model_instance.priority_weights['critical_vital_alert_points'] - 5 # small delta for other factors

    # Task overdue increases score
    score_overdue = priority_model_instance.calculate_priority_score(base_features.copy(), days_task_overdue=10)
    assert score_overdue >= score_base_only + (10 * priority_model_instance.priority_weights['task_overdue_factor_per_day']) - 5


def test_priority_model_generate_bulk_priorities(priority_model_instance, sample_health_records_df_main_sentinel):
    if sample_health_records_df_main_sentinel.empty: pytest.skip("Sample health records for bulk priority test are empty.")
    
    # `generate_followup_priorities` expects 'ai_risk_score'.
    # The sample_health_records_df_main_sentinel fixture already has AI scores applied by apply_ai_models.
    df_for_prio = sample_health_records_df_main_sentinel.copy()
    
    # Add 'days_task_overdue' if not present (as model expects it or defaults it)
    if 'days_task_overdue' not in df_for_prio.columns:
        df_for_prio['days_task_overdue'] = np.random.randint(0, 7, size=len(df_for_prio)) # Sample overdue days

    priority_scores_series = priority_model_instance.generate_followup_priorities(df_for_prio)
    assert isinstance(priority_scores_series, pd.Series)
    assert len(priority_scores_series) == len(df_for_prio)
    assert priority_scores_series.notna().all()
    assert priority_scores_series.min() >= 0 and priority_scores_series.max() <= 100


# --- Tests for SupplyForecastingModel (AI-Simulated) ---
@pytest.fixture(scope="module")
def supply_model_ai_instance():
    return SupplyForecastingModel()

def test_supply_model_ai_get_params(supply_model_ai_instance):
    # Test it gets params for a key drug from config
    if app_config.KEY_DRUG_SUBSTRINGS_SUPPLY:
        key_drug = app_config.KEY_DRUG_SUBSTRINGS_SUPPLY[0]
        params = supply_model_ai_instance._get_item_params(key_drug)
        assert "coeffs" in params and "trend" in params and "noise_std" in params
    # Test fallback for unknown item
    params_unknown = supply_model_ai_instance._get_item_params("Unknown Test Item 99")
    assert params_unknown["trend"] == 0.0005 # Should match default fallback

def test_supply_model_ai_forecast_structure_and_depletion(supply_model_ai_instance):
    # Input DF for AI model needs specific columns: item, current_stock, avg_daily_consumption_historical, last_stock_update_date
    current_supply_state_df = pd.DataFrame({
        'item': [app_config.KEY_DRUG_SUBSTRINGS_SUPPLY[0] if app_config.KEY_DRUG_SUBSTRINGS_SUPPLY else "ItemX", "ItemY"],
        'current_stock': [200.0, 80.0],
        'avg_daily_consumption_historical': [15.0, 5.0],
        'last_stock_update_date': pd.to_datetime(['2023-11-01', '2023-11-01'])
    })
    forecast_len_days = 10 # Short forecast for test
    ai_forecast_result_df = supply_model_ai_instance.forecast_supply_levels_advanced(current_supply_state_df, forecast_days_out=forecast_len_days)
    
    assert isinstance(ai_forecast_result_df, pd.DataFrame)
    if not ai_forecast_result_df.empty:
        assert len(ai_forecast_result_df['item'].unique()) <= 2
        assert len(ai_forecast_result_df) <= 2 * forecast_len_days # Each item gets rows for each forecast day
        
        expected_fc_cols = ['item', 'forecast_date', 'forecasted_stock_level', 
                            'forecasted_days_of_supply', 'predicted_daily_consumption', 
                            'estimated_stockout_date_ai']
        for col_fc in expected_fc_cols:
            assert col_fc in ai_forecast_result_df.columns, f"AI Supply forecast missing column: {col_fc}"
        
        # Check stock depletion for the first item
        item1_name = current_supply_state_df['item'].iloc[0]
        item1_fc_df = ai_forecast_result_df[ai_forecast_result_df['item'] == item1_name]
        if not item1_fc_df.empty and len(item1_fc_df) > 1:
            assert item1_fc_df['forecasted_stock_level'].iloc[-1] < item1_fc_df['forecasted_stock_level'].iloc[0] or item1_fc_df['forecasted_stock_level'].iloc[0] == 0


# --- Tests for Central apply_ai_models Function ---
def test_apply_ai_models_adds_sentinel_columns(sample_health_records_df_main_sentinel):
    if sample_health_records_df_main_sentinel.empty: pytest.skip("Sample health data empty for apply_ai_models test.")
    
    # The fixture already has AI scores applied. To test `apply_ai_models` properly adds them,
    # we should drop them from a copy first.
    df_before_ai = sample_health_records_df_main_sentinel.copy()
    if 'ai_risk_score' in df_before_ai.columns: df_before_ai = df_before_ai.drop(columns=['ai_risk_score'])
    if 'ai_followup_priority_score' in df_before_ai.columns: df_before_ai = df_before_ai.drop(columns=['ai_followup_priority_score'])
        
    enriched_df_applied, _ = apply_ai_models(df_before_ai) # Pass copy
    
    assert 'ai_risk_score' in enriched_df_applied.columns
    assert 'ai_followup_priority_score' in enriched_df_applied.columns
    assert len(enriched_df_applied) == len(df_before_ai)
    if not enriched_df_applied.empty:
        assert enriched_df_applied['ai_risk_score'].notna().all()
        assert enriched_df_applied['ai_followup_priority_score'].notna().all()

def test_apply_ai_models_handles_empty_input():
    df_empty_ai = pd.DataFrame()
    enriched_df, supply_df = apply_ai_models(df_empty_ai.copy()) # Pass copy
    assert enriched_df.empty
    assert supply_df is None # Or an empty DataFrame based on function's exact return for this case
