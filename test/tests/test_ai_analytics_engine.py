# sentinel_project_root/test/tests/test_ai_analytics_engine.py
# Pytest tests for AI simulation logic in utils.ai_analytics_engine.py for Sentinel.

import pytest
import pandas as pd
import numpy as np
# from datetime import datetime, timedelta # Not directly used in this test file after fixture usage

# Classes and functions to be tested
from utils.ai_analytics_engine import (
    RiskPredictionModel,
    FollowUpPrioritizer,
    SupplyForecastingModel,
    apply_ai_models
)
from config import app_config # For thresholds, keys, etc.

# Fixtures (e.g., sample_health_records_df_main_sentinel) are sourced from conftest.py

# --- Tests for RiskPredictionModel ---

@pytest.fixture(scope="module")
def risk_model_instance() -> RiskPredictionModel:
    """Provides a single instance of RiskPredictionModel for tests in this module."""
    return RiskPredictionModel()

def test_risk_model_condition_base_score_sentinel(risk_model_instance: RiskPredictionModel):
    if not app_config.KEY_CONDITIONS_FOR_ACTION:
        pytest.skip("No KEY_CONDITIONS_FOR_ACTION defined in app_config to test.")
        
    for condition in app_config.KEY_CONDITIONS_FOR_ACTION:
        expected_score = risk_model_instance.condition_base_scores.get(condition, 0.0)
        assert risk_model_instance._get_condition_base_score(condition) == expected_score, \
            f"Base score mismatch for key condition: {condition}"
        assert risk_model_instance._get_condition_base_score(condition.lower()) == expected_score, \
            f"Base score mismatch for lowercase key condition: {condition.lower()}"

    # Test multi-condition string (should pick the max relevant score or combine based on model logic)
    # Current _get_condition_base_score takes the max if multiple are found.
    multi_cond_str = f"{app_config.KEY_CONDITIONS_FOR_ACTION[0]};Pneumonia"
    expected_multi_score = max(
        risk_model_instance.condition_base_scores.get(app_config.KEY_CONDITIONS_FOR_ACTION[0], 0.0),
        risk_model_instance.condition_base_scores.get("Pneumonia", 0.0)
    )
    assert risk_model_instance._get_condition_base_score(multi_cond_str) == expected_multi_score, \
        "Multi-condition base score mismatch."
    
    assert risk_model_instance._get_condition_base_score("NonExistentCondition") == 0.0
    assert risk_model_instance._get_condition_base_score(None) == 0.0
    assert risk_model_instance._get_condition_base_score("") == 0.0
    assert risk_model_instance._get_condition_base_score("Wellness Visit") < 0, \
        "Wellness Visit should have a negative base score."

def test_risk_model_predict_score_with_sentinel_factors(risk_model_instance: RiskPredictionModel):
    base_features = pd.Series({
        'condition': 'Wellness Visit', 'age': 35, 'chronic_condition_flag': 0,
        'min_spo2_pct': 98, 'vital_signs_temperature_celsius': 37.0,
        'fall_detected_today': 0, 'ambient_heat_index_c': 28.0,
        'ppe_compliant_flag': 1, 'signs_of_fatigue_observed_flag': 0,
        'rapid_psychometric_distress_score': 1.0, 'hrv_rmssd_ms': 55.0,
        'medication_adherence_self_report': 'Good', 'tb_contact_traced': 0
    })
    base_risk = risk_model_instance.predict_risk_score(base_features.copy())
    assert 0 <= base_risk <= 100, "Base risk score out of bounds."

    # Test Critical Low SpO2
    features_low_spo2 = base_features.copy()
    features_low_spo2['min_spo2_pct'] = app_config.ALERT_SPO2_CRITICAL_LOW_PCT - 1
    score_low_spo2 = risk_model_instance.predict_risk_score(features_low_spo2)
    spo2_factor_points = risk_model_instance.base_risk_factors['min_spo2_pct']['factor_low'] * \
                         risk_model_instance.base_risk_factors['min_spo2_pct']['weight']
    assert score_low_spo2 >= base_risk + spo2_factor_points - 15, "Critical SpO2 did not increase risk sufficiently." # Allow for interaction damping

    # Test High Fever
    features_high_fever = base_features.copy()
    features_high_fever['vital_signs_temperature_celsius'] = app_config.ALERT_BODY_TEMP_HIGH_FEVER_C + 0.1
    score_high_fever = risk_model_instance.predict_risk_score(features_high_fever)
    fever_factor_points = risk_model_instance.base_risk_factors['vital_signs_temperature_celsius']['factor_super_high'] * \
                          risk_model_instance.base_risk_factors['vital_signs_temperature_celsius']['weight']
    assert score_high_fever >= base_risk + fever_factor_points - 15, "High fever did not increase risk sufficiently."

    # Test Fall Detected
    features_fall = base_features.copy(); features_fall['fall_detected_today'] = 1
    score_fall = risk_model_instance.predict_risk_score(features_fall)
    fall_factor_points = risk_model_instance.base_risk_factors['fall_detected_today']['factor_true'] * \
                         risk_model_instance.base_risk_factors['fall_detected_today']['weight']
    assert score_fall >= base_risk + fall_factor_points - 10, "Fall detection did not increase risk sufficiently."

    # Test Chronic Condition Flag
    features_chronic = base_features.copy(); features_chronic['chronic_condition_flag'] = 1
    score_chronic = risk_model_instance.predict_risk_score(features_chronic)
    assert score_chronic >= base_risk + risk_model_instance.CHRONIC_CONDITION_FLAG_RISK_POINTS - 5, "Chronic condition flag impact incorrect."

    # Test PPE Non-Compliant (flag_value = 0 for risk)
    features_no_ppe = base_features.copy(); features_no_ppe['ppe_compliant_flag'] = 0
    score_no_ppe = risk_model_instance.predict_risk_score(features_no_ppe)
    ppe_factor_points = risk_model_instance.base_risk_factors['ppe_compliant_flag']['factor_true'] * \
                        risk_model_instance.base_risk_factors['ppe_compliant_flag']['weight']
    assert score_no_ppe >= base_risk + ppe_factor_points - 10, "PPE non-compliance did not increase risk sufficiently."
    
    # Test High Ambient Heat Index (Danger level)
    features_high_heat = base_features.copy(); features_high_heat['ambient_heat_index_c'] = app_config.ALERT_AMBIENT_HEAT_INDEX_DANGER_C + 1
    score_high_heat = risk_model_instance.predict_risk_score(features_high_heat)
    heat_factor_points = risk_model_instance.base_risk_factors['ambient_heat_index_c']['factor_super_high'] * \
                         risk_model_instance.base_risk_factors['ambient_heat_index_c']['weight']
    assert score_high_heat >= base_risk + heat_factor_points - 10, "High ambient heat did not increase risk sufficiently."


def test_risk_model_bulk_predict(risk_model_instance: RiskPredictionModel, sample_health_records_df_main_sentinel: pd.DataFrame):
    if sample_health_records_df_main_sentinel.empty:
        pytest.skip("Sample health records empty for bulk risk prediction test.")
    
    df_to_score_bulk = sample_health_records_df_main_sentinel.copy()
    # Drop existing AI score if we want to test its fresh calculation
    if 'ai_risk_score' in df_to_score_bulk.columns:
        df_to_score_bulk = df_to_score_bulk.drop(columns=['ai_risk_score'])

    risk_scores_series_bulk = risk_model_instance.predict_bulk_risk_scores(df_to_score_bulk)
    assert isinstance(risk_scores_series_bulk, pd.Series), "Bulk prediction did not return a Series."
    assert len(risk_scores_series_bulk) == len(df_to_score_bulk), "Bulk prediction Series length mismatch."
    assert risk_scores_series_bulk.notna().all(), "Risk scores from bulk prediction should not be NaN after clipping."
    assert risk_scores_series_bulk.min() >= 0 and risk_scores_series_bulk.max() <= 100, \
        "Risk scores from bulk prediction are out of 0-100 bounds."

# --- Tests for FollowUpPrioritizer ---
@pytest.fixture(scope="module")
def priority_model_instance() -> FollowUpPrioritizer:
    return FollowUpPrioritizer()

def test_priority_model_helper_logics(priority_model_instance: FollowUpPrioritizer):
    assert priority_model_instance._has_active_critical_vitals_alert(pd.Series({'min_spo2_pct': app_config.ALERT_SPO2_CRITICAL_LOW_PCT - 1})) is True, "Critical SpO2 not detected by helper."
    assert priority_model_instance._has_active_critical_vitals_alert(pd.Series({'vital_signs_temperature_celsius': app_config.ALERT_BODY_TEMP_HIGH_FEVER_C + 0.1})) is True, "High fever not detected by helper."
    assert priority_model_instance._is_pending_urgent_task(pd.Series({'referral_status': 'Pending', 'condition': app_config.KEY_CONDITIONS_FOR_ACTION[0]})) is True, "Pending urgent task not detected."
    assert priority_model_instance._has_acute_condition_severity(pd.Series({'condition': 'Pneumonia', 'min_spo2_pct': app_config.ALERT_SPO2_WARNING_LOW_PCT -1 })) is True, "Acute pneumonia severity not detected."
    assert priority_model_instance._contextual_hazard_present(pd.Series({'ambient_heat_index_c': app_config.ALERT_AMBIENT_HEAT_INDEX_DANGER_C + 1})) is True, "Contextual hazard (heat) not detected."
    assert priority_model_instance._has_active_critical_vitals_alert(pd.Series({'min_spo2_pct': 98, 'vital_signs_temperature_celsius': 37.0})) is False, "Healthy vitals incorrectly flagged."

def test_priority_model_calculate_score_components(priority_model_instance: FollowUpPrioritizer):
    base_features_prio = pd.Series({'ai_risk_score': 30.0}) # Low base AI risk
    score_base = priority_model_instance.calculate_priority_score(base_features_prio.copy())
    
    features_crit_vitals = base_features_prio.copy(); features_crit_vitals['fall_detected_today'] = 1
    score_crit_vitals = priority_model_instance.calculate_priority_score(features_crit_vitals.copy())
    assert score_crit_vitals >= score_base + priority_model_instance.priority_weights['critical_vital_alert_points'] - 10, "Critical vitals alert points not added sufficiently."

    score_task_overdue = priority_model_instance.calculate_priority_score(base_features_prio.copy(), days_task_overdue=5)
    assert score_task_overdue >= score_base + (5 * priority_model_instance.priority_weights['task_overdue_factor_per_day']) - 5, "Task overdue factor not applied correctly."

def test_priority_model_generate_bulk_priorities(priority_model_instance: FollowUpPrioritizer, sample_health_records_df_main_sentinel: pd.DataFrame):
    if sample_health_records_df_main_sentinel.empty:
        pytest.skip("Sample health records empty for bulk priority test.")
    
    df_for_prio_bulk = sample_health_records_df_main_sentinel.copy()
    # `generate_followup_priorities` expects 'ai_risk_score'. The fixture already has it.
    # Add 'days_task_overdue' if not present, as model expects it or defaults it.
    if 'days_task_overdue' not in df_for_prio_bulk.columns:
        df_for_prio_bulk['days_task_overdue'] = np.random.randint(0, 5, size=len(df_for_prio_bulk))

    priority_scores_series_bulk = priority_model_instance.generate_followup_priorities(df_for_prio_bulk)
    assert isinstance(priority_scores_series_bulk, pd.Series), "Bulk priority generation did not return a Series."
    assert len(priority_scores_series_bulk) == len(df_for_prio_bulk), "Bulk priority Series length mismatch."
    assert priority_scores_series_bulk.notna().all(), "Priority scores from bulk generation should not be NaN."
    assert priority_scores_series_bulk.min() >= 0 and priority_scores_series_bulk.max() <= 100, \
        "Priority scores from bulk generation are out of 0-100 bounds."

# --- Tests for SupplyForecastingModel (AI-Simulated) ---
@pytest.fixture(scope="module")
def supply_model_ai_instance() -> SupplyForecastingModel:
    return SupplyForecastingModel()

def test_supply_model_ai_get_params(supply_model_ai_instance: SupplyForecastingModel):
    if not app_config.KEY_DRUG_SUBSTRINGS_SUPPLY:
        pytest.skip("No KEY_DRUG_SUBSTRINGS_SUPPLY in app_config to test AI supply model params.")
    
    first_key_drug = app_config.KEY_DRUG_SUBSTRINGS_SUPPLY[0]
    params_drug = supply_model_ai_instance._get_item_params(first_key_drug)
    assert isinstance(params_drug, dict)
    assert all(k in params_drug for k in ["coeffs", "trend", "noise_std"]), f"Missing keys in params for '{first_key_drug}'."

    params_unknown_item = supply_model_ai_instance._get_item_params("ThisItemDoesNotExistInConfig999")
    assert params_unknown_item["trend"] == 0.0001, "Fallback trend for unknown item is incorrect." # Check against expected default

def test_supply_model_ai_forecast_structure_and_depletion(supply_model_ai_instance: SupplyForecastingModel):
    item_name_supply_test = app_config.KEY_DRUG_SUBSTRINGS_SUPPLY[0] if app_config.KEY_DRUG_SUBSTRINGS_SUPPLY else "TestItemSupply"
    
    current_supply_df_test = pd.DataFrame({
        'item': [item_name_supply_test, "AnotherTestItem"],
        'current_stock': [250.0, 100.0],
        'avg_daily_consumption_historical': [12.0, 6.0], # Ensure this is positive
        'last_stock_update_date': pd.to_datetime(['2023-12-01', '2023-12-01'])
    })
    forecast_horizon_days = 15
    ai_forecast_df = supply_model_ai_instance.forecast_supply_levels_advanced(current_supply_df_test, forecast_days_out=forecast_horizon_days)
    
    assert isinstance(ai_forecast_df, pd.DataFrame), "AI Supply forecast did not return a DataFrame."
    if not ai_forecast_df.empty:
        assert len(ai_forecast_df['item'].unique()) <= 2, "Forecast contains unexpected number of items."
        assert len(ai_forecast_df) <= 2 * forecast_horizon_days, "Forecast contains too many daily records."
        
        expected_forecast_cols = ['item', 'forecast_date', 'forecasted_stock_level', 
                                  'forecasted_days_of_supply', 'predicted_daily_consumption', 
                                  'estimated_stockout_date_ai']
        for col_fc_check in expected_forecast_cols:
            assert col_fc_check in ai_forecast_df.columns, f"AI Supply forecast missing column: {col_fc_check}"
        
        # Check stock depletion for the first item if it has consumption
        item1_data_fc = ai_forecast_df[ai_forecast_df['item'] == item_name_supply_test]
        if not item1_data_fc.empty and len(item1_data_fc) > 1 and current_supply_df_test['avg_daily_consumption_historical'].iloc[0] > 1e-6:
            assert item1_data_fc['forecasted_stock_level'].iloc[-1] < item1_data_fc['forecasted_stock_level'].iloc[0] or \
                   item1_data_fc['forecasted_stock_level'].iloc[0] == 0, "Stock did not deplete for item with consumption."

# --- Tests for Central apply_ai_models Function ---
def test_apply_ai_models_adds_sentinel_columns(sample_health_records_df_main_sentinel: pd.DataFrame):
    if sample_health_records_df_main_sentinel.empty:
        pytest.skip("Sample health data empty for apply_ai_models test.")
    
    df_input_for_apply_ai = sample_health_records_df_main_sentinel.copy()
    # Drop AI scores if they exist from the fixture to test their fresh addition
    cols_to_drop_if_exist = ['ai_risk_score', 'ai_followup_priority_score']
    for col_drop in cols_to_drop_if_exist:
        if col_drop in df_input_for_apply_ai.columns:
            df_input_for_apply_ai = df_input_for_apply_ai.drop(columns=[col_drop])
            
    enriched_df_result, _ = apply_ai_models(df_input_for_apply_ai) # Pass copy
    
    assert 'ai_risk_score' in enriched_df_result.columns, "'ai_risk_score' not added by apply_ai_models."
    assert 'ai_followup_priority_score' in enriched_df_result.columns, "'ai_followup_priority_score' not added."
    assert len(enriched_df_result) == len(df_input_for_apply_ai), "Row count changed after apply_ai_models."
    if not enriched_df_result.empty:
        assert enriched_df_result['ai_risk_score'].notna().all(), "NaNs found in 'ai_risk_score' after apply_ai_models."
        assert enriched_df_result['ai_followup_priority_score'].notna().all(), "NaNs found in 'ai_followup_priority_score'."

def test_apply_ai_models_handles_empty_or_invalid_input():
    # Test with empty DataFrame
    df_empty = pd.DataFrame()
    enriched_df_empty, supply_df_empty = apply_ai_models(df_empty.copy())
    assert isinstance(enriched_df_empty, pd.DataFrame), "Should return DataFrame for empty input."
    assert 'ai_risk_score' in enriched_df_empty.columns and 'ai_followup_priority_score' in enriched_df_empty.columns, \
        "Empty output DF from apply_ai_models missing expected AI columns."
    assert enriched_df_empty.empty, "Enriched DF should be empty for empty input."
    assert supply_df_empty is None, "Supply DF should be None for empty health_df input." # Or empty DF, depends on exact impl.

    # Test with None input
    enriched_df_none, supply_df_none = apply_ai_models(None) # type: ignore
    assert isinstance(enriched_df_none, pd.DataFrame) and enriched_df_none.empty
    assert 'ai_risk_score' in enriched_df_none.columns and 'ai_followup_priority_score' in enriched_df_none.columns
    assert supply_df_none is None
