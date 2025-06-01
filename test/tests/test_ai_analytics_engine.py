# sentinel_project_root/test/tests/test_ai_analytics_engine.py
# Pytest tests for AI simulation logic in utils.ai_analytics_engine.py for Sentinel.

import pytest
import pandas as pd
import numpy as np
# from typing import Callable # Not directly used, but good to have if SUT takes callables

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

    # Test multi-condition string (model's _get_condition_base_score takes the max)
    if len(app_config.KEY_CONDITIONS_FOR_ACTION) >= 2:
        cond1 = app_config.KEY_CONDITIONS_FOR_ACTION[0]
        cond2 = "Pneumonia" # Assume Pneumonia is also in condition_base_scores
        multi_cond_str = f"{cond1}; {cond2}"
        expected_multi_score = max(
            risk_model_instance.condition_base_scores.get(cond1, 0.0),
            risk_model_instance.condition_base_scores.get(cond2, 0.0)
        )
        assert risk_model_instance._get_condition_base_score(multi_cond_str) == expected_multi_score, \
            "Multi-condition base score did not pick the max relevant score."
    
    assert risk_model_instance._get_condition_base_score("NonExistentCondition") == 0.0, "Non-existent condition should yield 0 base score."
    assert risk_model_instance._get_condition_base_score(None) == 0.0, "None condition should yield 0 base score."
    assert risk_model_instance._get_condition_base_score("") == 0.0, "Empty condition string should yield 0 base score."
    assert risk_model_instance._get_condition_base_score("Wellness Visit") < 0, \
        "Wellness Visit should have a negative base score as per model config."

def test_risk_model_predict_score_with_sentinel_factors(risk_model_instance: RiskPredictionModel):
    # Base features for a relatively healthy individual
    base_features = pd.Series({
        'condition': 'Wellness Visit', 'age': 35, 'chronic_condition_flag': 0,
        'min_spo2_pct': 98.0, 'vital_signs_temperature_celsius': 37.0,
        'fall_detected_today': 0, 'ambient_heat_index_c': 28.0,
        'ppe_compliant_flag': 1, 'signs_of_fatigue_observed_flag': 0,
        'rapid_psychometric_distress_score': 1.0, 'hrv_rmssd_ms': 55.0,
        'medication_adherence_self_report': 'Good', 'tb_contact_traced': 0
    })
    base_risk = risk_model_instance.predict_risk_score(base_features.copy())
    assert 0 <= base_risk <= 100, "Base risk score out of bounds."

    # Test Critical Low SpO2 impact
    features_low_spo2 = base_features.copy()
    features_low_spo2['min_spo2_pct'] = app_config.ALERT_SPO2_CRITICAL_LOW_PCT - 2 # Clearly below critical
    score_low_spo2 = risk_model_instance.predict_risk_score(features_low_spo2)
    # Calculate expected point addition from this factor alone
    spo2_factor_details = risk_model_instance.base_risk_factors['min_spo2_pct']
    spo2_points_added = spo2_factor_details['factor_low'] * spo2_factor_details['weight']
    # Allow a delta for interaction with negative base score of "Wellness Visit" and other small factors
    assert score_low_spo2 >= (base_risk + spo2_points_added - abs(risk_model_instance.condition_base_scores.get("Wellness Visit",0)) - 10), \
        "Critical SpO2 did not increase risk sufficiently considering base wellness score."

    # Test High Fever impact
    features_high_fever = base_features.copy()
    features_high_fever['vital_signs_temperature_celsius'] = app_config.ALERT_BODY_TEMP_HIGH_FEVER_C + 0.2
    score_high_fever = risk_model_instance.predict_risk_score(features_high_fever)
    fever_factor_details = risk_model_instance.base_risk_factors['vital_signs_temperature_celsius']
    fever_points_added = fever_factor_details['factor_super_high'] * fever_factor_details['weight']
    assert score_high_fever >= (base_risk + fever_points_added - abs(risk_model_instance.condition_base_scores.get("Wellness Visit",0)) - 10), \
        "High fever did not increase risk sufficiently."

    # Test Fall Detected
    features_fall = base_features.copy(); features_fall['fall_detected_today'] = 1
    score_fall = risk_model_instance.predict_risk_score(features_fall)
    fall_factor_details = risk_model_instance.base_risk_factors['fall_detected_today']
    fall_points_added = fall_factor_details['factor_true'] * fall_factor_details['weight']
    assert score_fall >= (base_risk + fall_points_added - abs(risk_model_instance.condition_base_scores.get("Wellness Visit",0)) - 5), \
        "Fall detection did not increase risk sufficiently."


def test_risk_model_bulk_predict(risk_model_instance: RiskPredictionModel, sample_health_records_df_main_sentinel: pd.DataFrame):
    if sample_health_records_df_main_sentinel.empty:
        pytest.skip("Sample health records empty for bulk risk prediction test.")
    
    df_to_score = sample_health_records_df_main_sentinel.copy()
    # Drop existing AI score if we want to test its fresh calculation by this model instance
    if 'ai_risk_score' in df_to_score.columns:
        df_to_score = df_to_score.drop(columns=['ai_risk_score'])

    risk_scores_series = risk_model_instance.predict_bulk_risk_scores(df_to_score)
    assert isinstance(risk_scores_series, pd.Series), "Bulk prediction did not return a Series."
    assert len(risk_scores_series) == len(df_to_score), "Bulk prediction Series length mismatch."
    assert risk_scores_series.notna().all(), "Risk scores from bulk prediction should not be NaN (due to clipping)."
    assert risk_scores_series.min() >= 0 and risk_scores_series.max() <= 100, \
        "Risk scores from bulk prediction are out of 0-100 bounds."

# --- Tests for FollowUpPrioritizer ---
@pytest.fixture(scope="module")
def priority_model_instance() -> FollowUpPrioritizer:
    return FollowUpPrioritizer()

def test_priority_model_helper_logics(priority_model_instance: FollowUpPrioritizer):
    assert priority_model_instance._has_active_critical_vitals_alert(pd.Series({'min_spo2_pct': app_config.ALERT_SPO2_CRITICAL_LOW_PCT - 1})) is True
    assert priority_model_instance._has_active_critical_vitals_alert(pd.Series({'vital_signs_temperature_celsius': app_config.ALERT_BODY_TEMP_HIGH_FEVER_C + 0.1})) is True
    assert priority_model_instance._has_active_critical_vitals_alert(pd.Series({'fall_detected_today': 1})) is True
    assert priority_model_instance._has_active_critical_vitals_alert(pd.Series({'min_spo2_pct': 98, 'vital_signs_temperature_celsius': 37.0, 'fall_detected_today': 0})) is False

    assert priority_model_instance._is_pending_urgent_task(pd.Series({'referral_status': 'Pending', 'condition': app_config.KEY_CONDITIONS_FOR_ACTION[0]})) is True
    assert priority_model_instance._is_pending_urgent_task(pd.Series({'worker_task_priority': 'Urgent'})) is True
    assert priority_model_instance._is_pending_urgent_task(pd.Series({'referral_status': 'Completed'})) is False

    assert priority_model_instance._has_acute_condition_severity(pd.Series({'condition': 'Pneumonia', 'min_spo2_pct': app_config.ALERT_SPO2_WARNING_LOW_PCT -1 })) is True
    assert priority_model_instance._has_acute_condition_severity(pd.Series({'condition': 'Sepsis'})) is True
    assert priority_model_instance._has_acute_condition_severity(pd.Series({'condition': 'Wellness Visit'})) is False

    assert priority_model_instance._contextual_hazard_present(pd.Series({'ambient_heat_index_c': app_config.ALERT_AMBIENT_HEAT_INDEX_DANGER_C + 1})) is True
    assert priority_model_instance._contextual_hazard_present(pd.Series({'ambient_heat_index_c': 25})) is False


def test_priority_model_calculate_score_components(priority_model_instance: FollowUpPrioritizer):
    base_features = pd.Series({'ai_risk_score': 25.0}) # Low base AI risk
    score_base_only = priority_model_instance.calculate_priority_score(base_features.copy())
    
    features_critical_vitals = base_features.copy(); features_critical_vitals['min_spo2_pct'] = app_config.ALERT_SPO2_CRITICAL_LOW_PCT - 1
    score_critical_vitals = priority_model_instance.calculate_priority_score(features_critical_vitals.copy())
    assert score_critical_vitals >= score_base_only + priority_model_instance.priority_weights['critical_vital_alert_points'] - 10, "Critical vitals points not sufficiently added."

    score_overdue_task = priority_model_instance.calculate_priority_score(base_features.copy(), days_task_overdue=7)
    assert score_overdue_task >= score_base_only + (7 * priority_model_instance.priority_weights['task_overdue_factor_per_day']) - 5, "Task overdue factor impact incorrect."


def test_priority_model_generate_bulk_priorities(priority_model_instance: FollowUpPrioritizer, sample_health_records_df_main_sentinel: pd.DataFrame):
    if sample_health_records_df_main_sentinel.empty:
        pytest.skip("Sample health records empty for bulk priority test.")
    
    df_for_prio = sample_health_records_df_main_sentinel.copy()
    # The fixture should already have 'ai_risk_score'. Add 'days_task_overdue'.
    if 'days_task_overdue' not in df_for_prio.columns:
        df_for_prio['days_task_overdue'] = np.random.randint(0, 8, size=len(df_for_prio))

    priority_scores_series = priority_model_instance.generate_followup_priorities(df_for_prio)
    assert isinstance(priority_scores_series, pd.Series)
    assert len(priority_scores_series) == len(df_for_prio)
    assert priority_scores_series.notna().all(), "Priority scores from bulk gen should not be NaN."
    assert priority_scores_series.min() >= 0 and priority_scores_series.max() <= 100, "Priority scores out of 0-100 bounds."


# --- Tests for SupplyForecastingModel (AI-Simulated) ---
@pytest.fixture(scope="module")
def supply_model_ai_instance() -> SupplyForecastingModel:
    return SupplyForecastingModel()

def test_supply_model_ai_get_item_params(supply_model_ai_instance: SupplyForecastingModel):
    if not app_config.KEY_DRUG_SUBSTRINGS_SUPPLY:
        pytest.skip("No KEY_DRUG_SUBSTRINGS_SUPPLY in app_config to test supply model params.")
    
    a_key_drug = app_config.KEY_DRUG_SUBSTRINGS_SUPPLY[0]
    params = supply_model_ai_instance._get_item_params(a_key_drug)
    assert isinstance(params, dict)
    assert all(k in params for k in ["coeffs", "trend", "noise_std"]), f"Missing keys in params for drug '{a_key_drug}'."

    params_unknown = supply_model_ai_instance._get_item_params("TotallyRandomNewItem123")
    assert params_unknown["trend"] == 0.0001, "Fallback trend for unknown item differs from expected default."


def test_supply_model_ai_forecast_output(supply_model_ai_instance: SupplyForecastingModel):
    test_item_name = app_config.KEY_DRUG_SUBSTRINGS_SUPPLY[0] if app_config.KEY_DRUG_SUBSTRINGS_SUPPLY else "TestSupplyItem"
    
    supply_input_df = pd.DataFrame({
        'item': [test_item_name, "AnotherItem"],
        'current_stock': [300.0, 150.0],
        'avg_daily_consumption_historical': [10.0, 5.0], # Ensure positive consumption
        'last_stock_update_date': pd.to_datetime(['2023-11-15', '2023-11-15'])
    })
    forecast_days = 20
    forecast_output_df = supply_model_ai_instance.forecast_supply_levels_advanced(supply_input_df, forecast_days_out=forecast_days)
    
    assert isinstance(forecast_output_df, pd.DataFrame), "AI Supply forecast did not return a DataFrame."
    if not forecast_output_df.empty:
        assert len(forecast_output_df['item'].unique()) <= 2
        assert len(forecast_output_df) <= 2 * forecast_days
        
        expected_cols = ['item', 'forecast_date', 'forecasted_stock_level', 
                         'forecasted_days_of_supply', 'predicted_daily_consumption', 
                         'estimated_stockout_date_ai']
        for col_name_fc in expected_cols:
            assert col_name_fc in forecast_output_df.columns, f"AI Supply forecast output missing column: {col_name_fc}"
        
        # Check stock depletion for the first test item
        item1_forecast_data = forecast_output_df[forecast_output_df['item'] == test_item_name]
        if not item1_forecast_data.empty and len(item1_forecast_data) > 1:
            initial_stock = item1_forecast_data['forecasted_stock_level'].iloc[0]
            final_stock = item1_forecast_data['forecasted_stock_level'].iloc[-1]
            # Stock should deplete or stay at 0 if it started at 0 or consumption is minimal
            assert final_stock < initial_stock or initial_stock == 0 or (np.isclose(final_stock, initial_stock) and supply_input_df.loc[supply_input_df['item']==test_item_name, 'avg_daily_consumption_historical'].iloc[0] < 1e-5), \
                   f"Stock for '{test_item_name}' did not deplete as expected."


# --- Tests for Central apply_ai_models Function ---
def test_apply_ai_models_adds_columns_and_maintains_rows(sample_health_records_df_main_sentinel: pd.DataFrame):
    if sample_health_records_df_main_sentinel.empty:
        pytest.skip("Sample health data empty for apply_ai_models test.")
    
    df_input_ai_apply = sample_health_records_df_main_sentinel.copy()
    # Drop AI scores if they exist from the fixture (as fixture itself calls apply_ai_models)
    cols_to_drop_for_test = ['ai_risk_score', 'ai_followup_priority_score']
    for col_to_drop in cols_to_drop_for_test:
        if col_to_drop in df_input_ai_apply.columns:
            df_input_ai_apply = df_input_ai_apply.drop(columns=[col_to_drop])
            
    enriched_df_output, _ = apply_ai_models(df_input_ai_apply.copy()) # Pass copy
    
    assert 'ai_risk_score' in enriched_df_output.columns, "'ai_risk_score' column not added by apply_ai_models."
    assert 'ai_followup_priority_score' in enriched_df_output.columns, "'ai_followup_priority_score' column not added."
    assert len(enriched_df_output) == len(df_input_ai_apply), "Row count changed after apply_ai_models."
    if not enriched_df_output.empty:
        assert enriched_df_output['ai_risk_score'].notna().all(), "NaNs found in 'ai_risk_score' after apply_ai_models."
        assert enriched_df_output['ai_followup_priority_score'].notna().all(), "NaNs found in 'ai_followup_priority_score'."

def test_apply_ai_models_handles_empty_and_none_input():
    # Test with empty DataFrame
    empty_df = pd.DataFrame(columns=['encounter_id', 'patient_id']) # Example with some base columns
    enriched_df_empty_out, supply_df_empty_out = apply_ai_models(empty_df.copy())
    
    assert isinstance(enriched_df_empty_out, pd.DataFrame), "Should return DataFrame for empty input."
    assert 'ai_risk_score' in enriched_df_empty_out.columns and 'ai_followup_priority_score' in enriched_df_empty_out.columns, \
        "Empty output DF from apply_ai_models missing expected AI columns."
    assert enriched_df_empty_out.empty, "Enriched DF should be empty for empty input."
    assert supply_df_empty_out is None, "Supply DF should be None when health_df is empty and no supply_status_df passed."

    # Test with None input
    enriched_df_none_out, supply_df_none_out = apply_ai_models(None) # type: ignore
    assert isinstance(enriched_df_none_out, pd.DataFrame), "Should return DataFrame for None input."
    assert 'ai_risk_score' in enriched_df_none_out.columns and 'ai_followup_priority_score' in enriched_df_none_out.columns
    assert enriched_df_none_out.empty, "Enriched DF should be empty for None input."
    assert supply_df_none_out is None, "Supply DF should be None for None health_df input."
