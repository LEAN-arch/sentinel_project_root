# sentinel_project_root/test/tests/test_core_data_processing.py
# Pytest tests for the refactored functions in utils.core_data_processing.py
# Aligned with "Sentinel Health Co-Pilot" redesign.

import pytest
import pandas as pd
import geopandas as gpd
import numpy as np
from datetime import date, datetime, timedelta # For date comparisons and creating sample data

# Functions to be tested (from the refactored utils.core_data_processing)
from utils.core_data_processing import (
    _clean_column_names,
    _convert_to_numeric,
    hash_geodataframe,
    load_health_records,
    load_iot_clinic_environment_data,
    load_zone_data,
    enrich_zone_geodata_with_health_aggregates,
    get_overall_kpis,
    get_chw_summary,
    get_patient_alerts_for_chw, # Returns List[Dict]
    get_clinic_summary,
    get_clinic_environmental_summary,
    get_patient_alerts_for_clinic, # Returns DataFrame
    get_district_summary_kpis,
    get_trend_data,
    get_supply_forecast_data # Simple linear forecast
)

# Import app_config to use its thresholds and keys in test assertions
from config import app_config # The NEW, redesigned app_config

# Fixtures are imported automatically by pytest from conftest.py
# We expect: sample_health_records_df_main_sentinel, sample_iot_clinic_df_main_sentinel,
#            sample_zone_geometries_gdf_main_sentinel, sample_enriched_gdf_main_sentinel, etc.

# --- Tests for Helper Functions ---
def test_clean_column_names_functionality():
    df_dirty = pd.DataFrame(columns=['Test Column', 'Another-Col SPACE', 'already_good'])
    df_cleaned = _clean_column_names(df_dirty.copy())
    assert list(df_cleaned.columns) == ['test_column', 'another_col_space', 'already_good']
    assert list(_clean_column_names(pd.DataFrame()).columns) == []
    assert list(_clean_column_names(df_cleaned.copy()).columns) == ['test_column', 'another_col_space', 'already_good']

def test_convert_to_numeric_functionality():
    s_dirty = pd.Series(['101', '23.5', 'bad_data', None, '55'])
    s_numeric_nan = _convert_to_numeric(s_dirty.copy())
    pd.testing.assert_series_equal(s_numeric_nan, pd.Series([101.0, 23.5, np.nan, np.nan, 55.0]), check_dtype=False)
    s_numeric_zero = _convert_to_numeric(s_dirty.copy(), default_value=0)
    pd.testing.assert_series_equal(s_numeric_zero, pd.Series([101.0, 23.5, 0.0, 0.0, 55.0]), check_dtype=False)
    s_already_num = pd.Series([1, 2.0, 3])
    pd.testing.assert_series_equal(_convert_to_numeric(s_already_num.copy()), pd.Series([1.0, 2.0, 3.0]), check_dtype=False)

def test_hash_geodataframe_functionality(sample_zone_geometries_gdf_main_sentinel):
    if sample_zone_geometries_gdf_main_sentinel.empty:
        pytest.skip("Sample GDF for hashing is empty.")
    gdf_hash1 = hash_geodataframe(sample_zone_geometries_gdf_main_sentinel.copy())
    assert isinstance(gdf_hash1, str) and gdf_hash1 is not None
    assert hash_geodataframe(None) is None
    assert hash_geodataframe(gpd.GeoDataFrame()) is not None # Hash for empty structure
    gdf_modified = sample_zone_geometries_gdf_main_sentinel.copy()
    if 'population' in gdf_modified.columns and not gdf_modified.empty:
        pop_val = gdf_modified.loc[0, 'population']
        gdf_modified.loc[0, 'population'] = (pop_val + 100) if pd.notna(pop_val) else 100
        gdf_hash2 = hash_geodataframe(gdf_modified)
        assert isinstance(gdf_hash2, str) and gdf_hash1 != gdf_hash2
    else:
        logger.warning("Skipping GDF hash modification test; 'population' missing or GDF empty.")


# --- Tests for Data Loading ---
# (Simplified structure tests, actual loading depends on files existing which is hard to mock without tmp_path for all)
def test_load_health_records_sentinel_cols(sample_health_records_df_main_sentinel):
    df = sample_health_records_df_main_sentinel
    assert isinstance(df, pd.DataFrame)
    key_cols = ['patient_id', 'encounter_date', 'ai_risk_score', 'min_spo2_pct']
    for col in key_cols: assert col in df.columns
    if not df.empty: assert pd.api.types.is_datetime64_any_dtype(df['encounter_date'])

def test_load_iot_data_sentinel_cols(sample_iot_clinic_df_main_sentinel):
    df = sample_iot_clinic_df_main_sentinel
    assert isinstance(df, pd.DataFrame)
    key_cols = ['timestamp', 'clinic_id', 'room_name', 'avg_co2_ppm']
    for col in key_cols: assert col in df.columns
    if not df.empty: assert pd.api.types.is_datetime64_any_dtype(df['timestamp'])

def test_load_zone_data_sentinel_cols(sample_zone_geometries_gdf_main_sentinel): # This fixture simulates output of load_zone_data
    gdf = sample_zone_geometries_gdf_main_sentinel
    assert isinstance(gdf, gpd.GeoDataFrame)
    key_cols = ['zone_id', 'name', 'population', 'geometry']
    for col in key_cols: assert col in gdf.columns
    if not gdf.empty: assert gdf.crs is not None

# --- Test for Enrichment ---
def test_enrich_zone_geodata_values_sentinel(sample_enriched_gdf_main_sentinel, sample_health_records_df_main_sentinel):
    if sample_enriched_gdf_main_sentinel.empty or sample_health_records_df_main_sentinel.empty:
        pytest.skip("Cannot test enrichment values with empty input fixtures for Sentinel.")
    gdf = sample_enriched_gdf_main_sentinel
    health_df = sample_health_records_df_main_sentinel
    
    # Test a specific aggregated value for a known zone, e.g., 'ZoneA'
    zone_to_test = 'ZoneA'
    if zone_to_test not in gdf['zone_id'].tolist():
        pytest.skip(f"Zone '{zone_to_test}' not found in sample_enriched_gdf_main_sentinel.")
    
    # Example 1: active_tb_cases (or any other condition from KEY_CONDITIONS_FOR_ACTION)
    tb_cond_col_name = f"active_tb_cases" # Ensure this matches the dynamic column name generation
    if tb_cond_col_name in gdf.columns:
        expected_tb_zone_a = health_df[(health_df['zone_id'] == zone_to_test) & (health_df['condition'].str.contains("TB", case=False, na=False))]['patient_id'].nunique()
        actual_tb_zone_a = gdf[gdf['zone_id'] == zone_to_test][tb_cond_col_name].iloc[0]
        assert actual_tb_zone_a == expected_tb_zone_a, f"Mismatch in {tb_cond_col_name} for {zone_to_test}"

    # Example 2: avg_risk_score
    expected_avg_risk_zone_a = health_df[health_df['zone_id'] == zone_to_test]['ai_risk_score'].mean()
    actual_avg_risk_zone_a = gdf[gdf['zone_id'] == zone_to_test]['avg_risk_score'].iloc[0]
    if pd.notna(expected_avg_risk_zone_a) and pd.notna(actual_avg_risk_zone_a):
        assert np.isclose(actual_avg_risk_zone_a, expected_avg_risk_zone_a)
    else:
        assert pd.isna(actual_avg_risk_zone_a) == pd.isna(expected_avg_risk_zone_a)


# --- Tests for KPI and Summary Calculation Functions (Sentinel Context) ---

def test_get_clinic_summary_sentinel_structure(sample_health_records_df_main_sentinel):
    df_period = sample_health_records_df_main_sentinel[
        (sample_health_records_df_main_sentinel['encounter_date'] >= pd.Timestamp('2023-10-01')) &
        (sample_health_records_df_main_sentinel['encounter_date'] <= pd.Timestamp('2023-10-07'))
    ]
    if df_period.empty: pytest.skip("No data for clinic summary test period in sample.")
    
    summary = get_clinic_summary(df_period)
    assert isinstance(summary, dict)
    expected_top_keys = [
        "overall_avg_test_turnaround_conclusive_days", "perc_critical_tests_tat_met",
        "total_pending_critical_tests_patients", "sample_rejection_rate_perc",
        "key_drug_stockouts_count", "test_summary_details"
    ]
    for key in expected_top_keys: assert key in summary, f"Key '{key}' missing in clinic_summary"
    
    assert isinstance(summary["test_summary_details"], dict)
    # Check structure of one entry in test_summary_details
    # Ensure display names from app_config.KEY_TEST_TYPES_FOR_ANALYSIS are used as keys
    example_test_display_name = app_config.KEY_TEST_TYPES_FOR_ANALYSIS["RDT-Malaria"]["display_name"]
    if example_test_display_name in summary["test_summary_details"]:
        test_detail = summary["test_summary_details"][example_test_display_name]
        expected_detail_keys = ["positive_rate_perc", "avg_tat_days", "perc_met_tat_target",
                                "pending_count_patients", "rejected_count_patients", "total_conclusive_tests"]
        for d_key in expected_detail_keys: assert d_key in test_detail, f"Detail key '{d_key}' missing for {example_test_display_name}"


def test_get_clinic_environmental_summary_sentinel_kpis(sample_iot_clinic_df_main_sentinel):
    if sample_iot_clinic_df_main_sentinel.empty: pytest.skip("No IoT data for env summary test.")
    summary = get_clinic_environmental_summary(sample_iot_clinic_df_main_sentinel)
    assert isinstance(summary, dict)
    # Test specific KPI values based on sample_iot_clinic_df_main_sentinel and app_config thresholds
    # e.g., avg_co2_overall_ppm, rooms_co2_very_high_alert_latest_count
    assert 'avg_co2_overall_ppm' in summary
    assert 'rooms_co2_very_high_alert_latest_count' in summary # Based on ALERT_AMBIENT_CO2_VERY_HIGH_PPM
    # If sample has CO2 > ALERT_AMBIENT_CO2_VERY_HIGH_PPM for some latest room readings:
    if (sample_iot_clinic_df_main_sentinel.sort_values('timestamp').drop_duplicates(['clinic_id', 'room_name'], keep='last')['avg_co2_ppm'] > app_config.ALERT_AMBIENT_CO2_VERY_HIGH_PPM).any():
        assert summary['rooms_co2_very_high_alert_latest_count'] > 0
    else:
        assert summary['rooms_co2_very_high_alert_latest_count'] == 0


def test_get_patient_alerts_for_clinic_sentinel(sample_health_records_df_main_sentinel):
    df_period = sample_health_records_df_main_sentinel[
        (sample_health_records_df_main_sentinel['encounter_date'].dt.date >= date(2023,10,1)) &
        (sample_health_records_df_main_sentinel['encounter_date'].dt.date <= date(2023,10,7)) &
        (sample_health_records_df_main_sentinel['ai_risk_score'] >= app_config.RISK_SCORE_MODERATE_THRESHOLD) # Focus on patients that would trigger alerts
    ]
    if df_period.empty: pytest.skip("No relevant data for clinic patient alert testing.")
    
    alerts_df = get_patient_alerts_for_clinic(df_period)
    assert isinstance(alerts_df, pd.DataFrame)
    if not alerts_df.empty:
        # Check for key columns output by the refactored get_patient_alerts_for_clinic
        expected_alert_df_cols = ['patient_id', 'encounter_date', 'Alert Reason', 'Priority Score']
        for col in expected_alert_df_cols:
            assert col in alerts_df.columns, f"Clinic alerts DF missing column: {col}"
        assert not alerts_df['Priority Score'].isnull().any()


def test_get_district_summary_kpis_sentinel(sample_enriched_gdf_main_sentinel):
    if sample_enriched_gdf_main_sentinel.empty: pytest.skip("Enriched GDF empty for district KPI test.")
    kpis = get_district_summary_kpis(sample_enriched_gdf_main_sentinel)
    assert isinstance(kpis, dict)
    # Check Sentinel-specific district KPI keys and non-NaN for calculated population-weighted averages
    expected_dist_keys = [
        "total_population_district", "population_weighted_avg_ai_risk_score",
        "zones_meeting_high_risk_criteria_count", "district_avg_facility_coverage_score",
        "district_total_active_tb_cases", # Assumes tb is a KEY_CONDITION_FOR_ACTION
        "district_overall_key_disease_prevalence_per_1000", "district_population_weighted_avg_steps"
    ]
    for key in expected_dist_keys: assert key in kpis, f"District KPI key '{key}' missing"
    
    if kpis["total_population_district"] > 0:
        assert pd.notna(kpis["population_weighted_avg_ai_risk_score"])
        assert pd.notna(kpis["district_avg_facility_coverage_score"])
    else: # If total population is 0, weighted averages might be NaN or simple mean
        assert pd.isna(kpis["population_weighted_avg_ai_risk_score"]) or isinstance(kpis["population_weighted_avg_ai_risk_score"], float)


def test_get_supply_forecast_data_sentinel(sample_health_records_df_main_sentinel):
    # Test for a specific item present in the sample data
    if sample_health_records_df_main_sentinel.empty: pytest.skip("Health records empty for supply forecast test.")
    
    test_item = None
    if 'item' in sample_health_records_df_main_sentinel.columns and app_config.KEY_DRUG_SUBSTRINGS_SUPPLY:
        for drug_sub in app_config.KEY_DRUG_SUBSTRINGS_SUPPLY:
            if sample_health_records_df_main_sentinel['item'].str.contains(drug_sub, case=False, na=False).any():
                test_item = sample_health_records_df_main_sentinel[sample_health_records_df_main_sentinel['item'].str.contains(drug_sub, case=False, na=False)]['item'].iloc[0]
                break
    if not test_item: pytest.skip("No key drugs from config found in sample data for supply forecast.")

    forecast_df = get_supply_forecast_data(sample_health_records_df_main_sentinel, item_filter_list=[test_item])
    assert isinstance(forecast_df, pd.DataFrame)
    if not forecast_df.empty:
        expected_forecast_cols = ['item', 'date', 'forecasted_stock_level', 'forecasted_days_of_supply',
                                  'estimated_stockout_date_linear', 'initial_days_supply_at_forecast_start']
        for col in expected_forecast_cols:
            assert col in forecast_df.columns, f"Supply forecast DF missing column: {col}"
        assert forecast_df['item'].iloc[0] == test_item
        assert pd.api.types.is_datetime64_any_dtype(forecast_df['date'])
        assert pd.api.types.is_datetime64_any_dtype(forecast_df['estimated_stockout_date_linear']) or forecast_df['estimated_stockout_date_linear'].isnull().all()


# --- Graceful Handling Tests ---
def test_graceful_handling_empty_inputs(empty_health_df_sentinel_schema, empty_iot_df_sentinel_schema, empty_enriched_gdf_sentinel_schema):
    # Overall KPIs
    assert get_overall_kpis(empty_health_df_sentinel_schema) is not None # Should return default dict
    # CHW Summary
    assert get_chw_summary(empty_health_df_sentinel_schema) is not None
    # CHW Alerts
    assert isinstance(get_patient_alerts_for_chw(empty_health_df_sentinel_schema), list)
    # Clinic Summary
    clinic_sum_empty = get_clinic_summary(empty_health_df_sentinel_schema)
    assert isinstance(clinic_sum_empty, dict) and "test_summary_details" in clinic_sum_empty
    # Clinic Env Summary
    assert get_clinic_environmental_summary(empty_iot_df_sentinel_schema) is not None
    # Clinic Patient Alerts
    assert isinstance(get_patient_alerts_for_clinic(empty_health_df_sentinel_schema), pd.DataFrame) # Returns empty DF
    # District KPIs
    assert get_district_summary_kpis(empty_enriched_gdf_sentinel_schema) is not None
    # Trend Data
    assert isinstance(get_trend_data(empty_health_df_sentinel_schema, 'ai_risk_score'), pd.Series) # Returns empty Series
    # Supply Forecast
    assert isinstance(get_supply_forecast_data(empty_health_df_sentinel_schema), pd.DataFrame) # Returns empty DF

def test_handling_missing_critical_columns(sample_health_records_df_main_sentinel):
    if sample_health_records_df_main_sentinel.empty: pytest.skip("Sample data empty.")
    
    df_missing_risk = sample_health_records_df_main_sentinel.drop(columns=['ai_risk_score'], errors='ignore')
    # get_overall_kpis should still run and return nan or default for avg_patient_risk
    kpis_no_risk = get_overall_kpis(df_missing_risk)
    assert pd.isna(kpis_no_risk['avg_patient_risk'])

    df_missing_condition = sample_health_records_df_main_sentinel.drop(columns=['condition'], errors='ignore')
    # active_tb_cases should be 0 or based on dynamic col presence handled in function
    kpis_no_cond = get_overall_kpis(df_missing_condition)
    tb_col_name_in_kpi = f"active_tb_cases_current" # Key in get_overall_kpis if TB is a key condition
    assert kpis_no_cond.get(tb_col_name_in_kpi, 0) == 0
