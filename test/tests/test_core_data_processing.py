# sentinel_project_root/test/tests/test_core_data_processing.py
# Pytest tests for functions in utils.core_data_processing.py for Sentinel.

import pytest
import pandas as pd
import geopandas as gpd
import numpy as np
from datetime import date, datetime, timedelta
import inspect # For logging function names if needed (removed for brevity)

# Functions to be tested
from utils.core_data_processing import (
    _clean_column_names,
    _convert_to_numeric,
    hash_geodataframe,
    load_health_records, # Tested via fixture usage mainly
    load_iot_clinic_environment_data, # Tested via fixture
    load_zone_data, # Tested via fixture
    enrich_zone_geodata_with_health_aggregates, # Tested with fixtures
    get_overall_kpis,
    get_chw_summary,
    # get_patient_alerts_for_chw, # This logic moved to alert_generator.py component
    get_clinic_summary,
    get_clinic_environmental_summary,
    get_patient_alerts_for_clinic,
    get_district_summary_kpis,
    get_trend_data,
    get_supply_forecast_data # Simple linear forecast
)
from config import app_config # For thresholds, keys, etc.
import logging # For checking logger messages if needed, or for this file's own logs
logger = logging.getLogger(__name__)


# --- Tests for Helper Functions ---
def test_clean_column_names_functionality():
    df_dirty_cols = pd.DataFrame(columns=['Test Column One', 'Another-Col WITH_Space', 'already_good', ' leading_space', 'trailing_space ', 'col(paren)'])
    df_cleaned_cols = _clean_column_names(df_dirty_cols.copy())
    expected_cols = ['test_column_one', 'another_col_with_space', 'already_good', 'leading_space', 'trailing_space', 'col_paren']
    assert list(df_cleaned_cols.columns) == expected_cols, "Column cleaning did not produce expected names."
    
    df_empty = pd.DataFrame()
    assert list(_clean_column_names(df_empty.copy()).columns) == [], "Cleaning empty DataFrame failed."
    # Test idempotency
    assert list(_clean_column_names(df_cleaned_cols.copy()).columns) == expected_cols, "Re-cleaning already clean columns changed them."

def test_convert_to_numeric_functionality():
    series_mixed_dirty = pd.Series(['101.0', '23.5', 'bad_value', None, '55', True, False]) # Added booleans
    
    series_to_nan = _convert_to_numeric(series_mixed_dirty.copy(), default_value=np.nan)
    expected_nan = pd.Series([101.0, 23.5, np.nan, np.nan, 55.0, 1.0, 0.0], dtype=float) # Booleans convert to 1.0/0.0
    pd.testing.assert_series_equal(series_to_nan, expected_nan, check_dtype=True, check_exact=False, rtol=1e-5)

    series_to_zero = _convert_to_numeric(series_mixed_dirty.copy(), default_value=0)
    expected_zero = pd.Series([101.0, 23.5, 0.0, 0.0, 55.0, 1.0, 0.0], dtype=float)
    pd.testing.assert_series_equal(series_to_zero, expected_zero, check_dtype=True, check_exact=False, rtol=1e-5)

    series_already_numeric = pd.Series([1, 2.5, 3, np.nan])
    pd.testing.assert_series_equal(_convert_to_numeric(series_already_numeric.copy()), pd.Series([1.0, 2.5, 3.0, np.nan], dtype=float), check_dtype=True)
    pd.testing.assert_series_equal(_convert_to_numeric(pd.Series([], dtype=object)), pd.Series([], dtype=float), check_dtype=True)


def test_hash_geodataframe_functionality(sample_zone_geometries_gdf_main_sentinel: gpd.GeoDataFrame):
    if not isinstance(sample_zone_geometries_gdf_main_sentinel, gpd.GeoDataFrame) or \
       sample_zone_geometries_gdf_main_sentinel.empty:
        pytest.skip("Sample GDF for hashing is invalid or empty.")
    
    gdf_to_hash = sample_zone_geometries_gdf_main_sentinel.copy()
    hash_val1 = hash_geodataframe(gdf_to_hash)
    assert isinstance(hash_val1, str) and hash_val1 is not None, "Hashing valid GDF failed."
    
    assert hash_geodataframe(None) is None, "Hashing None GDF should return None." # type: ignore
    assert hash_geodataframe(gpd.GeoDataFrame()) == "empty_gdf", "Hashing empty GDF returned unexpected value."

    # Test that modification changes the hash
    gdf_modified_hash = gdf_to_hash.copy()
    if 'population' in gdf_modified_hash.columns and not gdf_modified_hash.empty:
        original_pop_val = gdf_modified_hash.loc[0, 'population']
        gdf_modified_hash.loc[0, 'population'] = (original_pop_val + 1000) if pd.notna(original_pop_val) else 1000
        hash_val2 = hash_geodataframe(gdf_modified_hash)
        assert isinstance(hash_val2, str) and hash_val1 != hash_val2, "Modified GDF did not produce a different hash."
    else:
        logger.warning("Skipping GDF hash modification sub-test; 'population' col missing or GDF empty.")

# --- Tests for Data Loading (Schema and basic type checks using fixtures) ---
# These tests use fixtures that simulate the *output* of the load functions.
def test_load_health_records_simulated_output(sample_health_records_df_main_sentinel: pd.DataFrame):
    df = sample_health_records_df_main_sentinel # This fixture already applies AI models
    assert isinstance(df, pd.DataFrame), "Health records fixture is not a DataFrame."
    if df.empty: pytest.skip("Health records fixture is empty.")
    
    key_cols_health = ['patient_id', 'encounter_date', 'ai_risk_score', 'min_spo2_pct', 'condition']
    for col in key_cols_health:
        assert col in df.columns, f"Key column '{col}' missing in health records fixture."
    assert pd.api.types.is_datetime64_any_dtype(df['encounter_date']), "'encounter_date' is not datetime."
    assert pd.api.types.is_numeric_dtype(df['ai_risk_score']), "'ai_risk_score' is not numeric."

def test_load_iot_data_simulated_output(sample_iot_clinic_df_main_sentinel: pd.DataFrame):
    df = sample_iot_clinic_df_main_sentinel
    assert isinstance(df, pd.DataFrame), "IoT data fixture is not a DataFrame."
    if df.empty: pytest.skip("IoT data fixture is empty.")
        
    key_cols_iot = ['timestamp', 'clinic_id', 'room_name', 'avg_co2_ppm']
    for col in key_cols_iot:
        assert col in df.columns, f"Key column '{col}' missing in IoT data fixture."
    assert pd.api.types.is_datetime64_any_dtype(df['timestamp']), "'timestamp' is not datetime."
    assert pd.api.types.is_numeric_dtype(df['avg_co2_ppm']) or df['avg_co2_ppm'].isnull().all(), "'avg_co2_ppm' is not numeric or all NaN."

def test_load_zone_data_simulated_output(sample_zone_geometries_gdf_main_sentinel: gpd.GeoDataFrame):
    gdf = sample_zone_geometries_gdf_main_sentinel # This fixture simulates the merged output
    assert isinstance(gdf, gpd.GeoDataFrame), "Zone data fixture is not a GeoDataFrame."
    if gdf.empty: pytest.skip("Zone data fixture is empty.")
        
    key_cols_zone = ['zone_id', 'name', 'population', 'geometry'] # 'geometry' is the active geometry column
    for col in key_cols_zone:
        assert col in gdf.columns, f"Key column '{col}' missing in zone data fixture."
    assert gdf.crs is not None, "Zone data GDF has no CRS."
    assert str(gdf.crs).upper() == app_config.DEFAULT_CRS_STANDARD.upper(), "Zone data GDF has incorrect CRS."


# --- Test for Enrichment Logic ---
def test_enrich_zone_geodata_values_sentinel(
    sample_enriched_gdf_main_sentinel: Optional[gpd.GeoDataFrame], 
    sample_health_records_df_main_sentinel: pd.DataFrame
):
    if not isinstance(sample_enriched_gdf_main_sentinel, gpd.GeoDataFrame) or \
       sample_enriched_gdf_main_sentinel.empty or \
       sample_health_records_df_main_sentinel.empty:
        pytest.skip("Cannot test enrichment values with empty/invalid input fixtures for Sentinel.")
    
    gdf_enriched = sample_enriched_gdf_main_sentinel
    health_df_source = sample_health_records_df_main_sentinel
    
    zone_to_verify = 'ZoneA' # Assuming 'ZoneA' exists in sample data
    if zone_to_verify not in gdf_enriched.get('zone_id', pd.Series(dtype=str)).tolist():
        pytest.skip(f"Test zone '{zone_to_verify}' not found in the enriched GDF fixture.")
    
    # Example 1: active_tb_cases for ZoneA
    # Construct dynamic column name as per enrichment logic
    tb_col_dynamic = f"active_tb_cases" # Assumes 'TB' is in KEY_CONDITIONS_FOR_ACTION and formatted this way
    if tb_col_dynamic in gdf_enriched.columns:
        expected_tb_in_zone_a = health_df_source[
            (health_df_source['zone_id'] == zone_to_verify) & 
            (health_df_source.get('condition', pd.Series(dtype=str)).str.contains("TB", case=False, na=False))
        ]['patient_id'].nunique()
        
        actual_tb_in_zone_a = gdf_enriched[gdf_enriched['zone_id'] == zone_to_verify][tb_col_dynamic].iloc[0]
        assert actual_tb_in_zone_a == expected_tb_in_zone_a, f"Mismatch in '{tb_col_dynamic}' for '{zone_to_verify}'."

    # Example 2: avg_risk_score for ZoneA
    if 'avg_risk_score' in gdf_enriched.columns:
        expected_avg_risk_in_zone_a = health_df_source[health_df_source['zone_id'] == zone_to_verify]['ai_risk_score'].mean()
        actual_avg_risk_in_zone_a = gdf_enriched[gdf_enriched['zone_id'] == zone_to_verify]['avg_risk_score'].iloc[0]
        
        if pd.notna(expected_avg_risk_in_zone_a) and pd.notna(actual_avg_risk_in_zone_a):
            assert np.isclose(actual_avg_risk_in_zone_a, expected_avg_risk_in_zone_a, rtol=1e-3), \
                f"Mismatch in 'avg_risk_score' for '{zone_to_verify}'."
        else: # Both should be NaN or one is NaN (check based on expected logic if data causes NaN)
            assert pd.isna(actual_avg_risk_in_zone_a) == pd.isna(expected_avg_risk_in_zone_a), \
                f"NaN mismatch for 'avg_risk_score' in '{zone_to_verify}'."


# --- Tests for KPI and Summary Calculation Functions ---

def test_get_clinic_summary_sentinel_structure(sample_health_records_df_main_sentinel: pd.DataFrame):
    if sample_health_records_df_main_sentinel.empty:
        pytest.skip("Sample health records empty for clinic summary test.")
    
    # Use a relevant slice of data for the summary
    df_clinic_period = sample_health_records_df_main_sentinel[
        (sample_health_records_df_main_sentinel['encounter_date'] >= pd.Timestamp('2023-10-01')) &
        (sample_health_records_df_main_sentinel['encounter_date'] <= pd.Timestamp('2023-10-15')) &
        (sample_health_records_df_main_sentinel['clinic_id'] == 'CLINIC01') # Focus on one clinic
    ].copy() # Ensure it's a copy

    if df_clinic_period.empty: 
        pytest.skip("No data for CLINIC01 in the specified period for clinic summary test.")
    
    clinic_summary_output = get_clinic_summary(df_clinic_period)
    assert isinstance(clinic_summary_output, dict), "get_clinic_summary did not return a dict."
    
    expected_top_level_keys = [
        "overall_avg_test_turnaround_conclusive_days", "perc_critical_tests_tat_met",
        "total_pending_critical_tests_patients", "sample_rejection_rate_perc",
        "key_drug_stockouts_count", "test_summary_details"
    ]
    for key in expected_top_level_keys:
        assert key in clinic_summary_output, f"Key '{key}' missing in clinic_summary output."
    
    assert isinstance(clinic_summary_output["test_summary_details"], dict), "'test_summary_details' is not a dict."
    
    # Check structure of one entry in test_summary_details using a configured test type
    # Example: Malaria RDT
    malaria_rdt_config_key = "RDT-Malaria"
    if malaria_rdt_config_key in app_config.KEY_TEST_TYPES_FOR_ANALYSIS:
        malaria_display_name = app_config.KEY_TEST_TYPES_FOR_ANALYSIS[malaria_rdt_config_key]["display_name"]
        if malaria_display_name in clinic_summary_output["test_summary_details"]:
            malaria_test_detail = clinic_summary_output["test_summary_details"][malaria_display_name]
            expected_detail_metric_keys = [
                "positive_rate_perc", "avg_tat_days", "perc_met_tat_target",
                "pending_count_patients", "rejected_count_patients", "total_conclusive_tests"
            ]
            for detail_key in expected_detail_metric_keys:
                assert detail_key in malaria_test_detail, \
                    f"Detail key '{detail_key}' missing for '{malaria_display_name}' in test_summary_details."
        else:
            logger.info(f"Test display name '{malaria_display_name}' not found in test_summary_details. This might be okay if no such tests in sample period.")


def test_get_clinic_environmental_summary_sentinel_kpis(sample_iot_clinic_df_main_sentinel: pd.DataFrame):
    if sample_iot_clinic_df_main_sentinel.empty:
        pytest.skip("Sample IoT data empty for environmental summary test.")
    
    iot_summary_output = get_clinic_environmental_summary(sample_iot_clinic_df_main_sentinel)
    assert isinstance(iot_summary_output, dict), "get_clinic_environmental_summary did not return a dict."
    
    assert 'avg_co2_overall_ppm' in iot_summary_output, "Missing 'avg_co2_overall_ppm' in IoT summary."
    assert 'rooms_co2_very_high_alert_latest_count' in iot_summary_output, "Missing 'rooms_co2_very_high_alert_latest_count'."
    
    # Validate alert count based on sample data and config
    latest_readings_per_room = sample_iot_clinic_df_main_sentinel.sort_values('timestamp').drop_duplicates(['clinic_id', 'room_name'], keep='last')
    expected_co2_alerts = (latest_readings_per_room['avg_co2_ppm'] > app_config.ALERT_AMBIENT_CO2_VERY_HIGH_PPM).sum()
    assert iot_summary_output['rooms_co2_very_high_alert_latest_count'] == expected_co2_alerts, \
        "Mismatch in 'rooms_co2_very_high_alert_latest_count'."


def test_get_patient_alerts_for_clinic_sentinel(sample_health_records_df_main_sentinel: pd.DataFrame):
    if sample_health_records_df_main_sentinel.empty:
        pytest.skip("Sample health records empty for clinic patient alert testing.")
    
    # Filter for records that are more likely to trigger alerts (e.g., high risk score)
    df_alerts_input = sample_health_records_df_main_sentinel[
        (sample_health_records_df_main_sentinel['encounter_date'].dt.date >= date(2023,10,1)) &
        (sample_health_records_df_main_sentinel['encounter_date'].dt.date <= date(2023,10,10)) & # Example period
        (sample_health_records_df_main_sentinel.get('ai_risk_score', pd.Series(dtype=float)) >= app_config.RISK_SCORE_MODERATE_THRESHOLD)
    ].copy()

    if df_alerts_input.empty:
        pytest.skip("No relevant data (e.g., high risk patients) in sample for clinic patient alert testing for the chosen period.")
    
    clinic_alerts_df_output = get_patient_alerts_for_clinic(df_alerts_input)
    assert isinstance(clinic_alerts_df_output, pd.DataFrame), "get_patient_alerts_for_clinic did not return a DataFrame."
    
    if not clinic_alerts_df_output.empty:
        expected_clinic_alert_cols = ['patient_id', 'encounter_date', 'Alert Reason', 'Priority Score', 'condition', 'ai_risk_score']
        for col in expected_clinic_alert_cols:
            assert col in clinic_alerts_df_output.columns, f"Clinic alerts DataFrame missing column: {col}"
        assert not clinic_alerts_df_output['Priority Score'].isnull().any(), "Priority Score in clinic alerts should not be NaN."


def test_get_district_summary_kpis_sentinel(sample_enriched_gdf_main_sentinel: Optional[gpd.GeoDataFrame]):
    if not isinstance(sample_enriched_gdf_main_sentinel, gpd.GeoDataFrame) or sample_enriched_gdf_main_sentinel.empty:
        pytest.skip("Enriched GDF fixture empty or invalid for district KPI test.")
        
    district_kpis_output = get_district_summary_kpis(sample_enriched_gdf_main_sentinel)
    assert isinstance(district_kpis_output, dict), "get_district_summary_kpis did not return a dict."
    
    expected_district_kpi_keys = [
        "total_population_district", "population_weighted_avg_ai_risk_score",
        "zones_meeting_high_risk_criteria_count", "district_avg_facility_coverage_score",
        # Example for one dynamic condition key based on app_config
        f"district_total_active_{app_config.KEY_CONDITIONS_FOR_ACTION[0].lower().replace(' ', '_').replace('-', '_').replace('(severe)','')}_cases",
        "district_overall_key_disease_prevalence_per_1000", "district_population_weighted_avg_steps"
    ]
    for key in expected_district_kpi_keys:
        assert key in district_kpis_output, f"District KPI key '{key}' missing from output."
    
    if district_kpis_output.get("total_population_district", 0) > 0:
        assert pd.notna(district_kpis_output["population_weighted_avg_ai_risk_score"]), "Pop-weighted AI risk is NaN with non-zero population."
        assert pd.notna(district_kpis_output["district_avg_facility_coverage_score"]), "Pop-weighted facility coverage is NaN."
    else: # If total population is 0, weighted averages might be NaN or simple mean based on impl.
        assert pd.isna(district_kpis_output.get("population_weighted_avg_ai_risk_score", np.nan)) or \
               isinstance(district_kpis_output.get("population_weighted_avg_ai_risk_score"), float), \
               "Pop-weighted AI risk has unexpected value with zero population."

def test_get_supply_forecast_data_sentinel(sample_health_records_df_main_sentinel: pd.DataFrame):
    if sample_health_records_df_main_sentinel.empty:
        pytest.skip("Health records empty for supply forecast test.")
    
    item_for_forecast_test = None
    if 'item' in sample_health_records_df_main_sentinel.columns and app_config.KEY_DRUG_SUBSTRINGS_SUPPLY:
        for drug_substring in app_config.KEY_DRUG_SUBSTRINGS_SUPPLY:
            if sample_health_records_df_main_sentinel['item'].str.contains(drug_substring, case=False, na=False).any():
                item_for_forecast_test = sample_health_records_df_main_sentinel[
                    sample_health_records_df_main_sentinel['item'].str.contains(drug_substring, case=False, na=False)
                ]['item'].iloc[0]
                break # Found a testable item
    if not item_for_forecast_test:
        pytest.skip("No key drugs from app_config found in sample data for supply forecast test.")

    supply_forecast_df_output = get_supply_forecast_data(sample_health_records_df_main_sentinel, item_filter_list=[item_for_forecast_test])
    assert isinstance(supply_forecast_df_output, pd.DataFrame), "get_supply_forecast_data did not return a DataFrame."
    
    if not supply_forecast_df_output.empty:
        expected_supply_forecast_cols = ['item', 'date', 'forecasted_stock_level', 'forecasted_days_of_supply',
                                         'estimated_stockout_date_linear', 'initial_days_supply_at_forecast_start']
        for col in expected_supply_forecast_cols:
            assert col in supply_forecast_df_output.columns, f"Supply forecast DataFrame missing column: {col}"
        assert supply_forecast_df_output['item'].iloc[0] == item_for_forecast_test, "Forecasted item mismatch."
        assert pd.api.types.is_datetime64_any_dtype(supply_forecast_df_output['date']), "'date' column in supply forecast is not datetime."
        # estimated_stockout_date_linear can be NaT if stockout is beyond forecast period
        assert pd.api.types.is_datetime64_any_dtype(supply_forecast_df_output['estimated_stockout_date_linear']) or \
               supply_forecast_df_output['estimated_stockout_date_linear'].isnull().all(), \
               "'estimated_stockout_date_linear' has incorrect type or non-NaT nulls."


# --- Graceful Handling Tests for Empty or Flawed Inputs ---
def test_graceful_handling_empty_inputs_all_summaries(
    empty_health_df_sentinel_schema: pd.DataFrame, 
    empty_iot_df_sentinel_schema: pd.DataFrame, 
    empty_enriched_gdf_sentinel_schema: gpd.GeoDataFrame
):
    # Overall KPIs
    assert isinstance(get_overall_kpis(empty_health_df_sentinel_schema.copy()), dict), "get_overall_kpis failed on empty input."
    # CHW Summary
    assert isinstance(get_chw_summary(empty_health_df_sentinel_schema.copy()), dict), "get_chw_summary failed on empty input."
    # Clinic Summary
    clinic_summary_empty_res = get_clinic_summary(empty_health_df_sentinel_schema.copy())
    assert isinstance(clinic_summary_empty_res, dict) and "test_summary_details" in clinic_summary_empty_res, "get_clinic_summary structure error on empty."
    # Clinic Env Summary
    assert isinstance(get_clinic_environmental_summary(empty_iot_df_sentinel_schema.copy()), dict), "get_clinic_environmental_summary failed on empty."
    # Clinic Patient Alerts
    assert isinstance(get_patient_alerts_for_clinic(empty_health_df_sentinel_schema.copy()), pd.DataFrame), "get_patient_alerts_for_clinic failed on empty."
    # District KPIs
    assert isinstance(get_district_summary_kpis(empty_enriched_gdf_sentinel_schema.copy()), dict), "get_district_summary_kpis failed on empty."
    # Trend Data
    assert isinstance(get_trend_data(empty_health_df_sentinel_schema.copy(), 'ai_risk_score'), pd.Series), "get_trend_data failed on empty." # Expects empty series
    # Supply Forecast
    assert isinstance(get_supply_forecast_data(empty_health_df_sentinel_schema.copy()), pd.DataFrame), "get_supply_forecast_data failed on empty."


def test_handling_missing_critical_columns_in_summaries(sample_health_records_df_main_sentinel: pd.DataFrame):
    if sample_health_records_df_main_sentinel.empty:
        pytest.skip("Sample data empty, cannot test missing columns impact.")
    
    df_no_risk = sample_health_records_df_main_sentinel.drop(columns=['ai_risk_score'], errors='ignore')
    kpis_output_no_risk = get_overall_kpis(df_no_risk.copy())
    assert pd.isna(kpis_output_no_risk.get('avg_patient_ai_risk_period', np.nan)), \
        "avg_patient_ai_risk_period should be NaN when 'ai_risk_score' column is missing."

    df_no_condition = sample_health_records_df_main_sentinel.drop(columns=['condition'], errors='ignore')
    kpis_output_no_condition = get_overall_kpis(df_no_condition.copy())
    # Dynamic condition columns should default to 0 if 'condition' column is absent
    example_cond_key = app_config.KEY_CONDITIONS_FOR_ACTION[0]
    formatted_cond_key_kpi = f"active_{example_cond_key.lower().replace(' ', '_').replace('-', '_').replace('(severe)','')}_cases_period"
    assert kpis_output_no_condition.get(formatted_cond_key_kpi, 0) == 0, \
        f"{formatted_cond_key_kpi} should be 0 when 'condition' column is missing."
