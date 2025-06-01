# sentinel_project_root/test/tests/conftest.py
# Pytest fixtures for testing the "Sentinel Health Co-Pilot" application.
# These fixtures provide sample data aligned with the refactored system design,
# new schemas, and app_config settings.

import pytest
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon # For creating sample geometries
import numpy as np
from datetime import datetime, date, timedelta # For sample dates

# Crucially, import the refactored app_config and core data processing functions
# This ensures that fixtures generating complex data (like enriched GDF) use the actual logic.
# Assuming 'test' directory (where app_config, utils are) is on PYTHONPATH for tests.
try:
    from config import app_config
    from utils.core_data_processing import (
        load_health_records, # To simulate loading if CSVs are source of fixture data
        load_iot_clinic_environment_data,
        load_zone_data,
        enrich_zone_geodata_with_health_aggregates
    )
    from utils.ai_analytics_engine import apply_ai_models # If sample_health_records needs AI enrichment stage
except ImportError as e:
    # Attempt to adjust path if running pytest from project root and 'test' is not automatically on path
    import sys, os
    current_dir_conftest = os.path.dirname(os.path.abspath(__file__)) # tests directory
    project_test_root_conftest = os.path.abspath(os.path.join(current_dir_conftest, os.pardir)) # test directory
    if project_test_root_conftest not in sys.path:
        sys.path.insert(0, project_test_root_conftest)
    
    # Try imports again
    from config import app_config
    from utils.core_data_processing import (
        load_health_records, load_iot_clinic_environment_data, load_zone_data,
        enrich_zone_geodata_with_health_aggregates
    )
    from utils.ai_analytics_engine import apply_ai_models


# --- Fixture for Sample Health Records (Sentinel Lean Data Model + Key Fields) ---
@pytest.fixture(scope="session")
def sample_health_records_df_main_sentinel() -> pd.DataFrame:
    """
    Provides a comprehensive DataFrame of sample health records, reflecting the
    Sentinel lean data model, new app_config thresholds, and AI-enriched fields.
    This data is representative of what might be synced from PEDs or collected at facilities.
    """
    # Using datetime objects directly for dates
    base_date = datetime(2023, 10, 1, 8, 0, 0)
    date_list = [base_date + timedelta(hours=i*2, days=i//4) for i in range(42)]

    data = {
        'encounter_id': [f'SENC{i:03d}' for i in range(1, 43)],
        'patient_id': [f'SPID{i%15:03d}' for i in range(1, 43)], # ~15 unique patients
        'encounter_date': date_list,
        'encounter_type': ['CHW_HOME_VISIT', 'CHW_ALERT_RESPONSE', 'CLINIC_INTAKE', 'WORKER_SELF_CHECK'] * 10 + ['CHW_HOME_VISIT', 'CLINIC_INTAKE'],
        'age': np.random.randint(1, 85, 42).tolist(),
        'gender': np.random.choice(['Male', 'Female', 'Other'], 42).tolist(),
        'pregnancy_status': np.random.choice([0, 1, 0, 0], 42).tolist(), # 0=No, 1=Yes
        'chronic_condition_flag': np.random.choice([0, 1, 0], 42).tolist(), # 0=No, 1=Yes
        'zone_id': [f"Zone{chr(65+(i%4))}" for i in range(42)], # ZoneA, B, C, D
        'clinic_id': [f"CLINIC{ (i%3)+1 :02d}" for i in range(42)], # CLINIC01, 02, 03
        'chw_id': [f"CHW{ (i%5)+1 :03d}" for i in range(42)], # CHW001-005
        
        # Sensor Streams (illustrative, values should hit thresholds in app_config)
        'hrv_rmssd_ms': np.random.uniform(15, 70, 42).round(1).tolist(), # Some below app_config.STRESS_HRV_LOW_THRESHOLD_MS
        'min_spo2_pct': np.random.choice(
            [app_config.ALERT_SPO2_CRITICAL_LOW_PCT-2, app_config.ALERT_SPO2_CRITICAL_LOW_PCT+1,
             app_config.ALERT_SPO2_WARNING_LOW_PCT-1, app_config.ALERT_SPO2_WARNING_LOW_PCT+2, 98, 99], 42).tolist(),
        'vital_signs_temperature_celsius': np.random.choice(
            [app_config.ALERT_BODY_TEMP_FEVER_C-0.5, app_config.ALERT_BODY_TEMP_FEVER_C+0.2,
             app_config.ALERT_BODY_TEMP_HIGH_FEVER_C-0.2, app_config.ALERT_BODY_TEMP_HIGH_FEVER_C+0.3, 37.0], 42).tolist(),
        'max_skin_temp_celsius': [t + np.random.uniform(-0.2, 0.2) for t in np.random.normal(37.2, 0.5, 42)], # Correlated with body temp
        'movement_activity_level': np.random.choice([0,1,2,3], 42).tolist(), # 0=None, 1=Low, 2=Med, 3=High
        'fall_detected_today': np.random.choice([0,0,0,0,1], 42).tolist(), # Some falls
        
        # Contextual & Behavioral
        'ambient_heat_index_c': np.random.uniform(28, 45, 42).round(1).tolist(), # Some above ALERT_AMBIENT_HEAT_INDEX_DANGER_C
        'ppe_compliant_flag': np.random.choice([0,1,1,1], 42).tolist(), # 0=No, 1=Yes, some non-compliance
        'signs_of_fatigue_observed_flag': np.random.choice([0,0,1], 42).tolist(),
        'rapid_psychometric_distress_score': np.random.randint(0, 11, 42).tolist(), # 0-10 scale

        'condition': np.random.choice(app_config.KEY_CONDITIONS_FOR_ACTION + ['Wellness Visit', 'Minor Cold'], 42).tolist(),
        'patient_reported_symptoms': ['fever;cough', 'none', 'headache;fatigue', 'diarrhea', 'short breath'] * 8 + ['fever', 'cough'],
        
        'test_type': np.random.choice(list(app_config.KEY_TEST_TYPES_FOR_ANALYSIS.keys()) + ["None"], 42).tolist(),
        'test_result': np.random.choice(['Positive', 'Negative', 'Pending', 'N/A'], 42).tolist(),
        'test_turnaround_days': [np.random.uniform(0,5) if r != "Pending" and r != "N/A" else np.nan for r in ['Positive']*42 ], # Placeholder logic

        'referral_status': np.random.choice(['Pending', 'Completed', 'Initiated', 'N/A'], 42).tolist(),
        'referral_reason': ['Urgent consult', 'Lab test', 'Specialist review', 'Routine', 'None'] * 8 + ['Urgent', 'Lab'],
        
        'medication_adherence_self_report': np.random.choice(['Good', 'Fair', 'Poor', 'N/A'], 42).tolist(),
        'item': np.random.choice(app_config.KEY_DRUG_SUBSTRINGS_SUPPLY + ['Bandages', 'Gloves'], 42).tolist(),
        'item_stock_agg_zone': np.random.randint(0,100,42).tolist(), # Simulating stock for higher tier processing
        'consumption_rate_per_day': np.random.uniform(0.1,5,42).tolist()
        # Ensure any other key columns from `load_health_records` and `apply_ai_models` are here
        # (e.g., 'notes', 'sample_collection_date', specific sensor data columns if used)
    }
    df = pd.DataFrame(data)

    # Apply AI models to this raw data to get 'ai_risk_score' and 'ai_followup_priority_score'
    # This ensures the fixture provides data as it would be *after* initial Edge/Node AI processing.
    df_with_ai, _ = apply_ai_models(df.copy(), source_context="ConftestHealthFixtureGen") # Use copy

    # Simulate cleaning similar to load_health_records if apply_ai_models doesn't do it all
    # (though apply_ai_models input df is assumed to be somewhat clean already for model input)
    for col in ['age', 'hrv_rmssd_ms', 'min_spo2_pct', 'vital_signs_temperature_celsius', 'ambient_heat_index_c', 'rapid_psychometric_distress_score', 'test_turnaround_days']:
        if col in df_with_ai.columns: df_with_ai[col] = pd.to_numeric(df_with_ai[col], errors='coerce')
    for col in ['pregnancy_status', 'chronic_condition_flag', 'ppe_compliant_flag', 'signs_of_fatigue_observed_flag', 'fall_detected_today']:
        if col in df_with_ai.columns: df_with_ai[col] = pd.to_numeric(df_with_ai[col], errors='coerce').fillna(0).astype(int)

    return df_with_ai


@pytest.fixture(scope="session")
def sample_iot_clinic_df_main_sentinel() -> pd.DataFrame: # Renamed for Sentinel context
    """Provides sample IoT data for clinic environments, with values testing new app_config thresholds."""
    data = {
        'timestamp': pd.to_datetime([
            '2023-10-20T09:00:00Z', '2023-10-20T10:00:00Z', '2023-10-20T11:00:00Z',
            '2023-10-21T09:30:00Z', '2023-10-21T10:30:00Z'
        ]),
        'clinic_id': ['CLINIC01', 'CLINIC01', 'CLINIC02', 'CLINIC01', 'CLINIC02'],
        'room_name': ['Waiting Area', 'Consult Room 1', 'TB Screening Room', 'Waiting Area', 'Lab'],
        'zone_id': ['ZoneA', 'ZoneA', 'ZoneB', 'ZoneA', 'ZoneB'],
        'avg_co2_ppm': [800, 1700, app_config.ALERT_AMBIENT_CO2_VERY_HIGH_PPM + 150, 650, 950],
        'avg_pm25': [10.1, app_config.ALERT_AMBIENT_PM25_HIGH_UGM3 + 3, 15.7, 9.2, app_config.ALERT_AMBIENT_PM25_VERY_HIGH_UGM3 + 5],
        'avg_noise_db': [58, 62, app_config.ALERT_AMBIENT_NOISE_HIGH_DBA + 7, 52, 68],
        'waiting_room_occupancy': [7, np.nan, 12, 9, np.nan] # Note: 12 > TARGET_CLINIC_WAITING_ROOM_OCCUPANCY_MAX
    }
    return pd.DataFrame(data)


@pytest.fixture(scope="session")
def sample_zone_geometries_gdf_main_sentinel() -> gpd.GeoDataFrame:
    """
    Provides a sample GeoDataFrame with zone geometries and basic attributes (name, zone_id).
    This would be the base input to `enrich_zone_geodata_with_health_aggregates` after `load_zone_data`.
    The `load_zone_data` function itself is complex, so this fixture simplifies getting a testable GDF.
    """
    # Using the definition from previous conceptual conftest.py for load_zone_data output structure.
    # This implicitly tests the merging logic within `load_zone_data` if it were called here,
    # but for conftest, we provide the merged result directly for other tests to consume.
    
    attributes_data = {'zone_id': ['ZoneA', 'ZoneB', 'ZoneC', 'ZoneD'],
                       'name': ['Alpha Region', 'Beta District', 'Gamma Sector', 'Delta Area'],
                       'population': [10000, 15000, 7500, 12000],
                       'socio_economic_index': [0.65, 0.40, 0.70, 0.50],
                       'num_clinics': [2,1,1,2],
                       'predominant_hazard_type':['HEAT','FLOOD','DUST','NONE']
                      }
    attr_df = pd.DataFrame(attributes_data)

    geom_data = [{"zone_id": "ZoneA", "geometry": Polygon([[0,0],[0,1],[1,1],[1,0],[0,0]])},
                 {"zone_id": "ZoneB", "geometry": Polygon([[1,0],[1,1],[2,1],[2,0],[1,0]])},
                 {"zone_id": "ZoneC", "geometry": Polygon([[0,1],[0,2],[1,2],[1,1],[0,1]])},
                 {"zone_id": "ZoneD", "geometry": Polygon([[1,1],[1,2],[2,2],[2,1],[1,1]])}]
    geom_gdf = gpd.GeoDataFrame(geom_data, geometry='geometry', crs=app_config.DEFAULT_CRS_STANDARD)
    
    merged_gdf = geom_gdf.merge(attr_df, on='zone_id', how='left')
    # Ensure key numeric types after merge
    for col in ['population', 'socio_economic_index', 'num_clinics']:
        if col in merged_gdf: merged_gdf[col] = pd.to_numeric(merged_gdf[col], errors='coerce')
    return merged_gdf


@pytest.fixture(scope="session")
def sample_enriched_gdf_main_sentinel(
    sample_zone_geometries_gdf_main_sentinel, # This is the GDF that already has base attributes
    sample_health_records_df_main_sentinel,
    sample_iot_clinic_df_main_sentinel
) -> gpd.GeoDataFrame:
    """
    Generates a sample enriched GDF by calling the REFFACTORED
    `enrich_zone_geodata_with_health_aggregates` function with other Sentinel fixtures.
    This ensures the test fixture matches the actual data processing pipeline output.
    """
    # `sample_zone_geometries_gdf_main_sentinel` already simulates the output of `load_zone_data`
    # (i.e., geometries merged with basic attributes like name, population from zone_attributes.csv).
    # It's ready to be passed to the enrichment function.
    base_gdf = sample_zone_geometries_gdf_main_sentinel.copy() # Use a copy

    # Health records are already "enriched" with AI scores by its own fixture
    health_for_enrich = sample_health_records_df_main_sentinel.copy()
    iot_for_enrich = sample_iot_clinic_df_main_sentinel.copy()

    enriched_gdf = enrich_zone_geodata_with_health_aggregates(
        zone_gdf=base_gdf,
        health_df=health_for_enrich,
        iot_df=iot_for_enrich,
        source_context="ConftestEnrichment/Sentinel" # For logging context
    )
    return enriched_gdf


# --- Fixtures for Empty Schemas (Updated for Sentinel Model) ---
@pytest.fixture
def empty_health_df_sentinel_schema() -> pd.DataFrame:
    # This list must match the columns generally produced by `load_health_records`
    # after cleaning, AND columns added by `apply_ai_models`.
    # It represents the comprehensive schema for enriched health records.
    cols = [
        'encounter_id', 'patient_id', 'encounter_date', 'encounter_date_obj', # Added by load_health_records if not there
        'encounter_type', 'age', 'gender', 'pregnancy_status', 'chronic_condition_flag',
        'zone_id', 'clinic_id', 'chw_id',
        'hrv_rmssd_ms', 'min_spo2_pct', 'vital_signs_temperature_celsius', 'max_skin_temp_celsius',
        'movement_activity_level', 'fall_detected_today', 'ambient_heat_index_c',
        'ppe_compliant_flag', 'signs_of_fatigue_observed_flag', 'rapid_psychometric_distress_score',
        'condition', 'patient_reported_symptoms', 'test_type', 'test_result', 'test_turnaround_days',
        'referral_status', 'referral_reason', 'referred_to_facility_id',
        'medication_adherence_self_report',
        'item', 'item_stock_agg_zone', 'consumption_rate_per_day', # Note: original had 'item_dispensed_chw_kit' which is more specific
        'ai_risk_score', 'ai_followup_priority_score', # From AI models
        'notes', 'sample_collection_date', 'sample_registered_lab_date', 'rejection_reason', 'sample_status',
        'diagnosis_code_icd10', 'physician_id', 'avg_spo2', 'avg_daily_steps', # These are from original expanded health_records schema
        'resting_heart_rate', 'avg_sleep_duration_hrs', 'sleep_score_pct', 'stress_level_score',
        'screening_hpv_status', 'hiv_viral_load_copies_ml', 'key_chronic_conditions_summary',
        'chw_visit', 'tb_contact_traced', 'referral_outcome', 'referral_outcome_date',
        'patient_latitude', 'patient_longitude'
    ]
    return pd.DataFrame(columns=list(set(cols))) # Use set to ensure no duplicates if merging lists

@pytest.fixture
def empty_iot_df_sentinel_schema() -> pd.DataFrame:
    cols = [
        'timestamp', 'clinic_id', 'room_name', 'zone_id', 'avg_co2_ppm', 'max_co2_ppm',
        'avg_pm25', 'voc_index', 'avg_temp_celsius', 'avg_humidity_rh', 'avg_noise_db',
        'waiting_room_occupancy', 'patient_throughput_per_hour', 'sanitizer_dispenses_per_hour'
    ]
    return pd.DataFrame(columns=cols)

@pytest.fixture
def empty_zone_attributes_sentinel_schema() -> pd.DataFrame:
    cols = ['zone_id', 'name', 'population', 'socio_economic_index', 'num_clinics',
            'avg_travel_time_clinic_min', 'predominant_hazard_type', 'typical_workforce_exposure_level']
    return pd.DataFrame(columns=cols)

@pytest.fixture
def empty_enriched_gdf_sentinel_schema() -> gpd.GeoDataFrame:
    # Must precisely match all columns output by `enrich_zone_geodata_with_health_aggregates`
    # This list is conceptual and based on the refactored enrichment function.
    expected_enriched_cols = [
        'zone_id', 'name', 'geometry', 'population', 'socio_economic_index', 'num_clinics',
        'avg_travel_time_clinic_min', 'predominant_hazard_type', 'typical_workforce_exposure_level', # from base attributes
        'total_population_health_data', 'avg_risk_score', 'total_patient_encounters',
        # Dynamically added condition columns, e.g., 'active_tb_cases', 'active_malaria_cases', ...
        # from app_config.KEY_CONDITIONS_FOR_ACTION
        'total_active_key_infections', 'prevalence_per_1000', 'facility_coverage_score',
        'avg_daily_steps_zone', 'zone_avg_co2', 'population_density',
        'total_referrals_made', 'successful_referrals',
        'avg_test_turnaround_critical', 'perc_critical_tests_tat_met'
    ]
    # Add active_{condition}_cases columns dynamically based on current app_config
    for cond_key in app_config.KEY_CONDITIONS_FOR_ACTION:
        col_name = f"active_{cond_key.lower().replace(' ', '_').replace('-', '_').replace('(severe)','')}_cases"
        if col_name not in expected_enriched_cols:
            expected_enriched_cols.append(col_name)
    
    return gpd.GeoDataFrame(columns=list(set(expected_enriched_cols)), geometry='geometry', crs=app_config.DEFAULT_CRS_STANDARD)


# --- Generic Plotting Data Fixtures (likely still useful for testing _web plotters) ---
@pytest.fixture(scope="session")
def sample_series_data_sentinel() -> pd.Series:
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'])
    return pd.Series([10, 12, 9, 15, 13], index=dates, name="MetricValues")

@pytest.fixture(scope="session")
def sample_bar_df_sentinel() -> pd.DataFrame:
    return pd.DataFrame({
        'category_col': ['CatA', 'CatB', 'CatA', 'CatC', 'CatB'],
        'value_col': [20, 15, 25, 30, 18],
        'group_col': ['Group1', 'Group1', 'Group2', 'Group1', 'Group2']
    })

@pytest.fixture(scope="session")
def sample_donut_df_sentinel() -> pd.DataFrame:
    return pd.DataFrame({
        'status_labels': ['Critical', 'Warning', 'Okay', 'Unknown'],
        'counts_values': [5, 12, 30, 3]
    })

# (sample_heatmap_df can be defined similarly if heatmaps are still used extensively)
