# sentinel_project_root/test/tests/conftest.py
# Pytest fixtures for testing the "Sentinel Health Co-Pilot" application.

import pytest
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon # Polygon is used, Point was not.
import numpy as np
from datetime import datetime, date, timedelta

# Standardized import block for conftest
import sys
import os

# Add the 'test' directory (parent of 'tests', contains 'config' and 'utils') to sys.path
# This allows utils and config to be imported directly by test files and this conftest.
current_conftest_dir = os.path.dirname(os.path.abspath(__file__)) # .../tests
project_test_directory = os.path.abspath(os.path.join(current_conftest_dir, os.pardir)) # .../test

if project_test_directory not in sys.path:
    sys.path.insert(0, project_test_directory)

# Now import from config and utils
try:
    from config import app_config
    from utils.core_data_processing import (
        # load_health_records, # Not called directly by fixtures here, but used by sample_health_records_df_main_sentinel implicitly
        # load_iot_clinic_environment_data,
        # load_zone_data, # sample_zone_geometries_gdf_main_sentinel simulates its output
        enrich_zone_geodata_with_health_aggregates # Called by sample_enriched_gdf_main_sentinel
    )
    from utils.ai_analytics_engine import apply_ai_models # Called by sample_health_records_df_main_sentinel
except ImportError as e_import:
    print(f"CRITICAL ERROR in conftest.py: Could not import project modules. Check PYTHONPATH or structure. Error: {e_import}")
    # Pytest will likely fail to collect tests if this happens.
    # Consider raising the error to make it explicit.
    raise


# --- Fixture for Sample Health Records (Sentinel Lean Data Model + Key Fields) ---
@pytest.fixture(scope="session")
def sample_health_records_df_main_sentinel() -> pd.DataFrame:
    """
    Provides a DataFrame of sample health records, AI-enriched, reflecting the Sentinel model.
    """
    base_date_health = datetime(2023, 10, 1, 9, 0, 0)
    num_records = 50 # Increased sample size for better coverage
    date_list_health = [base_date_health + timedelta(hours=i*3, days=i//6) for i in range(num_records)]

    data_health = {
        'encounter_id': [f'SENC{i:03d}' for i in range(1, num_records + 1)],
        'patient_id': [f'SPID{(i % 20):03d}' for i in range(1, num_records + 1)], # ~20 unique patients
        'encounter_date': date_list_health,
        'encounter_type': np.random.choice(['CHW_HOME_VISIT', 'CHW_ALERT_RESPONSE', 'CLINIC_INTAKE', 'WORKER_SELF_CHECK'], num_records).tolist(),
        'age': np.random.randint(0, 90, num_records).tolist(), # Include infants
        'gender': np.random.choice(['Male', 'Female', 'Other'], num_records).tolist(),
        'pregnancy_status': np.random.choice([0, 1, 0, 0, 0], num_records).tolist(), # 0=No, 1=Yes
        'chronic_condition_flag': np.random.choice([0, 1, 0, 0], num_records).tolist(), # 0=No, 1=Yes
        'zone_id': [f"Zone{chr(65+(i%4))}" for i in range(num_records)], # ZoneA, B, C, D
        'clinic_id': [f"CLINIC{(i%3)+1:02d}" for i in range(num_records)], # CLINIC01, 02, 03
        'chw_id': [f"CHW{(i%5)+1:03d}" for i in range(num_records)], # CHW001-005
        
        'hrv_rmssd_ms': np.random.uniform(app_config.STRESS_HRV_LOW_THRESHOLD_MS - 5, 70, num_records).round(1).tolist(),
        'min_spo2_pct': np.random.choice(
            [app_config.ALERT_SPO2_CRITICAL_LOW_PCT - 2, app_config.ALERT_SPO2_CRITICAL_LOW_PCT, 
             app_config.ALERT_SPO2_WARNING_LOW_PCT - 1, app_config.ALERT_SPO2_WARNING_LOW_PCT, 97, 99], num_records).tolist(),
        'vital_signs_temperature_celsius': np.random.choice(
            [app_config.ALERT_BODY_TEMP_FEVER_C - 0.5, app_config.ALERT_BODY_TEMP_FEVER_C + 0.1,
             app_config.ALERT_BODY_TEMP_HIGH_FEVER_C - 0.1, app_config.ALERT_BODY_TEMP_HIGH_FEVER_C + 0.2, 37.0, 36.5], num_records).tolist(),
        'max_skin_temp_celsius': lambda df: df['vital_signs_temperature_celsius'] + np.random.uniform(-0.3, 0.3, len(df)), # Lambda for dynamic calculation
        'movement_activity_level': np.random.choice([0,1,2,3], num_records).tolist(),
        'fall_detected_today': np.random.choice([0,0,0,0,0,1], num_records).tolist(), # Fewer falls
        
        'ambient_heat_index_c': np.random.uniform(25, app_config.ALERT_AMBIENT_HEAT_INDEX_DANGER_C + 5, num_records).round(1).tolist(),
        'ppe_compliant_flag': np.random.choice([0,1,1,1,1], num_records).tolist(),
        'signs_of_fatigue_observed_flag': np.random.choice([0,0,0,1], num_records).tolist(),
        'rapid_psychometric_distress_score': np.random.randint(0, 11, num_records).tolist(),

        'condition': np.random.choice(app_config.KEY_CONDITIONS_FOR_ACTION + ['Wellness Visit', 'Minor Ailment', 'Injury'], num_records).tolist(),
        'patient_reported_symptoms': ['fever;cough', 'none', 'headache;fatigue;dizzy', 'diarrhea;vomiting', 'short breath;chest pain', 'rash'] * (num_records // 6) + ['fever'] * (num_records % 6),
        
        'test_type': np.random.choice(list(app_config.KEY_TEST_TYPES_FOR_ANALYSIS.keys()) + ["None", "PulseOx"], num_records).tolist(),
        'test_result': np.random.choice(['Positive', 'Negative', 'Pending', 'N/A', '88', '92'], num_records).tolist(), # Adding numeric results for PulseOx
        'test_turnaround_days': [np.random.uniform(0,7) if res not in ["Pending", "N/A"] else np.nan for res in np.random.choice(['Positive', 'Pending', 'N/A'], num_records)],

        'referral_status': np.random.choice(['Pending', 'Completed', 'Initiated', 'N/A', 'Declined'], num_records).tolist(),
        'referral_reason': ['Urgent Consult', 'Lab Work', 'Specialist Review', 'Routine Check', 'Emergency Transport', 'None'] * (num_records // 6) + ['Urgent'] * (num_records % 6) ,
        'referred_to_facility_id': [f"CLINIC{(i%2)+1:02d}" if stat != "N/A" else "N/A" for i, stat in enumerate(np.random.choice(['Pending', 'N/A'], num_records))],
        
        'medication_adherence_self_report': np.random.choice(['Good', 'Fair', 'Poor', 'N/A', 'Unknown'], num_records).tolist(),
        'item': np.random.choice(app_config.KEY_DRUG_SUBSTRINGS_SUPPLY + ['Bandages', 'Gloves', 'ORS Sachet'], num_records).tolist(),
        'quantity_dispensed': [np.random.randint(0,5) if itm != 'Bandages' else np.random.randint(0,20) for itm in np.random.choice(['ORS Sachet', 'Bandages'], num_records)],
        'item_stock_agg_zone': np.random.randint(0,150,num_records).tolist(),
        'consumption_rate_per_day': np.random.uniform(0.05, 3.5, num_records).round(2).tolist(),
        # Other columns from the provided CSV that might be relevant for full schema coverage
        'notes': ["Sample note " + str(i) for i in range(num_records)],
        'sample_collection_date': [d - timedelta(hours=np.random.randint(1,3)) if tt != "None" else pd.NaT for d, tt in zip(date_list_health, np.random.choice(['RDT-Malaria', "None"], num_records))],
        'sample_status': np.random.choice(['Accepted', 'Pending', 'Rejected', 'N/A'], num_records).tolist(),
        'rejection_reason': [np.random.choice(['Hemolyzed', 'Insufficient Quantity', 'Improper Labeling', 'N/A']) if s == 'Rejected' else 'N/A' for s in np.random.choice(['Accepted', 'Rejected', 'N/A'], num_records)]
    }
    df_health = pd.DataFrame(data_health)
    # Apply lambda for dynamically calculated columns
    df_health['max_skin_temp_celsius'] = data_health['max_skin_temp_celsius'](df_health)


    # Apply AI models to this "rawer" data (AI scores should not be in the initial dict)
    # The apply_ai_models function is expected to handle missing columns by adding defaults.
    df_health_enriched, _ = apply_ai_models(df_health.copy(), source_context="ConftestHealthFixtureGen")
    
    # Ensure datetime types are correct after all processing (apply_ai_models might not touch all date cols)
    for col_date in ['encounter_date', 'sample_collection_date']: # Add others if necessary
        if col_date in df_health_enriched.columns:
            df_health_enriched[col_date] = pd.to_datetime(df_health_enriched[col_date], errors='coerce')

    return df_health_enriched


@pytest.fixture(scope="session")
def sample_iot_clinic_df_main_sentinel() -> pd.DataFrame:
    """Provides sample IoT data for clinic environments."""
    iot_data = {
        'timestamp': pd.to_datetime([
            '2023-10-20T09:00:00Z', '2023-10-20T10:00:00Z', '2023-10-20T11:00:00Z', '2023-10-20T12:00:00Z',
            '2023-10-21T09:30:00Z', '2023-10-21T10:30:00Z', '2023-10-21T11:30:00Z'
        ]),
        'clinic_id': ['CLINIC01', 'CLINIC01', 'CLINIC02', 'CLINIC01', 'CLINIC01', 'CLINIC02', 'CLINIC03'],
        'room_name': ['WaitingArea_A', 'ConsultRoom1', 'TB Screening', 'WaitingArea_A', 'Lab_Main', 'TB Screening', 'Outdoor_Tent'],
        'zone_id': ['ZoneA', 'ZoneA', 'ZoneB', 'ZoneA', 'ZoneA', 'ZoneB', 'ZoneD'],
        'avg_co2_ppm': [850, 1650, app_config.ALERT_AMBIENT_CO2_VERY_HIGH_PPM + 200, 2700, 700, 1300, 550],
        'max_co2_ppm': [1000, 1900, app_config.ALERT_AMBIENT_CO2_VERY_HIGH_PPM + 300, 3000, 850, 1500, 650],
        'avg_pm25': [12.1, app_config.ALERT_AMBIENT_PM25_HIGH_UGM3 - 2, 18.7, app_config.ALERT_AMBIENT_PM25_VERY_HIGH_UGM3 + 10, 8.5, 22.0, 5.1],
        'voc_index': np.random.randint(50, 250, 7).tolist(),
        'avg_temp_celsius': np.random.uniform(22.0, 28.0, 7).round(1).tolist(),
        'avg_humidity_rh': np.random.uniform(40, 75, 7).round(0).tolist(),
        'avg_noise_db': [55, 60, app_config.ALERT_AMBIENT_NOISE_HIGH_DBA - 3, 72, 50, 65, app_config.ALERT_AMBIENT_NOISE_HIGH_DBA + 5],
        'waiting_room_occupancy': [8, np.nan, np.nan, app_config.TARGET_CLINIC_WAITING_ROOM_OCCUPANCY_MAX + 5, np.nan, np.nan, 3],
        'patient_throughput_per_hour': [6,10,4,5,7,3,8],
        'sanitizer_dispenses_per_hour': np.random.randint(1,15,7).tolist()
    }
    return pd.DataFrame(iot_data)


@pytest.fixture(scope="session")
def sample_zone_geometries_gdf_main_sentinel() -> gpd.GeoDataFrame:
    """
    Provides a sample GeoDataFrame with zone geometries and merged attributes.
    This simulates the direct output of `load_zone_data`.
    """
    attributes_data_fixture = {
        'zone_id': ['ZoneA', 'ZoneB', 'ZoneC', 'ZoneD'],
        'name': ['Alpha Region Central', 'Beta District South', 'Gamma Sector East', 'Delta Area West'],
        'population': [10500, 15200, 7300, 11800],
        'socio_economic_index': [0.62, 0.38, 0.75, 0.48],
        'num_clinics': [2, 1, 1, 2],
        'avg_travel_time_clinic_min': [20, 45, 15, 25],
        'predominant_hazard_type': ['HEAT_WAVE', 'FLOODING', 'DUST_STORMS', 'NONE_SPECIFIC'],
        'typical_workforce_exposure_level': ['HIGH', 'MODERATE', 'MODERATE_HIGH', 'LOW'],
        'primary_livelihood': ['Urban Informal', 'Subsistence Agriculture', 'Small Trade', 'Mixed Formal'],
        'water_source_main': ['Communal Tap Network', 'Borehole & River', 'Piped Utility Sporadic', 'Piped Utility Reliable'],
        'area_sqkm': [50.5, 120.0, 30.2, 75.8] # Added area for density calculation
    }
    attr_df_fixture = pd.DataFrame(attributes_data_fixture)

    geom_data_fixture = [
        {"zone_id": "ZoneA", "geometry": Polygon([[0,0],[0,1],[1,1],[1,0],[0,0]])},
        {"zone_id": "ZoneB", "geometry": Polygon([[1,0],[1,1],[2,1],[2,0],[1,0]])},
        {"zone_id": "ZoneC", "geometry": Polygon([[0,1],[0,2],[1,2],[1,1],[0,1]])},
        {"zone_id": "ZoneD", "geometry": Polygon([[1,1],[1,2],[2,2],[2,1],[1,1]])}
    ]
    geom_gdf_fixture = gpd.GeoDataFrame(geom_data_fixture, geometry='geometry', crs=app_config.DEFAULT_CRS_STANDARD)
    
    # Simulate the merge that load_zone_data would perform
    merged_gdf_fixture = geom_gdf_fixture.merge(attr_df_fixture, on='zone_id', how='left')
    
    # Ensure numeric types for relevant attribute columns after merge
    numeric_cols_in_gdf = ['population', 'socio_economic_index', 'num_clinics', 'avg_travel_time_clinic_min', 'area_sqkm']
    for col_gdf_num in numeric_cols_in_gdf:
        if col_gdf_num in merged_gdf_fixture.columns:
            merged_gdf_fixture[col_gdf_num] = pd.to_numeric(merged_gdf_fixture[col_gdf_num], errors='coerce')
    return merged_gdf_fixture


@pytest.fixture(scope="session")
def sample_enriched_gdf_main_sentinel(
    sample_zone_geometries_gdf_main_sentinel: gpd.GeoDataFrame, # This is already merged (attributes + geometry)
    sample_health_records_df_main_sentinel: pd.DataFrame,
    sample_iot_clinic_df_main_sentinel: pd.DataFrame
) -> Optional[gpd.GeoDataFrame]: # Can be None if enrichment fails
    """
    Generates a sample enriched GDF by calling `enrich_zone_geodata_with_health_aggregates`.
    """
    if not isinstance(sample_zone_geometries_gdf_main_sentinel, gpd.GeoDataFrame) or \
       sample_zone_geometries_gdf_main_sentinel.empty:
        pytest.skip("Base zone GDF fixture is invalid or empty for enrichment.")
        return None # Should not happen if fixture is correct

    base_gdf_for_enrich = sample_zone_geometries_gdf_main_sentinel.copy()
    health_for_enrich_copy = sample_health_records_df_main_sentinel.copy()
    iot_for_enrich_copy = sample_iot_clinic_df_main_sentinel.copy()

    enriched_gdf_result = enrich_zone_geodata_with_health_aggregates(
        zone_gdf=base_gdf_for_enrich,
        health_df=health_for_enrich_copy,
        iot_df=iot_for_enrich_copy,
        source_context="ConftestEnrichment/Sentinel"
    )
    return enriched_gdf_result


# --- Fixtures for Empty Schemas (Updated for Sentinel Model) ---
@pytest.fixture
def empty_health_df_sentinel_schema() -> pd.DataFrame:
    # Comprehensive list of columns after loading, cleaning, and AI enrichment
    cols = [
        'encounter_id', 'patient_id', 'encounter_date', 'encounter_date_obj',
        'encounter_type', 'age', 'gender', 'pregnancy_status', 'chronic_condition_flag',
        'zone_id', 'clinic_id', 'chw_id',
        'hrv_rmssd_ms', 'min_spo2_pct', 'vital_signs_temperature_celsius', 'max_skin_temp_celsius',
        'movement_activity_level', 'fall_detected_today', 'ambient_heat_index_c',
        'ppe_compliant_flag', 'signs_of_fatigue_observed_flag', 'rapid_psychometric_distress_score',
        'condition', 'patient_reported_symptoms', 'test_type', 'test_result', 'test_turnaround_days',
        'referral_status', 'referral_reason', 'referred_to_facility_id', 'referral_outcome', 'referral_outcome_date', 'referral_date',
        'medication_adherence_self_report',
        'item', 'quantity_dispensed', 'item_stock_agg_zone', 'consumption_rate_per_day',
        'ai_risk_score', 'ai_followup_priority_score', # From AI models
        'notes', 'sample_collection_date', 'sample_registered_lab_date', 'sample_status', 'rejection_reason',
        'diagnosis_code_icd10', 'physician_id', 'avg_spo2', 'avg_daily_steps',
        'resting_heart_rate', 'avg_sleep_duration_hrs', 'sleep_score_pct', 'stress_level_score',
        'screening_hpv_status', 'hiv_viral_load_copies_ml', 'key_chronic_conditions_summary',
        'chw_visit', 'tb_contact_traced', 
        'patient_latitude', 'patient_longitude'
    ]
    return pd.DataFrame(columns=list(set(cols))) # Use set to ensure unique column names

@pytest.fixture
def empty_iot_df_sentinel_schema() -> pd.DataFrame:
    cols = [
        'timestamp', 'clinic_id', 'room_name', 'zone_id', 'avg_co2_ppm', 'max_co2_ppm',
        'avg_pm25', 'voc_index', 'avg_temp_celsius', 'avg_humidity_rh', 'avg_noise_db',
        'waiting_room_occupancy', 'patient_throughput_per_hour', 'sanitizer_dispenses_per_hour'
    ]
    return pd.DataFrame(columns=cols)

@pytest.fixture
def empty_zone_attributes_sentinel_schema() -> pd.DataFrame: # Simulates zone_attributes.csv content
    cols = ['zone_id', 'name', 'population', 'socio_economic_index', 'num_clinics',
            'avg_travel_time_clinic_min', 'predominant_hazard_type', 
            'typical_workforce_exposure_level', 'primary_livelihood', 'water_source_main', 'area_sqkm']
    return pd.DataFrame(columns=cols)

@pytest.fixture
def empty_enriched_gdf_sentinel_schema() -> gpd.GeoDataFrame:
    # Columns expected after `enrich_zone_geodata_with_health_aggregates`
    base_attr_cols = ['zone_id', 'name', 'geometry', 'population', 'socio_economic_index', 'num_clinics',
                      'avg_travel_time_clinic_min', 'predominant_hazard_type', 'typical_workforce_exposure_level',
                      'primary_livelihood', 'water_source_main', 'area_sqkm'] # From load_zone_data
    
    health_agg_cols = [
        'total_population_health_data', 'avg_risk_score', 'total_patient_encounters',
        'total_referrals_made', 'successful_referrals',
        'avg_test_turnaround_critical', 'perc_critical_tests_tat_met',
        'total_active_key_infections', 'prevalence_per_1000',
        'avg_daily_steps_zone', 'facility_coverage_score', 'population_density'
    ]
    iot_agg_cols = ['zone_avg_co2']
    
    condition_specific_cols = []
    for cond_key in app_config.KEY_CONDITIONS_FOR_ACTION:
        col_name = f"active_{cond_key.lower().replace(' ', '_').replace('-', '_').replace('(severe)','')}_cases"
        condition_specific_cols.append(col_name)
    
    all_expected_cols = list(set(base_attr_cols + health_agg_cols + iot_agg_cols + condition_specific_cols))
    
    return gpd.GeoDataFrame(columns=all_expected_cols, geometry='geometry', crs=app_config.DEFAULT_CRS_STANDARD)


# --- Generic Plotting Data Fixtures ---
@pytest.fixture(scope="session")
def sample_series_data_sentinel() -> pd.Series:
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06'])
    return pd.Series([10, 12, 9, 15, 13, 14], index=dates, name="MetricValueSeries")

@pytest.fixture(scope="session")
def sample_bar_df_sentinel() -> pd.DataFrame:
    return pd.DataFrame({
        'category_axis': ['Alpha', 'Beta', 'Gamma', 'Alpha', 'Delta', 'Beta'],
        'value_axis': [22, 18, 28, 35, 15, 21],
        'grouping_col': ['GroupX', 'GroupY', 'GroupX', 'GroupY', 'GroupX', 'GroupY']
    })

@pytest.fixture(scope="session")
def sample_donut_df_sentinel() -> pd.DataFrame:
    return pd.DataFrame({
        'status_category': ['High Risk', 'Moderate Risk', 'Low Risk', 'Neutral/Unknown'],
        'patient_counts': [8, 15, 45, 5]
    })

@pytest.fixture(scope="session")
def sample_heatmap_df_sentinel() -> pd.DataFrame:
    # A sample correlation matrix or similar pivot table
    idx = ['MetricA', 'MetricB', 'MetricC', 'MetricD']
    cols = ['Zone1', 'Zone2', 'Zone3']
    data = np.random.rand(4, 3) * 2 - 1 # Values between -1 and 1
    return pd.DataFrame(data, index=idx, columns=cols)

@pytest.fixture(scope="session")
def sample_choropleth_gdf_sentinel(sample_zone_geometries_gdf_main_sentinel: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """ Creates a GDF suitable for choropleth map testing by adding a sample value column. """
    if not isinstance(sample_zone_geometries_gdf_main_sentinel, gpd.GeoDataFrame) or \
       sample_zone_geometries_gdf_main_sentinel.empty:
        return gpd.GeoDataFrame(columns=['zone_id', 'name', 'geometry', 'risk_score'], geometry='geometry', crs=app_config.DEFAULT_CRS_STANDARD) # Return empty with schema

    gdf_map_test = sample_zone_geometries_gdf_main_sentinel.copy()
    # Add a sample numeric column for coloring the choropleth
    np.random.seed(42) # for reproducibility
    gdf_map_test['risk_score'] = np.random.randint(20, 85, len(gdf_map_test))
    gdf_map_test['facility_count'] = np.random.randint(0, 5, len(gdf_map_test)) # Another example column
    return gdf_map_test
