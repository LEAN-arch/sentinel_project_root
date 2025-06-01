# sentinel_project_root/test/tests/conftest.py
# Pytest fixtures for testing the "Sentinel Health Co-Pilot" application.

import pytest
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import numpy as np
from datetime import datetime, date, timedelta
import sys
import os

# --- Path Setup for Imports ---
# Add the 'test' directory (parent of 'tests', contains 'config' and 'utils') to sys.path
# This allows utils and config to be imported directly by test files and this conftest.
_current_conftest_dir = os.path.dirname(os.path.abspath(__file__))
_project_test_dir = os.path.abspath(os.path.join(_current_conftest_dir, os.pardir))

if _project_test_dir not in sys.path:
    sys.path.insert(0, _project_test_dir)

# --- Critical Project Module Imports ---
try:
    from config import app_config
    from utils.core_data_processing import enrich_zone_geodata_with_health_aggregates
    from utils.ai_analytics_engine import apply_ai_models
except ImportError as e:
    # This is a critical failure for test setup.
    print(f"FATAL ERROR in conftest.py: Could not import core project modules. Tests will not run correctly.")
    print(f"PYTHONPATH currently is: {sys.path}")
    print(f"Attempted to add: {_project_test_dir}")
    print(f"Error details: {e}")
    raise # Re-raise the error to halt pytest collection if critical modules are missing.


# --- Fixture for Sample Health Records ---
@pytest.fixture(scope="session")
def sample_health_records_df_main_sentinel() -> pd.DataFrame:
    """
    Provides a comprehensive, AI-enriched DataFrame of sample health records.
    """
    base_date = datetime(2023, 10, 1, 9, 30, 0)
    num_records = 60 # Increased for more variety
    record_dates = [base_date + timedelta(hours=i * 2, days=i // 5) for i in range(num_records)]

    raw_data = {
        'encounter_id': [f'SENC_TEST_{i:03d}' for i in range(1, num_records + 1)],
        'patient_id': [f'SPID_TEST_{(i % 25):03d}' for i in range(1, num_records + 1)], # ~25 unique patients
        'encounter_date': record_dates,
        'encounter_type': np.random.choice(['CHW_HOME_VISIT', 'CHW_ALERT_RESPONSE', 'CLINIC_INTAKE', 'WORKER_SELF_CHECK', 'CHW_SCHEDULED_DOTS'], num_records).tolist(),
        'age': np.random.randint(0, 95, num_records).tolist(), # Include 0 for infants, up to 94
        'gender': np.random.choice(['Male', 'Female', 'Other', 'Unknown'], num_records, p=[0.45,0.45,0.05,0.05]).tolist(),
        'pregnancy_status': np.random.choice([0, 1, 0, 0, 0, 0], num_records).tolist(), # Weighted towards not pregnant
        'chronic_condition_flag': np.random.choice([0, 1, 0, 0], num_records).tolist(),
        'zone_id': [f"Zone{chr(65 + (i % 4))}" for i in range(num_records)], # ZoneA, B, C, D
        'clinic_id': [f"CLINIC{(i % 3) + 1:02d}" if (i%3!=0) else "HUB01" for i in range(num_records)], # Mix of clinics and hubs
        'chw_id': [f"CHW{(i % 6) + 1:03d}" if 'CHW' in et else "N/A" for i, et in enumerate(np.random.choice(['CHW_HOME_VISIT', 'CLINIC_INTAKE'], num_records))],
        
        'hrv_rmssd_ms': np.random.uniform(app_config.STRESS_HRV_LOW_THRESHOLD_MS - 8, 75, num_records).round(1).tolist(),
        'min_spo2_pct': np.random.choice([app_config.ALERT_SPO2_CRITICAL_LOW_PCT - 3, app_config.ALERT_SPO2_CRITICAL_LOW_PCT, app_config.ALERT_SPO2_WARNING_LOW_PCT - 1, app_config.ALERT_SPO2_WARNING_LOW_PCT, 96, 98, 99], num_records).tolist(),
        'vital_signs_temperature_celsius': np.random.choice([36.2, 37.0, app_config.ALERT_BODY_TEMP_FEVER_C + 0.3, app_config.ALERT_BODY_TEMP_HIGH_FEVER_C - 0.2, app_config.ALERT_BODY_TEMP_HIGH_FEVER_C + 0.3], num_records).tolist(),
        'max_skin_temp_celsius': lambda df: df['vital_signs_temperature_celsius'] + np.random.uniform(-0.4, 0.4, len(df)), # Dynamic calculation
        'movement_activity_level': np.random.choice([0, 1, 2, 3, 2], num_records).tolist(), # 0=None, 1=Low, 2=Med, 3=High
        'fall_detected_today': np.random.choice([0] * 10 + [1], num_records).tolist(), # Weighted towards no falls
        
        'ambient_heat_index_c': np.random.uniform(22, app_config.ALERT_AMBIENT_HEAT_INDEX_DANGER_C + 8, num_records).round(1).tolist(),
        'ppe_compliant_flag': np.random.choice([0, 1, 1, 1, 1, 1], num_records).tolist(), # Weighted towards compliant
        'signs_of_fatigue_observed_flag': np.random.choice([0, 0, 0, 0, 1], num_records).tolist(),
        'rapid_psychometric_distress_score': np.random.randint(0, 11, num_records).tolist(),

        'condition': np.random.choice(app_config.KEY_CONDITIONS_FOR_ACTION + ['Wellness Visit', 'Minor Cold', 'Injury', 'Hypertension Follow-up'], num_records).tolist(),
        'patient_reported_symptoms': ['fever;cough;fatigue', 'none', 'headache;dizzy', 'diarrhea', 'short breath;chest discomfort', 'skin rash;itching', 'general malaise'] * (num_records // 7) + ['fever'] * (num_records % 7),
        
        'test_type': np.random.choice(list(app_config.KEY_TEST_TYPES_FOR_ANALYSIS.keys()) + ["None", "PulseOx", "BP Check"], num_records).tolist(),
        'test_result': np.random.choice(['Positive', 'Negative', 'Pending', 'N/A', '89', '93', '120/80', '145/92'], num_records).tolist(),
        'test_turnaround_days': [np.random.uniform(0, 8).round(1) if res not in ["Pending", "N/A"] and not res.replace('.','',1).isdigit() else np.nan for res in np.random.choice(['Positive', 'Pending', 'N/A', '90'], num_records)],

        'referral_status': np.random.choice(['Pending', 'Completed', 'Initiated', 'N/A', 'Declined', 'Attended'], num_records).tolist(),
        'referral_reason': ['Urgent Consult', 'Lab Test Needed', 'Specialist Opinion', 'Routine Follow-up', 'Emergency Transport Required', 'None', 'Further Investigation'] * (num_records // 7) + ['Urgent'] * (num_records % 7),
        'referred_to_facility_id': [f"FACIL{(i%4)+1:02d}" if stat != "N/A" else "N/A" for i, stat in enumerate(np.random.choice(['Pending', 'N/A', 'Completed'], num_records))],
        
        'medication_adherence_self_report': np.random.choice(['Good', 'Fair', 'Poor', 'N/A', 'Unknown', 'Excellent'], num_records).tolist(),
        'item': np.random.choice(app_config.KEY_DRUG_SUBSTRINGS_SUPPLY + ['Bandages', 'Gloves', 'ORS Sachet', 'Syringe'], num_records).tolist(),
        'quantity_dispensed': [np.random.randint(0,3) if itm in app_config.KEY_DRUG_SUBSTRINGS_SUPPLY else np.random.randint(0,10) for itm in np.random.choice(app_config.KEY_DRUG_SUBSTRINGS_SUPPLY + ['Bandages'], num_records)],
        'item_stock_agg_zone': np.random.randint(0, 200, num_records).tolist(),
        'consumption_rate_per_day': np.random.uniform(0.01, 4.0, num_records).round(3).tolist(),
        
        'notes': ["Sample note for encounter " + str(i) for i in range(num_records)],
        'sample_collection_date': [d - timedelta(hours=np.random.randint(0,4), minutes=np.random.randint(0,59)) if tt not in ["None", "PulseOx", "BP Check"] else pd.NaT for d, tt in zip(record_dates, np.random.choice(list(app_config.KEY_TEST_TYPES_FOR_ANALYSIS.keys()) + ["None", "PulseOx", "BP Check"], num_records))],
        'sample_registered_lab_date': lambda df: [d + timedelta(hours=np.random.randint(0,2)) if pd.notna(d) else pd.NaT for d in df['sample_collection_date']],
        'sample_status': np.random.choice(['Accepted', 'Pending Collection', 'Rejected', 'N/A', 'In Transit'], num_records).tolist(),
        'rejection_reason': [np.random.choice(['Hemolyzed Sample', 'Insufficient Volume', 'Improper Labeling', 'N/A', 'Contaminated Specimen']) if s == 'Rejected' else 'N/A' for s in np.random.choice(['Accepted', 'Rejected', 'N/A'], num_records)],
        'avg_daily_steps': np.random.choice([np.nan] + list(range(500, 12000, 500)), num_records).tolist() # Add some NaNs
    }
    df = pd.DataFrame({k: v for k,v in raw_data.items() if not callable(v)})
    # Apply lambda functions for dynamically calculated columns
    if callable(raw_data['max_skin_temp_celsius']):
        df['max_skin_temp_celsius'] = raw_data['max_skin_temp_celsius'](df)
    if callable(raw_data['sample_registered_lab_date']):
        df['sample_registered_lab_date'] = raw_data['sample_registered_lab_date'](df)

    # Apply AI models (this function should handle missing columns by adding defaults)
    enriched_df, _ = apply_ai_models(df.copy(), source_context="ConftestHealthFixtureAI")
    
    # Final type coercions for consistency after all processing
    date_cols_final = ['encounter_date', 'sample_collection_date', 'sample_registered_lab_date']
    for col in date_cols_final:
        if col in enriched_df.columns:
            enriched_df[col] = pd.to_datetime(enriched_df[col], errors='coerce')
    
    return enriched_df


@pytest.fixture(scope="session")
def sample_iot_clinic_df_main_sentinel() -> pd.DataFrame:
    """Provides sample IoT data for clinic environments with varied values."""
    num_iot_records = 15
    base_iot_date = datetime(2023, 10, 20, 8, 0, 0)
    iot_timestamps = [base_iot_date + timedelta(hours=i) for i in range(num_iot_records)]
    
    iot_fixture_data = {
        'timestamp': iot_timestamps,
        'clinic_id': [f"CLINIC{(i%3)+1:02d}" for i in range(num_iot_records)],
        'room_name': np.random.choice(['WaitingArea_Main', 'ConsultRoom_1', 'Lab_SectionA', 'TB_Screening_Zone', 'Pharmacy_Dispense'], num_iot_records).tolist(),
        'zone_id': [f"Zone{chr(65+(i%4))}" for i in range(num_iot_records)],
        'avg_co2_ppm': np.random.choice([700, 1200, app_config.ALERT_AMBIENT_CO2_HIGH_PPM + 100, app_config.ALERT_AMBIENT_CO2_VERY_HIGH_PPM + 50, 2800], num_iot_records).tolist(),
        'max_co2_ppm': lambda df: (df['avg_co2_ppm'] * np.random.uniform(1.1, 1.3, len(df))).round(0),
        'avg_pm25': np.random.choice([10.0, app_config.ALERT_AMBIENT_PM25_HIGH_UGM3 + 5, 25.0, app_config.ALERT_AMBIENT_PM25_VERY_HIGH_UGM3 + 8, 60.0], num_iot_records).tolist(),
        'voc_index': np.random.randint(40, 300, num_iot_records).tolist(),
        'avg_temp_celsius': np.random.uniform(21.0, 29.5, num_iot_records).round(1).tolist(),
        'avg_humidity_rh': np.random.uniform(35, 80, num_iot_records).round(0).tolist(),
        'avg_noise_db': np.random.choice([50, 65, app_config.ALERT_AMBIENT_NOISE_HIGH_DBA - 5, app_config.ALERT_AMBIENT_NOISE_HIGH_DBA + 10, 92], num_iot_records).tolist(),
        'waiting_room_occupancy': [np.random.randint(2, app_config.TARGET_CLINIC_WAITING_ROOM_OCCUPANCY_MAX + 8) if 'Waiting' in r else np.nan for r in np.random.choice(['WaitingArea_Main', 'ConsultRoom_1'], num_iot_records)],
        'patient_throughput_per_hour': np.random.randint(app_config.TARGET_CLINIC_PATIENT_THROUGHPUT_MIN_PER_HOUR -2, 15, num_iot_records).tolist(),
        'sanitizer_dispenses_per_hour': np.random.randint(0, 20, num_iot_records).tolist()
    }
    df_iot = pd.DataFrame({k: v for k,v in iot_fixture_data.items() if not callable(v)})
    if callable(iot_fixture_data['max_co2_ppm']):
        df_iot['max_co2_ppm'] = iot_fixture_data['max_co2_ppm'](df_iot)
    return df_iot


@pytest.fixture(scope="session")
def sample_zone_geometries_gdf_main_sentinel() -> gpd.GeoDataFrame:
    """Provides a sample GeoDataFrame with zone geometries and merged base attributes."""
    attributes = {
        'zone_id': ['ZoneA', 'ZoneB', 'ZoneC', 'ZoneD', 'ZoneE'], # Added ZoneE
        'name': ['Alpha Central', 'Beta South Plains', 'Gamma East Hills', 'Delta West Coast', 'Epsilon North Valley'],
        'population': [11000, 16000, 7000, 12500, 9000],
        'socio_economic_index': [0.60, 0.35, 0.78, 0.45, 0.55],
        'num_clinics': [2, 1, 1, 3, 1],
        'avg_travel_time_clinic_min': [18, 50, 12, 22, 35],
        'predominant_hazard_type': ['HEAT_STRESS', 'FLOOD_RISK', 'AIR_QUALITY_POOR', 'NONE', 'LANDSLIDE_RISK'],
        'typical_workforce_exposure_level': ['HIGH', 'MODERATE_HIGH', 'HIGH', 'LOW', 'MODERATE'],
        'primary_livelihood': ['Urban Services', 'Agriculture', 'Factory/Industrial', 'Mixed Economy', 'Pastoralism'],
        'water_source_main': ['Piped Network', 'Borehole/Well', 'Piped Intermittent', 'Piped Network', 'River/Spring'],
        'area_sqkm': [45.0, 130.5, 28.0, 82.3, 65.0] 
    }
    attr_df = pd.DataFrame(attributes)

    geometries = [
        {"zone_id": "ZoneA", "geometry": Polygon([[0,0],[0,1.1],[1.1,1.1],[1.1,0],[0,0]])}, # Slightly larger
        {"zone_id": "ZoneB", "geometry": Polygon([[1,0],[1,1.2],[2.2,1.2],[2.2,0],[1,0]])},
        {"zone_id": "ZoneC", "geometry": Polygon([[0,1],[0,2.1],[1.1,2.1],[1.1,1],[0,1]])},
        {"zone_id": "ZoneD", "geometry": Polygon([[1,1],[1,2.2],[2.2,2.2],[2.2,1],[1,1]])},
        {"zone_id": "ZoneE", "geometry": Polygon([[-1,0],[-1,1],[0,1],[0,0],[-1,0]])} # New zone
    ]
    geom_gdf = gpd.GeoDataFrame(geometries, geometry='geometry', crs=app_config.DEFAULT_CRS_STANDARD)
    
    merged_gdf = geom_gdf.merge(attr_df, on='zone_id', how='left')
    numeric_attr_cols = ['population', 'socio_economic_index', 'num_clinics', 'avg_travel_time_clinic_min', 'area_sqkm']
    for col_num in numeric_attr_cols:
        if col_num in merged_gdf.columns:
            merged_gdf[col_num] = pd.to_numeric(merged_gdf[col_num], errors='coerce')
    return merged_gdf


@pytest.fixture(scope="session")
def sample_enriched_gdf_main_sentinel(
    sample_zone_geometries_gdf_main_sentinel: gpd.GeoDataFrame,
    sample_health_records_df_main_sentinel: pd.DataFrame,
    sample_iot_clinic_df_main_sentinel: pd.DataFrame
) -> Optional[gpd.GeoDataFrame]:
    if not isinstance(sample_zone_geometries_gdf_main_sentinel, gpd.GeoDataFrame) or sample_zone_geometries_gdf_main_sentinel.empty:
        pytest.skip("Base zone GDF fixture is invalid for creating enriched GDF.")
        return None # Should cause test to skip if this fixture is critical for it

    # Use copies to avoid modifying fixtures if enrichment function is not purely functional
    enriched_gdf = enrich_zone_geodata_with_health_aggregates(
        zone_gdf=sample_zone_geometries_gdf_main_sentinel.copy(),
        health_df=sample_health_records_df_main_sentinel.copy(),
        iot_df=sample_iot_clinic_df_main_sentinel.copy(),
        source_context="ConftestEnrichment/SentinelMain"
    )
    return enriched_gdf


# --- Fixtures for Empty Schemas ---
@pytest.fixture
def empty_health_df_sentinel_schema() -> pd.DataFrame:
    # This should represent the most comprehensive schema after all loading, cleaning, and AI enrichment.
    # Derived from health_records_expanded.csv structure and AI model outputs.
    cols = [
        'encounter_id', 'patient_id', 'encounter_date', 'encounter_date_obj', 'encounter_type', 
        'age', 'gender', 'pregnancy_status', 'chronic_condition_flag', 'zone_id', 'clinic_id', 'chw_id',
        'hrv_rmssd_ms', 'min_spo2_pct', 'vital_signs_temperature_celsius', 'max_skin_temp_celsius',
        'movement_activity_level', 'fall_detected_today', 'ambient_heat_index_c', 'ppe_compliant_flag',
        'signs_of_fatigue_observed_flag', 'rapid_psychometric_distress_score', 'condition', 
        'patient_reported_symptoms', 'test_type', 'test_result', 'test_turnaround_days',
        'sample_collection_date', 'sample_registered_lab_date', 'sample_status', 'rejection_reason',
        'referral_status', 'referral_reason', 'referred_to_facility_id', 'referral_outcome', 'referral_outcome_date', 'referral_date',
        'medication_adherence_self_report', 'item', 'quantity_dispensed', 'item_stock_agg_zone', 
        'consumption_rate_per_day', 'notes', 'diagnosis_code_icd10', 'physician_id', 'avg_spo2', 
        'avg_daily_steps', 'resting_heart_rate', 'avg_sleep_duration_hrs', 'sleep_score_pct', 
        'stress_level_score', 'screening_hpv_status', 'hiv_viral_load_copies_ml', 
        'key_chronic_conditions_summary', 'chw_visit', 'tb_contact_traced', 
        'patient_latitude', 'patient_longitude',
        'ai_risk_score', 'ai_followup_priority_score' # Added by AI engine
    ]
    return pd.DataFrame(columns=list(set(cols))) # Use set to ensure unique columns

@pytest.fixture
def empty_iot_df_sentinel_schema() -> pd.DataFrame:
    cols = [
        'timestamp', 'clinic_id', 'room_name', 'zone_id', 'avg_co2_ppm', 'max_co2_ppm',
        'avg_pm25', 'voc_index', 'avg_temp_celsius', 'avg_humidity_rh', 'avg_noise_db',
        'waiting_room_occupancy', 'patient_throughput_per_hour', 'sanitizer_dispenses_per_hour'
    ]
    return pd.DataFrame(columns=cols)

@pytest.fixture
def empty_zone_attributes_sentinel_schema() -> pd.DataFrame: # Simulates zone_attributes.csv
    cols = ['zone_id', 'name', 'population', 'socio_economic_index', 'num_clinics',
            'avg_travel_time_clinic_min', 'predominant_hazard_type', 
            'typical_workforce_exposure_level', 'primary_livelihood', 'water_source_main', 'area_sqkm']
    return pd.DataFrame(columns=cols)

@pytest.fixture
def empty_enriched_gdf_sentinel_schema() -> gpd.GeoDataFrame:
    # Columns expected after `enrich_zone_geodata_with_health_aggregates`
    base_cols_enriched = [
        'zone_id', 'name', 'geometry', 'population', 'socio_economic_index', 'num_clinics',
        'avg_travel_time_clinic_min', 'predominant_hazard_type', 'typical_workforce_exposure_level',
        'primary_livelihood', 'water_source_main', 'area_sqkm', # Base attributes from load_zone_data
        'total_population_health_data', 'avg_risk_score', 'total_patient_encounters',
        'total_referrals_made', 'successful_referrals',
        'avg_test_turnaround_critical', 'perc_critical_tests_tat_met',
        'total_active_key_infections', 'prevalence_per_1000',
        'avg_daily_steps_zone', 'zone_avg_co2', 'facility_coverage_score', 'population_density',
        'chw_density_per_10k' # Placeholder, needs chw_count_zone for actual calculation
    ]
    dynamic_condition_cols = [f"active_{cond_key.lower().replace(' ', '_').replace('-', '_').replace('(severe)','')}_cases" 
                              for cond_key in app_config.KEY_CONDITIONS_FOR_ACTION]
    
    all_cols_final_enriched = list(set(base_cols_enriched + dynamic_condition_cols))
    return gpd.GeoDataFrame(columns=all_cols_final_enriched, geometry='geometry', crs=app_config.DEFAULT_CRS_STANDARD)


# --- Generic Plotting Data Fixtures ---
@pytest.fixture(scope="session")
def sample_series_data_sentinel() -> pd.Series:
    idx_dates = pd.to_datetime(['2023-03-01', '2023-03-08', '2023-03-15', '2023-03-22', '2023-03-29', '2023-04-05'])
    return pd.Series([15, 18, 12, 20, 17, 22], index=idx_dates, name="WeeklyMetric")

@pytest.fixture(scope="session")
def sample_bar_df_sentinel() -> pd.DataFrame:
    return pd.DataFrame({
        'category_label': ['Condition A', 'Condition B', 'Condition C', 'Condition A', 'Condition D', 'Condition B'],
        'value_count': [25, 19, 30, 38, 12, 24],
        'group_by_col': ['Group1', 'Group2', 'Group1', 'Group2', 'Group1', 'Group2']
    })

@pytest.fixture(scope="session")
def sample_donut_df_sentinel() -> pd.DataFrame:
    return pd.DataFrame({
        'risk_level_label': ['Critical Risk', 'Warning Level', 'Acceptable', 'Data Pending'],
        'case_counts': [10, 25, 60, 5]
    })

@pytest.fixture(scope="session")
def sample_heatmap_df_sentinel() -> pd.DataFrame:
    # Correlation-like matrix
    rows_heatmap = ['Symptom_Fever', 'Symptom_Cough', 'Symptom_Fatigue', 'Low_SpO2_Alert']
    cols_heatmap = ['Zone_Alpha', 'Zone_Beta', 'Zone_Gamma']
    data_heatmap = np.array([[0.8, 0.2, 0.5], [0.7, 0.9, 0.3], [0.6, 0.5, 0.8], [0.9, 0.1, 0.2]])
    return pd.DataFrame(data_heatmap, index=rows_heatmap, columns=cols_heatmap)

@pytest.fixture(scope="session")
def sample_choropleth_gdf_sentinel(sample_zone_geometries_gdf_main_sentinel: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """ Creates a GDF suitable for choropleth map testing by adding sample value columns. """
    if not isinstance(sample_zone_geometries_gdf_main_sentinel, gpd.GeoDataFrame) or sample_zone_geometries_gdf_main_sentinel.empty:
        # Return an empty GDF with expected schema if base is invalid, tests using this should skip.
        return gpd.GeoDataFrame(columns=['zone_id', 'name', 'geometry', 'sample_risk_value', 'sample_facility_count'], 
                                geometry='geometry', crs=app_config.DEFAULT_CRS_STANDARD)

    gdf_for_map = sample_zone_geometries_gdf_main_sentinel.copy()
    # Add sample numeric columns for coloring the choropleth
    rng_map = np.random.RandomState(123) # For reproducible random values
    gdf_for_map['sample_risk_value'] = rng_map.randint(10, 95, len(gdf_for_map))
    gdf_for_map['sample_facility_count'] = rng_map.randint(0, 6, len(gdf_for_map))
    gdf_for_map['population_density_sample'] = (gdf_for_map['population'] / gdf_for_map['area_sqkm']).fillna(0).round(1) if 'population' in gdf_for_map and 'area_sqkm' in gdf_for_map else np.nan
    return gdf_for_map
