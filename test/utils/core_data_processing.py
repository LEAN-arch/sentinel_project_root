# sentinel_project_root/test/utils/core_data_processing.py
# Redesigned for "Sentinel Health Co-Pilot" - LMIC Edge-First System
# This module provides robust data loading, cleaning, and aggregation utilities
# primarily intended for:
#   1. Facility Node (Tier 2) and Cloud (Tier 3) backend processing.
#   2. Initial data provisioning and system setup for simulations or demos.
#   3. Simulation and testing environments.
import streamlit as st
import pandas as pd
import geopandas as gpd # << THE CRITICAL IMPORT
import numpy as np
import os
import sys 
import logging
from config import app_config
from typing import List, Dict, Any, Optional, Tuple, Union

logger = logging.getLogger(__name__)
GEOPANDAS_AVAILABLE = True

# --- I. Core Helper Functions ---
def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        logger.error(f"_clean_column_names expects a pandas DataFrame, got {type(df)}.")
        return df if df is not None else pd.DataFrame()
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
    return df

def _convert_to_numeric(series: pd.Series, default_value: Any = np.nan) -> pd.Series:
    if not isinstance(series, pd.Series):
        logger.debug(f"_convert_to_numeric given non-Series type: {type(series)}. Attempting conversion.")
        try:
            series = pd.Series(series, dtype=float if default_value is np.nan else type(default_value))
        except Exception as e_series:
            logger.error(f"Could not convert input of type {type(series)} to Series in _convert_to_numeric: {e_series}")
            length = len(series) if hasattr(series, '__len__') else 1
            dtype_val = type(default_value) if default_value is not np.nan else float
            return pd.Series([default_value] * length, dtype=dtype_val)
    return pd.to_numeric(series, errors='coerce').fillna(default_value)

def hash_geodataframe(gdf: gpd.GeoDataFrame) -> Optional[str]:
    if gdf is None or not isinstance(gdf, gpd.GeoDataFrame): return None
    try:
        geom_col_name = gdf.geometry.name if hasattr(gdf, 'geometry') and hasattr(gdf.geometry, 'name') else 'geometry'
        non_geom_cols, geom_hash_val = [], 0
        if geom_col_name in gdf.columns and hasattr(gdf[geom_col_name], 'is_empty') and not gdf[geom_col_name].is_empty.all():
            non_geom_cols = gdf.drop(columns=[geom_col_name], errors='ignore').columns.tolist()
            valid_geoms = gdf[geom_col_name][gdf[geom_col_name].is_valid & ~gdf[geom_col_name].is_empty]
            if not valid_geoms.empty: geom_hash_val = pd.util.hash_array(valid_geoms.to_wkt().values).sum()
            else: geom_hash_val = pd.util.hash_array(gdf[geom_col_name].astype(str).values).sum()
        else: non_geom_cols = gdf.columns.tolist()
        df_content_hash = 0
        if non_geom_cols:
            df_to_hash = gdf[non_geom_cols].copy()
            dt_cols_to_convert = df_to_hash.select_dtypes(include=['datetime64', 'datetime64[ns]', 'datetime64[ns, UTC]']).columns
            for col in dt_cols_to_convert:
                df_to_hash[col] = pd.to_datetime(df_to_hash[col], errors='coerce').astype('int64') // 10**9
            for col in df_to_hash.select_dtypes(include=['timedelta64', 'timedelta64[ns]']).columns:
                df_to_hash[col] = df_to_hash[col].astype('int64')
            try: df_content_hash = pd.util.hash_pandas_object(df_to_hash, index=True).sum()
            except TypeError as e_ht:
                logger.warning(f"Unhashable type in GDF for hashing: {e_ht}. Converting offenders to string.")
                for col_offending in df_to_hash.columns:
                    try: pd.util.hash_pandas_object(df_to_hash[[col_offending]], index=True)
                    except TypeError: df_to_hash[col_offending] = df_to_hash[col_offending].astype(str)
                df_content_hash = pd.util.hash_pandas_object(df_to_hash, index=True).sum()
        return f"{df_content_hash}-{geom_hash_val}"
    except Exception as e_hash_gdf_exc: logger.error(f"GDF hashing failed: {e_hash_gdf_exc}", exc_info=True); return str(gdf.head(1).to_string()) + str(gdf.shape)

def _robust_merge_agg(left_df: pd.DataFrame, right_df: pd.DataFrame, target_col_name: str, on_col: str = 'zone_id', default_fill_value: Any = 0.0) -> pd.DataFrame:
    if not isinstance(left_df, pd.DataFrame): return pd.DataFrame(columns=[on_col, target_col_name]) if left_df is None else left_df
    left_df_work = left_df.copy()
    if target_col_name not in left_df_work.columns: left_df_work[target_col_name] = default_fill_value
    else: left_df_work[target_col_name] = left_df_work[target_col_name].fillna(default_fill_value)
    if not isinstance(right_df, pd.DataFrame) or right_df.empty or on_col not in right_df.columns: return left_df_work
    value_col_candidates = [col for col in right_df.columns if col != on_col]
    if not value_col_candidates: return left_df_work
    value_col_in_right = value_col_candidates[0]
    try:
        left_df_work[on_col] = left_df_work.get(on_col, pd.Series(dtype=str)).astype(str).str.strip()
        right_df_for_merge = right_df[[on_col, value_col_in_right]].copy()
        right_df_for_merge[on_col] = right_df_for_merge[on_col].astype(str).str.strip()
    except Exception as e_type_rm: logger.error(f"Type conversion error for '{on_col}' in robust_merge for '{target_col_name}': {e_type_rm}"); return left_df_work
    temp_agg_col = f"__temp_agg_{target_col_name.replace(' ', '_')}_{np.random.randint(0, 1000000)}__"
    right_df_for_merge.rename(columns={value_col_in_right: temp_agg_col}, inplace=True)
    original_index = left_df_work.index; original_index_name = left_df_work.index.name
    reset_required = not isinstance(original_index, pd.RangeIndex) or original_index_name is not None
    if reset_required: left_df_for_merge = left_df_work.reset_index()
    else: left_df_for_merge = left_df_work
    merged_df = left_df_for_merge.merge(right_df_for_merge, on=on_col, how='left')
    if temp_agg_col in merged_df.columns:
        merged_df[target_col_name] = merged_df[temp_agg_col].combine_first(merged_df.get(target_col_name))
        merged_df.drop(columns=[temp_agg_col], inplace=True, errors='ignore')
    merged_df[target_col_name].fillna(default_fill_value, inplace=True)
    if reset_required:
        index_col_restore = original_index_name if original_index_name else 'index'
        if index_col_restore in merged_df.columns:
            merged_df.set_index(index_col_restore, inplace=True, drop=True)
            if original_index_name: merged_df.index.name = original_index_name
        elif len(merged_df) == len(original_index): merged_df.index = original_index
        else: logger.warning(f"Index restoration issue in robust_merge for {target_col_name}, using RangeIndex.")
    return merged_df


# --- II. Data Loading and Basic Cleaning Functions ---
@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS_WEB_REPORTS, show_spinner="Loading health records dataset...")
def load_health_records(file_path: Optional[str] = None, source_context: str = "FacilityNode") -> pd.DataFrame:
    actual_file_path = file_path or app_config.HEALTH_RECORDS_CSV
    logger.info(f"({source_context}) Attempting to load health records from: {actual_file_path}")
    is_streamlit_active_hr = "streamlit" in sys.modules and hasattr(st, 'session_state') and hasattr(st.session_state, 'run_count') # Check if in full Streamlit context
    if not os.path.exists(actual_file_path):
        logger.error(f"({source_context}) Health records file not found: {actual_file_path}")
        if is_streamlit_active_hr : st.error(f"🚨 Health records file '{os.path.basename(actual_file_path)}' not found.")
        return pd.DataFrame()
    try:
        df = pd.read_csv(actual_file_path, low_memory=False); df = _clean_column_names(df)
        logger.info(f"({source_context}) Loaded {len(df)} raw records from {actual_file_path}.")
        date_cols = ['encounter_date', 'sample_collection_date', 'sample_registered_lab_date', 'referral_date', 'referral_outcome_date']
        for col in date_cols:
            if col in df.columns: df[col] = pd.to_datetime(df.get(col), errors='coerce')
            else: df[col] = pd.NaT
        num_cols_map = { # Includes Sentinel lean data model fields and originals
            'test_turnaround_days': np.nan, 'quantity_dispensed': 0, 'item_stock_agg_zone': 0,
            'consumption_rate_per_day': 0.0, 'ai_risk_score': np.nan, 'ai_followup_priority_score': np.nan,
            'vital_signs_bp_systolic': np.nan, 'vital_signs_bp_diastolic': np.nan,
            'vital_signs_temperature_celsius': np.nan, 'max_skin_temp_celsius': np.nan,
            'min_spo2_pct': np.nan, 'avg_spo2': np.nan, 'avg_daily_steps': 0,
            'resting_heart_rate': np.nan, 'hrv_rmssd_ms': np.nan,
            'avg_sleep_duration_hrs': np.nan, 'sleep_score_pct': np.nan, 'stress_level_score': np.nan,
            'fall_detected_today': 0, 'age': np.nan, 'chw_visit': 0, 'tb_contact_traced': 0,
            'patient_latitude': np.nan, 'patient_longitude': np.nan, 'hiv_viral_load_copies_ml': np.nan,
            'pregnancy_status': 0, 'chronic_condition_flag':0, 'ppe_compliant_flag':1,
            'signs_of_fatigue_observed_flag': 0, 'rapid_psychometric_distress_score': np.nan,
            'movement_activity_level':np.nan, 'ambient_heat_index_c':np.nan
        }
        for col, default_val in num_cols_map.items():
            if col in df.columns: df[col] = _convert_to_numeric(df.get(col), default_val)
            else: df[col] = default_val
        str_cols = [
            'encounter_id','patient_id','encounter_type','condition','diagnosis_code_icd10',
            'test_type','test_result','item','zone_id','clinic_id','chw_id', 'physician_id',
            'notes','patient_reported_symptoms','gender','screening_hpv_status',
            'key_chronic_conditions_summary','medication_adherence_self_report',
            'referral_status','referral_reason','referred_to_facility_id','referral_outcome',
            'sample_status','rejection_reason'
        ]
        common_na_vals_str = ['', 'nan', 'None', 'N/A', '#N/A', 'np.nan', 'NaT', '<NA>']
        for col in str_cols:
            if col in df.columns:
                df[col] = df[col].fillna("Unknown").astype(str).str.strip().replace(common_na_vals_str, "Unknown", regex=False)
            else: df[col] = "Unknown"
        critical_core_cols = {'patient_id': "UnknownPID", 'encounter_date': pd.NaT, 'condition': "NoConditionData"}
        for col, def_val_c in critical_core_cols.items():
            if col not in df.columns or df[col].isnull().all():
                logger.warning(f"({source_context}) Core col '{col}' missing/all null. Filling default: {def_val_c}.")
                df[col] = def_val_c
        logger.info(f"({source_context}) Health records cleaned. Shape: {df.shape}")
        return df
    except Exception as e_load_hr:
        logger.error(f"({source_context}) Load/process health records error: {e_load_hr}", exc_info=True)
        if is_streamlit_active_hr : st.error(f"Failed loading/processing health records: {e_load_hr}")
        return pd.DataFrame()

@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS_WEB_REPORTS, show_spinner="Loading IoT environmental dataset...")
def load_iot_clinic_environment_data(file_path: Optional[str] = None, source_context: str = "FacilityNode") -> pd.DataFrame:
    actual_file_path = file_path or app_config.IOT_CLINIC_ENVIRONMENT_CSV
    logger.info(f"({source_context}) Attempting to load IoT data from: {actual_file_path}")
    is_streamlit_active_iot = "streamlit" in sys.modules and hasattr(st, 'session_state') and hasattr(st.session_state, 'run_count')
    
    if not os.path.exists(actual_file_path): # Corrected syntax from line 169 issue
        logger.warning(f"({source_context}) IoT data file not found: {actual_file_path}")
        if is_streamlit_active_iot: 
            st.info(f"ℹ️ IoT data file '{os.path.basename(actual_file_path)}' missing for {source_context}. Environmental monitoring may be limited.")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(actual_file_path, low_memory=False)
        df = _clean_column_names(df) # Apply cleaning
        logger.info(f"({source_context}) Loaded {len(df)} IoT records from {actual_file_path}.")
        
        if 'timestamp' not in df.columns: 
            logger.error(f"({source_context}) IoT data missing critical 'timestamp' column. Returning empty DataFrame.")
            return pd.DataFrame()
        df['timestamp'] = pd.to_datetime(df.get('timestamp'), errors='coerce')
        
        num_iot_cols_list = ['avg_co2_ppm','max_co2_ppm','avg_pm25','voc_index','avg_temp_celsius','avg_humidity_rh','avg_noise_db','waiting_room_occupancy','patient_throughput_per_hour','sanitizer_dispenses_per_hour']
        for col in num_iot_cols_list:
            if col in df.columns: df[col] = _convert_to_numeric(df.get(col), np.nan)
            else: df[col] = np.nan # Ensure column exists
        
        str_iot_cols_list = ['clinic_id','room_name','zone_id']
        common_na_values_iot = ['', 'nan', 'None', 'N/A', '#N/A', 'np.nan', 'NaT', '<NA>'] # From health records
        for col in str_iot_cols_list:
            if col in df.columns: 
                df[col] = df.get(col,pd.Series(dtype=str)).fillna("Unknown").astype(str).str.strip().replace(common_na_values_iot, "Unknown", regex=False)
            else: df[col] = "Unknown" # Ensure column exists
        
        logger.info(f"({source_context}) IoT data cleaning complete. Shape: {df.shape}")
        return df
    except Exception as e_load_iot: 
        logger.error(f"({source_context}) Error loading/processing IoT data from {actual_file_path}: {e_load_iot}", exc_info=True)
        return pd.DataFrame()

@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS_WEB_REPORTS, hash_funcs={gpd.GeoDataFrame: hash_geodataframe}, show_spinner="Loading zone geographic & attribute dataset...")
def load_zone_data(attributes_path: Optional[str] = None, geometries_path: Optional[str] = None, source_context: str = "FacilityNode") -> Optional[gpd.GeoDataFrame]:
    attr_path = attributes_path or app_config.ZONE_ATTRIBUTES_CSV
    geom_path = geometries_path or app_config.ZONE_GEOMETRIES_GEOJSON
    logger.info(f"({source_context}) Loading zone data: Attrs='{os.path.basename(attr_path)}', Geoms='{os.path.basename(geom_path)}'")
    is_streamlit_active_zone_load = "streamlit" in sys.modules and hasattr(st, 'session_state') and hasattr(st.session_state, 'run_count')
    
    error_msgs_load_zone = []
    if not os.path.exists(attr_path): error_msgs_load_zone.append(f"Attributes file missing: {os.path.basename(attr_path)}")
    if not os.path.exists(geom_path): error_msgs_load_zone.append(f"Geometries file missing: {os.path.basename(geom_path)}")
    if error_msgs_load_zone: 
        full_err_str = "; ".join(error_msgs_load_zone); logger.error(f"({source_context}) {full_err_str}")
        if is_streamlit_active_zone_load : st.error(f"🚨 GIS Data Error: {full_err_str}"); return None
    
    try:
        attrs_df_load = pd.read_csv(attr_path); attrs_df_load = _clean_column_names(attrs_df_load)
        geoms_gdf_load = gpd.read_file(geom_path); geoms_gdf_load = _clean_column_names(geoms_gdf_load)

        for df_check_zone, name_check_zone in [(attrs_df_load, "attributes"), (geoms_gdf_load, "geometries")]:
            if 'zone_id' not in df_check_zone.columns: 
                logger.error(f"({source_context}) 'zone_id' missing in {name_check_zone}. Cannot merge.")
                if is_streamlit_active_zone_load: st.error("🚨 Key 'zone_id' missing in GIS input files.")
                return None
            df_check_zone['zone_id'] = df_check_zone['zone_id'].astype(str).str.strip()

        if 'zone_display_name' in attrs_df_load.columns and 'name' not in attrs_df_load.columns: 
            attrs_df_load.rename(columns={'zone_display_name':'name'}, inplace=True)
        elif 'name' not in attrs_df_load.columns and 'zone_id' in attrs_df_load.columns: 
            attrs_df_load['name'] = "Zone " + attrs_df_load['zone_id'].astype(str)
        
        original_geom_col_name_in_geoms_gdf = geoms_gdf_load.geometry.name if hasattr(geoms_gdf_load, 'geometry') and hasattr(geoms_gdf_load.geometry, 'name') else 'geometry'
        
        mrg_gdf_load = geoms_gdf_load.merge(attrs_df_load, on="zone_id", how="left", suffixes=('_geom_src', '_attr_src'))
        
        for col_name_attr in attrs_df_load.columns:
            if col_name_attr == 'zone_id': continue
            attr_suffixed, geom_suffixed = f"{col_name_attr}_attr_src", f"{col_name_attr}_geom_src"
            if attr_suffixed in mrg_gdf_load.columns:
                mrg_gdf_load[col_name_attr] = mrg_gdf_load[attr_suffixed].fillna(mrg_gdf_load.get(geom_suffixed))
                mrg_gdf_load.drop(columns=[attr_suffixed, geom_suffixed], errors='ignore', inplace=True)
            elif geom_suffixed in mrg_gdf_load.columns and col_name_attr not in mrg_gdf_load.columns : mrg_gdf_load.rename(columns={geom_suffixed:col_name_attr}, inplace=True)
        
        if mrg_gdf_load.geometry.name != 'geometry':
            if 'geometry' in mrg_gdf_load.columns : mrg_gdf_load = mrg_gdf_load.set_geometry('geometry', drop=False, inplace=False)
            elif original_geom_col_name_in_geoms_gdf in mrg_gdf_load.columns : mrg_gdf_load = mrg_gdf_load.rename_geometry('geometry', col_name=original_geom_col_name_in_geoms_gdf)
            else: logger.error(f"({source_context}) No identifiable geometry column in merged GDF ({mrg_gdf_load.columns.tolist()})."); return None
        
        if not hasattr(mrg_gdf_load, 'crs') or mrg_gdf_load.crs is None: mrg_gdf_load = mrg_gdf_load.set_crs(app_config.DEFAULT_CRS_STANDARD, allow_override=True)
        elif str(mrg_gdf_load.crs).upper() != app_config.DEFAULT_CRS_STANDARD.upper(): mrg_gdf_load = mrg_gdf_load.to_crs(app_config.DEFAULT_CRS_STANDARD)
        
        default_zone_attr_cols_map = {'name':"Unknown Zone", 'population':0.0, 'num_clinics':0.0, 'socio_economic_index':0.5, 'avg_travel_time_clinic_min':30.0, 'predominant_hazard_type': "Unknown", 'typical_workforce_exposure_level': "Unknown", 'area_sqkm':np.nan}
        for col_zd_final, def_val_zd_final in default_zone_attr_cols_map.items():
            if col_zd_final not in mrg_gdf_load.columns: 
                mrg_gdf_load[col_zd_final] = def_val_zd_final if col_zd_final !='name' else ("Zone " + mrg_gdf_load['zone_id'].astype(str))
            elif col_zd_final in ['population','socio_economic_index','num_clinics','avg_travel_time_clinic_min','area_sqkm']: 
                mrg_gdf_load[col_zd_final] = _convert_to_numeric(mrg_gdf_load.get(col_zd_final), def_val_zd_final)
            elif col_zd_final == 'name' : 
                mrg_gdf_load[col_zd_final] = mrg_gdf_load.get(col_zd_final,"Unknown").astype(str).fillna("Zone "+mrg_gdf_load['zone_id'].astype(str))
        logger.info(f"({source_context}) Zone data loaded/merged. Shape: {mrg_gdf_load.shape}. CRS: {mrg_gdf_load.crs}")
        return mrg_gdf_load
    except ImportError as e_gpd_load:
        logger.critical(f"({source_context}) GeoPandas import error during zone data load: {e_gpd_load}. Ensure GeoPandas and its C dependencies (GDAL, GEOS, PROJ) are correctly installed in the environment.", exc_info=True)
        if is_streamlit_active_zone_load: st.error(f"🚨 CRITICAL GIS LIBRARY ERROR: GeoPandas (or its dependencies) not found or failed to load. Mapping and analytics features requiring geospatial data will be unavailable. Details: {e_gpd_load}")
        return None
    except Exception as e_load_zone_final: 
        logger.error(f"({source_context}) General error loading/merging zone data: {e_load_zone_final}", exc_info=True); 
        if is_streamlit_active_zone_load : st.error(f"GIS data processing error: {e_load_zone_final}"); 
        return None


# --- III. Data Enrichment Function ---
def enrich_zone_geodata_with_health_aggregates(
    zone_gdf: gpd.GeoDataFrame, health_df: Optional[pd.DataFrame], iot_df: Optional[pd.DataFrame] = None,
    source_context: str = "FacilityNode/ZoneEnrich"
) -> gpd.GeoDataFrame:
    logger.info(f"({source_context}) Starting zone GeoDataFrame enrichment.")
    if not isinstance(zone_gdf, gpd.GeoDataFrame) or zone_gdf.empty or 'zone_id' not in zone_gdf.columns:
        logger.warning(f"({source_context}) Invalid or empty zone_gdf for enrichment. Returning input or minimal GDF.")
        return zone_gdf if isinstance(zone_gdf, gpd.GeoDataFrame) else gpd.GeoDataFrame(columns=['zone_id', 'name', 'geometry', 'population'], crs=app_config.DEFAULT_CRS_STANDARD)

    enriched = zone_gdf.copy()
    if 'population' not in enriched.columns: enriched['population'] = 0.0
    enriched['population'] = _convert_to_numeric(enriched['population'], 0.0)

    agg_cols_to_init_list = [
        'total_population_health_data', 'avg_risk_score', 'total_patient_encounters',
        'total_referrals_made', 'successful_referrals', 'avg_test_turnaround_critical', 
        'perc_critical_tests_tat_met', 'prevalence_per_1000', 'total_active_key_infections',
        'avg_daily_steps_zone', 'zone_avg_co2', 'facility_coverage_score', 'population_density'
    ]
    for cond_key_enrich in app_config.KEY_CONDITIONS_FOR_ACTION:
        agg_cols_to_init_list.append(f"active_{cond_key_enrich.lower().replace(' ', '_').replace('-', '_').replace('(severe)','')}_cases")
    
    for col_init in agg_cols_to_init_list:
        default_for_init = np.nan if any(k_nan in col_init for k_nan in ['avg_', 'perc_', 'score', 'density']) else 0.0
        if col_init not in enriched.columns: enriched[col_init] = default_for_init
        else: enriched[col_init] = _convert_to_numeric(enriched[col_init], default_for_init)

    if health_df is not None and not health_df.empty and 'zone_id' in health_df.columns:
        hdf_enrich = health_df.copy(); hdf_enrich['zone_id'] = hdf_enrich['zone_id'].astype(str).str.strip()
        enriched = _robust_merge_agg(enriched, hdf_enrich.groupby('zone_id')['patient_id'].nunique().reset_index(name="val_col"), 'total_population_health_data')
        enriched = _robust_merge_agg(enriched, hdf_enrich.groupby('zone_id')['ai_risk_score'].mean().reset_index(name="val_col"), 'avg_risk_score', default_fill_value=np.nan)
        enriched = _robust_merge_agg(enriched, hdf_enrich.groupby('zone_id')['encounter_id'].nunique().reset_index(name="val_col"), 'total_patient_encounters')
        
        for cond_name_enrich in app_config.KEY_CONDITIONS_FOR_ACTION:
            col_name_dyn = f"active_{cond_name_enrich.lower().replace(' ', '_').replace('-', '_').replace('(severe)','')}_cases"
            cond_filt_enrich = hdf_enrich.get('condition', pd.Series(dtype=str)).str.contains(cond_name_enrich, case=False, na=False)
            enriched = _robust_merge_agg(enriched, hdf_enrich[cond_filt_enrich].groupby('zone_id')['patient_id'].nunique().reset_index(name="val_col"), col_name_dyn)
        
        actionable_cols_sum = [col for col in enriched.columns if col.startswith("active_") and col.endswith("_cases") and any(action_cond.lower().replace(' ', '_').replace('-', '_').replace('(severe)','') in col for action_cond in app_config.KEY_CONDITIONS_FOR_ACTION)]
        if actionable_cols_sum: enriched['total_active_key_infections'] = enriched[actionable_cols_sum].sum(axis=1, skipna=True)

        if 'referral_status' in hdf_enrich.columns:
            made_refs_df = hdf_enrich[hdf_enrich['referral_status'].notna() & (~hdf_enrich['referral_status'].str.lower().isin(['n/a', 'unknown']))]
            enriched = _robust_merge_agg(enriched, made_refs_df.groupby('zone_id')['encounter_id'].nunique().reset_index(name="val_col"), 'total_referrals_made')
            if 'referral_outcome' in hdf_enrich.columns:
                succ_outcomes = ['completed', 'service provided', 'attended consult', 'attended followup', 'attended']
                succ_refs_df = hdf_enrich[hdf_enrich.get('referral_outcome',pd.Series(dtype=str)).str.lower().isin(succ_outcomes)]
                enriched = _robust_merge_agg(enriched, succ_refs_df.groupby('zone_id')['encounter_id'].nunique().reset_index(name="val_col"), 'successful_referrals')

        crit_tests_list_enr = app_config.CRITICAL_TESTS_LIST
        if crit_tests_list_enr and 'test_type' in hdf_enrich.columns and 'test_turnaround_days' in hdf_enrich.columns:
            tat_enrich_df = hdf_enrich[(hdf_enrich['test_type'].isin(crit_tests_list_enr)) & (hdf_enrich['test_turnaround_days'].notna()) & (~hdf_enrich.get('test_result',pd.Series(dtype=str)).str.lower().isin(['pending', 'rejected sample', 'unknown', 'indeterminate']))].copy()
            if not tat_enrich_df.empty:
                enriched = _robust_merge_agg(enriched, tat_enrich_df.groupby('zone_id')['test_turnaround_days'].mean().reset_index(name="val_col"), 'avg_test_turnaround_critical', default_fill_value=np.nan)
                def _check_tat_met_core_enr_final(row_c_enr): 
                    cfg_c_enr = app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get(str(row_c_enr.get('test_type','')))
                    tgt_days_enr = cfg_c_enr.get('target_tat_days') if cfg_c_enr and isinstance(cfg_c_enr,dict) else app_config.TARGET_TEST_TURNAROUND_DAYS
                    return pd.notna(row_c_enr['test_turnaround_days']) and pd.notna(tgt_days_enr) and row_c_enr['test_turnaround_days'] <= tgt_days_enr
                tat_enrich_df['tat_met_flag_core_enr'] = tat_enrich_df.apply(_check_tat_met_core_enr_final, axis=1)
                perc_met_agg_core_enr = tat_enrich_df.groupby('zone_id')['tat_met_flag_core_enr'].mean().reset_index(name="val_col"); perc_met_agg_core_enr["val_col"] *= 100
                enriched = _robust_merge_agg(enriched, perc_met_agg_core_enr, 'perc_critical_tests_tat_met', default_fill_value=np.nan) # Perc can be NaN
        
        if 'avg_daily_steps' in hdf_enrich.columns: enriched = _robust_merge_agg(enriched, hdf_enrich.groupby('zone_id')['avg_daily_steps'].mean().reset_index(name="val_col"), 'avg_daily_steps_zone', default_fill_value=np.nan)

    if iot_df is not None and not iot_df.empty and all(c in iot_df.columns for c in ['zone_id','avg_co2_ppm']):
        iot_enrich_df_agg = iot_df.copy(); iot_enrich_df_agg['zone_id'] = iot_enrich_df_agg['zone_id'].astype(str).str.strip()
        enriched = _robust_merge_agg(enriched, iot_enrich_df_agg.groupby('zone_id')['avg_co2_ppm'].mean().reset_index(name="val_col"), 'zone_avg_co2', default_fill_value=np.nan)
    
    if 'total_active_key_infections' in enriched.columns and 'population' in enriched.columns:
         enriched['prevalence_per_1000'] = enriched.apply(lambda r: (r.get('total_active_key_infections',0.0)/r.get('population',1.0)) * 1000.0 if pd.notna(r.get('population')) and r.get('population',0.0)>0 else 0.0, axis=1).fillna(0.0)
    if 'num_clinics' in enriched.columns and 'population' in enriched.columns:
        enriched['facility_coverage_score'] = enriched.apply(lambda r: min(100.0, (r.get('num_clinics',0.0)/r.get('population',1.0))*20000.0) if pd.notna(r.get('population')) and r.get('population',0.0)>0 and pd.notna(r.get('num_clinics')) else 0.0, axis=1).fillna(0.0)
    elif 'facility_coverage_score' not in enriched.columns: enriched['facility_coverage_score']=0.0

    if 'geometry' in enriched.columns and hasattr(enriched, 'crs') and enriched.crs and 'population' in enriched.columns:
        if 'area_sqkm' in enriched.columns and enriched['area_sqkm'].notna().any() and enriched['area_sqkm'].gt(0).any() : # Area must be pre-calculated or passed in attributes
            valid_area_mask_enr_final = enriched['area_sqkm'].notna() & enriched['area_sqkm'].gt(0)
            enriched.loc[valid_area_mask_enr_final, 'population_density'] = (enriched.loc[valid_area_mask_enr_final,'population'] / enriched.loc[valid_area_mask_enr_final,'area_sqkm'])
            enriched['population_density'].fillna(0.0, inplace=True)
        elif 'population_density' not in enriched.columns : 
            enriched['population_density'] = np.nan
            logger.info(f"({source_context}) Population density: 'area_sqkm' column missing/invalid or GDF needs projected CRS for area calculation to be accurate.")
    
    for col_final_enr_fill in agg_cols_to_init_list:
        default_final_val = np.nan if any(k_nan in col_final_enr_fill for k_nan in ['avg_', 'perc_', 'score', 'density']) else 0.0
        if col_final_enr_fill in enriched.columns: enriched[col_final_enr_fill] = pd.to_numeric(enriched.get(col_final_enr_fill, default_final_val), errors='coerce').fillna(default_final_val)
        else: enriched[col_final_enr_fill] = default_final_val
    
    logger.info(f"({source_context}) Zone GDF enrichment complete. Shape: {enriched.shape}. Output columns: {enriched.columns.tolist()}")
    return enriched


# --- IV. KPI & Summary Calculation Functions (Full Implementations) ---

def get_overall_kpis(health_df: pd.DataFrame, date_filter_start: Optional[str]=None, date_filter_end: Optional[str]=None, source_context: str = "OverallKPIs") -> Dict[str, Any]:
    logger.info(f"({source_context}) Calculating overall KPIs for period: {date_filter_start or 'all'} to {date_filter_end or 'all'}.")
    kpis_dict: Dict[str, Any] = {
        "total_patients": 0, "avg_patient_risk": np.nan,
        "malaria_rdt_positive_rate_period": np.nan, "hiv_rapid_positive_rate_period": np.nan,
        "key_supply_stockout_alerts": 0
    }
    for cond_kpi_key_gen in app_config.KEY_CONDITIONS_FOR_ACTION:
        kpis_dict[f"active_{cond_kpi_key_gen.lower().replace(' ', '_').replace('-', '_').replace('(severe)','')}_cases_current"] = 0

    if health_df is None or health_df.empty: logger.warning(f"({source_context}) No health data provided."); return kpis_dict
    df_kpi_calc = health_df.copy()
    if 'encounter_date' not in df_kpi_calc.columns or df_kpi_calc['encounter_date'].isnull().all():
        logger.warning(f"({source_context}) 'encounter_date' missing for overall KPIs or all null."); return kpis_dict
    
    df_kpi_calc['encounter_date'] = pd.to_datetime(df_kpi_calc['encounter_date'], errors='coerce')
    df_kpi_calc.dropna(subset=['encounter_date'], inplace=True)
    
    start_dt_kpi = pd.to_datetime(date_filter_start, errors='coerce') if date_filter_start else None
    end_dt_kpi = pd.to_datetime(date_filter_end, errors='coerce') if date_filter_end else None
    if start_dt_kpi: df_kpi_calc = df_kpi_calc[df_kpi_calc['encounter_date'] >= start_dt_kpi]
    if end_dt_kpi: df_kpi_calc = df_kpi_calc[df_kpi_calc['encounter_date'] <= end_dt_kpi]
    if df_kpi_calc.empty: logger.info(f"({source_context}) No data after date filtering for overall KPIs."); return kpis_dict

    if 'patient_id' in df_kpi_calc: kpis_dict["total_patients"] = df_kpi_calc['patient_id'].nunique()
    if 'ai_risk_score' in df_kpi_calc and df_kpi_calc['ai_risk_score'].notna().any(): kpis_dict["avg_patient_risk"] = df_kpi_calc['ai_risk_score'].mean()
    else: kpis_dict["avg_patient_risk"] = np.nan # Explicitly NaN if no scores

    if 'condition' in df_kpi_calc.columns:
        for cond_key_kpi in app_config.KEY_CONDITIONS_FOR_ACTION:
            kpi_col_name_kpi = f"active_{cond_key_kpi.lower().replace(' ', '_').replace('-', '_').replace('(severe)','')}_cases_current"
            kpis_dict[kpi_col_name_kpi] = df_kpi_calc[df_kpi_calc['condition'].str.contains(cond_key_kpi, case=False, na=False)]['patient_id'].nunique()

    for test_key_from_config, kpi_name_for_output in [
        ("RDT-Malaria", "malaria_rdt_positive_rate_period"), 
        ("HIV-Rapid", "hiv_rapid_positive_rate_period")
    ]:
        if test_key_from_config in app_config.KEY_TEST_TYPES_FOR_ANALYSIS and \
           'test_type' in df_kpi_calc.columns and 'test_result' in df_kpi_calc.columns:
            test_df_analysis = df_kpi_calc[
                (df_kpi_calc['test_type'] == test_key_from_config) & 
                (~df_kpi_calc.get('test_result', pd.Series(dtype=str)).astype(str).str.lower().isin(["pending", "rejected sample", "unknown", "indeterminate", "n/a", ""]))
            ]
            if not test_df_analysis.empty and len(test_df_analysis) > 0:
                kpis_dict[kpi_name_for_output] = (test_df_analysis[test_df_analysis['test_result'].astype(str).str.lower() == 'positive'].shape[0] / len(test_df_analysis)) * 100
            else: kpis_dict[kpi_name_for_output] = 0.0 # No conclusive tests of this type implies 0% rate from actuals
        else: kpis_dict[kpi_name_for_output] = np.nan # Test type not in data or columns missing

    if all(c in df_kpi_calc for c in ['item', 'item_stock_agg_zone', 'consumption_rate_per_day', 'encounter_date']) and app_config.KEY_DRUG_SUBSTRINGS_SUPPLY:
        supply_df_for_kpi = df_kpi_calc.sort_values('encounter_date').drop_duplicates(subset=['item', 'zone_id'], keep='last') # Get latest stock per item/zone
        supply_df_for_kpi['consumption_rate_per_day'] = supply_df_for_kpi['consumption_rate_per_day'].replace(0, np.nan)
        supply_df_for_kpi['days_supply_kpi_calc'] = supply_df_for_kpi['item_stock_agg_zone'] / supply_df_for_kpi['consumption_rate_per_day']
        supply_df_for_kpi.dropna(subset=['days_supply_kpi_calc'], inplace=True)
        key_drug_supply_kpi_df = supply_df_for_kpi[supply_df_for_kpi['item'].str.contains('|'.join(app_config.KEY_DRUG_SUBSTRINGS_SUPPLY), case=False, na=False)]
        kpis_dict['key_supply_stockout_alerts'] = key_drug_supply_kpi_df[key_drug_supply_kpi_df['days_supply_kpi_calc'] < app_config.CRITICAL_SUPPLY_DAYS_REMAINING]['item'].nunique()
    return kpis_dict

def get_chw_summary(health_df_daily: pd.DataFrame, source_context: str = "CHWSummary") -> Dict[str, Any]:
    logger.info(f"({source_context}) Calculating CHW daily summary.")
    summary_chw: Dict[str, Any] = {
        "visits_today":0, "avg_patient_risk_visited_today":np.nan,
        "high_ai_prio_followups_today": 0, "patients_critical_spo2_today": 0,
        "patients_high_fever_today": 0, "avg_patient_steps_visited_today":np.nan,
        "patients_fall_detected_today":0, "pending_critical_condition_referrals": 0
    }
    if health_df_daily is None or health_df_daily.empty: logger.warning(f"({source_context}) No daily data provided for CHW summary."); return summary_chw
    
    chw_df_s = health_df_daily.copy() # Work on a copy
    if 'patient_id' in chw_df_s and chw_df_s['patient_id'].notna().any():
        summary_chw["visits_today"]=chw_df_s['patient_id'].nunique()
        # Metrics based on unique patients for averages, but total encounters for event counts
        unique_patients_df_chw = chw_df_s.drop_duplicates(subset=['patient_id'], keep='first')

        if 'ai_risk_score' in unique_patients_df_chw.columns and unique_patients_df_chw['ai_risk_score'].notna().any():
            summary_chw["avg_patient_risk_visited_today"]=unique_patients_df_chw['ai_risk_score'].mean()
        
        if 'ai_followup_priority_score' in chw_df_s.columns and chw_df_s['ai_followup_priority_score'].notna().any():
            summary_chw["high_ai_prio_followups_today"] = chw_df_s[chw_df_s['ai_followup_priority_score'] >= app_config.FATIGUE_INDEX_HIGH_THRESHOLD]['patient_id'].nunique()
        
        if 'min_spo2_pct' in chw_df_s.columns and chw_df_s['min_spo2_pct'].notna().any():
            summary_chw["patients_critical_spo2_today"]=chw_df_s[chw_df_s['min_spo2_pct'] < app_config.ALERT_SPO2_CRITICAL_LOW_PCT]['patient_id'].nunique()
        
        temp_col_chw_sum = next((tc for tc in ['vital_signs_temperature_celsius','max_skin_temp_celsius'] if tc in chw_df_s and chw_df_s[tc].notna().any()), None)
        if temp_col_chw_sum:
            summary_chw["patients_high_fever_today"]=chw_df_s[chw_df_s[temp_col_chw_sum] >= app_config.ALERT_BODY_TEMP_HIGH_FEVER_C]['patient_id'].nunique()
        
        if 'avg_daily_steps' in unique_patients_df_chw.columns and unique_patients_df_chw['avg_daily_steps'].notna().any():
            summary_chw["avg_patient_steps_visited_today"]=unique_patients_df_chw['avg_daily_steps'].mean()
        
        if 'fall_detected_today' in chw_df_s.columns and chw_df_s['fall_detected_today'].notna().any():
            summary_chw["patients_fall_detected_today"]=chw_df_s[chw_df_s['fall_detected_today'] > 0]['patient_id'].nunique()

    if all(c in chw_df_s.columns for c in ['condition', 'referral_status', 'referral_reason']):
        crit_cond_set_chw_s = set(app_config.KEY_CONDITIONS_FOR_ACTION)
        urgent_keywords_s = ['urgent', 'emergency', 'critical', 'severe', 'immediate']
        
        chw_df_s_copy_for_lambda = chw_df_s.copy() # Use a copy for lambda to avoid SettingWithCopy
        chw_df_s_copy_for_lambda['is_crit_ref_chw_s'] = chw_df_s_copy_for_lambda.apply(
            lambda r_s_lambda: (str(r_s_lambda.get('referral_status','Unknown')).lower() == 'pending' and 
                       (any(ck_s_lambda.lower() in str(r_s_lambda.get('condition','')).lower() for ck_s_lambda in crit_cond_set_chw_s) or
                        any(uk_s_lambda.lower() in str(r_s_lambda.get('referral_reason','')).lower() for uk_s_lambda in urgent_keywords_s))
                      ), axis=1
        )
        summary_chw["pending_critical_condition_referrals"] = chw_df_s_copy_for_lambda[chw_df_s_copy_for_lambda['is_crit_ref_chw_s']]['patient_id'].nunique()
    return summary_chw

def get_patient_alerts_for_chw(health_df_daily: pd.DataFrame, source_context: str = "CHWAlertGenFull", risk_threshold_moderate: int = app_config.RISK_SCORE_MODERATE_THRESHOLD, risk_threshold_high: int = app_config.RISK_SCORE_HIGH_THRESHOLD) -> List[Dict[str, Any]]:
    logger.info(f"({source_context}) Generating CHW patient alerts list from daily data (rows: {len(health_df_daily) if health_df_daily is not None else 0}).")
    if health_df_daily is None or health_df_daily.empty: return []
    
    df_alerts_src_chw = health_df_daily.copy()
    processed_alerts_final_chw: List[Dict[str, Any]] = []
    
    cols_to_ensure_chw_alert_final = {
        'patient_id': "UnknownPatient", 'encounter_date': pd.NaT, 'condition': "N/A", 'zone_id': "UnknownZone",
        'age': np.nan, 'ai_risk_score': np.nan, 'ai_followup_priority_score': np.nan,
        'min_spo2_pct': 100.0, 'vital_signs_temperature_celsius': 37.0, 'max_skin_temp_celsius': 37.0,
        'fall_detected_today': 0, 'referral_status': "N/A", 'referral_reason': "N/A",
        'medication_adherence_self_report': "Unknown"
    }
    for col_chw_a, def_chw_a in cols_to_ensure_chw_alert_final.items():
        if col_chw_a not in df_alerts_src_chw.columns: df_alerts_src_chw[col_chw_a] = def_chw_a
        elif 'date' in col_chw_a: df_alerts_src_chw[col_chw_a] = pd.to_datetime(df_alerts_src_chw[col_chw_a], errors='coerce')
        elif isinstance(def_chw_a, (float, int)): df_alerts_src_chw[col_chw_a] = pd.to_numeric(df_alerts_src_chw[col_chw_a], errors='coerce').fillna(def_chw_a if pd.notna(def_chw_a) else np.nan)
        elif isinstance(def_chw_a, str): df_alerts_src_chw[col_chw_a] = df_alerts_src_chw[col_chw_a].astype(str).fillna(def_chw_a)


    temp_col_chw_alerts = next((tc_chw for tc_chw in ['vital_signs_temperature_celsius', 'max_skin_temp_celsius'] if tc_chw in df_alerts_src_chw and df_alerts_src_chw[tc_chw].notna().any()), None)

    for _, record_chw in df_alerts_src_chw.iterrows():
        record_alerts_list_chw: List[Dict[str, Any]] = []
        pat_id_chw_a, cond_chw_a, zone_chw_a = str(record_chw.get('patient_id')), str(record_chw.get('condition')), str(record_chw.get('zone_id'))
        enc_date_chw_a_obj = pd.to_datetime(record_chw.get('encounter_date'))
        enc_date_chw_a = enc_date_chw_a_obj.strftime('%Y-%m-%d') if pd.notna(enc_date_chw_a_obj) else 'N/A'
        ctx_chw_a = f"Cond: {cond_chw_a} | Zone: {zone_chw_a} | Date: {enc_date_chw_a}"
        
        def add_chw_alert_to_list(level, reason, details, action_code, score, trigger_val_str):
            record_alerts_list_chw.append({"patient_id":pat_id_chw_a, "encounter_date":enc_date_chw_a_obj, "alert_level":level, "primary_reason":reason, "brief_details":details, "suggested_action_code":action_code, "raw_priority_score":score, "context_info":ctx_chw_a, "triggering_value":trigger_val_str})

        spo2_chw = record_chw.get('min_spo2_pct')
        if pd.notna(spo2_chw):
            if spo2_chw < app_config.ALERT_SPO2_CRITICAL_LOW_PCT: add_chw_alert_to_list("CRITICAL", "Crit. Low SpO2", f"SpO2: {spo2_chw:.0f}%", "ACT_SPO2_URGENT", 98+(app_config.ALERT_SPO2_CRITICAL_LOW_PCT-spo2_chw), f"SpO2 {spo2_chw:.0f}%")
            elif spo2_chw < app_config.ALERT_SPO2_WARNING_LOW_PCT: add_chw_alert_to_list("WARNING", "Low SpO2", f"SpO2: {spo2_chw:.0f}%", "ACT_SPO2_MONITOR", 75+(app_config.ALERT_SPO2_WARNING_LOW_PCT-spo2_chw), f"SpO2 {spo2_chw:.0f}%")
        
        temp_val_chw_a = record_chw.get(temp_col_chw_alerts) if temp_col_chw_alerts else np.nan
        if pd.notna(temp_val_chw_a):
            if temp_val_chw_a >= app_config.ALERT_BODY_TEMP_HIGH_FEVER_C: add_chw_alert_to_list("CRITICAL", "High Fever", f"Temp: {temp_val_chw_a:.1f}°C", "ACT_FEVER_URGENT", 95+(temp_val_chw_a-app_config.ALERT_BODY_TEMP_HIGH_FEVER_C)*2, f"Temp {temp_val_chw_a:.1f}°C")
            elif temp_val_chw_a >= app_config.ALERT_BODY_TEMP_FEVER_C: add_chw_alert_to_list("WARNING", "Fever", f"Temp: {temp_val_chw_a:.1f}°C", "ACT_FEVER_MONITOR", 70+(temp_val_chw_a-app_config.ALERT_BODY_TEMP_FEVER_C), f"Temp {temp_val_chw_a:.1f}°C")
        
        if record_chw.get('fall_detected_today',0) > 0: add_chw_alert_to_list("CRITICAL", "Fall Detected", f"Falls: {int(record_chw.get('fall_detected_today',0))}", "ACT_FALL_ASSESS", 92, "Fall > 0")
        
        ai_prio_chw, ai_risk_chw = record_chw.get('ai_followup_priority_score'), record_chw.get('ai_risk_score')
        if pd.notna(ai_prio_chw) and ai_prio_chw >= risk_threshold_high: add_chw_alert_to_list("WARNING", "High AI Prio.", f"AI Prio: {ai_prio_chw:.0f}", "ACT_REVIEW_AI_PRIO", ai_prio_chw, f"AI Prio {ai_prio_chw:.0f}")
        elif pd.notna(ai_risk_chw) and ai_risk_chw >= risk_threshold_high and not (pd.notna(ai_prio_chw) and ai_prio_chw >= risk_threshold_high): add_chw_alert_to_list("INFO", "High AI Risk", f"AI Risk: {ai_risk_chw:.0f}", "ACT_MONITOR_AI_RISK", ai_risk_chw, f"AI Risk {ai_risk_chw:.0f}")
        elif pd.notna(ai_prio_chw) and ai_prio_chw >= risk_threshold_moderate : add_chw_alert_to_list("INFO", "Mod. AI Prio.", f"AI Prio: {ai_prio_chw:.0f}", "ACT_REVIEW_AI_PRIO", ai_prio_chw, f"AI Prio {ai_prio_chw:.0f}")
        
        if str(record_chw.get('referral_status', '')).lower() == 'pending' and any(ck_chw_a.lower() in str(record_chw.get('condition','')).lower() for ck_chw_a in app_config.KEY_CONDITIONS_FOR_ACTION):
            add_chw_alert_to_list("WARNING", "Pending Critical Ref.", f"For: {record_chw.get('condition')}", "ACT_FOLLOWUP_REF", 80, "PendCritRef")
        
        if record_alerts_list_chw: processed_alerts_final_chw.append(max(record_alerts_list_chw, key=lambda x_alert: x_alert['raw_priority_score']))
    
    if not processed_alerts_final_chw: return []
    alerts_chw_final_df = pd.DataFrame(processed_alerts_final_chw)
    if not alerts_chw_final_df.empty :
        alerts_chw_final_df['encounter_date_obj_for_dedup'] = pd.to_datetime(alerts_chw_final_df['encounter_date']).dt.date
        alerts_chw_final_df.sort_values(by=["raw_priority_score","alert_level"], ascending=[False,True], inplace=True) # Sort to keep highest prio
        alerts_chw_final_df.drop_duplicates(subset=["patient_id", "encounter_date_obj_for_dedup", "primary_reason"], keep="first", inplace=True) # More specific dedupe
        alerts_chw_final_df.drop(columns=['encounter_date_obj_for_dedup'], inplace=True, errors='ignore')
        return alerts_chw_final_df.to_dict(orient='records')
    return []


def get_clinic_summary(health_df_period: pd.DataFrame, source_context: str = "ClinicSummaryFull") -> Dict[str, Any]:
    logger.info(f"({source_context}) Calculating full clinic summary.")
    summary: Dict[str, Any] = {
        "overall_avg_test_turnaround_conclusive_days": np.nan, "perc_critical_tests_tat_met": 0.0,
        "total_pending_critical_tests_patients": 0, "sample_rejection_rate_perc": 0.0,
        "key_drug_stockouts_count": 0, "test_summary_details": {}
    }
    if health_df_period is None or health_df_period.empty: logger.warning(f"({source_context}) No data for clinic summary."); return summary
    df_cs_full = health_df_period.copy()
    # Ensure all necessary columns for string operations exist and are strings
    for col_cs_str_full in ['test_type','test_result','sample_status','item','zone_id', 'condition', 'referral_status', 'referral_outcome', 'rejection_reason', 'patient_id', 'encounter_id']: 
        df_cs_full[col_cs_str_full] = df_cs_full.get(col_cs_str_full, pd.Series(dtype=str)).fillna("Unknown").astype(str)
    df_cs_full['encounter_date'] = pd.to_datetime(df_cs_full.get('encounter_date'), errors='coerce')
    # Ensure numeric columns are numeric, fill NaNs appropriately for calculations
    for num_col_cs, def_cs_val in {'test_turnaround_days': np.nan, 'item_stock_agg_zone': 0.0, 'consumption_rate_per_day': 0.0001}.items():
        df_cs_full[num_col_cs] = _convert_to_numeric(df_cs_full.get(num_col_cs), def_cs_val)
    
    concl_df_cs_full = df_cs_full[~df_cs_full['test_result'].str.lower().isin(['pending','rejected sample','unknown','n/a','nan','indeterminate','']) & df_cs_full['test_turnaround_days'].notna()].copy()
    if not concl_df_cs_full.empty and concl_df_cs_full['test_turnaround_days'].notna().any(): summary["overall_avg_test_turnaround_conclusive_days"] = concl_df_cs_full['test_turnaround_days'].mean()
    
    crit_keys_cs_full = app_config.CRITICAL_TESTS_LIST
    if crit_keys_cs_full: # Only if critical tests are defined
        crit_concl_df_cs_full = concl_df_cs_full[concl_df_cs_full['test_type'].isin(crit_keys_cs_full)].copy()
        if not crit_concl_df_cs_full.empty:
            def _chk_tat_cs_fn_full(r_cs_f): 
                cfg_s_f = app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get(str(r_cs_f.get('test_type','')),{})
                tgt_days_f = cfg_s_f.get('target_tat_days', app_config.TARGET_TEST_TURNAROUND_DAYS)
                return pd.notna(r_cs_f['test_turnaround_days']) and pd.notna(tgt_days_f) and r_cs_f['test_turnaround_days'] <= tgt_days_f
            crit_concl_df_cs_full['tat_met_cs_full'] = crit_concl_df_cs_full.apply(_chk_tat_cs_fn_full,axis=1)
            if not crit_concl_df_cs_full['tat_met_cs_full'].empty : summary["perc_critical_tests_tat_met"]=(crit_concl_df_cs_full['tat_met_cs_full'].mean()*100)
        summary["total_pending_critical_tests_patients"] = df_cs_full[(df_cs_full['test_type'].isin(crit_keys_cs_full)) & (df_cs_full['test_result'].str.lower()=='pending')]['patient_id'].nunique()

    all_proc_samp_cs_full = df_cs_full[~df_cs_full['sample_status'].str.lower().isin(['pending','unknown','n/a','nan',''])].copy()
    if not all_proc_samp_cs_full.empty: summary["sample_rejection_rate_perc"] = (all_proc_samp_cs_full[all_proc_samp_cs_full['sample_status'].str.lower()=='rejected'].shape[0]/len(all_proc_samp_cs_full))*100 if len(all_proc_samp_cs_full) >0 else 0.0
    
    test_sum_details_cs_full={} # Detailed stats per test type
    for o_key_cs_f,cfg_p_cs_f in app_config.KEY_TEST_TYPES_FOR_ANALYSIS.items():
        d_name_cs_f = cfg_p_cs_f.get("display_name",o_key_cs_f)
        actual_keys_cs_f_list = cfg_p_cs_f.get("types_in_group",[o_key_cs_f]) if isinstance(cfg_p_cs_f.get("types_in_group"),list) else [cfg_p_cs_f.get("types_in_group",o_key_cs_f)]
        grp_df_cs_f = df_cs_full[df_cs_full['test_type'].isin(actual_keys_cs_f_list)]
        stats_cs_f = {"positive_rate_perc":np.nan,"avg_tat_days":np.nan,"perc_met_tat_target":0.0,"pending_count_patients":0,"rejected_count_patients":0,"total_conclusive_tests":0}
        if grp_df_cs_f.empty: test_sum_details_cs_full[d_name_cs_f]=stats_cs_f; continue
        grp_c_sum_cs_f = grp_df_cs_f[~grp_df_cs_f['test_result'].str.lower().isin(['pending','rejected sample','unknown','n/a','nan','indeterminate','']) & grp_df_cs_f['test_turnaround_days'].notna()].copy()
        stats_cs_f["total_conclusive_tests"]=len(grp_c_sum_cs_f)
        if not grp_c_sum_cs_f.empty:
            stats_cs_f["positive_rate_perc"]=(grp_c_sum_cs_f[grp_c_sum_cs_f['test_result'].str.lower()=='positive'].shape[0]/len(grp_c_sum_cs_f))*100 if len(grp_c_sum_cs_f)>0 else 0.0
            if grp_c_sum_cs_f['test_turnaround_days'].notna().any():stats_cs_f["avg_tat_days"]=grp_c_sum_cs_f['test_turnaround_days'].mean()
            tgt_tat_spec_cs_f=cfg_p_cs_f.get("target_tat_days",app_config.TARGET_TEST_TURNAROUND_DAYS)
            grp_c_sum_cs_f.loc[:,'tat_met_s_cs_f']=grp_c_sum_cs_f['test_turnaround_days']<=tgt_tat_spec_cs_f if pd.notna(tgt_tat_spec_cs_f) else False
            if not grp_c_sum_cs_f['tat_met_s_cs_f'].empty: stats_cs_f["perc_met_tat_target"]=grp_c_sum_cs_f['tat_met_s_cs_f'].mean()*100
        stats_cs_f["pending_count_patients"]=grp_df_cs_f[grp_df_cs_f['test_result'].str.lower()=='pending']['patient_id'].nunique()
        stats_cs_f["rejected_count_patients"]=grp_df_cs_f[grp_df_cs_f['sample_status'].str.lower()=='rejected']['patient_id'].nunique()
        test_sum_details_cs_full[d_name_cs_f]=stats_cs_f
    summary["test_summary_details"]=test_sum_details_cs_full

    if all(c_supply in df_cs_full.columns for c_supply in ['item','item_stock_agg_zone','consumption_rate_per_day','encounter_date']) and app_config.KEY_DRUG_SUBSTRINGS_SUPPLY:
        supply_df_sum_cs = df_cs_full[df_cs_full['item'].str.contains('|'.join(app_config.KEY_DRUG_SUBSTRINGS_SUPPLY), case=False, na=False)].copy()
        if not supply_df_sum_cs.empty:
            supply_df_sum_cs['encounter_date'] = pd.to_datetime(supply_df_sum_cs['encounter_date'], errors='coerce') # Ensure datetime again just before use
            supply_df_sum_cs.dropna(subset=['encounter_date'], inplace=True)
            if not supply_df_sum_cs.empty:
                latest_supply_cs = supply_df_sum_cs.sort_values('encounter_date').drop_duplicates(subset=['item', 'zone_id'], keep='last') # Consider zone for stock location
                latest_supply_cs['consumption_rate_per_day'] = latest_supply_cs['consumption_rate_per_day'].replace(0, np.nan) # Avoid DivByZero
                latest_supply_cs['days_of_supply_calc_val'] = latest_supply_cs['item_stock_agg_zone'] / latest_supply_cs['consumption_rate_per_day']
                summary['key_drug_stockouts_count'] = latest_supply_cs[latest_supply_cs['days_of_supply_calc_val'] < app_config.CRITICAL_SUPPLY_DAYS_REMAINING]['item'].nunique()
    return summary


def get_clinic_environmental_summary(iot_df_period: pd.DataFrame, source_context: str = "ClinicEnvSummaryFull") -> Dict[str, Any]:
    logger.info(f"({source_context}) Calculating full clinic environmental summary.")
    summary_env_full: Dict[str, Any] = {"avg_co2_overall_ppm":np.nan, "rooms_co2_very_high_alert_latest_count":0, "avg_pm25_overall_ugm3":np.nan, "rooms_pm25_very_high_alert_latest_count":0, "avg_waiting_room_occupancy_persons":np.nan, "waiting_room_high_occupancy_alert_latest_flag":False, "avg_noise_overall_dba":np.nan, "rooms_noise_high_alert_latest_count":0}
    if iot_df_period is None or iot_df_period.empty or 'timestamp' not in iot_df_period.columns: logger.warning(f"({source_context}) No valid IoT data for env summary."); return summary_env_full
    df_iot_sum_full = iot_df_period.copy()
    df_iot_sum_full['timestamp'] = pd.to_datetime(df_iot_sum_full['timestamp'], errors='coerce')
    df_iot_sum_full.dropna(subset=['timestamp'], inplace=True)
    if df_iot_sum_full.empty : logger.warning(f"({source_context}) IoT data empty after timestamp cleaning for env summary."); return summary_env_full

    num_cols_env_f = ['avg_co2_ppm','avg_pm25','waiting_room_occupancy','avg_noise_db']
    for col_ef in num_cols_env_f: df_iot_sum_full[col_ef] = _convert_to_numeric(df_iot_sum_full.get(col_ef), np.nan)
    
    if df_iot_sum_full['avg_co2_ppm'].notna().any(): summary_env_full["avg_co2_overall_ppm"] = df_iot_sum_full['avg_co2_ppm'].mean()
    if df_iot_sum_full['avg_pm25'].notna().any(): summary_env_full["avg_pm25_overall_ugm3"] = df_iot_sum_full['avg_pm25'].mean()
    if df_iot_sum_full.get('room_name',pd.Series(dtype=str)).str.contains('Waiting',case=False,na=False).any() and \
       df_iot_sum_full.get('waiting_room_occupancy',pd.Series(dtype=float)).notna().any():
        summary_env_full["avg_waiting_room_occupancy_persons"] = df_iot_sum_full.loc[df_iot_sum_full.get('room_name','').str.contains('Waiting',case=False,na=False), 'waiting_room_occupancy'].mean()
    if df_iot_sum_full['avg_noise_db'].notna().any(): summary_env_full["avg_noise_overall_dba"] = df_iot_sum_full['avg_noise_db'].mean()
    
    if all(c_ef in df_iot_sum_full for c_ef in ['clinic_id','room_name','timestamp']):
        latest_reads_ef = df_iot_sum_full.sort_values('timestamp').drop_duplicates(subset=['clinic_id','room_name'], keep='last')
        if not latest_reads_ef.empty:
            if 'avg_co2_ppm' in latest_reads_ef and latest_reads_ef['avg_co2_ppm'].notna().any(): summary_env_full["rooms_co2_very_high_alert_latest_count"] = latest_reads_ef[latest_reads_ef['avg_co2_ppm'] > app_config.ALERT_AMBIENT_CO2_VERY_HIGH_PPM].shape[0]
            if 'avg_pm25' in latest_reads_ef and latest_reads_ef['avg_pm25'].notna().any(): summary_env_full["rooms_pm25_very_high_alert_latest_count"] = latest_reads_ef[latest_reads_ef['avg_pm25'] > app_config.ALERT_AMBIENT_PM25_VERY_HIGH_UGM3].shape[0]
            waiting_rooms_latest = latest_reads_ef[latest_reads_ef.get('room_name','').str.contains('Waiting',case=False,na=False)]
            if not waiting_rooms_latest.empty and 'waiting_room_occupancy' in waiting_rooms_latest and waiting_rooms_latest['waiting_room_occupancy'].notna().any():
                 summary_env_full["waiting_room_high_occupancy_alert_latest_flag"] = (waiting_rooms_latest['waiting_room_occupancy'] > app_config.TARGET_CLINIC_WAITING_ROOM_OCCUPANCY_MAX).any()
            if 'avg_noise_db' in latest_reads_ef and latest_reads_ef['avg_noise_db'].notna().any(): summary_env_full["rooms_noise_high_alert_latest_count"] = latest_reads_ef[latest_reads_ef['avg_noise_db'] > app_config.ALERT_AMBIENT_NOISE_HIGH_DBA].shape[0]
    return summary_env_full


def get_patient_alerts_for_clinic(health_df_period: pd.DataFrame, risk_threshold_moderate: int = app_config.RISK_SCORE_MODERATE_THRESHOLD, source_context: str = "ClinicAlertsFull") -> pd.DataFrame:
    logger.info(f"({source_context}) Generating full patient alerts for clinic.")
    expected_cols_clinic_alert_df = ['patient_id', 'encounter_date', 'Alert Reason', 'Priority Score', 'Key Details', 'ai_risk_score', 'condition', 'Context', 'suggested_action_code']
    if health_df_period is None or health_df_period.empty: return pd.DataFrame(columns=expected_cols_clinic_alert_df)
    
    # Use CHW alert logic as base
    base_alerts_list_clinic = get_patient_alerts_for_chw(health_df_period, source_context=f"{source_context}/BaseLogic", risk_threshold_moderate=risk_threshold_moderate, risk_threshold_high=app_config.RISK_SCORE_HIGH_THRESHOLD) # Make sure to pass risk thresholds here
    
    alerts_clinic_df = pd.DataFrame(base_alerts_list_clinic) if base_alerts_list_clinic else pd.DataFrame() # Ensure DF even if list is empty
    # Standardize column names from CHW alert output
    rename_map_clinic_alerts = {'primary_reason': 'Alert Reason', 'raw_priority_score': 'Priority Score', 'brief_details': 'Key Details', 'context_info': 'Context'}
    if not alerts_clinic_df.empty: alerts_clinic_df.rename(columns=rename_map_clinic_alerts, inplace=True, errors='ignore')

    # Add clinic-specific rules like repeated high-risk encounters
    if 'ai_risk_score' in health_df_period.columns and 'patient_id' in health_df_period.columns and 'encounter_date' in health_df_period.columns:
        high_risk_encounters_clinic = health_df_period[health_df_period['ai_risk_score'] >= app_config.RISK_SCORE_HIGH_THRESHOLD]
        if not high_risk_encounters_clinic.empty:
            repeated_high_risk_counts_clinic = high_risk_encounters_clinic.groupby('patient_id')['encounter_date'].nunique()
            patients_repeated_hr_clinic = repeated_high_risk_counts_clinic[repeated_high_risk_counts_clinic > 2].index
            
            for pat_id_clinic_repeat in patients_repeated_hr_clinic:
                num_visits_clinic = repeated_high_risk_counts_clinic.get(pat_id_clinic_repeat,0)
                if not alerts_clinic_df.empty and pat_id_clinic_repeat in alerts_clinic_df['patient_id'].values:
                    idx_existing_clinic_alert = alerts_clinic_df[alerts_clinic_df['patient_id'] == pat_id_clinic_repeat].index[0]
                    alerts_clinic_df.loc[idx_existing_clinic_alert, 'Alert Reason'] = (alerts_clinic_df.loc[idx_existing_clinic_alert, 'Alert Reason'] or "") + "; Repeated High Risk Encounters"
                    alerts_clinic_df.loc[idx_existing_clinic_alert, 'Priority Score'] = max(alerts_clinic_df.loc[idx_existing_clinic_alert, 'Priority Score'] or 0, 96)
                    alerts_clinic_df.loc[idx_existing_clinic_alert, 'Key Details'] = (alerts_clinic_df.loc[idx_existing_clinic_alert, 'Key Details'] or "") + f" ({num_visits_clinic} high risk visits)"
                else:
                    latest_rec_clinic_repeat = health_df_period[health_df_period['patient_id'] == pat_id_clinic_repeat].sort_values('encounter_date',ascending=False).iloc[0]
                    new_alert_entry_clinic = pd.DataFrame([{'patient_id': pat_id_clinic_repeat, 'encounter_date': latest_rec_clinic_repeat['encounter_date'],'condition': latest_rec_clinic_repeat.get('condition',"N/A"),'Alert Reason': f"Repeated High Risk Encounters ({num_visits_clinic} visits)",'Priority Score': 96, 'Key Details': f"Last AI Risk: {latest_rec_clinic_repeat.get('ai_risk_score', np.nan):.0f}", 'ai_risk_score': latest_rec_clinic_repeat.get('ai_risk_score', np.nan)}])
                    alerts_clinic_df = pd.concat([alerts_clinic_df, new_alert_entry_clinic], ignore_index=True)
    
    for col_fa_clinic in expected_cols_clinic_alert_df: # Ensure all expected columns are present
        if col_fa_clinic not in alerts_clinic_df.columns: alerts_clinic_df[col_fa_clinic] = pd.NA
    
    return alerts_clinic_df.sort_values(by='Priority Score', ascending=False).reset_index(drop=True) if not alerts_clinic_df.empty else pd.DataFrame(columns=expected_cols_clinic_alert_df)


def get_district_summary_kpis(enriched_zone_gdf: Optional[gpd.GeoDataFrame], source_context: str = "DHOReportKPIsFull") -> Dict[str, Any]:
    logger.info(f"({source_context}) Calculating full district summary KPIs.")
    kpis_dist_full: Dict[str, Any] = {"total_population_district":0, "population_weighted_avg_ai_risk_score":np.nan, "zones_meeting_high_risk_criteria_count":0, "district_avg_facility_coverage_score":np.nan, "district_overall_key_disease_prevalence_per_1000":np.nan, "district_population_weighted_avg_steps":np.nan, "district_avg_clinic_co2_ppm":np.nan}
    for cond_key_d_kpi in app_config.KEY_CONDITIONS_FOR_ACTION: kpis_dist_full[f"district_total_active_{cond_key_d_kpi.lower().replace(' ', '_').replace('-', '_').replace('(severe)','')}_cases"] = 0
    if not isinstance(enriched_zone_gdf, gpd.GeoDataFrame) or enriched_zone_gdf.empty: logger.warning(f"({source_context}) Enriched GDF empty/invalid."); return kpis_dist_full
    gdf_dist_kpi_full = enriched_zone_gdf.copy()
    cols_to_num_dist = ['population','avg_risk_score','total_active_key_infections','facility_coverage_score','avg_daily_steps_zone','zone_avg_co2']
    for cond_k_d in app_config.KEY_CONDITIONS_FOR_ACTION: cols_to_num_dist.append(f"active_{cond_k_d.lower().replace(' ', '_').replace('-', '_').replace('(severe)','')}_cases")
    for col_d_num in cols_to_num_dist: gdf_dist_kpi_full[col_d_num] = _convert_to_numeric(gdf_dist_kpi_full.get(col_d_num, 0.0), np.nan if 'avg_' in col_d_num or 'score' in col_d_num else 0.0)
    if 'population' in gdf_dist_kpi_full.columns: kpis_dist_full["total_population_district"] = gdf_dist_kpi_full['population'].sum()
    tot_pop_d_kpi = kpis_dist_full["total_population_district"]
    if pd.notna(tot_pop_d_kpi) and tot_pop_d_kpi > 0:
        for mc_d, kc_d in [('avg_risk_score','population_weighted_avg_ai_risk_score'),('facility_coverage_score','district_avg_facility_coverage_score'),('avg_daily_steps_zone','district_population_weighted_avg_steps')]:
            if mc_d in gdf_dist_kpi_full.columns and gdf_dist_kpi_full[mc_d].notna().any() and 'population' in gdf_dist_kpi_full.columns:
                vw_d, vv_d = gdf_dist_kpi_full.loc[gdf_dist_kpi_full[mc_d].notna(),'population'].fillna(0), gdf_dist_kpi_full.loc[gdf_dist_kpi_full[mc_d].notna(),mc_d]
                if not vv_d.empty: kpis_dist_full[kc_d] = np.average(vv_d, weights=vw_d) if vw_d.sum() > 0 else vv_d.mean()
                else: kpis_dist_full[kc_d] = np.nan
        if 'total_active_key_infections' in gdf_dist_kpi_full.columns: kpis_dist_full["district_overall_key_disease_prevalence_per_1000"] = (gdf_dist_kpi_full['total_active_key_infections'].sum()/tot_pop_d_kpi)*1000 if tot_pop_d_kpi > 0 else 0.0
    else: # Fallbacks
        logger.warning(f"({source_context}) District total population {tot_pop_d_kpi}. Using unweighted averages for some KPIs.")
        for mc_d, kc_d in [('avg_risk_score','population_weighted_avg_ai_risk_score'),('facility_coverage_score','district_avg_facility_coverage_score'),('avg_daily_steps_zone','district_population_weighted_avg_steps')]: kpis_dist_full[kc_d] = gdf_dist_kpi_full[mc_d].mean() if mc_d in gdf_dist_kpi_full and gdf_dist_kpi_full[mc_d].notna().any() else np.nan
        kpis_dist_full["district_overall_key_disease_prevalence_per_1000"] = np.nan
    if 'avg_risk_score' in gdf_dist_kpi_full.columns: kpis_dist_full["zones_meeting_high_risk_criteria_count"] = gdf_dist_kpi_full[gdf_dist_kpi_full['avg_risk_score'] >= app_config.DISTRICT_ZONE_HIGH_RISK_AVG_SCORE].shape[0]
    for cond_d_kpi in app_config.KEY_CONDITIONS_FOR_ACTION:
        col_n_d = f"active_{cond_d_kpi.lower().replace(' ', '_').replace('-', '_').replace('(severe)','')}_cases"; kpi_out_n_d = f"district_total_{col_n_d}"
        kpis_dist_full[kpi_out_n_d] = int(gdf_dist_kpi_full.get(col_n_d,0).sum()) if col_n_d in gdf_dist_kpi_full else 0
    if 'zone_avg_co2' in gdf_dist_kpi_full and gdf_dist_kpi_full['zone_avg_co2'].notna().any(): kpis_dist_full["district_avg_clinic_co2_ppm"] = gdf_dist_kpi_full[gdf_dist_kpi_full['zone_avg_co2'] > 0]['zone_avg_co2'].mean()
    return kpis_dist_full

def get_trend_data(df: pd.DataFrame, value_col: str, date_col: str = 'encounter_date', period: str = 'D', agg_func: str = 'mean', filter_col: Optional[str] = None, filter_val: Optional[Any] = None, source_context: str = "TrendDataUtilFull") -> pd.Series:
    logger.debug(f"({source_context}) Generating full trend for '{value_col}', period '{period}', agg '{agg_func}'.")
    if not isinstance(df, pd.DataFrame) or df.empty or date_col not in df.columns or value_col not in df.columns:
        logger.debug(f"({source_context}) Invalid input or missing cols for trend: df empty={df.empty if isinstance(df,pd.DataFrame) else 'NotDF'}, date_col='{date_col}' in? {date_col in (df.columns if isinstance(df,pd.DataFrame) else []) }, value_col='{value_col}' in? {value_col in (df.columns if isinstance(df,pd.DataFrame) else []) }")
        return pd.Series(dtype='float64')
    tdf_calc = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(tdf_calc[date_col]): tdf_calc[date_col] = pd.to_datetime(tdf_calc[date_col], errors='coerce')
    tdf_calc.dropna(subset=[date_col], inplace=True)
    if agg_func not in ['nunique', 'count', 'size']: tdf_calc.dropna(subset=[value_col], inplace=True)
    if tdf_calc.empty: logger.debug(f"({source_context}) DF empty after date/NA handling for trend."); return pd.Series(dtype='float64')
    if filter_col and filter_col in tdf_calc.columns and filter_val is not None:
        tdf_calc = tdf_calc[tdf_calc[filter_col] == filter_val]
        if tdf_calc.empty: logger.debug(f"({source_context}) DF empty after filter '{filter_col}=={filter_val}'."); return pd.Series(dtype='float64')
    tdf_calc.set_index(date_col, inplace=True)
    if agg_func in ['mean', 'sum', 'median', 'std', 'var'] and not pd.api.types.is_numeric_dtype(tdf_calc[value_col]):
        tdf_calc[value_col] = _convert_to_numeric(tdf_calc[value_col], np.nan); tdf_calc.dropna(subset=[value_col], inplace=True)
        if tdf_calc.empty: logger.debug(f"({source_context}) DF empty post-numeric conversion for agg."); return pd.Series(dtype='float64')
    try:
        res_trend = tdf_calc.groupby(pd.Grouper(freq=period))
        if agg_func == 'nunique': s_out = res_trend[value_col].nunique()
        elif agg_func == 'sum': s_out = res_trend[value_col].sum()
        elif agg_func == 'median': s_out = res_trend[value_col].median()
        elif agg_func == 'count': s_out = res_trend[value_col].count()
        elif agg_func == 'size': s_out = res_trend.size()
        elif agg_func == 'std': s_out = res_trend[value_col].std()
        elif agg_func == 'var': s_out = res_trend[value_col].var()
        else: s_out = res_trend[value_col].mean()
    except Exception as e_trend_final: logger.error(f"({source_context}) Trend resampling error ({value_col}/{agg_func}): {e_trend_final}", exc_info=True); return pd.Series(dtype='float64')
    return s_out


def get_supply_forecast_data(health_df: pd.DataFrame, forecast_days_out: int = 30, item_filter_list: Optional[List[str]] = None, source_context: str = "SupplyForecastLinearFull") -> pd.DataFrame:
    logger.info(f"({source_context}) Calculating full linear supply forecast for {forecast_days_out} days.")
    def_cols_fc_lin = ['item', 'date', 'initial_stock_at_forecast_start', 'base_consumption_rate_per_day', 'forecasted_stock_level', 'forecasted_days_of_supply', 'estimated_stockout_date_linear', 'lower_ci_days_supply', 'upper_ci_days_supply', 'initial_days_supply_at_forecast_start']
    req_cols_fc_lin_f = ['item', 'encounter_date', 'item_stock_agg_zone', 'consumption_rate_per_day']
    if health_df is None or health_df.empty or not all(c_fcl in health_df.columns for c_fcl in req_cols_fc_lin_f): logger.warning(f"({source_context}) Missing req cols for linear supply forecast: {req_cols_fc_lin_f}"); return pd.DataFrame(columns=def_cols_fc_lin)
    df_fc_lin_s = health_df.copy()
    df_fc_lin_s['encounter_date'] = pd.to_datetime(df_fc_lin_s['encounter_date'], errors='coerce')
    df_fc_lin_s.dropna(subset=['encounter_date', 'item'], inplace=True)
    df_fc_lin_s['item_stock_agg_zone'] = _convert_to_numeric(df_fc_lin_s.get('item_stock_agg_zone'), 0.0)
    df_fc_lin_s['consumption_rate_per_day'] = _convert_to_numeric(df_fc_lin_s.get('consumption_rate_per_day'), 0.0001)
    if df_fc_lin_s.empty: return pd.DataFrame(columns=def_cols_fc_lin)
    latest_stat_fc_lin = df_fc_lin_s.sort_values('encounter_date').drop_duplicates(subset=['item'], keep='last')
    if item_filter_list: latest_stat_fc_lin = latest_stat_fc_lin[latest_stat_fc_lin['item'].isin(item_filter_list)]
    if latest_stat_fc_lin.empty: return pd.DataFrame(columns=def_cols_fc_lin)
    forecasts_lin_l = []
    cons_std_dev_f = 0.25
    for _, row_fc_it in latest_stat_fc_lin.iterrows():
        it_fc, stk_fc, cons_r_fc, last_dt_fc = row_fc_it['item'], row_fc_it.get('item_stock_agg_zone',0.0), row_fc_it.get('consumption_rate_per_day',0.0001), row_fc_it['encounter_date']
        if pd.isna(stk_fc) or stk_fc < 0: stk_fc = 0.0
        cons_r_fc = max(0.0001, cons_r_fc)
        fc_dates_lin_f = pd.date_range(start=last_dt_fc + pd.Timedelta(days=1), periods=forecast_days_out, freq='D')
        init_dos_fc_lin_f = stk_fc / cons_r_fc if cons_r_fc > 0 else (np.inf if stk_fc > 0 else 0)
        est_stockout_dt_fc_lin_f = last_dt_fc + pd.to_timedelta(init_dos_fc_lin_f, unit='D') if np.isfinite(init_dos_fc_lin_f) else pd.NaT
        for day_idx_fc, current_fc_dt_lin_f in enumerate(fc_dates_lin_f):
            days_out_f = day_idx_fc + 1
            fc_stk_lvl = max(0, stk_fc - (cons_r_fc * days_out_f))
            fc_days_sup = fc_stk_lvl / cons_r_fc if cons_r_fc > 0 else (np.inf if fc_stk_lvl > 0 else 0)
            cons_r_ci_low = max(0.0001, cons_r_fc * (1 - cons_std_dev_f)); cons_r_ci_upp = cons_r_fc * (1 + cons_std_dev_f)
            stk_high_cons = max(0, stk_fc - (cons_r_ci_upp * days_out_f)); ds_high_cons = stk_high_cons / cons_r_ci_upp if cons_r_ci_upp > 0 else (np.inf if stk_high_cons > 0 else 0)
            stk_low_cons = max(0, stk_fc - (cons_r_ci_low * days_out_f)); ds_low_cons = stk_low_cons / cons_r_ci_low if cons_r_ci_low > 0 else (np.inf if stk_low_cons > 0 else 0)
            forecasts_lin_l.append({'item':it_fc, 'date':current_fc_dt_lin_f, 'initial_stock_at_forecast_start':stk_fc, 'base_consumption_rate_per_day':cons_r_fc, 'forecasted_stock_level':fc_stk_lvl, 'forecasted_days_of_supply':fc_days_sup, 'estimated_stockout_date_linear':est_stockout_dt_fc_lin_f, 'lower_ci_days_supply':ds_high_cons, 'upper_ci_days_supply':ds_low_cons, 'initial_days_supply_at_forecast_start':init_dos_fc_lin_f})
    if not forecasts_lin_l: return pd.DataFrame(columns=def_cols_fc_lin)
    return pd.DataFrame(forecasts_lin_l)
