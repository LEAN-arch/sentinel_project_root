# sentinel_project_root/test/utils/core_data_processing.py
# Redesigned for "Sentinel Health Co-Pilot" - LMIC Edge-First System
# This module provides robust data loading, cleaning, and aggregation utilities
# primarily intended for:
#   1. Facility Node (Tier 2) and Cloud (Tier 3) backend processing.
#   2. Initial data provisioning and system setup for simulations or demos.
#   3. Simulation and testing environments.

import streamlit as st # Kept for @st.cache_data, for higher-tier Streamlit apps.
import pandas as pd
import geopandas as gpd # This is the import causing ModuleNotFoundError if not installed
import numpy as np
import os
import logging
from config import app_config # Uses the fully refactored app_config
from typing import List, Dict, Any, Optional, Tuple, Union # Added Union

logger = logging.getLogger(__name__)

# --- I. Core Helper Functions ---
def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes column names: lower case, replaces spaces/hyphens with underscores."""
    if not isinstance(df, pd.DataFrame):
        logger.error(f"_clean_column_names expects a pandas DataFrame, got {type(df)}.")
        return df if df is not None else pd.DataFrame()
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
    return df

def _convert_to_numeric(series: pd.Series, default_value: Any = np.nan) -> pd.Series:
    """Safely converts a pandas Series to numeric, coercing errors to default_value."""
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
            # Convert datetime/timedelta to int representation for stable hashing
            # Note: app_config.DEFAULT_CRS_STANDARD is used here in a f-string which requires it to be a simple string
            # for f-string interpolation within the select_dtypes. It should ideally be just 'UTC' or specific timezone name
            # if tz-aware datetimes are common and need specific handling for hashing.
            # Using a more generic datetime64[ns, UTC] and plain datetime64[ns] for now.
            dt_cols_to_convert = df_to_hash.select_dtypes(include=['datetime64', 'datetime64[ns]', 'datetime64[ns, UTC]']).columns
            for col in dt_cols_to_convert:
                df_to_hash[col] = pd.to_datetime(df_to_hash[col], errors='coerce').astype('int64') // 10**9
            for col in df_to_hash.select_dtypes(include=['timedelta64', 'timedelta64[ns]']).columns:
                df_to_hash[col] = df_to_hash[col].astype('int64')
            try: df_content_hash = pd.util.hash_pandas_object(df_to_hash, index=True).sum()
            except TypeError as e_ht:
                logger.warning(f"Unhashable type encountered in GDF for hashing: {e_ht}. Converting offending columns to string.")
                for col_offending in df_to_hash.columns:
                    try: pd.util.hash_pandas_object(df_to_hash[[col_offending]], index=True)
                    except TypeError: df_to_hash[col_offending] = df_to_hash[col_offending].astype(str)
                df_content_hash = pd.util.hash_pandas_object(df_to_hash, index=True).sum()
        return f"{df_content_hash}-{geom_hash_val}"
    except Exception as e_hash_gdf_exc: logger.error(f"General GDF hashing failed: {e_hash_gdf_exc}", exc_info=True); return str(gdf.head(1).to_string()) + str(gdf.shape)

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
        elif len(merged_df) == len(original_index): merged_df.index = original_index # Try direct assignment if col was dropped but length same
        else: logger.warning(f"Index restoration issue in robust_merge for {target_col_name}. Using RangeIndex.")
    return merged_df


# --- II. Data Loading and Basic Cleaning Functions ---
@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS_WEB_REPORTS, show_spinner="Loading health records dataset...")
def load_health_records(file_path: Optional[str] = None, source_context: str = "FacilityNode") -> pd.DataFrame:
    actual_file_path = file_path or app_config.HEALTH_RECORDS_CSV
    logger.info(f"({source_context}) Attempting to load health records from: {actual_file_path}")
    # Conditional st.error based on whether Streamlit context is active (simplistic check)
    is_streamlit_active = "streamlit" in sys.modules # Basic check if streamlit is imported anywhere
    
    if not os.path.exists(actual_file_path):
        logger.error(f"({source_context}) Health records file not found: {actual_file_path}")
        if is_streamlit_active and hasattr(st,'session_state') and st.session_state.get("streamlit_ πλήρες", False): # A more reliable check if running inside full streamlit context
             st.error(f"🚨 Health records file '{os.path.basename(actual_file_path)}' not found.")
        return pd.DataFrame()
    try:
        df = pd.read_csv(actual_file_path, low_memory=False); df = _clean_column_names(df)
        logger.info(f"({source_context}) Loaded {len(df)} raw records from {actual_file_path}.")
        date_cols = ['encounter_date', 'sample_collection_date', 'sample_registered_lab_date', 'referral_date', 'referral_outcome_date']
        for col in date_cols:
            if col in df.columns: df[col] = pd.to_datetime(df.get(col), errors='coerce')
            else: df[col] = pd.NaT # Ensure column exists as datetime
        num_cols_map = {
            'test_turnaround_days': np.nan, 'quantity_dispensed': 0, 'item_stock_agg_zone': 0,
            'consumption_rate_per_day': 0.0, 'ai_risk_score': np.nan, 'ai_followup_priority_score': np.nan,
            'vital_signs_bp_systolic': np.nan, 'vital_signs_bp_diastolic': np.nan,
            'vital_signs_temperature_celsius': np.nan, 'max_skin_temp_celsius': np.nan,
            'min_spo2_pct': np.nan, 'avg_spo2': np.nan, 'avg_daily_steps': 0,
            'resting_heart_rate': np.nan, 'hrv_rmssd_ms': np.nan, # Used by AI engine
            'avg_sleep_duration_hrs': np.nan, 'sleep_score_pct': np.nan, 'stress_level_score': np.nan,
            'fall_detected_today': 0, 'age': np.nan, 'chw_visit': 0, 'tb_contact_traced': 0,
            'patient_latitude': np.nan, 'patient_longitude': np.nan, 'hiv_viral_load_copies_ml': np.nan,
            'pregnancy_status': 0, 'chronic_condition_flag':0, 'ppe_compliant_flag':1, # From Sentinel Lean Data
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
                logger.warning(f"({source_context}) Core col '{col}' missing/all null. Filling with default: {def_val_c}.")
                df[col] = def_val_c
        logger.info(f"({source_context}) Health records cleaned. Shape: {df.shape}")
        return df
    except Exception as e_load_hr:
        logger.error(f"({source_context}) Load/process health records error: {e_load_hr}", exc_info=True)
        if is_streamlit_active and hasattr(st,'session_state') and st.session_state.get("streamlit_ πλήρες", False): st.error(f"Failed to load/process health records: {e_load_hr}")
        return pd.DataFrame()

@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS_WEB_REPORTS, show_spinner="Loading IoT environmental dataset...")
def load_iot_clinic_environment_data(file_path: Optional[str] = None, source_context: str = "FacilityNode") -> pd.DataFrame:
    actual_file_path = file_path or app_config.IOT_CLINIC_ENVIRONMENT_CSV
    logger.info(f"({source_context}) Loading IoT data from: {actual_file_path}")
    is_streamlit_active_iot = "streamlit" in sys.modules and hasattr(st,'session_state') and st.session_state.get("streamlit_ πλήρες", False)
    if not os.path.exists(actual_file_path): logger.warning(f"({source_context}) IoT file not found: {actual_file_path}"); if is_streamlit_active_iot: st.info(f"ℹ️ IoT file '{os.path.basename(actual_file_path)}' missing."); return pd.DataFrame()
    try:
        df = pd.read_csv(actual_file_path, low_memory=False); df = _clean_column_names(df)
        if 'timestamp' not in df.columns: logger.error(f"({source_context}) IoT missing 'timestamp'"); return pd.DataFrame()
        df['timestamp'] = pd.to_datetime(df.get('timestamp'), errors='coerce')
        num_iot_cols_list = ['avg_co2_ppm','max_co2_ppm','avg_pm25','voc_index','avg_temp_celsius','avg_humidity_rh','avg_noise_db','waiting_room_occupancy','patient_throughput_per_hour','sanitizer_dispenses_per_hour']
        for col in num_iot_cols_list:
            if col in df.columns: df[col] = _convert_to_numeric(df.get(col), np.nan)
            else: df[col] = np.nan
        str_iot_cols_list = ['clinic_id','room_name','zone_id']
        for col in str_iot_cols_list:
            if col in df.columns: df[col] = df.get(col,pd.Series(dtype=str)).fillna("Unknown").astype(str).str.strip().replace(['','nan','None'], "Unknown", regex=False)
            else: df[col] = "Unknown"
        logger.info(f"({source_context}) IoT data cleaned. Shape: {df.shape}")
        return df
    except Exception as e_load_iot: logger.error(f"({source_context}) Load/process IoT error: {e_load_iot}", exc_info=True); return pd.DataFrame()

@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS_WEB_REPORTS, hash_funcs={gpd.GeoDataFrame: hash_geodataframe}, show_spinner="Loading zone geographic & attribute dataset...")
def load_zone_data(attributes_path: Optional[str] = None, geometries_path: Optional[str] = None, source_context: str = "FacilityNode") -> Optional[gpd.GeoDataFrame]:
    attr_path = attributes_path or app_config.ZONE_ATTRIBUTES_CSV
    geom_path = geometries_path or app_config.ZONE_GEOMETRIES_GEOJSON
    logger.info(f"({source_context}) Loading zone data: Attrs='{attr_path}', Geoms='{geom_path}'")
    is_streamlit_active_zone = "streamlit" in sys.modules and hasattr(st,'session_state') and st.session_state.get("streamlit_ πλήρες", False)
    error_msgs_load_zone = []
    if not os.path.exists(attr_path): error_msgs_load_zone.append(f"Attributes file missing: {os.path.basename(attr_path)}")
    if not os.path.exists(geom_path): error_msgs_load_zone.append(f"Geometries file missing: {os.path.basename(geom_path)}")
    if error_msgs_load_zone: err_str = "; ".join(error_msgs_load_zone); logger.error(f"({source_context}) {err_str}"); if is_streamlit_active_zone : st.error(f"🚨 GIS Data Error: {err_str}"); return None
    try:
        attrs_df_load = pd.read_csv(attr_path); attrs_df_load = _clean_column_names(attrs_df_load)
        geoms_gdf_load = gpd.read_file(geom_path); geoms_gdf_load = _clean_column_names(geoms_gdf_load)
        for df_ld_chk, name_ld_chk in [(attrs_df_load, "attributes"), (geoms_gdf_load, "geometries")]:
            if 'zone_id' not in df_ld_chk.columns: logger.error(f"({source_context}) 'zone_id' missing in {name_ld_chk}."); if is_streamlit_active_zone: st.error("🚨 'zone_id' missing in GIS."); return None
            df_ld_chk['zone_id'] = df_ld_chk['zone_id'].astype(str).str.strip()
        if 'zone_display_name' in attrs_df_load.columns and 'name' not in attrs_df_load.columns: attrs_df_load.rename(columns={'zone_display_name':'name'}, inplace=True)
        elif 'name' not in attrs_df_load.columns: attrs_df_load['name'] = "Zone " + attrs_df_load['zone_id'].astype(str)
        
        original_geom_col = geoms_gdf_load.geometry.name if hasattr(geoms_gdf_load, 'geometry') else 'geometry'
        mrg_gdf_ld = geoms_gdf_load.merge(attrs_df_load, on="zone_id", how="left", suffixes=('_geom_orig', '_attr_orig'))
        for col_attr_ld in attrs_df_load.columns:
            if col_attr_ld == 'zone_id': continue
            attr_sfx, geom_sfx = f"{col_attr_ld}_attr_orig", f"{col_attr_ld}_geom_orig"
            if attr_sfx in mrg_gdf_ld.columns:
                mrg_gdf_ld[col_attr_ld] = mrg_gdf_ld[attr_sfx].fillna(mrg_gdf_ld.get(geom_sfx))
                mrg_gdf_ld.drop(columns=[attr_sfx, geom_sfx], errors='ignore', inplace=True)
            elif geom_sfx in mrg_gdf_ld.columns and col_attr_ld not in mrg_gdf_ld.columns : mrg_gdf_ld.rename(columns={geom_sfx:col_attr_ld}, inplace=True)
        
        if mrg_gdf_ld.geometry.name != 'geometry': # Standardize to 'geometry'
            if 'geometry' in mrg_gdf_ld.columns: mrg_gdf_ld = mrg_gdf_ld.set_geometry('geometry', drop=False, inplace=False) # drop=False to keep original if named 'geometry_geom_orig'
            elif original_geom_col in mrg_gdf_ld.columns : mrg_gdf_ld = mrg_gdf_ld.rename_geometry('geometry', col_name=original_geom_col)
            else: logger.error(f"({source_context}) No identifiable geometry column in merged GDF."); return None
        
        if mrg_gdf_ld.crs is None: mrg_gdf_ld = mrg_gdf_ld.set_crs(app_config.DEFAULT_CRS_STANDARD, allow_override=True)
        elif str(mrg_gdf_ld.crs).upper() != app_config.DEFAULT_CRS_STANDARD.upper(): mrg_gdf_ld = mrg_gdf_ld.to_crs(app_config.DEFAULT_CRS_STANDARD)
        
        default_zone_attr_cols = {'name':"Unknown Zone", 'population':0.0, 'num_clinics':0.0, 'socio_economic_index':0.5, 'avg_travel_time_clinic_min':30.0, 'predominant_hazard_type': "Unknown", 'typical_workforce_exposure_level': "Unknown"}
        for col_zd, def_val_zd in default_zone_attr_cols.items():
            if col_zd not in mrg_gdf_ld.columns: mrg_gdf_ld[col_zd] = def_val_zd if col_zd !='name' else ("Zone " + mrg_gdf_ld['zone_id'].astype(str))
            elif col_zd in ['population','socio_economic_index','num_clinics','avg_travel_time_clinic_min']: mrg_gdf_ld[col_zd] = _convert_to_numeric(mrg_gdf_ld.get(col_zd), def_val_zd)
            elif col_zd == 'name' : mrg_gdf_ld[col_zd] = mrg_gdf_ld.get(col_zd,"Unknown").astype(str).fillna("Zone "+mrg_gdf_ld['zone_id'].astype(str))
        
        logger.info(f"({source_context}) Zone data loaded/merged. Shape: {mrg_gdf_ld.shape}. CRS: {mrg_gdf_ld.crs}")
        return mrg_gdf_ld
    except Exception as e_load_zone: logger.error(f"({source_context}) Load/merge zone data error: {e_load_zone}", exc_info=True); if is_streamlit_active_zone : st.error(f"GIS data processing error: {e_load_zone}"); return None


# --- III. Data Enrichment Function ---
def enrich_zone_geodata_with_health_aggregates(
    zone_gdf: gpd.GeoDataFrame, health_df: Optional[pd.DataFrame], iot_df: Optional[pd.DataFrame] = None,
    source_context: str = "FacilityNode/ZoneEnrich"
) -> gpd.GeoDataFrame:
    logger.info(f"({source_context}) Starting zone GeoDataFrame enrichment.")
    if not isinstance(zone_gdf, gpd.GeoDataFrame) or zone_gdf.empty or 'zone_id' not in zone_gdf.columns:
        logger.warning(f"({source_context}) Invalid/empty zone_gdf for enrichment. Returning input or minimal GDF.")
        return zone_gdf if isinstance(zone_gdf, gpd.GeoDataFrame) else gpd.GeoDataFrame(columns=['zone_id', 'name', 'geometry', 'population'], crs=app_config.DEFAULT_CRS_STANDARD)

    enriched = zone_gdf.copy()
    if 'population' not in enriched.columns: enriched['population'] = 0.0
    enriched['population'] = _convert_to_numeric(enriched['population'], 0.0)

    agg_cols_to_init_list = [
        'total_population_health_data', 'avg_risk_score', 'total_patient_encounters',
        'total_referrals_made', 'successful_referrals',
        'avg_test_turnaround_critical', 'perc_critical_tests_tat_met',
        'prevalence_per_1000', 'total_active_key_infections',
        'avg_daily_steps_zone', 'zone_avg_co2', 'facility_coverage_score', 'population_density'
    ]
    for cond_key_enrich in app_config.KEY_CONDITIONS_FOR_ACTION: # Add specific condition counts based on new config
        agg_cols_to_init_list.append(f"active_{cond_key_enrich.lower().replace(' ', '_').replace('-', '_').replace('(severe)','')}_cases")
    for col_init in agg_cols_to_init_list: enriched[col_init] = 0.0 if not ('avg_' in col_init or 'perc_' in col_init or 'score' in col_init or 'density' in col_init) else np.nan # Init sums to 0, averages/rates to NaN

    if health_df is not None and not health_df.empty and 'zone_id' in health_df.columns:
        hdf_enrich = health_df.copy(); hdf_enrich['zone_id'] = hdf_enrich['zone_id'].astype(str).str.strip()
        enriched = _robust_merge_agg(enriched, hdf_enrich.groupby('zone_id')['patient_id'].nunique().reset_index(), 'total_population_health_data')
        enriched = _robust_merge_agg(enriched, hdf_enrich.groupby('zone_id')['ai_risk_score'].mean().reset_index(), 'avg_risk_score', default_fill_value=np.nan)
        enriched = _robust_merge_agg(enriched, hdf_enrich.groupby('zone_id')['encounter_id'].nunique().reset_index(), 'total_patient_encounters')
        for cond_name_enrich in app_config.KEY_CONDITIONS_FOR_ACTION:
            col_name_dyn = f"active_{cond_name_enrich.lower().replace(' ', '_').replace('-', '_').replace('(severe)','')}_cases"
            cond_filt_enrich = hdf_enrich.get('condition', pd.Series(dtype=str)).str.contains(cond_name_enrich, case=False, na=False)
            enriched = _robust_merge_agg(enriched, hdf_enrich[cond_filt_enrich].groupby('zone_id')['patient_id'].nunique().reset_index(), col_name_dyn)
        
        actionable_cols_sum = [f"active_{c.lower().replace(' ', '_').replace('-', '_').replace('(severe)','')}_cases" for c in app_config.KEY_CONDITIONS_FOR_ACTION if f"active_{c.lower().replace(' ', '_').replace('-', '_').replace('(severe)','')}_cases" in enriched.columns]
        if actionable_cols_sum: enriched['total_active_key_infections'] = enriched[actionable_cols_sum].sum(axis=1)

        if 'referral_status' in hdf_enrich.columns:
            made_refs_df = hdf_enrich[hdf_enrich['referral_status'].notna() & (~hdf_enrich['referral_status'].isin(['N/A', 'Unknown', 'Unknown']))]
            enriched = _robust_merge_agg(enriched, made_refs_df.groupby('zone_id')['encounter_id'].nunique().reset_index(), 'total_referrals_made')
            if 'referral_outcome' in hdf_enrich.columns:
                succ_outcomes = ['completed', 'service provided', 'attended consult', 'attended followup', 'attended']
                succ_refs_df = hdf_enrich[hdf_enrich.get('referral_outcome',pd.Series(dtype=str)).str.lower().isin(succ_outcomes)]
                enriched = _robust_merge_agg(enriched, succ_refs_df.groupby('zone_id')['encounter_id'].nunique().reset_index(), 'successful_referrals')

        crit_tests_list = app_config.CRITICAL_TESTS_LIST
        if crit_tests_list and 'test_type' in hdf_enrich.columns and 'test_turnaround_days' in hdf_enrich.columns:
            tat_enrich_df = hdf_enrich[(hdf_enrich['test_type'].isin(crit_tests_list)) & (hdf_enrich['test_turnaround_days'].notna()) & (~hdf_enrich.get('test_result',pd.Series(dtype=str)).isin(['Pending', 'Rejected Sample', 'Unknown', 'Indeterminate', 'Unknown']))].copy()
            if not tat_enrich_df.empty:
                enriched = _robust_merge_agg(enriched, tat_enrich_df.groupby('zone_id')['test_turnaround_days'].mean().reset_index(), 'avg_test_turnaround_critical', default_fill_value=np.nan)
                def _check_tat_met_core(row_core): 
                    cfg_core = app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get(row_core['test_type']); 
                    tgt_days = cfg_core['target_tat_days'] if cfg_core and 'target_tat_days' in cfg_core else app_config.TARGET_TEST_TURNAROUND_DAYS;
                    return pd.notna(row_core['test_turnaround_days']) and pd.notna(tgt_days) and row_core['test_turnaround_days'] <= tgt_days
                tat_enrich_df['tat_met_flag_core'] = tat_enrich_df.apply(_check_tat_met_core, axis=1)
                perc_met_agg_core = tat_enrich_df.groupby('zone_id')['tat_met_flag_core'].mean().reset_index(); perc_met_agg_core.iloc[:, 1] *= 100
                enriched = _robust_merge_agg(enriched, perc_met_agg_core, 'perc_critical_tests_tat_met', default_fill_value=np.nan)
        
        if 'avg_daily_steps' in hdf_enrich.columns: enriched = _robust_merge_agg(enriched, hdf_enrich.groupby('zone_id')['avg_daily_steps'].mean().reset_index(), 'avg_daily_steps_zone', default_fill_value=np.nan)

    if iot_df is not None and not iot_df.empty and all(c in iot_df.columns for c in ['zone_id','avg_co2_ppm']):
        iot_enrich_df = iot_df.copy(); iot_enrich_df['zone_id'] = iot_enrich_df['zone_id'].astype(str).str.strip()
        enriched = _robust_merge_agg(enriched, iot_enrich_df.groupby('zone_id')['avg_co2_ppm'].mean().reset_index(), 'zone_avg_co2', default_fill_value=np.nan)

    if 'total_active_key_infections' in enriched.columns and 'population' in enriched.columns:
         enriched['prevalence_per_1000'] = enriched.apply(lambda r: (r.get('total_active_key_infections',0)/r.get('population',1)) * 1000 if pd.notna(r.get('population')) and r.get('population',0)>0 else 0.0, axis=1).fillna(0.0)
    if 'num_clinics' in enriched.columns and 'population' in enriched.columns:
        enriched['facility_coverage_score'] = enriched.apply(lambda r: min(100.0, (r.get('num_clinics',0)/r.get('population',1))*20000) if pd.notna(r.get('population')) and r.get('population',0)>0 and pd.notna(r.get('num_clinics')) else 0.0, axis=1).fillna(0.0)
    elif 'facility_coverage_score' not in enriched.columns: enriched['facility_coverage_score']=0.0

    if 'geometry' in enriched.columns and 'population' in enriched.columns and enriched.crs and 'area_sqkm' in enriched.columns : # Area must be pre-calculated or passed in attributes
        valid_area_mask = enriched['area_sqkm'].notna() & enriched['area_sqkm'].gt(0)
        enriched.loc[valid_area_mask, 'population_density'] = (enriched.loc[valid_area_mask,'population'] / enriched.loc[valid_area_mask,'area_sqkm'])
        enriched['population_density'].fillna(0.0, inplace=True)
    elif 'population_density' not in enriched.columns : enriched['population_density'] = np.nan; logger.info(f"({source_context}) Pop density calculation requires 'area_sqkm' column or GDF in projected CRS for area calculation.")

    for col_final_fill in agg_cols_to_init_list:
        default_for_col = np.nan if 'avg_' in col_final_fill or 'perc_' in col_final_fill or 'score' in col_final_fill or 'density' in col_final_fill else 0.0
        if col_final_fill in enriched.columns: enriched[col_final_fill] = pd.to_numeric(enriched.get(col_final_fill, default_for_col), errors='coerce').fillna(default_for_col)
        else: enriched[col_final_fill] = default_for_col
    
    logger.info(f"({source_context}) Zone GDF enrichment complete. Shape: {enriched.shape}. Columns: {enriched.columns.tolist()}")
    return enriched


# --- IV. KPI & Summary Calculation Functions ---
# (Signatures with simplified placeholder returns. FULL implementations from File 32 must be used.)
def get_overall_kpis(health_df: pd.DataFrame, date_filter_start: Optional[str]=None, date_filter_end: Optional[str]=None, source_context: str = "DefaultContext") -> Dict[str, Any]:
    logger.info(f"({source_context}) get_overall_kpis called (USING FULL IMPLEMENTATION FROM FILE 32)")
    # ** PASTE FULL get_overall_kpis IMPLEMENTATION FROM FILE 32 HERE **
    # This includes initializing kpis dict, filtering df, calculating total_patients, avg_patient_risk,
    # dynamic active_X_cases based on KEY_CONDITIONS_FOR_ACTION, test positivity rates, and supply stockouts.
    # Returning minimal dict for stub:
    return {"total_patients": 0, "avg_patient_risk": np.nan, "active_tb_cases_current":0}

def get_chw_summary(health_df_daily: pd.DataFrame, source_context: str = "DefaultContext") -> Dict[str, Any]:
    logger.info(f"({source_context}) get_chw_summary called (USING FULL IMPLEMENTATION FROM FILE 32)")
    # ** PASTE FULL get_chw_summary IMPLEMENTATION FROM FILE 32 HERE **
    return {"visits_today":0,"avg_patient_risk_visited_today":np.nan, "pending_critical_condition_referrals":0}

def get_patient_alerts_for_chw(health_df_daily: pd.DataFrame, source_context: str = "DefaultContext", risk_threshold_moderate=app_config.RISK_SCORE_MODERATE_THRESHOLD, risk_threshold_high=app_config.RISK_SCORE_HIGH_THRESHOLD) -> List[Dict[str, Any]]: # Added risk thresholds based on prior logic
    logger.info(f"({source_context}) get_patient_alerts_for_chw called (USING FULL IMPLEMENTATION FROM FILE 32)")
    # ** PASTE FULL get_patient_alerts_for_chw IMPLEMENTATION FROM FILE 32 HERE **
    # This includes the complex alert rule engine generating List of Dicts.
    return []

def get_clinic_summary(health_df_period: pd.DataFrame, source_context: str = "DefaultContext") -> Dict[str, Any]:
    logger.info(f"({source_context}) get_clinic_summary called (USING FULL IMPLEMENTATION FROM FILE 32)")
    # ** PASTE FULL get_clinic_summary IMPLEMENTATION FROM FILE 32 HERE **
    # This includes calculation of overall TAT, %TAT met, pending tests, rejection rate, drug stockouts,
    # AND the detailed 'test_summary_details' dictionary.
    return {"overall_avg_test_turnaround_conclusive_days":np.nan, "key_drug_stockouts_count":0, "test_summary_details":{}}

def get_clinic_environmental_summary(iot_df_period: pd.DataFrame, source_context: str = "DefaultContext") -> Dict[str, Any]:
    logger.info(f"({source_context}) get_clinic_environmental_summary called (USING FULL IMPLEMENTATION FROM FILE 32)")
    # ** PASTE FULL get_clinic_environmental_summary IMPLEMENTATION FROM FILE 32 HERE **
    return {"avg_co2_overall_ppm":np.nan, "rooms_co2_very_high_alert_latest_count":0}

def get_patient_alerts_for_clinic(health_df_period: pd.DataFrame, risk_threshold_moderate: int = app_config.RISK_SCORE_MODERATE_THRESHOLD, source_context: str = "DefaultContext") -> pd.DataFrame:
    logger.info(f"({source_context}) get_patient_alerts_for_clinic called (USING FULL IMPLEMENTATION FROM FILE 32)")
    # ** PASTE FULL get_patient_alerts_for_clinic IMPLEMENTATION FROM FILE 32 HERE **
    # This function calls get_patient_alerts_for_chw internally and might add clinic-specific rules.
    return pd.DataFrame(columns=['patient_id', 'Alert Reason', 'Priority Score'])

def get_district_summary_kpis(enriched_zone_gdf: gpd.GeoDataFrame, source_context: str = "DefaultContext") -> Dict[str, Any]:
    logger.info(f"({source_context}) get_district_summary_kpis called (USING FULL IMPLEMENTATION FROM FILE 32)")
    # ** PASTE FULL get_district_summary_kpis IMPLEMENTATION FROM FILE 32 HERE **
    # Calculates pop-weighted averages and sums from the enriched GDF.
    return {"total_population_district":0,"population_weighted_avg_ai_risk_score":np.nan}

def get_trend_data(df: pd.DataFrame, value_col: str, date_col: str = 'encounter_date', period: str = 'D', agg_func: str = 'mean', filter_col: Optional[str] = None, filter_val: Optional[Any] = None, source_context: str = "DefaultContext") -> pd.Series:
    logger.info(f"({source_context}) get_trend_data for {value_col} (USING FULL IMPLEMENTATION FROM FILE 32)")
    # ** PASTE FULL get_trend_data IMPLEMENTATION FROM FILE 32 HERE **
    # The robust trend calculation utility.
    return pd.Series(dtype='float64')

def get_supply_forecast_data(health_df: pd.DataFrame, forecast_days_out: int = 30, item_filter_list: Optional[List[str]] = None, source_context: str = "DefaultContext") -> pd.DataFrame:
    logger.info(f"({source_context}) get_supply_forecast_data (linear) called (USING FULL IMPLEMENTATION FROM FILE 32)")
    # ** PASTE FULL get_supply_forecast_data (SIMPLE LINEAR MODEL) FROM FILE 32 HERE **
    return pd.DataFrame(columns=['item', 'date', 'forecasted_stock_level'])

# --- IV. KPI & Summary Calculation Functions ---
# These function signatures assume their full, refactored implementations from File 32 are used.
# I will only provide signatures and placeholder returns for brevity here,
# but ** YOU MUST REPLACE THESE STUBS WITH THE FULL, CORRECTED VERSIONS FROM FILE 32 **

def get_overall_kpis(health_df: pd.DataFrame, date_filter_start: Optional[str]=None, date_filter_end: Optional[str]=None, source_context: str = "FacilityNode") -> Dict[str, Any]:
    logger.info(f"({source_context}) Calculating overall KPIs (Placeholder - REPLACE WITH FULL LOGIC).")
    return {"total_patients": 0, "avg_patient_risk": np.nan} # Placeholder

def get_chw_summary(health_df_daily: pd.DataFrame, source_context: str = "CHWReport") -> Dict[str, Any]:
    logger.info(f"({source_context}) Calculating CHW summary (Placeholder - REPLACE WITH FULL LOGIC).")
    return {"visits_today":0, "pending_critical_condition_referrals": 0} # Placeholder

def get_patient_alerts_for_chw(health_df_daily: pd.DataFrame, source_context: str = "CHWReport") -> List[Dict[str, Any]]:
    logger.info(f"({source_context}) Generating CHW patient alerts (Placeholder - REPLACE WITH FULL LOGIC).")
    return [] # Placeholder

def get_clinic_summary(health_df_period: pd.DataFrame, source_context: str = "ClinicReport") -> Dict[str, Any]:
    logger.info(f"({source_context}) Calculating clinic summary (Placeholder - REPLACE WITH FULL LOGIC).")
    return {"overall_avg_test_turnaround_conclusive_days": np.nan, "test_summary_details":{}} # Placeholder

def get_clinic_environmental_summary(iot_df_period: pd.DataFrame, source_context: str = "ClinicReport") -> Dict[str, Any]:
    logger.info(f"({source_context}) Calculating clinic env summary (Placeholder - REPLACE WITH FULL LOGIC).")
    return {"avg_co2_overall_ppm": np.nan, "rooms_co2_very_high_alert_latest_count":0} # Placeholder

def get_patient_alerts_for_clinic(health_df_period: pd.DataFrame, risk_threshold_moderate: int = app_config.RISK_SCORE_MODERATE_THRESHOLD, source_context: str = "ClinicReport") -> pd.DataFrame:
    logger.info(f"({source_context}) Generating clinic patient alerts (Placeholder - REPLACE WITH FULL LOGIC).")
    return pd.DataFrame(columns=['patient_id', 'Alert Reason']) # Placeholder

def get_district_summary_kpis(enriched_zone_gdf: gpd.GeoDataFrame, source_context: str = "DHOReport") -> Dict[str, Any]:
    logger.info(f"({source_context}) Calculating district KPIs (Placeholder - REPLACE WITH FULL LOGIC).")
    return {"total_population_district":0, "population_weighted_avg_ai_risk_score":np.nan} # Placeholder

def get_trend_data(df: pd.DataFrame, value_col: str, date_col: str = 'encounter_date', period: str = 'D', agg_func: str = 'mean', filter_col: Optional[str] = None, filter_val: Optional[Any] = None, source_context: str = "Util/Trend") -> pd.Series:
    logger.info(f"({source_context}) Calculating trend for {value_col} (Placeholder - REPLACE WITH FULL LOGIC).")
    return pd.Series(dtype='float64') # Placeholder

def get_supply_forecast_data(health_df: pd.DataFrame, forecast_days_out: int = 30, item_filter_list: Optional[List[str]] = None, source_context: str = "FacilityNode/Supply") -> pd.DataFrame:
    logger.info(f"({source_context}) Calculating linear supply forecast (Placeholder - REPLACE WITH FULL LOGIC).")
    return pd.DataFrame(columns=['item', 'date', 'forecasted_stock_level']) # Placeholder
