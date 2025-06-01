# sentinel_project_root/test/utils/core_data_processing.py
# Redesigned for "Sentinel Health Co-Pilot" - LMIC Edge-First System
# This module provides robust data loading, cleaning, and aggregation utilities
# primarily intended for:
#   1. Facility Node (Tier 2) and Cloud (Tier 3) backend processing.
#   2. Initial data provisioning and system setup for simulations or demos.
#   3. Simulation and testing environments.

import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import os
import logging
from config import app_config
from typing import List, Dict, Any, Optional, Tuple, Union

logger = logging.getLogger(__name__)

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
    is_streamlit_active = "streamlit" in sys.modules and hasattr(st, 'session_state') and hasattr(st.session_state, 'run_count')

    if not os.path.exists(actual_file_path):
        logger.error(f"({source_context}) Health records file not found: {actual_file_path}")
        if is_streamlit_active : st.error(f"🚨 Health records file '{os.path.basename(actual_file_path)}' not found.")
        return pd.DataFrame()
    try:
        df = pd.read_csv(actual_file_path, low_memory=False); df = _clean_column_names(df)
        logger.info(f"({source_context}) Loaded {len(df)} raw records from {actual_file_path}.")
        date_cols = ['encounter_date', 'sample_collection_date', 'sample_registered_lab_date', 'referral_date', 'referral_outcome_date']
        for col in date_cols:
            if col in df.columns: df[col] = pd.to_datetime(df.get(col), errors='coerce')
            else: df[col] = pd.NaT
        num_cols_map = {
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
        if is_streamlit_active: st.error(f"Failed loading/processing health records: {e_load_hr}")
        return pd.DataFrame()

@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS_WEB_REPORTS, show_spinner="Loading IoT environmental dataset...")
def load_iot_clinic_environment_data(file_path: Optional[str] = None, source_context: str = "FacilityNode") -> pd.DataFrame:
    actual_file_path = file_path or app_config.IOT_CLINIC_ENVIRONMENT_CSV
    logger.info(f"({source_context}) Loading IoT data from: {actual_file_path}")
    is_streamlit_active_iot = "streamlit" in sys.modules and hasattr(st, 'session_state') and hasattr(st.session_state, 'run_count')
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
    is_streamlit_active_zone = "streamlit" in sys.modules and hasattr(st, 'session_state') and hasattr(st.session_state, 'run_count')
    error_msgs_load_zone = []
    if not os.path.exists(attr_path): error_msgs_load_zone.append(f"Attrs file missing: {os.path.basename(attr_path)}")
    if not os.path.exists(geom_path): error_msgs_load_zone.append(f"Geoms file missing: {os.path.basename(geom_path)}")
    if error_msgs_load_zone: full_err_msg = "; ".join(error_msgs_load_zone); logger.error(f"({source_context}) {full_err_msg}"); if is_streamlit_active_zone : st.error(f"🚨 GIS Data Error: {full_err_msg}"); return None
    try:
        attrs_df_load = pd.read_csv(attr_path); attrs_df_load = _clean_column_names(attrs_df_load)
        geoms_gdf_load = gpd.read_file(geom_path); geoms_gdf_load = _clean_column_names(geoms_gdf_load)
        for df_ld_chk, name_ld_chk in [(attrs_df_load, "attributes"), (geoms_gdf_load, "geometries")]:
            if 'zone_id' not in df_ld_chk.columns: logger.error(f"({source_context}) 'zone_id' missing in {name_ld_chk}."); if is_streamlit_active_zone: st.error("🚨 'zone_id' missing in GIS."); return None
            df_ld_chk['zone_id'] = df_ld_chk['zone_id'].astype(str).str.strip()
        if 'zone_display_name' in attrs_df_load.columns and 'name' not in attrs_df_load.columns: attrs_df_load.rename(columns={'zone_display_name':'name'}, inplace=True)
        elif 'name' not in attrs_df_load.columns: attrs_df_load['name'] = "Zone " + attrs_df_load['zone_id'].astype(str)
        
        original_geom_col_name_in_geoms_gdf = geoms_gdf_load.geometry.name if hasattr(geoms_gdf_load, 'geometry') and hasattr(geoms_gdf_load.geometry, 'name') else 'geometry'
        
        mrg_gdf_ld = geoms_gdf_load.merge(attrs_df_load, on="zone_id", how="left", suffixes=('_geom_orig', '_attr_orig'))
        
        for col_name_attr in attrs_df_load.columns:
            if col_name_attr == 'zone_id': continue
            attr_suffixed, geom_suffixed = f"{col_name_attr}_attr_orig", f"{col_name_attr}_geom_orig"
            if attr_suffixed in mrg_gdf_ld.columns:
                mrg_gdf_ld[col_name_attr] = mrg_gdf_ld[attr_suffixed].fillna(mrg_gdf_ld.get(geom_suffixed))
                mrg_gdf_ld.drop(columns=[attr_suffixed, geom_suffixed], errors='ignore', inplace=True)
            elif geom_suffixed in mrg_gdf_ld.columns and col_name_attr not in mrg_gdf_ld.columns: mrg_gdf_ld.rename(columns={geom_suffixed:col_name_attr}, inplace=True)
        
        if mrg_gdf_ld.geometry.name != 'geometry':
            if 'geometry' in mrg_gdf_ld.columns: mrg_gdf_ld = mrg_gdf_ld.set_geometry('geometry', drop=False, inplace=False)
            elif original_geom_col_name_in_geoms_gdf in mrg_gdf_ld.columns: mrg_gdf_ld = mrg_gdf_ld.rename_geometry('geometry', col_name=original_geom_col_name_in_geoms_gdf)
            else: logger.error(f"({source_context}) No identifiable geometry column in merged GDF ({mrg_gdf_ld.columns.tolist()})."); return None

        if mrg_gdf_ld.crs is None: mrg_gdf_ld = mrg_gdf_ld.set_crs(app_config.DEFAULT_CRS_STANDARD, allow_override=True)
        elif str(mrg_gdf_ld.crs).upper() != app_config.DEFAULT_CRS_STANDARD.upper(): mrg_gdf_ld = mrg_gdf_ld.to_crs(app_config.DEFAULT_CRS_STANDARD)
        
        default_zone_attr_cols = {'name':"Unknown Zone", 'population':0.0, 'num_clinics':0.0, 'socio_economic_index':0.5, 'avg_travel_time_clinic_min':30.0, 'predominant_hazard_type': "Unknown", 'typical_workforce_exposure_level': "Unknown", 'area_sqkm':np.nan}
        for col_zd, def_val_zd in default_zone_attr_cols.items():
            if col_zd not in mrg_gdf_ld.columns: mrg_gdf_ld[col_zd] = def_val_zd if col_zd !='name' else ("Zone " + mrg_gdf_ld['zone_id'].astype(str))
            elif col_zd in ['population','socio_economic_index','num_clinics','avg_travel_time_clinic_min','area_sqkm']: mrg_gdf_ld[col_zd] = _convert_to_numeric(mrg_gdf_ld.get(col_zd), def_val_zd)
            elif col_zd == 'name' : mrg_gdf_ld[col_zd] = mrg_gdf_ld.get(col_zd,"Unknown").astype(str).fillna("Zone "+mrg_gdf_ld['zone_id'].astype(str))
        logger.info(f"({source_context}) Zone data loaded/merged. Shape: {mrg_gdf_ld.shape}. CRS: {mrg_gdf_ld.crs}")
        return mrg_gdf_ld
    except ImportError as e_gpd:
        logger.error(f"({source_context}) Geopandas import error during zone data load: {e_gpd}. Ensure geopandas and its dependencies are installed correctly.", exc_info=True)
        if is_streamlit_active_zone: st.error(f"🚨 GIS Library Error: GeoPandas not found or not working. Mapping features disabled. Details: {e_gpd}")
        return None # Cannot return GDF without geopandas
    except Exception as e_load_zone: logger.error(f"({source_context}) Load/merge zone data error: {e_load_zone}", exc_info=True); if is_streamlit_active_zone : st.error(f"GIS data processing error: {e_load_zone}"); return None

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
    
    for col_init in agg_cols_to_init_list: # Ensure columns exist before merge_agg and set default type for averages/rates
        default_for_init = np.nan if ('avg_' in col_init or 'perc_' in col_init or 'score' in col_init or 'density' in col_init) else 0.0
        if col_init not in enriched.columns: enriched[col_init] = default_for_init
        else: enriched[col_init] = _convert_to_numeric(enriched[col_init], default_for_init) # Ensure numeric

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
        if actionable_cols_sum: enriched['total_active_key_infections'] = enriched[actionable_cols_sum].sum(axis=1, skipna=True)
        if 'referral_status' in hdf_enrich.columns:
            made_refs_df = hdf_enrich[hdf_enrich['referral_status'].notna() & (~hdf_enrich['referral_status'].isin(['N/A', 'Unknown']))]
            enriched = _robust_merge_agg(enriched, made_refs_df.groupby('zone_id')['encounter_id'].nunique().reset_index(), 'total_referrals_made')
            if 'referral_outcome' in hdf_enrich.columns:
                succ_outcomes = ['completed', 'service provided', 'attended consult', 'attended followup', 'attended']
                succ_refs_df = hdf_enrich[hdf_enrich.get('referral_outcome',pd.Series(dtype=str)).str.lower().isin(succ_outcomes)]
                enriched = _robust_merge_agg(enriched, succ_refs_df.groupby('zone_id')['encounter_id'].nunique().reset_index(), 'successful_referrals')
        crit_tests_list_enr = app_config.CRITICAL_TESTS_LIST
        if crit_tests_list_enr and 'test_type' in hdf_enrich.columns and 'test_turnaround_days' in hdf_enrich.columns:
            tat_enrich_df = hdf_enrich[(hdf_enrich['test_type'].isin(crit_tests_list_enr)) & (hdf_enrich['test_turnaround_days'].notna()) & (~hdf_enrich.get('test_result',pd.Series(dtype=str)).isin(['Pending', 'Rejected Sample', 'Unknown', 'Indeterminate']))].copy()
            if not tat_enrich_df.empty:
                enriched = _robust_merge_agg(enriched, tat_enrich_df.groupby('zone_id')['test_turnaround_days'].mean().reset_index(), 'avg_test_turnaround_critical', default_fill_value=np.nan)
                def _check_tat_met_core_enr(row_c_enr): 
                    cfg_c_enr = app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get(str(row_c_enr.get('test_type','')))
                    tgt_days_enr = cfg_c_enr['target_tat_days'] if cfg_c_enr and 'target_tat_days' in cfg_c_enr else app_config.TARGET_TEST_TURNAROUND_DAYS
                    return pd.notna(row_c_enr['test_turnaround_days']) and pd.notna(tgt_days_enr) and row_c_enr['test_turnaround_days'] <= tgt_days_enr
                tat_enrich_df['tat_met_flag_core_enr'] = tat_enrich_df.apply(_check_tat_met_core_enr, axis=1)
                perc_met_agg_core_enr = tat_enrich_df.groupby('zone_id')['tat_met_flag_core_enr'].mean().reset_index(); perc_met_agg_core_enr.iloc[:, 1] *= 100
                enriched = _robust_merge_agg(enriched, perc_met_agg_core_enr, 'perc_critical_tests_tat_met', default_fill_value=np.nan) # Perc can be NaN
        if 'avg_daily_steps' in hdf_enrich.columns: enriched = _robust_merge_agg(enriched, hdf_enrich.groupby('zone_id')['avg_daily_steps'].mean().reset_index(), 'avg_daily_steps_zone', default_fill_value=np.nan)

    if iot_df is not None and not iot_df.empty and all(c in iot_df.columns for c in ['zone_id','avg_co2_ppm']):
        iot_enrich_df_agg = iot_df.copy(); iot_enrich_df_agg['zone_id'] = iot_enrich_df_agg['zone_id'].astype(str).str.strip()
        enriched = _robust_merge_agg(enriched, iot_enrich_df_agg.groupby('zone_id')['avg_co2_ppm'].mean().reset_index(), 'zone_avg_co2', default_fill_value=np.nan)
    
    if 'total_active_key_infections' in enriched.columns and 'population' in enriched.columns:
         enriched['prevalence_per_1000'] = enriched.apply(lambda r: (r.get('total_active_key_infections',0)/r.get('population',1)) * 1000 if pd.notna(r.get('population')) and r.get('population',0)>0 else 0.0, axis=1).fillna(0.0)
    if 'num_clinics' in enriched.columns and 'population' in enriched.columns:
        enriched['facility_coverage_score'] = enriched.apply(lambda r: min(100.0, (r.get('num_clinics',0)/r.get('population',1))*20000) if pd.notna(r.get('population')) and r.get('population',0)>0 and pd.notna(r.get('num_clinics')) else 0.0, axis=1).fillna(0.0)
    elif 'facility_coverage_score' not in enriched.columns: enriched['facility_coverage_score']=0.0

    if 'geometry' in enriched.columns and enriched.is_valid.all() and enriched.crs and 'population' in enriched.columns :
        if 'area_sqkm' in enriched.columns and enriched['area_sqkm'].notna().any() and enriched['area_sqkm'].gt(0).any() :
            valid_area_mask_enr = enriched['area_sqkm'].notna() & enriched['area_sqkm'].gt(0)
            enriched.loc[valid_area_mask_enr, 'population_density'] = (enriched.loc[valid_area_mask_enr,'population'] / enriched.loc[valid_area_mask_enr,'area_sqkm'])
            enriched['population_density'].fillna(0.0, inplace=True)
        elif 'population_density' not in enriched.columns:
            enriched['population_density'] = np.nan; logger.info(f"({source_context}) Population density: 'area_sqkm' column missing/invalid or GDF needs projected CRS for area calc.")
    
    for col_final_enr in agg_cols_to_init_list:
        default_final = np.nan if any(k_nan in col_final_enr for k_nan in ['avg_', 'perc_', 'score', 'density']) else 0.0
        if col_final_enr in enriched.columns: enriched[col_final_enr] = pd.to_numeric(enriched.get(col_final_enr, default_final), errors='coerce').fillna(default_final)
        else: enriched[col_final_enr] = default_final
    
    logger.info(f"({source_context}) Zone GDF enrichment complete. Shape: {enriched.shape}. Output columns: {enriched.columns.tolist()}")
    return enriched


# --- IV. KPI & Summary Calculation Functions (Full Implementations) ---

def get_overall_kpis(health_df: pd.DataFrame, date_filter_start: Optional[str]=None, date_filter_end: Optional[str]=None, source_context: str = "OverallKPIs") -> Dict[str, Any]:
    logger.info(f"({source_context}) Calculating overall KPIs for period: {date_filter_start or 'all'} to {date_filter_end or 'all'}.")
    kpis: Dict[str, Any] = {
        "total_patients": 0, "avg_patient_risk": np.nan,
        "malaria_rdt_positive_rate_period": np.nan, "hiv_rapid_positive_rate_period": np.nan, # Added HIV
        "key_supply_stockout_alerts": 0
    }
    # Dynamically add keys for KEY_CONDITIONS_FOR_ACTION to ensure they are initialized
    for cond_kpi_k in app_config.KEY_CONDITIONS_FOR_ACTION:
        kpis[f"active_{cond_kpi_k.lower().replace(' ', '_').replace('-', '_').replace('(severe)','')}_cases_current"] = 0

    if health_df is None or health_df.empty: logger.warning(f"({source_context}) No health data provided."); return kpis
    df_kpi_calc = health_df.copy()
    if 'encounter_date' not in df_kpi_calc.columns or df_kpi_calc['encounter_date'].isnull().all():
        logger.warning(f"({source_context}) 'encounter_date' missing or all null."); return kpis
    
    df_kpi_calc['encounter_date'] = pd.to_datetime(df_kpi_calc['encounter_date'], errors='coerce')
    df_kpi_calc.dropna(subset=['encounter_date'], inplace=True)
    
    start_dt_kpi = pd.to_datetime(date_filter_start, errors='coerce') if date_filter_start else None
    end_dt_kpi = pd.to_datetime(date_filter_end, errors='coerce') if date_filter_end else None
    if start_dt_kpi: df_kpi_calc = df_kpi_calc[df_kpi_calc['encounter_date'] >= start_dt_kpi]
    if end_dt_kpi: df_kpi_calc = df_kpi_calc[df_kpi_calc['encounter_date'] <= end_dt_kpi]
    if df_kpi_calc.empty: logger.info(f"({source_context}) No data after date filtering."); return kpis

    if 'patient_id' in df_kpi_calc: kpis["total_patients"] = df_kpi_calc['patient_id'].nunique()
    if 'ai_risk_score' in df_kpi_calc and df_kpi_calc['ai_risk_score'].notna().any(): kpis["avg_patient_risk"] = df_kpi_calc['ai_risk_score'].mean()
    
    if 'condition' in df_kpi_calc.columns:
        for cond_key_kpi in app_config.KEY_CONDITIONS_FOR_ACTION:
            kpi_col_name_kpi = f"active_{cond_key_kpi.lower().replace(' ', '_').replace('-', '_').replace('(severe)','')}_cases_current"
            kpis[kpi_col_name_kpi] = df_kpi_calc[df_kpi_calc['condition'].str.contains(cond_key_kpi, case=False, na=False)]['patient_id'].nunique()

    for test_key_from_config, kpi_name_for_output in [
        ("RDT-Malaria", "malaria_rdt_positive_rate_period"), 
        ("HIV-Rapid", "hiv_rapid_positive_rate_period") # Example for another test
    ]:
        if test_key_from_config in app_config.KEY_TEST_TYPES_FOR_ANALYSIS and \
           'test_type' in df_kpi_calc.columns and 'test_result' in df_kpi_calc.columns:
            test_df_analysis = df_kpi_calc[
                (df_kpi_calc['test_type'] == test_key_from_config) & 
                (~df_kpi_calc.get('test_result', pd.Series(dtype=str)).isin(["Pending", "Rejected Sample", "Unknown", "Indeterminate", "N/A", ""]))
            ]
            if not test_df_analysis.empty:
                kpis[kpi_name_for_output] = (test_df_analysis[test_df_analysis['test_result'] == 'Positive'].shape[0] / len(test_df_analysis)) * 100
            else: kpis[kpi_name_for_output] = 0.0 # No conclusive tests of this type, so 0% positive rate
        else: kpis[kpi_name_for_output] = np.nan # Test type not in data or columns missing

    if all(c in df_kpi_calc for c in ['item', 'item_stock_agg_zone', 'consumption_rate_per_day', 'encounter_date']) and app_config.KEY_DRUG_SUBSTRINGS_SUPPLY:
        supply_df_for_kpi = df_kpi_calc.sort_values('encounter_date').drop_duplicates(subset=['item', 'zone_id'], keep='last')
        supply_df_for_kpi['consumption_rate_per_day'] = supply_df_for_kpi['consumption_rate_per_day'].replace(0, np.nan) # Avoid division by zero for days_supply
        supply_df_for_kpi['days_supply_kpi_calc'] = supply_df_for_kpi['item_stock_agg_zone'] / supply_df_for_kpi['consumption_rate_per_day']
        supply_df_for_kpi.dropna(subset=['days_supply_kpi_calc'], inplace=True)
        key_drug_supply_kpi_df = supply_df_for_kpi[supply_df_for_kpi['item'].str.contains('|'.join(app_config.KEY_DRUG_SUBSTRINGS_SUPPLY), case=False, na=False)]
        kpis['key_supply_stockout_alerts'] = key_drug_supply_kpi_df[key_drug_supply_kpi_df['days_supply_kpi_calc'] < app_config.CRITICAL_SUPPLY_DAYS_REMAINING]['item'].nunique()
    return kpis


def get_chw_summary(health_df_daily: pd.DataFrame, source_context: str = "CHWSummary") -> Dict[str, Any]:
    logger.info(f"({source_context}) Calculating CHW daily summary.")
    summary: Dict[str, Any] = {
        "visits_today":0, "avg_patient_risk_visited_today":np.nan,
        "high_ai_prio_followups_today": 0, # Patients with high AI Followup score
        "patients_critical_spo2_today": 0, # SpO2 < CRITICAL_LOW
        "patients_high_fever_today": 0,    # Temp >= HIGH_FEVER
        "avg_patient_steps_visited_today":np.nan, "patients_fall_detected_today":0,
        "pending_critical_condition_referrals": 0 # CHW generated urgent referrals today
    }
    if health_df_daily is None or health_df_daily.empty: logger.warning(f"({source_context}) No daily data for CHW summary."); return summary
    
    chw_df_sum = health_df_daily.copy()
    if 'patient_id' in chw_df_sum: summary["visits_today"] = chw_df_sum['patient_id'].nunique()

    # Metrics based on unique patients visited
    if not chw_df_sum.empty and 'patient_id' in chw_df_sum.columns:
        unique_patients_df_chw = chw_df_sum.drop_duplicates(subset=['patient_id'], keep='first') # Base actions on unique patients seen

        if 'ai_risk_score' in unique_patients_df_chw.columns and unique_patients_df_chw['ai_risk_score'].notna().any():
            summary["avg_patient_risk_visited_today"] = unique_patients_df_chw['ai_risk_score'].mean()
        
        if 'ai_followup_priority_score' in chw_df_sum.columns and chw_df_sum['ai_followup_priority_score'].notna().any(): # Use full df for score check then unique patients
            summary["high_ai_prio_followups_today"] = chw_df_sum[chw_df_sum['ai_followup_priority_score'] >= app_config.FATIGUE_INDEX_HIGH_THRESHOLD]['patient_id'].nunique()
        
        if 'min_spo2_pct' in chw_df_sum.columns and chw_df_sum['min_spo2_pct'].notna().any(): # Use full df for vitals checks
            summary["patients_critical_spo2_today"] = chw_df_sum[chw_df_sum['min_spo2_pct'] < app_config.ALERT_SPO2_CRITICAL_LOW_PCT]['patient_id'].nunique()
        
        temp_col_for_chw_sum = next((tc for tc in ['vital_signs_temperature_celsius','max_skin_temp_celsius'] if tc in chw_df_sum and chw_df_sum[tc].notna().any()), None)
        if temp_col_for_chw_sum:
            summary["patients_high_fever_today"] = chw_df_sum[chw_df_sum[temp_col_for_chw_sum] >= app_config.ALERT_BODY_TEMP_HIGH_FEVER_C]['patient_id'].nunique()
        
        if 'avg_daily_steps' in unique_patients_df_chw.columns and unique_patients_df_chw['avg_daily_steps'].notna().any():
            summary["avg_patient_steps_visited_today"] = unique_patients_df_chw['avg_daily_steps'].mean()
        
        if 'fall_detected_today' in chw_df_sum.columns and chw_df_sum['fall_detected_today'].notna().any(): # Check falls in all encounters
            summary["patients_fall_detected_today"] = chw_df_sum[chw_df_sum['fall_detected_today'] > 0]['patient_id'].nunique()

    if all(c in chw_df_sum.columns for c in ['condition', 'referral_status', 'referral_reason']):
        crit_cond_set_chw_sum = set(app_config.KEY_CONDITIONS_FOR_ACTION)
        urgent_keywords_for_sum = ['urgent', 'emergency', 'critical', 'severe', 'immediate']
        
        # Define on a copy to avoid SettingWithCopyWarning
        chw_df_sum_copy = chw_df_sum.copy()
        chw_df_sum_copy['is_crit_ref_chw_sum'] = chw_df_sum_copy.apply(
            lambda r_sum: (str(r_sum.get('referral_status','Unknown')).lower() == 'pending' and 
                       (any(ck_sum.lower() in str(r_sum.get('condition','')).lower() for ck_sum in crit_cond_set_chw_sum) or
                        any(uk_sum.lower() in str(r_sum.get('referral_reason','')).lower() for uk_sum in urgent_keywords_for_sum))
                      ), axis=1
        )
        summary["pending_critical_condition_referrals"] = chw_df_sum_copy[chw_df_sum_copy['is_crit_ref_chw_sum']]['patient_id'].nunique()
    return summary

def get_patient_alerts_for_chw(health_df_daily: pd.DataFrame, source_context: str = "CHWAlertGen", risk_threshold_moderate=app_config.RISK_SCORE_MODERATE_THRESHOLD, risk_threshold_high=app_config.RISK_SCORE_HIGH_THRESHOLD) -> List[Dict[str, Any]]:
    logger.info(f"({source_context}) Generating CHW patient alerts list.")
    if health_df_daily is None or health_df_daily.empty: return []
    df_alerts_src = health_df_daily.copy()
    processed_alerts_output: List[Dict[str, Any]] = []
    
    cols_to_ensure_chw_alert = { # Ensure cols exist, default to benign or identifiable values
        'patient_id': "UnknownPatient", 'encounter_date': pd.NaT, 'condition': "N/A", 'zone_id': "UnknownZone",
        'age': np.nan, 'ai_risk_score': np.nan, 'ai_followup_priority_score': np.nan,
        'min_spo2_pct': 100.0, 'vital_signs_temperature_celsius': 37.0, 'max_skin_temp_celsius': 37.0,
        'fall_detected_today': 0, 'referral_status': "N/A", 'referral_reason': "N/A",
        'medication_adherence_self_report': "Unknown"
    }
    for col_a, def_a in cols_to_ensure_chw_alert.items():
        if col_a not in df_alerts_src.columns: df_alerts_src[col_a] = def_a
        # Basic type ensuring, fuller cleaning assumed done by load_health_records
        elif col_a == 'encounter_date': df_alerts_src[col_a] = pd.to_datetime(df_alerts_src[col_a], errors='coerce')
        elif isinstance(def_a, (float, int)): df_alerts_src[col_a] = pd.to_numeric(df_alerts_src[col_a], errors='coerce').fillna(def_a)

    temp_col_alerts = next((tc for tc in ['vital_signs_temperature_celsius', 'max_skin_temp_celsius'] if tc in df_alerts_src and df_alerts_src[tc].notna().any()), None)

    for _, record in df_alerts_src.iterrows():
        record_alerts: List[Dict[str, Any]] = []
        p_id, cond_rec, zone_rec, enc_date_rec = str(record.get('patient_id')), str(record.get('condition')), str(record.get('zone_id')), pd.to_datetime(record.get('encounter_date')).strftime('%Y-%m-%d') if pd.notna(record.get('encounter_date')) else 'N/A'
        ctx = f"Cond: {cond_rec} | Zone: {zone_rec} | Date: {enc_date_rec}"
        def add_alert(level, reason, details, action_code, score, trigger_val):
            record_alerts.append({"patient_id":p_id, "alert_level":level, "primary_reason":reason, "brief_details":details, "suggested_action_code":action_code, "raw_priority_score":score, "context_info":ctx, "triggering_value":trigger_val, "encounter_date": record.get('encounter_date')})

        spo2 = record.get('min_spo2_pct')
        if pd.notna(spo2):
            if spo2 < app_config.ALERT_SPO2_CRITICAL_LOW_PCT: add_alert("CRITICAL", "Crit. Low SpO2", f"SpO2: {spo2:.0f}%", "ACT_SPO2_URGENT", 98+(app_config.ALERT_SPO2_CRITICAL_LOW_PCT-spo2), f"SpO2 {spo2:.0f}%")
            elif spo2 < app_config.ALERT_SPO2_WARNING_LOW_PCT: add_alert("WARNING", "Low SpO2", f"SpO2: {spo2:.0f}%", "ACT_SPO2_MONITOR", 75+(app_config.ALERT_SPO2_WARNING_LOW_PCT-spo2), f"SpO2 {spo2:.0f}%")
        
        temp_val_rec = record.get(temp_col_alerts) if temp_col_alerts else np.nan
        if pd.notna(temp_val_rec):
            if temp_val_rec >= app_config.ALERT_BODY_TEMP_HIGH_FEVER_C: add_alert("CRITICAL", "High Fever", f"Temp: {temp_val_rec:.1f}°C", "ACT_FEVER_URGENT", 95+(temp_val_rec-app_config.ALERT_BODY_TEMP_HIGH_FEVER_C)*2, f"Temp {temp_val_rec:.1f}°C")
            elif temp_val_rec >= app_config.ALERT_BODY_TEMP_FEVER_C: add_alert("WARNING", "Fever", f"Temp: {temp_val_rec:.1f}°C", "ACT_FEVER_MONITOR", 70+(temp_val_rec-app_config.ALERT_BODY_TEMP_FEVER_C), f"Temp {temp_val_rec:.1f}°C")
        
        if record.get('fall_detected_today',0) > 0: add_alert("CRITICAL", "Fall Detected", f"Falls: {int(record.get('fall_detected_today',0))}", "ACT_FALL_ASSESS", 92, "Fall > 0")
        
        ai_prio = record.get('ai_followup_priority_score'); ai_risk = record.get('ai_risk_score')
        if pd.notna(ai_prio) and ai_prio >= risk_threshold_high: add_alert("WARNING", "High AI Prio.", f"AI Prio: {ai_prio:.0f}", "ACT_REVIEW_AI_PRIO", ai_prio, f"AI Prio {ai_prio:.0f}")
        elif pd.notna(ai_risk) and ai_risk >= risk_threshold_high and not (pd.notna(ai_prio) and ai_prio >= risk_threshold_high): add_alert("INFO", "High AI Risk", f"AI Risk: {ai_risk:.0f}", "ACT_MONITOR_AI_RISK", ai_risk, f"AI Risk {ai_risk:.0f}")
        elif pd.notna(ai_prio) and ai_prio >= risk_threshold_moderate : add_alert("INFO", "Mod. AI Prio.", f"AI Prio: {ai_prio:.0f}", "ACT_REVIEW_AI_PRIO", ai_prio, f"AI Prio {ai_prio:.0f}")
        
        if str(record.get('referral_status', '')).lower() == 'pending' and any(ck.lower() in str(record.get('condition','')).lower() for ck in app_config.KEY_CONDITIONS_FOR_ACTION):
            add_alert("WARNING", "Pending Critical Ref.", f"For: {record.get('condition')}", "ACT_FOLLOWUP_REF", 80, "PendCritRef")
        
        if record_alerts: processed_alerts_output.append(max(record_alerts, key=lambda x: x['raw_priority_score']))
    
    if not processed_alerts_output: return []
    alerts_final_df = pd.DataFrame(processed_alerts_output)
    if 'encounter_date' in alerts_final_df: alerts_final_df['encounter_date_obj_dedup'] = pd.to_datetime(alerts_final_df['encounter_date']).dt.date
    else: alerts_final_df['encounter_date_obj_dedup'] = pd.to_datetime(for_date).date() # fallback if enc_date was missing

    alerts_final_df.sort_values(by=["raw_priority_score","alert_level"], ascending=[False,True], inplace=True)
    alerts_final_df.drop_duplicates(subset=["patient_id", "encounter_date_obj_dedup"], keep="first", inplace=True)
    return alerts_final_df.to_dict(orient='records')


def get_clinic_summary(health_df_period: pd.DataFrame, source_context: str = "ClinicSummary") -> Dict[str, Any]:
    logger.info(f"({source_context}) Calculating clinic summary.")
    summary: Dict[str, Any] = {
        "overall_avg_test_turnaround_conclusive_days": np.nan, "perc_critical_tests_tat_met": 0.0,
        "total_pending_critical_tests_patients": 0, "sample_rejection_rate_perc": 0.0,
        "key_drug_stockouts_count": 0, "test_summary_details": {}
    }
    if health_df_period is None or health_df_period.empty: logger.warning(f"({source_context}) No data for clinic summary."); return summary
    df_cs = health_df_period.copy()
    for col_cs_str in ['test_type','test_result','sample_status','item','zone_id']: df_cs[col_cs_str] = df_cs.get(col_cs_str, pd.Series(dtype=str)).fillna("Unknown").astype(str)
    df_cs['encounter_date'] = pd.to_datetime(df_cs.get('encounter_date'), errors='coerce')
    df_cs['test_turnaround_days'] = _convert_to_numeric(df_cs.get('test_turnaround_days'), np.nan)
    
    concl_df_cs = df_cs[~df_cs['test_result'].isin(['Pending','Rejected Sample','Unknown','N/A','nan','Indeterminate']) & df_cs['test_turnaround_days'].notna()].copy()
    if not concl_df_cs.empty and concl_df_cs['test_turnaround_days'].notna().any(): summary["overall_avg_test_turnaround_conclusive_days"] = concl_df_cs['test_turnaround_days'].mean()
    
    crit_keys_cs = app_config.CRITICAL_TESTS_LIST
    if crit_keys_cs:
        crit_concl_df_cs = concl_df_cs[concl_df_cs['test_type'].isin(crit_keys_cs)].copy()
        if not crit_concl_df_cs.empty:
            def _chk_tat_cs_fn(r_cs): cfg_s=app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get(r_cs['test_type']); return pd.notna(r_cs['test_turnaround_days']) and pd.notna(cfg_s.get('target_tat_days')) and r_cs['test_turnaround_days'] <= cfg_s['target_tat_days'] if cfg_s else False
            crit_concl_df_cs['tat_met_cs'] = crit_concl_df_cs.apply(_chk_tat_cs_fn,axis=1)
            if not crit_concl_df_cs['tat_met_cs'].empty : summary["perc_critical_tests_tat_met"]=(crit_concl_df_cs['tat_met_cs'].mean()*100)
        if 'patient_id' in df_cs.columns: summary["total_pending_critical_tests_patients"] = df_cs[(df_cs['test_type'].isin(crit_keys_cs)) & (df_cs['test_result']=='Pending')]['patient_id'].nunique()

    all_proc_samp_cs = df_cs[~df_cs['sample_status'].isin(['Pending','Unknown','N/A','nan'])].copy()
    if not all_proc_samp_cs.empty: summary["sample_rejection_rate_perc"] = (all_proc_samp_cs[all_proc_samp_cs['sample_status']=='Rejected'].shape[0]/len(all_proc_samp_cs))*100 if len(all_proc_samp_cs) >0 else 0.0
    
    test_sum_details_cs={}
    for o_key_cs,cfg_p_cs in app_config.KEY_TEST_TYPES_FOR_ANALYSIS.items():
        d_name_cs=cfg_p_cs.get("display_name",o_key_cs); actual_keys_cs=cfg_p_cs.get("types_in_group",[o_key_cs]) if isinstance(cfg_p_cs.get("types_in_group"),list) else [cfg_p_cs.get("types_in_group",o_key_cs)]
        grp_df_for_cs = df_cs[df_cs['test_type'].isin(actual_keys_cs)]
        stats_s_cs = {"positive_rate_perc":np.nan,"avg_tat_days":np.nan,"perc_met_tat_target":0.0,"pending_count_patients":0,"rejected_count_patients":0,"total_conclusive_tests":0}
        if grp_df_for_cs.empty: test_sum_details_cs[d_name_cs]=stats_s_cs; continue
        grp_c_sum_cs = grp_df_for_cs[~grp_df_for_cs['test_result'].isin(['Pending','Rejected Sample','Unknown','N/A','nan','Indeterminate']) & grp_df_for_cs['test_turnaround_days'].notna()].copy()
        stats_s_cs["total_conclusive_tests"]=len(grp_c_sum_cs)
        if not grp_c_sum_cs.empty:
            stats_s_cs["positive_rate_perc"]=(grp_c_sum_cs[grp_c_sum_cs['test_result']=='Positive'].shape[0]/len(grp_c_sum_cs))*100 if len(grp_c_sum_cs)>0 else 0.0
            if grp_c_sum_cs['test_turnaround_days'].notna().any():stats_s_cs["avg_tat_days"]=grp_c_sum_cs['test_turnaround_days'].mean()
            tgt_tat_spec_cs=cfg_p_cs.get("target_tat_days",app_config.TARGET_TEST_TURNAROUND_DAYS)
            grp_c_sum_cs.loc[:,'tat_met_s_cs']=grp_c_sum_cs['test_turnaround_days']<=tgt_tat_spec_cs if pd.notna(tgt_tat_spec_cs) else False
            if not grp_c_sum_cs['tat_met_s_cs'].empty: stats_s_cs["perc_met_tat_target"]=grp_c_sum_cs['tat_met_s_cs'].mean()*100
        if 'patient_id' in grp_df_for_cs.columns:
            if 'test_result' in grp_df_for_cs.columns: stats_s_cs["pending_count_patients"]=grp_df_for_cs[grp_df_for_cs['test_result']=='Pending']['patient_id'].nunique()
            if 'sample_status' in grp_df_for_cs.columns: stats_s_cs["rejected_count_patients"]=grp_df_for_cs[grp_df_for_cs['sample_status']=='Rejected']['patient_id'].nunique()
        test_sum_details_cs[d_name_cs]=stats_s_cs
    summary["test_summary_details"]=test_sum_details_cs

    if all(c in df_cs for c in ['item','item_stock_agg_zone','consumption_rate_per_day', 'encounter_date']) and app_config.KEY_DRUG_SUBSTRINGS_SUPPLY:
        key_drugs_df_sum_cs = df_cs[df_cs['item'].str.contains('|'.join(app_config.KEY_DRUG_SUBSTRINGS_SUPPLY), case=False, na=False)].copy()
        if not key_drugs_df_sum_cs.empty:
            key_drugs_df_sum_cs['encounter_date'] = pd.to_datetime(key_drugs_df_sum_cs['encounter_date'], errors='coerce')
            key_drugs_df_sum_cs.dropna(subset=['encounter_date'], inplace=True)
            if not key_drugs_df_sum_cs.empty:
                latest_key_supply_sum_cs = key_drugs_df_sum_cs.sort_values('encounter_date').drop_duplicates(subset=['item', 'zone_id'], keep='last') # zone_id important for stock scope
                latest_key_supply_sum_cs['consumption_rate_per_day'] = latest_key_supply_sum_cs['consumption_rate_per_day'].replace(0, np.nan)
                latest_key_supply_sum_cs['days_of_supply_calc_cs'] = latest_key_supply_sum_cs['item_stock_agg_zone'] / latest_key_supply_sum_cs['consumption_rate_per_day']
                summary['key_drug_stockouts_count'] = latest_key_supply_sum_cs[latest_key_supply_sum_cs['days_of_supply_calc_cs'] < app_config.CRITICAL_SUPPLY_DAYS_REMAINING]['item'].nunique()
    return summary

def get_clinic_environmental_summary(iot_df_period: pd.DataFrame, source_context: str = "ClinicEnvSummary") -> Dict[str, Any]:
    logger.info(f"({source_context}) Calculating clinic environmental summary.")
    summary_env: Dict[str, Any] = {"avg_co2_overall_ppm":np.nan, "rooms_co2_very_high_alert_latest_count":0, "avg_pm25_overall_ugm3":np.nan, "rooms_pm25_very_high_alert_latest_count":0, "avg_waiting_room_occupancy_persons":np.nan, "waiting_room_high_occupancy_alert_latest_flag":False, "avg_noise_overall_dba":np.nan, "rooms_noise_high_alert_latest_count":0}
    if iot_df_period is None or iot_df_period.empty or 'timestamp' not in iot_df_period.columns: logger.warning(f"({source_context}) No valid IoT data for env summary."); return summary_env
    df_iot_env_sum = iot_df_period.copy()
    df_iot_env_sum['timestamp'] = pd.to_datetime(df_iot_env_sum['timestamp'], errors='coerce') # Ensure datetime
    df_iot_env_sum.dropna(subset=['timestamp'], inplace=True)
    if df_iot_env_sum.empty : logger.warning(f"({source_context}) IoT data empty after timestamp cleaning for env summary."); return summary_env

    num_cols_env_sum = ['avg_co2_ppm','avg_pm25','waiting_room_occupancy','avg_noise_db']
    for col_env in num_cols_env_sum: df_iot_env_sum[col_env] = _convert_to_numeric(df_iot_env_sum.get(col_env), np.nan)
    
    if df_iot_env_sum['avg_co2_ppm'].notna().any(): summary_env["avg_co2_overall_ppm"] = df_iot_env_sum['avg_co2_ppm'].mean()
    if df_iot_env_sum['avg_pm25'].notna().any(): summary_env["avg_pm25_overall_ugm3"] = df_iot_env_sum['avg_pm25'].mean()
    if df_iot_env_sum.get('waiting_room_occupancy',pd.Series(dtype=float)).notna().any(): summary_env["avg_waiting_room_occupancy_persons"] = df_iot_env_sum['waiting_room_occupancy'][df_iot_env_sum.get('room_name','').str.contains('Waiting',case=False,na=False)].mean() # Avg only for waiting rooms
    if df_iot_env_sum['avg_noise_db'].notna().any(): summary_env["avg_noise_overall_dba"] = df_iot_env_sum['avg_noise_db'].mean() # Use dba in key for clarity
    
    if all(c in df_iot_env_sum for c in ['clinic_id','room_name','timestamp']):
        latest_reads_env_sum = df_iot_env_sum.sort_values('timestamp').drop_duplicates(subset=['clinic_id','room_name'], keep='last')
        if not latest_reads_env_sum.empty:
            if 'avg_co2_ppm' in latest_reads_env_sum and latest_reads_env_sum['avg_co2_ppm'].notna().any(): summary_env["rooms_co2_very_high_alert_latest_count"] = latest_reads_env_sum[latest_reads_env_sum['avg_co2_ppm'] > app_config.ALERT_AMBIENT_CO2_VERY_HIGH_PPM].shape[0]
            if 'avg_pm25' in latest_reads_env_sum and latest_reads_env_sum['avg_pm25'].notna().any(): summary_env["rooms_pm25_very_high_alert_latest_count"] = latest_reads_env_sum[latest_reads_env_sum['avg_pm25'] > app_config.ALERT_AMBIENT_PM25_VERY_HIGH_UGM3].shape[0]
            if 'waiting_room_occupancy' in latest_reads_env_sum and latest_reads_env_sum.get('room_name','').str.contains('Waiting',case=False,na=False).any(): # check specifically in rooms named "Waiting"
                 summary_env["waiting_room_high_occupancy_alert_latest_flag"] = (latest_reads_env_sum[latest_reads_env_sum.get('room_name','').str.contains('Waiting',case=False,na=False)]['waiting_room_occupancy'] > app_config.TARGET_CLINIC_WAITING_ROOM_OCCUPANCY_MAX).any()
            if 'avg_noise_db' in latest_reads_env_sum and latest_reads_env_sum['avg_noise_db'].notna().any(): summary_env["rooms_noise_high_alert_latest_count"] = latest_reads_env_sum[latest_reads_env_sum['avg_noise_db'] > app_config.ALERT_AMBIENT_NOISE_HIGH_DBA].shape[0]
    return summary_env


def get_patient_alerts_for_clinic(health_df_period: pd.DataFrame, risk_threshold_moderate: int = app_config.RISK_SCORE_MODERATE_THRESHOLD, source_context: str = "ClinicAlerts") -> pd.DataFrame:
    logger.info(f"({source_context}) Generating patient alerts for clinic.")
    if health_df_period is None or health_df_period.empty: return pd.DataFrame(columns=['patient_id', 'encounter_date', 'Alert Reason', 'Priority Score', 'Key Details']) # Return empty with schema
    
    # For clinic, we might have more complex rules or want a different alert structure than CHW direct alerts
    # Leverage the CHW alert generation for common criticals, then add clinic-specific logic if needed
    # The refactored get_patient_alerts_for_chw returns a list of dicts, convert to DataFrame
    base_alerts_list = get_patient_alerts_for_chw(health_df_period, source_context=f"{source_context}/BaseCHWLogic", risk_threshold_moderate=risk_threshold_moderate, risk_threshold_high=app_config.RISK_SCORE_HIGH_THRESHOLD) # pass thresholds
    
    if not base_alerts_list: return pd.DataFrame(columns=['patient_id', 'encounter_date', 'Alert Reason', 'Priority Score', 'Key Details'])
    
    alerts_df = pd.DataFrame(base_alerts_list)
    # Standardize column names if they came from get_patient_alerts_for_chw output directly
    rename_map_alerts = {
        'primary_reason': 'Alert Reason', 'raw_priority_score': 'Priority Score',
        'brief_details': 'Key Details', 'context_info': 'Context'
        # 'suggested_action_code' might also be useful for clinic staff
    }
    alerts_df.rename(columns=rename_map_alerts, inplace=True, errors='ignore')

    # Example clinic-specific addition: flag patients with multiple high-risk AI scores in the period
    if 'ai_risk_score' in health_df_period.columns and 'patient_id' in health_df_period.columns and 'encounter_date' in health_df_period.columns:
        high_risk_encs = health_df_period[health_df_period['ai_risk_score'] >= app_config.RISK_SCORE_HIGH_THRESHOLD]
        if not high_risk_encs.empty:
            repeated_high_risk_counts = high_risk_encs.groupby('patient_id')['encounter_date'].nunique()
            patients_with_repeated_high_risk = repeated_high_risk_counts[repeated_high_risk_counts > 2].index # e.g., >2 high risk encounters in period
            
            for pat_id_repeat in patients_with_repeated_high_risk:
                num_visits = repeated_high_risk_counts.get(pat_id_repeat,0)
                if pat_id_repeat in alerts_df['patient_id'].values: # If already alerted, enhance reason/priority
                    idx_existing_alert = alerts_df[alerts_df['patient_id'] == pat_id_repeat].index[0] # First alert for this patient
                    alerts_df.loc[idx_existing_alert, 'Alert Reason'] = alerts_df.loc[idx_existing_alert, 'Alert Reason'] + "; Repeated High Risk Encounters"
                    alerts_df.loc[idx_existing_alert, 'Priority Score'] = max(alerts_df.loc[idx_existing_alert, 'Priority Score'], 96) # Boost to very high
                    alerts_df.loc[idx_existing_alert, 'Key Details'] = (alerts_df.loc[idx_existing_alert, 'Key Details'] or "") + f" ({num_visits} high risk visits)"

                else: # Add new alert for this repeated risk if not already present
                    latest_rec_repeat = health_df_period[health_df_period['patient_id'] == pat_id_repeat].sort_values('encounter_date',ascending=False).iloc[0]
                    new_alert_entry = pd.DataFrame([{
                        'patient_id': pat_id_repeat, 'encounter_date': latest_rec_repeat['encounter_date'],
                        'condition': latest_rec_repeat.get('condition',"N/A"),
                        'Alert Reason': f"Repeated High Risk Encounters ({num_visits} visits)",
                        'Priority Score': 96, 'Key Details': f"Last AI Risk: {latest_rec_repeat.get('ai_risk_score', np.nan):.0f}",
                        'ai_risk_score': latest_rec_repeat.get('ai_risk_score', np.nan), # Include for consistency
                        # Add other relevant fields as needed for the clinic alert table
                    }])
                    alerts_df = pd.concat([alerts_df, new_alert_entry], ignore_index=True)

    # Ensure essential columns exist even after potential concat
    final_alert_cols = ['patient_id', 'encounter_date', 'Alert Reason', 'Priority Score', 'Key Details', 'ai_risk_score', 'condition', 'Context', 'suggested_action_code']
    for col_fa in final_alert_cols:
        if col_fa not in alerts_df.columns: alerts_df[col_fa] = pd.NA # Use pd.NA for generic missing

    return alerts_df.sort_values(by='Priority Score', ascending=False).reset_index(drop=True)


def get_district_summary_kpis(enriched_zone_gdf: Optional[gpd.GeoDataFrame], source_context: str = "DHOReportKPIs") -> Dict[str, Any]:
    logger.info(f"({source_context}) Calculating district summary KPIs.")
    kpis_dist: Dict[str, Any] = {
        "total_population_district": 0, "population_weighted_avg_ai_risk_score": np.nan,
        "zones_meeting_high_risk_criteria_count": 0, "district_avg_facility_coverage_score": np.nan,
        # Dynamically add for KEY_CONDITIONS_FOR_ACTION
        "district_overall_key_disease_prevalence_per_1000": np.nan,
        "district_population_weighted_avg_steps": np.nan, "district_avg_clinic_co2_ppm": np.nan
    }
    for cond_key_dist_kpi in app_config.KEY_CONDITIONS_FOR_ACTION: # Initialize keys for all actionable conditions
         kpis_dist[f"district_total_active_{cond_key_dist_kpi.lower().replace(' ', '_').replace('-', '_').replace('(severe)','')}_cases"] = 0

    if not isinstance(enriched_zone_gdf, gpd.GeoDataFrame) or enriched_zone_gdf.empty: logger.warning(f"({source_context}) Enriched GDF is empty or invalid."); return kpis_dist
    
    gdf_dist_kpi = enriched_zone_gdf.copy()
    num_cols_for_dist_kpi = ['population','avg_risk_score','total_active_key_infections','facility_coverage_score','avg_daily_steps_zone','zone_avg_co2']
    for cond_k in app_config.KEY_CONDITIONS_FOR_ACTION: num_cols_for_dist_kpi.append(f"active_{cond_k.lower().replace(' ', '_').replace('-', '_').replace('(severe)','')}_cases")
    for col_dist_num in num_cols_for_dist_kpi:
        gdf_dist_kpi[col_dist_num] = _convert_to_numeric(gdf_dist_kpi.get(col_dist_num, 0.0), np.nan if 'avg_' in col_dist_num or 'score' in col_dist_num else 0.0)

    if 'population' in gdf_dist_kpi.columns: kpis_dist["total_population_district"] = gdf_dist_kpi['population'].sum()
    
    total_pop_dist_kpi = kpis_dist["total_population_district"]
    if pd.notna(total_pop_dist_kpi) and total_pop_dist_kpi > 0:
        for metric_col_dist, kpi_key_dist in [
            ('avg_risk_score', 'population_weighted_avg_ai_risk_score'),
            ('facility_coverage_score', 'district_avg_facility_coverage_score'),
            ('avg_daily_steps_zone', 'district_population_weighted_avg_steps')]:
            if metric_col_dist in gdf_dist_kpi.columns and gdf_dist_kpi[metric_col_dist].notna().any() and 'population' in gdf_dist_kpi.columns:
                valid_weights_dist = gdf_dist_kpi.loc[gdf_dist_kpi[metric_col_dist].notna(), 'population'].fillna(0) # Fill NaN weights with 0
                valid_values_dist = gdf_dist_kpi.loc[gdf_dist_kpi[metric_col_dist].notna(), metric_col_dist]
                if not valid_values_dist.empty:
                    if valid_weights_dist.sum() > 0: kpis_dist[kpi_key_dist] = np.average(valid_values_dist, weights=valid_weights_dist)
                    else: kpis_dist[kpi_key_dist] = valid_values_dist.mean() # Unweighted mean if sum of weights is 0
                else: kpis_dist[kpi_key_dist] = np.nan # No valid values
            else: kpis_dist[kpi_key_dist] = np.nan # Metric col or population missing
        
        if 'total_active_key_infections' in gdf_dist_kpi.columns:
             kpis_dist["district_overall_key_disease_prevalence_per_1000"] = (gdf_dist_kpi['total_active_key_infections'].sum() / total_pop_dist_kpi) * 1000 if total_pop_dist_kpi > 0 else 0.0
    else: # Fallback to unweighted means if total population is zero or NaN
        logger.warning(f"({source_context}) District total population is zero or NaN. Using unweighted averages for some KPIs.")
        for metric_col_dist, kpi_key_dist in [('avg_risk_score','population_weighted_avg_ai_risk_score'),('facility_coverage_score','district_avg_facility_coverage_score'),('avg_daily_steps_zone','district_population_weighted_avg_steps')]:
            kpis_dist[kpi_key_dist] = gdf_dist_kpi[metric_col_dist].mean() if metric_col_dist in gdf_dist_kpi and gdf_dist_kpi[metric_col_dist].notna().any() else np.nan
        kpis_dist["district_overall_key_disease_prevalence_per_1000"] = np.nan # Cannot calculate without population

    if 'avg_risk_score' in gdf_dist_kpi.columns: kpis_dist["zones_meeting_high_risk_criteria_count"] = gdf_dist_kpi[gdf_dist_kpi['avg_risk_score'] >= app_config.DISTRICT_ZONE_HIGH_RISK_AVG_SCORE].shape[0]
    for cond_k_dist in app_config.KEY_CONDITIONS_FOR_ACTION:
        col_name_kpi_dist = f"active_{cond_k_dist.lower().replace(' ', '_').replace('-', '_').replace('(severe)','')}_cases"
        kpi_out_name = f"district_total_{col_name_kpi_dist}"
        kpis_dist[kpi_out_name] = int(gdf_dist_kpi.get(col_name_kpi_dist,0).sum()) if col_name_kpi_dist in gdf_dist_kpi else 0
        
    if 'zone_avg_co2' in gdf_dist_kpi and gdf_dist_kpi['zone_avg_co2'].notna().any(): kpis_dist["district_avg_clinic_co2_ppm"] = gdf_dist_kpi[gdf_dist_kpi['zone_avg_co2'] > 0]['zone_avg_co2'].mean() # Avg for zones with actual readings
    return kpis_dist


def get_trend_data(df: pd.DataFrame, value_col: str, date_col: str = 'encounter_date', period: str = 'D', agg_func: str = 'mean', filter_col: Optional[str] = None, filter_val: Optional[Any] = None, source_context: str = "TrendDataUtil") -> pd.Series:
    logger.debug(f"({source_context}) Generating trend for '{value_col}', period '{period}', agg '{agg_func}'.")
    if not isinstance(df, pd.DataFrame) or df.empty or date_col not in df.columns or value_col not in df.columns:
        logger.debug(f"({source_context}) Invalid input df or missing cols ('{date_col}', '{value_col}') for trend.")
        return pd.Series(dtype='float64')
    
    trend_df_calc = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(trend_df_calc[date_col]): trend_df_calc[date_col] = pd.to_datetime(trend_df_calc[date_col], errors='coerce')
    trend_df_calc.dropna(subset=[date_col], inplace=True)
    if agg_func not in ['nunique', 'count', 'size']: trend_df_calc.dropna(subset=[value_col], inplace=True) # Drop NaNs in value_col for most aggs
    if trend_df_calc.empty: logger.debug(f"({source_context}) DataFrame empty after date/NA handling for trend."); return pd.Series(dtype='float64')
    
    if filter_col and filter_col in trend_df_calc.columns and filter_val is not None:
        trend_df_calc = trend_df_calc[trend_df_calc[filter_col] == filter_val]
        if trend_df_calc.empty: logger.debug(f"({source_context}) DataFrame empty after filter '{filter_col}=={filter_val}'."); return pd.Series(dtype='float64')
    
    trend_df_calc.set_index(date_col, inplace=True)
    if agg_func in ['mean', 'sum', 'median', 'std', 'var'] and not pd.api.types.is_numeric_dtype(trend_df_calc[value_col]):
        trend_df_calc[value_col] = _convert_to_numeric(trend_df_calc[value_col], np.nan); trend_df_calc.dropna(subset=[value_col], inplace=True)
        if trend_df_calc.empty: logger.debug(f"({source_context}) DataFrame empty after numeric conversion for aggregation."); return pd.Series(dtype='float64')
    try:
        resampled_trend = trend_df_calc.groupby(pd.Grouper(freq=period)) # Use Grouper directly
        if agg_func == 'nunique': trend_series_out = resampled_trend[value_col].nunique()
        elif agg_func == 'sum': trend_series_out = resampled_trend[value_col].sum()
        elif agg_func == 'median': trend_series_out = resampled_trend[value_col].median()
        elif agg_func == 'count': trend_series_out = resampled_trend[value_col].count() # Counts non-NA values
        elif agg_func == 'size': trend_series_out = resampled_trend.size() # Counts all rows in group
        elif agg_func == 'std': trend_series_out = resampled_trend[value_col].std()
        elif agg_func == 'var': trend_series_out = resampled_trend[value_col].var()
        else: trend_series_out = resampled_trend[value_col].mean() # Default
    except Exception as e_trend: logger.error(f"({source_context}) Trend resampling error for {value_col} ({agg_func}): {e_trend}", exc_info=True); return pd.Series(dtype='float64')
    return trend_series_out


def get_supply_forecast_data(health_df: pd.DataFrame, forecast_days_out: int = 30, item_filter_list: Optional[List[str]] = None, source_context: str = "SupplyForecastLinear") -> pd.DataFrame:
    logger.info(f"({source_context}) Calculating simple linear supply forecast for {forecast_days_out} days.")
    default_cols_fc_linear = ['item', 'date', 'initial_stock_at_forecast_start', 'base_consumption_rate_per_day', 'forecasted_stock_level', 'forecasted_days_of_supply', 'estimated_stockout_date_linear', 'lower_ci_days_supply', 'upper_ci_days_supply', 'initial_days_supply_at_forecast_start']
    req_cols_fc_linear = ['item', 'encounter_date', 'item_stock_agg_zone', 'consumption_rate_per_day']
    if health_df is None or health_df.empty or not all(c in health_df.columns for c in req_cols_fc_linear):
        logger.warning(f"({source_context}) Missing required columns for linear supply forecast: {req_cols_fc_linear}")
        return pd.DataFrame(columns=default_cols_fc_linear)

    df_fc_linear_src = health_df.copy()
    df_fc_linear_src['encounter_date'] = pd.to_datetime(df_fc_linear_src['encounter_date'], errors='coerce')
    df_fc_linear_src.dropna(subset=['encounter_date', 'item'], inplace=True)
    df_fc_linear_src['item_stock_agg_zone'] = _convert_to_numeric(df_fc_linear_src.get('item_stock_agg_zone'), 0.0)
    df_fc_linear_src['consumption_rate_per_day'] = _convert_to_numeric(df_fc_linear_src.get('consumption_rate_per_day'), 0.0001) # Avoid zero for rate if possible
    if df_fc_linear_src.empty: return pd.DataFrame(columns=default_cols_fc_linear)

    latest_status_fc_linear = df_fc_linear_src.sort_values('encounter_date').drop_duplicates(subset=['item'], keep='last') # Assuming clinic-wide item tracking
    if item_filter_list: latest_status_fc_linear = latest_status_fc_linear[latest_status_fc_linear['item'].isin(item_filter_list)]
    if latest_status_fc_linear.empty: return pd.DataFrame(columns=default_cols_fc_linear)
    
    forecasts_linear_list = []
    consumption_std_dev_factor = 0.25 # For very simple CI estimation, e.g. +/- 25%
    for _, row_fc_item in latest_status_fc_linear.iterrows():
        item_fc_lin, stock_fc_lin, cons_rate_fc_lin, last_dt_fc_lin = row_fc_item['item'], row_fc_item.get('item_stock_agg_zone',0.0), row_fc_item.get('consumption_rate_per_day',0.0001), row_fc_item['encounter_date']
        if pd.isna(stock_fc_lin) or stock_fc_lin < 0: stock_fc_lin = 0.0
        cons_rate_fc_lin = max(0.0001, cons_rate_fc_lin) # Ensure minimal consumption rate > 0
        
        fc_dates_linear = pd.date_range(start=last_dt_fc_lin + pd.Timedelta(days=1), periods=forecast_days_out, freq='D')
        init_dos_fc_lin = stock_fc_lin / cons_rate_fc_lin if cons_rate_fc_lin > 0 else (np.inf if stock_fc_lin > 0 else 0)
        est_stockout_dt_fc_lin = last_dt_fc_lin + pd.to_timedelta(init_dos_fc_lin, unit='D') if np.isfinite(init_dos_fc_lin) else pd.NaT
        
        for day_idx_fc_lin, current_fc_date_lin in enumerate(fc_dates_linear):
            days_out_fc_lin = day_idx_fc_lin + 1
            fc_stock_level_lin = max(0, stock_fc_lin - (cons_rate_fc_lin * days_out_fc_lin))
            fc_days_supply_lin = fc_stock_level_lin / cons_rate_fc_lin if cons_rate_fc_lin > 0 else (np.inf if fc_stock_level_lin > 0 else 0)
            
            # Simple CI based on consumption rate variation
            cons_rate_ci_lower = max(0.0001, cons_rate_fc_lin * (1 - consumption_std_dev_factor))
            cons_rate_ci_upper = cons_rate_fc_lin * (1 + consumption_std_dev_factor)
            stock_at_high_cons_rate = max(0, stock_fc_lin - (cons_rate_ci_upper * days_out_fc_lin))
            days_supply_at_high_cons = stock_at_high_cons_rate / cons_rate_ci_upper if cons_rate_ci_upper > 0 else (np.inf if stock_at_high_cons_rate > 0 else 0) # This is lower bound of days supply
            stock_at_low_cons_rate = max(0, stock_fc_lin - (cons_rate_ci_lower * days_out_fc_lin))
            days_supply_at_low_cons = stock_at_low_cons_rate / cons_rate_ci_lower if cons_rate_ci_lower > 0 else (np.inf if stock_at_low_cons_rate > 0 else 0) # This is upper bound of days supply
            
            forecasts_linear_list.append({
                'item': item_fc_lin, 'date': current_fc_date_lin,
                'initial_stock_at_forecast_start': stock_fc_lin,
                'base_consumption_rate_per_day': cons_rate_fc_lin,
                'forecasted_stock_level': fc_stock_level_lin,
                'forecasted_days_of_supply': fc_days_supply_lin,
                'estimated_stockout_date_linear': est_stockout_dt_fc_lin,
                'lower_ci_days_supply': days_supply_at_high_cons,
                'upper_ci_days_supply': days_supply_at_low_cons,
                'initial_days_supply_at_forecast_start': init_dos_fc_lin
            })
    if not forecasts_linear_list: return pd.DataFrame(columns=default_cols_fc_linear)
    return pd.DataFrame(forecasts_linear_list)
