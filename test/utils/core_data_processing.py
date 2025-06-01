# sentinel_project_root/test/utils/core_data_processing.py
# Core data loading, cleaning, and aggregation utilities for Sentinel Health Co-Pilot.

import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import os
import logging
from config import app_config
from typing import List, Dict, Any, Optional, Tuple, Union # Added Union
from datetime import datetime, date, timedelta # For date manipulation

logger = logging.getLogger(__name__)

# --- I. Core Helper Functions ---
def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        logger.error(f"_clean_column_names expects a pandas DataFrame, got {type(df)}.")
        return df if df is not None else pd.DataFrame()
    df.columns = df.columns.str.lower().str.replace('[^0-9a-zA-Z_]', '_', regex=True).str.replace('_+', '_', regex=True).str.strip('_')
    return df

def _convert_to_numeric(series: pd.Series, default_value: Any = np.nan) -> pd.Series:
    if not isinstance(series, pd.Series):
        logger.debug(f"_convert_to_numeric given non-Series type: {type(series)}. Attempting conversion.")
        try:
            # Attempt to create a series, inferring dtype if default_value is numeric, else object
            dtype_to_use = float if pd.api.types.is_number(default_value) or default_value is np.nan else object
            series = pd.Series(series, dtype=dtype_to_use)
        except Exception as e_series:
            logger.error(f"Could not convert input of type {type(series)} to Series in _convert_to_numeric: {e_series}")
            length = len(series) if hasattr(series, '__len__') else 1
            return pd.Series([default_value] * length, dtype=type(default_value) if default_value is not np.nan else float)
    return pd.to_numeric(series, errors='coerce').fillna(default_value)


def hash_geodataframe(gdf: Optional[gpd.GeoDataFrame]) -> Optional[str]:
    if not isinstance(gdf, gpd.GeoDataFrame) or gdf is None:
        return None
    if gdf.empty:
        return "empty_gdf" # Consistent hash for empty GDF
    try:
        # Hash non-geometry part first for stability
        df_part = pd.DataFrame(gdf.drop(columns=gdf.geometry.name, errors='ignore'))
        # Convert datetime types to integer (nanoseconds) for consistent hashing
        for col in df_part.select_dtypes(include=['datetime64[ns]', 'datetime64', 'datetimetz']).columns:
            df_part[col] = pd.to_datetime(df_part[col], errors='coerce').astype('int64')
        
        df_hash = pd.util.hash_pandas_object(df_part, index=True).sum()
        
        # Hash geometry part (WKT representation is standard)
        # Ensure valid geometries before converting to WKT
        geom_wkt_series = gdf.geometry.apply(lambda geom: geom.wkt if geom and geom.is_valid else None)
        geom_hash = pd.util.hash_array(geom_wkt_series.fillna('').values).sum()
        
        return f"{df_hash}-{geom_hash}"
    except Exception as e:
        logger.error(f"Robust Hashing GeoDataFrame failed: {e}", exc_info=True)
        # Fallback: hash based on shape and head to avoid full data string for large GDFs
        return f"fallback-{gdf.shape}-{pd.util.hash_pandas_object(gdf.head(1).astype(str), index=True).sum()}"


def _robust_merge_agg(
    left_df: pd.DataFrame, right_df: Optional[pd.DataFrame], target_col_name: str,
    on_col: str = 'zone_id', default_fill_value: Any = 0.0
) -> pd.DataFrame:
    """ Robustly merges aggregated right_df into left_df. """
    if not isinstance(left_df, pd.DataFrame):
        logger.error(f"Left df in _robust_merge_agg is not a DataFrame: {type(left_df)}")
        return pd.DataFrame(columns=[on_col, target_col_name]) if left_df is None else left_df.copy()

    # Ensure target column exists in left_df, filled with default
    if target_col_name not in left_df.columns:
        left_df[target_col_name] = default_fill_value
    else:
        left_df[target_col_name] = left_df[target_col_name].fillna(default_fill_value)

    if not isinstance(right_df, pd.DataFrame) or right_df.empty or on_col not in right_df.columns:
        logger.debug(f"Right_df for '{target_col_name}' is empty or missing '{on_col}'. Left_df returned with default for '{target_col_name}'.")
        return left_df

    # Identify the value column in right_df (assuming it's the one not 'on_col')
    value_col_candidates = [col for col in right_df.columns if col != on_col]
    if not value_col_candidates:
        logger.debug(f"No value column found in right_df for '{target_col_name}'. Left_df returned.")
        return left_df
    value_col_in_right = value_col_candidates[0] # Take the first one

    # Prepare for merge: copy, ensure compatible 'on_col' types (string for safety)
    left_df_merged = left_df.copy()
    try:
        left_df_merged[on_col] = left_df_merged[on_col].astype(str).str.strip()
        right_df_prepared = right_df[[on_col, value_col_in_right]].copy()
        right_df_prepared[on_col] = right_df_prepared[on_col].astype(str).str.strip()
    except KeyError:
        logger.error(f"KeyError: '{on_col}' not found in left_df during type conversion for merging '{target_col_name}'. Returning original left_df.")
        return left_df # Or handle more gracefully if appropriate
    except Exception as e:
        logger.error(f"Type conversion error for '{on_col}' in _robust_merge_agg targeting '{target_col_name}': {e}. Returning original left_df.")
        return left_df

    # Use a temporary unique column name for the merge to avoid conflicts
    temp_agg_col = f"__temp_agg_{target_col_name}_{os.urandom(4).hex()}__"
    right_df_prepared.rename(columns={value_col_in_right: temp_agg_col}, inplace=True)

    # Preserve original index of left_df
    original_index = left_df_merged.index
    original_index_name = left_df_merged.index.name

    # Reset index for merge if it's not a simple RangeIndex or has a name
    reset_index_for_merge = not isinstance(original_index, pd.RangeIndex) or original_index_name is not None
    if reset_index_for_merge:
        left_df_merged = left_df_merged.reset_index()
    
    # Perform the merge
    left_df_merged = left_df_merged.merge(right_df_prepared, on=on_col, how='left')

    # Update the target column: use merged value if it exists (notna), otherwise keep original value
    if temp_agg_col in left_df_merged.columns:
        # `combine_first` is good here: it takes value from temp_agg_col if not NaN, else from original target_col_name
        left_df_merged[target_col_name] = left_df_merged[temp_agg_col].combine_first(left_df_merged[target_col_name])
        left_df_merged.drop(columns=[temp_agg_col], inplace=True)
    
    # Ensure final fill with default value for any NaNs that might remain or were introduced
    left_df_merged[target_col_name].fillna(default_fill_value, inplace=True)

    # Restore original index if it was reset
    if reset_index_for_merge:
        index_col_to_set = original_index_name if original_index_name else 'index' # Default name if unnamed
        if index_col_to_set in left_df_merged.columns:
            left_df_merged.set_index(index_col_to_set, inplace=True, drop=True)
            left_df_merged.index.name = original_index_name # Restore original index name
        else: # Should not happen if logic is correct, but as a safeguard:
            logger.warning(f"Original index column '{index_col_to_set}' lost during merge for '{target_col_name}'. Attempting to reassign original index if lengths match.")
            if len(left_df_merged) == len(original_index):
                left_df_merged.index = original_index
            else:
                logger.error(f"Could not restore original index for '{target_col_name}' due to length mismatch. Index may be incorrect.")
                
    return left_df_merged

# --- II. Data Loading and Basic Cleaning Functions ---
@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS_WEB_REPORTS, show_spinner="Loading health records dataset...")
def load_health_records(file_path: Optional[str] = None, source_context: str = "DataLoader") -> pd.DataFrame:
    actual_file_path = file_path or app_config.HEALTH_RECORDS_CSV
    logger.info(f"({source_context}) Loading health records from: {actual_file_path}")
    if not os.path.exists(actual_file_path):
        logger.error(f"({source_context}) Health records file not found: {actual_file_path}")
        return pd.DataFrame() # Return empty DF, let caller handle
    try:
        df = pd.read_csv(actual_file_path, low_memory=False)
        df = _clean_column_names(df) # Standardize column names
        logger.info(f"({source_context}) Loaded {len(df)} raw records. Columns: {df.columns.tolist()}")

        # Date conversions
        date_cols = ['encounter_date', 'sample_collection_date', 'sample_registered_lab_date', 
                     'referral_date', 'referral_outcome_date'] # Assuming 'referral_date' might exist
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            else: # Add if missing, as NaT
                df[col] = pd.NaT 
                logger.debug(f"({source_context}) Date column '{col}' not found, added as NaT.")
        
        if 'encounter_date' in df.columns: # For convenience in filtering
             df['encounter_date_obj'] = df['encounter_date'].dt.date


        # Numeric conversions with appropriate defaults
        numeric_cols_defaults = {
            # Vitals & Sensor Data
            'age': np.nan, 'hrv_rmssd_ms': np.nan, 'min_spo2_pct': np.nan, 
            'vital_signs_temperature_celsius': np.nan, 'max_skin_temp_celsius': np.nan,
            'movement_activity_level': np.nan, 'fall_detected_today': 0, # fall is a count or flag
            'ambient_heat_index_c': np.nan, 'rapid_psychometric_distress_score': np.nan,
            'vital_signs_bp_systolic': np.nan, 'vital_signs_bp_diastolic': np.nan,
            'avg_spo2': np.nan, 'avg_daily_steps': np.nan, 'resting_heart_rate': np.nan,
            'avg_sleep_duration_hrs': np.nan, 'sleep_score_pct': np.nan, 'stress_level_score': np.nan,
            'hiv_viral_load_copies_ml': np.nan,
            # Flags (typically 0 or 1)
            'pregnancy_status': 0, 'chronic_condition_flag': 0, 'ppe_compliant_flag': 1, # Default to compliant
            'signs_of_fatigue_observed_flag': 0, 'chw_visit': 0, 'tb_contact_traced': 0,
            # Test & Supply related numerics
            'test_turnaround_days': np.nan, 'quantity_dispensed': 0, 
            'item_stock_agg_zone': 0, 'consumption_rate_per_day': 0.0,
            # AI Scores (will be filled by AI engine, but ensure numeric type)
            'ai_risk_score': np.nan, 'ai_followup_priority_score': np.nan,
            # Geo
            'patient_latitude': np.nan, 'patient_longitude': np.nan
        }
        for col, default in numeric_cols_defaults.items():
            if col in df.columns:
                df[col] = _convert_to_numeric(df[col], default_value=default)
            else: # Add if missing
                df[col] = default
                logger.debug(f"({source_context}) Numeric column '{col}' not found, added with default: {default}.")
        
        # String column cleaning (fill NA with "Unknown", strip, normalize common NA strings)
        string_cols = [
            'encounter_id', 'patient_id', 'encounter_type', 'gender', 'zone_id', 'clinic_id', 'chw_id',
            'condition', 'patient_reported_symptoms', 'test_type', 'test_result', 
            'referral_status', 'referral_reason', 'referred_to_facility_id', 'referral_outcome',
            'medication_adherence_self_report', 'item', 'notes', 'diagnosis_code_icd10', 
            'physician_id', 'screening_hpv_status', 'key_chronic_conditions_summary',
            'sample_status', 'rejection_reason'
        ]
        common_na_strings = ['', 'nan', 'None', 'N/A', '#N/A', 'np.nan', 'NaT', '<NA>', 'null'] # Expanded list
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().replace(common_na_strings, "Unknown", regex=False)
                df[col] = df[col].fillna("Unknown") # Catch any remaining NaNs after astype(str)
            else: # Add if missing
                df[col] = "Unknown"
                logger.debug(f"({source_context}) String column '{col}' not found, added as 'Unknown'.")

        logger.info(f"({source_context}) Health records cleaning complete. Shape: {df.shape}")
        return df
    except FileNotFoundError: # Already handled, but good practice for read_csv
        logger.error(f"({source_context}) File not found: {actual_file_path} (should have been caught by os.path.exists).")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"({source_context}) Error loading/processing health records from {actual_file_path}: {e}", exc_info=True)
        return pd.DataFrame()

@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS_WEB_REPORTS, show_spinner="Loading IoT environmental dataset...")
def load_iot_clinic_environment_data(file_path: Optional[str] = None, source_context: str = "DataLoader") -> pd.DataFrame:
    actual_file_path = file_path or app_config.IOT_CLINIC_ENVIRONMENT_CSV
    logger.info(f"({source_context}) Loading IoT data from: {actual_file_path}")
    if not os.path.exists(actual_file_path):
        logger.warning(f"({source_context}) IoT data file not found: {actual_file_path}.")
        return pd.DataFrame()
    try:
        df = pd.read_csv(actual_file_path, low_memory=False)
        df = _clean_column_names(df)
        logger.info(f"({source_context}) Loaded {len(df)} IoT records. Columns: {df.columns.tolist()}")

        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        else:
            logger.error(f"({source_context}) IoT data missing critical 'timestamp' column. Returning empty DataFrame.")
            return pd.DataFrame()

        numeric_iot_cols = ['avg_co2_ppm', 'max_co2_ppm', 'avg_pm25', 'voc_index', 
                            'avg_temp_celsius', 'avg_humidity_rh', 'avg_noise_db',
                            'waiting_room_occupancy', 'patient_throughput_per_hour', 
                            'sanitizer_dispenses_per_hour']
        for col in numeric_iot_cols:
            if col in df.columns:
                df[col] = _convert_to_numeric(df[col], default_value=np.nan)
            else:
                df[col] = np.nan
                logger.debug(f"({source_context}) IoT numeric column '{col}' not found, added as NaN.")
        
        string_iot_cols = ['clinic_id', 'room_name', 'zone_id']
        common_na_strings = ['', 'nan', 'None', 'N/A', '#N/A', 'np.nan', 'NaT', '<NA>', 'null']
        for col in string_iot_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().replace(common_na_strings, "Unknown", regex=False)
                df[col] = df[col].fillna("Unknown")
            else:
                df[col] = "Unknown"
                logger.debug(f"({source_context}) IoT string column '{col}' not found, added as 'Unknown'.")
        
        logger.info(f"({source_context}) IoT data cleaning complete. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"({source_context}) Error loading/processing IoT data from {actual_file_path}: {e}", exc_info=True)
        return pd.DataFrame()

@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS_WEB_REPORTS, hash_funcs={gpd.GeoDataFrame: hash_geodataframe}, show_spinner="Loading zone geographic & attribute dataset...")
def load_zone_data(attributes_path: Optional[str] = None, geometries_path: Optional[str] = None, source_context: str = "DataLoader") -> Optional[gpd.GeoDataFrame]:
    attr_path = attributes_path or app_config.ZONE_ATTRIBUTES_CSV
    geom_path = geometries_path or app_config.ZONE_GEOMETRIES_GEOJSON
    logger.info(f"({source_context}) Loading zone attributes from '{attr_path}' and geometries from '{geom_path}'")

    if not os.path.exists(attr_path) or not os.path.exists(geom_path):
        missing_files = []
        if not os.path.exists(attr_path): missing_files.append(f"Attributes: '{os.path.basename(attr_path)}'")
        if not os.path.exists(geom_path): missing_files.append(f"Geometries: '{os.path.basename(geom_path)}'")
        logger.error(f"({source_context}) Zone data file(s) not found: {', '.join(missing_files)}")
        return None
    
    try:
        attrs_df = pd.read_csv(attr_path); attrs_df = _clean_column_names(attrs_df)
        geoms_gdf = gpd.read_file(geom_path); geoms_gdf = _clean_column_names(geoms_gdf)

        for df_check, name_check in [(attrs_df, "attributes"), (geoms_gdf, "geometries")]:
            if 'zone_id' not in df_check.columns:
                logger.error(f"({source_context}) 'zone_id' missing in zone {name_check}. Cannot merge.")
                return None
            df_check['zone_id'] = df_check['zone_id'].astype(str).str.strip() # Standardize merge key

        # Standardize 'name' column from attributes if it's named differently
        if 'zone_display_name' in attrs_df.columns and 'name' not in attrs_df.columns:
            attrs_df.rename(columns={'zone_display_name': 'name'}, inplace=True)
        elif 'name' not in attrs_df.columns and 'zone_id' in attrs_df.columns: # Fallback if no name column
            attrs_df['name'] = "Zone " + attrs_df['zone_id']
            
        original_geom_col_name = geoms_gdf.geometry.name # Store original geometry column name

        # Merge, ensuring suffixes handle any overlaps beyond zone_id
        merged_gdf = geoms_gdf.merge(attrs_df, on="zone_id", how="left", suffixes=('_geom', '_attr'))

        # Resolve column conflicts: prioritize attribute versions if both exist
        for col_name in attrs_df.columns:
            if col_name == 'zone_id': continue # Skip merge key
            attr_suffixed_col = f"{col_name}_attr"
            geom_suffixed_col = f"{col_name}_geom"
            
            if attr_suffixed_col in merged_gdf.columns: # Attribute version exists
                # Fill NaNs in attr version with geom version if geom version exists
                merged_gdf[col_name] = merged_gdf[attr_suffixed_col].fillna(merged_gdf.get(geom_suffixed_col))
                cols_to_drop = [attr_suffixed_col]
                if geom_suffixed_col in merged_gdf.columns: cols_to_drop.append(geom_suffixed_col)
                merged_gdf.drop(columns=cols_to_drop, inplace=True)
            elif geom_suffixed_col in merged_gdf.columns and col_name not in merged_gdf.columns: # Only geom version exists (unlikely if attrs_df has it)
                 merged_gdf.rename(columns={geom_suffixed_col: col_name}, inplace=True)


        # Ensure 'geometry' is the active geometry column
        if merged_gdf.geometry.name != 'geometry':
            if 'geometry' in merged_gdf.columns: # A column named 'geometry' exists but isn't active
                merged_gdf = merged_gdf.set_geometry('geometry', drop=False, inplace=False)
            elif original_geom_col_name in merged_gdf.columns: # Original geometry column still exists
                merged_gdf = merged_gdf.set_geometry(original_geom_col_name, drop=False, inplace=False)
            else: # This case should be rare if loading was successful
                logger.error(f"({source_context}) Could not determine or set active geometry column for GDF.")
                return None

        # CRS Handling
        if merged_gdf.crs is None:
            merged_gdf = merged_gdf.set_crs(app_config.DEFAULT_CRS_STANDARD, allow_override=True)
            logger.info(f"({source_context}) GDF CRS was None, set to default: {app_config.DEFAULT_CRS_STANDARD}")
        elif str(merged_gdf.crs).upper() != app_config.DEFAULT_CRS_STANDARD.upper():
            logger.info(f"({source_context}) Reprojecting GDF from {merged_gdf.crs} to {app_config.DEFAULT_CRS_STANDARD}")
            merged_gdf = merged_gdf.to_crs(app_config.DEFAULT_CRS_STANDARD)

        # Ensure default zone attribute columns exist and have correct types
        default_numeric_zone_cols = {'population': 0.0, 'num_clinics': 0, 'socio_economic_index': 0.5, 'avg_travel_time_clinic_min': 30.0, 'area_sqkm': np.nan}
        for col, default_val in default_numeric_zone_cols.items():
            if col not in merged_gdf.columns: merged_gdf[col] = default_val
            merged_gdf[col] = _convert_to_numeric(merged_gdf.get(col), default_value=default_val)
        
        if 'name' not in merged_gdf.columns or merged_gdf['name'].isnull().all():
             merged_gdf['name'] = "Zone " + merged_gdf['zone_id'].astype(str)
        else:
            merged_gdf['name'] = merged_gdf['name'].astype(str).fillna("Zone " + merged_gdf['zone_id'].astype(str))

        # Ensure other Sentinel-specific attributes exist
        sentinel_attrs = ['predominant_hazard_type', 'typical_workforce_exposure_level', 'primary_livelihood', 'water_source_main']
        for attr in sentinel_attrs:
            if attr not in merged_gdf.columns: merged_gdf[attr] = "Unknown"
            else: merged_gdf[attr] = merged_gdf[attr].astype(str).fillna("Unknown")
            
        logger.info(f"({source_context}) Zone data loaded and merged: {len(merged_gdf)} zones. CRS: {merged_gdf.crs}. Columns: {merged_gdf.columns.tolist()}")
        return merged_gdf
    except Exception as e:
        logger.error(f"({source_context}) Error loading/merging zone data: {e}", exc_info=True)
        return None


# --- III. Data Enrichment and Aggregation Functions ---
@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS_WEB_REPORTS, hash_funcs={gpd.GeoDataFrame: hash_geodataframe}, show_spinner="Enriching zone geographic data with health aggregates...")
def enrich_zone_geodata_with_health_aggregates(
    zone_gdf: Optional[gpd.GeoDataFrame],
    health_df: Optional[pd.DataFrame],
    iot_df: Optional[pd.DataFrame] = None,
    source_context: str = "DataEnricher"
) -> Optional[gpd.GeoDataFrame]:
    logger.info(f"({source_context}) Starting zone GeoDataFrame enrichment process.")
    if not isinstance(zone_gdf, gpd.GeoDataFrame) or zone_gdf.empty or 'zone_id' not in zone_gdf.columns:
        logger.warning(f"({source_context}) Invalid or empty zone_gdf for enrichment. Cannot proceed.")
        return zone_gdf # Return original or None if it was None

    enriched_gdf = zone_gdf.copy() # Work on a copy

    # Ensure base population column exists and is numeric for calculations
    if 'population' not in enriched_gdf.columns:
        enriched_gdf['population'] = 0.0
        logger.warning(f"({source_context}) 'population' column missing in GDF, added with 0.0.")
    enriched_gdf['population'] = _convert_to_numeric(enriched_gdf['population'], 0.0)

    # Initialize all aggregation columns to ensure they exist
    agg_cols_to_init = {
        'total_population_health_data': 0, 'avg_risk_score': np.nan, 'total_patient_encounters': 0,
        'total_referrals_made': 0, 'successful_referrals': 0,
        'avg_test_turnaround_critical': np.nan, 'perc_critical_tests_tat_met': 0.0,
        'total_active_key_infections': 0, 'prevalence_per_1000': 0.0,
        'avg_daily_steps_zone': np.nan, 'zone_avg_co2': np.nan,
        'facility_coverage_score': 0.0, 'population_density': np.nan
    }
    for cond_key in app_config.KEY_CONDITIONS_FOR_ACTION:
        agg_cols_to_init[f"active_{cond_key.lower().replace(' ', '_').replace('-', '_').replace('(severe)','')}_cases"] = 0
    
    for col, default_val_init in agg_cols_to_init.items():
        if col not in enriched_gdf.columns:
            enriched_gdf[col] = default_val_init

    # Process Health Data Aggregations
    if isinstance(health_df, pd.DataFrame) and not health_df.empty and 'zone_id' in health_df.columns:
        health_df_agg = health_df.copy()
        health_df_agg['zone_id'] = health_df_agg['zone_id'].astype(str).str.strip()

        enriched_gdf = _robust_merge_agg(enriched_gdf, health_df_agg.groupby('zone_id')['patient_id'].nunique().reset_index(name='count'), 'total_population_health_data', default_fill_value=0)
        enriched_gdf = _robust_merge_agg(enriched_gdf, health_df_agg.groupby('zone_id')['ai_risk_score'].mean().reset_index(name='mean_val'), 'avg_risk_score', default_fill_value=np.nan)
        enriched_gdf = _robust_merge_agg(enriched_gdf, health_df_agg.groupby('zone_id')['encounter_id'].nunique().reset_index(name='count'), 'total_patient_encounters', default_fill_value=0)

        for condition_name in app_config.KEY_CONDITIONS_FOR_ACTION:
            col_name_cond = f"active_{condition_name.lower().replace(' ', '_').replace('-', '_').replace('(severe)','')}_cases"
            condition_filter_mask = health_df_agg.get('condition', pd.Series(dtype=str)).str.contains(condition_name, case=False, na=False)
            if condition_filter_mask.any(): # Only aggregate if any records match
                agg_data_cond = health_df_agg[condition_filter_mask].groupby('zone_id')['patient_id'].nunique().reset_index(name='count')
                enriched_gdf = _robust_merge_agg(enriched_gdf, agg_data_cond, col_name_cond, default_fill_value=0)
            else: # Ensure column exists with 0 if no cases
                 enriched_gdf[col_name_cond] = enriched_gdf.get(col_name_cond, 0.0).fillna(0.0)

        actionable_condition_cols_list = [f"active_{c.lower().replace(' ', '_').replace('-', '_').replace('(severe)','')}_cases" for c in app_config.KEY_CONDITIONS_FOR_ACTION if f"active_{c.lower().replace(' ', '_').replace('-', '_').replace('(severe)','')}_cases" in enriched_gdf.columns]
        if actionable_condition_cols_list: # Check if list is not empty
            enriched_gdf['total_active_key_infections'] = enriched_gdf[actionable_condition_cols_list].sum(axis=1, skipna=True).fillna(0)

        # Referral Aggregations
        if 'referral_status' in health_df_agg.columns:
            made_refs_df = health_df_agg[health_df_agg['referral_status'].notna() & (~health_df_agg['referral_status'].isin(['N/A', 'Unknown', 'No Referral']))] # Count if status is anything other than explicitly no referral
            enriched_gdf = _robust_merge_agg(enriched_gdf, made_refs_df.groupby('zone_id')['encounter_id'].nunique().reset_index(name='count'), 'total_referrals_made', default_fill_value=0)
            if 'referral_outcome' in health_df_agg.columns:
                successful_outcomes = ['completed', 'service provided', 'attended consult', 'attended followup', 'attended', 'admitted'] # Added 'admitted'
                successful_refs_df = health_df_agg[health_df_agg.get('referral_outcome', pd.Series(dtype=str)).str.lower().isin(successful_outcomes)]
                enriched_gdf = _robust_merge_agg(enriched_gdf, successful_refs_df.groupby('zone_id')['encounter_id'].nunique().reset_index(name='count'), 'successful_referrals', default_fill_value=0)
        
        # Critical Test TAT Aggregations
        crit_tests = app_config.CRITICAL_TESTS_LIST
        if crit_tests and all(c in health_df_agg.columns for c in ['test_type', 'test_turnaround_days', 'test_result']):
            tat_analysis_df = health_df_agg[
                (health_df_agg['test_type'].isin(crit_tests)) &
                (health_df_agg['test_turnaround_days'].notna()) &
                (~health_df_agg.get('test_result', pd.Series(dtype=str)).isin(['Pending', 'Rejected Sample', 'Unknown', 'Indeterminate']))
            ].copy() # Conclusive critical tests
            if not tat_analysis_df.empty:
                enriched_gdf = _robust_merge_agg(enriched_gdf, tat_analysis_df.groupby('zone_id')['test_turnaround_days'].mean().reset_index(name='mean_val'), 'avg_test_turnaround_critical', default_fill_value=np.nan)
                
                def _check_tat_met_for_enrich(row_tat_check): # Closure for app_config access
                    config_for_test = app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get(row_tat_check['test_type'])
                    target_days = config_for_test.get('target_tat_days', app_config.TARGET_TEST_TURNAROUND_DAYS) if config_for_test else app_config.TARGET_TEST_TURNAROUND_DAYS
                    return pd.notna(row_tat_check['test_turnaround_days']) and pd.notna(target_days) and row_tat_check['test_turnaround_days'] <= target_days
                
                tat_analysis_df['tat_met'] = tat_analysis_df.apply(_check_tat_met_for_enrich, axis=1)
                perc_met_df = tat_analysis_df.groupby('zone_id')['tat_met'].mean().reset_index(name='mean_val')
                perc_met_df['mean_val'] *= 100 # Convert proportion to percentage
                enriched_gdf = _robust_merge_agg(enriched_gdf, perc_met_df, 'perc_critical_tests_tat_met', default_fill_value=0.0)

        if 'avg_daily_steps' in health_df_agg.columns:
            enriched_gdf = _robust_merge_agg(enriched_gdf, health_df_agg.groupby('zone_id')['avg_daily_steps'].mean().reset_index(name='mean_val'), 'avg_daily_steps_zone', default_fill_value=np.nan)
    else:
        logger.warning(f"({source_context}) Health data for enrichment is empty or missing 'zone_id'. Most health aggregates will be default.")

    # Process IoT Data Aggregations
    if isinstance(iot_df, pd.DataFrame) and not iot_df.empty and all(c in iot_df.columns for c in ['zone_id', 'avg_co2_ppm']):
        iot_df_agg_src = iot_df.copy()
        iot_df_agg_src['zone_id'] = iot_df_agg_src['zone_id'].astype(str).str.strip()
        enriched_gdf = _robust_merge_agg(enriched_gdf, iot_df_agg_src.groupby('zone_id')['avg_co2_ppm'].mean().reset_index(name='mean_val'), 'zone_avg_co2', default_fill_value=np.nan)
    else:
        logger.info(f"({source_context}) IoT data not provided or suitable for CO2 aggregation.")

    # Calculate Derived Metrics
    if 'total_active_key_infections' in enriched_gdf.columns and 'population' in enriched_gdf.columns:
        enriched_gdf['prevalence_per_1000'] = enriched_gdf.apply(
            lambda r: (r['total_active_key_infections'] / r['population']) * 1000 if pd.notna(r['population']) and r['population'] > 0 else 0.0,
            axis=1
        ).fillna(0.0)

    # Facility Coverage Score (example logic, can be refined)
    # Assumes 'num_clinics' and 'population' are available in GDF from load_zone_data.
    if 'num_clinics' in enriched_gdf.columns and 'population' in enriched_gdf.columns:
        enriched_gdf['facility_coverage_score'] = enriched_gdf.apply(
            lambda r: min(100.0, (r.get('num_clinics', 0) / (r.get('population', 1) / 10000)) * 100 ) # Clinics per 10k pop, capped at 100%
                      if pd.notna(r.get('population')) and r.get('population') > 0 and pd.notna(r.get('num_clinics')) else 0.0,
            axis=1
        ).fillna(0.0)
    elif 'facility_coverage_score' not in enriched_gdf.columns: # Ensure column exists
        enriched_gdf['facility_coverage_score'] = 0.0

    # Population Density
    if 'geometry' in enriched_gdf.columns and 'population' in enriched_gdf.columns and enriched_gdf.crs:
        if 'area_sqkm' in enriched_gdf.columns and enriched_gdf['area_sqkm'].notna().any() and enriched_gdf['area_sqkm'].gt(0).any():
            enriched_gdf['population_density'] = (enriched_gdf['population'] / enriched_gdf['area_sqkm']).fillna(np.nan) # Keep NaN if area is 0/NaN
        else: # Try to calculate area if not provided, with CRS warning
            try:
                if not any(keyword in str(enriched_gdf.crs).lower() for keyword in ["equal_area", "utm", "projected", "metre", "meter"]):
                    logger.warning(f"({source_context}) GDF CRS '{enriched_gdf.crs}' may not be suitable for accurate area calculation. Population density might be affected.")
                # Area calculation (units depend on CRS; if EPSG:4326, area is in sq degrees, not ideal)
                # For simplicity, if area_sqkm not present, we'll leave population_density as NaN or 0.
                # A production system would reproject to an equal-area CRS.
                # Example if GDF were in meters: enriched_gdf['area_calc'] = enriched_gdf.geometry.area / 10**6
                # enriched_gdf['population_density'] = (enriched_gdf['population'] / enriched_gdf['area_calc']).fillna(np.nan)
                if 'population_density' not in enriched_gdf.columns:
                    enriched_gdf['population_density'] = np.nan
                logger.info(f"({source_context}) 'area_sqkm' column missing or invalid. Population density calculation may be limited or based on GDF's direct .area property (CRS dependent).")
            except Exception as e_area:
                logger.warning(f"({source_context}) Could not calculate area/density: {e_area}")
                if 'population_density' not in enriched_gdf.columns: enriched_gdf['population_density'] = np.nan

    # Final type conversions and NaN filling for aggregated columns
    for col_final_type, default_final_type in agg_cols_to_init.items():
        if col_final_type in enriched_gdf.columns:
            fill_val_final = np.nan if isinstance(default_final_type, float) and np.isnan(default_final_type) else default_final_type
            enriched_gdf[col_final_type] = pd.to_numeric(enriched_gdf[col_final_type], errors='coerce').fillna(fill_val_final)
        else: # Should not happen if initialized correctly
            enriched_gdf[col_final_type] = default_final_type

    logger.info(f"({source_context}) Zone GeoDataFrame enrichment complete. GDF shape: {enriched_gdf.shape}. Result columns: {enriched_gdf.columns.tolist()}")
    return enriched_gdf


# --- IV. KPI & Summary Calculation Functions ---
# These functions are now more detailed, using the helper functions and app_config.

@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS_WEB_REPORTS)
def get_overall_kpis(health_df: Optional[pd.DataFrame], date_filter_start: Optional[Any]=None, date_filter_end: Optional[Any]=None, source_context: str = "GlobalKPIs") -> Dict[str, Any]:
    logger.info(f"({source_context}) Calculating overall system KPIs.")
    kpis: Dict[str, Any] = {
        "total_patients_period": 0, "avg_patient_ai_risk_period": np.nan,
        "malaria_rdt_positive_rate_period": np.nan,
        "key_supply_stockout_alerts_period": 0,
        "total_encounters_period": 0
    }
    # Dynamically add keys for KEY_CONDITIONS_FOR_ACTION
    for cond_key in app_config.KEY_CONDITIONS_FOR_ACTION:
        kpis[f"active_{cond_key.lower().replace(' ', '_').replace('-', '_').replace('(severe)','')}_cases_period"] = 0

    if not isinstance(health_df, pd.DataFrame) or health_df.empty:
        logger.warning(f"({source_context}) No health data for overall KPIs."); return kpis
    
    df_kpi = health_df.copy()
    # Ensure encounter_date exists and is datetime
    if 'encounter_date' not in df_kpi.columns or not pd.api.types.is_datetime64_any_dtype(df_kpi['encounter_date']):
        if 'encounter_date' in df_kpi.columns:
            df_kpi['encounter_date'] = pd.to_datetime(df_kpi['encounter_date'], errors='coerce')
        else:
            logger.warning(f"({source_context}) 'encounter_date' column missing. Cannot calculate time-bound KPIs.")
            return kpis # Or calculate non-time bound if any
    df_kpi.dropna(subset=['encounter_date'], inplace=True)

    # Apply date filters if provided
    try:
        start_dt = pd.to_datetime(date_filter_start).date() if date_filter_start else None
        end_dt = pd.to_datetime(date_filter_end).date() if date_filter_end else None
        if start_dt: df_kpi = df_kpi[df_kpi['encounter_date'].dt.date >= start_dt]
        if end_dt: df_kpi = df_kpi[df_kpi['encounter_date'].dt.date <= end_dt]
    except Exception as e_date_filter:
        logger.error(f"({source_context}) Error applying date filters for overall KPIs: {e_date_filter}")
        # Proceed with unfiltered data or return current kpis if this is critical
    
    if df_kpi.empty: logger.info(f"({source_context}) No data after date filtering for overall KPIs."); return kpis

    if 'patient_id' in df_kpi.columns: kpis["total_patients_period"] = df_kpi['patient_id'].nunique()
    if 'encounter_id' in df_kpi.columns: kpis["total_encounters_period"] = df_kpi['encounter_id'].nunique()
    
    if 'ai_risk_score' in df_kpi.columns and df_kpi['ai_risk_score'].notna().any():
        kpis["avg_patient_ai_risk_period"] = df_kpi['ai_risk_score'].mean()
    
    if 'condition' in df_kpi.columns and 'patient_id' in df_kpi.columns:
        for cond_key in app_config.KEY_CONDITIONS_FOR_ACTION:
            kpi_col_name = f"active_{cond_key.lower().replace(' ', '_').replace('-', '_').replace('(severe)','')}_cases_period"
            mask_cond = df_kpi['condition'].str.contains(cond_key, case=False, na=False)
            kpis[kpi_col_name] = df_kpi[mask_cond]['patient_id'].nunique() if mask_cond.any() else 0

    malaria_rdt_orig_key = "RDT-Malaria" # Original key from KEY_TEST_TYPES_FOR_ANALYSIS
    if malaria_rdt_orig_key in app_config.KEY_TEST_TYPES_FOR_ANALYSIS and \
       'test_type' in df_kpi.columns and 'test_result' in df_kpi.columns:
        
        conclusive_malaria_tests_df = df_kpi[
            (df_kpi['test_type'] == malaria_rdt_orig_key) &
            (~df_kpi.get('test_result', pd.Series(dtype=str)).isin(
                ["Pending", "Rejected Sample", "Unknown", "Indeterminate", "N/A", ""])) # Filter for conclusive results
        ]
        if not conclusive_malaria_tests_df.empty:
            positive_malaria_count = conclusive_malaria_tests_df[conclusive_malaria_tests_df['test_result'] == 'Positive'].shape[0]
            kpis["malaria_rdt_positive_rate_period"] = (positive_malaria_count / len(conclusive_malaria_tests_df)) * 100
        else:
            kpis["malaria_rdt_positive_rate_period"] = 0.0 # No conclusive tests, so 0% positivity

    # Key Supply Stockout Alerts (Simplified logic: count distinct key drugs below critical threshold)
    if all(c in df_kpi.columns for c in ['item', 'item_stock_agg_zone', 'consumption_rate_per_day', 'encounter_date']) and app_config.KEY_DRUG_SUBSTRINGS_SUPPLY:
        # Consider latest stock status per item within the period
        latest_supply_status_df = df_kpi.sort_values('encounter_date').drop_duplicates(subset=['item', 'zone_id'], keep='last') # Assuming stock is zone-specific
        latest_supply_status_df['consumption_rate_per_day'] = _convert_to_numeric(latest_supply_status_df['consumption_rate_per_day'], 0.001).replace(0, 0.001) # Avoid div by zero
        latest_supply_status_df['days_supply_calc'] = latest_supply_status_df['item_stock_agg_zone'] / latest_supply_status_df['consumption_rate_per_day']
        
        key_drugs_df = latest_supply_status_df[
            latest_supply_status_df['item'].str.contains('|'.join(app_config.KEY_DRUG_SUBSTRINGS_SUPPLY), case=False, na=False)
        ]
        if not key_drugs_df.empty:
            kpis['key_supply_stockout_alerts_period'] = key_drugs_df[
                key_drugs_df['days_supply_calc'] < app_config.CRITICAL_SUPPLY_DAYS_REMAINING
            ]['item'].nunique() # Count unique items, not zones

    logger.info(f"({source_context}) Overall KPIs calculated: {kpis}")
    return kpis


@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS_WEB_REPORTS) # Cache CHW summary for a period
def get_chw_summary(health_df_daily_for_chw: Optional[pd.DataFrame], source_context: str = "CHWSummary") -> Dict[str, Any]:
    logger.info(f"({source_context}) Calculating CHW daily summary for provided data.")
    summary: Dict[str, Any] = {
        "visits_today": 0, "avg_patient_risk_visited_today": np.nan, 
        "high_ai_prio_followups_today": 0, "patients_critical_spo2_today": 0, 
        "patients_high_fever_today": 0, "avg_patient_steps_visited_today": np.nan,
        "patients_fall_detected_today": 0, "pending_critical_condition_referrals": 0,
        "worker_self_fatigue_index_today": np.nan # If CHW self-assessment data is included
    }
    if not isinstance(health_df_daily_for_chw, pd.DataFrame) or health_df_daily_for_chw.empty:
        logger.warning(f"({source_context}) No health data for CHW summary."); return summary
    
    chw_df = health_df_daily_for_chw.copy() # Work on a copy

    # Ensure patient_id column is present for unique patient counts
    if 'patient_id' not in chw_df.columns: 
        logger.warning(f"({source_context}) 'patient_id' missing, cannot calculate visit counts etc."); return summary
    
    # Filter out records that might be worker self-checks unless explicitly handled
    # Assuming 'encounter_type' exists and is cleaned.
    patient_records_df = chw_df[~chw_df.get('encounter_type', pd.Series(dtype=str)).str.contains("WORKER_SELF", case=False, na=False)]
    
    if patient_records_df.empty:
        logger.info(f"({source_context}) No patient-specific records found for CHW summary after filtering out self-checks.")
    else:
        summary["visits_today"] = patient_records_df['patient_id'].nunique()

        if 'ai_risk_score' in patient_records_df.columns and patient_records_df['ai_risk_score'].notna().any():
            unique_patient_risk_df = patient_records_df.drop_duplicates(subset=['patient_id'])
            summary["avg_patient_risk_visited_today"] = unique_patient_risk_df['ai_risk_score'].mean()
        
        if 'ai_followup_priority_score' in patient_records_df.columns and patient_records_df['ai_followup_priority_score'].notna().any():
            summary["high_ai_prio_followups_today"] = patient_records_df[
                patient_records_df['ai_followup_priority_score'] >= app_config.FATIGUE_INDEX_HIGH_THRESHOLD # Using this as generic high prio
            ]['patient_id'].nunique()
        
        if 'min_spo2_pct' in patient_records_df.columns and patient_records_df['min_spo2_pct'].notna().any():
            summary["patients_critical_spo2_today"] = patient_records_df[
                patient_records_df['min_spo2_pct'] < app_config.ALERT_SPO2_CRITICAL_LOW_PCT
            ]['patient_id'].nunique()
        
        temp_col_chw = next((tc for tc in ['vital_signs_temperature_celsius', 'max_skin_temp_celsius'] 
                            if tc in patient_records_df.columns and patient_records_df[tc].notna().any()), None)
        if temp_col_chw:
            summary["patients_high_fever_today"] = patient_records_df[
                patient_records_df[temp_col_chw] >= app_config.ALERT_BODY_TEMP_HIGH_FEVER_C
            ]['patient_id'].nunique()
            
        if 'avg_daily_steps' in patient_records_df.columns and patient_records_df['avg_daily_steps'].notna().any():
            unique_patient_steps_df = patient_records_df.drop_duplicates(subset=['patient_id'])
            summary["avg_patient_steps_visited_today"] = unique_patient_steps_df['avg_daily_steps'].mean()
        
        if 'fall_detected_today' in patient_records_df.columns and patient_records_df['fall_detected_today'].notna().any():
            summary["patients_fall_detected_today"] = patient_records_df[
                patient_records_df['fall_detected_today'] > 0 # Assuming 1=fall, 0=no fall
            ]['patient_id'].nunique()

        if all(c in patient_records_df.columns for c in ['condition', 'referral_status', 'referral_reason']):
            crit_conds_set = set(app_config.KEY_CONDITIONS_FOR_ACTION)
            urgent_keywords = {'urgent', 'emergency', 'critical', 'severe'} # Set for faster lookup
            
            def _is_pending_critical_referral(row):
                if str(row.get('referral_status','')).lower() != 'pending': return False
                cond_lower = str(row.get('condition','')).lower()
                reason_lower = str(row.get('referral_reason','')).lower()
                is_crit_cond = any(ck.lower() in cond_lower for ck in crit_conds_set)
                is_urgent_reason = any(uk in reason_lower for uk in urgent_keywords)
                return is_crit_cond or is_urgent_reason
            
            patient_records_df['is_pending_crit_ref'] = patient_records_df.apply(_is_pending_critical_referral, axis=1)
            summary["pending_critical_condition_referrals"] = patient_records_df[patient_records_df['is_pending_crit_ref']]['patient_id'].nunique()

    # Extract CHW self-fatigue if available from original df (before filtering for patient records)
    worker_self_check_records = chw_df[chw_df.get('encounter_type', pd.Series(dtype=str)).str.contains("WORKER_SELF_CHECK", case=False, na=False)]
    if not worker_self_check_records.empty:
        # Assuming ai_followup_priority_score from WORKER_SELF_CHECK is the fatigue index
        # Or a dedicated column like 'worker_fatigue_index'
        fatigue_col_candidate = 'ai_followup_priority_score' # Or a more specific column if it exists
        if fatigue_col_candidate in worker_self_check_records.columns and worker_self_check_records[fatigue_col_candidate].notna().any():
            # Take the max fatigue score reported by the worker for the day
            summary["worker_self_fatigue_index_today"] = worker_self_check_records[fatigue_col_candidate].max()

    logger.info(f"({source_context}) CHW daily summary calculated: {summary}")
    return summary


# (get_patient_alerts_for_chw, get_clinic_summary, get_clinic_environmental_summary,
#  get_patient_alerts_for_clinic, get_district_summary_kpis, get_trend_data,
#  get_supply_forecast_data - implementations will be added/verified below this line)
# For now, I'll just provide the fully implemented get_clinic_summary as an example of
# fleshing out the previous stubs, then the others will follow a similar pattern of using
# app_config and the input data to calculate their respective metrics.

# ... (Previous placeholder get_patient_alerts_for_chw and others) ...

# --- IV. KPI & Summary Calculation Functions - DETAILED IMPLEMENTATIONS ---

# get_patient_alerts_for_chw now relies on the alert_generator.py component module,
# so this function in core_data_processing might be simplified or removed if
# alerts are always generated via that component. For now, let's assume
# a basic version could live here for very simple alert flags if needed by other processes.
# However, the prompt asked for consolidation, so the alert_generator.py logic is preferred.
# For this refactor, I will assume get_patient_alerts_for_chw is a wrapper or direct call
# to the more detailed logic in pages.chw_components_sentinel.alert_generator
# This function might not be needed in core_data_processing if alerts are solely a display concern.
# Decision: Remove from core_data_processing and let alert_generator.py handle it for CHW dashboard.

@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS_WEB_REPORTS)
def get_clinic_summary(health_df_period: Optional[pd.DataFrame], source_context: str = "ClinicSummary") -> Dict[str, Any]:
    logger.info(f"({source_context}) Calculating comprehensive clinic summary.")
    summary: Dict[str, Any] = {
        "overall_avg_test_turnaround_conclusive_days": np.nan,
        "perc_critical_tests_tat_met": 0.0,
        "total_pending_critical_tests_patients": 0,
        "sample_rejection_rate_perc": 0.0,
        "key_drug_stockouts_count": 0, # Based on latest status of key drugs
        "test_summary_details": {} # Detailed stats per test type display name
    }
    if not isinstance(health_df_period, pd.DataFrame) or health_df_period.empty:
        logger.warning(f"({source_context}) No health data for clinic summary."); return summary
    
    df = health_df_period.copy() # Work on a copy

    # Ensure required columns for test analysis exist
    test_cols_clinic = ['test_type', 'test_result', 'test_turnaround_days', 'sample_status', 'patient_id']
    for col in test_cols_clinic:
        if col not in df.columns:
            logger.warning(f"({source_context}) Missing column '{col}' for clinic test summary calculations.")
            # Add with default to prevent errors, but results will be affected
            df[col] = "Unknown" if col in ['test_type', 'test_result', 'sample_status'] else \
                      np.nan if col == 'test_turnaround_days' else "UnknownPID"
    
    df['test_turnaround_days'] = _convert_to_numeric(df['test_turnaround_days'], np.nan)

    # Conclusive tests (not pending, not rejected, not unknown/indeterminate)
    conclusive_mask = ~df.get('test_result', pd.Series(dtype=str)).isin(["Pending", "Rejected Sample", "Unknown", "Indeterminate", "N/A", ""])
    conclusive_tests_df = df[conclusive_mask].copy()

    if not conclusive_tests_df.empty and 'test_turnaround_days' in conclusive_tests_df and conclusive_tests_df['test_turnaround_days'].notna().any():
        summary["overall_avg_test_turnaround_conclusive_days"] = conclusive_tests_df['test_turnaround_days'].mean()

    # Critical Tests TAT Met
    critical_test_keys = app_config.CRITICAL_TESTS_LIST
    if critical_test_keys:
        crit_conclusive_tests_df = conclusive_tests_df[conclusive_tests_df['test_type'].isin(critical_test_keys)].copy()
        if not crit_conclusive_tests_df.empty:
            def _check_tat_met_clinic(row):
                test_cfg = app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get(row['test_type'])
                target_days = test_cfg.get('target_tat_days', app_config.TARGET_TEST_TURNAROUND_DAYS) if test_cfg else app_config.TARGET_TEST_TURNAROUND_DAYS
                return pd.notna(row['test_turnaround_days']) and pd.notna(target_days) and row['test_turnaround_days'] <= target_days
            
            crit_conclusive_tests_df['tat_met'] = crit_conclusive_tests_df.apply(_check_tat_met_clinic, axis=1)
            summary["perc_critical_tests_tat_met"] = crit_conclusive_tests_df['tat_met'].mean() * 100 if not crit_conclusive_tests_df['tat_met'].empty else 0.0
    
    # Pending Critical Tests (Patients)
    pending_crit_df = df[
        (df['test_type'].isin(critical_test_keys)) &
        (df.get('test_result', pd.Series(dtype=str)) == "Pending")
    ]
    if not pending_crit_df.empty and 'patient_id' in pending_crit_df.columns:
        summary["total_pending_critical_tests_patients"] = pending_crit_df['patient_id'].nunique()

    # Sample Rejection Rate
    total_tests_with_status = df[df.get('sample_status', pd.Series(dtype=str)).notna() & (df.get('sample_status',pd.Series(dtype=str)) != 'Unknown')]['encounter_id'].nunique() # Count unique encounters with tests
    rejected_samples_count = df[df.get('sample_status',pd.Series(dtype=str)) == "Rejected"]['encounter_id'].nunique()
    if total_tests_with_status > 0:
        summary["sample_rejection_rate_perc"] = (rejected_samples_count / total_tests_with_status) * 100
    
    # Key Drug Stockouts (using latest status per item in the period)
    if all(c in df.columns for c in ['item', 'item_stock_agg_zone', 'consumption_rate_per_day', 'encounter_date']) and app_config.KEY_DRUG_SUBSTRINGS_SUPPLY:
        latest_stock_df = df.sort_values('encounter_date').drop_duplicates(subset=['item'], keep='last') # Assuming stock is clinic-wide, not zone-specific for this summary
        latest_stock_df['consumption_rate_per_day'] = _convert_to_numeric(latest_stock_df['consumption_rate_per_day'], 0.001).replace(0, 0.001)
        latest_stock_df['days_supply'] = latest_stock_df['item_stock_agg_zone'] / latest_stock_df['consumption_rate_per_day']
        
        key_drugs_stock_df = latest_stock_df[
            latest_stock_df['item'].str.contains('|'.join(app_config.KEY_DRUG_SUBSTRINGS_SUPPLY), case=False, na=False)
        ]
        if not key_drugs_stock_df.empty:
            summary["key_drug_stockouts_count"] = key_drugs_stock_df[
                key_drugs_stock_df['days_supply'] < app_config.CRITICAL_SUPPLY_DAYS_REMAINING
            ]['item'].nunique()

    # Detailed Test Summary
    test_details_summary_map = {}
    # Iterate over configured test types to ensure all are considered
    for test_orig_key, test_config in app_config.KEY_TEST_TYPES_FOR_ANALYSIS.items():
        test_disp_name = test_config.get("display_name", test_orig_key)
        detail = {}
        
        # Filter for this specific test type
        df_this_test_all = df[df['test_type'] == test_orig_key]
        df_this_test_conclusive = conclusive_tests_df[conclusive_tests_df['test_type'] == test_orig_key]

        if not df_this_test_conclusive.empty:
            detail["positive_rate_perc"] = (df_this_test_conclusive[df_this_test_conclusive['test_result'] == 'Positive'].shape[0] / len(df_this_test_conclusive)) * 100 if len(df_this_test_conclusive) > 0 else 0.0
            detail["avg_tat_days"] = df_this_test_conclusive['test_turnaround_days'].mean() if df_this_test_conclusive['test_turnaround_days'].notna().any() else np.nan
            
            # % Met TAT Target for this specific test
            target_days_this_test = test_config.get('target_tat_days', app_config.TARGET_TEST_TURNAROUND_DAYS)
            met_tat_this_test_series = df_this_test_conclusive[
                (df_this_test_conclusive['test_turnaround_days'].notna()) &
                (df_this_test_conclusive['test_turnaround_days'] <= target_days_this_test)
            ]
            detail["perc_met_tat_target"] = (len(met_tat_this_test_series) / len(df_this_test_conclusive)) * 100 if len(df_this_test_conclusive) > 0 else 0.0
            detail["total_conclusive_tests"] = len(df_this_test_conclusive)
        else: # No conclusive tests for this type
            detail["positive_rate_perc"] = 0.0; detail["avg_tat_days"] = np.nan;
            detail["perc_met_tat_target"] = 0.0; detail["total_conclusive_tests"] = 0

        # Pending and Rejected counts for this specific test
        detail["pending_count_patients"] = df_this_test_all[df_this_test_all.get('test_result',pd.Series(dtype=str)) == "Pending"]['patient_id'].nunique()
        detail["rejected_count_patients"] = df_this_test_all[df_this_test_all.get('sample_status',pd.Series(dtype=str)) == "Rejected"]['patient_id'].nunique()
        
        test_details_summary_map[test_disp_name] = detail
    summary["test_summary_details"] = test_details_summary_map

    logger.info(f"({source_context}) Clinic summary calculated: {list(summary.keys())}")
    return summary


@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS_WEB_REPORTS)
def get_clinic_environmental_summary(iot_df_period: Optional[pd.DataFrame], source_context: str = "ClinicEnvSummary") -> Dict[str, Any]:
    logger.info(f"({source_context}) Calculating clinic environmental summary.")
    summary: Dict[str, Any] = {
        "avg_co2_overall_ppm": np.nan, "rooms_co2_very_high_alert_latest_count": 0,
        "avg_pm25_overall_ugm3": np.nan, "rooms_pm25_very_high_alert_latest_count": 0,
        "avg_waiting_room_occupancy_overall_persons": np.nan, "waiting_room_high_occupancy_alert_latest_flag": False,
        "avg_noise_overall_dba": np.nan, "rooms_noise_high_alert_latest_count": 0,
        "avg_temp_overall_celsius": np.nan, "avg_humidity_overall_rh": np.nan,
        "latest_readings_timestamp": None
    }
    if not isinstance(iot_df_period, pd.DataFrame) or iot_df_period.empty:
        logger.warning(f"({source_context}) No IoT data for environmental summary."); return summary

    df_iot = iot_df_period.copy()
    if 'timestamp' not in df_iot.columns or not pd.api.types.is_datetime64_any_dtype(df_iot['timestamp']):
        logger.warning(f"({source_context}) IoT data missing or invalid 'timestamp'. Environmental summary will be limited."); return summary
    
    df_iot.dropna(subset=['timestamp'], inplace=True)
    if df_iot.empty: logger.warning(f"({source_context}) IoT data has no valid timestamps."); return summary
    
    summary["latest_readings_timestamp"] = df_iot['timestamp'].max()

    # Overall averages across all rooms/readings in the period
    if 'avg_co2_ppm' in df_iot.columns: summary["avg_co2_overall_ppm"] = df_iot['avg_co2_ppm'].mean()
    if 'avg_pm25' in df_iot.columns: summary["avg_pm25_overall_ugm3"] = df_iot['avg_pm25'].mean()
    if 'avg_noise_db' in df_iot.columns: summary["avg_noise_overall_dba"] = df_iot['avg_noise_db'].mean()
    if 'avg_temp_celsius' in df_iot.columns: summary["avg_temp_overall_celsius"] = df_iot['avg_temp_celsius'].mean()
    if 'avg_humidity_rh' in df_iot.columns: summary["avg_humidity_overall_rh"] = df_iot['avg_humidity_rh'].mean()

    # Alerts based on latest reading per room in the period
    latest_room_readings_df = df_iot.sort_values('timestamp').drop_duplicates(subset=['clinic_id', 'room_name'], keep='last')
    
    if not latest_room_readings_df.empty:
        if 'avg_co2_ppm' in latest_room_readings_df.columns:
            summary["rooms_co2_very_high_alert_latest_count"] = latest_room_readings_df[
                latest_room_readings_df['avg_co2_ppm'] > app_config.ALERT_AMBIENT_CO2_VERY_HIGH_PPM
            ].shape[0]
        if 'avg_pm25' in latest_room_readings_df.columns:
            summary["rooms_pm25_very_high_alert_latest_count"] = latest_room_readings_df[
                latest_room_readings_df['avg_pm25'] > app_config.ALERT_AMBIENT_PM25_VERY_HIGH_UGM3
            ].shape[0]
        if 'avg_noise_db' in latest_room_readings_df.columns:
            summary["rooms_noise_high_alert_latest_count"] = latest_room_readings_df[
                latest_room_readings_df['avg_noise_db'] > app_config.ALERT_AMBIENT_NOISE_HIGH_DBA
            ].shape[0]
        
        # Waiting room occupancy
        waiting_rooms_latest_df = latest_room_readings_df[
            latest_room_readings_df.get('room_name', pd.Series(dtype=str)).str.contains("Waiting", case=False, na=False)
        ]
        if not waiting_rooms_latest_df.empty and 'waiting_room_occupancy' in waiting_rooms_latest_df.columns:
            summary["avg_waiting_room_occupancy_overall_persons"] = waiting_rooms_latest_df['waiting_room_occupancy'].mean() # Avg of latest occupancy for waiting rooms
            summary["waiting_room_high_occupancy_alert_latest_flag"] = (
                waiting_rooms_latest_df['waiting_room_occupancy'] > app_config.TARGET_CLINIC_WAITING_ROOM_OCCUPANCY_MAX
            ).any()

    logger.info(f"({source_context}) Clinic environmental summary calculated: {summary}")
    return summary

@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS_WEB_REPORTS)
def get_patient_alerts_for_clinic(health_df_period: Optional[pd.DataFrame], risk_threshold_moderate: int = app_config.RISK_SCORE_MODERATE_THRESHOLD, source_context: str = "ClinicPatientAlerts") -> pd.DataFrame:
    logger.info(f"({source_context}) Generating patient alerts for clinic view.")
    cols_for_clinic_alerts = ['patient_id', 'encounter_date', 'condition', 'Alert Reason', 'Priority Score', 'ai_risk_score', 'age', 'gender', 'zone_id', 'referred_to_facility_id']
    
    if not isinstance(health_df_period, pd.DataFrame) or health_df_period.empty:
        logger.warning(f"({source_context}) No health data for clinic patient alerts."); return pd.DataFrame(columns=cols_for_clinic_alerts)

    df = health_df_period.copy()
    alerts = []

    # Ensure necessary columns exist for rule evaluation
    required_alert_cols = { 'patient_id': "UnknownPID", 'encounter_date': pd.NaT, 'condition': "N/A", 
                            'ai_risk_score': np.nan, 'ai_followup_priority_score': np.nan,
                            'min_spo2_pct': np.nan, 'vital_signs_temperature_celsius': np.nan, 
                            'max_skin_temp_celsius': np.nan, 'referral_status': "Unknown",
                            'age': np.nan, 'gender': "Unknown", 'zone_id': "UnknownZone",
                            'referred_to_facility_id': "UnknownFacility" }
    for col, default in required_alert_cols.items():
        if col not in df.columns: df[col] = default
        elif 'date' in col: df[col] = pd.to_datetime(df[col], errors='coerce')
        # Apply _convert_to_numeric for relevant columns if they exist
        elif col in ['ai_risk_score', 'ai_followup_priority_score', 'min_spo2_pct', 
                     'vital_signs_temperature_celsius', 'max_skin_temp_celsius', 'age']:
            df[col] = _convert_to_numeric(df[col], default_value=default)


    for _, row in df.iterrows():
        alert_reason = None; priority = row.get('ai_followup_priority_score', row.get('ai_risk_score', 0)) # Base priority
        
        # Rule: High AI Risk Score for Clinic Review
        if pd.notna(row.get('ai_risk_score')) and row['ai_risk_score'] >= risk_threshold_moderate:
            alert_reason = f"High AI Risk ({row['ai_risk_score']:.0f})"
            priority = max(priority, row['ai_risk_score'])

        # Rule: Critical Low SpO2
        if pd.notna(row.get('min_spo2_pct')) and row['min_spo2_pct'] < app_config.ALERT_SPO2_CRITICAL_LOW_PCT:
            alert_reason = f"Critical SpO2 ({row['min_spo2_pct']:.0f}%)"
            priority = max(priority, 95 + (app_config.ALERT_SPO2_CRITICAL_LOW_PCT - row['min_spo2_pct']))

        # Rule: High Fever
        temp_val_clinic_alert = row.get('vital_signs_temperature_celsius', row.get('max_skin_temp_celsius'))
        if pd.notna(temp_val_clinic_alert) and temp_val_clinic_alert >= app_config.ALERT_BODY_TEMP_HIGH_FEVER_C:
            alert_reason = f"High Fever ({temp_val_clinic_alert:.1f}°C)"
            priority = max(priority, 90 + (temp_val_clinic_alert - app_config.ALERT_BODY_TEMP_HIGH_FEVER_C))

        # Rule: Pending Critical Referral needing follow-up at this clinic
        if str(row.get('referral_status','')).lower() == 'pending' and \
           str(row.get('referred_to_facility_id','')).lower() == str(df.get('clinic_id', pd.Series(dtype=str)).iloc[0] if 'clinic_id' in df.columns and not df['clinic_id'].empty else "this_clinic").lower() and \
           any(kc.lower() in str(row.get('condition','')).lower() for kc in app_config.KEY_CONDITIONS_FOR_ACTION):
            alert_reason = f"Pending Inbound Critical Referral: {row.get('condition')}"
            priority = max(priority, 85)

        if alert_reason:
            alerts.append({
                'patient_id': row.get('patient_id'), 'encounter_date': row.get('encounter_date'),
                'condition': row.get('condition'), 'Alert Reason': alert_reason,
                'Priority Score': round(min(priority, 100),1), # Cap at 100
                'ai_risk_score': row.get('ai_risk_score'), 'age': row.get('age'),
                'gender': row.get('gender'), 'zone_id': row.get('zone_id'),
                'referred_to_facility_id': row.get('referred_to_facility_id')
            })
    
    if not alerts: return pd.DataFrame(columns=cols_for_clinic_alerts)
    
    alerts_df_clinic = pd.DataFrame(alerts)
    alerts_df_clinic.sort_values(by='Priority Score', ascending=False, inplace=True)
    # Keep highest priority alert per patient for the period for this list
    alerts_df_clinic.drop_duplicates(subset=['patient_id'], keep='first', inplace=True) 
    
    logger.info(f"({source_context}) Generated {len(alerts_df_clinic)} unique patient alerts for clinic view.")
    return alerts_df_clinic[cols_for_clinic_alerts].head(50) # Limit to top 50 for display

@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS_WEB_REPORTS, hash_funcs={gpd.GeoDataFrame: hash_geodataframe})
def get_district_summary_kpis(enriched_zone_gdf: Optional[gpd.GeoDataFrame], source_context: str = "DistrictKPIs") -> Dict[str, Any]:
    logger.info(f"({source_context}) Calculating district summary KPIs from enriched GDF.")
    kpis: Dict[str, Any] = {
        "total_zones_in_gdf": 0, "total_population_district": 0.0,
        "population_weighted_avg_ai_risk_score": np.nan,
        "zones_meeting_high_risk_criteria_count": 0,
        "district_avg_facility_coverage_score": np.nan, # Population weighted
        "district_overall_key_disease_prevalence_per_1000": np.nan,
        "district_population_weighted_avg_steps": np.nan,
        "district_avg_clinic_co2_ppm": np.nan, # Simple average of zonal means
        "total_critical_tests_pending_district": 0
    }
    # Dynamic keys for KEY_CONDITIONS_FOR_ACTION
    for cond_key in app_config.KEY_CONDITIONS_FOR_ACTION:
        kpis[f"district_total_active_{cond_key.lower().replace(' ', '_').replace('-', '_').replace('(severe)','')}_cases"] = 0

    if not isinstance(enriched_zone_gdf, gpd.GeoDataFrame) or enriched_zone_gdf.empty:
        logger.warning(f"({source_context}) Enriched GDF empty or invalid for district KPIs."); return kpis

    gdf = enriched_zone_gdf.copy()
    kpis["total_zones_in_gdf"] = gdf['zone_id'].nunique() if 'zone_id' in gdf.columns else len(gdf)
    
    # Ensure population is numeric and handle NaNs/zeros for weighting
    gdf['population'] = _convert_to_numeric(gdf.get('population'), 0.0)
    total_pop_for_weighting = gdf['population'].sum()
    kpis["total_population_district"] = total_pop_for_weighting

    if total_pop_for_weighting > 0:
        if 'avg_risk_score' in gdf.columns:
            gdf['weighted_risk_numerator'] = gdf['avg_risk_score'].fillna(0) * gdf['population']
            kpis["population_weighted_avg_ai_risk_score"] = gdf['weighted_risk_numerator'].sum() / total_pop_for_weighting
        
        if 'facility_coverage_score' in gdf.columns:
            gdf['weighted_facility_cov_numerator'] = gdf['facility_coverage_score'].fillna(0) * gdf['population']
            kpis["district_avg_facility_coverage_score"] = gdf['weighted_facility_cov_numerator'].sum() / total_pop_for_weighting

        if 'avg_daily_steps_zone' in gdf.columns:
            gdf['weighted_steps_numerator'] = gdf['avg_daily_steps_zone'].fillna(app_config.TARGET_DAILY_STEPS * 0.5) * gdf['population'] # Fill NaN steps with a moderate default for weighting
            kpis["district_population_weighted_avg_steps"] = gdf['weighted_steps_numerator'].sum() / total_pop_for_weighting
    else: # If total population is 0, use simple means or NaNs
        logger.warning(f"({source_context}) Total district population is 0. Weighted averages will be NaN or simple means.")
        if 'avg_risk_score' in gdf.columns: kpis["population_weighted_avg_ai_risk_score"] = gdf['avg_risk_score'].mean() # Simple mean as fallback
        if 'facility_coverage_score' in gdf.columns: kpis["district_avg_facility_coverage_score"] = gdf['facility_coverage_score'].mean()
        if 'avg_daily_steps_zone' in gdf.columns: kpis["district_population_weighted_avg_steps"] = gdf['avg_daily_steps_zone'].mean()


    if 'avg_risk_score' in gdf.columns:
        kpis["zones_meeting_high_risk_criteria_count"] = gdf[
            gdf['avg_risk_score'] >= app_config.DISTRICT_ZONE_HIGH_RISK_AVG_SCORE
        ].shape[0]

    total_active_key_infections_district = 0
    for cond_key in app_config.KEY_CONDITIONS_FOR_ACTION:
        col_name_dist_kpi = f"active_{cond_key.lower().replace(' ', '_').replace('-', '_').replace('(severe)','')}_cases"
        if col_name_dist_kpi in gdf.columns:
            sum_val = gdf[col_name_dist_kpi].sum()
            kpis[f"district_total_active_{cond_key.lower().replace(' ', '_').replace('-', '_').replace('(severe)','')}_cases"] = sum_val
            total_active_key_infections_district += sum_val
    
    if total_pop_for_weighting > 0:
        kpis["district_overall_key_disease_prevalence_per_1000"] = (total_active_key_infections_district / total_pop_for_weighting) * 1000
    
    if 'zone_avg_co2' in gdf.columns and gdf['zone_avg_co2'].notna().any():
        kpis["district_avg_clinic_co2_ppm"] = gdf['zone_avg_co2'].mean() # Simple average of zonal means for CO2

    # Placeholder for district total pending critical tests (would require GDF to have this per zone)
    if 'pending_critical_tests_zone_count' in gdf.columns: # Example column name
        kpis["total_critical_tests_pending_district"] = gdf['pending_critical_tests_zone_count'].sum()

    logger.info(f"({source_context}) District summary KPIs calculated: {kpis}")
    return kpis

@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS_WEB_REPORTS)
def get_trend_data(
    df: Optional[pd.DataFrame], value_col: str, date_col: str = 'encounter_date', 
    period: str = 'D', agg_func: Union[str, Callable] = 'mean', # Allow custom agg func
    filter_col: Optional[str] = None, filter_val: Optional[Any] = None,
    source_context: str = "TrendCalculator"
) -> pd.Series:
    logger.debug(f"({source_context}) Generating trend data for '{value_col}' by '{period}', agg: '{agg_func}'.")
    
    if not isinstance(df, pd.DataFrame) or df.empty:
        logger.debug(f"({source_context}) Input DataFrame for trend is empty."); return pd.Series(dtype='float64')
    if date_col not in df.columns:
        logger.error(f"({source_context}) Date column '{date_col}' not found for trend calculation."); return pd.Series(dtype='float64')
    if value_col not in df.columns:
        logger.error(f"({source_context}) Value column '{value_col}' not found for trend calculation."); return pd.Series(dtype='float64')

    df_trend = df.copy()
    df_trend[date_col] = pd.to_datetime(df_trend[date_col], errors='coerce')
    df_trend.dropna(subset=[date_col, value_col], inplace=True) # Must have date and value
    
    if df_trend.empty: logger.debug(f"({source_context}) No valid data after cleaning for trend."); return pd.Series(dtype='float64')

    if filter_col and filter_val is not None and filter_col in df_trend.columns:
        df_trend = df_trend[df_trend[filter_col] == filter_val]
        if df_trend.empty: logger.debug(f"({source_context}) No data after applying filter: {filter_col}=={filter_val}."); return pd.Series(dtype='float64')
    
    try:
        # Ensure value_col is numeric for common agg_funcs like 'mean', 'sum'
        if isinstance(agg_func, str) and agg_func in ['mean', 'sum', 'median', 'std', 'var']:
            df_trend[value_col] = pd.to_numeric(df_trend[value_col], errors='coerce')
            df_trend.dropna(subset=[value_col], inplace=True) # Remove rows where conversion failed
            if df_trend.empty: logger.debug(f"({source_context}) No numeric data in '{value_col}' after coercion."); return pd.Series(dtype='float64')

        trend_series = df_trend.set_index(date_col)[value_col].resample(period).agg(agg_func)
        # Fill NaNs resulting from resample gaps if agg func is count-like, else keep NaNs for averages
        if isinstance(agg_func, str) and agg_func in ['count', 'nunique', 'size']:
            trend_series = trend_series.fillna(0)
        
        logger.debug(f"({source_context}) Trend for '{value_col}' generated with {len(trend_series)} points.")
        return trend_series
    except Exception as e:
        logger.error(f"({source_context}) Error generating trend for '{value_col}': {e}", exc_info=True)
        return pd.Series(dtype='float64')

@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS_WEB_REPORTS)
def get_supply_forecast_data(
    health_df: Optional[pd.DataFrame], # Health data containing item usage and stock info
    forecast_days_out: int = 30, 
    item_filter_list: Optional[List[str]] = None,
    source_context: str = "SimpleSupplyForecast"
) -> pd.DataFrame:
    logger.info(f"({source_context}) Generating simple linear supply forecast for {forecast_days_out} days.")
    output_cols = ['item', 'date', 'forecasted_stock_level', 'forecasted_days_of_supply', 
                   'estimated_stockout_date_linear', 'initial_stock_at_forecast_start', 
                   'base_consumption_rate_per_day']
    
    if not isinstance(health_df, pd.DataFrame) or health_df.empty:
        logger.warning(f"({source_context}) No health data for supply forecast."); return pd.DataFrame(columns=output_cols)

    required_cols = ['item', 'encounter_date', 'item_stock_agg_zone', 'consumption_rate_per_day']
    if not all(col in health_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in health_df.columns]
        logger.error(f"({source_context}) Missing required columns for supply forecast: {missing}."); return pd.DataFrame(columns=output_cols)

    df_supply = health_df[required_cols].copy()
    df_supply['encounter_date'] = pd.to_datetime(df_supply['encounter_date'], errors='coerce')
    df_supply.dropna(subset=['encounter_date', 'item'], inplace=True)
    df_supply['item_stock_agg_zone'] = _convert_to_numeric(df_supply['item_stock_agg_zone'], 0)
    df_supply['consumption_rate_per_day'] = _convert_to_numeric(df_supply['consumption_rate_per_day'], 0.001).replace(0, 0.001) # Avoid zero

    if item_filter_list:
        df_supply = df_supply[df_supply['item'].isin(item_filter_list)]
    if df_supply.empty: 
        logger.info(f"({source_context}) No items to forecast after filtering."); return pd.DataFrame(columns=output_cols)

    # Get latest status for each item
    latest_item_status_df = df_supply.sort_values('encounter_date').drop_duplicates(subset=['item'], keep='last')
    
    all_forecast_records = []
    forecast_start_date = pd.Timestamp(date.today()) # Use today as forecast start for simplicity

    for _, item_row in latest_item_status_df.iterrows():
        item_name = item_row['item']
        current_stock = item_row['item_stock_agg_zone']
        daily_consumption = item_row['consumption_rate_per_day']
        
        initial_dos = (current_stock / daily_consumption) if daily_consumption > 0 else np.inf
        stockout_date_calc = forecast_start_date + pd.to_timedelta(initial_dos, unit='D') if np.isfinite(initial_dos) else pd.NaT

        running_stock = current_stock
        for day_offset in range(forecast_days_out):
            fc_date = forecast_start_date + pd.Timedelta(days=day_offset)
            running_stock = max(0, current_stock - (daily_consumption * day_offset)) # Linear depletion
            dos_at_fc_date = (running_stock / daily_consumption) if daily_consumption > 0 else (np.inf if running_stock > 0 else 0)
            
            all_forecast_records.append({
                'item': item_name, 'date': fc_date,
                'forecasted_stock_level': running_stock,
                'forecasted_days_of_supply': dos_at_fc_date,
                'estimated_stockout_date_linear': stockout_date_calc,
                'initial_stock_at_forecast_start': current_stock,
                'base_consumption_rate_per_day': daily_consumption,
                'initial_days_supply_at_forecast_start': initial_dos
            })
            
    if not all_forecast_records: return pd.DataFrame(columns=output_cols)
    
    final_fc_df = pd.DataFrame(all_forecast_records)
    final_fc_df['estimated_stockout_date_linear'] = pd.to_datetime(final_fc_df['estimated_stockout_date_linear'], errors='coerce')
    logger.info(f"({source_context}) Simple linear supply forecast generated for {len(latest_item_status_df)} items.")
    return final_fc_df
