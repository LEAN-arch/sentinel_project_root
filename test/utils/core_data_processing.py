# sentinel_project_root/test/utils/core_data_processing.py
# MODIFIED VERSION - REMOVING GEOPANDAS DEPENDENCY

import streamlit as st
import pandas as pd
# import geopandas as gpd # <<<< REMOVED
import numpy as np
import os
import sys
import logging
import json # For reading GeoJSON manually
from config import app_config
from typing import List, Dict, Any, Optional, Tuple, Union

logger = logging.getLogger(__name__)
# GEOPANDAS_AVAILABLE = False # Set this globally if removing gpd everywhere

# --- I. Core Helper Functions ---
# _clean_column_names, _convert_to_numeric, _robust_merge_agg remain largely the same
# hash_geodataframe is REMOVED or significantly altered to hash_dataframe_with_potential_geom_strings
# For simplicity, let's assume hash_geodataframe is just removed if no GDFs exist.
# If DataFrames still need complex hashing for streamlit caching, that's a separate issue.
# @st.cache_data's default hashing might be sufficient for pandas DataFrames without geo objects.

# ... (Keep _clean_column_names, _convert_to_numeric, _robust_merge_agg as they are useful) ...
# ... (Remove hash_geodataframe function)

# --- II. Data Loading and Basic Cleaning Functions ---
# load_health_records and load_iot_clinic_environment_data remain the same as they don't use geopandas.

@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS_WEB_REPORTS, show_spinner="Loading zone attribute data...")
def load_zone_data_no_geopandas( # Renamed for clarity of this alternative
    attributes_path: Optional[str] = None,
    geometries_path: Optional[str] = None, # Path to GeoJSON to extract properties
    source_context: str = "FacilityNode"
) -> Optional[pd.DataFrame]: # Returns a Pandas DataFrame, not GeoDataFrame
    """
    Loads zone attributes from CSV and basic properties (like zone_id, name) from GeoJSON properties.
    Does NOT process or store complex geometries. Output is a Pandas DataFrame.
    This version avoids a direct GeoPandas dependency for an easier install on Streamlit Cloud.
    Map visualizations will be limited to what st.map or px.scatter_mapbox can do with lat/lon.
    """
    attr_path = attributes_path or app_config.ZONE_ATTRIBUTES_CSV
    geom_path = geometries_path or app_config.ZONE_GEOMETRIES_GEOJSON # Still used for properties
    logger.info(f"({source_context}) [NoGeoPandas] Loading zone data: Attrs='{os.path.basename(attr_path)}', GeomProps='{os.path.basename(geom_path)}'")
    is_streamlit_active_zone_alt = "streamlit" in sys.modules and hasattr(st, 'session_state') and hasattr(st.session_state, 'run_count')

    error_msgs_load_zone_alt = []
    if not os.path.exists(attr_path): error_msgs_load_zone_alt.append(f"Attributes file missing: {os.path.basename(attr_path)}")
    if not os.path.exists(geom_path): error_msgs_load_zone_alt.append(f"GeoJSON file missing (for properties): {os.path.basename(geom_path)}")
    if error_msgs_load_zone_alt: 
        full_err_str_alt = "; ".join(error_msgs_load_zone_alt)
        logger.error(f"({source_context}) [NoGeoPandas] {full_err_str_alt}")
        if is_streamlit_active_zone_alt : st.error(f"🚨 Zone Data Error: {full_err_str_alt}")
        return None
    
    try:
        # 1. Load attributes from CSV
        attrs_df = pd.read_csv(attr_path); attrs_df = _clean_column_names(attrs_df)
        if 'zone_id' not in attrs_df.columns:
            logger.error(f"({source_context}) [NoGeoPandas] 'zone_id' missing in attributes CSV '{attr_path}'.")
            if is_streamlit_active_zone_alt: st.error("🚨 'zone_id' missing in zone attributes file.")
            return None
        attrs_df['zone_id'] = attrs_df['zone_id'].astype(str).str.strip()
        # Standardize 'name' column
        if 'zone_display_name' in attrs_df.columns and 'name' not in attrs_df.columns: 
            attrs_df.rename(columns={'zone_display_name':'name'}, inplace=True)
        elif 'name' not in attrs_df.columns: 
            attrs_df['name'] = "Zone " + attrs_df['zone_id']

        # 2. Load properties from GeoJSON
        geojson_properties_list = []
        with open(geom_path, 'r', encoding='utf-8') as f_geojson:
            geojson_data = json.load(f_geojson)
            if 'features' in geojson_data and isinstance(geojson_data['features'], list):
                for feature in geojson_data['features']:
                    if 'properties' in feature and isinstance(feature['properties'], dict):
                        props = feature['properties'].copy() # Take a copy of properties
                        # Store geometry as WKT string or centroid if absolutely needed and simple to parse,
                        # otherwise ignore geometry for now to avoid Shapely dependency if possible.
                        # For this simplified version, we mostly care about zone_id and name from properties.
                        if 'geometry' in feature and isinstance(feature['geometry'], dict) and feature['geometry'].get('type') == 'Polygon':
                             # Very basic centroid calculation if needed for st.map. Requires numpy.
                             # coords = feature['geometry']['coordinates'][0] # Assuming no holes
                             # if coords and len(coords) > 0:
                             #    lons = [c[0] for c in coords]; lats = [c[1] for c in coords]
                             #    props['centroid_lon'] = np.mean(lons)
                             #    props['centroid_lat'] = np.mean(lats)
                             props['geometry_type_from_geojson'] = 'Polygon' # Could store type
                        geojson_properties_list.append(props)
            else:
                 logger.warning(f"({source_context}) [NoGeoPandas] GeoJSON file '{geom_path}' does not have a valid 'features' list.")

        geom_props_df = pd.DataFrame(geojson_properties_list)
        if not geom_props_df.empty:
            geom_props_df = _clean_column_names(geom_props_df) # Clean names from GeoJSON properties
            if 'zone_id' not in geom_props_df.columns:
                logger.warning(f"({source_context}) [NoGeoPandas] 'zone_id' missing in GeoJSON properties. Will rely solely on attributes CSV.")
                geom_props_df = pd.DataFrame(columns=['zone_id']) # Empty to avoid merge issues
            else:
                geom_props_df['zone_id'] = geom_props_df['zone_id'].astype(str).str.strip()
        else: # If no properties extracted, create empty df with zone_id to prevent merge error
            geom_props_df = pd.DataFrame(columns=['zone_id'])


        # 3. Merge attributes_df with geom_props_df
        # Prioritize columns from attributes_df in case of name conflicts.
        # Perform an outer merge to keep all zones from both, then decide on source of truth.
        if not geom_props_df.empty and 'zone_id' in geom_props_df.columns:
            merged_df = pd.merge(attrs_df, geom_props_df, on='zone_id', how='outer', suffixes=('_attr', '_geomprop'))
            # Consolidate columns, preferring _attr if both exist
            for col_base in attrs_df.columns:
                if col_base == 'zone_id': continue
                attr_col_name = f"{col_base}_attr"
                geomprop_col_name = f"{col_base}_geomprop"
                if attr_col_name in merged_df.columns and geomprop_col_name in merged_df.columns:
                    merged_df[col_base] = merged_df[attr_col_name].fillna(merged_df[geomprop_col_name])
                    merged_df.drop(columns=[attr_col_name, geomprop_col_name], inplace=True)
                elif attr_col_name in merged_df.columns: # Only in attrs
                    merged_df.rename(columns={attr_col_name: col_base}, inplace=True)
            # For columns only in geom_props_df (that were not in attrs_df)
            for col_base_g in geom_props_df.columns:
                if col_base_g == 'zone_id': continue
                geomprop_col_name_g = f"{col_base_g}_geomprop" # Original name if only from geom_props
                if geomprop_col_name_g in merged_df.columns and col_base_g not in merged_df.columns:
                     merged_df.rename(columns={geomprop_col_name_g: col_base_g}, inplace=True)

        else: # No properties from GeoJSON, just use attributes_df
            merged_df = attrs_df.copy()
            logger.info(f"({source_context}) [NoGeoPandas] No valid properties extracted from GeoJSON, using attributes CSV only.")


        # 4. Ensure essential Sentinel attributes and numeric conversions
        default_zone_attr_map_alt = {
            'name':"Unknown Zone", 'population':0.0, 'num_clinics':0.0, 
            'socio_economic_index':0.5, 'avg_travel_time_clinic_min':30.0, 
            'predominant_hazard_type': "Unknown", 
            'typical_workforce_exposure_level': "Unknown",
            'area_sqkm':np.nan # Area MUST come from attributes if not using GeoPandas .area
        }
        for col_za, def_val_za in default_zone_attr_map_alt.items():
            if col_za not in merged_df.columns: 
                merged_df[col_za] = def_val_za if col_za !='name' else ("Zone " + merged_df.get('zone_id',"").astype(str))
            elif col_za in ['population','socio_economic_index','num_clinics','avg_travel_time_clinic_min','area_sqkm']: 
                merged_df[col_za] = _convert_to_numeric(merged_df.get(col_za), def_val_za)
            elif col_za == 'name' : 
                merged_df[col_za] = merged_df.get(col_za,"Unknown").astype(str).fillna("Zone "+merged_df.get('zone_id',"").astype(str))
        
        # Drop rows where zone_id is NaN after all merges (should not happen if inputs are good)
        merged_df.dropna(subset=['zone_id'], inplace=True)

        logger.info(f"({source_context}) [NoGeoPandas] Zone data loaded and merged (Pandas DF). Shape: {merged_df.shape}")
        return merged_df

    except Exception as e_load_zone_alt: 
        logger.error(f"({source_context}) [NoGeoPandas] Error loading/merging zone data: {e_load_zone_alt}", exc_info=True)
        if is_streamlit_active_zone_alt : st.error(f"Zone data processing error (NoGeoPandas): {e_load_zone_alt}")
        return None

# --- Other functions in core_data_processing.py ---
# enrich_zone_geodata_with_health_aggregates would need to be significantly changed:
# - It would take zone_df (Pandas DataFrame) instead of zone_gdf.
# - All GDF-specific operations (like .crs, .geometry, .area, spatial joins) must be removed.
# - Population density would rely purely on an 'area_sqkm' column from zone_attributes.csv.
# - Map-related outputs or GDF caching specific logic needs removal/adaptation.

# All KPI functions (get_overall_kpis, get_chw_summary, etc.) and get_trend_data, get_supply_forecast_data
# should largely remain UNCHANGED by removing GeoPandas, as they operate on Pandas DataFrames from
# load_health_records or iot_data, or on the *non-geometric attributes* of the (now Pandas DataFrame)
# output of `load_zone_data_no_geopandas` when it's passed to `enrich_zone_data_no_geopandas`.

# THEREFORE, THE FULL Implementations for those KPI/Summary/Trend/Supply functions
# (as painstakingly provided in File 32 - FULL version response) would be included here,
# after this modified `load_zone_data_no_geopandas`.

# For brevity here, I am NOT re-pasting all those long KPI/summary functions again.
# **YOU MUST PASTE THE COMPLETE, FULL VERSIONS of all other functions from the
# previous File 32 (Corrected and Complete) into this file, replacing any stubs.**
# Only `load_zone_data` changes its name and implementation.
# And `enrich_zone_geodata_with_health_aggregates` changes to take a pd.DataFrame.
