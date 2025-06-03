# sentinel_project_root/test/utils/core_data_processing.py
# Core data loading, cleaning, and aggregation utilities for Sentinel Health Co-Pilot.

import streamlit as st
import pandas as pd
import numpy as np
import os
import logging
import json
from config import app_config
from typing import List, Dict, Any, Optional, Union, Callable
from datetime import datetime, date, timedelta
from .protocol_executor import execute_escalation_protocol

logger = logging.getLogger(__name__)

# --- I. Core Helper Functions ---
def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        logger.error(f"_clean_column_names expects a pandas DataFrame, got {type(df)}.")
        return pd.DataFrame()
    df.columns = df.columns.str.lower().str.replace('[^0-9a-zA-Z_]', '_', regex=True).str.replace('_+', '_', regex=True).str.strip('_')
    return df

def _convert_to_numeric(series: pd.Series, default_value: Any = np.nan) -> pd.Series:
    if not isinstance(series, pd.Series):
        try:
            dtype = float if pd.api.types.is_number(default_value) or default_value is np.nan else object
            series = pd.Series(series, dtype=dtype)
        except Exception as e:
            logger.error(f"Could not convert input to Series: {e}")
            length = len(series) if hasattr(series, '__len__') else 1
            return pd.Series([default_value] * length, dtype=type(default_value) if default_value is not np.nan else float)
    return pd.to_numeric(series, errors='coerce').fillna(default_value)

def _robust_merge_agg(
    left_df: pd.DataFrame, right_df: Optional[pd.DataFrame], target_col_name: str,
    on_col: str = 'zone_id', default_fill_value: Any = 0.0
) -> pd.DataFrame:
    if not isinstance(left_df, pd.DataFrame):
        logger.error(f"Left df is not a DataFrame: {type(left_df)}")
        return pd.DataFrame(columns=[on_col, target_col_name])
    left_df = left_df.copy()
    if target_col_name not in left_df.columns:
        left_df[target_col_name] = default_fill_value
    else:
        left_df[target_col_name] = left_df[target_col_name].fillna(default_fill_value)
    if not isinstance(right_df, pd.DataFrame) or right_df.empty or on_col not in right_df.columns:
        return left_df
    value_col = [col for col in right_df.columns if col != on_col][0]
    try:
        left_df[on_col] = left_df[on_col].astype(str).str.strip()
        right_df = right_df[[on_col, value_col]].copy()
        right_df[on_col] = right_df[on_col].astype(str).str.strip()
    except Exception as e:
        logger.error(f"Type conversion error in _robust_merge_agg: {e}")
        return left_df
    temp_agg_col = f"__temp_agg_{target_col_name}_{os.urandom(4).hex()}__"
    right_df.rename(columns={value_col: temp_agg_col}, inplace=True)
    original_index = left_df.index
    left_df = left_df.reset_index()
    left_df = left_df.merge(right_df, on=on_col, how='left')
    if temp_agg_col in left_df.columns:
        left_df[target_col_name] = left_df[temp_agg_col].combine_first(left_df[target_col_name])
        left_df.drop(columns=[temp_agg_col], inplace=True)
    left_df[target_col_name] = left_df[target_col_name].fillna(default_fill_value)
    left_df.set_index('index', inplace=True, drop=True)
    left_df.index.name = None
    return left_df

# --- II. Data Loading and Cleaning Functions ---
@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS)
def load_health_records(file_path: Optional[str] = None, source_context: str = "DataLoader") -> pd.DataFrame:
    actual_file_path = file_path or app_config.HEALTH_RECORDS_CSV
    logger.info(f"({source_context}) Loading health records from: {actual_file_path}")
    if not os.path.exists(actual_file_path):
        logger.error(f"({source_context}) Health records file not found: {actual_file_path}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(actual_file_path, low_memory=False)
        df = _clean_column_names(df)
        logger.info(f"({source_context}) Loaded {len(df)} raw records. Columns: {df.columns.tolist()}")
        date_cols = ['encounter_date', 'sample_collection_date', 'sample_registered_lab_date']
        for col in date_cols:
            df[col] = pd.to_datetime(df.get(col, pd.Series()), errors='coerce')
        if 'encounter_date' in df.columns:
            df['encounter_date_obj'] = df['encounter_date'].dt.date
        numeric_cols_defaults = {
            'age': np.nan, 'min_spo2_pct': np.nan, 'vital_signs_temperature_celsius': np.nan,
            'fall_detected_today': 0, 'ambient_heat_index_c': np.nan, 'test_turnaround_days': np.nan,
            'ai_risk_score': np.nan, 'ai_followup_priority_score': np.nan, 'item_stock_agg_zone': 0,
            'consumption_rate_per_day': 0.0
        }
        for col, default in numeric_cols_defaults.items():
            df[col] = _convert_to_numeric(df.get(col, pd.Series([default] * len(df))), default)
        string_cols = ['encounter_id', 'patient_id', 'gender', 'zone_id', 'clinic_id', 'condition', 'test_type', 'test_result', 'item']
        for col in string_cols:
            df[col] = df.get(col, pd.Series(['Unknown'] * len(df))).astype(str).str.strip().replace(['nan', 'None', 'N/A'], 'Unknown')
        logger.info(f"({source_context}) Health records processed: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading/processing health records: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS)
def load_iot_data(file_path: Optional[str] = None, source_context: str = "DataLoader") -> pd.DataFrame:
    actual_file_path = file_path or app_config.IOT_CLINIC_ENVIRONMENT_CSV
    logger.info(f"({source_context}) Loading IoT data from: {actual_file_path}")
    if not os.path.exists(actual_file_path):
        logger.warning(f"({source_context}) IoT data file not found: {actual_file_path}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(actual_file_path, low_memory=False)
        df = _clean_column_names(df)
        df['timestamp'] = pd.to_datetime(df.get('timestamp', pd.Series()), errors='coerce')
        numeric_cols = ['avg_co2_ppm', 'avg_pm25', 'avg_temp_celsius', 'waiting_room_occupancy']
        for col in numeric_cols:
            df[col] = _convert_to_numeric(df.get(col, pd.Series([np.nan] * len(df))), np.nan)
        string_cols = ['clinic_id', 'room_name', 'zone_id']
        for col in string_cols:
            df[col] = df.get(col, pd.Series(['Unknown'] * len(df))).astype(str).str.strip().replace(['nan', 'None', 'N/A'], 'Unknown')
        logger.info(f"({source_context}) IoT data processed: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading IoT data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS)
def load_zone_data(attributes_path: Optional[str] = None, geometries_path: Optional[str] = None, source_context: str = "DataLoader") -> pd.DataFrame:
    attr_path = attributes_path or app_config.ZONE_ATTRIBUTES_CSV
    geom_path = geometries_path or app_config.ZONE_GEOMETRIES_GEOJSON
    logger.info(f"({source_context}) Loading zone attributes from '{attr_path}' and geometries from '{geom_path}'")
    if not os.path.exists(attr_path) or not os.path.exists(geom_path):
        logger.error(f"({source_context}) Zone data file(s) not found")
        return pd.DataFrame()
    try:
        attrs_df = pd.read_csv(attr_path)
        attrs_df = _clean_column_names(attrs_df)
        with open(geom_path) as f:
            geojson_data = json.load(f)
        geom_df = pd.DataFrame([
            {"zone_id": f["properties"].get("zone_id"), "geometry": json.dumps(f["geometry"])}
            for f in geojson_data["features"]
        ])
        merged_df = attrs_df.merge(geom_df, on="zone_id", how="left")
        if 'name' not in merged_df.columns:
            merged_df['name'] = "Zone " + merged_df['zone_id'].astype(str)
        numeric_cols = {'population': 0.0, 'num_clinics': 0}
        for col, default in numeric_cols.items():
            merged_df[col] = _convert_to_numeric(merged_df.get(col, pd.Series([default] * len(merged_df))), default)
        logger.info(f"({source_context}) Zone data loaded: {len(merged_df)} zones")
        return merged_df
    except Exception as e:
        logger.error(f"Error loading zone data: {e}")
        return pd.DataFrame()

# --- III. Data Enrichment and Aggregation Functions ---
@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS)
def enrich_zone_data_with_health_aggregates(
    zone_df: Optional[pd.DataFrame],
    health_df: Optional[pd.DataFrame],
    iot_df: Optional[pd.DataFrame] = None,
    source_context: str = "DataEnricher"
) -> pd.DataFrame:
    logger.info(f"({source_context}) Starting zone data enrichment")
    if not isinstance(zone_df, pd.DataFrame) or zone_df.empty:
        logger.warning(f"({source_context}) Invalid or empty zone_df")
        return pd.DataFrame()
    enriched_df = zone_df.copy()
    agg_cols = {
        'total_population_health_data': 0, 'avg_risk_score': np.nan,
        'total_patient_encounters': 0, 'total_active_key_infections': 0,
        'zone_avg_co2': np.nan
    }
    for cond in app_config.KEY_CONDITIONS:
        agg_cols[f"active_{cond.lower().replace(' ', '_').replace('-', '_')}_cases"] = 0
    for col, default in agg_cols.items():
        enriched_df[col] = default
    if isinstance(health_df, pd.DataFrame) and not health_df.empty:
        health_df_agg = health_df[health_df['zone_id'].notna()].copy()
        health_df_agg['zone_id'] = health_df_agg['zone_id'].astype(str).str.strip()
        enriched_df = _robust_merge_agg(enriched_df, health_df_agg.groupby('zone_id')['patient_id'].nunique().reset_index(name='count'), 'total_population_health_data')
        enriched_df = _robust_merge_agg(enriched_df, health_df_agg.groupby('zone_id')['ai_risk_score'].mean().reset_index(name='mean_val'), 'avg_risk_score')
        enriched_df = _robust_merge_agg(enriched_df, health_df_agg.groupby('zone_id')['encounter_id'].nunique().reset_index(name='count'), 'total_patient_encounters')
        for condition in app_config.KEY_CONDITIONS:
            col_name = f"active_{condition.lower().replace(' ', '_').replace('-', '_')}_cases"
            mask = health_df_agg['condition'].str.contains(condition, case=False, na=False)
            if mask.any():
                agg_data = health_df_agg[mask].groupby('zone_id')['patient_id'].nunique().reset_index(name='count')
                enriched_df = _robust_merge_agg(enriched_df, agg_data, col_name)
        condition_cols = [f"active_{c.lower().replace(' ', '_').replace('-', '_')}_cases" for c in app_config.KEY_CONDITIONS]
        enriched_df['total_active_key_infections'] = enriched_df[condition_cols].sum(axis=1).fillna(0)
    if isinstance(iot_df, pd.DataFrame) and not iot_df.empty:
        iot_df_agg = iot_df[iot_df['zone_id'].notna()].copy()
        iot_df_agg['zone_id'] = iot_df_agg['zone_id'].astype(str).str.strip()
        enriched_df = _robust_merge_agg(enriched_df, iot_df_agg.groupby('zone_id')['avg_co2_ppm'].mean().reset_index(name='mean_val'), 'zone_avg_co2')
    logger.info(f"({source_context}) Zone data enrichment complete: {enriched_df.shape}")
    return enriched_df

# --- IV. KPI & Summary Calculation Functions ---
@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS)
def get_overall_kpis(health_df: Optional[pd.DataFrame], date_filter_start: Optional[Any]=None, date_filter_end: Optional[Any]=None, source_context: str = "GlobalKPIs") -> Dict[str, Any]:
    logger.info(f"({source_context}) Calculating overall KPIs")
    kpis = {
        "total_patients_period": 0, "avg_patient_ai_risk_period": np.nan,
        "malaria_rdt_positive_rate_period": np.nan, "key_supply_stockout_alerts_period": 0,
        "total_encounters_period": 0
    }
    for cond in app_config.KEY_CONDITIONS:
        kpis[f"active_{cond.lower().replace(' ', '_').replace('-', '_')}_cases_period"] = 0
    if not isinstance(health_df, pd.DataFrame) or health_df.empty:
        return kpis
    df = health_df.copy()
    df['encounter_date'] = pd.to_datetime(df.get('encounter_date', pd.Series()), errors='coerce')
    try:
        start_dt = pd.to_datetime(date_filter_start).date() if date_filter_start else None
        end_dt = pd.to_datetime(date_filter_end).date() if date_filter_end else None
        if start_dt:
            df = df[df['encounter_date'].dt.date >= start_dt]
        if end_dt:
            df = df[df['encounter_date'].dt.date <= end_dt]
    except Exception as e:
        logger.error(f"Error applying date filters: {e}")
    if df.empty:
        return kpis
    kpis["total_patients_period"] = df['patient_id'].nunique()
    kpis["total_encounters_period"] = df['encounter_id'].nunique()
    if 'ai_risk_score' in df.columns:
        kpis["avg_patient_ai_risk_period"] = df['ai_risk_score'].mean()
    for cond in app_config.KEY_CONDITIONS:
        kpi_col = f"active_{cond.lower().replace(' ', '_').replace('-', '_')}_cases_period"
        mask = df['condition'].str.contains(cond, case=False, na=False)
        kpis[kpi_col] = df[mask]['patient_id'].nunique() if mask.any() else 0
    if 'test_type' in df.columns and 'test_result' in df.columns:
        malaria_tests = df[df['test_type'] == 'RDT-Malaria']
        conclusive = malaria_tests[~malaria_tests['test_result'].isin(['Pending', 'Rejected Sample', 'Unknown', 'Indeterminate'])]
        if not conclusive.empty:
            positive = conclusive[conclusive['test_result'] == 'Positive'].shape[0]
            kpis["malaria_rdt_positive_rate_period"] = (positive / len(conclusive)) * 100
    if 'item' in df.columns and 'item_stock_agg_zone' in df.columns:
        latest_stock = df.sort_values('encounter_date').drop_duplicates(['item', 'zone_id'], keep='last')
        latest_stock['days_supply'] = latest_stock['item_stock_agg_zone'] / latest_stock['consumption_rate_per_day'].replace(0, 0.001)
        key_drugs = latest_stock[latest_stock['item'].isin(app_config.KEY_DRUGS)]
        kpis['key_supply_stockout_alerts_period'] = key_drugs[key_drugs['days_supply'] < app_config.CRITICAL_SUPPLY_DAYS]['item'].nunique()
    return kpis

@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS)
def get_chw_summary(health_df_daily: Optional[pd.DataFrame], source_context: str = "CHWSummary") -> Dict[str, Any]:
    logger.info(f"({source_context}) Calculating CHW summary")
    summary = {
        "visits_today": 0, "avg_patient_risk_visited_today": np.nan,
        "high_ai_prio_followups_today": 0, "patients_critical_spo2_today": 0,
        "patients_high_fever_today": 0, "avg_patient_steps_visited_today": np.nan,
        "patients_fall_detected_today": 0, "pending_critical_condition_referrals": 0
    }
    if not isinstance(health_df_daily, pd.DataFrame) or health_df_daily.empty:
        return summary
    df = health_df_daily.copy()
    patient_records = df[~df.get('encounter_type', pd.Series([''] * len(df))).str.contains("WORKER_SELF", case=False, na=False)]
    if not patient_records.empty:
        summary["visits_today"] = patient_records['patient_id'].nunique()
        if 'ai_risk_score' in patient_records.columns:
            summary["avg_patient_risk_visited_today"] = patient_records.drop_duplicates('patient_id')['ai_risk_score'].mean()
        if 'ai_followup_priority_score' in patient_records.columns:
            summary["high_ai_prio_followups_today"] = patient_records[patient_records['ai_followup_priority_score'] >= app_config.FATIGUE_INDEX_HIGH]['patient_id'].nunique()
        if 'min_spo2_pct' in patient_records.columns:
            summary["patients_critical_spo2_today"] = patient_records[patient_records['min_spo2_pct'] < app_config.ALERT_SPO2_CRITICAL]['patient_id'].nunique()
        temp_col = next((c for c in ['vital_signs_temperature_celsius', 'max_skin_temp_celsius'] if c in patient_records.columns), None)
        if temp_col:
            summary["patients_high_fever_today"] = patient_records[patient_records[temp_col] >= app_config.ALERT_BODY_TEMP_FEV]['patient_id'].nunique()
        if 'avg_daily_steps' in patient_records.columns:
            summary["avg_patient_steps_visited_today"] = patient_records.drop_duplicates('patient_id')['avg_daily_steps'].mean()
        if 'fall_detected_today' in patient_records.columns:
            summary["patients_fall_detected_today"] = patient_records[patient_records['fall_detected_today'] > 0]['patient_id'].nunique()
        if 'condition' in patient_records.columns and 'referral_status' in patient_records.columns:
            crit_refs = patient_records[
                (patient_records['referral_status'].str.lower() == 'pending') &
                (patient_records['condition'].str.contains('|'.join(app_config.KEY_CONDITIONS), case=False, na=False))
            ]
            summary["pending_critical_condition_referrals"] = crit_refs['patient_id'].nunique()
    return summary

@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS)
def get_clinic_summary(health_df_period: Optional[pd.DataFrame], source_context: str = "ClinicSummary") -> Dict[str, Any]:
    logger.info(f"({source_context}) Calculating clinic summary")
    summary = {
        "overall_avg_test_turnaround_conclusive_days": np.nan,
        "perc_critical_tests_tat_met": 0.0, "total_pending_critical_tests_patients": 0,
        "sample_rejection_rate_perc": 0.0, "key_drug_stockouts_count": 0,
        "test_summary_details": {}
    }
    if not isinstance(health_df_period, pd.DataFrame) or health_df_period.empty:
        return summary
    df = health_df_period.copy()
    for col in ['test_type', 'test_result', 'test_turnaround_days', 'sample_status', 'patient_id']:
        if col not in df.columns:
            df[col] = "Unknown" if col in ['test_type', 'test_result', 'sample_status'] else np.nan if col == 'test_turnaround_days' else "UnknownPID"
    df['test_turnaround_days'] = _convert_to_numeric(df['test_turnaround_days'], np.nan)
    conclusive_tests = df[~df['test_result'].isin(["Pending", "Rejected Sample", "Unknown", "Indeterminate"])]
    if not conclusive_tests.empty:
        summary["overall_avg_test_turnaround_conclusive_days"] = conclusive_tests['test_turnaround_days'].mean()
    crit_tests = df[df['test_type'].isin(app_config.CRITICAL_TESTS)]
    if not crit_tests.empty:
        crit_conclusive = crit_tests[~crit_tests['test_result'].isin(["Pending", "Rejected Sample", "Unknown", "Indeterminate"])]
        if not crit_conclusive.empty:
            crit_conclusive['tat_met'] = crit_conclusive.apply(
                lambda r: r['test_turnaround_days'] <= app_config.TEST_TYPES.get(r['test_type'], {}).get('tat_days', app_config.TEST_TURNAROUND_DAYS),
                axis=1
            )
            summary["perc_critical_tests_tat_met"] = crit_conclusive['tat_met'].mean() * 100
        summary["total_pending_critical_tests_patients"] = crit_tests[crit_tests['test_result'] == "Pending"]['patient_id'].nunique()
    total_tests = df[df['sample_status'].notna() & (df['sample_status'] != 'Unknown')]['encounter_id'].nunique()
    rejected_tests = df[df['sample_status'] == "Rejected"]['encounter_id'].nunique()
    if total_tests > 0:
        summary["sample_rejection_rate_perc"] = (rejected_tests / total_tests) * 100
    if 'item' in df.columns:
        latest_stock = df.sort_values('encounter_date').drop_duplicates(['item'], keep='last')
        latest_stock['days_supply'] = latest_stock['item_stock_agg_zone'] / latest_stock['consumption_rate_per_day'].replace(0, 0.001)
        key_drugs = latest_stock[latest_stock['item'].isin(app_config.KEY_DRUGS)]
        summary["key_drug_stockouts_count"] = key_drugs[key_drugs['days_supply'] < app_config.CRITICAL_SUPPLY_DAYS]['item'].nunique()
    test_details = {}
    for test, cfg in app_config.TEST_TYPES.items():
        test_df = df[df['test_type'] == test]
        conclusive = test_df[~test_df['test_result'].isin(["Pending", "Rejected Sample", "Unknown", "Indeterminate"])]
        details = {
            "positive_rate_perc": (conclusive[conclusive['test_result'] == 'Positive'].shape[0] / len(conclusive) * 100) if not conclusive.empty else 0.0,
            "avg_tat_days": conclusive['test_turnaround_days'].mean() if not conclusive.empty else np.nan,
            "perc_met_tat_target": (conclusive[conclusive['test_turnaround_days'] <= cfg.get('tat_days', app_config.TEST_TURNAROUND_DAYS)].shape[0] / len(conclusive) * 100) if not conclusive.empty else 0.0,
            "total_conclusive_tests": len(conclusive),
            "pending_count_patients": test_df[test_df['test_result'] == "Pending"]['patient_id'].nunique(),
            "rejected_count_patients": test_df[test_df['sample_status'] == "Rejected"]['patient_id'].nunique()
        }
        test_details[cfg.get("display_name", test)] = details
    summary["test_summary_details"] = test_details
    return summary

@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS)
def get_clinic_environmental_summary(iot_df_period: Optional[pd.DataFrame], source_context: str = "ClinicEnvSummary") -> Dict[str, Any]:
    logger.info(f"({source_context}) Calculating clinic environmental summary")
    summary = {
        "avg_co2_overall_ppm": np.nan, "rooms_co2_very_high_alert_latest_count": 0,
        "avg_pm25_overall_ugm3": np.nan, "rooms_pm25_very_high_alert_latest_count": 0,
        "avg_waiting_room_occupancy_overall_persons": np.nan, "waiting_room_high_occupancy_alert_latest_flag": False,
        "avg_noise_overall_dba": np.nan, "rooms_noise_high_alert_latest_count": 0,
        "avg_temp_overall_celsius": np.nan, "avg_humidity_overall_rh": np.nan,
        "latest_readings_timestamp": None
    }
    if not isinstance(iot_df_period, pd.DataFrame) or iot_df_period.empty:
        return summary
    df = iot_df_period.copy()
    df['timestamp'] = pd.to_datetime(df.get('timestamp', pd.Series()), errors='coerce')
    df = df.dropna(subset=['timestamp'])
    if df.empty:
        return summary
    summary["latest_readings_timestamp"] = df['timestamp'].max()
    if 'avg_co2_ppm' in df.columns:
        summary["avg_co2_overall_ppm"] = df['avg_co2_ppm'].mean()
    if 'avg_pm25' in df.columns:
        summary["avg_pm25_overall_ugm3"] = df['avg_pm25'].mean()
    if 'avg_noise_db' in df.columns:
        summary["avg_noise_overall_dba"] = df['avg_noise_db'].mean()
    if 'avg_temp_celsius' in df.columns:
        summary["avg_temp_overall_celsius"] = df['avg_temp_celsius'].mean()
    if 'avg_humidity_rh' in df.columns:
        summary["avg_humidity_overall_rh"] = df['avg_humidity_rh'].mean()
    latest_rooms = df.sort_values('timestamp').drop_duplicates(['clinic_id', 'room_name'], keep='last')
    if 'avg_co2_ppm' in latest_rooms.columns:
        summary["rooms_co2_very_high_alert_latest_count"] = latest_rooms[latest_rooms['avg_co2_ppm'] > app_config.ALERT_AMBIENT_CO2_VERY_HIGH].shape[0]
    if 'avg_pm25' in latest_rooms.columns:
        summary["rooms_pm25_very_high_alert_latest_count"] = latest_rooms[latest_rooms['avg_pm25'] > app_config.ALERT_AMBIENT_PM25_VERY_HIGH].shape[0]
    if 'avg_noise_db' in latest_rooms.columns:
        summary["rooms_noise_high_alert_latest_count"] = latest_rooms[latest_rooms['avg_noise_db'] > app_config.ALERT_AMBIENT_NOISE_HIGH].shape[0]
    waiting_rooms = latest_rooms[latest_rooms['room_name'].str.contains("Waiting", case=False, na=False)]
    if not waiting_rooms.empty and 'waiting_room_occupancy' in waiting_rooms.columns:
        summary["avg_waiting_room_occupancy_overall_persons"] = waiting_rooms['waiting_room_occupancy'].mean()
        summary["waiting_room_high_occupancy_alert_latest_flag"] = (waiting_rooms['waiting_room_occupancy'] > app_config.CLINIC_WAITING_ROOM_MAX).any()
    return summary

@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS)
def get_patient_alerts_for_clinic(health_df_period: Optional[pd.DataFrame], source_context: str = "ClinicPatientAlerts") -> pd.DataFrame:
    logger.info(f"({source_context}) Generating patient alerts for clinic")
    cols = ['patient_id', 'encounter_date', 'condition', 'Alert Reason', 'Priority Score', 'ai_risk_score', 'age', 'gender', 'zone_id', 'referred_to_facility_id']
    if not isinstance(health_df_period, pd.DataFrame) or health_df_period.empty:
        return pd.DataFrame(columns=cols)
    df = health_df_period.copy()
    for col, default in {
        'patient_id': "UnknownPID", 'encounter_date': pd.NaT, 'condition': "N/A",
        'ai_risk_score': np.nan, 'ai_followup_priority_score': np.nan, 'min_spo2_pct': np.nan, 
        'vital_signs_temperature_celsius': np.nan, 'max_skin_temp_celsius': np.nan, 
        'referral_status': "Unknown", 'age': np.nan, 'gender': "Unknown", 
        'zone_id': "UnknownZone", 'referred_to_facility_id': "UnknownFacility"
    }.items():
        df[col] = _convert_to_numeric(df.get(col, pd.Series([default] * len(df))), default) if col in ['ai_risk_score', 'ai_followup_priority_score', 'min_spo2_pct', 'vital_signs_temperature_celsius', 'max_skin_temp_celsius', 'age'] else df.get(col, pd.Series([default] * len(df))).fillna(default)
    df['encounter_date'] = pd.to_datetime(df['encounter_date'], errors='coerce')
    alerts = []
    for _, row in df.iterrows():
        alert_reason = None
        priority = row.get('ai_followup_priority_score', row.get('ai_risk_score', 0))
        if pd.notna(row['ai_risk_score']) and row['ai_risk_score'] >= app_config.RISK_SCORE_MODERATE:
            alert_reason = f"High AI Risk ({row['ai_risk_score']:.0f})"
            priority = max(priority, row['ai_risk_score'])
        if pd.notna(row['min_spo2_pct']) and row['min_spo2_pct'] < app_config.ALERT_SPO2_CRITICAL:
            alert_reason = f"Critical SpO2 ({row['min_spo2_pct']:.0f}%)"
            priority = max(priority, 95)
            execute_escalation_protocol("PATIENT_CRITICAL_SPO2_LOW", row.to_dict())
        temp_val = row.get('vital_signs_temperature_celsius', row.get('max_skin_temp_celsius'))
        if pd.notna(temp_val) and temp_val >= app_config.ALERT_BODY_TEMP_FEV:
            alert_reason = f"High Fever ({temp_val:.1f}°C)"
            priority = max(priority, 90)
        if row['referral_status'].lower() == 'pending' and any(kc.lower() in row['condition'].lower() for kc in app_config.KEY_CONDITIONS):
            alert_reason = f"Pending Critical Referral: {row['condition']}"
            priority = max(priority, 85)
        if alert_reason:
            alerts.append({
                'patient_id': row['patient_id'], 'encounter_date': row['encounter_date'],
                'condition': row['condition'], 'Alert Reason': alert_reason,
                'Priority Score': round(min(priority, 100), 1), 'ai_risk_score': row['ai_risk_score'],
                'age': row['age'], 'gender': row['gender'], 'zone_id': row['zone_id'],
                'referred_to_facility_id': row['referred_to_facility_id']
            })
    if not alerts:
        return pd.DataFrame(columns=cols)
    alerts_df = pd.DataFrame(alerts).sort_values('Priority Score', ascending=False)
    alerts_df = alerts_df.drop_duplicates('patient_id', keep='first')
    logger.info(f"({source_context}) Generated {len(alerts_df)} clinic alerts")
    return alerts_df[cols].head(50)

@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS)
def get_district_summary_kpis(enriched_zone_df: Optional[pd.DataFrame], source_context: str = "DistrictKPIs") -> Dict[str, Any]:
    logger.info(f"({source_context}) Calculating district KPIs")
    kpis = {
        "total_zones_in_df": 0, "total_population_district": 0.0,
        "population_weighted_avg_ai_risk_score": np.nan,
        "zones_meeting_high_risk_criteria_count": 0,
        "district_avg_facility_coverage_score": np.nan,
        "district_overall_key_disease_prevalence_per_1000": np.nan,
        "district_population_weighted_avg_steps": np.nan,
        "district_avg_clinic_co2_ppm": np.nan
    }
    for cond in app_config.KEY_CONDITIONS:
        kpis[f"district_total_active_{cond.lower().replace(' ', '_').replace('-', '_')}_cases"] = 0
    if not isinstance(enriched_zone_df, pd.DataFrame) or enriched_zone_df.empty:
        return kpis
    df = enriched_zone_df.copy()
    kpis["total_zones_in_df"] = df['zone_id'].nunique()
    df['population'] = _convert_to_numeric(df.get('population', pd.Series([0.0] * len(df))), 0.0)
    total_pop = df['population'].sum()
    kpis["total_population_district"] = total_pop
    if total_pop > 0:
        if 'avg_risk_score' in df.columns:
            kpis["population_weighted_avg_ai_risk_score"] = (df['avg_risk_score'].fillna(0) * df['population']).sum() / total_pop
        if 'avg_daily_steps_zone' in df.columns:
            kpis["district_population_weighted_avg_steps"] = (df['avg_daily_steps_zone'].fillna(app_config.TARGET_DAILY_STEPS * 0.5) * df['population']).sum() / total_pop
    if 'avg_risk_score' in df.columns:
        kpis["zones_meeting_high_risk_criteria_count"] = df[df['avg_risk_score'] >= app_config.DISTRICT_ZONE_HIGH_RISK_SCORE].shape[0]
    total_infections = 0
    for cond in app_config.KEY_CONDITIONS:
        col = f"active_{cond.lower().replace(' ', '_').replace('-', '_')}_cases"
        if col in df.columns:
            sum_val = df[col].sum()
            kpis[f"district_total_active_{cond.lower().replace(' ', '_').replace('-', '_')}_cases"] = sum_val
            total_infections += sum_val
    if total_pop > 0:
        kpis["district_overall_key_disease_prevalence_per_1000"] = (total_infections / total_pop) * 1000
    if 'zone_avg_co2' in df.columns:
        kpis["district_avg_clinic_co2_ppm"] = df['zone_avg_co2'].mean()
    return kpis

@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS)
def get_trend_data(
    df: Optional[pd.DataFrame], value_col: str, date_col: str = 'encounter_date',
    period: str = 'D', agg_func: Union[str, Callable] = 'mean',
    filter_col: Optional[str] = None, filter_val: Optional[Any] = None,
    source_context: str = "TrendCalculator"
) -> pd.Series:
    logger.debug(f"({source_context}) Generating trend for '{value_col}'")
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.Series(dtype='float64')
    df = df.copy()
    if date_col not in df.columns or value_col not in df.columns:
        logger.error(f"Missing {date_col} or {value_col}")
        return pd.Series(dtype='float64')
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col, value_col])
    if filter_col and filter_val is not None and filter_col in df.columns:
        df = df[df[filter_col] == filter_val]
    if df.empty:
        return pd.Series(dtype='float64')
    try:
        if isinstance(agg_func, str) and agg_func in ['mean', 'sum', 'median']:
            df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
            df = df.dropna(subset=[value_col])
        trend = df.set_index(date_col)[value_col].resample(period).agg(agg_func)
        if agg_func in ['count', 'nunique']:
            trend = trend.fillna(0)
        return trend
    except Exception as e:
        logger.error(f"Error generating trend: {e}")
        return pd.Series(dtype='float64')

@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS)
def get_supply_forecast_data(
    health_df: Optional[pd.DataFrame], forecast_days_out: int = 30,
    item_filter_list: Optional[List[str]] = None, source_context: str = "SupplyForecast"
) -> pd.DataFrame:
    logger.info(f"({source_context}) Generating supply forecast")
    cols = ['item', 'date', 'forecasted_stock_level', 'forecasted_days_of_supply', 'estimated_stockout_date_linear', 'initial_stock_at_forecast_start', 'base_consumption_rate_per_day']
    if not isinstance(health_df, pd.DataFrame) or health_df.empty:
        return pd.DataFrame(columns=cols)
    df = health_df[['item', 'encounter_date', 'item_stock_agg_zone', 'consumption_rate_per_day']].copy()
    df['encounter_date'] = pd.to_datetime(df['encounter_date'], errors='coerce')
    df = df.dropna(subset=['encounter_date', 'item'])
    df['item_stock_agg_zone'] = _convert_to_numeric(df['item_stock_agg_zone'], 0)
    df['consumption_rate_per_day'] = _convert_to_numeric(df['consumption_rate_per_day'], 0.001).replace(0, 0.001)
    if item_filter_list:
        df = df[df['item'].isin(item_filter_list)]
    if df.empty:
        return pd.DataFrame(columns=cols)
    latest_status = df.sort_values('encounter_date').drop_duplicates('item', keep='last')
    records = []
    start_date = pd.Timestamp(date.today())
    for _, row in latest_status.iterrows():
        item = row['item']
        stock = row['item_stock_agg_zone']
        consumption = row['consumption_rate_per_day']
        initial_dos = stock / consumption if consumption > 0 else np.inf
        stockout_date = start_date + pd.to_timedelta(initial_dos, unit='D') if np.isfinite(initial_dos) else pd.NaT
        running_stock = stock
        for day in range(forecast_days_out):
            fc_date = start_date + pd.Timedelta(days=day)
            running_stock = max(0, stock - (consumption * day))
            dos = running_stock / consumption if consumption > 0 else np.inf
            records.append({
                'item': item, 'date': fc_date, 'forecasted_stock_level': running_stock,
                'forecasted_days_of_supply': dos, 'estimated_stockout_date_linear': stockout_date,
                'initial_stock_at_forecast_start': stock, 'base_consumption_rate_per_day': consumption
            })
    if not records:
        return pd.DataFrame(columns=cols)
    final_df = pd.DataFrame(records)
    final_df['estimated_stockout_date_linear'] = pd.to_datetime(final_df['estimated_stockout_date_linear'], errors='coerce')
    return final_df
