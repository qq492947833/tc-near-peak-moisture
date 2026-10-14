"""IBTrACS preprocessing used by the wind-defined lifecycle algorithm.

Only a single internal missing USA wind bracketed by exact six-hour valid
reports is filled. WMO winds are not mixed into the USA wind series.
"""
import numpy as np
import pandas as pd

def prepare_records(raw, start_year=1982, end_year=2023):
    df = raw.copy()
    df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'], errors='coerce')
    df = df[df['ISO_TIME'].notna()].copy()
    numeric_columns = ['SEASON', 'NUMBER', 'LAT', 'LON', 'WMO_WIND', 'WMO_PRES', 'DIST2LAND', 'LANDFALL', 'USA_WIND', 'USA_PRES']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in ['SEASON', 'NUMBER', 'BASIN', 'SUBBASIN', 'NAME', 'NATURE', 'WMO_WIND', 'WMO_PRES', 'TRACK_TYPE', 'DIST2LAND', 'LANDFALL', 'USA_STATUS', 'USA_RECORD', 'USA_WIND', 'USA_PRES']:
        if col not in df.columns:
            df[col] = np.nan
    study_window_mask = (df['ISO_TIME'].dt.year >= start_year) & (df['ISO_TIME'].dt.year <= end_year)
    candidate_sids = df.loc[study_window_mask & df['SID'].notna(), 'SID'].astype(str).unique()
    df = df[df['SID'].astype(str).isin(candidate_sids)].copy()
    synoptic_mask = df['ISO_TIME'].dt.hour.isin([0, 6, 12, 18]) & (df['ISO_TIME'].dt.minute == 0) & (df['ISO_TIME'].dt.second == 0)
    df = df[synoptic_mask].copy()
    track_type = df['TRACK_TYPE'].astype('string').str.strip().str.upper()
    df = df[track_type.isna() | (track_type == 'MAIN')].copy()
    df = df.sort_values(['SID', 'ISO_TIME']).drop_duplicates(subset=['SID', 'ISO_TIME'], keep='first').reset_index(drop=True)
    df['WIND_ORIGINAL'] = df['USA_WIND'].astype(float)
    df['WIND'] = df['WIND_ORIGINAL'].copy()
    df['WIND_WAS_FILLED'] = False
    grp = df.groupby('SID', sort=False)
    prev_time = grp['ISO_TIME'].shift(1)
    next_time = grp['ISO_TIME'].shift(-1)
    prev_wind = grp['WIND'].shift(1)
    next_wind = grp['WIND'].shift(-1)
    single_6h_gap = df['WIND'].isna() & prev_wind.notna() & next_wind.notna() & (df['ISO_TIME'] - prev_time == pd.Timedelta(hours=6)) & (next_time - df['ISO_TIME'] == pd.Timedelta(hours=6))
    df.loc[single_6h_gap, 'WIND'] = (prev_wind[single_6h_gap].to_numpy() + next_wind[single_6h_gap].to_numpy()) / 2.0
    df.loc[single_6h_gap, 'WIND_WAS_FILLED'] = True
    df['PRES'] = df['USA_PRES'].astype(float)
    pressure_fallback = df['PRES'].isna() & df['WMO_PRES'].notna()
    df.loc[pressure_fallback, 'PRES'] = df.loc[pressure_fallback, 'WMO_PRES'].astype(float)
    tropical_usa_status = {'TD', 'TS', 'TY', 'ST', 'TC', 'HU', 'HR'}
    usa_status = df['USA_STATUS'].astype('string').str.strip().str.upper()
    nature = df['NATURE'].astype('string').str.strip().str.upper()
    status_available = usa_status.notna() & (usa_status != '')
    df['IS_TROPICAL'] = usa_status.isin(tropical_usa_status)
    df.loc[~status_available, 'IS_TROPICAL'] = nature[~status_available].eq('TS')
    df['IS_TROPICAL'] = df['IS_TROPICAL'].fillna(False).astype(bool)
    df['LAT'] = pd.to_numeric(df['LAT'], errors='coerce')
    df['LON'] = pd.to_numeric(df['LON'], errors='coerce')
    df['LON_180'] = (df['LON'] + 180.0) % 360.0 - 180.0
    df['LON_360'] = df['LON'] % 360.0
    key_columns = ['SID', 'SEASON', 'BASIN', 'NAME', 'ISO_TIME', 'LAT', 'LON_180', 'LON_360', 'WIND', 'WIND_ORIGINAL', 'WIND_WAS_FILLED', 'PRES', 'IS_TROPICAL', 'DIST2LAND', 'LANDFALL', 'NATURE', 'USA_STATUS']
    ibtracs = df[key_columns].copy()
    return ibtracs
