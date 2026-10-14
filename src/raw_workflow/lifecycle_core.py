"""Original wind-defined lifecycle algorithm, without notebook display or file writes.

Inputs use the documented six-hour USA wind series. The first-LMI maturity
anchor and original four-stage quality criteria are preserved.
"""
import numpy as np
import pandas as pd

def _haversine_km(lat1, lon1, lat2, lon2):
    if any((pd.isna(v) for v in [lat1, lon1, lat2, lon2])):
        return np.nan
    r = 6371.0088
    p1 = np.deg2rad(lat1)
    p2 = np.deg2rad(lat2)
    dp = np.deg2rad(lat2 - lat1)
    dl = np.deg2rad(lon2 - lon1)
    a = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(a))

def _count_true_runs(mask):
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return 0
    return int(mask[0]) + int(np.sum(mask[1:] & ~mask[:-1]))

def _hours(t1, t0):
    return (pd.Timestamp(t1) - pd.Timestamp(t0)).total_seconds() / 3600.0

def identify_tc_crisis_lifecycle(time, lat, lon, wind, pressure=None, basin=None, is_tropical=None, dist2land=None, landfall=None, wind_filled_flag=None, sid='', name='', season=np.nan, genesis_wind=35.0, genesis_lat_limit=40.0, min_lifecycle_points=5, min_peak_gain=15.0, min_stage_duration_hours=6.0, min_wind_valid_fraction=0.8, max_gap_hours=12.0, escalation_progress=0.25, maturity_progress=0.9, escalation_rate_threshold=10.0, resolution_rate_threshold=10.0, ri_threshold=30.0, rw_threshold=30.0, reintensification_threshold=10.0):
    time = pd.to_datetime(np.asarray(time), errors='coerce')
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    wind = np.asarray(wind, dtype=float)
    n = len(time)

    def _optional_numeric(x):
        if x is None:
            return np.full(n, np.nan, dtype=float)
        return np.asarray(x, dtype=float)
    pressure = _optional_numeric(pressure)
    dist2land = _optional_numeric(dist2land)
    landfall = _optional_numeric(landfall)
    if basin is None:
        basin = np.full(n, '', dtype=object)
    else:
        basin = np.asarray(basin, dtype=object)
    if is_tropical is None:
        is_tropical = np.ones(n, dtype=bool)
    else:
        is_tropical = np.asarray(is_tropical, dtype=bool)
    if wind_filled_flag is None:
        wind_filled_flag = np.zeros(n, dtype=bool)
    else:
        wind_filled_flag = np.asarray(wind_filled_flag, dtype=bool)
    x = pd.DataFrame({'time': time, 'lat': lat, 'lon': lon, 'wind': wind, 'pressure': pressure, 'basin': basin, 'is_tropical': is_tropical, 'dist2land': dist2land, 'landfall': landfall, 'wind_filled_flag': wind_filled_flag})
    x = x[x['time'].notna()].sort_values('time').drop_duplicates('time', keep='first').reset_index(drop=True)
    empty_track_columns = ['time', 'lat', 'lon', 'lon_360', 'wind', 'pressure', 'basin', 'dist2land', 'landfall', 'landfall_flag', 'wind_filled_flag', 'delta_v24_forward', 'delta_v24_backward', 'stage_code']

    def _invalid(reason, **extra):
        summary = {'sid': str(sid), 'name': '' if pd.isna(name) else str(name), 'season': float(season) if pd.notna(season) else np.nan, 'valid_lifecycle': 0, 'qc_reason': reason}
        summary.update(extra)
        return {'summary': summary, 'track': pd.DataFrame(columns=empty_track_columns)}
    if len(x) == 0:
        return _invalid('no_valid_time')
    genesis_candidates = x['is_tropical'].to_numpy() & x['wind'].notna().to_numpy() & (x['wind'].to_numpy() >= genesis_wind)
    genesis_idx_all = np.where(genesis_candidates)[0]
    if genesis_idx_all.size == 0:
        return _invalid('no_tropical_storm_genesis')
    g = int(genesis_idx_all[0])
    if genesis_lat_limit is not None:
        if pd.isna(x.loc[g, 'lat']) or abs(float(x.loc[g, 'lat'])) > float(genesis_lat_limit):
            return _invalid('genesis_outside_latitude_limit', genesis_lat=float(x.loc[g, 'lat']) if pd.notna(x.loc[g, 'lat']) else np.nan)
    tropical_after_g = np.where(x['is_tropical'].to_numpy() & (np.arange(len(x)) >= g))[0]
    if tropical_after_g.size == 0:
        return _invalid('no_tropical_phase_after_genesis')
    e = int(tropical_after_g[-1])
    life = x.iloc[g:e + 1].copy().reset_index(drop=True)
    life['lon_360'] = life['lon'] % 360.0
    if len(life) < min_lifecycle_points:
        return _invalid('too_few_lifecycle_points', n_lifecycle_obs=int(len(life)))
    if len(life) >= 2:
        all_gaps = life['time'].diff().dt.total_seconds().div(3600.0).dropna()
        max_record_gap = float(all_gaps.max()) if len(all_gaps) else 0.0
    else:
        max_record_gap = 0.0
    wind_valid = life['wind'].notna()
    wind_valid_fraction = float(wind_valid.mean())
    valid_wind_times = life.loc[wind_valid, 'time']
    if len(valid_wind_times) >= 2:
        wind_gaps = valid_wind_times.diff().dt.total_seconds().div(3600.0).dropna()
        max_wind_gap = float(wind_gaps.max()) if len(wind_gaps) else 0.0
    else:
        max_wind_gap = np.inf
    if wind_valid_fraction < min_wind_valid_fraction:
        return _invalid('low_wind_valid_fraction', n_lifecycle_obs=int(len(life)), wind_valid_fraction=wind_valid_fraction, max_record_gap_hours=max_record_gap, max_wind_gap_hours=max_wind_gap)
    if max_wind_gap > max_gap_hours:
        return _invalid('wind_gap_too_large', n_lifecycle_obs=int(len(life)), wind_valid_fraction=wind_valid_fraction, max_record_gap_hours=max_record_gap, max_wind_gap_hours=max_wind_gap)
    w = life['wind'].to_numpy(dtype=float)
    finite_w = np.isfinite(w)
    if not finite_w.any():
        return _invalid('no_valid_wind_in_lifecycle')
    lmi = float(np.nanmax(w))
    lmi_indices = np.where(np.isclose(w, lmi, equal_nan=False))[0]
    lmi_first_idx = int(lmi_indices[0])
    lmi_last_idx = int(lmi_indices[-1])
    genesis_wind_actual = float(w[0])
    peak_gain = lmi - genesis_wind_actual
    if peak_gain < min_peak_gain:
        return _invalid('peak_gain_too_small_for_four_stage_framework', genesis_wind=genesis_wind_actual, lmi=lmi, peak_gain=peak_gain, n_lifecycle_obs=int(len(life)), wind_valid_fraction=wind_valid_fraction, max_record_gap_hours=max_record_gap, max_wind_gap_hours=max_wind_gap)
    tindex = pd.DatetimeIndex(life['time'])
    s = pd.Series(w, index=tindex)
    future_w = s.reindex(tindex + pd.Timedelta(hours=24)).to_numpy(dtype=float)
    past_w = s.reindex(tindex - pd.Timedelta(hours=24)).to_numpy(dtype=float)
    d24_fwd = future_w - w
    d24_back = w - past_w
    life['delta_v24_forward'] = d24_fwd
    life['delta_v24_backward'] = d24_back
    progress = (w - genesis_wind_actual) / peak_gain
    mature_threshold_wind = genesis_wind_actual + maturity_progress * peak_gain
    maturity_start_idx = lmi_first_idx
    while maturity_start_idx > 0:
        prev = maturity_start_idx - 1
        gap_h = _hours(life.loc[maturity_start_idx, 'time'], life.loc[prev, 'time'])
        if gap_h <= max_gap_hours and np.isfinite(w[prev]) and (w[prev] >= mature_threshold_wind):
            maturity_start_idx = prev
        else:
            break
    pre_maturity_idx = np.arange(len(life)) <= maturity_start_idx
    below_progress = np.where(pre_maturity_idx & np.isfinite(progress) & (progress < escalation_progress))[0]
    if below_progress.size > 0:
        escalation_progress_idx = min(int(below_progress[-1]) + 1, maturity_start_idx)
    else:
        escalation_progress_idx = 0
    while escalation_progress_idx < maturity_start_idx and (not np.isfinite(progress[escalation_progress_idx]) or progress[escalation_progress_idx] < escalation_progress):
        escalation_progress_idx += 1
    rate_candidates = np.where((np.arange(len(life)) >= escalation_progress_idx) & (np.arange(len(life)) <= maturity_start_idx) & np.isfinite(d24_fwd) & (d24_fwd >= escalation_rate_threshold))[0]
    escalation_rate_found = rate_candidates.size > 0
    if escalation_rate_found:
        escalation_rate_idx = int(rate_candidates[0])
        escalation_idx = max(escalation_progress_idx, escalation_rate_idx)
    else:
        escalation_idx = escalation_progress_idx
    escalation_idx = int(min(escalation_idx, maturity_start_idx))
    high_band = (np.arange(len(life)) >= maturity_start_idx) & np.isfinite(w) & (w >= mature_threshold_wind)
    high_indices = np.where(high_band)[0]
    if high_indices.size > 0:
        last_high_idx = int(high_indices[-1])
    else:
        last_high_idx = lmi_last_idx
    resolution_idx = min(last_high_idx + 1, len(life) - 1)
    resolution_fallback = False
    if resolution_idx < len(life):
        back_change = d24_back[resolution_idx]
        resolution_rate_confirmed = bool(np.isfinite(back_change) and back_change <= -resolution_rate_threshold)
    else:
        resolution_rate_confirmed = False
    if not resolution_rate_confirmed:
        resolution_fallback = True
    stage = np.zeros(len(life), dtype=np.int8)
    idx = np.arange(len(life))
    stage[(idx >= 0) & (idx < escalation_idx)] = 1
    stage[(idx >= escalation_idx) & (idx < maturity_start_idx)] = 2
    stage[(idx >= maturity_start_idx) & (idx < resolution_idx)] = 3
    stage[idx >= resolution_idx] = 4
    stage[stage == 0] = 1
    life['stage_code'] = stage
    t_lmi_first = life.loc[lmi_first_idx, 'time']
    mir_mask = np.isfinite(d24_fwd) & (life['time'] + pd.Timedelta(hours=24) <= t_lmi_first)
    if mir_mask.any():
        mir_idx = int(np.nanargmax(np.where(mir_mask, d24_fwd, np.nan)))
        mir = float(d24_fwd[mir_idx])
    else:
        mir_idx = -1
        mir = np.nan
    t_lmi_last = life.loc[lmi_last_idx, 'time']
    t_end = life.loc[len(life) - 1, 'time']
    mwr_mask = np.isfinite(d24_fwd) & (life['time'] >= t_lmi_last) & (life['time'] + pd.Timedelta(hours=24) <= t_end)
    if mwr_mask.any():
        mwr_idx = int(np.nanargmax(np.where(mwr_mask, -d24_fwd, np.nan)))
        mwr = float(-d24_fwd[mwr_idx])
    else:
        mwr_idx = -1
        mwr = np.nan
    pre_lmi_mask = np.isfinite(d24_fwd) & (life['time'] + pd.Timedelta(hours=24) <= t_lmi_first)
    positive_intensification = pre_lmi_mask & (d24_fwd > 0)
    intensification_duration_h = float(np.sum(positive_intensification) * 6.0)
    mean_intensification_rate24 = float(np.nanmean(d24_fwd[positive_intensification])) if positive_intensification.any() else np.nan
    post_lmi_mask = np.isfinite(d24_fwd) & (life['time'] >= t_lmi_last) & (life['time'] + pd.Timedelta(hours=24) <= t_end)
    weakening_windows = post_lmi_mask & (d24_fwd < 0)
    weakening_duration_h = float(np.sum(weakening_windows) * 6.0)
    mean_weakening_rate24 = float(np.nanmean(-d24_fwd[weakening_windows])) if weakening_windows.any() else np.nan
    ri_event_count = _count_true_runs(positive_intensification & (d24_fwd >= ri_threshold))
    rw_event_count = _count_true_runs(weakening_windows & (-d24_fwd >= rw_threshold))
    post_first_lmi_reint = (np.arange(len(life)) >= lmi_first_idx) & np.isfinite(d24_fwd) & (d24_fwd >= reintensification_threshold)
    reintensification_count = _count_true_runs(post_first_lmi_reint)
    near_lmi_episode_count = _count_true_runs(high_band)
    t_genesis = life.loc[0, 'time']
    t_escalation = life.loc[escalation_idx, 'time']
    t_maturity = life.loc[maturity_start_idx, 'time']
    t_resolution = life.loc[resolution_idx, 'time']
    t_termination = life.loc[len(life) - 1, 'time']
    lifetime_h = _hours(t_termination, t_genesis)
    development_duration_h = _hours(t_escalation, t_genesis)
    escalation_duration_h = _hours(t_maturity, t_escalation)
    maturity_duration_h = _hours(t_resolution, t_maturity)
    resolution_duration_h = _hours(t_termination, t_resolution)
    pre_lmi_duration_h = _hours(t_lmi_first, t_genesis)
    post_lmi_duration_h = _hours(t_termination, t_lmi_last)
    stage_durations = {'development_duration_hours': development_duration_h, 'escalation_duration_hours': escalation_duration_h, 'maturity_duration_hours': maturity_duration_h, 'resolution_duration_hours': resolution_duration_h}
    short_stages = [stage_name for stage_name, duration_h in stage_durations.items() if not np.isfinite(duration_h) or duration_h < min_stage_duration_hours]
    if short_stages:
        return _invalid('stage_duration_too_short', genesis_wind=genesis_wind_actual, lmi=lmi, peak_gain=peak_gain, n_lifecycle_obs=int(len(life)), wind_valid_fraction=wind_valid_fraction, max_record_gap_hours=max_record_gap, max_wind_gap_hours=max_wind_gap, development_duration_hours=development_duration_h, escalation_duration_hours=escalation_duration_h, maturity_duration_hours=maturity_duration_h, resolution_duration_hours=resolution_duration_h, short_stage_names=';'.join(short_stages))

    def _endpoint_rate(i0, i1, weakening=False):
        if i1 <= i0:
            return np.nan
        dt_h = _hours(life.loc[i1, 'time'], life.loc[i0, 'time'])
        if dt_h <= 0 or not (np.isfinite(w[i0]) and np.isfinite(w[i1])):
            return np.nan
        dv = w[i1] - w[i0]
        if weakening:
            dv = -dv
        return float(dv / dt_h * 24.0)
    development_rate24 = _endpoint_rate(0, escalation_idx, weakening=False)
    escalation_rate24 = _endpoint_rate(escalation_idx, maturity_start_idx, weakening=False)
    resolution_rate24 = _endpoint_rate(resolution_idx, len(life) - 1, weakening=True)
    denom_t = pre_lmi_duration_h + post_lmi_duration_h
    lifecycle_asymmetry_time = (pre_lmi_duration_h - post_lmi_duration_h) / denom_t if denom_t > 0 else np.nan
    if np.isfinite(mean_intensification_rate24) and np.isfinite(mean_weakening_rate24) and (mean_intensification_rate24 > 0) and (mean_weakening_rate24 > 0):
        lifecycle_asymmetry_rate = float(np.log(mean_intensification_rate24 / mean_weakening_rate24))
    else:
        lifecycle_asymmetry_rate = np.nan
    ts_mask = np.isfinite(w) & (w >= 34.0)
    ace = float(0.0001 * np.nansum(w[ts_mask] ** 2))
    wind_ms = w[ts_mask] * 0.514444
    pdi = float(np.nansum(wind_ms ** 3 * (6.0 * 3600.0)))
    track_distance_km = 0.0
    segment_speeds = []
    for i in range(len(life) - 1):
        d = _haversine_km(life.loc[i, 'lat'], life.loc[i, 'lon'], life.loc[i + 1, 'lat'], life.loc[i + 1, 'lon'])
        dt_h = _hours(life.loc[i + 1, 'time'], life.loc[i, 'time'])
        if np.isfinite(d) and dt_h > 0:
            track_distance_km += d
            if dt_h <= max_gap_hours:
                segment_speeds.append(d / dt_h)
    mean_translation_speed_kmh = float(np.mean(segment_speeds)) if segment_speeds else np.nan
    landfall_flag = np.isfinite(life['landfall'].to_numpy(dtype=float)) & np.isclose(life['landfall'].to_numpy(dtype=float), 0.0)
    life['landfall_flag'] = landfall_flag.astype(np.int8)
    landfall_event_count = _count_true_runs(landfall_flag)
    if landfall_flag.any():
        lf_idx = int(np.where(landfall_flag)[0][0])
        first_landfall_time = life.loc[lf_idx, 'time']
        first_landfall_stage_code = int(stage[lf_idx])
        first_landfall_wind = float(w[lf_idx]) if np.isfinite(w[lf_idx]) else np.nan
        first_landfall_lat = float(life.loc[lf_idx, 'lat'])
        first_landfall_lon = float(life.loc[lf_idx, 'lon'])
        first_landfall_intensity_ratio = first_landfall_wind / lmi if np.isfinite(first_landfall_wind) and lmi > 0 else np.nan
        hours_lmi_to_landfall = _hours(first_landfall_time, t_lmi_first)
        hours_mir_to_landfall = _hours(first_landfall_time, life.loc[mir_idx, 'time']) if mir_idx >= 0 else np.nan
    else:
        lf_idx = -1
        first_landfall_time = pd.NaT
        first_landfall_stage_code = 0
        first_landfall_wind = np.nan
        first_landfall_lat = np.nan
        first_landfall_lon = np.nan
        first_landfall_intensity_ratio = np.nan
        hours_lmi_to_landfall = np.nan
        hours_mir_to_landfall = np.nan

    def _value_at(i, col):
        if i < 0:
            return np.nan
        v = life.loc[i, col]
        return float(v) if pd.notna(v) else np.nan

    def _str_at(i, col):
        if i < 0:
            return ''
        v = life.loc[i, col]
        return '' if pd.isna(v) else str(v)
    summary = {'sid': str(sid), 'name': '' if pd.isna(name) else str(name), 'season': float(season) if pd.notna(season) else np.nan, 'basin_genesis': _str_at(0, 'basin'), 'basin_lmi': _str_at(lmi_first_idx, 'basin'), 'valid_lifecycle': 1, 'qc_reason': 'OK', 'n_lifecycle_obs': int(len(life)), 'wind_valid_fraction': wind_valid_fraction, 'n_wind_filled': int(life['wind_filled_flag'].sum()), 'max_record_gap_hours': max_record_gap, 'max_wind_gap_hours': max_wind_gap, 'genesis_time': t_genesis, 'escalation_time': t_escalation, 'maturity_time': t_maturity, 'lmi_time': t_lmi_first, 'lmi_last_time': t_lmi_last, 'resolution_time': t_resolution, 'termination_time': t_termination, 'mir_time': life.loc[mir_idx, 'time'] if mir_idx >= 0 else pd.NaT, 'mwr_time': life.loc[mwr_idx, 'time'] if mwr_idx >= 0 else pd.NaT, 'genesis_lat': _value_at(0, 'lat'), 'genesis_lon': _value_at(0, 'lon'), 'escalation_lat': _value_at(escalation_idx, 'lat'), 'escalation_lon': _value_at(escalation_idx, 'lon'), 'maturity_lat': _value_at(maturity_start_idx, 'lat'), 'maturity_lon': _value_at(maturity_start_idx, 'lon'), 'lmi_lat': _value_at(lmi_first_idx, 'lat'), 'lmi_lon': _value_at(lmi_first_idx, 'lon'), 'resolution_lat': _value_at(resolution_idx, 'lat'), 'resolution_lon': _value_at(resolution_idx, 'lon'), 'termination_lat': _value_at(len(life) - 1, 'lat'), 'termination_lon': _value_at(len(life) - 1, 'lon'), 'mir_lat': _value_at(mir_idx, 'lat'), 'mir_lon': _value_at(mir_idx, 'lon'), 'mwr_lat': _value_at(mwr_idx, 'lat'), 'mwr_lon': _value_at(mwr_idx, 'lon'), 'genesis_wind': genesis_wind_actual, 'lmi': lmi, 'peak_gain': peak_gain, 'mature_threshold_wind': mature_threshold_wind, 'termination_wind': _value_at(len(life) - 1, 'wind'), 'mir': mir, 'mwr': mwr, 'lifetime_hours': lifetime_h, 'development_duration_hours': development_duration_h, 'escalation_duration_hours': escalation_duration_h, 'maturity_duration_hours': maturity_duration_h, 'resolution_duration_hours': resolution_duration_h, 'pre_lmi_duration_hours': pre_lmi_duration_h, 'post_lmi_duration_hours': post_lmi_duration_h, 'lmi_plateau_duration_hours': _hours(t_lmi_last, t_lmi_first), 'intensification_duration_hours': intensification_duration_h, 'mean_intensification_rate24': mean_intensification_rate24, 'weakening_duration_hours': weakening_duration_h, 'mean_weakening_rate24': mean_weakening_rate24, 'development_rate24': development_rate24, 'escalation_rate24': escalation_rate24, 'resolution_rate24': resolution_rate24, 'lifecycle_asymmetry_time': lifecycle_asymmetry_time, 'lifecycle_asymmetry_rate': lifecycle_asymmetry_rate, 'ri_event_count': int(ri_event_count), 'rw_event_count': int(rw_event_count), 'reintensification_event_count': int(reintensification_count), 'near_lmi_episode_count': int(near_lmi_episode_count), 'ace': ace, 'pdi_m3_s2': pdi, 'track_distance_km': float(track_distance_km), 'mean_translation_speed_kmh': mean_translation_speed_kmh, 'escalation_rate_criterion_found': int(escalation_rate_found), 'resolution_rate_criterion_confirmed': int(resolution_rate_confirmed), 'resolution_used_final_exit_rule': int(resolution_fallback), 'distance_to_land_genesis_km': _value_at(0, 'dist2land'), 'distance_to_land_mir_km': _value_at(mir_idx, 'dist2land'), 'distance_to_land_lmi_km': _value_at(lmi_first_idx, 'dist2land'), 'landfall_event_count': int(landfall_event_count), 'first_landfall_time': first_landfall_time, 'first_landfall_stage_code': int(first_landfall_stage_code), 'first_landfall_wind': first_landfall_wind, 'first_landfall_lat': first_landfall_lat, 'first_landfall_lon': first_landfall_lon, 'first_landfall_intensity_ratio': first_landfall_intensity_ratio, 'hours_lmi_to_landfall': hours_lmi_to_landfall, 'hours_mir_to_landfall': hours_mir_to_landfall}
    life_out = life[['time', 'lat', 'lon', 'lon_360', 'wind', 'pressure', 'basin', 'dist2land', 'landfall', 'landfall_flag', 'wind_filled_flag', 'delta_v24_forward', 'delta_v24_backward', 'stage_code']].copy()
    return {'summary': summary, 'track': life_out}
