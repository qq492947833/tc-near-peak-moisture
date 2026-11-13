"""Continuous first-LMI episode and legal six-hour R/U/F accounting."""
import numpy as np
import pandas as pd

def contiguous_window(times, winds, anchor_time, lmi, threshold, relative=False):
    times = np.asarray(times, dtype='datetime64[ns]')
    winds = np.asarray(winds, dtype=float)
    anchor = np.flatnonzero(times == np.datetime64(anchor_time, 'ns'))
    if len(anchor) != 1 or not np.isfinite(winds[anchor[0]]) or winds[anchor[0]] != lmi:
        raise ValueError('First-LMI anchor must exist uniquely and have finite wind exactly equal to saved LMI.')
    anchor = int(anchor[0])
    qualifies = np.isfinite(winds) & (winds >= threshold if relative else np.abs(winds - lmi) <= threshold)
    if not qualifies[anchor]:
        raise ValueError('Anchor does not satisfy threshold.')
    ends, reasons, stop_indices = ([], [], [])
    for direction in [-1, 1]:
        j = anchor
        while True:
            k = j + direction
            if k < 0 or k >= len(times):
                reason, stop = ('lifecycle_endpoint', None)
                break
            if abs(times[k] - times[j]) != np.timedelta64(6, 'h'):
                reason, stop = ('non_6h_gap', k)
                break
            if not np.isfinite(winds[k]):
                reason, stop = ('missing_wind', k)
                break
            if not qualifies[k]:
                reason, stop = ('threshold_failure', k)
                break
            j = k
        ends.append(j)
        reasons.append(reason)
        stop_indices.append(stop)
    selected = np.zeros(len(times), dtype=bool)
    selected[ends[0]:ends[1] + 1] = True
    labels = np.where(np.arange(len(times)) < ends[0], 0, np.where(np.arange(len(times)) <= ends[1], 1, 2))
    return {'selected': selected, 'labels': labels, 'first_i': ends[0], 'last_i': ends[1], 'left_stop_reason': reasons[0], 'right_stop_reason': reasons[1], 'left_stop_i': stop_indices[0], 'right_stop_i': stop_indices[1], 'qualifies': qualifies}

def valid_intervals(times, winds):
    times = np.asarray(times, dtype='datetime64[ns]')
    winds = np.asarray(winds, float)
    return (np.diff(times) == np.timedelta64(6, 'h')) & np.isfinite(winds[:-1]) & np.isfinite(winds[1:])

def window_metrics(times, winds, window):
    times = np.asarray(times, dtype='datetime64[ns]')
    selected = window['selected']
    indices = np.flatnonzero(selected)
    first, last = (int(indices[0]), int(indices[-1]))
    valid = valid_intervals(times, winds)
    return {'first_selected_time': pd.Timestamp(times[first]), 'last_selected_time': pd.Timestamp(times[last]), 'first_after_time': pd.Timestamp(times[last + 1]) if last + 1 < len(times) else pd.NaT, 'record_span_h': float((times[last] - times[first]) / np.timedelta64(1, 'h')), 'covered_duration_h': float(6 * np.sum(selected[:-1] & valid)), 'selected_record_n': len(indices), 'selected_finite_record_n': int(np.sum(selected & np.isfinite(winds))), 'left_stop_reason': window['left_stop_reason'], 'right_stop_reason': window['right_stop_reason'], 'left_endpoint_truncated': window['left_stop_reason'] == 'lifecycle_endpoint', 'right_endpoint_truncated': window['right_stop_reason'] == 'lifecycle_endpoint', 'selected_has_missing_wind': bool((~np.isfinite(np.asarray(winds)[indices])).any()), 'selected_has_non6h_gap': bool((np.diff(times[indices]) != np.timedelta64(6, 'h')).any()), 'search_stopped_missing_wind': 'missing_wind' in [window['left_stop_reason'], window['right_stop_reason']], 'search_stopped_gap': 'non_6h_gap' in [window['left_stop_reason'], window['right_stop_reason']]}

def true_runs(mask):
    idx = np.flatnonzero(mask)
    return np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1) if len(idx) else []

def decompose_track(track, row):
    g = track.copy().reset_index(drop=True)
    t = g.UTC
    w = g.wind_kt.to_numpy(float)
    finite = np.isfinite(w)
    gap = (t.shift(-1) - t).dt.total_seconds() / 3600
    valid = gap.eq(6).to_numpy() & finite & np.r_[finite[1:], False]
    valid &= t.shift(-1).le(pd.Timestamp(row['termination_time'])).to_numpy()
    r = g.R90C_selected.to_numpy(bool)
    u = finite & (w >= float(row['mature_threshold_wind'])) & t.ge(pd.Timestamp(row['R90C_first_selected_time'])).to_numpy() & t.le(pd.Timestamp(row['termination_time'])).to_numpy()
    f = g.F_stage.eq(3).to_numpy()
    g['R_selected'] = r
    g['U_selected'] = u
    g['F_selected'] = f
    g['valid_6h_interval'] = valid
    g['interval_end'] = t.shift(-1)
    g['interval_gap_h'] = gap
    for name, mask in [('R', r), ('U', u), ('F', f)]:
        g[f'{name}_covered_h'] = np.where(mask & valid, 6.0, 0.0)
    g['maturity_component'] = np.select([f & r, f & u & ~r, f & finite & ~u, f & ~finite], ['First continuous episode', 'Later near-LMI episode', 'Below-threshold bridge', 'Unknown wind'], default='Outside maturity')
    tr, tu, tf = [float(g[f'{k}_covered_h'].sum()) for k in ['R', 'U', 'F']]
    m = dict(T_R_h=tr, T_U_h=tu, T_F_h=tf, T_later_h=tu - tr, T_bridge_h=tf - tu, T_added_h=tf - tr, U_fraction_of_F=tu / tf if tf > 0 else np.nan, R_record_n=int(r.sum()), U_record_n=int(u.sum()), F_record_n=int(f.sum()), R_not_U_record_n=int((r & ~u).sum()), U_not_F_record_n=int((u & ~f).sum()), R_not_U_interval_n=int((r & ~u & valid).sum()), U_not_F_interval_n=int((u & ~f & valid).sum()), F_unknown_wind_record_n=int((f & ~finite).sum()), F_uncovered_record_interval_n=int((f & ~valid).sum()))
    return (g, m)

def reintensification_episodes(track, lmi_time, threshold):
    g = track.reset_index(drop=True)
    dv = pd.to_numeric(g.delta_v24_forward, errors='coerce').to_numpy(float)
    qualifies = g.UTC.ge(lmi_time).to_numpy() & np.isfinite(dv) & (dv >= threshold)
    out = []
    for no, idx in enumerate(true_runs(qualifies), 1):
        first, last = (g.UTC.iloc[idx[0]], g.UTC.iloc[idx[-1]])
        end = last + pd.Timedelta(hours=24)
        window = g[g.UTC.between(first, end)]
        stages = sorted(window.F_stage.astype(int).unique().tolist())
        out.append(dict(sid=g.sid.iloc[0], event_id=no, first_window_start=first, last_window_start=last, last_window_end=end, max_24h_gain_kt=float(dv[idx].max()), qualifying_window_n=len(idx), start_F_stage=int(g.F_stage.iloc[idx[0]]), involved_F_stages=';'.join(map(str, stages)), cross_stage=len(stages) > 1, window_end_present=bool(g.UTC.eq(end).any()), qualifying_record_indices=';'.join(map(str, g.record_index.iloc[idx].tolist()))))
    return out
