"""Limited MWT extension: frozen-support summaries, without field recalculation.

All input functions are explicit and return objects. Original sources are read-only.
No original driver is imported. Complete legal intervals, equal nested weights,
seed and centered-null algorithm reproduce the accepted MWT estimators.
"""
import os
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '1'
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd

from configuration import OUTPUT_ROOT as ROOT, TEMPLATE_ROOT as W, initialize_output

A = W / 'addendum_20261023'
SEED = 650117
B = 2000
GAMMA = ['Gamma_K_h', 'Gamma_K_p', 'Gamma_K_total',
         'Gamma_A_h', 'Gamma_A_p', 'Gamma_A_total']
AUX = ['APE_ref_correction', 'APE_compressibility']
INPUTS = ['vertical_input_K_Wkg', 'vertical_input_A_Wkg', 'vertical_input_P_Wkg']
IDENTITY = ['variant', 'window', 'region', 'level', 'metric']


def json_clean(value):
    if isinstance(value, dict):
        return {str(k): json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_clean(v) for v in value]
    if isinstance(value, np.generic):
        return json_clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(json_clean(value), ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def utc_seconds(series):
    return pd.to_datetime(series, utc=True).astype('int64').to_numpy() // 10**9


def load_frozen_templates(original_root=W):
    original_root = Path(original_root)
    add = original_root / 'addendum_20261023'
    events = pd.read_parquet(add / 'source_data/new_return_events.parquet')
    events = events[events.ledger_eligible].sort_values(['sid', 'event_id']).copy()
    assert len(events) == 133 and events.sid.nunique() == 123
    intervals = pd.read_parquet(add / 'source_data/new_intervals.parquet')
    intervals = intervals[intervals.valid_6h & intervals.valid_h.eq(6)].copy()
    intervals['left_s'] = utc_seconds(intervals.UTC)
    intervals['right_s'] = utc_seconds(intervals.interval_end)
    events['valley_s'] = utc_seconds(events.valley_UTC)
    events['return_s'] = utc_seconds(events.return_UTC)
    rise = []
    for i, e in enumerate(events.itertuples(index=False)):
        it = intervals[intervals.sid.eq(e.sid) & intervals.left_s.ge(e.valley_s)
                       & intervals.right_s.le(e.return_s)]
        assert float(it.valid_h.sum()) * 3600 == e.return_s - e.valley_s
        for r in it.itertuples(index=False):
            assert r.right_s - r.left_s == 21600
            rise.append({'event_row': i, 'sid': e.sid, 'event_id': e.event_id,
                         'left_s': r.left_s, 'right_s': r.right_s,
                         'seconds': float(r.valid_h) * 3600})
    rise = pd.DataFrame(rise)
    pre_events = pd.read_csv(original_root / 'source_data/events_frozen.csv', dtype={'sid': str})
    pre_events['t0_s'] = utc_seconds(pre_events.t0)
    assert pre_events.sid.nunique() == 117 and len(pre_events) == 234
    pre = []
    for e in pre_events.sort_values(['sid', 'event_class']).itertuples(index=False):
        for lag in [-24, -18, -12, -6]:
            pre.append({'sid': e.sid, 'event_id': e.event_id, 'event_class': e.event_class,
                        'lag_h': lag, 'UTC_s': e.t0_s + lag * 3600})
    pre = pd.DataFrame(pre)
    pairs = pd.read_parquet(add / 'source_data/rise_comparable_state_pairs.parquet')
    pairs = pairs[pairs.matched].sort_values(['sid', 'event_id', 'pair_id']).copy()
    roles = pd.read_parquet(add / 'source_data/rise_point_roles.parquet')
    request = pd.read_csv(add / 'source_data/rise_unique_point_requests.csv', dtype={'sid': str, 'request_id': str})
    request['UTC_s'] = utc_seconds(request.UTC)
    roles = roles[roles.role.isin(['matched_rise', 'matched_Resolution'])]
    pivot = roles.pivot(index='pair_id', columns='role', values='request_id').reindex(pairs.pair_id)
    lookup = request.set_index('request_id')
    for side, role in [('rise', 'matched_rise'), ('decay', 'matched_Resolution')]:
        rid = pivot[role].to_numpy()
        pairs[side + '_request_id'] = rid
        pairs[side + '_s'] = lookup.UTC_s.reindex(rid).to_numpy()
        assert np.array_equal(lookup.sid.reindex(rid).to_numpy(), pairs.sid.to_numpy())
    assert len(pairs) == 165 and pairs.sid.nunique() == 108 and pairs.event_id.nunique() == 115
    return {'events': events.reset_index(drop=True), 'rise_intervals': rise,
            'pre': pre, 'pairs': pairs, 'requests': request}


def enforce_gamma_support(frame):
    """Use identical observation support for each horizontal/vertical/total triplet."""
    keys = ['sid', 'UTC_s', 'variant', 'window', 'region', 'level']
    result = frame.copy()
    changed = []
    for energy in ['K', 'A']:
        metrics = ['Gamma_' + energy + '_' + part for part in ['h', 'p', 'total']]
        for variant, part in result[result.metric.isin(metrics) & result.variant.ne('old_support')].groupby('variant', sort=True):
            # Alternative-stencil/conservative-bracket variants deliberately
            # contain only pressure and total terms, not another stored copy of
            # horizontal transfer. Their existing paired support stays intact.
            if not set(metrics).issubset(set(part.metric)):
                changed.append({'energy': energy, 'variant': variant,
                                'finite_values_masked_for_triplet': 0,
                                'status': 'PARTIAL_COMPONENT_SENSITIVITY_PRESERVED'})
                continue
            wide = part.pivot(index=keys, columns='metric', values='value')
            eligible = wide[metrics].notna().all(axis=1)
            idx = pd.MultiIndex.from_frame(result[keys])
            candidate = result.metric.isin(metrics) & result.variant.eq(variant)
            good = eligible.reindex(idx).fillna(False).to_numpy()
            mask = candidate & ~good
            changed.append({'energy': energy, 'variant': variant,
                            'finite_values_masked_for_triplet': int((mask & result.value.notna()).sum()),
                            'status': 'COMPLETE_TRIPLET_COMMON_SUPPORT'})
            result.loc[mask, 'value'] = np.nan
    return result, changed


def add_old_same_temporal_support(frame):
    keys = ['sid', 'UTC_s', 'window', 'region', 'level', 'metric']
    old = frame[frame.variant.eq('old_support')].copy()
    primary = frame[frame.variant.eq('primary')]
    if old.empty:
        return frame
    keep = primary.set_index(keys).value.notna()
    ok = keep.reindex(pd.MultiIndex.from_frame(old[keys])).fillna(False).to_numpy()
    old.loc[~ok, 'value'] = np.nan
    old['variant'] = 'old_same_temporal_support'
    return pd.concat([frame, old], ignore_index=True)


def series_values(series, sid, seconds):
    index = pd.MultiIndex.from_arrays([np.asarray(sid, str), np.asarray(seconds, np.int64)], names=['sid', 'UTC_s'])
    return series.reindex(index).to_numpy(float)


def pre_event_values(series, template):
    rows = template.copy()
    rows['value'] = series_values(series, rows.sid, rows.UTC_s)
    wide = rows.pivot(index=['sid', 'lag_h'], columns='event_class', values='value')
    joint = wide[['internal', 'final']].notna().all(axis=1)
    rows_out = []
    for sid, part in wide.groupby(level='sid', sort=True):
        use = part.loc[joint.reindex(part.index)]
        n = len(use)
        ok = n >= 3
        temporary = float(use.internal.mean()) if ok else np.nan
        decay = float(use.final.mean()) if ok else np.nan
        rows_out.append({'sid': sid, 'event_id': sid + '|paired_departures',
                         'value': temporary - decay, 'temporary_mean': temporary,
                         'decay_mean': decay, 'n_common_offsets': n,
                         'valid': ok, 'failure_reason': '' if ok else 'FEWER_THAN_THREE_COMMON_PRE_OFFSETS'})
    return pd.DataFrame(rows_out)


def rise_values(series, templates, integrate_rate=True):
    events = templates['events']
    intervals = templates['rise_intervals']
    left = series_values(series, intervals.sid, intervals.left_s)
    right = series_values(series, intervals.sid, intervals.right_s)
    good = np.isfinite(left) & np.isfinite(right)
    event_row = intervals.event_row.to_numpy(int)
    n_events = len(events)
    n_interval = np.bincount(event_row, minlength=n_events)
    n_valid = np.bincount(event_row, weights=good, minlength=n_events)
    duration = np.bincount(event_row, weights=intervals.seconds, minlength=n_events)
    integral = np.bincount(event_row, weights=np.where(good, .5 * (left + right) * intervals.seconds, 0), minlength=n_events)
    complete = (n_interval > 0) & (n_valid == n_interval)
    integral[~complete] = np.nan
    endpoint = series_values(series, events.sid, events.return_s) - series_values(series, events.sid, events.valley_s)
    base = events[['sid', 'event_id', 'return_order']].copy()
    base['duration_h'] = duration / 3600
    base['n_intervals'] = n_interval
    base['n_valid_intervals'] = n_valid
    outputs = {}
    for name, value in [('rise_mean', integral / duration), ('rise_endpoint_change', endpoint)]:
        df = base.copy()
        df['value'] = value
        df['valid'] = np.isfinite(value)
        df['failure_reason'] = np.where(df.valid, '', 'MISSING_ENDPOINT' if name == 'rise_endpoint_change' else 'INCOMPLETE_LEGAL_INTERVAL_ENDPOINTS')
        outputs[name] = df
    if integrate_rate:
        df = base.copy()
        df['value'] = integral
        df['valid'] = complete
        df['failure_reason'] = np.where(complete, '', 'INCOMPLETE_LEGAL_INTERVAL_ENDPOINTS')
        outputs['rise_rate_integral'] = df
    return outputs


def matched_values(series, pairs):
    rows = pairs[['sid', 'event_id', 'pair_id', 'wind_bin']].copy()
    rows['rise_value'] = series_values(series, pairs.sid, pairs.rise_s)
    rows['decay_value'] = series_values(series, pairs.sid, pairs.decay_s)
    rows['value'] = rows.rise_value - rows.decay_value
    rows['valid'] = np.isfinite(rows.value)
    rows['failure_reason'] = np.where(rows.valid, '', 'NONFINITE_MATCHED_MEMBER')
    valid = rows[rows.valid]
    event = valid.groupby(['sid', 'event_id'], as_index=False)[['value', 'rise_value', 'decay_value']].mean()
    counts = valid.groupby(['sid', 'event_id']).size()
    event['n_joint_bins'] = counts.reindex(pd.MultiIndex.from_frame(event[['sid', 'event_id']])).to_numpy()
    event['valid'] = True
    return event, rows


def summarize_units(units, replicates=B, seed=SEED):
    good = units[np.isfinite(units.value)]
    sid_values = good.groupby('sid').value.mean().sort_index()
    a = sid_values.to_numpy(float)
    n = len(a)
    estimate = float(a.mean()) if n else np.nan
    if n < 2:
        return {'estimate': estimate, 'ci_lower': np.nan, 'ci_upper': np.nan, 'p': np.nan,
                'n_sid': n, 'n_events': len(good), 'n_attempted': replicates,
                'n_estimable': 0, 'n_failed': replicates, 'status': 'INSUFFICIENT_SID_SUPPORT'}, np.full(replicates, np.nan), sid_values
    index = np.random.default_rng(seed).integers(0, n, size=(replicates, n))
    draws = a[index].mean(axis=1)
    lo, hi = np.quantile(draws, [.025, .975])
    p = (1 + np.count_nonzero(np.abs(draws - estimate) >= abs(estimate))) / (replicates + 1)
    return {'estimate': estimate, 'ci_lower': float(lo), 'ci_upper': float(hi), 'p': float(p),
            'n_sid': n, 'n_events': len(good), 'n_attempted': replicates,
            'n_estimable': replicates, 'n_failed': 0, 'status': 'ESTIMATED'}, draws, sid_values


def family_definition(row):
    if row['variant'] != 'primary' or row['region'] != 'r0_300':
        return 'descriptive', None
    comp, metric, level = row['comparison'], row['metric'], row['level']
    if comp == 'rise_endpoint_change':
        return ('rise_endpoint_states_12', 12) if metric in ['KE', 'APE'] else ('descriptive', None)
    if metric in GAMMA:
        return comp + '_canonical_36', 36
    if metric in AUX:
        return comp + '_reference_terms_12', 12
    if comp == 'rise_rate_integral':
        if level == '500-850' and (metric == 'C_A_to_K' or metric in INPUTS):
            return 'rise_integral_layer_conversion_inputs_12', 12
        return 'descriptive', None
    if level == '500-850' and metric in ['KE', 'APE', 'C_A_to_K']:
        return comp + '_layer_state_conversion_9', 9
    return 'descriptive', None


def apply_bh(summary):
    summary = summary.copy()
    families = [family_definition(r) for r in summary.to_dict('records')]
    summary['family_id'] = [v[0] for v in families]
    summary['family_size'] = [v[1] for v in families]
    summary['q'] = np.nan
    checks = []
    for family, rows in summary[summary.family_id.ne('descriptive')].groupby('family_id', sort=True):
        m = int(rows.family_size.iloc[0])
        if len(rows) != m:
            raise AssertionError(f'Prespecified family incomplete: {family}: {len(rows)} != {m}')
        good = rows[np.isfinite(rows.p)].sort_values('p')
        q = np.minimum.accumulate((good.p.to_numpy() * m / np.arange(1, len(good) + 1))[::-1])[::-1]
        summary.loc[good.index, 'q'] = np.clip(q, 0, 1)
        checks.append({'family_id': family, 'predefined_size': m, 'included_rows': len(rows),
                       'estimable_p': len(good), 'missing_p_retained_in_denominator': m - len(good)})
    return summary, checks


def verify_additivity(summary, draws, sid_support):
    records = []
    key = ['comparison', 'variant', 'window', 'region', 'level']
    for identity, rows in summary.groupby(key, sort=True):
        for energy in ['K', 'A']:
            names = ['Gamma_' + energy + '_' + p for p in ['h', 'p', 'total']]
            pick = rows[rows.metric.isin(names)].set_index('metric')
            if len(pick) != 3:
                continue
            ks = [pick.loc[n, 'draw_key'] for n in names]
            support_equal = sid_support[ks[0]] == sid_support[ks[1]] == sid_support[ks[2]]
            a, b, total = [draws[k] for k in ks]
            finite = np.isfinite(a) & np.isfinite(b) & np.isfinite(total)
            err = float(np.max(np.abs(total[finite] - a[finite] - b[finite]))) if finite.any() else np.nan
            mag = float(np.max(np.abs(total[finite]))) if finite.any() else np.nan
            point_error = float(pick.loc[names[2], 'estimate'] - pick.loc[names[0], 'estimate'] - pick.loc[names[1], 'estimate'])
            passed = support_equal and (not finite.any() or err <= 2e-10 * max(1., mag))
            records.append({**dict(zip(key, identity)), 'energy': energy, 'same_sid_order': support_equal,
                            'n_joint_finite_draws': int(finite.sum()), 'max_abs_draw_error': err,
                            'point_error': point_error, 'passed': bool(passed)})
    return records


def compare_old_outputs(summary, original_root=W):
    original_root = Path(original_root)
    ptr = json.loads((original_root / 'statistics/latest_completed.json').read_text())
    pre_path = Path(ptr['path'])
    pre_path = pre_path if pre_path.is_absolute() else original_root / 'statistics' / pre_path
    pre = pd.read_parquet(pre_path / 'effects.parquet')
    rise = pd.read_parquet(original_root / 'addendum_20261023/mwt_rise_statistics/run_14934f2ae8f2eb7c39af/summary.parquet')
    mapping = {'rise_mean': 'rise_mean', 'rise_endpoint_change': 'rise_endpoint_change',
               'rise_rate_integral': 'rise_rate_integral', 'wind_matched_raw': 'matched_rise_minus_Resolution'}
    checks = []
    for row in summary[summary.variant.eq('old_support')].itertuples(index=False):
        if row.comparison == 'pre_event':
            old = pre[pre.summary.eq('pre') & pre.scheme.eq('main') & pre.window.eq(row.window)
                      & pre.region.eq(row.region) & pre.level.astype(str).eq(row.level) & pre.metric.eq(row.metric)]
        else:
            old = rise[rise.estimand.eq(mapping[row.comparison]) & rise['sample'].eq('all_events')
                       & rise.scheme.eq('main') & rise.window.eq(row.window) & rise.region.eq(row.region)
                       & rise.level.astype(str).eq(row.level) & rise.metric.eq(row.metric)
                       & rise.support_mode.eq('native_diagnostic')]
        if len(old) != 1:
            checks.append({'draw_key': row.draw_key, 'status': 'NO_UNIQUE_FROZEN_LOOKUP', 'matches': len(old)})
            continue
        ref = old.iloc[0]
        new = np.array([row.estimate, row.ci_lower, row.ci_upper], float)
        previous = np.array([ref.estimate, ref.lower, ref.upper], float)
        finite = np.isfinite(new) & np.isfinite(previous)
        err = float(np.max(np.abs(new[finite] - previous[finite]))) if finite.any() else np.nan
        agree = np.allclose(new, previous, rtol=2e-9, atol=2e-10, equal_nan=True)
        checks.append({'draw_key': row.draw_key, 'comparison': row.comparison, 'metric': row.metric,
                       'level': row.level, 'region': row.region, 'window': row.window,
                       'max_abs_estimate_ci_error': err, 'same_n_sid': int(ref.n_sid) == int(row.n_sid),
                       'status': 'PASS' if agree and int(ref.n_sid) == int(row.n_sid) else 'DIFFERENCE'})
    return checks


def run_statistics(root=ROOT, original_root=W):
    root = Path(root).resolve()
    initialize_output(root)
    output = root / 'statistics'
    output.mkdir(exist_ok=True)
    templates = load_frozen_templates(original_root)
    expected_sid = set(templates['events'].sid) | set(templates['pre'].sid)
    paths = [root / 'diagnostics' / sid / 'request_values.parquet' for sid in sorted(expected_sid)]
    absent = [str(p) for p in paths if not p.is_file()]
    if absent:
        raise ValueError('Missing frozen-SID extraction files: ' + str(absent))
    extracted, input_receipts = [], []
    for path in paths:
        path.resolve().relative_to(root)
        receipt_path = path.parent / 'COMPLETE.json'
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if receipt.get('sid') != path.parent.name or receipt.get('status') != 'COMPLETE' or receipt.get('output_sha256') != digest:
            raise AssertionError('Incomplete or changed diagnostic extraction: ' + str(path))
        if receipt.get('derivation_version') != 2 or receipt.get('mask_schema_version') != 3:
            raise AssertionError('Final derivation=2 / mask_schema=3 required before statistics: ' + str(path))
        block = pd.read_parquet(path)
        assert set(block.sid.astype(str)) == {path.parent.name}
        assert len(block) == receipt['rows']
        assert block.UTC.nunique() == receipt['n_requests']
        for column in ['valid_mask_sha256', 'geometric_mask_sha256']:
            if column not in block or not block[column].astype('string').str.fullmatch(r'[0-9a-f]{64}').fillna(False).all():
                raise AssertionError('Invalid final request mask fingerprint: ' + str(path) + ' / ' + column)
        extracted.append(block)
        input_receipts.append({'sid': path.parent.name, 'path': str(path), 'sha256': digest,
                               'receipt': str(receipt_path),
                               'receipt_sha256': hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
                               'derivation_version': 2, 'mask_schema_version': 3})
    frame = pd.concat(extracted, ignore_index=True)
    save_json(output / 'raw_statistics_input_receipts.json', input_receipts)
    if set(frame.sid.astype(str)) != expected_sid:
        raise AssertionError('Incomplete or extraneous SID extraction: missing=' +
                             str(sorted(expected_sid - set(frame.sid.astype(str)))) +
                             '; extra=' + str(sorted(set(frame.sid.astype(str)) - expected_sid)))
    frame['level'] = frame.level.astype(str)
    frame['UTC_s'] = utc_seconds(frame.UTC)
    frame['sid'] = frame.sid.astype(str)
    if frame.duplicated(['sid', 'UTC_s'] + IDENTITY).any():
        raise AssertionError('Duplicate scientific observations')
    frame.loc[~np.isfinite(frame.value), 'value'] = np.nan
    frame, mask_checks = enforce_gamma_support(frame)
    frame = add_old_same_temporal_support(frame)
    summaries, unit_records, matched_records = [], [], []
    draws, sid_support = {}, {}
    identities = list(frame.groupby(IDENTITY, sort=True, dropna=False))
    for count, (identity, block) in enumerate(identities):
        ident = dict(zip(IDENTITY, identity))
        metric = ident['metric']
        rate = metric not in ['KE', 'APE']
        unit = 'W kg-1' if rate else 'J kg-1'
        values = block.set_index(['sid', 'UTC_s']).value
        pre = pre_event_values(values, templates['pre'])
        jobs = {'pre_event': pre}
        phase = rise_values(values, templates, integrate_rate=rate)
        # Endpoint changes are requested for reservoirs; rate endpoint changes
        # are not a physical integrated supply and are omitted.
        if rate:
            phase.pop('rise_endpoint_change')
        jobs.update(phase)
        matched, matched_rows = matched_values(values, templates['pairs'])
        jobs['wind_matched_raw'] = matched
        for comparison, units in jobs.items():
            estimate, bs, sid = summarize_units(units)
            output_unit = 'J kg-1' if comparison == 'rise_rate_integral' else unit
            key = hashlib.sha256(json.dumps([*identity, comparison], default=str).encode()).hexdigest()[:24]
            draws[key] = bs
            sid_support[key] = sid.index.to_list()
            row = {**ident, 'comparison': comparison, 'unit': output_unit, **estimate,
                   'draw_key': key, 'seed': SEED, 'bootstrap_unit': 'SID',
                   'n_matched_units': int(matched_rows.valid.sum()) if comparison == 'wind_matched_raw' else None,
                   'n_event_pairs': estimate['n_sid'] if comparison == 'pre_event' else None,
                   'direction': {'pre_event': 'temporary departure minus start of decay',
                                 'wind_matched_raw': 'reintensification minus decay at matched wind',
                                 'rise_mean': 'actual valley-to-return time mean',
                                 'rise_rate_integral': 'actual valley-to-return trapezoid integral in seconds',
                                 'rise_endpoint_change': 'return minus valley'}[comparison],
                   'interval_method': '2000-draw SID percentile 95%',
                   'p_method': 'centered bootstrap two-sided plus-one count',
                   'source_path': str(root / 'diagnostics')}
            if comparison == 'pre_event':
                row['n_events'] = 2 * row['n_sid']
            sides = [('temporary', 'temporary_mean'), ('decay', 'decay_mean')] if comparison == 'pre_event' else (
                    [('rise', 'rise_value'), ('decay', 'decay_value')] if comparison == 'wind_matched_raw' else [])
            for side, column in sides:
                side_units = units.assign(value=units[column])
                side_units.loc[~np.isfinite(units.value), 'value'] = np.nan
                side_summary, side_draws, side_sid = summarize_units(side_units)
                assert side_sid.index.to_list() == sid.index.to_list()
                row[side + '_mean'] = side_summary['estimate']
                row[side + '_ci_lower'] = side_summary['ci_lower']
                row[side + '_ci_upper'] = side_summary['ci_upper']
                draws[key + '_' + side] = side_draws
                sid_support[key + '_' + side] = side_sid.index.to_list()
                row[side + '_draw_key'] = key + '_' + side
            summaries.append(row)
            unit_records.append(units.assign(**ident, comparison=comparison, draw_key=key, unit=output_unit))
        matched_records.append(matched_rows.assign(**ident))
        if (count + 1) % 25 == 0 or count == len(identities) - 1:
            save_json(root / 'checkpoints/raw_statistics.json',
                      {'state': 'SUMMARIZING', 'identities_done': count + 1, 'identities_total': len(identities)})
            print(json.dumps({'identities_done': count + 1, 'identities_total': len(identities)}), flush=True)
    summary, family_checks = apply_bh(pd.DataFrame(summaries))
    algebra = verify_additivity(summary, draws, sid_support)
    old_checks = compare_old_outputs(summary, original_root)
    summary.to_csv(output / 'raw_summary.csv', index=False, encoding='utf-8-sig')
    summary.to_parquet(output / 'raw_summary.parquet', index=False)
    pd.concat(unit_records, ignore_index=True).to_parquet(output / 'unit_support_and_values.parquet', index=False)
    pd.concat(matched_records, ignore_index=True).to_parquet(output / 'matched_unit_values.parquet', index=False)
    np.savez_compressed(output / 'raw_bootstrap_draws.npz', **draws)
    save_json(output / 'raw_bootstrap_sid_order.json', {'seed': SEED, 'attempts': B, 'sid_order_by_draw_key': sid_support})
    pd.DataFrame(algebra).to_csv(output / 'algebra_consistency.csv', index=False)
    pd.DataFrame(old_checks).to_csv(output / 'old_support_reproduction.csv', index=False)
    save_json(output / 'raw_statistics_QA.json', {'inputs': [str(p) for p in paths], 'input_sids': frame.sid.nunique(),
              'summary_rows': len(summary), 'gamma_triplet_masks': mask_checks,
              'families': family_checks, 'algebra_checks': len(algebra),
              'algebra_failed': sum(not r['passed'] for r in algebra),
              'old_comparisons': len(old_checks),
              'old_differences': sum(r['status'] != 'PASS' for r in old_checks),
              'notes': ['old_same_temporal_support matches primary finite observation support; spatial masking differences remain explicit',
                        'Complete fixed families retain missing targets in the family-size multiplier',
                        'All draws use equal SID weights; actual events and shared bins are averaged within SID as specified']})
    save_json(root / 'checkpoints/raw_statistics.json', {'state': 'COMPLETE_PENDING_INDEPENDENT_REVIEW', 'rows': len(summary)})
    if any(not r['passed'] for r in algebra):
        raise AssertionError('Gamma total is not additive on identical SID/draw support')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=str(ROOT))
    parser.add_argument('--original-root', default=str(W))
    args = parser.parse_args()
    run_statistics(Path(args.root), Path(args.original_root))
