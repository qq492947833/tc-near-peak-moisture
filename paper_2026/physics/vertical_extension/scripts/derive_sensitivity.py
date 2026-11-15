"""Paired numerical/layer sensitivities from frozen task-local diagnostics.

No source experiment is rerun. Common observations are reconstructed from exact
request values before SID/event aggregation; marginal CIs are never subtracted.
Only explicitly paired numerical variants are compared, with descriptive q=NA.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from raw_statistics import (ROOT, W, B, SEED, load_frozen_templates, utc_seconds,
                            enforce_gamma_support, pre_event_values, rise_values,
                            matched_values, summarize_units, save_json)


def sensitivity_definitions():
    return [
        {'sensitivity_id': 'sparse_minus_four_level', 'alternative_variant': 'sparse_layer',
         'reference_variant': 'primary', 'alternative_level': '500-850', 'reference_level': '500-850'},
        {'sensitivity_id': 'alternate_pressure_stencil', 'alternative_variant': 'alternate_derivative',
         'reference_variant': 'same_support_primary_alternate', 'alternative_level': None, 'reference_level': None},
        {'sensitivity_id': 'alternate_stencil_and_cprime', 'alternative_variant': 'combined_derivative_cprime',
         'reference_variant': 'same_support_primary_alternate', 'alternative_level': None, 'reference_level': None},
        {'sensitivity_id': 'bracket_minus_conservative', 'alternative_variant': 'bracket_derivative',
         'reference_variant': 'same_support_primary_bracket', 'alternative_level': None, 'reference_level': None},
        {'sensitivity_id': 'joint_process_layer_minus_700', 'alternative_variant': 'joint_layer_processes',
         'reference_variant': 'joint_layer_processes', 'alternative_level': '500-850', 'reference_level': '700'},
        {'sensitivity_id': 'joint_transfer_layer_minus_700', 'alternative_variant': 'joint_with_transfer',
         'reference_variant': 'joint_with_transfer', 'alternative_level': '500-850', 'reference_level': '700'},
    ]


def load_complete_requests(root, templates):
    root = Path(root).resolve()
    expected = sorted(set(templates['events'].sid) | set(templates['pre'].sid))
    pieces, manifest = [], []
    for sid in expected:
        path = root / 'diagnostics' / sid / 'request_values.parquet'
        path.resolve().relative_to(root)
        receipt_path = path.parent / 'COMPLETE.json'
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if receipt.get('status') != 'COMPLETE' or receipt.get('sid') != sid or receipt.get('output_sha256') != sha:
            raise AssertionError('Changed or incomplete extraction ' + str(path))
        if receipt.get('derivation_version') != 2 or receipt.get('mask_schema_version') != 3:
            raise AssertionError('Nonfinal diagnostic schema (derivation=2, mask_schema=3 required): ' + str(path))
        block = pd.read_parquet(path)
        assert set(block.sid.astype(str)) == {sid} and len(block) == receipt['rows']
        for column in ['valid_mask_sha256', 'geometric_mask_sha256']:
            if column not in block or not block[column].astype('string').str.fullmatch(r'[0-9a-f]{64}').fillna(False).all():
                raise AssertionError('Missing or invalid exact grid-mask fingerprint: ' + str(path) + ' / ' + column)
        pieces.append(block)
        manifest.append({'sid': sid, 'path': str(path), 'sha256': sha,
                         'receipt_sha256': hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
                         'derivation_version': 2, 'mask_schema_version': 3})
    result = pd.concat(pieces, ignore_index=True)
    result['level'] = result.level.astype(str)
    result['sid'] = result.sid.astype(str)
    result['UTC_s'] = utc_seconds(result.UTC)
    result.loc[~np.isfinite(result.value), 'value'] = np.nan
    result, common_gamma_log = enforce_gamma_support(result)
    return result, manifest, common_gamma_log


def request_pairs(alternative, reference):
    keys = ['sid', 'UTC_s']
    hashes = ['valid_mask_sha256', 'geometric_mask_sha256']
    for side in [alternative, reference]:
        if not set(hashes).issubset(side):
            raise AssertionError('Nonfinal request schema: exact grid-mask hashes required')
        for column in hashes:
            if not side[column].astype('string').str.fullmatch(r'[0-9a-f]{64}').fillna(False).all():
                raise AssertionError('Invalid exact grid-mask hash: ' + column)
    columns = keys + ['value', 'valid_area_fraction', 'n_valid_cells', 'n_geometric_cells'] + hashes
    paired = alternative[columns].merge(reference[columns], on=keys, how='outer',
                                        validate='one_to_one', suffixes=('_alternative', '_reference'))
    finite = np.isfinite(paired.value_alternative) & np.isfinite(paired.value_reference)
    fingerprint = (paired.valid_mask_sha256_alternative.eq(paired.valid_mask_sha256_reference)
                   & paired.geometric_mask_sha256_alternative.eq(paired.geometric_mask_sha256_reference))
    aggregates = (paired.n_valid_cells_alternative.eq(paired.n_valid_cells_reference)
                  & paired.n_geometric_cells_alternative.eq(paired.n_geometric_cells_reference)
                  & np.isclose(paired.valid_area_fraction_alternative,
                               paired.valid_area_fraction_reference, rtol=0, atol=2e-12))
    if (finite & fingerprint & ~aggregates).any():
        raise AssertionError('Identical grid masks have inconsistent cell counts or area fractions')
    spatial = fingerprint & aggregates
    paired['joint_finite'] = finite
    paired['exact_mask_fingerprint_equal'] = fingerprint
    paired['same_recorded_spatial_support'] = spatial
    paired['spatial_mismatch_with_finite_values'] = finite & ~spatial
    paired.loc[~finite, ['value_alternative', 'value_reference']] = np.nan
    paired = paired.sort_values(keys).reset_index(drop=True)
    return paired


def reconstruct_units(series, templates, rate):
    jobs = {'pre_event': pre_event_values(series, templates['pre'])}
    phase = rise_values(series, templates, integrate_rate=rate)
    if rate:
        phase.pop('rise_endpoint_change')
    jobs.update(phase)
    matched, bin_rows = matched_values(series, templates['pairs'])
    jobs['wind_matched_raw'] = matched
    return jobs, bin_rows


def pair_units(alternative, reference):
    keys = ['sid', 'event_id']
    use = ['value', 'n_common_offsets', 'n_joint_bins', 'duration_h', 'n_intervals', 'n_valid_intervals']
    a = alternative[keys + [c for c in use if c in alternative]]
    r = reference[keys + [c for c in use if c in reference]]
    pairs = a.merge(r, on=keys, how='outer', validate='one_to_one', suffixes=('_alternative', '_reference'))
    pairs['value'] = pairs.value_alternative - pairs.value_reference
    pairs['valid'] = np.isfinite(pairs.value)
    pairs['failure_reason'] = np.where(pairs.valid, '', 'NO_COMMON_ESTIMABLE_EVENT_TARGET')
    for col in ['n_common_offsets', 'n_joint_bins', 'duration_h', 'n_intervals']:
        ca, cr = col + '_alternative', col + '_reference'
        if ca in pairs and cr in pairs:
            good = pairs.valid & pairs[ca].notna() & pairs[cr].notna()
            if not np.allclose(pairs.loc[good, ca], pairs.loc[good, cr], rtol=0, atol=1e-12):
                raise AssertionError('Reconstructed paired units have different support: ' + col)
    return pairs


def compare_saved_units(paired, saved, identity, comparison, side):
    variant, level = identity[side + '_variant'], identity[side + '_level']
    select = (saved.variant.eq(variant) & saved.level.eq(level) & saved.window.eq(identity['window'])
              & saved.region.eq(identity['region']) & saved.metric.eq(identity['metric'])
              & saved.comparison.eq(comparison))
    old = saved.loc[select, ['sid', 'event_id', 'value']]
    if old.duplicated(['sid', 'event_id']).any():
        raise AssertionError('Saved unit target repeated')
    merged = paired[['sid', 'event_id', 'value_' + side]].merge(old, on=['sid', 'event_id'],
                           how='outer', validate='one_to_one', suffixes=('', '_saved'))
    current = merged['value_' + side].to_numpy(float)
    previous = merged.value.to_numpy(float)
    both = np.isfinite(current) & np.isfinite(previous)
    return {'saved_unit_rows': len(old), 'n_previous_finite': int(np.isfinite(previous).sum()),
            'n_common_reconstructed_finite': int(np.isfinite(current).sum()),
            'n_support_changed': int((np.isfinite(current) != np.isfinite(previous)).sum()),
            'max_value_difference_on_common_units': float(np.max(np.abs(current[both] - previous[both]))) if both.any() else None}


def run(root=ROOT, original_root=W):
    root = Path(root).resolve()
    from configuration import initialize_output
    initialize_output(root)
    templates = load_frozen_templates(original_root)
    frame, input_manifest, gamma_log = load_complete_requests(root, templates)
    saved_path = root / 'statistics/unit_support_and_values.parquet'
    saved = pd.read_parquet(saved_path)
    saved['level'] = saved.level.astype(str)
    if set(saved.sid.astype(str)) - set(templates['events'].sid):
        raise AssertionError('Unexpected SID in unit source')
    summaries, paired_units, request_qa, unit_qa = [], [], [], []
    draw_values, draw_sid = {}, {}
    blocked = []
    group_count = 0
    for definition in sensitivity_definitions():
        alternative = frame[frame.variant.eq(definition['alternative_variant'])]
        reference = frame[frame.variant.eq(definition['reference_variant'])]
        if definition['alternative_level'] is not None:
            alternative = alternative[alternative.level.eq(definition['alternative_level'])]
            reference = reference[reference.level.eq(definition['reference_level'])]
        for (window, region, level, metric), alt in alternative.groupby(['window', 'region', 'level', 'metric'], sort=True):
            ref_level = definition['reference_level'] or level
            ref = reference[reference.window.eq(window) & reference.region.eq(region)
                            & reference.level.eq(ref_level) & reference.metric.eq(metric)]
            if ref.empty:
                # Layer boundary inputs have no corresponding 700-hPa object.
                if definition['reference_level'] == '700' and metric.startswith('vertical_input_'):
                    continue
                raise AssertionError('Missing sensitivity reference: ' + str((definition, window, region, level, metric)))
            identity = {**definition, 'alternative_level': level, 'reference_level': ref_level,
                        'window': int(window), 'region': region, 'metric': metric}
            pair = request_pairs(alt, ref)
            n_spatial = int(pair.spatial_mismatch_with_finite_values.sum())
            qa = {**identity, 'n_request_union': len(pair), 'n_joint_finite_requests': int(pair.joint_finite.sum()),
                  'n_request_finite_support_mismatches': int((alt.value.notna().sum() + ref.value.notna().sum()) - 2 * pair.joint_finite.sum()),
                  'n_spatial_mask_mismatch_requests': n_spatial,
                  'spatial_equality_method': 'exact packed-grid SHA256 plus count and area consistency',
                  'status': 'SPATIAL_REAGGREGATION_REQUIRED' if n_spatial else 'COMMON_REQUEST_SUPPORT'}
            request_qa.append(qa)
            if n_spatial:
                bad = pair[pair.spatial_mismatch_with_finite_values].assign(**identity)
                blocked.append(bad)
                continue
            index = pd.MultiIndex.from_frame(pair[['sid', 'UTC_s']])
            alt_series = pd.Series(pair.value_alternative.to_numpy(), index=index)
            ref_series = pd.Series(pair.value_reference.to_numpy(), index=index)
            rate = metric not in ['KE', 'APE']
            alt_jobs, alt_bins = reconstruct_units(alt_series, templates, rate)
            ref_jobs, ref_bins = reconstruct_units(ref_series, templates, rate)
            assert np.array_equal(alt_bins.valid.to_numpy(), ref_bins.valid.to_numpy())
            for comparison in alt_jobs:
                units = pair_units(alt_jobs[comparison], ref_jobs[comparison])
                info, draws, sids = summarize_units(units)
                key = hashlib.sha256(json.dumps([identity, comparison], sort_keys=True).encode()).hexdigest()[:24]
                draw_values[key] = draws
                draw_sid[key] = sids.index.to_list()
                row = {**identity, 'comparison': comparison, **info, 'draw_key': key,
                       'unit': 'J kg-1' if not rate or comparison == 'rise_rate_integral' else 'W kg-1',
                       'direction': 'alternative minus reference on common observations and paired units',
                       'family_id': 'descriptive_numerical_or_layer_sensitivity', 'q': np.nan,
                       'n_matched_units': int(alt_bins.valid.sum()) if comparison == 'wind_matched_raw' else None,
                       'n_event_pairs': info['n_sid'] if comparison == 'pre_event' else None,
                       'p_method': 'SID centered bootstrap two-sided plus-one', 'seed': SEED,
                       'interval_method': '2000-draw SID percentile 95%',
                       'support_status': 'EXACT_COMMON_REQUEST_SUPPORT', 'source_path': str(saved_path)}
                if comparison == 'pre_event':
                    row['n_events'] = 2 * info['n_sid']
                for side in ['alternative', 'reference']:
                    side_units = units.assign(value=units['value_' + side])
                    side_units.loc[~units.valid, 'value'] = np.nan
                    side_info, side_draws, side_sids = summarize_units(side_units)
                    assert side_sids.index.to_list() == sids.index.to_list()
                    row[side + '_estimate'] = side_info['estimate']
                    row[side + '_ci_lower'] = side_info['ci_lower']
                    row[side + '_ci_upper'] = side_info['ci_upper']
                    draw_values[key + '_' + side] = side_draws
                    draw_sid[key + '_' + side] = side_sids.index.to_list()
                    unit_qa.append({**identity, 'comparison': comparison, 'side': side,
                                    **compare_saved_units(units, saved, identity, comparison, side)})
                finite = np.isfinite(draws)
                draw_error = float(np.max(np.abs(draws[finite] - draw_values[key + '_alternative'][finite]
                                                + draw_values[key + '_reference'][finite]))) if finite.any() else np.nan
                row['paired_draw_identity_max_error'] = draw_error
                if finite.any() and draw_error > 1e-9 * max(1., float(np.max(np.abs(draws[finite])))):
                    raise AssertionError('Paired difference draw identity failed')
                summaries.append(row)
                paired_units.append(units.assign(**identity, comparison=comparison, draw_key=key, unit=row['unit']))
            group_count += 1
            if group_count % 20 == 0:
                save_json(root / 'checkpoints/sensitivity_statistics.json',
                          {'state': 'RUNNING', 'group_count': group_count, 'summary_rows': len(summaries)})
                print(json.dumps({'sensitivity_groups': group_count, 'summary_rows': len(summaries)}), flush=True)
    output = root / 'statistics'
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / 'sensitivity_summary.csv', index=False, encoding='utf-8-sig')
    summary.to_parquet(output / 'sensitivity_summary.parquet', index=False)
    pd.concat(paired_units, ignore_index=True).to_parquet(output / 'sensitivity_paired_unit_values.parquet', index=False)
    pd.DataFrame(request_qa).to_csv(output / 'sensitivity_request_support_QA.csv', index=False)
    pd.DataFrame(unit_qa).to_csv(output / 'sensitivity_saved_unit_readback.csv', index=False)
    np.savez_compressed(output / 'sensitivity_bootstrap_draws.npz', **draw_values)
    save_json(output / 'sensitivity_bootstrap_sid_order.json', {'seed': SEED, 'n_attempted': B, 'sid_order': draw_sid})
    if blocked:
        pd.concat(blocked, ignore_index=True).to_csv(output / 'sensitivity_spatial_support_unresolved.csv', index=False)
    save_json(output / 'sensitivity_QA.json',
              {'status': 'COMPLETE_PENDING_REVIEW' if not blocked else 'PARTIAL_SPATIAL_REAGGREGATION_REQUIRED',
               'summary_rows': len(summary), 'request_groups_checked': len(request_qa),
               'groups_requiring_spatial_reaggregation': len(blocked),
               'input_receipts': input_manifest, 'gamma_support': gamma_log,
               'saved_units_sha256': hashlib.sha256(saved_path.read_bytes()).hexdigest(),
               'families': 'No new formal families. All q values are NA; P values descriptive.',
               'pairing': 'Exact request-level common finite values, then original temporal and nested-weight estimators',
               'spatial_rule': 'Exact valid/geometric grid-mask SHA256 equality, supplemented by valid-cell and area consistency; derivation_version=2 and mask_schema_version=3 required. Regional means cannot repair a true grid-mask mismatch.'})
    save_json(root / 'checkpoints/sensitivity_statistics.json',
              {'state': 'COMPLETE_PENDING_REVIEW' if not blocked else 'PARTIAL_SPATIAL_REAGGREGATION_REQUIRED',
               'summary_rows': len(summary), 'unresolved_spatial_groups': len(blocked)})
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=str(ROOT))
    args = parser.parse_args()
    run(Path(args.root))
