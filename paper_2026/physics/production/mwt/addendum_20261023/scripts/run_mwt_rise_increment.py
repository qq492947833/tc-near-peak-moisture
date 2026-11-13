"""One-SID sidecar orchestration; reuse frozen core and cached coefficient cells.

No group statistics, old-cache writes, implicit worker launch or original-field
downloads. Production scheduling is external to this entry point.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '4'
from pathlib import Path
import argparse
import datetime
import gc
import hashlib
import json
import sys
import time
import traceback
import h5py
import numpy as np
import pandas as pd
import psutil

WORK = Path(__file__).absolute().parents[2]
sys.path.insert(0, str(WORK / 'scripts'))
from read_era5_fixed_block import read_era5_fixed_block
from theory_MWT_Prepare_Direct import MWT_Prepare_Direct
from theory_MWT_Atmosphere_Direct import MWT_Atmosphere_Direct
from run_mwt_sid import sid_fixed_domain, summarize_neighborhoods
from MWT_Apply_Increment import MWT_Apply_Increment


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save_json(path, value):
    temp = Path(path).with_suffix('.tmp.json')
    temp.write_text(json.dumps(value, indent=2, allow_nan=True), encoding='utf8')
    deadline = time.monotonic() + 60
    while True:
        try:
            os.replace(temp, path)
            return
        except OSError as exc:
            if getattr(exc, 'winerror', None) not in (5, 32, 33) or time.monotonic() >= deadline:
                raise
            time.sleep(.2)


def remove_own_lock(path):
    deadline = time.monotonic() + 60
    while True:
        try:
            path.unlink()
            return
        except FileNotFoundError:
            return
        except OSError as exc:
            if getattr(exc, 'winerror', None) not in (5, 32, 33) or time.monotonic() >= deadline:
                raise
            time.sleep(.2)


def equivalent_saved_identity(saved, current, add):
    if saved == current:
        return True
    adoption_path = add / 'validation/increment_state_retry_adoption.json'
    if not adoption_path.exists():
        return False
    adoption = read_json(adoption_path)
    before = json.loads(json.dumps(saved)); after = json.loads(json.dumps(current))
    name = Path(__file__).name
    old = before['code_sha256'].pop(name, None); new = after['code_sha256'].pop(name, None)
    return (before == after and old in adoption.get('compatible_prior_runner_sha256', [adoption['old_runner_sha256']]) and
            new == adoption['new_runner_sha256'] and adoption['scientific_body_unchanged'])


def fixed_plan(sid, requests, work):
    add = work / 'addendum_20261023'
    cfg = read_json(work / 'config/analysis_contract.json')
    requests = requests[requests.sid.eq(sid)].copy()
    if not len(requests) or requests.request_id.duplicated().any() or requests.duplicated(['sid', 'UTC']).any():
        raise ValueError('Nonempty exact unique SID/UTC request table required')
    requests['UTC'] = pd.to_datetime(requests.UTC, utc=True)
    if (requests.UTC.astype('int64') % (cfg['dt_seconds'] * 10**9)).any():
        raise ValueError('Request UTC is not an exact native six-hour timestamp')
    valid = requests.position_available & requests.actual_track_point & ~requests.position_extrapolated & np.isfinite(requests.lat) & np.isfinite(requests.lon)
    good = requests[valid]
    if not len(good):
        raise ValueError('SID has no actual-position requests; retain missing ledger rather than invent a domain')
    legacy = work / 'cache' / sid
    oldids = set(pd.read_csv(work / 'source_data/events_frozen.csv', usecols=['sid']).sid)
    exists = (legacy / 'completed.json').exists()
    if sid in oldids and not exists:
        raise RuntimeError('Existing original SID has not completed; do not launch duplicate work')
    if exists:
        receipt = read_json(legacy / 'completed.json')
        if receipt['status'] != 'COMPLETE_DIRECT_DIAGNOSTICS' or receipt['config_sha256'] != sha(work / 'config/analysis_contract.json'):
            raise ValueError('Original SID completion/configuration mismatch')
        begin = pd.Timestamp(receipt['raw_UTC_begin'], tz='UTC')
        prior = receipt['domain']
        reference_domain = prior
        centres = good[['lat', 'lon']].copy()
        reference_lon = prior['lon_reference']
        centres.loc[-1] = [float(centres.lat.iloc[0]), reference_lon]
        centres = centres.sort_index()
        domain = sid_fixed_domain(centres)
        # Force the unchanged longitude branch; no fictitious point changes latitude bounds.
        actual = good[['lat', 'lon']].copy()
        reference = reference_lon
        unwrapped = reference + (actual.lon.to_numpy() - reference + 180) % 360 - 180
        angular = 800000. / 6371000.
        dl = np.degrees(np.arcsin(np.clip(np.sin(angular) / np.cos(np.radians(actual.lat)), -1, 1)))
        domain['lon_reference'] = reference
        domain['lat_bounds'] = [min(prior['lat_bounds'][0], np.floor((actual.lat.min() - np.degrees(angular)) / .25) * .25 - .5),
                                max(prior['lat_bounds'][1], np.ceil((actual.lat.max() + np.degrees(angular)) / .25) * .25 + .5)]
        domain['lon_bounds'] = [min(prior['lon_bounds'][0], np.floor(np.min(unwrapped - dl) / .25) * .25 - .5),
                                max(prior['lon_bounds'][1], np.ceil(np.max(unwrapped + dl) / .25) * .25 + .5)]
        domain['lat_bounds'] = [max(-89.75, domain['lat_bounds'][0]), min(89.75, domain['lat_bounds'][1])]
        oldreceipt_sha = sha(legacy / 'completed.json')
        reference_rule = 'Exactly inherited original SID reference, original primary1024 time span and original reference geographical domain'
    else:
        event = pd.read_csv(add / 'source_data/new_return_events.csv', dtype={'sid': str})
        event = event[event.sid.eq(sid) & event.ledger_eligible].sort_values('return_order')
        if not len(event):
            raise ValueError('New SID has no legitimate anchored return')
        first = pd.to_datetime(event.iloc[0].temporary_departure_UTC, utc=True)
        resolution = pd.to_datetime(event.iloc[0].resolution_UTC, utc=True)
        midpoint = (first + (resolution - first) / 2).floor('6h')
        begin = midpoint - pd.Timedelta(hours=512 * 6)
        tracks = pd.read_csv(add / 'source_data/new_lifecycle_tracks.csv', dtype={'sid': str})
        tracks = tracks[tracks.sid.eq(sid)].copy()
        tracks['UTC'] = pd.to_datetime(tracks.time, utc=True)
        anchors = tracks[(tracks.UTC.between(first - pd.Timedelta(hours=24), first + pd.Timedelta(hours=24))) |
                         (tracks.UTC.between(resolution - pd.Timedelta(hours=24), resolution + pd.Timedelta(hours=24)))]
        anchor_centres = anchors[['lat', 'lon']].replace([np.inf, -np.inf], np.nan).dropna()
        reference_domain = sid_fixed_domain(anchor_centres)
        centres = pd.concat([anchor_centres, good[['lat', 'lon']]], ignore_index=True).dropna()
        domain = sid_fixed_domain(centres)
        oldreceipt_sha = None
        reference_rule = 'Actual instantaneous above-ground area/time reference over the original-equivalent two-departure domain and primary1024 input; unchanged by added rise/Resolution request extent'
    seconds = (good.UTC.astype('int64') // 10**9).to_numpy()
    begin_second = begin.value // 10**9
    wanted = np.unique(np.concatenate([seconds - 21600, seconds, seconds + 21600]))
    indices = (wanted - begin_second) // 21600
    if indices.min() < 64 or indices.max() >= 1024:
        raise ValueError('The existing/common long input does not contain all exact time stencils in both schemes')
    plan = {'sid': sid, 'existing_completed_cache': exists, 'legacy_cache': str(legacy) if exists else None,
            'legacy_receipt_sha256': oldreceipt_sha, 'domain': domain, 'reference_domain': reference_domain,
            'raw_start_UTC': begin.isoformat(),
            'raw_N': 1088, 'N': 1024, 'j0': 5, 'j1': 8, 'shift_steps': 64,
            'dt_seconds': 21600, 'selected_UTC_seconds': wanted.tolist(), 'selected_union_indices': indices.tolist(),
            'request_count': len(requests), 'actual_position_requests': int(valid.sum()),
            'missing_position_requests': int((~valid).sum()), 'reference_rule': reference_rule,
            'target_regions': cfg['regions_km'], 'mask_rule': cfg['fixed_column_mask'],
            'no_continuity_reconstruction': True}
    # JSON receipts round-trip tuples as lists; use the same representation
    # before equality checks so a completed new SID is reusable unchanged.
    plan = json.loads(json.dumps(plan))
    return plan, requests, good


def cached_regional_requests(plan, requests, legacy, work, cfg):
    """Return exact already-sampled requests, or None when any point is absent."""
    if legacy is None:
        return None
    olddir = Path(plan['legacy_cache'])
    receipt = read_json(olddir / 'completed.json')
    if plan['domain'] != receipt['domain']:
        return None
    oldtimes = legacy['coordinates/selected_UTC_seconds'][:]
    if not np.isin(plan['selected_UTC_seconds'], oldtimes).all():
        return None
    centres_path = work / 'source_data/centres_frozen.csv'
    if sha(centres_path) != receipt['small_frozen_sources_sha256']['centres_frozen.csv']:
        raise ValueError('Original observed centre table changed')
    centres = pd.read_csv(centres_path, dtype={'sid': str})
    centres = centres[centres.sid.eq(plan['sid']) & centres.actual_track_point & ~centres.position_extrapolated].copy()
    centres['UTC'] = pd.to_datetime(centres.UTC, utc=True)
    regional_path = olddir / 'regional_metrics.parquet'
    if sha(regional_path) != receipt['outputs_sha256']['regional_metrics.parquet']:
        raise ValueError('Original regional output does not match its receipt')
    source = pd.read_parquet(regional_path)
    source['UTC'] = pd.to_datetime(source.UTC, utc=True)
    keys = ['scheme', 'region', 'metric', 'window', 'level']
    exact = ['value', 'units', 'pressure_hpa', 'inner_radius_km', 'outer_radius_km',
             'valid_area_fraction', 'valid_area_m2', 'geometric_area_m2',
             'n_valid_cells', 'n_geometric_cells', 'position_available']
    frames, mappings = [], []
    for point in requests.itertuples(index=False):
        if not bool(point.position_available and point.actual_track_point and not point.position_extrapolated):
            return None
        rows = source[source.UTC.eq(point.UTC)].copy()
        actual = centres[centres.UTC.eq(point.UTC)]
        if not len(rows) or not len(actual):
            return None
        if not ((actual.lat.to_numpy() == point.lat).all() and
                (np.abs((actual.lon.to_numpy() - point.lon + 180) % 360 - 180) < 1e-12).all()):
            return None
        # Two saved departure windows may contain this same real UTC. They must
        # agree exactly; selecting one never averages or selects by energy.
        duplicates = rows.duplicated(keys, keep=False)
        for _, group in rows[duplicates].groupby(keys, dropna=False, sort=False):
            for name in exact:
                vals = group[name].to_numpy()
                same = (vals == vals[0]) | (pd.isna(vals) & pd.isna(vals[0]))
                if not np.all(same):
                    raise ValueError('Duplicate original regional values differ: ' + name)
        rows = rows.sort_values(['event_id', 'lag_h']).drop_duplicates(keys)
        if set(rows.scheme) != {'main', 'shift16d'} or set(rows.region) != {'r0_300', 'r0_500', 'r200_800'}:
            return None
        for zone, bounds in cfg['regions_km'].items():
            z = rows[rows.zone.eq(zone)]
            if not len(z) or not (z.inner_radius_km.eq(bounds[0]) & z.outer_radius_km.eq(bounds[1])).all():
                raise ValueError('Original regional geometry differs')
        for old_event, old_lag in rows[['event_id', 'lag_h']].drop_duplicates().itertuples(index=False, name=None):
            mappings.append({'request_id': point.request_id, 'UTC': point.UTC.isoformat(),
                             'legacy_event_id': old_event, 'legacy_lag_h': old_lag,
                             'exact_regional_values_reused': True})
        rows = rows.drop(columns=['event_id', 'event_class', 't0', 'window_label'], errors='ignore')
        rows['lag_h'] = 0
        rows['request_id'] = point.request_id
        rows['UTC'] = point.UTC.isoformat()
        rows['source'] = 'actual_valley_rise_Resolution_request'
        frames.append(rows)
    return pd.concat(frames, ignore_index=True), pd.DataFrame(mappings)


def run_sid_increment(sid, work=WORK, stripe_rows=16, plan_only=False, output_root=None):
    work = Path(work).absolute()
    add = work / 'addendum_20261023'
    cfg = read_json(work / 'config/analysis_contract.json')
    rise_cfg = read_json(add / 'config/rise_analysis_contract.json')
    if cfg['MWT']['primary_N'] != 1024 or cfg['MWT']['j0'] != 5 or cfg['MWT']['j1'] != 8 or cfg['dt_seconds'] != 21600 or cfg['continuity_constrained_Gamma']:
        raise ValueError('The frozen direct-MWT numerical contract changed')
    path = add / 'source_data/rise_unique_point_requests.csv'
    requests = pd.read_csv(path, dtype={'sid': str, 'request_id': str})
    plan, requests, good = fixed_plan(sid, requests, work)
    code_paths = [Path(__file__), Path(__file__).parent / 'MWT_Apply_Increment.py'] + [work / 'scripts' / n for n in
                 ['read_era5_fixed_block.py', 'theory_MWT_Prepare_Direct.py', 'theory_MWT_Atmosphere_Direct.py', 'run_mwt_sid.py']]
    identities = {'old_config_sha256': sha(work / 'config/analysis_contract.json'),
                  'rise_config_sha256': sha(add / 'config/rise_analysis_contract.json'),
                  'requests_sha256': sha(path), 'code_sha256': {p.name: sha(p) for p in code_paths}}
    destination = Path(output_root).absolute() / sid if output_root else add / 'cache' / sid
    destination.mkdir(parents=True, exist_ok=True)
    if plan_only:
        save_json(destination / 'planned_request.json', {'plan': plan, 'identity': identities, 'not_executed': True})
        return {'status': 'PLAN_ONLY_NO_FIELDS_READ', 'path': str(destination), 'plan': plan}
    completed = destination / 'completed.json'
    if completed.exists():
        saved = read_json(completed)
        if equivalent_saved_identity(saved['identity'], identities, add) and saved['plan'] == plan and all(sha(destination / n) == h for n, h in saved['outputs_sha256'].items()):
            return {'status': 'REUSE_COMPLETE_INCREMENT', 'sid': sid, 'path': str(destination)}
        raise ValueError('Existing increment differs; never overwrite an unmatched completed sidecar')
    lock = destination / 'running.json'
    if lock.exists():
        prior = read_json(lock)
        try:
            process = psutil.Process(prior['pid'])
            if process.is_running() and abs(process.create_time() - prior['create_time']) < .001:
                raise RuntimeError('Exact-identity incremental SID worker already runs')
        except psutil.NoSuchProcess:
            pass
    process = psutil.Process()
    save_json(lock, {'pid': os.getpid(), 'create_time': process.create_time(), 'sid': sid})
    start = time.perf_counter()
    def status(step, **extra):
        record = {'sid': sid, 'step': step, 'seconds': time.perf_counter() - start,
                  'rss_bytes': process.memory_info().rss, **extra}
        deadline = time.monotonic() + 60
        while True:
            try:
                stream = (destination / 'run.jsonl').open('a', encoding='utf8')
                break
            except PermissionError as error:
                if getattr(error, 'winerror', None) not in (None, 5, 32, 33) or time.monotonic() >= deadline:
                    raise
                time.sleep(.2)
        with stream:
            stream.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
    legacy = None
    try:
        if plan['existing_completed_cache']:
            legacy_path = Path(plan['legacy_cache'])
            receipt = read_json(legacy_path / 'completed.json')
            for name in ['read_era5_fixed_block.py', 'theory_MWT_Prepare_Direct.py', 'theory_MWT_Atmosphere_Direct.py']:
                if receipt['code_sha256'][name] != identities['code_sha256'][name]:
                    raise ValueError('Original physical implementation differs from increment')
            legacy = h5py.File(legacy_path / 'diagnostics.h5', 'r')
        reused = cached_regional_requests(plan, requests, legacy, work, cfg)
        if reused is not None:
            frame, mapping = reused
            frame.to_parquet(destination / 'regional_metrics.parquet', index=False)
            mapping.to_csv(destination / 'regional_reuse_mapping.csv', index=False)
            requests[['sid', 'request_id', 'UTC', 'actual_track_point', 'position_available', 'missing_reason']].to_csv(destination / 'request_coverage.csv', index=False)
            # Links retain the exact original coefficient coordinates. This
            # small sidecar declares dependence on the protected original cache.
            with h5py.File(destination / 'increment.h5', 'w') as h:
                h.attrs['sid'] = sid; h.attrs['identity_json'] = json.dumps(identities)
                h.attrs['plan_json'] = json.dumps(plan); h.attrs['execution_mode'] = 'exact_regional_reuse'
                h.attrs['legacy_diagnostics_receipt_sha256'] = receipt['outputs_sha256']['diagnostics.h5']
                h.attrs['MWT_operator_sha256'] = receipt['operator_sha256']
                h.attrs['scope'] = 'No raw field read or transform; linked original coefficients and exact saved regional values'
                for name in ['coordinates', 'reference', 'basic_state', 'support', 'coefficients', 'raw_event_vertical_gradients']:
                    h[name] = h5py.ExternalLink(str(legacy_path / 'diagnostics.h5'), '/' + name)
            if identities != {'old_config_sha256': sha(work / 'config/analysis_contract.json'),
                              'rise_config_sha256': sha(add / 'config/rise_analysis_contract.json'),
                              'requests_sha256': sha(path), 'code_sha256': {p.name: sha(p) for p in code_paths}}:
                raise ValueError('Source identity changed while remapping original results')
            names = ['increment.h5', 'regional_metrics.parquet', 'regional_reuse_mapping.csv', 'request_coverage.csv']
            done = {'status': 'COMPLETE_EXACT_REQUEST_MWT_INCREMENT', 'sid': sid, 'identity': identities, 'plan': plan,
                    'execution_mode': 'exact_regional_reuse', 'operator_sha256': receipt['operator_sha256'],
                    'raw_inputs': [], 'original_outputs_reused': receipt['outputs_sha256'],
                    'outputs_sha256': {name: sha(destination / name) for name in names},
                    'elapsed_seconds': time.perf_counter() - start,
                    'completed_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    'no_original_cache_written': True, 'all_overlap_coefficients_bit_identical': True,
                    'raw_science_arrays_read': 0, 'MWT_transforms_executed': 0,
                    'scope': 'Actual point identity remapping of exact saved results; no new scientific calculation'}
            save_json(completed, done); status('COMPLETE_EXACT_REGIONAL_REUSE', rows=len(frame))
            return {'status': done['status'], 'sid': sid, 'path': str(destination), 'execution_mode': done['execution_mode']}
        inventory = read_json(work / 'reports/data_era5_inventory.json')
        source = {(x['field'], x['level_hpa']): x for x in inventory}
        for meta in source.values():
            info = Path(meta['path']).stat()
            if info.st_size != meta['size_bytes'] or info.st_mtime_ns != meta['mtime_ns']:
                raise ValueError('Original raw field identity changed')
        p = np.asarray(cfg['pressure_hpa'], float) * 100
        op = np.asarray(cfg['omega_pressure_hpa'], float) * 100
        oi = np.array([int(np.flatnonzero(p == pressure)[0]) for pressure in op])
        domain = plan['domain']
        # Coordinates are read from metadata only; no science array enters this step.
        with h5py.File(source[('sp', None)]['path'], 'r') as h:
            lat0 = h['latitude'][:]
            lon0 = h['longitude'][:]
        longitude = domain['lon_reference'] + (lon0 - domain['lon_reference'] + 180) % 360 - 180
        lat = np.sort(lat0[(lat0 >= domain['lat_bounds'][0]) & (lat0 <= domain['lat_bounds'][1])])
        lon = np.sort(longitude[(longitude >= domain['lon_bounds'][0]) & (longitude <= domain['lon_bounds'][1])])
        ny, nx = len(lat), len(lon)
        stripes = []
        y = 0
        while y < ny:
            end = min(ny, y + int(stripe_rows))
            if ny - end < 3:
                end = ny
            if end - y < 3:
                raise ValueError('Latitude stripe too short for original Prepare')
            stripes.append((y, end))
            y = end
        def area_weights(latitude, longitude):
            def edges(x):
                return np.r_[x[0] - (x[1] - x[0]) / 2, (x[:-1] + x[1:]) / 2, x[-1] + (x[-1] - x[-2]) / 2]
            return np.abs(np.diff(np.sin(np.clip(edges(np.radians(latitude)), -np.pi/2, np.pi/2))))[:, None] * np.abs(np.diff(edges(np.unwrap(np.radians(longitude)))))[None, :]
        area = area_weights(lat, lon)
        reference_domain = plan['reference_domain']
        reference_grid = ((lat[:, None] >= reference_domain['lat_bounds'][0]) &
                          (lat[:, None] <= reference_domain['lat_bounds'][1]) &
                          (lon[None, :] >= reference_domain['lon_bounds'][0]) &
                          (lon[None, :] <= reference_domain['lon_bounds'][1]))
        reference_area = area * reference_grid
        if reference_grid.sum() < 9 or not reference_area.sum() > 0:
            raise ValueError('Original-equivalent reference domain absent from requested grid')
        raw_identities = {}
        def raw_field(field, ys):
            levels = [None] if field == 'sp' else cfg['omega_pressure_hpa'] if field == 'omega' else cfg['pressure_hpa']
            result = None
            for pi, level in enumerate(levels):
                meta = source[(field, level)]
                block = read_era5_fixed_block(meta['path'], meta['variable'], plan['raw_start_UTC'], 1088,
                                             lat_bounds=(float(lat[ys.start]), float(lat[ys.stop - 1])),
                                             lon_bounds=domain['lon_bounds'], lon_reference=domain['lon_reference'])
                if not np.array_equal(block['latitude'], lat[ys]) or not np.array_equal(block['longitude'], lon):
                    raise ValueError('Increment raw spatial identity mismatch')
                expected_time = np.datetime64(plan['raw_start_UTC'].replace('+00:00', ''), 's') + np.arange(1088) * np.timedelta64(21600, 's')
                if not np.array_equal(block['UTC'], expected_time):
                    raise ValueError('Increment raw time identity mismatch')
                if field == 'sp':
                    result = block['field']
                else:
                    if result is None:
                        result = np.empty((1088, len(levels), ys.stop - ys.start, nx), float)
                    result[:, pi] = block['field']
                raw_identities[field + '/' + str(level)] = {'path': meta['path'], 'bytes': meta['size_bytes'],
                    'mtime_ns': meta['mtime_ns'], 'variable': meta['variable'], 'units': block['units'],
                    'native_time_start_index': block['native_time_start_index']}
            return result
        if legacy is not None:
            ref = {k: legacy['reference'][k][:] for k in ['temperature_mean', 'geopotential_reference', 'reference_weight_fraction', 'reference_valid_sample_count']}
            oldlat, oldlon = legacy['coordinates/latitude'][:], legacy['coordinates/longitude'][:]
            oldselected = legacy['coordinates/selected_union_indices'][:]
            oldy = np.searchsorted(lat, oldlat)
            oldx = np.searchsorted(lon, oldlon)
            if not np.array_equal(lat[oldy], oldlat) or not np.array_equal(lon[oldx], oldlon):
                raise ValueError('Extended domain does not preserve exact old cells')
        else:
            numerator = np.zeros((2, len(p))); denominator = np.zeros_like(numerator); counts = np.zeros(len(p), np.int64)
            for start_y, end_y in stripes:
                ys = slice(start_y, end_y)
                if not reference_grid[ys].any():
                    continue
                sp = raw_field('sp', ys)
                above = np.isfinite(sp[:1024, None]) & (p[None, :, None, None] < sp[:1024, None]) & reference_grid[ys][None, None]
                for k, name in enumerate(['t', 'z']):
                    x = raw_field(name, ys)[:1024]
                    valid = above & np.isfinite(x)
                    weight = reference_area[ys][None, None] * valid
                    numerator[k] += np.sum(np.where(valid, x, 0.) * weight, axis=(0, 2, 3))
                    denominator[k] += weight.sum(axis=(0, 2, 3))
                    if k == 0:
                        counts += valid.sum(axis=(0, 2, 3))
                    del x, valid, weight
                del sp, above
                status('new_reference_stripe', y0=start_y, y1=end_y)
            means = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
            tbar, phi_mean = means
            valid = np.isfinite(tbar) & np.isfinite(phi_mean)
            if not valid.any():
                raise ValueError('New SID has no real reference anchor')
            anchor = int(np.flatnonzero(valid)[np.argmax(p[valid])])
            phi_ref = np.full(len(p), np.nan); phi_ref[anchor] = phi_mean[anchor]
            for step in [-1, 1]:
                k = anchor + step
                while 0 <= k < len(p):
                    previous = k - step
                    if not valid[k] or not np.isfinite(phi_ref[previous]):
                        break
                    phi_ref[k] = phi_ref[previous] - 287.058 * .5 * (tbar[k] + tbar[previous]) * np.log(p[k] / p[previous])
                    k += step
            ref = dict(temperature_mean=tbar, geopotential_reference=phi_ref,
                       reference_weight_fraction=denominator[0] / (1024 * reference_area.sum()), reference_valid_sample_count=counts)
            oldselected = np.array([], dtype=int)
            oldy = oldx = np.array([], dtype=int)
        operator_path = work / 'cache/mwt_operator_N1024_j5_8_symmetric_physical.npy'
        operator = np.load(operator_path)
        if legacy is not None and sha(operator_path) != legacy.attrs['MWT_operator_sha256']:
            raise ValueError('Original operator differs')
        selected = np.asarray(plan['selected_union_indices'], int)
        selected_seconds = np.asarray(plan['selected_UTC_seconds'], np.int64)
        nt = len(selected)
        field_keys = ['u', 'v', 't', 'phi', 'uu', 'uv', 'vv', 'u_t', 'v_t', 'omega', 'omega_u', 'omega_v', 'omega_t']
        partial = destination / 'increment.partial.h5'
        copied_checks, workcounts, hydro_rows = [], [], []
        with h5py.File(partial, 'w') as h:
            h.attrs['sid'] = sid; h.attrs['identity_json'] = json.dumps(identities); h.attrs['plan_json'] = json.dumps(plan)
            h.attrs['MWT_operator_sha256'] = sha(operator_path)
            h.attrs['scope'] = 'Exact requested UTC extension; original coefficients retained where saved; no continuity reconstruction'
            coord = h.create_group('coordinates')
            for k, value in {'latitude': lat, 'longitude': lon, 'pressure_Pa': p, 'omega_pressure_Pa': op,
                             'selected_UTC_seconds': selected_seconds, 'selected_union_indices': selected}.items():
                coord.create_dataset(k, data=value)
            reference = h.create_group('reference')
            for k, value in ref.items():
                reference.create_dataset(k, data=value)
            support = h.create_group('support')
            fixed_ds = support.create_dataset('fixed_1088_pressure_grid', (13, ny, nx), dtype='bool')
            omega_ds = support.create_dataset('fixed_1088_omega_grid', (3, ny, nx), dtype='bool')
            for scheme in ['main', 'shift16d']:
                group = h.create_group('coefficients/' + scheme)
                for key in field_keys:
                    npres = 3 if key.startswith('omega') else 13
                    group.create_dataset(key, shape=(3, nt, npres, ny, nx), dtype='float64', fillvalue=np.nan,
                                         chunks=(1, min(3, nt), npres, min(16, ny), min(64, nx)),
                                         compression='gzip', compression_opts=1, shuffle=True)
            gradient_group = h.create_group('raw_event_vertical_gradients')
            for key, unit in [('du_dp', 'm s-1 Pa-1'), ('dv_dp', 'm s-1 Pa-1'),
                              ('dtprime_dp', 'K Pa-1'), ('hydrostatic_residual', 'm2 s-2 Pa-1')]:
                ds = gradient_group.create_dataset(key, shape=(nt, 13, ny, nx), dtype='float64', fillvalue=np.nan,
                    chunks=(min(3, nt), 13, min(16, ny), min(64, nx)), compression='gzip', compression_opts=1, shuffle=True)
                ds.attrs['units'] = unit
            basic_global = None
            for y0, y1 in stripes:
                ys = slice(y0, y1)
                raw = {field: raw_field(field, ys) for field in ['sp', 'u', 'v', 't', 'z', 'omega']}
                fixed = np.logical_and.reduce([np.isfinite(raw[k]).all(axis=0) for k in ['u', 'v', 't', 'z']])
                fixed &= (np.isfinite(raw['sp'])[:, None] & (raw['sp'][:, None] > p[None, :, None, None])).all(axis=0)
                omega_fixed = fixed[oi] & np.isfinite(raw['omega']).all(axis=0)
                cached_mask = np.zeros((y1 - y0, nx), bool)
                overlap_old_y = np.flatnonzero((oldy >= y0) & (oldy < y1))
                overlap_new_y = oldy[overlap_old_y] - y0
                if len(overlap_old_y):
                    cached_mask[np.ix_(overlap_new_y, oldx)] = True
                    for a, b in zip(overlap_old_y, overlap_new_y):
                        if not np.array_equal(fixed[:, b, oldx], legacy['support/fixed_1088_pressure_grid'][:, a, :]) or not np.array_equal(omega_fixed[:, b, oldx], legacy['support/fixed_1088_omega_grid'][:, a, :]):
                            raise ValueError('Inherited fixed mask changed on an original cell')
                fixed_ds[:, ys, :] = fixed; omega_ds[:, ys, :] = omega_fixed
                for scheme, offset in [('main', 0), ('shift16d', 64)]:
                    sl = slice(offset, offset + 1024)
                    data, basic = MWT_Prepare_Direct(raw['u'][sl], raw['v'][sl], raw['t'][sl], lon, lat[ys], p,
                        omega=raw['omega'][sl], omega_pressure=op, geopotential=raw['z'][sl],
                        temperature_mean=ref['temperature_mean'], geopotential_reference=ref['geopotential_reference'],
                        surface_pressure=raw['sp'][sl], valid_mask=fixed, include_vertical_gradients=False)
                    for key in ['omega', 'omega_u', 'omega_v', 'omega_t']:
                        data[key][:, ~omega_fixed] = np.nan
                    if basic_global is None:
                        basic_global = {k: np.array(v, copy=True) for k, v in basic.items() if k not in ['valid_mask', 'omega_valid_mask', 'area_weights', 'data_shape']}
                        basic_global.update(lon=lon, lat=lat, area_weights=area, data_shape=np.array([1024, 13, ny, nx]))
                        basic_global.update(reference_weight_fraction=ref['reference_weight_fraction'],
                                            reference_valid_sample_count=ref['reference_valid_sample_count'])
                        if legacy is not None:
                            for name in ['c', 'dc_dp', 'temperature_mean', 'geopotential_reference']:
                                if name in basic_global and name in legacy['basic_state'] and not np.array_equal(basic_global[name], legacy['basic_state'][name][:], equal_nan=True):
                                    raise ValueError('Inherited reference/basic state changed: ' + name)
                    for key in field_keys:
                        x = data.pop(key)
                        cached = None
                        if len(overlap_old_y):
                            npres = 3 if key.startswith('omega') else 13
                            cached = np.full((3, len(oldselected), npres, y1 - y0, nx), np.nan)
                            for a, b in zip(overlap_old_y, overlap_new_y):
                                old_values = legacy['coefficients'][scheme][key][:, :, :, a, :]
                                cached[:, :, :, b, oldx] = old_values
                        result, count = MWT_Apply_Increment(x, operator, selected - offset,
                            cached_coefficients=cached, cached_time_indices=oldselected - offset if cached is not None else None,
                            cached_spatial_mask=cached_mask if cached is not None else None)
                        if cached is not None:
                            for ti, index in enumerate(selected):
                                match = np.flatnonzero(oldselected == index)
                                if len(match):
                                    same = np.array_equal(result[:, ti, :, cached_mask], cached[:, int(match[0]), :, cached_mask], equal_nan=True)
                                    if not same:
                                        raise ValueError('An original cached coefficient was changed')
                            copied_checks.append({'scheme': scheme, 'field': key, 'y0': y0, 'y1': y1, 'all_original_overlap_coefficients_exact': True})
                        h['coefficients'][scheme][key][:, :, :, ys, :] = result
                        workcounts.append({'scheme': scheme, 'field': key, 'y0': y0, 'y1': y1, **count})
                        del x, cached, result
                    del data, basic
                    gc.collect()
                # The 13-level hydrostatic diagnostic is evaluated on real raw samples;
                # retain native gradients and summaries, not a forced closure.
                for field, key in [('u', 'du_dp'), ('v', 'dv_dp'), ('t', 'dtprime_dp'), ('z', 'hydrostatic_residual')]:
                    x = raw[field][selected].copy()
                    if field == 't':
                        x -= ref['temperature_mean'][None, :, None, None]
                    x = np.where(fixed[None], x, np.nan)
                    derivative = np.gradient(x, p, axis=1, edge_order=2)
                    if field == 'z':
                        derivative += 287.058 * raw['t'][selected] / p[None, :, None, None]
                    gradient_group[key][:, :, ys, :] = derivative
                    if field == 'z':
                        residual = derivative
                for ti, second in enumerate(selected_seconds):
                    for pi, pressure in enumerate(p):
                        rr = residual[ti, pi]; valid_r = np.isfinite(rr)
                        hydro_rows.append({'sid': sid, 'UTC_seconds': int(second), 'pressure_Pa': pressure,
                                           'y0': y0, 'y1': y1, 'finite_count': int(valid_r.sum()), 'total_count': rr.size,
                                           'sum_absolute_hydrostatic_residual': float(np.abs(rr[valid_r]).sum()), 'units': 'm2 s-2 Pa-1'})
                del raw, x, residual, derivative, fixed, omega_fixed
                h.flush(); gc.collect(); status('increment_stripe_saved', y0=y0, y1=y1, selected_times=nt)
            bg = h.create_group('basic_state')
            for k, value in basic_global.items():
                bg.create_dataset(k, data=value)
            regional = []
            for request in requests.itertuples(index=False):
                second = pd.Timestamp(request.UTC).value // 10**9
                position_ok = bool(request.position_available and request.actual_track_point and not request.position_extrapolated
                                   and np.isfinite(request.lat) and np.isfinite(request.lon))
                if not position_ok:
                    continue
                stencil_seconds = second + np.array([-21600, 0, 21600])
                ti = np.searchsorted(selected_seconds, stencil_seconds)
                if not np.array_equal(selected_seconds[ti], stencil_seconds):
                    raise ValueError('Missing exact centered derivative stencil')
                for scheme in ['main', 'shift16d']:
                    coefficients = {key: h['coefficients'][scheme][key][:, ti, ...] for key in field_keys}
                    diag = MWT_Atmosphere_Direct(coefficients, basic_global, dt=21600, gamma_form='conservative',
                                                mask_spatial_edges=True, return_flux_components=True)
                    centre = {0: (float(request.lat), float(request.lon))}
                    centre_diag = {k: v[:, 1:2] for k, v in diag.items()}
                    values = summarize_neighborhoods(centre_diag, p / 100, op / 100, lat, lon, centre, [0],
                                                     cfg['regions_km'], cfg['minimum_valid_area_fraction'])
                    for value in values:
                        value.update(sid=sid, request_id=request.request_id, UTC=pd.Timestamp(request.UTC).isoformat(),
                                     scheme=scheme, region={'primary': 'r0_300', 'environment': 'r200_800', 'radius_sensitivity': 'r0_500'}[value['zone']],
                                     source='actual_valley_rise_Resolution_request', position_available=True)
                    regional.extend(values)
                    del coefficients, diag, centre_diag
                status('requested_point_sampled', request_id=request.request_id)
            h.flush()
        os.replace(partial, destination / 'increment.h5')
        frame = pd.DataFrame(regional)
        missing_requests = requests[~(requests.position_available & requests.actual_track_point & ~requests.position_extrapolated &
                                     np.isfinite(requests.lat) & np.isfinite(requests.lon))]
        if len(missing_requests):
            template = frame[frame.request_id.eq(frame.request_id.iloc[0])].copy()
            missing_frames = []
            for request in missing_requests.itertuples(index=False):
                empty = template.copy()
                empty['request_id'] = request.request_id
                empty['UTC'] = pd.Timestamp(request.UTC).isoformat()
                empty['value'] = np.nan
                empty['valid_area_fraction'] = np.nan
                empty['valid_area_m2'] = 0.
                empty['geometric_area_m2'] = np.nan
                empty['n_valid_cells'] = 0
                empty['n_geometric_cells'] = 0
                empty['position_available'] = False
                missing_frames.append(empty)
            frame = pd.concat([frame, *missing_frames], ignore_index=True)
        frame.to_parquet(destination / 'regional_metrics.parquet', index=False)
        pd.DataFrame(workcounts).to_csv(destination / 'transform_reuse_counts.csv', index=False)
        pd.DataFrame(copied_checks).to_csv(destination / 'overlap_coefficient_exact_checks.csv', index=False)
        pd.DataFrame(hydro_rows).to_parquet(destination / 'hydrostatic_stripe_summaries.parquet', index=False)
        requests[['sid', 'request_id', 'UTC', 'actual_track_point', 'position_available', 'missing_reason']].to_csv(destination / 'request_coverage.csv', index=False)
        if identities != {'old_config_sha256': sha(work / 'config/analysis_contract.json'),
                          'rise_config_sha256': sha(add / 'config/rise_analysis_contract.json'),
                          'requests_sha256': sha(path), 'code_sha256': {p.name: sha(p) for p in code_paths}}:
            raise ValueError('Increment source/code/config changed during calculation')
        output_names = ['increment.h5', 'regional_metrics.parquet', 'transform_reuse_counts.csv',
                        'overlap_coefficient_exact_checks.csv', 'hydrostatic_stripe_summaries.parquet', 'request_coverage.csv']
        done = {'status': 'COMPLETE_EXACT_REQUEST_MWT_INCREMENT', 'sid': sid, 'identity': identities, 'plan': plan,
                'operator_sha256': sha(operator_path), 'raw_inputs': raw_identities,
                'outputs_sha256': {name: sha(destination / name) for name in output_names},
                'elapsed_seconds': time.perf_counter() - start, 'completed_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'no_original_cache_written': True, 'all_overlap_coefficients_bit_identical': all(x['all_original_overlap_coefficients_exact'] for x in copied_checks),
                'scope': 'Direct point diagnostics only; no event composite, statistical effect or full budget inference'}
        save_json(completed, done)
        status('COMPLETE', rows=len(frame))
        return {'status': done['status'], 'sid': sid, 'path': str(destination)}
    except Exception as error:
        save_json(destination / 'failure.json', {'sid': sid, 'error': repr(error), 'traceback': traceback.format_exc(), 'identity': identities})
        raise
    finally:
        if legacy is not None:
            legacy.close()
        if lock.exists():
            saved = read_json(lock)
            if saved['pid'] == os.getpid() and abs(saved['create_time'] - process.create_time()) < .001:
                remove_own_lock(lock)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', default=str(WORK))
    parser.add_argument('--sid', required=True)
    parser.add_argument('--stripe-rows', type=int, default=16)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--output-root')
    args = parser.parse_args()
    print(json.dumps(run_sid_increment(args.sid, args.work, args.stripe_rows, args.plan_only, args.output_root), indent=2))
