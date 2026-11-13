"""Small implementation checks and replay of one saved coefficient stencil."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).absolute().parent
FUNCTIONS = ('MWT', 'MWT_Prepare_Direct', 'MWT_Apply_Operator',
             'MWT_Atmosphere_Direct', 'MWT_Apply_Increment')


def load_functions():
    functions = {}
    records = []
    for name in FUNCTIONS:
        path = ROOT / 'src' / (name + '.py')
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        definitions = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(x.name for x in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module)
        if definitions != [name] or set(imports) - {'numpy', 'operator'}:
            raise AssertionError('Unexpected function or dependency: ' + name)
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        functions[name] = getattr(module, name)
        records.append({'file': 'src/' + path.name,
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'imports': imports})
    return functions, records


def compare(name, actual, expected, records, units='1', eps_multiplier=512):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape:
        raise AssertionError(name + ': shape differs')
    finite = np.isfinite(expected)
    if not np.array_equal(np.isfinite(actual), finite):
        raise AssertionError(name + ': finite-value mask differs')
    difference = np.abs(actual[finite] - expected[finite])
    scale = np.maximum(1., np.abs(expected[finite]))
    max_abs = float(difference.max(initial=0))
    scaled = float((difference / scale).max(initial=0))
    if scaled > eps_multiplier * np.finfo(np.float64).eps:
        raise AssertionError(name + ': arithmetic tolerance exceeded: ' + str(scaled))
    records.append({'name': name, 'units': units, 'finite_values': int(finite.sum()),
                    'missing_values': int((~finite).sum()), 'maximum_absolute_difference': max_abs,
                    'maximum_scaled_difference': scaled,
                    'scaled_tolerance': eps_multiplier * np.finfo(np.float64).eps})


def analytical_input():
    n = 64
    pressure = 100. * np.array([50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000])
    lat, lon = np.linspace(12., 16., 5), np.linspace(165., 170., 6)
    time, p = np.arange(n)[:, None, None, None], pressure[None, :, None, None]
    y, x = lat[None, None, :, None], lon[None, None, None, :]
    shape = (n, len(pressure), len(lat), len(lon))
    u = np.broadcast_to(8 + np.sin(2*np.pi*time/16) + .001*(p/100-700) + .1*(y-14) + .05*(x-168), shape).copy()
    v = np.broadcast_to(3 + .5*np.cos(2*np.pi*time/8) + .02*(y-14) + .03*(x-168), shape).copy()
    temperature = np.broadcast_to(250 + 20*np.log(p/70000) + .6*np.sin(2*np.pi*time/32) + .02*(y-14) + .03*(x-168), shape).copy()
    geopotential = np.broadcast_to(-287.058*(250*np.log(p/70000)+10*np.log(p/70000)**2)
                                  + 2*(x-168)+3*(y-14)+.3*np.cos(2*np.pi*time/32), shape).copy()
    omega = np.broadcast_to(.1*np.sin(2*np.pi*time/16)*(p/70000), shape).copy()[:, [7, 9, 10]]
    sp = np.full((n, len(lat), len(lon)), 102000.)
    return dict(u=u, v=v, temperature=temperature, lon=lon, lat=lat, pressure=pressure,
                geopotential=geopotential, omega=omega, surface_pressure=sp)


def verify_analytical(functions, output):
    checks = []
    raw = analytical_input()
    np.savez_compressed(output / 'analytical_input.npz', **raw)
    prepare = functions['MWT_Prepare_Direct']
    mwt = functions['MWT']
    apply = functions['MWT_Apply_Operator']
    increment = functions['MWT_Apply_Increment']
    diagnose = functions['MWT_Atmosphere_Direct']
    fields, basic = prepare(**raw)
    n, j0, j1 = 64, 1, 4
    settings = dict(j0=j0, j1=j1, boundary='symmetric', normalization='physical', nan_policy='propagate')
    field_units = {'u': 'm s-1', 'v': 'm s-1', 't': 'K', 'phi': 'm2 s-2',
                   'uu': 'm2 s-2', 'uv': 'm2 s-2', 'vv': 'm2 s-2',
                   'u_t': 'm K s-1', 'v_t': 'm K s-1', 'omega': 'Pa s-1',
                   'omega_u': 'Pa m s-2', 'omega_v': 'Pa m s-2', 'omega_t': 'Pa K s-1'}
    op = mwt(np.eye(n), return_reconstruction=False, **settings)
    coefficients, reconstruction = mwt(fields, **settings)
    operator_coefficients = apply(fields, op)
    for key in fields:
        compare('operator_' + key, operator_coefficients[key], coefficients[key], checks,
                field_units[key], eps_multiplier=4096)
        compare('reconstruction_' + key, reconstruction[key].sum(axis=0), fields[key], checks,
                field_units[key], eps_multiplier=4096)
    original = mwt(fields['u'], j0=j0, j1=j1, boundary='symmetric', normalization='original',
                   return_reconstruction=False)
    compare('physical_normalization', coefficients['u'], np.sqrt(n)*original, checks, 'm s-1')
    if np.allclose(coefficients['u']**2, coefficients['uu']):
        raise AssertionError('Nonlinear products unexpectedly treated as coefficient products')
    checks.append({'name': 'original_product_transformed_separately', 'pass': True})
    indices = np.array([16, 17, 18, 19, 20])
    for key in fields:
        compare('selected_operator_' + key, apply(fields[key], op, indices), coefficients[key][:, indices],
                checks, field_units[key], eps_multiplier=4096)
    bad = fields['u'].copy()
    bad[10, 9, 2, 2] = np.nan
    missing = apply(bad, op, indices)
    if not np.isnan(missing[:, :, 9, 2, 2]).all():
        raise AssertionError('Incomplete time column was filled')
    cache_times = indices[:2]
    cache = apply(bad, op, cache_times)
    spatial = np.zeros(bad.shape[-2:], bool)
    spatial[1:4, 1:4] = True
    result, counts = increment(bad, op, indices, cache, cache_times, spatial)
    compare('increment_saved_and_new_rows', result, missing, checks, 'm s-1', eps_multiplier=4096)
    if counts['new_output_time_rows'] != 3 or counts['existing_output_time_rows'] != 2:
        raise AssertionError('Increment did not distinguish saved/new rows')
    checks.append({'name': 'increment_counts', **counts})
    low_surface = dict(raw)
    low_surface['surface_pressure'] = np.full_like(raw['surface_pressure'], 98000.)
    masked, masked_basic = prepare(**low_surface)
    if not np.isnan(masked['u'][:, -1]).all() or not np.isfinite(masked['u'][:, 9]).all():
        raise AssertionError('Underground-layer mask propagated to valid 700 hPa')
    if not np.isnan(masked_basic['temperature_mean'][-1]) or not np.isfinite(masked_basic['c'][9]):
        raise AssertionError('Reference missing-layer rule failed')
    checks.append({'name': 'underground_1000_retains_700', 'pass': True})
    malformed = dict(raw)
    malformed['omega'] = np.zeros_like(raw['u'])
    try:
        prepare(**malformed)
    except ValueError:
        checks.append({'name': 'mismatched_omega_layers_rejected', 'pass': True})
    else:
        raise AssertionError('Thirteen-level omega accepted without matching pressure identity')
    diagnostic = diagnose(coefficients, basic)
    if len(diagnostic) != 25:
        raise AssertionError('Diagnostic inventory changed')
    oi = basic['omega_indices']
    centre = (..., slice(1, -1), slice(1, -1))
    c = basic['c'][oi, None, None]
    independent = {
        'C_A_to_K': -287.058/basic['omega_pressure'][:, None, None] * coefficients['t'][:, :, oi] * coefficients['omega'],
        'Q_K_p': .5*(coefficients['u'][:, :, oi]*coefficients['omega_u'] + coefficients['v'][:, :, oi]*coefficients['omega_v']),
        'Q_A_p': .5*c*coefficients['t'][:, :, oi]*coefficients['omega_t'],
        'Q_P_p': coefficients['omega']*coefficients['phi'][:, :, oi],
    }
    for key, expected in independent.items():
        compare('formula_' + key, diagnostic[key][centre], expected[centre], checks,
                'W kg-1' if key == 'C_A_to_K' else 'Pa m2 s-3')
    for label in 'KAP':
        boundary = independent['Q_' + label + '_p'][:, :, 0] - independent['Q_' + label + '_p'][:, :, 2]
        for suffix, denominator, units in [('Wkg', 35000., 'W kg-1'), ('Wm2', 9.8, 'W m-2')]:
            compare('boundary_' + label + '_' + suffix,
                    diagnostic['vertical_input_' + label + '_' + suffix][centre], (boundary/denominator)[centre], checks, units)
    return {'input_type': 'deterministic analytical implementation fixture',
            'N': n, 'j0': j0, 'j1': j1, 'dt_seconds': 21600,
            'not_used_for_event_science_or_endpoint_inference': True,
            'operator_basis': 'Exact supplied MWT applied to identity matrix', 'checks': checks}


def verify_saved(functions, output):
    meta = json.loads((ROOT / 'inputs/saved_coefficients_example.json').read_text(encoding='utf-8'))
    semantics = json.loads((ROOT / 'config/diagnostic_semantics.json').read_text(encoding='utf-8'))
    metric_rows = semantics.get('metrics', [])
    units = {row['metric']: row['units_stored'] for row in metric_rows}
    with np.load(ROOT / 'inputs/saved_coefficients_example.npz', allow_pickle=False) as source:
        coef = {k.split('__', 1)[1]: source[k] for k in source.files if k.startswith('coefficient__')}
        basic = {k.split('__', 1)[1]: source[k] for k in source.files if k.startswith('basic__')}
        expected = {k.split('__', 1)[1]: source[k] for k in source.files if k.startswith('expected__')}
        utc = source['UTC_seconds']
        if not np.array_equal(np.diff(utc), [meta['dt_seconds']]*2):
            raise AssertionError('UTC interval differs from diagnostic dt')
        if not np.array_equal(np.isfinite(coef['u']).all(axis=(0, 1)), source['fixed_pressure_mask']):
            raise AssertionError('Saved pressure-column support differs')
        if not np.array_equal(np.isfinite(coef['omega']).all(axis=(0, 1)), source['fixed_omega_mask']):
            raise AssertionError('Saved omega-column support differs')
    actual = functions['MWT_Atmosphere_Direct'](coef, basic, dt=meta['dt_seconds'],
                                                gamma_form=meta['gamma_form'])
    if set(actual) != set(expected) or len(actual) != 25:
        raise AssertionError('Saved 25-diagnostic inventory mismatch')
    checks, replay = [], {}
    for key in sorted(actual):
        centre = actual[key][:, 1, ..., 1, 1]
        compare(key, centre, expected[key], checks, units.get(key, 'see diagnostic_semantics.json'))
        replay[key] = centre
    np.savez_compressed(output / 'saved_coefficient_replay.npz', **replay)
    return {'sid': meta['sid'], 'event_id': meta['event_id'], 'event_class': meta['event_class'],
            'scheme': meta['scheme'], 'checks': checks,
            'scope': 'Single-event local diagnostic replay from saved coefficients; not raw ERA5 extraction, full MWT, population inference or physical closure validation.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    sys.dont_write_bytecode = True
    functions, identity = load_functions()
    provenance = json.loads((ROOT / 'config/source_provenance.json').read_text(encoding='utf-8'))
    expected = {r['path']: r['sha256'] for r in provenance['core_functions']}
    for entry in identity:
        if entry['sha256'] != expected[entry['file']]:
            raise AssertionError('Core source differs from recorded implementation')
    result = {'status': 'COMPLETE_SMALL_IMPLEMENTATION_AND_COEFFICIENT_REPLAY',
              'python': sys.version.split()[0], 'numpy': np.__version__, 'source_functions': identity,
              'analytical': verify_analytical(functions, output),
              'saved_coefficients': verify_saved(functions, output),
              'raw_ERA5_recalculated': False, 'population_statistics_recalculated': False}
    (output / 'verification.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'analytical_checks': len(result['analytical']['checks']),
                      'saved_diagnostic_checks': len(result['saved_coefficients']['checks'])}))


if __name__ == '__main__':
    main()
