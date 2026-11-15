def run_checks(output_path):
    import json
    from pathlib import Path
    import numpy as np
    from vertical_terms import pressure_derivative, pressure_layer_mean, vertical_terms

    p = np.array([40000., 50000., 60000., 70000., 85000., 92500.])
    checks = []

    def check(label, residual, threshold=1e-12):
        residual = float(residual)
        checks.append(dict(check=label, max_abs_error=residual, tolerance=threshold,
                           passed=bool(np.isfinite(residual) and residual <= threshold)))

    x = p / 1e5
    check('nonuniform_quadratic_derivative_Pa',
          np.max(np.abs(pressure_derivative(x*x, p, 0) - 2*p/1e10)))
    check('alternative_three_point_quadratic_derivative_Pa',
          abs(pressure_derivative(x*x, p, 0, {70000.: [50000., 70000., 92500.]})[3]
              - 2*p[3]/1e10))
    check('alternative_five_point_quartic_derivative_Pa',
          abs(pressure_derivative(x**4, p, 0,
              {70000.: [40000., 50000., 60000., 70000., 85000.]})[3]
              - 4*x[3]**3/1e5))
    shape = (3, 5, len(p), 2, 2)
    expand = lambda profile: np.broadcast_to(np.asarray(profile)[None, None, :, None, None], shape).copy()
    u = expand(1 + x); v = expand(2 - 2*x); t = expand(3 + 0*x)
    wu = expand(4 + 0*x); wv = expand(5 + 0*x); wt = expand(6 + 0*x)
    c = 2 + x; tref = 270 - 5*x
    z = vertical_terms(u, v, t, wu, wv, wt, p, c, tref)
    expected_k = 0.5*(4/1e5 - 10/1e5)
    check('linear_K_vertical_conservative', np.max(np.abs(z['Gamma_K_p'] - expected_k)))
    check('linear_K_vertical_bracket', np.max(np.abs(z['Gamma_K_p_bracket'] - expected_k)))
    check('cprime_correction_separated_from_A_canonical', np.max(np.abs(z['Gamma_A_p'])))
    check('constant_T_A_bracket_zero', np.max(np.abs(z['Gamma_A_p_bracket'])))
    check('cprime_correction_positive_value', np.max(np.abs(z['S_A_cprime'] - 9e-5)))
    check('thermodynamic_correction', np.max(np.abs(z['S_A_thermodynamic']
          - expand(287.058*18/(p*tref)))))
    layer = pressure_layer_mean(u, p)
    check('linear_layer_mean_exact', np.max(np.abs(layer['mean'] - 1.675)))
    check('layer_column_integral_units_factor',
          np.max(np.abs(layer['column_integral'] - layer['mean']*35000/9.8)), 1e-10)
    distorted = u.copy(); distorted[0, 0, 2, 0, 0] = np.nan
    mask = pressure_layer_mean(distorted, p)
    checks.append(dict(check='missing_layer_not_partial_mean', passed=bool(
        not mask['common_valid'][0,0,0,0] and np.isnan(mask['mean'][0,0,0,0])),
        max_abs_error=None, tolerance=None))
    sparse = pressure_layer_mean(u, p, levels_pa=(50000.,70000.,85000.))
    check('linear_sparse_and_dense_integral_match',
          np.max(np.abs(sparse['mean']-layer['mean'])))
    checks.append(dict(check='different_shape_rejected', passed=False,
                       max_abs_error=None,tolerance=None))
    try:
        vertical_terms(u, v[:,:-1], t, wu, wv, wt, p, c, tref)
    except ValueError:
        checks[-1]['passed'] = True
    payload = {'status':'PASS' if all(c['passed'] for c in checks) else 'FAIL',
               'scope':'Analytical numerical checks only; no scientific event or ERA5 calculation',
               'checks':checks}
    Path(output_path).write_text(json.dumps(payload,indent=2),encoding='utf-8')
    return payload


if __name__ == '__main__':
    import sys
    result = run_checks(sys.argv[1])
    print(result['status'],len(result['checks']),'checks')
    if result['status'] != 'PASS':
        raise SystemExit(1)
