"""Pressure-direction MWT terms for the isolated 2026-11-15 diagnostic.

No I/O, global configuration, transformed-field multiplication approximation,
or change to the frozen horizontal calculation is performed by this module.
All input coefficients use the existing physical MWT normalization.
"""


def pressure_derivative(values, pressure_pa, axis=2, templates=None):
    """Differentiate on an uneven Pa grid using local polynomial weights.

    Default is numpy.gradient(edge_order=2), i.e. three-point centered interior
    differences and three-point one-sided edges. ``templates`` optionally maps
    a target pressure in Pa to an explicit sequence of at least three pressures
    in the supplied grid; the resulting derivative is interpolatory and uses no
    pressure imputation. Unspecified levels retain the default stencil.
    NaNs propagate through the stencil. Returns an array with the input shape.
    """
    import numpy as np

    x = np.asarray(values, dtype=float)
    p = np.asarray(pressure_pa, dtype=float)
    axis = np.core.numeric.normalize_axis_index(axis, x.ndim)
    if p.ndim != 1 or len(p) < 3 or x.shape[axis] != len(p):
        raise ValueError('Pressure and array dimensions disagree or fewer than three levels')
    if not np.isfinite(p).all() or not np.all(np.diff(p) > 0) or np.any(p <= 0):
        raise ValueError('Pressure must be finite, strictly increasing, positive Pa')
    out = np.gradient(x, p, axis=axis, edge_order=2)
    if templates is None:
        return out
    for target, support in templates.items():
        target = float(target)
        support = np.asarray(support, dtype=float)
        if support.ndim != 1 or len(support) < 3 or len(np.unique(support)) != len(support):
            raise ValueError('An alternative stencil requires at least three distinct levels')
        ti = np.flatnonzero(p == target)
        si = [np.flatnonzero(p == v) for v in support]
        if len(ti) != 1 or any(len(k) != 1 for k in si):
            raise ValueError('All stencil levels and target must exist in input pressure')
        si = np.asarray([int(k[0]) for k in si])
        delta = support - target
        scale = float(np.max(np.abs(delta)))
        if scale <= 0:
            raise ValueError('Degenerate pressure stencil')
        z = delta / scale
        powers = np.vstack([z ** k for k in range(len(z))])
        rhs = np.zeros(len(z)); rhs[1] = 1.0 / scale
        weights = np.linalg.solve(powers, rhs)
        xx = np.moveaxis(x, axis, 0)
        replacement = np.tensordot(weights, xx[si], axes=(0, 0))
        oo = np.moveaxis(out, axis, 0)
        oo[int(ti[0])] = replacement
    return out


def vertical_terms(u, v, t, omega_u, omega_v, omega_t, pressure_pa, c,
                   t_reference, dc_dp=None, gas=287.058, templates=None):
    """Return canonical pressure contributions and separate APE corrections.

    Coefficient arrays are (window,time,pressure,latitude,longitude). ``t`` is
    the coefficient of T-prime about the frozen reference. omega_u/v/t are
    MWT(original omega times original u/v/T-prime), not coefficient products.
    Pressure is Pa; c is the frozen dry-APE coefficient (J kg-1 K-2), reference
    temperature K, and optional dc_dp its frozen derivative per Pa.

    Primary contributions use the original conservative advective/flux split.
    Analytically equivalent bracket forms are separate numerical-sensitivity
    outputs. S_A_cprime and S_A_thermodynamic are diagnosed separately and are
    never added to Gamma_A_p. All rate outputs are W kg-1; Q_*_p have units
    Pa m2 s-3 and must be differenced across boundaries before normalization.
    """
    import numpy as np

    arrays = [np.asarray(a, dtype=float) for a in (u, v, t, omega_u, omega_v, omega_t)]
    shape = arrays[0].shape
    if len(shape) != 5 or any(a.shape != shape for a in arrays):
        raise ValueError('All coefficients must share five axes/window-time-pressure-lat-lon')
    if any(np.isinf(a).any() for a in arrays):
        raise ValueError('Infinite coefficients are invalid')
    p, c, tref = (np.asarray(a, dtype=float) for a in (pressure_pa, c, t_reference))
    if p.shape != (shape[2],) or c.shape != p.shape or tref.shape != p.shape:
        raise ValueError('Reference profiles must match pressure')
    if np.any(c[np.isfinite(c)] <= 0) or np.any(tref[np.isfinite(tref)] <= 0):
        raise ValueError('Valid reference temperature and dry-APE c must be positive')
    if not np.isfinite(gas) or gas <= 0:
        raise ValueError('Gas constant must be positive')
    u, v, t, wu, wv, wt = arrays
    d = lambda a: pressure_derivative(a, p, axis=2, templates=templates)
    du, dv, dt, dwu, dwv, dwt = [d(a) for a in arrays]
    c5 = c[None, None, :, None, None]
    tref5 = tref[None, None, :, None, None]
    if dc_dp is None:
        dc = pressure_derivative(c, p, axis=0, templates=templates)
    else:
        dc = np.asarray(dc_dp, dtype=float)
        if dc.shape != p.shape or np.isinf(dc).any():
            raise ValueError('dc_dp profile mismatch')
    dc5 = dc[None, None, :, None, None]
    qk = 0.5 * (u * wu + v * wv)
    qa = 0.5 * c5 * t * wt
    sa_cprime = 0.5 * t * wt * dc5
    sa_thermo = (float(gas) / p[None, None, :, None, None]) * t * wt / tref5
    gkp = -(u * dwu + v * dwv) + d(qk)
    gap = -c5 * t * dwt + d(qa) - sa_cprime
    gkp_bracket = 0.5 * (wu * du - u * dwu + wv * dv - v * dwv)
    gap_bracket = 0.5 * c5 * (wt * dt - t * dwt)
    return {
        'Gamma_K_p': gkp, 'Gamma_A_p': gap,
        'Gamma_K_p_bracket': gkp_bracket, 'Gamma_A_p_bracket': gap_bracket,
        'Q_K_p': qk, 'Q_A_p': qa,
        'S_A_cprime': sa_cprime, 'S_A_thermodynamic': sa_thermo,
        'du_dp': du, 'dv_dp': dv, 'dt_dp': dt,
        'domega_u_dp': dwu, 'domega_v_dp': dwv, 'domega_t_dp': dwt,
        'dc_dp': dc,
    }


def pressure_layer_mean(values, pressure_pa, levels_pa=(50000., 60000., 70000., 85000.),
                        axis=2, gravity=9.8):
    """Return a common-column trapezoidal pressure mean and dp/g integral.

    Every selected level must be finite in a column. Mean units equal those of
    ``values``; column-integral units gain kg m-2. No missing level is filled,
    and no partial-column normalization is performed. A three-level call may
    be used only as the explicitly labeled sparse-layer sensitivity.
    """
    import numpy as np

    a = np.asarray(values, dtype=float)
    p = np.asarray(pressure_pa, dtype=float)
    levels = np.asarray(levels_pa, dtype=float)
    axis = np.core.numeric.normalize_axis_index(axis, a.ndim)
    if p.ndim != 1 or a.shape[axis] != len(p) or not np.all(np.diff(p) > 0):
        raise ValueError('Pressure dimension/order mismatch')
    if len(levels) < 2 or not np.all(np.diff(levels) > 0):
        raise ValueError('Layer levels must be increasing')
    if not np.isfinite(gravity) or gravity <= 0:
        raise ValueError('gravity must be positive')
    ind = [np.flatnonzero(p == v) for v in levels]
    if any(len(k) != 1 for k in ind):
        raise ValueError('Every requested level must be present')
    selected = np.take(a, [int(k[0]) for k in ind], axis=axis)
    common = np.isfinite(selected).all(axis=axis)
    integral = np.trapz(selected, levels, axis=axis)
    integral = np.where(common, integral, np.nan)
    return {
        'mean': integral / (levels[-1] - levels[0]),
        'column_integral': integral / float(gravity),
        'common_valid': common,
        'pressure_thickness_pa': float(levels[-1] - levels[0]),
    }
