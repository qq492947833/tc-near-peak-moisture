"""70-hPa T/q from 100 and 50 hPa, values linear in log pressure.

The alternative is the single fixed pressure-linear sensitivity. Neither uses
log humidity, extrapolation, or a replacement for an unavailable native field.
"""
import numpy as np

def interpolate_variable(pressures, values, target, pressure_units='hPa', variable='t', units='K', method='logp'):
    p = np.asarray(pressures, dtype=np.float64)
    a = np.asarray(np.ma.filled(np.ma.asarray(values, dtype=float), np.nan), dtype=np.float64)
    target = np.float64(target)
    if pressure_units == 'Pa':
        p = p / 100.0
        target = target / 100.0
    elif pressure_units != 'hPa':
        raise ValueError('pressure must explicitly be hPa or Pa')
    if p.shape != (2,) or a.shape[0] != 2 or len(set(p.tolist())) != 2 or (not np.all(np.isfinite(p))) or (not np.all(p > 0)):
        raise ValueError('two unique finite positive pressure endpoints required')
    order = np.argsort(p)
    p = p[order]
    a = a[order]
    if not np.array_equal(p, [50.0, 100.0]):
        raise ValueError('Exactly 50/100 hPa endpoints are required')
    if variable == 't':
        if units != 'K':
            raise ValueError('temperature must be K, conversion must be explicit before this boundary')
        valid = np.isfinite(a).all(axis=0) & (a > 100.0).all(axis=0) & (a < 400.0).all(axis=0)
    elif variable == 'q':
        if units not in ('kg kg-1', 'kg kg**-1', 'kg/kg'):
            raise ValueError('specific humidity must be kg/kg, not mixing ratio')
        valid = np.isfinite(a).all(axis=0) & (a >= 0).all(axis=0) & (a < 0.1).all(axis=0)
    else:
        raise ValueError('Only temperature and specific humidity are supported')
    if not np.isfinite(target) or target < p[0] or target > p[1]:
        return np.full(a.shape[1:], np.nan, dtype=np.float64)
    if method == 'logp':
        w50 = np.log(target / 100.0) / np.log(50.0 / 100.0)
    elif method == 'linear_p':
        w50 = (target - 100.0) / (50.0 - 100.0)
    else:
        raise ValueError('only mainlogp and fixedlinear_p sensitivity')
    return np.where(valid, (1.0 - w50) * a[1] + w50 * a[0], np.nan)

def joint70(t100, t50, q100, q50, method='logp'):
    t = interpolate_variable([100.0, 50.0], np.stack([t100, t50]), 70.0, variable='t', units='K', method=method)
    q = interpolate_variable([100.0, 50.0], np.stack([q100, q50]), 70.0, variable='q', units='kg kg-1', method=method)
    joint = np.isfinite(t) & np.isfinite(q)
    return (np.where(joint, t, np.nan), np.where(joint, q, np.nan), joint)

def align_native(values, source_lat, source_lon, target_lat, target_lon):
    a = np.asarray(np.ma.filled(np.ma.asarray(values, dtype=float), np.nan), dtype=np.float64)
    sy = np.asarray(source_lat, dtype=np.float64)
    sx = np.mod(np.asarray(source_lon, dtype=np.float64), 360.0)
    ty = np.asarray(target_lat, dtype=np.float64)
    tx = np.mod(np.asarray(target_lon, dtype=np.float64), 360.0)
    if a.shape != (len(sy), len(sx)) or len(set(sy)) != len(sy) or len(set(sx)) != len(sx):
        raise ValueError('shape/duplicate coordinate conflict')
    iy = {v: i for i, v in enumerate(sy)}
    ix = {v: i for i, v in enumerate(sx)}
    try:
        ii = [iy[v] for v in ty]
        jj = [ix[v] for v in tx]
    except KeyError as e:
        raise ValueError('exact coordinate absent; no nearest or regridding') from e
    return a[np.ix_(ii, jj)]
