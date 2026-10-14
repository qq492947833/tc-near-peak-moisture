"""Float64 finite-change accounting on fixed cyclone-relative boundary nodes."""
import numpy as np

G0 = 9.80665
PRESSURES_PA = np.array([50000., 60000., 70000., 85000., 92500.])

def trap_weights(x):
    x = np.asarray(x, dtype=np.float64)
    assert x.ndim == 1 and len(x) >= 2 and np.all(np.diff(x) > 0)
    w = np.zeros_like(x)
    w[:-1] += np.diff(x) / 2
    w[1:] += np.diff(x) / 2
    return w

def integrate(f, times_s, pressures_pa, dl, sector=None):
    """f has time, pressure, azimuth axes; no missing-node replacement."""
    f = np.asarray(f, dtype=np.float64)
    assert f.shape == (len(times_s), len(pressures_pa), len(dl))
    w = trap_weights(times_s)[:, None, None] * trap_weights(pressures_pa)[None, :, None] * np.asarray(dl)[None, None, :] / G0
    if sector is not None:
        w = w * np.asarray(sector)[None, None, :]
    return float(np.sum(f * w))

def local_components(qt, qf, at, af, bt, bf, kind='out'):
    """q/relative motion first, then absolute normal wind/boundary motion.

    This prescribed hierarchy is not a symmetric three-factor Shapley value.
    Inputs correspond in north-clockwise azimuth, pressure and relative lag;
    they are neither the same geographical locations nor the same air parcels.
    """
    if kind == 'net':
        ins = local_components(qt, qf, at, af, bt, bf, 'in')
        outs = local_components(qt, qf, at, af, bt, bf, 'out')
        return {k: ins[k] - outs[k] for k in ins}
    sign = 1. if kind == 'out' else -1.
    assert kind in ('out', 'in')
    h = lambda a, b: np.maximum(sign * (a - b), 0.)
    htt, hft, htf, hff = h(at, bt), h(af, bt), h(at, bf), h(af, bf)
    cq = .5 * (qt - qf) * (htt + hff)
    cv = .5 * (htt - hff) * (qt + qf)
    ca = .5 * ((htt - hft) + (htf - hff))
    cb = .5 * ((htt - htf) + (hft - hff))
    return {'temporary': qt * htt, 'final': qf * hff,
            'difference': qt * htt - qf * hff,
            'humidity': cq, 'relative_motion': cv,
            'absolute_normal_wind': .5 * (qt + qf) * ca,
            'boundary_normal_motion': .5 * (qt + qf) * cb}

def integrated_components(qt, qf, at, af, bt, bf, times_s, pressures_pa, dl, kind, sector=None):
    return {k: integrate(v, times_s, pressures_pa, dl, sector)
            for k, v in local_components(qt, qf, at, af, bt, bf, kind).items()}

def outward_diagnostics(q, a, b, times_s, pressures_pa, dl):
    vn = a - b
    indicator = (vn > 0).astype(float)
    L = lambda x: integrate(x, times_s, pressures_pa, dl)
    base, active = L(np.ones_like(q)), L(indicator)
    air = L(np.maximum(vn, 0))
    water = L(q * np.maximum(vn, 0))
    return {'outward_fraction': active / base,
            'conditional_normal_speed_m_s': air / active if active > 0 else np.nan,
            'outward_air_mass_kg': air,
            'flux_weighted_humidity_kg_kg': water / air if air > 0 else np.nan,
            'outward_water_kg': water}

def refine_grid(data, times_s, pressures_pa, factor):
    """Bilinear interpolation in lag/pressure at unchanged azimuth indices.

    This numerical integration sensitivity creates no new atmospheric data.
    Refining a and b separately is identical to refining their difference.
    """
    assert factor in (1, 2, 4)
    data = np.asarray(data, dtype=np.float64)
    if factor == 1:
        return data.copy(), np.asarray(times_s).copy(), np.asarray(pressures_pa).copy()
    def refined(x):
        return np.concatenate([np.linspace(x[i], x[i+1], factor+1)[:-1] for i in range(len(x)-1)] + [np.asarray(x[-1:])])
    tt, pp = refined(np.asarray(times_s)), refined(np.asarray(pressures_pa))
    # Endpoints included exactly; no extrapolation, replacement, or support changes.
    tmp = np.empty((len(tt), len(pressures_pa), data.shape[2]))
    for k in range(len(pressures_pa)):
        for z in range(data.shape[2]):
            tmp[:, k, z] = np.interp(tt, times_s, data[:, k, z])
    out = np.empty((len(tt), len(pp), data.shape[2]))
    for i in range(len(tt)):
        for z in range(data.shape[2]):
            out[i, :, z] = np.interp(pp, pressures_pa, tmp[i, :, z])
    return out, tt, pp

def bootstrap_summary(values, indices):
    v = np.asarray(values, dtype=np.float64)
    assert v.ndim == 1 and indices.shape[1] == len(v)
    if not np.isfinite(v).all():
        return {'n': len(v), 'finite_n': int(np.isfinite(v).sum()), 'estimate': None,
                'low': None, 'high': None, 'status': 'NA conditional quantity undefined; no draw replacement'}
    boot = v[indices].mean(axis=1)
    return {'n': len(v), 'finite_n': len(v), 'estimate': float(v.mean()),
            'low': float(np.quantile(boot, .025)), 'high': float(np.quantile(boot, .975)),
            'status': 'COMPUTED'}
