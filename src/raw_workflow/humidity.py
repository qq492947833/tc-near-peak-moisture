"""Four-combination RH decomposition on fixed support at two moving domains.

Pressure is Pa; q is specific humidity in kg/kg; temperature is K. Each domain
uses one original-cell area mask valid at both times. The symmetric domain term
is separate from support change. No interpolated time points are introduced.
"""
import numpy as np

def rh(q, t, p):
    eps = 287.0597 / 461.525
    alpha = np.clip((t - 250.16) / (273.16 - 250.16), 0, 1) ** 2
    es = alpha * 611.21 * np.exp(17.502 * (t - 273.16) / (t - 32.19)) + (1 - alpha) * 611.21 * np.exp(22.587 * (t - 273.16) / (t + 0.7))
    return 100 * p * q / (eps + (1 - eps) * q) / es

def decomp(q0, t0, q1, t1, p):
    a = rh(q0, t0, p)
    b = rh(q1, t1, p)
    c = rh(q1, t0, p)
    d = rh(q0, t1, p)
    return (b - a, 0.5 * (c - a + (b - d)), 0.5 * (d - a + (b - c)))

def pair(row, cache):
    allgeom = []
    out = []
    for gi in [0, 1]:
        bb = []
        for ti in [0, 1]:
            utc = row['UTC' + str(ti)]
            lat = row['lat' + str(gi)]
            lon = row['lon' + str(gi)]
            b = cache.reference(utc, lat, lon, 800.0)
            a = {k: cache.get(k, utc, lat, lon).values for k in ['q500', 'q700', 't500', 't700']}
            bb.append((b, a))
        assert np.array_equal(bb[0][0].latitude_indices, bb[1][0].latitude_indices) and np.array_equal(bb[0][0].longitude_indices, bb[1][0].longitude_indices)
        allgeom.append(bb)
    for level in [500, 700]:
        fixed = []
        full = []
        temporalq = []
        temporalt = []
        support = []
        for gi, bb in enumerate(allgeom):
            b = bb[0][0]
            area = b.cell_area_m2
            dom = (b.distance_km >= 200) & (b.distance_km < 800)
            a, c = (bb[0][1], bb[1][1])
            q0, t0, q1, t1 = (a['q' + str(level)], a['t' + str(level)], c['q' + str(level)], c['t' + str(level)])
            m = dom & np.isfinite(q0) & np.isfinite(t0) & np.isfinite(q1) & np.isfinite(t1)
            frac = float(area[m].sum() / area[dom].sum())
            support.append(frac)

            def mean(x, mask):
                return float(np.sum(x[mask] * area[mask]) / area[mask].sum()) if mask.any() else np.nan
            f0 = rh(q0, t0, level * 100)
            f1 = rh(q1, t1, level * 100)
            fixed.append([mean(f0, m), mean(f1, m)] if frac >= 0.9 else [np.nan, np.nan])
            qcontr, tcontr = decomp(q0, t0, q1, t1, level * 100)[1:]
            temporalq.append(mean(qcontr, m) if frac >= 0.9 else np.nan)
            temporalt.append(mean(tcontr, m) if frac >= 0.9 else np.nan)
            original = []
            for f in [f0, f1]:
                own = dom & np.isfinite(f)
                original.append(mean(f, own) if area[own].sum() / area[dom].sum() >= 0.9 else np.nan)
            full.append(original)
        f00, f01 = fixed[0]
        f10, f11 = fixed[1]
        temporal = 0.5 * (f01 - f00 + (f11 - f10))
        spatial = 0.5 * (f10 - f00 + (f11 - f01))
        total = f11 - f00
        original = full[1][1] - full[0][0]
        out.append(dict(sid=row['sid'], boundary=row['boundary'], level_hpa=level, units='percentage points', G0_t0=f00, G0_t1=f01, G1_t0=f10, G1_t1=f11, temporal_symmetric=temporal, domain_symmetric=spatial, time_q_contribution=float(np.mean(temporalq)), time_T_contribution=float(np.mean(temporalt)), fixed_support_endpoint_change=total, original_support_endpoint_change=original, support_change=original - total, G0_joint_fraction=support[0], G1_joint_fraction=support[1], decomposition_identity=total - temporal - spatial))
    return out
