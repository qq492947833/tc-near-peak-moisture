"""Moving-node 700-hPa endpoint and tendency calculations.

The 32 x 64 equal-area nodes are spatial quadrature, not new observations. The
original source remains 6-hourly. A common finite mask across both events and
all five times must be applied before temporal integration (see integrate_pair).
"""
import numpy as np
import pandas as pd
from .native_grid import spherical_gradient, vertical_advection_700
R=6371008.8

def nodes(lat, lon, inner, outer):
    c0, c1 = (np.cos(inner * 1000 / R), np.cos(outer * 1000 / R))
    dist = np.arccos(c0 + (np.arange(32) + 0.5) / 32 * (c1 - c0))
    theta = 2 * np.pi * (np.arange(64) + 0.5) / 64
    d, t = np.meshgrid(dist, theta, indexing='ij')
    phi = np.deg2rad(lat)
    lam = np.deg2rad(lon)
    outlat = np.arcsin(np.sin(phi) * np.cos(d) + np.cos(phi) * np.sin(d) * np.cos(t))
    outlon = lam + np.arctan2(np.sin(t) * np.sin(d) * np.cos(phi), np.cos(d) - np.sin(phi) * np.sin(outlat))
    return (np.rad2deg(outlat), np.rad2deg(outlon))

def interp(a, b, xy):
    la, lo = xy
    anchor = 0.5 * (b.longitude[0] + b.longitude[-1])
    lo = anchor + (lo - anchor + 180) % 360 - 180
    y = (la - b.latitude[0]) / (b.latitude[1] - b.latitude[0])
    x = (lo - b.longitude[0]) / (b.longitude[1] - b.longitude[0])
    iy = np.floor(y).astype(int)
    ix = np.floor(x).astype(int)
    fy = y - iy
    fx = x - ix
    valid = (iy >= 0) & (iy < a.shape[0] - 1) & (ix >= 0) & (ix < a.shape[1] - 1)
    ii = np.clip(iy, 0, a.shape[0] - 2)
    jj = np.clip(ix, 0, a.shape[1] - 2)
    a00, a01, a10, a11 = (a[ii, jj], a[ii, jj + 1], a[ii + 1, jj], a[ii + 1, jj + 1])
    valid &= np.isfinite(a00) & np.isfinite(a01) & np.isfinite(a10) & np.isfinite(a11)
    return np.where(valid, (1 - fy) * ((1 - fx) * a00 + fx * a01) + fy * ((1 - fx) * a10 + fx * a11), np.nan)

def calculate(row, cache, prior_position):
    utc = row['UTC']
    lat = row['lat']
    lon = row['lon']
    b = cache.reference(utc, lat, lon, 800.0)
    a = {k: cache.get(k, utc, lat, lon).values for k in ['q700', 'q600', 'q850', 'w700', 'u700', 'v700']}
    q = a['q700']
    gx, gy = spherical_gradient(q, b.latitude, b.longitude)
    minus = cache.get('q700', pd.Timestamp(utc) - pd.Timedelta(hours=6), lat, lon).values
    plus = cache.get('q700', pd.Timestamp(utc) + pd.Timedelta(hours=6), lat, lon).values
    dt = (plus - minus) / 43200.0
    alt = (plus - q) / 21600.0 if row['lag_h'] == 0 else (q - minus) / 21600.0 if row['lag_h'] == 24 else dt
    vert = vertical_advection_700(a['q600'], q, a['q850'], a['w700'])
    sec = -a['w700'] * (a['q850'] - a['q600']) / 25000.0
    fields = dict(q=q * 1000, dt=dt * 86400000, alt_dt=alt * 86400000, gx=gx, gy=gy, u=a['u700'], v=a['v700'], vertical3=vert * 86400000, vertical_secant=sec * 86400000)
    out = {}
    for name, lo, hi in [('inner', 100, 300), ('outer', 300, 800)]:
        xy = nodes(lat, lon, lo, hi)
        prev = nodes(*prior_position, lo, hi)
        cnu = R * np.cos(np.deg2rad((xy[0] + prev[0]) / 2)) * np.deg2rad((xy[1] - prev[1] + 180) % 360 - 180) / 21600.0
        cnv = R * np.deg2rad(xy[0] - prev[0]) / 21600.0
        f = {k: interp(v, b, xy) for k, v in fields.items()}
        motion_center = (row['motion_u_ms'] * f['gx'] + row['motion_v_ms'] * f['gy']) * 86400000.0
        motion_node = (cnu * f['gx'] + cnv * f['gy']) * 86400000.0
        f.update(moving_center=f['dt'] + motion_center, moving_nodes=f['dt'] + motion_node, moving_nodes_endpointFD=f['alt_dt'] + motion_node, horizontal_nodes=(-(f['u'] - cnu) * f['gx'] - (f['v'] - cnv) * f['gy']) * 86400000.0)
        f['residual_nodes'] = f['moving_nodes'] - f['horizontal_nodes'] - f['vertical3']
        out[name] = {k: v for k, v in f.items() if k in ['q', 'moving_center', 'moving_nodes', 'moving_nodes_endpointFD', 'horizontal_nodes', 'vertical3', 'vertical_secant', 'residual_nodes']}
    return out


def integrate_pair(events,sid):
    """Apply one finite node mask to two events x five times x every field.

    events is [temporary_times, final_times]; each sequence contains outputs of
    calculate at 0,6,12,18,24 h, in that order. Values are integrated in days;
    q is in g/kg and all tendencies in g/kg/day. No missing times are filled.
    """
    if len(events)!=2 or any(len(event)!=5 for event in events):
        raise ValueError('Two events and all five actual times are required')
    result=[]
    for radial in ['inner','outer']:
        fields=list(events[0][0][radial])
        arrays={k:np.array([[event[t][radial][k] for t in range(5)] for event in events]) for k in fields}
        if any(a.shape!=(2,5,32,64) for a in arrays.values()):
            raise ValueError('The fixed 32 x 64 node geometry is required')
        common=np.isfinite(np.stack(list(arrays.values()))).all((0,1,2))
        fraction=float(common.mean())
        for side,event_class in enumerate(['internal','final']):
            output=dict(sid=sid,radial=radial,event_class=event_class,
                        joint_fraction=fraction,support_pass=fraction>=.9,
                        n_original_nodes=2048,n_joint_nodes=int(common.sum()),units='g kg-1')
            for k in fields:
                means=np.array([np.mean(arrays[k][side,t][common]) for t in range(5)]) if fraction>=.9 else np.full(5,np.nan)
                output['delta_q' if k=='q' else 'integral_'+k]=float(means[-1]-means[0]) if k=='q' else float(np.trapezoid(means,dx=.25))
            output['endpoint_minus_integrated_nodes']=output['delta_q']-output['integral_moving_nodes']
            output['endpoint_minus_integrated_center']=output['delta_q']-output['integral_moving_center']
            output['endpoint_minus_endpointFD_nodes']=output['delta_q']-output['integral_moving_nodes_endpointFD']
            output['vertical_gradient_difference']=output['integral_vertical_secant']-output['integral_vertical3']
            result.append(output)
    return result
