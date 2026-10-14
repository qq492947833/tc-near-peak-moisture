"""Joint-support profile preparation retaining every original native level."""
import numpy as np
NATIVE_LEVELS=np.array([1000,925,850,700,600,500,400,300,250,200,150,100,50])
P13=NATIVE_LEVELS
P14=np.insert(P13,12,70)

def prepare13(original):
    """Retain original arrays; never remove a missing lower prefix."""
    p = np.asarray(original['pressure_hpa'])
    t = np.asarray(original['temperature_K'])
    q = np.asarray(original['specific_humidity_kgkg'])
    f = np.asarray(original['joint_Tq_valid_fraction'])
    if not np.array_equal(p, NATIVE_LEVELS) or t.shape != (13,) or q.shape != (13,) or (f.shape != (13,)):
        raise ValueError('original fixed13 shape/pressure mismatch')
    if np.any(np.isfinite(f) & ((f < 0) | (f > 1 + 1e-12))):
        raise ValueError('invalid area fraction')
    good = np.isfinite(t) & (t > 100) & (t < 400) & np.isfinite(q) & (q >= 0) & (q < 0.1) & np.isfinite(f) & (f >= 0.9)
    return dict(pressure_hpa=p.copy(), temperature_K=np.where(good, t, np.nan), specific_humidity_kgkg=np.where(good, q, np.nan), joint_Tq_valid_fraction=f.copy(), valid_mask=good, valid=bool(good.all()), unsupported_levels_hpa=p[~good].tolist())

def add70(prepared, t70=np.nan, q70=np.nan, f70=np.nan):
    return dict(pressure_hpa=np.insert(prepared['pressure_hpa'], 12, 70), temperature_K=np.insert(prepared['temperature_K'], 12, t70), specific_humidity_kgkg=np.insert(prepared['specific_humidity_kgkg'], 12, q70), joint_Tq_valid_fraction=np.insert(prepared['joint_Tq_valid_fraction'], 12, f70))


def augment_from_native(original,t100,t50,q100,q50,cell_area_m2,original_annulus):
    """Derive 70 hPa on joint native cells before area averaging.

    T/q endpoint fields must already use exact shared coordinates and same-cell
    surface-pressure masks. original_annulus is the full 200 <= r < 800 km
    denominator; missing cells cannot be removed from that denominator.
    """
    from .pi_interpolation import joint70
    from .native_grid import area_weighted_support
    prepared=prepare13(original)
    result={k:np.asarray(v).copy() for k,v in original.items()}
    for method,label in [('logp','logp'),('linear_p','plinear')]:
        t,q,_=joint70(t100,t50,q100,q50,method)
        ts=area_weighted_support(t,cell_area_m2,original_annulus,min_fraction=.9)
        qs=area_weighted_support(q,cell_area_m2,original_annulus,min_fraction=.9)
        if ts['valid_area_m2']!=qs['valid_area_m2']:
            raise ValueError('Derived temperature and humidity support must coincide')
        augmented=add70(prepared,ts['mean'],qs['mean'],ts['valid_fraction'])
        result['T_'+label+'_K']=augmented['temperature_K']
        result['q_'+label+'_kgkg']=augmented['specific_humidity_kgkg']
        result['pressure_aug_hpa']=augmented['pressure_hpa']
        result['joint_fraction_aug']=augmented['joint_Tq_valid_fraction']
    return result
