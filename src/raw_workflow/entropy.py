"""Signed entropy and ventilation diagnostics with original physical support."""
import numpy as np
CPD, RD, RV, LVO = 1005.7, 287.04, 461.5, 2.555e6
EPS=RD/RV

def saturation_vapor_pressure_pa(temperature_k):
    """Same Bolton-style es as pinned pyPI es_cc; temperature K, pressure Pa."""
    tc = np.asarray(temperature_k, dtype=float) - 273.15
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        return 611.2 * np.exp(17.67 * tc / (243.5 + tc))

def pseudoadiabatic_entropy(temperature_k, mixing_ratio_kgkg, pressure_pa):
    """TE main Eq. (3); mixing ratio is NOT specific humidity.

    Returns NaN for missing/invalid states; r=0 is the exact dry-air limit.
    Supersaturation is retained, with no artificial RH clipping.
    Additive entropy reference constant cancels only with consistent units.
    """
    t, r, p = np.broadcast_arrays(np.asarray(temperature_k, float), np.asarray(mixing_ratio_kgkg, float), np.asarray(pressure_pa, float))
    es = saturation_vapor_pressure_pa(t)
    with np.errstate(divide='ignore', invalid='ignore'):
        e = p * r / (EPS + r)
        rh = e / es
        vapor_log = np.where(r == 0, 0.0, r * np.log(rh))
        s = CPD * np.log(t) - RD * np.log(p - e) + LVO * r / t - RV * vapor_log
    valid = np.isfinite(t) & np.isfinite(r) & np.isfinite(p) & (t > 100) & (t < 400) & (r >= 0) & (p > e) & (p > 0) & np.isfinite(es) & (es > 0)
    return np.where(valid, s, np.nan)

def entropy600_fields(T_k, q_kgkg, sp_pa):
    """600-hPa actual/saturated entropy fields; call before spatial averaging."""
    t, q, sp = np.broadcast_arrays(np.asarray(T_k, float), np.asarray(q_kgkg, float), np.asarray(sp_pa, float))
    p = 60000.0
    valid_t_sp = np.isfinite(t) & np.isfinite(sp) & (sp >= p)
    valid_q = np.isfinite(q) & (q >= 0) & (q < 0.1)
    es = saturation_vapor_pressure_pa(t)
    with np.errstate(divide='ignore', invalid='ignore'):
        r = q / (1 - q)
        rs = EPS * es / (p - es)
    s = pseudoadiabatic_entropy(t, r, p)
    ss = pseudoadiabatic_entropy(t, rs, p)
    valid_sat = valid_t_sp & np.isfinite(ss) & (es < p)
    valid_actual = valid_t_sp & valid_q & np.isfinite(s)
    return dict(entropy_jkgk=np.where(valid_actual, s, np.nan), saturated_entropy_jkgk=np.where(valid_sat, ss, np.nan), valid_actual_mask=valid_actual, valid_saturated_mask=valid_sat, valid_mask=valid_actual & valid_sat, pressure_hpa=600.0)

def assemble_ventilation(shear_200_850_ms, core600_saturated_entropy_jkgk, environment600_entropy_jkgk, pi_result, *, support_confirmed, support_metadata=None):
    """Tang-Emanuel algebra, event-level spatial definitions supplied explicitly.

    Raw signed values are retained if the entropy deficit is negative, but such a
    state is not interpreted as adverse ventilation. No clipping or zero filling.
    """
    out = dict(ventilation_index=np.nan, raw_signed_index=np.nan, chi600=np.nan, entropy_deficit_jkgk=np.nan, sea_air_entropy_difference_jkgk=np.nan, quality_status='INVALID_COMPONENT', valid=False, support_metadata=support_metadata or {}, definition='shear_200_850 * (mean_saturated_entropy_core0_100 - mean_entropy_environment100_300) / PI_RMW_sea_air_entropy_difference / PI')
    if support_confirmed is not True:
        return out
    sst_bl = pi_result.get('sea_air_entropy_difference_jkgk', np.nan)
    pi = pi_result.get('vmax_ms', np.nan)
    vals = np.array([shear_200_850_ms, core600_saturated_entropy_jkgk, environment600_entropy_jkgk, sst_bl, pi], float)
    if not pi_result.get('pi_valid', False) or not np.all(np.isfinite(vals)) or shear_200_850_ms < 0 or (sst_bl <= 0) or (pi <= 0):
        return out
    deficit = core600_saturated_entropy_jkgk - environment600_entropy_jkgk
    chi = deficit / sst_bl
    value = shear_200_850_ms * chi / pi
    out.update(entropy_deficit_jkgk=float(deficit), sea_air_entropy_difference_jkgk=float(sst_bl), chi600=float(chi), raw_signed_index=float(value))
    if deficit < 0:
        out['quality_status'] = 'NEGATIVE_ENTROPY_DEFICIT_SIGNED_DIAGNOSTIC'
        return out
    out.update(ventilation_index=float(value), quality_status='PASS', valid=True)
    return out
