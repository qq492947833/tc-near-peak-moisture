"""FLTCCF PI/ventilation diagnostics; no ERA5 I/O or fitting.

Pinned tcpyPI v1.4.2, commit 872dcf42667818fe889bd3fb5a7952076be666e5,
is vendored unchanged under vendor/tcpyPI with its MIT license. The diagnostic
extension adds return values only: it does not change PI's original five outputs.
"""
from __future__ import annotations
import ast
import hashlib
import importlib
from pathlib import Path
import numpy as np

UPSTREAM_COMMIT = "872dcf42667818fe889bd3fb5a7952076be666e5"
UPSTREAM_SHA256 = "34c823867b14f82503c3bb868e7ee19d008d927c219fb9514c600757a3bed022"
DEFAULT_CONFIG = dict(CKCD=0.9, ascent_flag=1, diss_flag=1, V_reduc=0.8,
                      ptop=50.0, miss_handle=1, outflow_source="cape_star")
# Tang & Emanuel (2012), main Eq. (3): pseudoadiabatic entropy constants.
CPD, RD, RV, LVO = 1005.7, 287.04, 461.5, 2.555e6
EPS = RD / RV
_REFERENCE = importlib.import_module("vendor.tcpyPI.pi")

def _build_diagnostic_function():
    source_path = Path(_REFERENCE.__file__)
    raw = source_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != UPSTREAM_SHA256:
        raise RuntimeError("Pinned upstream PI identity changed; refuse generated diagnostic.")
    tree = ast.parse(raw.decode("utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_pi_numba")
    fn.name = "_pi_with_rmw_numba"
    extra = ("rmw_pressure_used", "rmw_temperature", "rmw_mixing_ratio",
             "rmw_saturated_mixing_ratio", "rmw_pressure_next", "rmw_iterations")
    # Initialize diagnostics before any early return. Existing calculations unchanged.
    init = ast.parse("\n".join(x + " = np.nan" for x in extra)).body
    fn.body[1:1] = init
    class Extend(ast.NodeTransformer):
        def visit_Return(self, node):
            assert isinstance(node.value, ast.Tuple) and len(node.value.elts) == 5
            node.value.elts.extend(ast.Name(id=n, ctx=ast.Load()) for n in extra)
            return node
        def visit_Assign(self, node):
            target_names = {x.id for x in ast.walk(node.targets[0]) if isinstance(x, ast.Name)}
            if "CAPEM" in target_names:
                captures = ast.parse("rmw_pressure_used=PP\nrmw_temperature=TP\nrmw_mixing_ratio=RP").body
                return captures + [node]
            if "RS0" in target_names:
                return [node] + ast.parse("rmw_saturated_mixing_ratio=RP").body
            if "PM" in target_names and isinstance(node.value, ast.Name) and node.value.id == "PNEW":
                return [node] + ast.parse("rmw_pressure_next=PM").body
            return node
        def visit_AugAssign(self,node):
            if isinstance(node.target, ast.Name) and node.target.id == "NP":
                return [node] + ast.parse("rmw_iterations=NP").body
            return node
    fn = Extend().visit(fn)
    generated = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    namespace = dict(vars(_REFERENCE))
    exec(compile(generated, str(source_path) + "::rmw_return_extension", "exec"), namespace)
    return namespace[fn.name]

pi_reference = _REFERENCE.pi
pi_diagnostic_raw = _build_diagnostic_function()

def _config(config):
    result = dict(DEFAULT_CONFIG)
    if config:
        unknown = set(config) - set(result)
        if unknown:
            raise ValueError("Unknown PI configuration fields: " + repr(sorted(unknown)))
        result.update(config)
    if result["miss_handle"] != 1:
        raise ValueError("v5 never permits ignoring missing required profile data")
    if result["outflow_source"] not in ("cape_star", "cape_env"):
        raise ValueError("Unknown outflow_source")
    return result

def _failure(reason, estimand, metadata=None):
    return dict(vmax_ms=np.nan, pmin_hpa=np.nan, iflag=None, outflow_temperature_k=np.nan,
                outflow_pressure_hpa=np.nan, pi_valid=False, quality_status=reason,
                estimand=estimand, rmw_pressure_used_hpa=np.nan, rmw_pressure_next_hpa=np.nan,
                rmw_temperature_k=np.nan, rmw_mixing_ratio_kgkg=np.nan,
                rmw_saturated_mixing_ratio_kgkg=np.nan, rmw_iterations=np.nan,
                entropy_bl_jkgk=np.nan, entropy_sst_sat_jkgk=np.nan,
                sea_air_entropy_difference_jkgk=np.nan, metadata=metadata or {})

def saturation_vapor_pressure_pa(temperature_k):
    """Same Bolton-style es as pinned pyPI es_cc; temperature K, pressure Pa."""
    tc = np.asarray(temperature_k, dtype=float) - 273.15
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return 611.2 * np.exp(17.67 * tc / (243.5 + tc))

def pseudoadiabatic_entropy(temperature_k, mixing_ratio_kgkg, pressure_pa):
    """TE main Eq. (3); mixing ratio is NOT specific humidity.

    Returns NaN for missing/invalid states; r=0 is the exact dry-air limit.
    Supersaturation is retained, with no artificial RH clipping.
    Additive entropy reference constant cancels only with consistent units.
    """
    t, r, p = np.broadcast_arrays(np.asarray(temperature_k,float),
        np.asarray(mixing_ratio_kgkg,float), np.asarray(pressure_pa,float))
    es = saturation_vapor_pressure_pa(t)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = p * r / (EPS + r)
        rh = e / es
        vapor_log = np.where(r == 0, 0.0, r * np.log(rh))
        s = CPD * np.log(t) - RD * np.log(p-e) + LVO*r/t - RV*vapor_log
    valid = np.isfinite(t)&np.isfinite(r)&np.isfinite(p)&(t>100)&(t<400)&(r>=0)&(p>e)&(p>0)&np.isfinite(es)&(es>0)
    return np.where(valid,s,np.nan)

def _solve(t, q, p, sst, msl, cfg, estimand, metadata):
    r_gkg = 1000.0 * q / (1.0-q)
    args = dict(cfg)
    args["outflow_source_flag"] = 0 if args.pop("outflow_source") == "cape_star" else 1
    out = pi_diagnostic_raw(float(sst-273.15),float(msl/100),
        p.copy(),(t-273.15).copy(),r_gkg.copy(),**args)
    result = _failure("UPSTREAM_FAILURE", estimand, metadata)
    original_keys = ("vmax_ms","pmin_hpa","iflag","outflow_temperature_k","outflow_pressure_hpa")
    result.update(dict(zip(original_keys,out[:5])))
    result["iflag"] = int(out[2])
    result["raw_original_outputs"] = tuple(out[:5])
    result["config"] = cfg
    result["upstream_commit"] = UPSTREAM_COMMIT
    if out[2] != 1 or not np.all(np.isfinite(out[:5])) or out[0] <= 0:
        result["quality_status"] = "UPSTREAM_IFL_"+str(out[2]) if out[2]!=1 else "NONPOSITIVE_OR_NONFINITE_PI"
        # Preserve original outputs in raw record. Do not fabricate physical outputs on failure.
        for name in ("vmax_ms","pmin_hpa","outflow_temperature_k","outflow_pressure_hpa"):
            result[name]=np.nan
        return result
    extra_keys=("rmw_pressure_used_hpa","rmw_temperature_k","rmw_mixing_ratio_kgkg",
        "rmw_saturated_mixing_ratio_kgkg","rmw_pressure_next_hpa","rmw_iterations")
    result.update(dict(zip(extra_keys,out[5:])))
    pp,tb,rb,rs,_,_=out[5:]
    sb = float(pseudoadiabatic_entropy(tb,rb,pp*100))
    ss = float(pseudoadiabatic_entropy(sst,rs,pp*100))
    result.update(pi_valid=True,quality_status="PASS",entropy_bl_jkgk=sb,
        entropy_sst_sat_jkgk=ss, sea_air_entropy_difference_jkgk=ss-sb)
    if not np.isfinite(ss-sb) or ss-sb <= 0:
        result["quality_status"]="PI_VALID_ENTROPY_DIFFERENCE_INVALID"
    return result

def _validate_arrays(T_k,q_kgkg,pressure_hpa):
    t,q,p=(np.asarray(x,dtype=float) for x in (T_k,q_kgkg,pressure_hpa))
    if t.ndim!=1 or q.shape!=t.shape or p.shape!=t.shape or len(p)<3:
        return None,"PROFILE_SHAPE"
    if not np.all(np.isfinite(p)&(p>0)) or not np.all(np.diff(p)<0):
        return None,"PRESSURE_NOT_STRICTLY_DESCENDING_HPA"
    return (t,q,p),None

def _prepared(t,q,p,sst,msl,keep,cfg,estimand,metadata):
    # Only a contiguous lower prefix may be removed. Internal holes remain invalid.
    relevant = p >= cfg["ptop"]
    if not np.any(p<=cfg["ptop"]):
        return _failure("PROFILE_DOES_NOT_REACH_PTOP",estimand,metadata)
    indices=np.flatnonzero(keep & relevant)
    if len(indices)<3:
        return _failure("TOO_FEW_VALID_LEVELS",estimand,metadata)
    if not np.array_equal(indices,np.arange(indices[0],np.flatnonzero(relevant)[-1]+1)):
        return _failure("INTERNAL_PROFILE_SUPPORT_GAP",estimand,metadata)
    use=indices
    if p[use[-1]]>cfg["ptop"]:
        return _failure("PTOP_SUPPORT_MISSING",estimand,metadata)
    t,q,p=t[use],q[use],p[use]
    metadata.update(used_pressure_hpa=p.tolist(),lowest_valid_pressure_hpa=float(p[0]),
                    top_valid_pressure_hpa=float(p[-1]),valid_level_count=len(p))
    if not np.all(np.isfinite(t)&(t>100)&(t<400)&np.isfinite(q)&(q>=0)&(q<0.1)):
        return _failure("MISSING_OR_INVALID_REQUIRED_T_Q",estimand,metadata)
    if not np.isfinite(sst) or not 100<sst<400 or not np.isfinite(msl) or not 20000<msl<120000:
        return _failure("INVALID_SST_K_OR_MSL_PA",estimand,metadata)
    return _solve(t,q,p,float(sst),float(msl),cfg,estimand,metadata)

def pi_profile(T_k,q_kgkg,pressure_hpa,sst_k,msl_pa,sp_pa,config=None):
    """One actual gridpoint profile. Explicit same-grid, same-UTC SP is required."""
    cfg=_config(config);est="PI_OF_ONE_GRIDPOINT_PROFILE"
    arrays,error=_validate_arrays(T_k,q_kgkg,pressure_hpa)
    if error:return _failure(error,est)
    t,q,p=arrays
    if not np.isfinite(sp_pa) or not 20000<sp_pa<120000:
        return _failure("INVALID_SAME_GRID_SP_PA",est)
    keep=p*100<=sp_pa
    meta=dict(pressure_hpa=p.tolist(),above_ground_mask=keep.tolist(),
              removed_below_ground_count=int((~keep).sum()),sp_pa=float(sp_pa),
              profile_origin="one_gridpoint_same_UTC",sp_qc_confirmed=True)
    result=_prepared(t,q,p,sst_k,msl_pa,keep,cfg,est,meta)
    if "lowest_valid_pressure_hpa" in meta:
        meta["surface_pressure_gap_hpa"]=float(sp_pa/100-meta["lowest_valid_pressure_hpa"])
    return result

def pi_environment_profile(T_k,q_kgkg,pressure_hpa,sst_k,msl_pa,
        valid_area_fraction_by_level,*,sp_qc_confirmed,original_area_denominator_confirmed,
        surface_support_confirmed,minimum_area_fraction=0.9,support_metadata=None,config=None):
    """PI of area-mean environmental profile; NOT mean of gridded PI.

    Caller must apply same-grid/time SP and original-denominator QC before averaging.
    Support must refer to joint T/q support for each layer. No average SP is used.
    The SST/ocean-area rule is confirmed by caller and retained in support_metadata.
    """
    cfg=_config(config);est="PI_OF_ENVIRONMENTAL_AREA_MEAN_PROFILE"
    arrays,error=_validate_arrays(T_k,q_kgkg,pressure_hpa)
    if error:return _failure(error,est)
    # Reject 'False' strings, which Python otherwise treats as true.
    if sp_qc_confirmed is not True or original_area_denominator_confirmed is not True or surface_support_confirmed is not True:
        return _failure("CALLER_SUPPORT_NOT_EXPLICITLY_CONFIRMED",est)
    t,q,p=arrays; support=np.asarray(valid_area_fraction_by_level,dtype=float)
    if support.shape!=p.shape or not 0<minimum_area_fraction<=1:
        return _failure("INVALID_LAYER_SUPPORT_METADATA",est)
    if np.any(np.isfinite(support)&((support<0)|(support>1+1e-12))):
        return _failure("INVALID_LAYER_SUPPORT_RANGE",est)
    keep=np.isfinite(support)&(support>=minimum_area_fraction)
    meta=dict(support_metadata or {})
    meta.update(pressure_hpa=p.tolist(),valid_area_fraction_by_level=support.tolist(),
        kept_layer_mask=keep.tolist(),minimum_area_fraction=minimum_area_fraction,
        profile_origin="200-800_km_environmental_area_mean",sp_qc_confirmed=True,
        original_area_denominator_confirmed=True,surface_support_confirmed=True,
        surface_pressure_gap_hpa=None,
        surface_gap_note="No single surface pressure exists for an area-mean profile; supply actual gridpoint gap summary in support_metadata")
    return _prepared(t,q,p,sst_k,msl_pa,keep,cfg,est,meta)

def entropy600_fields(T_k,q_kgkg,sp_pa):
    """600-hPa actual/saturated entropy fields; call before spatial averaging."""
    t,q,sp=np.broadcast_arrays(np.asarray(T_k,float),np.asarray(q_kgkg,float),np.asarray(sp_pa,float))
    p=60000.0
    valid_t_sp=np.isfinite(t)&np.isfinite(sp)&(sp>=p)
    valid_q=np.isfinite(q)&(q>=0)&(q<0.1)
    es=saturation_vapor_pressure_pa(t)
    with np.errstate(divide="ignore",invalid="ignore"):
        r=q/(1-q);rs=EPS*es/(p-es)
    s=pseudoadiabatic_entropy(t,r,p);ss=pseudoadiabatic_entropy(t,rs,p)
    valid_sat=valid_t_sp&np.isfinite(ss)&(es<p)
    valid_actual=valid_t_sp&valid_q&np.isfinite(s)
    # Saturated core entropy requires T/SP, not a measured q. Keep separate support.
    return dict(entropy_jkgk=np.where(valid_actual,s,np.nan),
                saturated_entropy_jkgk=np.where(valid_sat,ss,np.nan),
                valid_actual_mask=valid_actual,valid_saturated_mask=valid_sat,
                valid_mask=valid_actual&valid_sat,pressure_hpa=600.0)

def assemble_ventilation(shear_200_850_ms,core600_saturated_entropy_jkgk,
        environment600_entropy_jkgk,pi_result,*,support_confirmed,support_metadata=None):
    """Tang-Emanuel algebra, event-level spatial definitions supplied explicitly.

    Raw signed values are retained if the entropy deficit is negative, but such a
    state is not interpreted as adverse ventilation. No clipping or zero filling.
    """
    out=dict(ventilation_index=np.nan,raw_signed_index=np.nan,chi600=np.nan,
        entropy_deficit_jkgk=np.nan,sea_air_entropy_difference_jkgk=np.nan,
        quality_status="INVALID_COMPONENT",valid=False,support_metadata=support_metadata or {},
        definition="shear_200_850 * (mean_saturated_entropy_core0_100 - mean_entropy_environment100_300) / PI_RMW_sea_air_entropy_difference / PI")
    if support_confirmed is not True:return out
    sst_bl=pi_result.get("sea_air_entropy_difference_jkgk",np.nan)
    pi=pi_result.get("vmax_ms",np.nan)
    vals=np.array([shear_200_850_ms,core600_saturated_entropy_jkgk,environment600_entropy_jkgk,sst_bl,pi],float)
    if not pi_result.get("pi_valid",False) or not np.all(np.isfinite(vals)) or shear_200_850_ms<0 or sst_bl<=0 or pi<=0:return out
    deficit=core600_saturated_entropy_jkgk-environment600_entropy_jkgk
    chi=deficit/sst_bl;value=shear_200_850_ms*chi/pi
    out.update(entropy_deficit_jkgk=float(deficit),sea_air_entropy_difference_jkgk=float(sst_bl),
               chi600=float(chi),raw_signed_index=float(value))
    if deficit<0:
        out["quality_status"]="NEGATIVE_ENTROPY_DEFICIT_SIGNED_DIAGNOSTIC"
        return out
    out.update(ventilation_index=float(value),quality_status="PASS",valid=True)
    return out
