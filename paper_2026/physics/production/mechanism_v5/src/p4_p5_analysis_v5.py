"""P4/P5 fixed-source paired process analysis and independent numerical checks.

Only new v5 outputs are writable. No raw ERA5 read, trajectory rematching,
old-result replacement, H2 fit, or bootstrap success top-up is implemented here.
Scientific analysis requires an explicit complete extraction/thermo coverage gate.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, csv, hashlib, json, math, os
import numpy as np
from scipy.optimize import linprog
from v5_statistics import bootstrap_weights, weighted_fit, target_estimable, summarize_draws
from native_fields_v5 import spherical_divergence, area_weighted_support, utc_datetime

PROCESS_COLUMNS = {
 "midRH": ("mid_rh_pct", "pp"),
 "q700_storm_relative_horizontal_advection": ("q700_horizontal_storm_relative_gkg_day", "g kg-1 day-1"),
 "q850_moisture_flux_convergence": ("q850_moisture_flux_convergence_gkg_day", "g kg-1 day-1"),
 "shear200_500": ("shear_200_500_vector_magnitude_ms", "m s-1"),
 "PI_headroom": ("PI_headroom_ms", "m s-1"),
}
BOOTSTRAP_REPEATS = 2000
BOOTSTRAP_SEED = 510082026

def stamp():
 return datetime.now(timezone.utc).isoformat()

def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 return h.hexdigest()

def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,float) and not math.isfinite(x):return None
 return x

def canonical(x):
 return json.dumps(clean(x),sort_keys=True,separators=(",",":"),ensure_ascii=False)

def digest(x):
 return hashlib.sha256(canonical(x).encode()).hexdigest()

def atomic_json(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_name(path.name+f".tmp.{os.getpid()}")
 tmp.write_text(json.dumps(clean(obj),ensure_ascii=False,indent=2),encoding="utf-8")
 os.replace(tmp,path)

def explicit_bool(value):
 if isinstance(value,(bool,np.bool_)):return bool(value)
 if isinstance(value,(int,np.integer)) and value in (0,1):return bool(value)
 s=str(value).strip().lower()
 if s in ("true","1"):return True
 if s in ("false","0"):return False
 raise ValueError(f"not an explicit Boolean: {value!r}")

def num(value):
 try:
  v=float(value)
  return v if np.isfinite(v) else np.nan
 except (ValueError,TypeError):return np.nan

def bh_fixed_family(p,nominal_m=None):
 """Unavailable target occupies its prespecified family slot; never reduces m."""
 p=np.asarray(p,float);m=len(p) if nominal_m is None else int(nominal_m)
 if m<len(p):raise ValueError("family smaller than supplied prespecified targets")
 if np.any(np.isfinite(p)&((p<0)|(p>1))):raise ValueError("invalid P")
 vals=np.ones(m);vals[:len(p)]=np.where(np.isfinite(p),p,1.)
 order=np.argsort(vals,kind="stable")
 qsort=np.minimum(1.,np.minimum.accumulate((vals[order]*m/np.arange(1,m+1))[::-1])[::-1])
 q=np.empty(m);q[order]=qsort
 q=q[:len(p)];q[~np.isfinite(p)]=np.nan
 return q

def sid_equal_weights(sid,valid=None):
 """Equal total weight per represented SID, renormalizing only its valid pairs."""
 sid=np.asarray(sid,str);valid=np.ones(len(sid),bool) if valid is None else np.asarray(valid,bool)
 out=np.zeros(len(sid))
 for s in np.unique(sid[valid]):
  ix=valid&(sid==s);out[ix]=1./ix.sum()
 return out

def cluster_mean(sid,y,draws=BOOTSTRAP_REPEATS,seed=BOOTSTRAP_SEED):
 """SID mean of within-SID valid-pair mean, plus exact sampled-SID draws."""
 sid=np.asarray(sid,str);y=np.asarray(y,float)
 unique=np.unique(sid[np.isfinite(y)])
 per=np.array([np.mean(y[(sid==s)&np.isfinite(y)]) for s in unique])
 if len(unique)<2:
  return dict(estimate=np.nan,draws=np.full(draws,np.nan),sid=unique,per_sid=per,
              sampled_sid_indices=np.empty((0,0),int),n_sid=len(unique),n_pairs=int(np.isfinite(y).sum()))
 indices,w=bootstrap_weights(len(unique),draws,seed)
 out=per[indices].mean(axis=1)
 return dict(estimate=float(per.mean()),draws=out,sid=unique,per_sid=per,
             sampled_sid_indices=indices,n_sid=len(unique),n_pairs=int(np.isfinite(y).sum()))

def model_target_draws(sid,X,y,contrast,draws=BOOTSTRAP_REPEATS,seed=BOOTSTRAP_SEED):
 """Fixed target under SID multiplicity weights. Failed targets remain NaN."""
 sid=np.asarray(sid,str);X=np.asarray(X,float);y=np.asarray(y,float);c=np.asarray(contrast,float)
 valid=np.isfinite(X).all(axis=1)&np.isfinite(y)
 sid=sid[valid];X=X[valid];y=y[valid]
 if not len(y):raise ValueError("no supported model rows")
 unique,inv=np.unique(sid,return_inverse=True)
 base=sid_equal_weights(sid)
 fit=weighted_fit(X,y,base,contrast=c)
 estimate=float(c@fit["beta"]) if fit["estimable"] else np.nan
 indices,mult=bootstrap_weights(len(unique),draws,seed)
 values=np.full(draws,np.nan);records=[]
 for b in range(draws):
  w=base*mult[b,inv];keep=w>0
  f=weighted_fit(X[keep],y[keep],w[keep],contrast=c)
  if f["estimable"]:values[b]=float(c@f["beta"])
  records.append(dict(draw=b,success=bool(f["estimable"]),rank=int(f["rank"]),
    sampled_distinct_sid=int(np.count_nonzero(mult[b])),positive_weight_rows=int(keep.sum()),
    residual_row_df=int(f["df"]),failure_reason="" if f["estimable"] else "TARGET_NOT_ESTIMABLE"))
 return dict(estimate=estimate,draws=values,draw_records=records,sid=unique,
   sampled_sid_indices=indices,input_valid=valid,base_weights=base,
   fitted_beta=fit["beta"],n_sid=len(unique),n_pairs=len(y),rank=int(fit["rank"]),
   residual_row_df=int(fit["df"]),cluster_count_minus_rank=int(len(unique)-fit["rank"]),
   reference="SID percentile bootstrap; neither row nor cluster df is a t-reference")

def empirical_background_contrast(X,w):
 """Actual observed design distribution, never an invented all-zero delta state."""
 X=np.asarray(X,float);w=np.asarray(w,float)
 if not np.isfinite(X).all() or np.any(w<0) or w.sum()<=0:raise ValueError("invalid observed design")
 return np.average(X,axis=0,weights=w)

def joint_support_diagnostics(background,target,split,categories=None):
 """Outcome-blind convex-hull intersection witness, affine support and leverage.

This diagnostic does not trim/rematch SID or create a new significance family.
A feasible intersection does not establish causal positivity or full common support.
"""
 B=np.asarray(background,float);a=np.asarray(target,float)
 if B.ndim==1:B=B[:,None]
 good=np.isfinite(B).all(axis=1)&np.isfinite(a)
 B=B[good];a=a[good]
 cat=np.full(len(a),"all") if categories is None else np.asarray(categories,str)[good]
 results=[]
 for name in sorted(set(cat)):
  sel=cat==name;Z=B[sel];av=a[sel]
  left=av<split;right=av>=split
  item=dict(stratum=name,n_left=int(left.sum()),n_right=int(right.sum()),split=float(split))
  if not left.any() or not right.any():
   item.update(status="NO_BOTH_SIDES",intersection_witness=None);results.append(item);continue
  mu=Z.mean(axis=0);scale=Z.std(axis=0);scale=np.where(scale>0,scale,1.)
  Z=(Z-mu)/scale
  # Convex combinations of actual rows; no independently recombined marginal ranges.
  L=Z[left];R=Z[right]
  Aeq=np.vstack([np.column_stack([L.T,-R.T]),
   np.r_[np.ones(len(L)),np.zeros(len(R))],np.r_[np.zeros(len(L)),np.ones(len(R))]])
  beq=np.r_[np.zeros(Z.shape[1]),1.,1.]
  lp=linprog(np.zeros(len(L)+len(R)),A_eq=Aeq,b_eq=beq,bounds=(0,None),method="highs")
  item.update(status="CONVEX_HULLS_INTERSECT" if lp.success else "NO_JOINT_HULL_INTERSECTION",
    solver_status=int(lp.status),solver_message=str(lp.message),
    intersection_witness=(lp.x[:len(L)]@L*scale+mu).tolist() if lp.success else None,
    marginal_overlap_low=np.maximum(B[sel][left].min(axis=0),B[sel][right].min(axis=0)).tolist(),
    marginal_overlap_high=np.minimum(B[sel][left].max(axis=0),B[sel][right].max(axis=0)).tolist())
  D=np.column_stack([np.ones(len(Z)),av,Z]);u,s,vt=np.linalg.svd(D,full_matrices=False)
  rank=int(np.sum(s>np.finfo(float).eps*max(D.shape)*s[0]))
  item.update(design_rank=rank,design_columns=D.shape[1],
    max_observed_leverage=float(np.sum(u[:,:rank]**2,axis=1).max()),
    target_range=[float(av.min()),float(av.max())])
  results.append(item)
 return dict(n_complete_background=int(good.sum()),n_missing_background=int((~good).sum()),
  split_used_for_diagnostic_only=float(split),strata=results,
  qualification="intersection is only a joint-support witness, not proof of broad positivity; no population trimming")

def wrap_delta_longitude(lon_after,lon_before):
 return (np.asarray(lon_after,float)-np.asarray(lon_before,float)+180.)%360.-180.

def native_mfc850(q,u,v,latitude,longitude):
 """Specific-humidity flux convergence: -div(q v), kg kg-1 s-1, native sphere."""
 q=np.asarray(q,float);u=np.asarray(u,float);v=np.asarray(v,float)
 valid=np.isfinite(q)&np.isfinite(u)&np.isfinite(v)
 return -spherical_divergence(np.where(valid,q*u,np.nan),np.where(valid,q*v,np.nan),latitude,longitude)

def cache_geometry(root,spec,radius):
 obj=dict(UTC=utc_datetime(spec["UTC"]).isoformat()+"Z",latitude=float(spec["lat"]),
   longitude=float(spec["lon"])%360,radius_km=float(radius),halo=1,
   reader_sha=sha(Path(root)/"src/native_fields_v5.py"))
 return digest(obj)[:20]

def _load_cache_geometry(root,spec,radius):
 folder=Path(root)/"cache/native_fields"/cache_geometry(root,spec,radius)
 meta=json.loads((folder/"geometry.json").read_text(encoding="utf-8"))
 if sha(folder/"sp.npz")!=meta["npz_sha256"]:raise ValueError("SP cache hash mismatch")
 with np.load(folder/"sp.npz",allow_pickle=False) as z:
  d={k:z[k].copy() for k in z.files}
 if json.loads(str(d["binding_json"].item()))!=meta["binding"]:raise ValueError("SP embedded binding mismatch")
 return folder,d

def _load_cached_field(root,spec,key,radius,reference):
 folder,g=_load_cache_geometry(root,spec,radius)
 path=folder/(key+".npz");meta=json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
 if sha(path)!=meta["npz_sha256"]:raise ValueError(f"native cache hash mismatch: {key}")
 with np.load(path,allow_pickle=False) as z:
  if json.loads(str(z["binding_json"].item()))!=meta["binding"]:raise ValueError("native embedded binding mismatch")
  val=z["values"].copy();mask=z["support_mask"].astype(bool)
 if not np.array_equal(np.isfinite(val),mask):raise ValueError("cache support/value mismatch")
 iy={int(j):i for i,j in enumerate(g["latitude_indices"])}
 ix={int(j):i for i,j in enumerate(g["longitude_indices"])}
 ii=[iy[int(j)] for j in reference["latitude_indices"]]
 jj=[ix[int(j)] for j in reference["longitude_indices"]]
 return val[np.ix_(ii,jj)],dict(path=str(path),sha256=meta["npz_sha256"])

def derive_mfc_from_checkpoint(root,checkpoint,output=None):
 """Read only a COMPLETE current request and its already-created native cache."""
 root=Path(root);checkpoint=Path(checkpoint)
 done=json.loads(checkpoint.read_text(encoding="utf-8"));b=done["binding"];s=b["normalised"]
 if done.get("status")!="COMPLETE":raise ValueError("request not complete")
 if b["driver_sha"]!=sha(root/"src/extract_environment_v5.py"):raise ValueError("obsolete driver checkpoint")
 if not s["variable_set"]=="full_process":raise ValueError("MFC requires full process request")
 _,g=_load_cache_geometry(root,s,800.)
 sources=[];values={}
 for key in ("q850","u850","v850"):
  radius=1500. if s["need_map"] and key in ("u850","v850") else 800.
  values[key],source=_load_cached_field(root,s,key,radius,g);sources.append(source)
 # Same-cell SP QC already applied by authority and independently checked here.
 invalid_sp=~np.isfinite(g["sp"])|(g["sp"]<85000)
 if any(np.isfinite(v[invalid_sp]).any() for v in values.values()):raise ValueError("below-ground native cache value")
 field=native_mfc850(values["q850"],values["u850"],values["v850"],g["latitude"],g["longitude"])
 domain=(g["distance"]>=200)&(g["distance"]<800)
 stat=area_weighted_support(field,g["area"],domain,min_fraction=.9)
 answer=dict(request_id=b["request"]["request_id"],request_hash=checkpoint.parent.name,
   process="q850_moisture_flux_convergence",definition="-div(q850*(u850,v850)) on native sphere",
   units="kg kg-1 s-1",statistics=stat,
   q850_moisture_flux_convergence_gkg_day=stat["mean"]*1000*86400,
   source_checkpoint=str(checkpoint),source_checkpoint_sha256=sha(checkpoint),source_native_cache=sources,
   no_raw_ERA5_read=True,original_area_denominator=True,created_utc=stamp())
 if output is not None:atomic_json(output,answer)
 return answer

def read_rows(path):
 with Path(path).open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))

def write_rows(path,rows):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 if not rows:raise ValueError("empty output requires explicit schema")
 fields=list(dict.fromkeys(k for r in rows for k in r))
 tmp=path.with_name(path.name+f".tmp.{os.getpid()}")
 with tmp.open("w",encoding="utf-8-sig",newline="") as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 os.replace(tmp,path)

def prepare_source_design(root,out):
 """Geometry/background only; no science estimates or substitute old core values."""
 root=Path(root);out=Path(out)
 requests={r["request_id"]:r for r in read_rows(root/"data/environment_requests_v5.csv")}
 P4=read_rows(root/"p4_p5/P4_boundary_requests.csv")
 P5=read_rows(root/"p4_p5/P5_matched_intensity_pairs.csv")
 time={r["sid"]:r for r in read_rows(root/"p4_p5/P5_per_SID_time_audit.csv")}
 p4lookup={(r["sid"],r["boundary"],int(r["lag_h"])):r for r in P4}
 sidboundary=sorted(set((r["sid"],r["boundary"]) for r in P4))
 pairs4=[]
 def backgrounds(pre,post,sid):
  t1=utc_datetime(pre["UTC"]);t2=utc_datetime(post["UTC"])
  genesis=utc_datetime(time[sid]["genesis_UTC"])
  la,lb=num(pre["lat"]),num(post["lat"]);loa,lob=num(pre["lon"]),num(post["lon"])
  delta_lon=float(wrap_delta_longitude(lob,loa))
  speed1=np.hypot(num(pre.get("motion_u_ms")),num(pre.get("motion_v_ms")))
  speed2=np.hypot(num(post.get("motion_u_ms")),num(post.get("motion_v_ms")))
  return dict(delta_wind_kt=num(post["wind_kt"])-num(pre["wind_kt"]),
    delta_latitude_deg=lb-la,delta_longitude_deg=delta_lon,
    midpoint_latitude_deg=.5*(la+lb),midpoint_longitude_deg=(loa+.5*delta_lon+180)%360-180,
    delta_age_h=(t2-t1).total_seconds()/3600,
    midpoint_age_h=((t1-genesis).total_seconds()+(t2-genesis).total_seconds())/7200,
    midpoint_season_sin=.5*(math.sin(2*math.pi*(t1.month-1)/12)+math.sin(2*math.pi*(t2.month-1)/12)),
    midpoint_season_cos=.5*(math.cos(2*math.pi*(t1.month-1)/12)+math.cos(2*math.pi*(t2.month-1)/12)),
    delta_motion_speed_ms=speed2-speed1,motion_background_available=bool(np.isfinite(speed1+speed2)))
 for sid,boundary in sidboundary:
  for label,l1,l2 in [("minus24_to_0",-24,0),("0_to_plus24",0,24),("minus24_to_plus24",-24,24)]:
   s1=p4lookup[sid,boundary,l1];s2=p4lookup[sid,boundary,l2]
   rid1="P4|"+s1["request_id"];rid2="P4|"+s2["request_id"]
   a=requests[rid1];b=requests[rid2]
   rec=dict(sid=sid,boundary=boundary,change_window=label,
     request_id_before=rid1,request_id_after=rid2,UTC_before=a["UTC"],UTC_after=b["UTC"],
     D_h=num(s1["D_h"]),E_h=num(s1["E_h"]),allocation=num(s1["D_over_DplusE"]),
     basin_genesis=s1["basin_genesis"],point_before_available=explicit_bool(s1["actual_track_point"]),
     point_after_available=explicit_bool(s2["actual_track_point"]),
     stage_before=s1["frozen_stage"],stage_after=s2["frozen_stage"],**backgrounds(a,b,sid))
   pairs4.append(rec)
 pairs5=[]
 for src in P5:
  rid1="P5|"+src["pair_id"]+"|pre";rid2="P5|"+src["pair_id"]+"|post"
  a=requests[rid1];b=requests[rid2]
  pairs5.append(dict(sid=src["sid"],pair_id=src["pair_id"],request_id_before=rid1,
    request_id_after=rid2,UTC_before=a["UTC"],UTC_after=b["UTC"],
    bin_index=int(src["bin_index"]),bin_lower_kt=num(src["bin_lower_kt"]),
    bin_upper_kt=num(src["bin_upper_kt"]),source_pair_weight=num(src["pair_weight_within_sid"]),
    A_signed=num(src["A_signed"]),Tpre_h=num(src["pre_lmi_duration_hours"]),
    Tpost_h=num(src["post_lmi_duration_hours"]),basin_genesis=src["basin_genesis"],
    **backgrounds(a,b,src["sid"])))
 write_rows(out/"P4_fixed_pair_design.csv",pairs4)
 write_rows(out/"P5_fixed_pair_design.csv",pairs5)
 status=dict(created_utc=stamp(),execution_status="PREPARED_ONLY",
   verification_status="SOURCE_GEOMETRY_PREPARED_PENDING_PROCESS_INPUTS",
   scientific_conclusion="NOT_COMPUTED",P4_rows=len(pairs4),P5_pairs=len(pairs5),
   P4_sids=len(set(r["sid"] for r in pairs4)),P5_sids=len(set(r["sid"] for r in pairs5)),
   source_hashes={f:sha(root/f) for f in ["analysis_spec_v5.json","p4_p5/source_spec.json",
    "p4_p5/P4_boundary_requests.csv","p4_p5/P5_matched_intensity_pairs.csv",
    "p4_p5/P5_per_SID_time_audit.csv","data/environment_requests_v5.csv"]},
   explicit_scope="fixed pairs/backgrounds only; old environmental *_value columns deliberately not copied")
 atomic_json(out/"prepared_status.json",status)
 return status

def phase_support_fit(W,phase,row_weights,max_iter=100):
 """Unpenalized logit of phase on fixed W, outcome-blind; no regularized fallback."""
 from scipy.special import expit
 W=np.asarray(W,float);y=np.asarray(phase,float);w=np.asarray(row_weights,float)
 if not (np.isfinite(W).all() and np.isfinite(y).all() and np.isfinite(w).all()):
  raise ValueError("phase support requires finite background")
 if np.any((y!=0)&(y!=1)) or np.any(w<=0):raise ValueError("invalid phase/weights")
 scale=np.sqrt(np.average(W*W,axis=0,weights=w));scale=np.where(scale>0,scale,1.)
 Z=W/scale
 u,s,vt=np.linalg.svd(np.sqrt(w)[:,None]*Z,full_matrices=False)
 tol=np.finfo(float).eps*max(Z.shape)*s[0];rank=int((s>tol).sum())
 V=vt[:rank].T;D=Z@V;b=np.zeros(rank)
 def loss(beta):
  eta=D@beta
  return float(np.sum(w*(np.logaddexp(0,eta)-y*eta)))
 converged=False
 for it in range(max_iter):
  prob=expit(D@b);grad=D.T@(w*(prob-y))
  hess=D.T@((w*prob*(1-prob))[:,None]*D)
  step=np.linalg.lstsq(hess,grad,rcond=None)[0]
  current=loss(b);alpha=1.
  while alpha>2**-25 and loss(b-alpha*step)>current:
   alpha*=.5
  bn=b-alpha*step
  if np.max(np.abs(D@(bn-b)))<1e-9 and np.max(np.abs(grad))/w.sum()<1e-9:
   b=bn;converged=True;break
  b=bn
  if np.max(np.abs(D@b))>40:
   break
 prob=expit(D@b)
 return dict(probability=prob,beta=(V@b)/scale,rank=rank,iterations=it+1,
   converged=converged,status="CONVERGED" if converged else "NO_FINITE_CONVERGED_UNPENALIZED_LOGIT",
   log_likelihood=-loss(b),max_absolute_linear_predictor=float(np.max(np.abs(D@b))),
   column_scale=scale)

def standardized_mean_difference(A,B,w):
 A=np.asarray(A,float);B=np.asarray(B,float);w=np.asarray(w,float)
 m0=np.average(A,axis=0,weights=w);m1=np.average(B,axis=0,weights=w)
 v0=np.average((A-m0)**2,axis=0,weights=w);v1=np.average((B-m1)**2,axis=0,weights=w)
 sd=np.sqrt((v0+v1)/2);delta=m1-m0
 out=np.divide(delta,sd,out=np.full(delta.shape,np.nan),where=sd>0)
 out[(sd==0)&(delta==0)]=0.
 return out

def _pooled_phase_point(sid,pre,post,W0,W1,cats,multiplicity,check_joint=True):
 """One complete adaptive-support functional, applied also within every SID draw."""
 sid=np.asarray(sid,str);unique,inv=np.unique(sid,return_inverse=True)
 mult=np.asarray(multiplicity,float);positive=mult[inv]>0
 ss=sid[positive];a=pre[positive];b=post[positive]
 A=W0[positive];B=W1[positive];cc=cats[positive]
 base=sid_equal_weights(ss)*mult[inv[positive]]
 W=np.vstack([A,B]);phase=np.r_[np.zeros(len(ss)),np.ones(len(ss))]
 logit=phase_support_fit(W,phase,np.r_[base/2,base/2])
 info=dict(logit_iterations=logit["iterations"],logit_status=logit["status"])
 if not logit["converged"]:return dict(success=False,reason="SUPPORT_LOGIT_NO_FINITE_MLE",**info)
 p0,p1=np.split(logit["probability"],2)
 keep=(p0>=.1)&(p0<=.9)&(p1>=.1)&(p1<=.9)
 info.update(n_sid_support=len(set(ss[keep])),n_pairs_support=int(keep.sum()))
 if len(set(ss[keep]))<2:return dict(success=False,reason="NO_PAIRED_COMMON_SUPPORT",**info)
 # Re-normalize supported shared bins within each SID, then apply SID draw count.
 supported_sid=ss[keep];support_mult={s:mult[i] for i,s in enumerate(unique)}
 sw=sid_equal_weights(supported_sid)*np.array([support_mult[s] for s in supported_sid])
 AA=A[keep];BB=B[keep];yy0=a[keep];yy1=b[keep]
 info.update(standardized_mean_difference_before_support=standardized_mean_difference(A,B,base),
   standardized_mean_difference_after_support=standardized_mean_difference(AA,BB,sw),
   raw_same_support=float(np.average(yy1-yy0,weights=sw)),support_row_mask=keep,
   positive_row_mask=positive,propensity_before=p0,propensity_after=p1)
 diagnostic=None
 if check_joint:
  diagnostic=joint_support_diagnostics(np.vstack([AA[:,1:],BB[:,1:]]),
    np.r_[np.zeros(len(AA)),np.ones(len(BB))],.5,np.r_[cc[keep],cc[keep]])
  if any(x["status"]!="CONVEX_HULLS_INTERSECT" for x in diagnostic["strata"]):
   return dict(success=False,reason="NO_JOINT_HULL_SUPPORT",joint_support=diagnostic,**info)
 X=np.column_stack([np.r_[np.zeros(len(AA)),np.ones(len(BB))],np.vstack([AA,BB])])
 c=np.zeros(X.shape[1]);c[0]=1
 fit=weighted_fit(X,np.r_[yy0,yy1],np.r_[sw/2,sw/2],contrast=c)
 if not fit["estimable"]:return dict(success=False,reason="PHASE_CONTRAST_NOT_ESTIMABLE",**info)
 return dict(success=True,reason="",estimate=float(c@fit["beta"]),
   rank=fit["rank"],coefficients=fit["beta"],joint_support=diagnostic,**info)

def phase_standardization(sid,before,after,W_before,W_after,basin=None,
                          draws=BOOTSTRAP_REPEATS,seed=BOOTSTRAP_SEED):
 """Data-adaptive overlap estimand: each SID draw re-fits support AND Y model.

All endpoints/bins are fixed before environmental outcomes. Every draw retains
its failure, support size and exact multiplicities; no accepted-draw top-up.
"""
 sid=np.asarray(sid,str);pre=np.asarray(before,float);post=np.asarray(after,float)
 W0=np.asarray(W_before,float);W1=np.asarray(W_after,float)
 raw=cluster_mean(sid,post-pre,draws,seed)
 valid=np.isfinite(pre)&np.isfinite(post)&np.isfinite(W0).all(axis=1)&np.isfinite(W1).all(axis=1)
 sid=sid[valid];pre=pre[valid];post=post[valid];W0=W0[valid];W1=W1[valid]
 cats=np.full(len(sid),"all") if basin is None else np.asarray(basin,str)[valid]
 unique=np.unique(sid)
 result=dict(input_valid=valid,estimand="data-adaptive common-support additive pooled-phase contrast over empirical W",
   ci_scope="SID percentile bootstrap re-estimating propensity support and outcome model in every draw",
   no_sid_fixed_effect=True,no_causal_claim=True,support_bounds=[.1,.9],
   raw_original_valid_population=raw,n_sid_background_complete=len(unique),
   n_pairs_background_complete=len(sid))
 if len(unique)<2:
  result.update(status="NOT_ESTIMATED_INSUFFICIENT_SIDS");return result
 point=_pooled_phase_point(sid,pre,post,W0,W1,cats,np.ones(len(unique)))
 result["point"]=point
 if not point["success"] and "raw_same_support" not in point:
  result.update(status="NOT_ESTIMATED_"+point["reason"]);return result
 indices,mult=bootstrap_weights(len(unique),draws,seed)
 values=np.full(draws,np.nan);rawsupport=np.full(draws,np.nan);records=[]
 for b in range(draws):
  f=_pooled_phase_point(sid,pre,post,W0,W1,cats,mult[b])
  if f["success"]:values[b]=f["estimate"]
  if "raw_same_support" in f:rawsupport[b]=f["raw_same_support"]
  records.append({k:f.get(k) for k in ("success","reason","logit_iterations","logit_status",
    "n_sid_support","n_pairs_support","rank")})
 result.update(status="ESTIMATED_ADAPTIVE_DESCRIPTIVE" if point["success"] else "NOT_ESTIMATED_"+point["reason"],
   adjusted=dict(estimate=point.get("estimate",np.nan),draws=values),
   raw_same_support=dict(estimate=point["raw_same_support"],draws=rawsupport),
   sampled_sid_indices=indices,sampled_sid_order=unique,draw_records=records,
   bootstrap_failure_count=int(np.isnan(values).sum()),
   n_sid_support=point["n_sid_support"],n_pairs_support=point["n_pairs_support"])
 return result


BACKGROUND_NUMERIC_NAMES = [
 "wind_kt","log1p_age_h","latitude_deg","longitude_sin","longitude_cos",
 "month_sin","month_cos","past_motion_speed_ms","ERA5_lsm_annulus_land_fraction"]

def background_vector(row,genesis_utc,basin_levels,basin):
 """Fixed physically named W; absent motion/land remain NaN, never 0 imputation."""
 dt=utc_datetime(row["UTC"]);age=(dt-utc_datetime(genesis_utc)).total_seconds()/3600
 lat=num(row.get("lat"));lon=num(row.get("lon"))
 wind=num(row.get("wind_kt"));mu=num(row.get("motion_u_ms"));mv=num(row.get("motion_v_ms"))
 ocean=num(row.get("flux_ERA5lsm_ocean_geometric_fraction"))
 vals=[wind,np.log1p(age) if np.isfinite(age) and age>=0 else np.nan,
   lat,np.sin(np.deg2rad(lon)),np.cos(np.deg2rad(lon)),
   math.sin(2*math.pi*(dt.month-1)/12),math.cos(2*math.pi*(dt.month-1)/12),
   np.hypot(mu,mv),1-ocean]
 names=["intercept"]+BACKGROUND_NUMERIC_NAMES+["basin_"+str(b) for b in basin_levels[1:]]
 return np.array([1.]+vals+[float(basin==b) for b in basin_levels[1:]],float),names

def paired_slope_design(target,W0,W1,basin,basin_levels):
 """Original-scale target + actual deltaW + fixed genesis basin background."""
 target=np.asarray(target,float)
 delta=np.asarray(W1,float)[:,1:1+len(BACKGROUND_NUMERIC_NAMES)]-np.asarray(W0,float)[:,1:1+len(BACKGROUND_NUMERIC_NAMES)]
 basin=np.asarray(basin,str)
 X=np.column_stack([np.ones(len(target)),target,delta]+[(basin==b).astype(float) for b in basin_levels[1:]])
 names=["intercept","continuous_target"]+["delta_"+n for n in BACKGROUND_NUMERIC_NAMES]+["basin_"+str(b) for b in basin_levels[1:]]
 return X,names

def build_pair_arrays(pairs,environment,time_table,process,kind,basin_levels):
 """No substitution from old six-core columns; process-specific support retained."""
 key,units=PROCESS_COLUMNS[process]
 records=[];a=[];b=[];w0=[];w1=[];target=[];sid=[];basin=[]
 for pair in pairs:
  r0=environment[pair["request_id_before"]];r1=environment[pair["request_id_after"]]
  s=pair["sid"];ba=pair["basin_genesis"]
  y0=num(r0.get(key));y1=num(r1.get(key))
  # Values accepted only through process QC, including PI validity.
  if not explicit_bool(r0.get(key+"__support_pass",False)):y0=np.nan
  if not explicit_bool(r1.get(key+"__support_pass",False)):y1=np.nan
  W0,names=background_vector(r0,time_table[s]["genesis_UTC"],basin_levels,ba)
  W1,_=background_vector(r1,time_table[s]["genesis_UTC"],basin_levels,ba)
  rec=dict(pair,process=process,units=units,value_before=y0,value_after=y1,
    difference_after_minus_before=y1-y0,process_pair_valid=bool(np.isfinite(y0+y1)),
    background_pair_valid=bool(np.isfinite(W0).all() and np.isfinite(W1).all()),
    source_before_complete=r0.get("source_complete_json",""),
    source_after_complete=r1.get("source_complete_json",""),
    source_before_complete_sha256=r0.get("source_complete_sha256",""),
    source_after_complete_sha256=r1.get("source_complete_sha256",""),
    valid_fraction_before=num(r0.get(key+"__valid_fraction")),
    valid_fraction_after=num(r1.get(key+"__valid_fraction")),
    QC_before=r0.get(key+"__reason",""),QC_after=r1.get(key+"__reason",""))
  for i,name in enumerate(names):
   rec["W_before__"+name]=W0[i];rec["W_after__"+name]=W1[i]
  records.append(rec);a.append(y0);b.append(y1);w0.append(W0);w1.append(W1)
  target.append(num(pair["allocation" if kind=="P4" else "A_signed"]));sid.append(s);basin.append(ba)
 return dict(records=records,sid=np.array(sid,str),before=np.array(a),after=np.array(b),
  W_before=np.array(w0),W_after=np.array(w1),target=np.array(target),basin=np.array(basin,str),
  background_names=names,units=units)

def run_pair_job(arrays,kind,process,job_id,out,include_slope,include_background=True,
                 draws=BOOTSTRAP_REPEATS,seed=BOOTSTRAP_SEED):
 """One prespecified process/window. CLI science gate must pass before calling."""
 import time
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 binding=dict(kind=kind,process=process,job_id=job_id,draws=draws,seed=seed,
  records_sha256=digest(arrays["records"]),analysis_sha256=sha(__file__),
  statistics_sha256=sha(Path(__file__).with_name("v5_statistics.py")),
  include_slope=include_slope,include_background=include_background)
 done=out/"COMPLETE.json"
 if done.exists():
  existing=json.loads(done.read_text(encoding="utf-8"))
  if existing["binding"]!=binding:raise ValueError("existing job differs; explicit new output revision required")
  for f in existing["outputs"]:
   if sha(out/f["name"])!=f["sha256"]:raise ValueError("existing job artifact changed")
  return json.loads((out/"summary.json").read_text(encoding="utf-8"))
 started=stamp();tick=time.perf_counter()
 atomic_json(out/"STARTED.json",dict(binding=binding,started_utc=started,pid=os.getpid()))
 write_rows(out/"paired_source_and_design.csv",arrays["records"])
 sid=arrays["sid"];a=arrays["before"];b=arrays["after"];delta=b-a
 raw=cluster_mean(sid,delta,draws,seed)
 raw_summary=summarize_draws(raw["estimate"],raw["draws"])
 raw_summary.update(n_sid=raw["n_sid"],n_pairs=raw["n_pairs"],attempted_draws=draws,
   failed_draws=int(np.isnan(raw["draws"]).sum()),estimand="SID-equal mean of its process-valid frozen pair differences")
 np.savez_compressed(out/"raw_draws.npz",draw=raw["draws"],sid=raw["sid"],per_sid=raw["per_sid"],
   sampled_sid_indices=raw["sampled_sid_indices"])
 result=dict(kind=kind,process=process,job_id=job_id,units=arrays["units"],raw=raw_summary)
 if include_slope:
  levels=sorted(set(arrays["basin"]))
  X,names=paired_slope_design(arrays["target"],arrays["W_before"],arrays["W_after"],arrays["basin"],levels)
  c=np.zeros(X.shape[1]);c[1]=1
  good=np.isfinite(X).all(axis=1)&np.isfinite(delta)
  if len(set(sid[good]))>=2:
   fit=model_target_draws(sid,X,delta,c,draws,seed)
   res=summarize_draws(fit["estimate"]*.1,fit["draws"]*.1)
   wg=sid_equal_weights(sid[good]);tg=arrays["target"][good]
   mu=np.average(tg,weights=wg);sd=float(np.sqrt(np.average((tg-mu)**2,weights=wg)))
   res.update(n_sid=fit["n_sid"],n_pairs=fit["n_pairs"],rank=fit["rank"],
    row_residual_df=fit["residual_row_df"],cluster_count_minus_rank=fit["cluster_count_minus_rank"],
    df_reference=fit["reference"],attempted_draws=draws,failed_draws=int(np.isnan(fit["draws"]).sum()),
    target_unit="per0.1"+("D/(D+E)" if kind=="P4" else "A"),
    target_SID_weighted_SD=sd,per_SD_estimate=fit["estimate"]*sd,
    per_SD_CI=np.quantile(fit["draws"][np.isfinite(fit["draws"])],[.025,.975])*sd if np.isfinite(fit["draws"]).any() else [np.nan,np.nan])
   result["continuous_slope"]=res
   np.savez_compressed(out/"slope_design_and_draws.npz",X=X,y=delta,sid=sid,column_names=np.array(names),
     input_valid=fit["input_valid"],base_weights=fit["base_weights"],contrast=c,
     coefficient_estimate=fit["fitted_beta"],draw=fit["draws"],sampled_sid_order=fit["sid"],
     sampled_sid_indices=fit["sampled_sid_indices"])
   write_rows(out/"slope_draw_status.csv",fit["draw_records"])
  else:result["continuous_slope"]=dict(status="NOT_ESTIMATED_INSUFFICIENT_BACKGROUND_COMPLETE_SIDS",n_sid=len(set(sid[good])))
 if include_background:
  adj=phase_standardization(sid,a,b,arrays["W_before"],arrays["W_after"],arrays["basin"],draws,seed)
  # P/q deliberately omitted for the descriptive background contrast.
  info={k:v for k,v in adj.items() if k not in ("raw_original_valid_population","point","adjusted","raw_same_support","sampled_sid_indices","sampled_sid_order","draw_records")}
  if "raw_same_support" in adj:
   for term in ("adjusted","raw_same_support"):
    vv=adj[term];d=vv["draws"];ok=d[np.isfinite(d)]
    info[term]=dict(estimate=vv["estimate"],CI=np.quantile(ok,[.025,.975]).tolist() if len(ok) else [np.nan,np.nan],
       valid_draws=len(ok),failed_draws=int(np.isnan(d).sum()),attempted_draws=draws,no_P_or_q=True)
   np.savez_compressed(out/"background_draws.npz",adjusted=adj["adjusted"]["draws"],
      raw_same_support=adj["raw_same_support"]["draws"],sampled_sid_indices=adj["sampled_sid_indices"],
      sampled_sid_order=adj["sampled_sid_order"])
   write_rows(out/"background_draw_status.csv",adj["draw_records"])
  atomic_json(out/"background_point_and_support.json",adj.get("point",{}))
  result["background_standardization"]=info
 atomic_json(out/"summary.json",result)
 files=[p for p in out.iterdir() if p.is_file() and p.name not in ("STARTED.json","COMPLETE.json")]
 atomic_json(done,dict(status="COMPUTED_PENDING_INDEPENDENT_SCIENCE_VERIFICATION",binding=binding,
   started_utc=started,finished_utc=stamp(),elapsed_seconds=time.perf_counter()-tick,
   outputs=[dict(name=p.name,sha256=sha(p),size=p.stat().st_size) for p in sorted(files)]))
 return result

def finalize_prespecified_families(summaries,out):
 """Full planned slots required; NA slots are kept in nominal m."""
 out=Path(out);rows=[]
 for s in summaries:
  base={k:s[k] for k in ("kind","process","job_id","units")}
  for target,key in (("raw","raw"),("continuous_slope","continuous_slope")):
   if key not in s:continue
   v=s[key]
   family=("P4_20_changes" if target=="raw" else "P4_10_allocation_slopes") if s["kind"]=="P4" else ("P5_5_pair_differences" if target=="raw" else "P5_5_continuous_A")
   rows.append(dict(base,target=target,family=family,**v))
 expected={"P4_20_changes":20,"P4_10_allocation_slopes":10,"P5_5_pair_differences":5,"P5_5_continuous_A":5}
 for family,m in expected.items():
  selected=[i for i,r in enumerate(rows) if r["family"]==family]
  if len(selected)!=m:raise ValueError(f"{family}: expected{m} frozen slots, received{len(selected)}")
  q=bh_fixed_family([num(rows[i].get("p")) for i in selected],m)
  for i,x in zip(selected,q):
   rows[i]["q"]=x;rows[i]["nominal_family_size"]=m
 write_rows(out/"P4_P5_primary_family_results.csv",rows)
 return rows


def extraction_ledger_index(launch_path):
 """Use the actual launch/barrier ledger bindings, never directory-name selection."""
 launch_path=Path(launch_path);launch=json.loads(launch_path.read_text(encoding="utf-8"));result={}
 for worker in launch["workers"]:
  ready=json.loads((launch_path.parent/"barrier"/f"ready_{worker['shard']}.json").read_text(encoding="utf-8"))
  if ready["pid"]!=worker["pid"]:raise ValueError("worker/barrier PID identity conflict")
  ledger=Path(ready["audit_directory"])/"request_ledger.jsonl"
  lines=ledger.read_text(encoding="utf-8").splitlines()
  for i,line in enumerate(lines):
   try:r=json.loads(line)
   except json.JSONDecodeError:
    if i==len(lines)-1:continue  # concurrent atomic-line append not yet complete
    raise
   rid=r["request_id"]
   if rid in result and result[rid].get("folder")!=r.get("folder"):raise ValueError("multiple source folders for request")
   result[rid]=r
 return launch,result

def verified_current_environment_row(root,request,ledger_row,thermo_row,mfc_output=None):
 root=Path(root);folder=Path(ledger_row["folder"]);complete=folder/"COMPLETE.json"
 done=json.loads(complete.read_text(encoding="utf-8"));binding=done["binding"]
 if done["status"]!="COMPLETE" or binding["request"]!=request:raise ValueError("extraction source binding mismatch")
 if binding["driver_sha"]!=sha(root/"src/extract_environment_v5.py"):raise ValueError("extraction driver mismatch")
 sem=json.loads((root/"config/extraction_semantics_v5.json").read_text(encoding="utf-8"))
 if binding["semantics_sha"]!=digest(sem):raise ValueError("extraction semantic binding mismatch")
 files={x["name"]:x["sha256"] for x in done["outputs"]}
 for name in ("request_values.json","scalars.csv"):
  if sha(folder/name)!=files[name]:raise ValueError("extraction consumed file checksum mismatch")
 val=json.loads((folder/"request_values.json").read_text(encoding="utf-8"))
 val.update(request)
 scalars={r["variable"]:r for r in read_rows(folder/"scalars.csv")}
 for process,(key,unit) in PROCESS_COLUMNS.items():
  if key in scalars:
   sc=scalars[key]
   val[key]=num(sc["value"])
   val[key+"__support_pass"]=explicit_bool(sc["support_pass"])
   val[key+"__valid_fraction"]=num(sc["valid_fraction"])
   val[key+"__reason"]=sc["reason"]
 if Path(thermo_row["request_folder"])!=folder:raise ValueError("thermo belongs to another extraction")
 tc=Path(thermo_row["derived_complete_path"])
 if sha(tc)!=thermo_row["derived_complete_sha256"]:raise ValueError("thermo COMPLETE mismatch")
 td=json.loads(tc.read_text(encoding="utf-8"))
 if td["binding"]["source"]["checkpoint_sha256"]!=sha(complete):raise ValueError("thermo bound to another extraction COMPLETE")
 t_files={r["name"]:r["sha256"] for r in td["outputs"]}
 if sha(tc.parent/"derived.json")!=t_files["derived.json"]:raise ValueError("thermo result changed")
 result=json.loads((tc.parent/"derived.json").read_text(encoding="utf-8"))
 # Aggregation may have identical typed values formatted as text, but the actual
 # per-request derived result is the numeric authority.
 if str(result["request_id"])!=request["request_id"]:raise ValueError("thermo request id conflict")
 if utc_datetime(result["wind_source_UTC"])!=utc_datetime(request["UTC"]):raise ValueError("headroom wind time conflict")
 pi_valid=explicit_bool(result["pi_valid"]);headroom=num(result["thermal_headroom_ms"])
 val["PI_headroom_ms"]=headroom if pi_valid else np.nan
 val["PI_headroom_ms__support_pass"]=bool(pi_valid and np.isfinite(headroom))
 val["PI_headroom_ms__valid_fraction"]=np.nan
 val["PI_headroom_ms__reason"]=result["pi_quality_status"]
 val["PI_source_complete_json"]=str(tc);val["PI_source_complete_sha256"]=sha(tc)
 mfc=derive_mfc_from_checkpoint(root,complete,mfc_output);key=PROCESS_COLUMNS["q850_moisture_flux_convergence"][0]
 val[key]=mfc[key];val[key+"__support_pass"]=mfc["statistics"]["support_pass"]
 val[key+"__valid_fraction"]=mfc["statistics"]["valid_fraction"]
 val[key+"__reason"]="PASS" if mfc["statistics"]["support_pass"] else "BELOW_ORIGINAL_AREA_SUPPORT"
 val["MFC_source_json"]=str(mfc_output) if mfc_output else ""
 val["source_complete_json"]=str(complete);val["source_complete_sha256"]=sha(complete)
 val["extraction_terminal_status"]="COMPLETE";val["thermo_terminal_status"]="COMPLETE"
 return val

def assemble_environment_inputs(root,launch_path,thermo_index,out):
 """All requested P4/P5 outcomes must be terminal before assembling any science."""
 root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
 launch,index=extraction_ledger_index(launch_path)
 requests=[r for r in read_rows(root/"data/environment_requests_v5.csv") if r["module"] in ("P4","P5")]
 if sha(root/"data/environment_requests_v5.csv")!=launch["source_identity"]["requests_sha256"]:
  raise ValueError("requested source table differs from active launch")
 thermo={r["request_id"]:r for r in read_rows(thermo_index)}
 pending=[];failed=[];needed=[];excluded=[]
 for r in requests:
  rid=r["request_id"];rec=index.get(rid)
  if rec is None:pending.append(dict(request_id=rid,reason="NO_TERMINAL_EXTRACTION"));continue
  if rec["status"]=="EXCLUDED_MISSING_ACTUAL_POSITION":excluded.append(r);continue
  if rec["status"] not in ("COMPLETE","REUSED_VERIFIED_CHECKPOINT"):
   failed.append(dict(request_id=rid,reason=rec["status"]));continue
  if rid not in thermo:pending.append(dict(request_id=rid,reason="NO_TERMINAL_THERMO"));continue
  needed.append(r)
 status=dict(created_utc=stamp(),expected_requests=len(requests),ready_requests=len(needed),
  excluded_no_position=len(excluded),pending_requests=len(pending),failed_requests=len(failed),
  extraction_launch=str(launch_path),thermo_index=str(thermo_index),
  science_fitting_status="NOT_RUN",status="READY_TO_ASSEMBLE" if not pending and not failed else "PREPARED_WAITING_FOR_INPUTS")
 if pending or failed:
  atomic_json(out/"coverage_status.json",status)
  write_rows(out/"outstanding_inputs.csv",pending+failed)
  return status
 rows=[]
 for i,r in enumerate(requests):
  rid=r["request_id"]
  if index[rid]["status"]=="EXCLUDED_MISSING_ACTUAL_POSITION":
   val=dict(r,extraction_terminal_status="EXCLUDED_MISSING_ACTUAL_POSITION",thermo_terminal_status="NOT_APPLICABLE_NO_POSITION")
   for key,unit in PROCESS_COLUMNS.values():
    val[key]=np.nan;val[key+"__support_pass"]=False;val[key+"__reason"]="NO_OBSERVED_POSITION"
  else:
   mfc_file=out/"mfc_native_cache_derivatives"/(index[rid]["request_hash"]+".json")
   val=verified_current_environment_row(root,r,index[rid],thermo[rid],mfc_file)
  rows.append(val)
  if (i+1)%250==0:atomic_json(out/"assembly_progress.json",dict(completed=i+1,total=len(requests),utc=stamp()))
 table=out/"environment_ready.csv";write_rows(table,rows)
 status.update(status="ASSEMBLED_INPUTS_PENDING_SCIENCE",ready_table=str(table),ready_table_sha256=sha(table),
  source_request_table_sha256=sha(root/"data/environment_requests_v5.csv"),
  thermo_index_sha256=sha(thermo_index),analysis_module_sha256=sha(__file__))
 atomic_json(out/"coverage_status.json",status)
 return status

def analyze_complete_inputs(root,ready_folder,out):
 """Run only the frozen 20+10+5+5 families after full source/thermo assembly."""
 root=Path(root);ready_folder=Path(ready_folder);out=Path(out)
 status=json.loads((ready_folder/"coverage_status.json").read_text(encoding="utf-8"))
 if status["status"]!="ASSEMBLED_INPUTS_PENDING_SCIENCE":raise ValueError("P4/P5 inputs are not complete")
 table=Path(status["ready_table"])
 if sha(table)!=status["ready_table_sha256"]:raise ValueError("ready table changed")
 if status["source_request_table_sha256"]!=sha(root/"data/environment_requests_v5.csv"):raise ValueError("source requests changed")
 if status["analysis_module_sha256"]!=sha(__file__):raise ValueError("analysis implementation changed since input assembly")
 env={r["request_id"]:r for r in read_rows(table)}
 time={r["sid"]:r for r in read_rows(root/"p4_p5/P5_per_SID_time_audit.csv")}
 levels=sorted(set(r["basin_genesis"] for r in time.values()))
 base=root/"p4_p5/analysis_prepared"
 pairs4=read_rows(base/"P4_fixed_pair_design.csv");pairs5=read_rows(base/"P5_fixed_pair_design.csv")
 summaries=[]
 for boundary in ("D_to_E","E_to_F"):
  for window in ("minus24_to_0","0_to_plus24"):
   pairs=[r for r in pairs4 if r["boundary"]==boundary and r["change_window"]==window]
   for process in PROCESS_COLUMNS:
    arr=build_pair_arrays(pairs,env,time,process,"P4",levels)
    job=boundary+"__"+window+"__"+process
    summaries.append(run_pair_job(arr,"P4",process,job,out/"jobs"/job,window=="minus24_to_0"))
    atomic_json(out/"progress.json",dict(finished_jobs=len(summaries),total_jobs=25,last_job=job,utc=stamp()))
 for process in PROCESS_COLUMNS:
  arr=build_pair_arrays(pairs5,env,time,process,"P5",levels);job="P5__"+process
  summaries.append(run_pair_job(arr,"P5",process,job,out/"jobs"/job,True))
  atomic_json(out/"progress.json",dict(finished_jobs=len(summaries),total_jobs=25,last_job=job,utc=stamp()))
 results=finalize_prespecified_families(summaries,out)
 atomic_json(out/"execution_status.json",dict(execution="COMPUTED",verification="PENDING_INDEPENDENT_SCIENCE_CHECK",
  scientific_conclusion="NOT_YET_REVIEWED",finished_utc=stamp(),jobs=25,primary_family_targets=len(results),
  inputs=str(ready_folder),no_ERA5_read=True,no_legacy_model_or_H2_rerun=True))
 return results

def main():
 p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--prepare",action="store_true")
 p.add_argument("--mfc-checkpoint");p.add_argument("--output")
 p.add_argument("--assemble",action="store_true");p.add_argument("--launch");p.add_argument("--thermo-index")
 p.add_argument("--analyze",action="store_true");p.add_argument("--ready-folder")
 a=p.parse_args();root=Path(a.root)
 if a.assemble:
  if not (a.launch and a.thermo_index and a.output):raise SystemExit("assemble requires explicit launch, thermo index and output")
  print(json.dumps(assemble_environment_inputs(root,a.launch,a.thermo_index,a.output)))
 elif a.analyze:
  # Authorized v5.1 safe-checkpoint migration; this gate changes no scientific functions.
  hold_path=root/'checkpoints/P4_P5_PI_HOLD.json'
  hold=json.loads(hold_path.read_text(encoding='utf-8-sig')) if hold_path.exists() else {}
  if hold.get('status')!='RELEASED_WITH_VERIFIED_INPUT':
   print(json.dumps(dict(status='P4_P5_PI_PHYSICAL_HOLD',reason='Full25-job analysis requires explicit input-verified release',hold_file=str(hold_path))))
   raise SystemExit(42)
  if not (a.ready_folder and a.output):raise SystemExit("analyze requires explicit ready-folder and output")
  print(json.dumps(dict(completed_targets=len(analyze_complete_inputs(root,a.ready_folder,a.output)))))
 elif a.prepare:
  out=Path(a.output) if a.output else root/"p4_p5/analysis_prepared"
  print(json.dumps(prepare_source_design(root,out),ensure_ascii=False))
 elif a.mfc_checkpoint:
  print(json.dumps(clean(derive_mfc_from_checkpoint(root,a.mfc_checkpoint,a.output)),ensure_ascii=False))
 else:raise SystemExit("explicit --prepare or --mfc-checkpoint required; no implicit scientific fits")
if __name__=="__main__":main()
