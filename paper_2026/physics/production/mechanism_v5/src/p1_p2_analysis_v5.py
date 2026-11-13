"""P1/P2 scalar/sector comparisons from COMPLETE v5 caches; no raw ERA5 access.
Real inference requires explicit source-verification receipt and all requested records.
The synthetic runner is separate and cannot write to production result directories.
"""
from pathlib import Path
from datetime import datetime,timezone
import os
for _k in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS","NUMEXPR_NUM_THREADS"):
 os.environ[_k]="1"
import argparse,ast,hashlib,json,sys,traceback
import numpy as np
import pandas as pd
from v5_statistics import bh,bootstrap_weights,paired_mean_draws,summarize_draws
ROOT=Path(__file__).resolve().parents[1]
LAGS=list(range(-48,49,6));PRE=[-24,-18,-12,-6];POST=[0,6,12,18,24]
PROCESSES={
 "shear200_850":("shear_200_850_vector_magnitude_ms","m s-1",1.),
 "shear500_850":("shear_500_850_vector_magnitude_ms","m s-1",1.),
 "shear200_500":("shear_200_500_vector_magnitude_ms","m s-1",1.),
 "midRH":("mid_rh_pct","%",1.),
 "q700_storm_relative_horizontal_advection":("q700_horizontal_storm_relative_kgkg_s","kg kg-1 s-1",86400000.),
 "q700_vertical_advection":("q700_vertical_advection_kgkg_s","kg kg-1 s-1",86400000.),
 "PI_headroom":("thermal_headroom_ms","m s-1",1.),
 "Tang_Emanuel_ventilation":("raw_signed_index","dimensionless",1.),
 "SST":("sst_C","degC",1.)}
UNITS={k:("g kg-1 day-1" if "advection" in k else "pp" if k=="midRH" else z[1]) for k,z in PROCESSES.items()}
DIRECTIONS={"shear200_850":1,"shear500_850":1,"shear200_500":1,"midRH":-1,"PI_headroom":-1,"Tang_Emanuel_ventilation":1}
LABELS={k:k for k in PROCESSES}
LABELS.update(midRH="500/700-hPa relative humidity",Tang_Emanuel_ventilation="Signed entropy-shear ventilation diagnostic",
 PI_headroom="Environmental-profile PI minus contemporaneous wind",q700_storm_relative_horizontal_advection="700-hPa storm-relative horizontal humidity advection",
 q700_vertical_advection="700-hPa vertical humidity advection")
SECTORS=["downshear","left_of_shear","upshear","right_of_shear"]
SECTORVARS={"r700_percent":"pp","q700_gkg":"g kg-1","q700_storm_relative_advection":"g kg-1 day-1"}
def stamp():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(tuple,list)):return [clean(v) for v in x]
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,Path):return str(x)
 if isinstance(x,float) and not np.isfinite(x):return None
 return x
def jsave(p,x):
 Path(p).parent.mkdir(parents=True,exist_ok=True)
 Path(p).write_text(json.dumps(clean(x),ensure_ascii=False,indent=2),encoding="utf-8")
def boolean(x):
 if isinstance(x,(bool,np.bool_)):return bool(x)
 if str(x).strip().lower() in ("true","1"):return True
 if str(x).strip().lower() in ("false","0"):return False
 raise ValueError("Explicit boolean required: "+repr(x))
def number(x):
 try:return float(x)
 except (ValueError,TypeError):return np.nan
def utc(x):return pd.Timestamp(x).tz_localize("UTC") if pd.Timestamp(x).tzinfo is None else pd.Timestamp(x).tz_convert("UTC")
def csvread(p):return pd.read_csv(p,keep_default_na=False,na_values=[""])
def native_functions():
 """Execute only existing pure array functions. This deliberately does not import netCDF4."""
 p=ROOT/"src/native_fields_v5.py";tree=ast.parse(p.read_text(encoding="utf-8-sig"))
 names={"_as_float","_field_grid","spherical_gradient","storm_relative_advection","area_weighted_support"}
 nodes=[z for z in tree.body if isinstance(z,ast.FunctionDef) and z.name in names]
 if len(nodes)!=len(names):raise ValueError("Native pure function definitions not found")
 env=dict(np=np,EARTH_RADIUS_M=6371008.8)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),"exec"),env)
 return env
def validate_requests(requests,events,strict=True):
 r=requests.copy();e=events.copy()
 if "module" in r:r=r[r.module.eq("P1")].copy()
 for c in ("current117","candidate123","need_map","need_profile","position_extrapolated"):
  r[c]=r[c].map(boolean)
 for c in ("current117","candidate123"):e[c]=e[c].map(boolean)
 if r.request_id.duplicated().any() or e.event_id.duplicated().any():raise ValueError("Duplicate request/event")
 if r.position_extrapolated.any():raise ValueError("Extrapolated request")
 r["lag_h"]=pd.to_numeric(r.lag_h).astype(int)
 if not set(r.lag_h)<=set(LAGS):raise ValueError("Unexpected lag")
 if set(r.event_class)!={"internal","final"}:raise ValueError("Wrong event classes")
 if set(r.center_kind)!={"fixed_t0","actual_track"}:raise ValueError("Wrong centre definitions")
 if set(r.event_id)!=set(e.event_id):raise ValueError("Request/event identity mismatch")
 if not e.groupby("sid").event_class.apply(lambda x:set(x)=={"internal","final"} and len(x)==2).all():raise ValueError("Not exactly one frozen pair per SID")
 for z in r.itertuples():
  if utc(z.UTC)!=utc(z.t0)+pd.Timedelta(hours=int(z.lag_h)):raise ValueError("UTC/lag mismatch")
 if strict:
  if e.sid.nunique()!=123 or e.loc[e.current117,"sid"].nunique()!=117:raise ValueError("Frozen117/123 mismatch")
  if len(r)!=123*2*2*len(LAGS):raise ValueError("Incomplete expected request inventory")
 for eid,g in r.groupby("event_id"):
  if set(zip(g.center_kind,g.lag_h))!={(c,l) for c in ("fixed_t0","actual_track") for l in LAGS}:raise ValueError("Event request grid incomplete")
 return r,e
def preangle(long,events):
 rows=[]
 for e in events.itertuples():
  z=long[(long.event_id==e.event_id)&(long.center_kind=="fixed_t0")&long.lag_h.isin(PRE)]
  p=z.pivot(index="lag_h",columns="variable",values="value").reindex(PRE)
  for k in ("shear_u","shear_v"):
   if k not in p:p[k]=np.nan
  g=np.isfinite(p[["shear_u","shear_v"]]).all(axis=1);n=int(g.sum())
  u=float(p.loc[g,"shear_u"].mean()) if n>=3 else np.nan
  v=float(p.loc[g,"shear_v"].mean()) if n>=3 else np.nan
  mag=np.hypot(u,v);valid=np.isfinite(mag) and mag>=2.
  rows.append(dict(sid=e.sid,event_id=e.event_id,event_class=e.event_class,n_pre_vector=n,mean_shear_u=u,mean_shear_v=v,
   mean_shear_magnitude=mag,fixed_angle_rad=float(np.arctan2(v,u)) if valid else np.nan,sector_eligible=valid,
   reason="PASS" if valid else "INSUFFICIENT_PRE_VECTOR" if n<3 else "PRE_VECTOR_MAGNITUDE_BELOW_2_MS",
   angle_source_center="fixed_t0",angle_source_lags="-24,-18,-12,-6",unrotated_scalar_retained=True))
 return pd.DataFrame(rows)
def native_sectors(payload,center_lat,center_lon,angle,cu,cv):
 f=native_functions()
 lat=np.asarray(payload["latitude"]);lon=np.asarray(payload["longitude"])
 phi=np.deg2rad(lat[:,None]);phi0=np.deg2rad(center_lat)
 dl=np.deg2rad((lon[None,:]-center_lon+180)%360-180)
 east=np.cos(phi)*np.sin(dl);north=np.cos(phi0)*np.sin(phi)-np.sin(phi0)*np.cos(phi)*np.cos(dl)
 angle_grid=np.arctan2(north,east)
 sector=np.floor(((angle_grid-angle+np.pi/4)%(2*np.pi))/(np.pi/2)).astype(int)
 ring=(payload["distance_km"]>=200)&(payload["distance_km"]<800)
 advection=f["storm_relative_advection"](payload["q700"],payload["u700"],payload["v700"],cu,cv,lat,lon)*86400000.
 vals={"r700_percent":np.asarray(payload["r700"]),"q700_gkg":np.asarray(payload["q700"])*1000,
       "q700_storm_relative_advection":advection}
 out=[]
 for i,name in enumerate(SECTORS):
  for var,a in vals.items():
   st=f["area_weighted_support"](a,payload["cell_area_m2"],ring&(sector==i),min_fraction=.9)
   out.append(dict(sector=name,variable=var,value=st["mean"],units=SECTORVARS[var],**{k:v for k,v in st.items() if k!="mean"},
    definition="700-hPa native scalar/advection; four fixed 90-degree sectors; no regridding before derivative",
    direction_fixed_rad=angle))
 return out
def verify_folder(folder,r,expected_driver=None,require_map=False):
 folder=Path(folder);cp=read(folder/"COMPLETE.json")
 if cp.get("status")!="COMPLETE":raise ValueError("Extraction incomplete "+str(folder))
 binding=cp["binding"];br=binding["request"]
 for k in ("request_id","sid","event_id","event_class","center_kind"):
  if str(br[k])!=str(r[k]):raise ValueError("Source identity mismatch "+k)
 if utc(br["UTC"])!=utc(r["UTC"]) or int(br["lag_h"])!=int(r["lag_h"]):raise ValueError("Source timing mismatch")
 for k in ("lat","lon","motion_u_ms","motion_v_ms"):
  if not np.isclose(number(br.get(k)),number(r.get(k)),rtol=0,atol=1e-10,equal_nan=True):raise ValueError("Source position/motion mismatch "+k)
 for k in ("need_map","need_profile","position_extrapolated"):
  if boolean(br.get(k))!=boolean(r.get(k)):raise ValueError("Source request semantics mismatch "+k)
 if expected_driver and binding["driver_sha"]!=expected_driver:raise ValueError("Driver version conflict")
 listed={z["name"]:z for z in cp["outputs"]}
 names=["scalars.csv","request_values.json"]
 if require_map:names+=["native_map_1500km.npz"]
 for n in names:
  if n not in listed or sha(folder/n)!=listed[n]["sha256"]:raise ValueError("Source artifact identity conflict: "+str(folder/n))
 return cp
def build_sources(requests,events,cache_index,thermo,track,expected_driver=None):
 """Only COMPLETE cache source tables/native npz. No environment effect summary here."""
 r,e=validate_requests(requests,events)
 ci=cache_index.set_index("request_id")
 if ci.index.duplicated().any():raise ValueError("Duplicate cache index")
 th=thermo.set_index("request_id")
 if th.index.duplicated().any():raise ValueError("Duplicate thermodynamic request")
 tk=track[["sid","UTC","legal_interval_v3"]].copy();tk["UTC"]=pd.to_datetime(tk.UTC,utc=True)
 if tk.duplicated(["sid","UTC"]).any():raise ValueError("Duplicate frozen track")
 leg={(z.sid,utc(z.UTC)):boolean(z.legal_interval_v3) for z in tk.itertuples()}
 rows=[];provenance=[];folders={};rr=[]
 for z in r.to_dict("records"):
  rid=z["request_id"];position=np.isfinite([number(z["lat"]),number(z["lon"])]).all()
  z["legal_6h_from_UTC"]=bool(z["center_kind"]=="actual_track" and leg.get((z["sid"],utc(z["UTC"])),False))
  rr.append(z)
  if not position:
   status=str(ci.loc[rid,"status"]) if rid in ci.index else "NOT_INDEXED"
   if status!="EXCLUDED_MISSING_ACTUAL_POSITION":raise ValueError("Missing-position exclusion not recorded "+rid)
   scalar={};folder=None;cp=None
  else:
   if rid not in ci.index:raise ValueError("Expected real-position request absent "+rid)
   zz=ci.loc[rid]
   if str(zz["status"]) not in ("COMPLETE","REUSED_VERIFIED_CHECKPOINT"):raise ValueError("Request not complete "+rid)
   folder=Path(zz["folder"]);cp=verify_folder(folder,z,expected_driver,z["need_map"]);folders[rid]=folder
   sc=csvread(folder/"scalars.csv")
   if sc.variable.duplicated().any():raise ValueError("Duplicate scalar in "+rid)
   scalar=sc.set_index("variable").to_dict("index")
  base={k:z[k] for k in ("request_id","sid","event_id","event_class","lag_h","center_kind","UTC","current117","candidate123")}
  base.update(legal_6h_from_UTC=z["legal_6h_from_UTC"],position_observed=bool(position),source_folder=str(folder) if folder else "")
  for var,(name,unit,factor) in PROCESSES.items():
   value=np.nan;reason="NO_ACTUAL_POSITION" if not position else "VARIABLE_NOT_REQUESTED";fraction=np.nan;support=False
   if name in scalar:
    a=scalar[name]
    if a["units"]!=unit:raise ValueError("Unit mismatch "+name+": "+str(a["units"]))
    fraction=number(a["valid_fraction"]);support=boolean(a["support_pass"]) and fraction>=.9
    value=number(a["value"])*factor if support else np.nan;reason=str(a["reason"])
   elif var in ("PI_headroom","Tang_Emanuel_ventilation") and z["need_profile"] and position:
    if rid not in th.index:raise ValueError("Thermodynamic derived request absent "+rid)
    a=th.loc[rid]
    support=boolean(a["pi_valid"]) if var=="PI_headroom" else boolean(a["signed_estimable"])
    value=number(a[name]) if support else np.nan
    dcp=Path(a["derived_complete_path"])
    if sha(dcp)!=str(a["derived_complete_sha256"]):raise ValueError("Derived checkpoint identity mismatch "+rid)
    dc=read(dcp)
    if dc.get("status")!="COMPLETE":raise ValueError("Derived checkpoint incomplete "+rid)
    if dc["binding"]["source"]["checkpoint_sha256"]!=sha(folder/"COMPLETE.json"):raise ValueError("Derived input checkpoint differs "+rid)
    df=dcp.parent/"derived.json"
    manifest={x["name"]:x["sha256"] for x in dc["outputs"]}
    if manifest.get("derived.json")!=sha(df):raise ValueError("Derived values identity mismatch "+rid)
    original=read(df)
    if original["request_id"]!=rid or not np.isclose(number(original[name]),number(a[name]),equal_nan=True,rtol=1e-12,atol=1e-12):raise ValueError("Derived aggregate/value mismatch "+rid)
    reason=str(a["pi_quality_status"] if var=="PI_headroom" else a["ventilation_quality_status"])
    if var=="PI_headroom" and not np.isfinite(value):reason="NO_CONTEMPORANEOUS_WIND_OR_INVALID_PI"
   rows.append(dict(base,variable=var,value=value,units=UNITS[var],support_pass=bool(support and np.isfinite(value)),valid_fraction=fraction,reason=reason))
  for var,name in [("shear_u","shear_200_850_u_ms"),("shear_v","shear_200_850_v_ms")]:
   a=scalar.get(name,{});good=boolean(a.get("support_pass",False)) and number(a.get("valid_fraction"))>=.9
   rows.append(dict(base,variable=var,value=number(a.get("value")) if good else np.nan,units="m s-1",support_pass=good,
                    valid_fraction=number(a.get("valid_fraction")),reason=a.get("reason","NO_SOURCE")))
  provenance.append(dict(request_id=rid,status="COMPLETE" if position else "EXCLUDED_MISSING_ACTUAL_POSITION",
    folder=str(folder) if folder else "",checkpoint_sha256=sha(folder/"COMPLETE.json") if folder else "",
    scalar_sha256=sha(folder/"scalars.csv") if folder else "",native_sha256=sha(folder/"native_map_1500km.npz") if folder and z["need_map"] else ""))
 long=pd.DataFrame(rows);req=pd.DataFrame(rr);angles=preangle(long,e);ai=angles.set_index("event_id");sectors=[]
 for z in req.to_dict("records"):
  aa=ai.loc[z["event_id"]]
  if not z["need_map"] or z["request_id"] not in folders or not aa["sector_eligible"]:continue
  with np.load(folders[z["request_id"]]/"native_map_1500km.npz",allow_pickle=False) as zz:payload={k:zz[k] for k in zz.files}
  ss=native_sectors(payload,float(z["lat"]),float(z["lon"]),float(aa.fixed_angle_rad),number(z["motion_u_ms"]),number(z["motion_v_ms"]))
  for x in ss:
   x.update({k:z[k] for k in ("request_id","sid","event_id","event_class","lag_h","center_kind","UTC","current117","candidate123")})
   x["physical_hemisphere"]="NH" if float(z["lat"])>=0 else "SH";x["SH_reflection_applied"]=False
   sectors.append(x)
 return long,pd.DataFrame(sectors),angles,req,pd.DataFrame(provenance)
def lookup_cube(long,sid,var,center_policy,extra=None):
 g=long[(long.sid==sid)&(long.variable==var)]
 if extra:
  for k,v in extra.items():g=g[g[k]==v]
 arr=np.full((2,len(LAGS)),np.nan);legal=np.zeros((2,len(LAGS)),bool);ids=np.empty((2,len(LAGS)),dtype=object);ids[:]=""
 for i,side in enumerate(("internal","final")):
  for j,lag in enumerate(LAGS):
   center="fixed_t0" if center_policy=="fixed_pre_along_post" and lag<0 else "actual_track"
   z=g[(g.event_class==side)&(g.lag_h==lag)&(g.center_kind==center)]
   if len(z)>1:raise ValueError("Duplicate analysis cell")
   if len(z):
    a=z.iloc[0];arr[i,j]=number(a.value);ids[i,j]=a.request_id
    if "legal_6h_from_UTC" in a:legal[i,j]=boolean(a.legal_6h_from_UTC)
 return arr,legal,ids
def pair_record(sid,var,summary,internal,final,units,policy,**kwargs):
 valid=np.isfinite(internal) and np.isfinite(final)
 return dict(sid=sid,variable=var,summary=summary,internal=internal,final=final,difference=internal-final if valid else np.nan,
             units=units,event_units=("%" if (var=="midRH" or var.endswith("|r700_percent")) and (summary=="pre_mean" or summary.startswith("lag_")) else units),
             process_label=LABELS.get(var,var),center_policy=policy,contrast_direction="internal minus final",**kwargs)
def scalar_pairs(long,sids,policy):
 rows=[];support=[]
 for sid in sids:
  for var in PROCESSES:
   a,legal,ids=lookup_cube(long,sid,var,policy);ip=[LAGS.index(x) for x in PRE]
   pre=a[:,ip];common=np.isfinite(pre).all(axis=0);npair=int(common.sum())
   pm=np.mean(pre[:,common],axis=1) if npair>=3 else np.array([np.nan,np.nan])
   rows.append(pair_record(sid,var,"pre_mean",*pm,UNITS[var],policy,n_common_pre=npair,
    source_request_ids=";".join(ids[:,ip][:,common].reshape(-1)),family="P2_pre_and_change_18"))
   k0=LAGS.index(0);k24=LAGS.index(24);joint=np.isfinite(a[:,[k0,k24]]).all()
   change=a[:,k24]-a[:,k0] if joint else np.array([np.nan,np.nan])
   rows.append(pair_record(sid,var,"change_t0_to24h",*change,UNITS[var],policy,
    source_request_ids=";".join(ids[:,[k0,k24]].reshape(-1)),family="P2_pre_and_change_18"))
   for j,lag in enumerate(LAGS):
    v=a[:,j] if np.isfinite(a[:,j]).all() else [np.nan,np.nan]
    rows.append(pair_record(sid,var,"lag_"+str(lag),*v,UNITS[var],policy,source_request_ids=";".join(ids[:,j]),
     family="DESCRIPTIVE_NO_PER_LAG_TEST"))
   if var in DIRECTIONS:
    k=[LAGS.index(x) for x in POST];xx=a[:,k];joint_interval=np.isfinite(xx[:,:-1]).all(axis=0)&np.isfinite(xx[:,1:]).all(axis=0)
    joint_interval&=legal[:,k[:-1]].all(axis=0)
    adverse=DIRECTIONS[var]*(xx-pm[:,None])>0
    hh=6*int(joint_interval.sum())
    ah=np.sum((adverse[:,:-1]&adverse[:,1:])&joint_interval[None,:],axis=1)*6.
    transitions=np.sum((adverse[:,:-1]!=adverse[:,1:])&joint_interval[None,:],axis=1)
    if not np.isfinite(pm).all() or hh==0:ah[:]=np.nan
    full=hh==24 and np.isfinite(pm).all();primary=ah if full else [np.nan,np.nan]
    rows.append(pair_record(sid,var,"adverse_endpoint_covered_hours_0_24",*primary,"h",policy,
       n_common_pre=npair,common_covered_h=hh,source_request_ids=";".join(ids[:,k].reshape(-1)),family="P2_persistence_6"))
    support.append(dict(sid=sid,variable=var,center_policy=policy,common_valid_pre=npair,common_covered_h=hh,
      internal_partial_adverse_h=ah[0],final_partial_adverse_h=ah[1],internal_transition_intervals=int(transitions[0]),
      final_transition_intervals=int(transitions[1]),full24_primary=full,
      interval_support=";".join(str(POST[j])+"to"+str(POST[j+1])+":"+str(bool(b)) for j,b in enumerate(joint_interval)),
      reason="PASS_FULL24" if full else "PRE_SUPPORT_BELOW3" if npair<3 else "INCOMPLETE_COMMON_24H_SUPPORT",
      definition="covered 6-h intervals adverse at both sampled endpoints; no continuous-between-samples claim; no interpolation"))
 return pd.DataFrame(rows),pd.DataFrame(support)
def sector_pairs(long,sids,policy):
 rows=[]
 if not len(long):return pd.DataFrame()
 for sid in sids:
  for sec in SECTORS:
   for var,unit in SECTORVARS.items():
    a,_,ids=lookup_cube(long,sid,var,policy,dict(sector=sec));ip=[LAGS.index(x) for x in PRE]
    common=np.isfinite(a[:,ip]).all(axis=0);n=int(common.sum());pm=a[:,ip][:,common].mean(axis=1) if n>=3 else [np.nan,np.nan]
    rows.append(pair_record(sid,sec+"|"+var,"pre_mean",*pm,unit,policy,n_common_pre=n,family="SECTOR_DESCRIPTIVE_NO_NEW_TEST"))
    for j,lag in enumerate(LAGS):
     v=a[:,j] if np.isfinite(a[:,j]).all() else [np.nan,np.nan]
     rows.append(pair_record(sid,sec+"|"+var,"lag_"+str(lag),*v,unit,policy,
       source_request_ids=";".join(ids[:,j]),family="SECTOR_DESCRIPTIVE_NO_NEW_TEST"))
 return pd.DataFrame(rows)
def sst_decomposition(long,sids):
 rows=[]
 for sid in sids:
  vals={}
  for side in ("internal","final"):
   g=long[(long.sid==sid)&(long.event_class==side)&(long.variable=="SST")]
   at={(z.center_kind,int(z.lag_h)):number(z.value) for z in g.itertuples()}
   v=np.array([at.get(("actual_track",0),np.nan),at.get(("actual_track",24),np.nan),
               at.get(("fixed_t0",0),np.nan),at.get(("fixed_t0",24),np.nan)])
   if np.isfinite(v).all() and not np.isclose(v[0],v[2],rtol=1e-10,atol=1e-10):raise ValueError("SST same t0 centre sources disagree")
   vals[side]={"along_total_change":v[1]-v[0],"fixed_location_change":v[3]-v[2],"displacement_difference":v[1]-v[3]} if np.isfinite(v).all() else {k:np.nan for k in ("along_total_change","fixed_location_change","displacement_difference")}
  for name in vals["internal"]:
   rows.append(pair_record(sid,"SST",name,vals["internal"][name],vals["final"][name],"degC","separate_fixed_and_along",
    family="SST_DECOMPOSITION_DESCRIPTIVE",definition="same four-source support; along total = fixed location + displacement"))
 return pd.DataFrame(rows)
def summarize_pairs(pairs,sids,out,B=2000,seed=510082026,inferential=False):
 out=Path(out);out.mkdir(parents=True,exist_ok=False)
 pairs=pairs.copy();pairs["target"]=pairs.variable+"|"+pairs.summary
 if pairs.duplicated(["sid","target"]).any():raise ValueError("Duplicate SID target")
 targets=list(dict.fromkeys(pairs.target));sidlist=list(sids);ix,w=bootstrap_weights(len(sidlist),B,seed)
 arrays={side:pairs.pivot(index="sid",columns="target",values=side).reindex(index=sidlist,columns=targets).to_numpy(float)
         for side in ("internal","final","difference")}
 all_draws={};all_est={};counts=None
 for side,a in arrays.items():
  # Both event means use EXACTLY the finite paired mask, including any summary-level support restriction.
  a[~np.isfinite(arrays["difference"])]=np.nan
  est,d,n=paired_mean_draws(a,w);all_draws[side]=d;all_est[side]=est;counts=n
 rows=[]
 meta=pairs.drop_duplicates("target").set_index("target")
 for j,t in enumerate(targets):
  z=meta.loc[t];d=summarize_draws(all_est["difference"][j],all_draws["difference"][:,j])
  row=dict(target=t,variable=z.variable,process_label=z.process_label,summary=z.summary,units=z.units,event_units=z.event_units,center_policy=z.center_policy,
   family=z.family,n_paired_sid=int(counts[j]),cohort_sid_count=len(sids),bootstrap_requested=B,
   **d,failed_draws=int(B-d["valid_draws"]),p_descriptive_only=np.nan,q=np.nan,contrast="internal minus final")
  for side in ("internal","final"):
   a=summarize_draws(all_est[side][j],all_draws[side][:,j])
   row.update({side+"_"+k:a[k] for k in ("estimate","lo","hi")})
  if not inferential or z.family not in ("P2_pre_and_change_18","P2_persistence_6"):row["p"]=np.nan
  rows.append(row)
 summary=pd.DataFrame(rows)
 if inferential:
  for family,m in (("P2_pre_and_change_18",18),("P2_persistence_6",6)):
   sel=summary.family.eq(family)
   if sel.sum()!=m:raise ValueError("Fixed family target count mismatch")
   summary.loc[sel,"q"]=bh(summary.loc[sel,"p"].to_numpy(),family_size=m)
 pairs.to_csv(out/"SID_paired_source.csv",index=False,encoding="utf-8-sig")
 summary.to_csv(out/"estimates_CI_P_q.csv",index=False,encoding="utf-8-sig")
 pd.DataFrame({"sid":sidlist}).to_csv(out/"bootstrap_SID_order.csv",index=False)
 np.savez_compressed(out/"bootstrap_draws.npz",SID=np.asarray(sidlist,dtype=str),target=np.asarray(targets,dtype=str),
  SID_draw_indices=ix,draw_multiplicity=w,internal_source=arrays["internal"],final_source=arrays["final"],
  difference_source=arrays["difference"],**{k+"_draws":v for k,v in all_draws.items()})
 jsave(out/"method.json",dict(created=stamp(),B=B,seed=seed,unit="SID; each draw carries both events and all lags",
  CI="direct target draw 2.5/97.5 percentiles",P="two-sided finite-draw sign-tail plus1; fixed18/6BH primary only",
  failure="No replacement/top-up; every nonfinite draw retained in NPZ; status table records counts",
  primary_persistence_population="same SID both events; >=3 common pre lags; complete common lawful0-24h interval support",
  inference=inferential,cohort_sid_count=len(sids),not_conditioned_on_48h_followup=True,
  descriptive_lag_CI="pointwise, not simultaneous; no per-lag significance testing"))
 return summary
def frozen_addendum():
 return dict(created_utc=stamp(),status="technical clarification frozen before real P1/P2 effects",authorization="root explicit approval",
  contrast="internal minus final",persistence_family=list(DIRECTIONS),
  SST="retains primary18 pre/change family and descriptive fixed/displacement decomposition; no persistence test",
  persistence_family_reason="thermal headroom already contains SST background; avoid redundant persistence endpoint, selected before outcomes",
  interval_rule="both sampled endpoints strictly worse than own supported premean; legal6h; no interpolation; crossing intervals separately counted",
  interval_interpretation="covered 6-h intervals adverse at both sampled endpoints; not continuously observed adverse conditions",
  pre_pairing="primary event premeans use same paired valid lags >=3/4; lag-specific comparisons use same observed lag on both events",
  primary_persistence="complete24h common supported lawful intervals only; partial covered/adverse hours retained descriptively, never extrapolated",
  primary_centers="fixed_t0 at negative lags, actual_track at nonnegative lags; separate all-along sensitivity",
  signed_ventilation="raw_signed_index and signed_estimable; retain negative values; higher relative value is a relative diagnostic, not proof of damage",
  sector="700hPa r/q/advection only; native derivative before rotation; fixed premean shear vector direction>=2m/s; weak events remain scalar",
  count_family="18 pre/change;6 persistence, fixed sizes with missing P retained; sensitivity/sector/lag no added significance families",
  bootstrap="2000 draws of original selected cohort SID, same draw IDs for all event sides/lags/processes; no complete48h gate")
def prepare_or_run(a):
 out=Path(a.out).resolve()
 if ROOT.resolve() not in out.parents:raise ValueError("Output outside new v5")
 if out.exists():raise ValueError("Output exists; no overwrite")
 receipt=read(a.receipt)
 if receipt.get("status")!="PASS_P1_ALL_REQUESTED_EXTRACTIONS_AND_SOURCES_VERIFIED":raise ValueError("Required complete source receipt not PASS")
 for key,path in (("requests",a.requests),("events",a.events),("cache_index",a.cache_index),("thermodynamics",a.thermodynamics)):
  rec=receipt["inputs"][key]
  if str(Path(rec["path"]).resolve())!=str(Path(path).resolve()) or rec["sha256"]!=sha(path):raise ValueError("Receipt mismatch "+key)
 spec=ROOT/"analysis_spec_v5.json";add=ROOT/"config/p1_p2_technical_clarification_v5.json"
 if not add.exists():raise ValueError("Technical definitions not frozen")
 s=read(spec)
 if s["shared"]["bootstrap"]["repeats"]!=2000 or s["P2"]["processes"]!=list(PROCESSES):raise ValueError("Analysis spec mismatch")
 long,sectors,angles,req,prov=build_sources(csvread(a.requests),csvread(a.events),csvread(a.cache_index),csvread(a.thermodynamics),
  pd.read_parquet(a.track),receipt.get("driver_sha256"))
 out.mkdir(parents=True)
 for name,df in [("process_sources",long),("sector_sources",sectors),("fixed_angle_sources",angles),("enriched_requests",req),("consumed_sources",prov)]:
  df.to_csv(out/(name+".csv"),index=False,encoding="utf-8-sig")
 events=csvread(a.events)
 for c in ("current117","candidate123"):events[c]=events[c].map(boolean)
 if a.run:
  for cohort,flag in (("current117","current117"),("candidate123","candidate123")):
   sid=sorted(events.loc[events[flag],"sid"].unique())
   for policy in ("fixed_pre_along_post","all_along"):
    pp,ss=scalar_pairs(long,sid,policy);label=cohort+"_"+policy
    summarize_pairs(pp,sid,out/label,2000,510082026,inferential=(cohort=="current117" and policy=="fixed_pre_along_post"))
    ss.to_csv(out/label/"interval_coverage_all_SID.csv",index=False,encoding="utf-8-sig")
    sec=sector_pairs(sectors,sid,policy)
    if len(sec):summarize_pairs(sec,sid,out/(label+"_sectors"),2000,510082026,False)
   summarize_pairs(sst_decomposition(long,sid),sid,out/(cohort+"_SST_decomposition"),2000,510082026,False)
 jsave(out/"execution_receipt.json",dict(started_utc=a.started,finished_utc=stamp(),argv=sys.argv,python=sys.executable,
  mode="REAL_P1_P2_STATISTICS" if a.run else "SOURCE_PREPARATION_ONLY",raw_ERA5_reads=0,code_sha256=sha(__file__),
  analysis_spec_sha256=sha(spec),clarification_sha256=sha(add),receipt_sha256=sha(a.receipt),
  actual_bootstrap_draws=2000 if a.run else 0,source_requests=len(req),scalar_rows=len(long),sector_rows=len(sectors)))
def main():
 p=argparse.ArgumentParser();p.add_argument("--freeze-technical",action="store_true");p.add_argument("--run",action="store_true")
 p.add_argument("--prepare-only",action="store_true");p.add_argument("--requests");p.add_argument("--events")
 p.add_argument("--cache-index");p.add_argument("--thermodynamics");p.add_argument("--receipt");p.add_argument("--out")
 p.add_argument("--track",default=str(ROOT.parent/"review_completion_v3_2/sources/normalized_track_sp_qc_v3_2.parquet"))
 a=p.parse_args();a.started=stamp()
 if a.freeze_technical:
  path=ROOT/"config/p1_p2_technical_clarification_v5.json"
  if path.exists():raise ValueError("Frozen technical file already exists; no overwrite")
  jsave(path,frozen_addendum());print(path);return
 if a.run==a.prepare_only:p.error("Select exactly one of --run or --prepare-only")
 for k in ("requests","events","cache_index","thermodynamics","receipt","out"):
  if not getattr(a,k):p.error("Required: --"+k.replace("_","-"))
 prepare_or_run(a)
if __name__=="__main__":main()

