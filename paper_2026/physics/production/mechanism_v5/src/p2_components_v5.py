"""P2 physical components on exactly the accepted P1 target support and saved SID draws.
No raw ERA5, no refit, no new draw, and no new P/q family.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,csv,json,hashlib,sys,os
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def stamp():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):
 with Path(p).open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def num(x):
 try:return float(x)
 except (TypeError,ValueError):return np.nan
def savecsv(p,rr):
 keys=list(dict.fromkeys(k for r in rr for k in r))
 with Path(p).open("w",encoding="utf-8-sig",newline="") as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rr)
def jread(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def jwrite(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding="utf-8")
FIELDS={
 "PI_headroom":[("pi_vmax_ms","m s-1"),("wind_ms","m s-1"),("thermal_headroom_ms","m s-1")],
 "Tang_Emanuel_ventilation":[("shear_200_850_ms","m s-1"),("pi_vmax_ms","m s-1"),
  ("entropy_deficit_jkgk","J kg-1 K-1"),("sea_air_entropy_difference_jkgk","J kg-1 K-1"),
  ("chi600","1"),("raw_signed_index","1"),("core600_saturated_entropy_jkgk","J kg-1 K-1"),
  ("environment600_entropy_jkgk","J kg-1 K-1")]}
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--results",type=Path,required=True)
 ap.add_argument("--thermo",type=Path,required=True);ap.add_argument("--out",type=Path,required=True);a=ap.parse_args()
 started=stamp()
 if a.out.exists():raise ValueError("New output directory required")
 main=a.results/"current117_fixed_pre_along_post"
 snapshot={r["request_id"]:r for r in rows(a.thermo)}
 pairs=[r for r in rows(main/"SID_paired_source.csv") if r["variable"] in FIELDS and r["summary"]!="adverse_endpoint_covered_hours_0_24"]
 with np.load(main/"bootstrap_draws.npz",allow_pickle=False) as z:
  saved={k:z[k].copy() for k in z.files}
 sids=saved["SID"];six={s:i for i,s in enumerate(sids)};targetix={s:i for i,s in enumerate(saved["target"])}
 checks=0;source={};ledger=[];maxerr={}
 def eq(label,a,b):
  nonlocal checks
  aa=np.asarray(a,float);bb=np.asarray(b,float)
  np.testing.assert_allclose(aa,bb,rtol=1e-10,atol=1e-10,equal_nan=True,err_msg=label)
  finite=np.isfinite(aa)&np.isfinite(bb);err=float(np.max(np.abs(aa[finite]-bb[finite]))) if finite.any() else 0.
  maxerr[label]=max(maxerr.get(label,0.),err);checks+=1
 for pair in pairs:
  if not np.isfinite(num(pair["difference"])):continue
  for rid in pair["source_request_ids"].split(";"):
   if rid in source:continue
   row=snapshot[rid];cp=Path(row["derived_complete_path"]);c=jread(cp)
   if sha(cp)!=row["derived_complete_sha256"]:raise ValueError("Derived COMPLETE identity conflict")
   h={x["name"]:x["sha256"] for x in c["outputs"]};dr=cp.parent/"derived.json"
   if sha(dr)!=h["derived.json"]:raise ValueError("Derived value identity conflict")
   r=jread(dr)
   if r["request_id"]!=rid:raise ValueError("Derived request conflict")
   r["wind_ms"]=num(r["wind_for_headroom_kt"])*1852/3600
   if np.isfinite(num(r["thermal_headroom_ms"])):
    eq("request_headroom_identity",num(r["thermal_headroom_ms"]),num(r["pi_vmax_ms"])-r["wind_ms"])
   if bool(r["signed_estimable"]):
    eq("request_entropy_deficit_identity",r["entropy_deficit_jkgk"],r["core600_saturated_entropy_jkgk"]-r["environment600_entropy_jkgk"])
    eq("request_chi_identity",r["chi600"],r["entropy_deficit_jkgk"]/r["sea_air_entropy_difference_jkgk"])
    eq("request_signed_VI_identity",r["raw_signed_index"],r["shear_200_850_ms"]*r["chi600"]/r["pi_vmax_ms"])
   source[rid]=r
   ledger.append(dict(request_id=rid,sid=r["sid"],event_id=r["event_id"],event_class=r["event_class"],UTC=r["UTC"],
    lag_h=r["lag_h"],center_kind=r["center_kind"],wind_source_UTC=r["wind_source_UTC"],
    wind_source_policy=r["wind_source_policy"],derived_complete=str(cp),derived_complete_sha256=sha(cp),
    **{field:r.get(field) for field in dict(FIELDS["PI_headroom"]+FIELDS["Tang_Emanuel_ventilation"])}))
 output=[];keys=[];meta={}
 for pair in pairs:
  for field,unit in FIELDS[pair["variable"]]:
   key=pair["target"]+"|component:"+field
   if key not in meta:keys.append(key);meta[key]=(pair["target"],pair["variable"],pair["summary"],field,unit)
   vals=[np.nan,np.nan];eligible=np.isfinite(num(pair["difference"]))
   if eligible:
    rr=[source[rid] for rid in pair["source_request_ids"].split(";")]
    for i,side in enumerate(("internal","final")):
     sel=[r for r in rr if r["event_class"]==side]
     if not sel or any(r["sid"]!=pair["sid"] for r in sel):raise ValueError("Shared support event identity conflict")
     yy=np.array([num(r.get(field)) for r in sel])
     if not np.isfinite(yy).all():raise ValueError("Component missing on accepted target support")
     if pair["summary"]=="pre_mean":
      if len(sel)<3:raise ValueError("Pre support below frozen minimum")
      vals[i]=float(yy.mean())
     elif pair["summary"]=="change_t0_to24h":
      at={int(r["lag_h"]):num(r[field]) for r in sel}
      if set(at)!={0,24}:raise ValueError("Change requires original endpoints")
      vals[i]=at[24]-at[0]
     elif pair["summary"].startswith("lag_"):
      if len(sel)!=1:raise ValueError("Lag requires exactly one request per event")
      vals[i]=yy[0]
     else:raise ValueError("Unexpected estimand")
   output.append(dict(sid=pair["sid"],parent_target=pair["target"],component_target=key,component=field,
    summary=pair["summary"],units=unit,internal=vals[0],final=vals[1],difference=vals[0]-vals[1],
    eligible_same_parent_support=eligible,source_request_ids=pair["source_request_ids"],
    support_policy="exact parent target finite paired mask; no added sample selection",P_q="NOT_TESTED_DESCRIPTIVE_COMPONENT"))
 kix={k:i for i,k in enumerate(keys)};shape=(len(sids),len(keys))
 arrays={x:np.full(shape,np.nan) for x in ("internal","final","difference")}
 for r in output:
  for side in arrays:arrays[side][six[r["sid"]],kix[r["component_target"]]]=r[side]
 weights=saved["draw_multiplicity"].astype(float);draws={};est={};ns={}
 for side,x in arrays.items():
  good=np.isfinite(x);n=good.sum(0);den=weights@good.astype(float)
  est[side]=np.divide(np.where(good,x,0).sum(0),n,out=np.full(n.shape,np.nan),where=n>=2)
  draws[side]=np.divide(weights@np.where(good,x,0),den,out=np.full(den.shape,np.nan),where=den>=2)
  draws[side][:,n<2]=np.nan;ns[side]=n
 for key,(parent,variable,summary,field,unit) in meta.items():
  j=kix[key];original=targetix[parent]
  for side in arrays:
   eq("parent_support_"+side,np.isfinite(arrays[side][:,j]).astype(int),np.isfinite(saved["difference_source"][:,original]).astype(int))
  if field in ("thermal_headroom_ms","raw_signed_index"):
   for side in arrays:
    eq("parent_source_"+side,arrays[side][:,j],saved[side+"_source"][:,original])
    eq("parent_draw_"+side,draws[side][:,j],saved[side+"_draws"][:,original])
 for summary in sorted(set(r["summary"] for r in pairs if r["variable"]=="PI_headroom")):
  js=[kix["PI_headroom|"+summary+"|component:"+x] for x in ("thermal_headroom_ms","pi_vmax_ms","wind_ms")]
  for side in arrays:
   eq("SID_headroom_linear_identity_"+side,arrays[side][:,js[0]],arrays[side][:,js[1]]-arrays[side][:,js[2]])
   eq("mean_headroom_linear_identity_"+side,est[side][js[0]],est[side][js[1]]-est[side][js[2]])
   eq("draw_headroom_linear_identity_"+side,draws[side][:,js[0]],draws[side][:,js[1]]-draws[side][:,js[2]])
 result=[]
 for j,key in enumerate(keys):
  parent,var,summary,field,unit=meta[key]
  rec=dict(parent_target=parent,variable=var,summary=summary,component=field,units=unit,n_paired_sid=int(ns["difference"][j]),
    attempted_draws=len(weights),CI_method="2.5/97.5 percentiles of reused original SID multiplicities",
    inference="DESCRIPTIVE_COMPONENT_CI_ONLY_NO_NEW_P_OR_Q",support="exact parent target paired times and SIDs")
  for side in arrays:
   dd=draws[side][:,j];valid=dd[np.isfinite(dd)];lo,hi=np.quantile(valid,[.025,.975]) if len(valid) else (np.nan,np.nan)
   rec.update({side+"_estimate":float(est[side][j]),side+"_lo":float(lo),side+"_hi":float(hi),
    side+"_valid_draws":len(valid),side+"_failed_draws":len(dd)-len(valid)})
  result.append(rec)
 a.out.mkdir(parents=True)
 savecsv(a.out/"request_components_source.csv",ledger);savecsv(a.out/"SID_component_paired_source.csv",output)
 savecsv(a.out/"component_estimates_CI.csv",result)
 shear=[r for r in rows(main/"estimates_CI_P_q.csv") if r["variable"] in ("shear200_850","shear500_850","shear200_500")]
 savecsv(a.out/"accepted_three_layer_shear_reference.csv",shear)
 np.savez_compressed(a.out/"components_saved_SID_draws.npz",SID=sids,target=np.asarray(keys,str),
  SID_draw_indices=saved["SID_draw_indices"],draw_multiplicity=saved["draw_multiplicity"],
  **{k+"_source":v for k,v in arrays.items()},**{k+"_draws":v for k,v in draws.items()})
 receipt=dict(status="COMPONENTS_COMPUTED_AND_IDENTITIES_VERIFIED",started_utc=started,finished_utc=stamp(),
  argv=[sys.executable]+sys.argv,exit_code=0,checks=checks,max_absolute_identity_errors=maxerr,
  requests=len(source),component_targets=len(keys),paired_source_rows=len(output),new_resamples=0,raw_ERA5_reads=0,
  CI_scope="pointwise descriptive component summaries on each existing parent estimand support, not nonlinear attribution",
  scientific_caution="headroom includes contemporaneous wind; aggregate VI is not a product or ratio of aggregate components",
  inputs={str(x):sha(x) for x in (a.thermo,main/"SID_paired_source.csv",main/"bootstrap_draws.npz")},
  outputs=[dict(name=x.name,sha256=sha(x)) for x in a.out.iterdir() if x.is_file()],code_sha256=sha(__file__))
 jwrite(a.out/"verification_receipt.json",receipt)
 for r in result:
  if r["summary"] in ("pre_mean","change_t0_to24h"):
   print(json.dumps({k:r[k] for k in ("variable","summary","component","n_paired_sid","difference_estimate","difference_lo","difference_hi")}))
 print(json.dumps(dict(status=receipt["status"],checks=checks,output=str(a.out))))
if __name__=="__main__":main()

