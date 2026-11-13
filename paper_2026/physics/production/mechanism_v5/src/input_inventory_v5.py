from _release_paths import release_path as _release_path
"""Read-only ERA5 v5 inventory; all scientific arrays bounded to 3 x 3 x 3 samples."""
from pathlib import Path
import csv, json, hashlib, os, re, sys, traceback
from datetime import datetime, timezone
import numpy as np
import netCDF4
ROOT=Path(str(_release_path('mechanism_v5','')))
BASE=Path(str(_release_path('raw_era5','')))
V32=ROOT.parent/"review_completion_v3_2"
def stamp(): return datetime.now(timezone.utc).isoformat()
def conv(x):
 if isinstance(x,np.ndarray): return x.tolist()
 if isinstance(x,np.generic): return x.item()
 if isinstance(x,bytes): return x.decode(errors="replace")
 if isinstance(x,Path): return str(x)
 raise TypeError(type(x).__name__)
def dump(path,obj): path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=conv),encoding="utf-8")
def sha(b): return hashlib.sha256(b).hexdigest()
def attrs(obj): return {a:obj.getncattr(a) for a in obj.ncattrs()}
def identity(p):
 s=p.stat(); return dict(path=str(p),size=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,file_id=s.st_ino,device=s.st_dev)
def log(o):
 with (ROOT/"audit"/"input_audit_execution_v5.jsonl").open("a",encoding="utf-8") as f: f.write(json.dumps(dict(utc=stamp(),**o),ensure_ascii=False,default=conv)+"\n")
files=sorted(p for p in BASE.glob("*.nc") if not p.name.startswith("wp_detr_"))
allrows=[]; details=[]; issues=[]
oldmeta=json.loads((V32/"input_audit"/"raw_metadata_v3_2.json").read_text(encoding="utf-8-sig"))
started=stamp()
log(dict(event="start",python=sys.executable,netCDF4=netCDF4.__version__,pid=os.getpid(),files=len(files),scope="coordinates complete; 27 values per dynamic scalar variable; no full raw-file hash"))
for k,p in enumerate(files,1):
 r=dict(filename=p.name,path=str(p),full_sha256="NOT_RECOMPUTED",status="PENDING")
 d=dict(identity=identity(p))
 try:
  with netCDF4.Dataset(p,"r") as ds:
   d["global_attrs"]=attrs(ds); d["format"]=ds.file_format
   d["variables"]={n:dict(dimensions=v.dimensions,shape=v.shape,dtype=str(v.dtype),attrs=attrs(v),chunking=v.chunking()) for n,v in ds.variables.items()}
   timedim=next((n for n in ("time","valid_time") if n in ds.variables),None)
   if timedim:
    tv=ds[timedim]; rawt=np.asarray(tv[:]); dates=netCDF4.num2date(rawt,tv.units,getattr(tv,"calendar","standard"),only_use_cftime_datetimes=True)
    # 67k time coordinates only, never load a global data field.
    epoch=netCDF4.date2num(dates,"hours since 1980-01-01 00:00:00","proleptic_gregorian")
    d["time"]={"name":timedim,"count":len(rawt),"start_utc":str(dates[0]),"end_utc":str(dates[-1]),"hours_step_unique":np.unique(np.diff(epoch)).tolist(),"monotonic_unique":bool(np.all(np.diff(epoch)>0)),"hours_UTC":sorted(set(int(x.hour) for x in dates)),"sha256_raw":sha(rawt.tobytes())}
    dynamic=(len(rawt)>1)
    fulltime=bool(len(rawt)==67208 and epoch[0]==0 and np.all(np.diff(epoch)==6) and str(dates[-1])=="2025-12-31 18:00:00")
   else: dynamic=False;fulltime=True;d["time"]={}
   coords={}
   for name in ("latitude","longitude"):
    a=np.asarray(ds[name][:]); coords[name]=dict(count=len(a),first=float(a[0]),last=float(a[-1]),step_unique=np.unique(np.diff(a)).tolist(),sha256=sha(a.tobytes()))
   d["coordinates"]=coords
   globalgrid=coords["latitude"]["count"]==721 and coords["longitude"]["count"]==1440 and coords["latitude"]["first"]==90 and coords["latitude"]["last"]==-90 and coords["latitude"]["step_unique"]==[-.25] and coords["longitude"]["first"]==0 and coords["longitude"]["last"]==359.75 and coords["longitude"]["step_unique"]==[.25]
   variables=[n for n,v in ds.variables.items() if "latitude" in v.dimensions and "longitude" in v.dimensions]
   sam=[]
   for vn in variables:
    v=ds[vn]; checks=[(0,320,600),(len(ds.dimensions[timedim])//2 if dynamic else 0,280,1200),(len(ds.dimensions[timedim])-1 if dynamic else 0,440,320)]
    for ti,la,lo in checks:
     idx=tuple(slice(ti,ti+1) if name==timedim else slice(la,la+3) if name=="latitude" else slice(lo,lo+3) if name=="longitude" else slice(0,1) for name in v.dimensions)
     a=np.ma.asarray(v[idx]); data=np.ma.filled(a,np.nan)
     finite=np.isfinite(data)
     sam.append(dict(variable=vn,time_index=ti,lat_slice=[la,la+3],lon_slice=[lo,lo+3],shape=list(data.shape),n_valid=int(finite.sum()),n_total=data.size,finite_min=float(np.nanmin(data)) if finite.any() else None,finite_max=float(np.nanmax(data)) if finite.any() else None,decoded_values=data.tolist(),values_sha256=sha(np.asarray(data).tobytes())))
   d["bounded_samples"]=sam
   var=ds[variables[0]]; a=attrs(var)
   d["metadata_sha256"]=sha(json.dumps({x:d[x] for x in ("global_attrs","variables","time","coordinates")},sort_keys=True,default=conv).encode())
   r.update(variable=";".join(variables),units=a.get("units"),level_hpa=a.get("pressure_level_hpa",""),derivation=a.get("calculation_formula",""),size=d["identity"]["size"],mtime_ns=d["identity"]["mtime_ns"],file_id=d["identity"]["file_id"],metadata_sha256=d["metadata_sha256"],time_start=d["time"].get("start_utc"),time_end=d["time"].get("end_utc"),n_time=d["time"].get("count"),regular_6h_1980_2025=fulltime if dynamic else "STATIC",global_025_grid=globalgrid,sample_finite_count=sum(s["n_valid"] for s in sam),sample_count=sum(s["n_total"] for s in sam),source_run=d["global_attrs"].get("merge_run_id",""),step_type=a.get("GRIB_stepType",""),status="AVAILABLE_CHECKED" if globalgrid and (fulltime or not dynamic) else "CONFLICT")
   if r["status"]=="CONFLICT": issues.append(dict(filename=p.name,reason="time or grid differs",time=d["time"],coordinates=coords))
   if variables[0] in ("tp","sp"):
    before=oldmeta[variables[0]]["identity"];d["v3_2_identity_matches"]={key:d["identity"][key]==before[key] for key in before}
    r["baseline_identity_match"]=all(d["v3_2_identity_matches"].values())
    if not r["baseline_identity_match"]: issues.append(dict(filename=p.name,reason="v3_2 file identity differs",comparison=d["v3_2_identity_matches"]))
   else:r["baseline_identity_match"]="NO_OLD_FULL_IDENTITY_CLAIM"
   r["processing_note"]= "derived signed wind difference, not raw pressure-level wind" if r["derivation"] else "SP pressure mask before pressure-level diagnostics; original area denominator" if r["level_hpa"]!="" else "surface fields: do not apply 850 hPa subsurface mask"
   if variables[0] in ("slhf","sshf"):r["processing_note"]="User-confirmed ERA5 official download/time concatenation only; hourly accumulated J m-2 sampled six-hourly; upward W m-2 = -value/3600; legacy standard_name is not sign authority"
   if variables[0]=="tp":r["processing_note"]="one-hour accumulation ending valid UTC, sampled every 6h; mm h-1=value*1000; not six-hour/daily total"
 except Exception as e:
  r.update(status="READ_ERROR",error=repr(e));d["traceback"]=traceback.format_exc();issues.append(dict(filename=p.name,reason="READ_ERROR",error=repr(e)))
 d["row"]=r;allrows.append(r);details.append(d)
 dump(ROOT/"audit"/"raw_metadata_v5.partial.json",details)
 log(dict(event="file_complete",index=k,filename=p.name,status=r["status"]))
 print(f"{k}/{len(files)} {p.name}: {r['status']}",flush=True)
# Evidence files are small and hashed in this audit. Their raw data full checks are inherited.
evidence=[]
merge=BASE/"merge_sp_tp_1980_2025"/"run_20260925_203014"
for p in [merge/"sp_validation.json",merge/"tp_validation.json",merge/"state.json",V32/"input_audit"/"inherited_full_validation.json",V32/"input_audit"/"tp_accumulation_definition_v3_2.json",V32/"analysis_spec_v3_2.yaml"]:
 evidence.append(dict(path=str(p),exists=p.exists(),size=p.stat().st_size if p.exists() else None,sha256=sha(p.read_bytes()) if p.exists() else None))
keys=list(dict.fromkeys(k for r in allrows for k in r))
with (ROOT/"input_inventory_v5.csv").open("w",newline="",encoding="utf-8-sig") as f:
 w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(allrows)
dump(ROOT/"audit"/"raw_metadata_v5.json",details)
dump(ROOT/"audit"/"input_evidence_bindings_v5.json",evidence)
summary=dict(started_utc=started,finished_utc=stamp(),python=sys.executable,netcdf4_version=netCDF4.__version__,files=len(files),available=sum(x["status"]=="AVAILABLE_CHECKED" for x in allrows),issues=issues,full_raw_hashes_recomputed=False,scope="complete time/grid coordinates and metadata; 3 deterministic 3x3 reads per file; no global scientific array",inventory_path=str(ROOT/"input_inventory_v5.csv"))
dump(ROOT/"audit"/"input_inventory_execution_summary.json",summary);log(dict(event="finished",**summary));print(json.dumps(summary,ensure_ascii=False),flush=True)
