"""v5.1 bounded single-worker extraction; default one-request pilot, --run-all is explicit.
No scientific fitting. Original ERA5/baselines are read-only. Checkpoints/cache live only under v5.
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime,timezone,timedelta
from dataclasses import replace
from contextlib import contextmanager
import msvcrt
import argparse,csv,json,hashlib,os,sys,time,traceback
for _thread_env in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
 os.environ[_thread_env]="1"
import numpy as np
from native_fields_v5 import NativeFieldReader,NativeBlock,utc_datetime,area_weighted_support,spherical_gradient,spherical_divergence,storm_relative_advection,vertical_advection_700,shear_components
ROOT=Path(__file__).resolve().parents[1]
LEVELS=np.array([1000,925,850,700,600,500,400,300,250,200,150,100,50],dtype=int)
CORE=("sst","u200","v200","u850","v850","r500","r700","d200","w500","vo850")
FULL=tuple(dict.fromkeys(CORE+("u500","v500","u700","v700","q600","q700","q850","w700","d850","z200","z500","msl","slhf","sshf","tp","tcwv")))
MAP=("q700","r700","u700","v700","z200","u200","v200","u850","v850")
def stamp():return datetime.now(timezone.utc).isoformat()
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,float) and not np.isfinite(x):return None
 if isinstance(x,Path):return str(x)
 return x
def canon(x):return json.dumps(clean(x),sort_keys=True,separators=(",",":"),ensure_ascii=False)
def digest(x):return hashlib.sha256(canon(x).encode()).hexdigest()
def filehash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic_json(path,x):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_name(path.name+f".{os.getpid()}.tmp")
 tmp.write_text(json.dumps(clean(x),ensure_ascii=False,indent=2),encoding="utf-8")
 os.replace(tmp,path)
def atomic_npz(path,**arrays):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_name(path.name+f".{os.getpid()}.tmp")
 with tmp.open("wb") as f:np.savez_compressed(f,**arrays)
 os.replace(tmp,path)
def atomic_csv(path,rows,fields=None):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 fields=fields or list(dict.fromkeys(k for r in rows for k in r))
 tmp=path.with_name(path.name+f".{os.getpid()}.tmp")
 with tmp.open("w",newline="",encoding="utf-8-sig") as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 os.replace(tmp,path)
def boolean(x):
 if isinstance(x,(bool,np.bool_)):return bool(x)
 if str(x).strip().lower() in ("true","1","yes"):return True
 if str(x).strip().lower() in ("false","0","no",""):return False
 raise ValueError(f"invalid explicit boolean: {x!r}")
def number(x):
 try:return float(x)
 except (TypeError,ValueError):return np.nan
def row_spec(row):
 vs=row.get("variable_set","full_process").strip().lower()
 if vs not in ("full_process","baseline_exposure","sst_for_displacement"):raise ValueError(f"undefined variable_set {vs}")
 prof=boolean(row.get("need_profile",False));maps=boolean(row.get("need_map",False))
 if vs!="full_process" and (prof or maps):raise ValueError("limited variable_set cannot silently trigger profile/map")
 lat=number(row["lat"])+0.0;lon=number(row["lon"])%360+0.0
 if not np.isfinite(lat+lon) or not(-90<lat<90):raise ValueError("missing/invalid actual centre, no extrapolation")
 utc=utc_datetime(row["UTC"]).isoformat()+"Z"
 return dict(UTC=utc,lat=lat,lon=lon,variable_set=vs,need_profile=prof,need_map=maps,
             motion_u_ms=number(row.get("motion_u_ms")),motion_v_ms=number(row.get("motion_v_ms")))
@contextmanager
def cache_item_lock(path,timeout_seconds=300):
 # OS byte-range locking serialises main requests AND overlapping t+/-6 cache keys.
 # The stable lock file remains; the OS releases its byte lock if a process exits.
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 f=path.open("a+b")
 if path.stat().st_size==0:f.write(b"0");f.flush()
 tick=time.perf_counter()
 try:
  while True:
   try:
    f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1);break
   except OSError:
    if time.perf_counter()-tick>timeout_seconds:raise TimeoutError(f"active cache item lock: {path}")
    time.sleep(.2)
  try:yield
  finally:
   f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
 finally:f.close()

class ExtractionCache:
 def __init__(self,reader,root,spec):
  self.reader=reader;self.root=root;self.spec=spec;self.new_raw_reads=0;self.cache_hits=0;self.refs={}
  self.reader_sha=filehash(Path(__file__).with_name("native_fields_v5.py"));self._source_checked=set()
 def geometry(self,utc,lat,lon,radius):
  obj=dict(UTC=utc_datetime(utc).isoformat()+"Z",latitude=float(lat),longitude=float(lon)%360,radius_km=float(radius),halo=1,reader_sha=self.reader_sha)
  return digest(obj)[:20],obj
 def verify_source_identity(self,key):
  if key in self._source_checked:return
  cfg=self.reader.map[key];expected=cfg.get("source_identity")
  if expected:
   st=Path(cfg["path"]).stat()
   current=dict(size=st.st_size,mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns,file_id=st.st_ino,device=st.st_dev)
   if any(current[n]!=expected[n] for n in current):raise ValueError(f"raw source identity changed since audit: {key}")
  self._source_checked.add(key)
 def reference(self,utc,lat,lon,radius):
  key,obj=self.geometry(utc,lat,lon,radius)
  self.verify_source_identity("sp")
  if key in self.refs:return self.refs[key]
  with cache_item_lock(self.root/key/"geometry_access.lock"):
   return self._reference_unlocked(utc,lat,lon,radius)
 def _reference_unlocked(self,utc,lat,lon,radius):
  key,obj=self.geometry(utc,lat,lon,radius)
  if key in self.refs:return self.refs[key]
  folder=self.root/key;folder.mkdir(parents=True,exist_ok=True);meta=folder/"geometry.json";p=folder/"sp.npz"
  source=self.reader.map["sp"];binding=dict(geometry=obj,sp_metadata_sha=source.get("metadata_sha256"),sp_identity=source.get("source_identity"))
  if p.exists():
   with np.load(p,allow_pickle=False) as z:
    vals={k:z[k].copy() for k in z.files}
   if json.loads(str(vals["binding_json"].item()))!=clean(binding):raise ValueError(f"cache geometry/source binding conflict {key}")
   if not meta.exists():
    atomic_json(meta,dict(binding=binding,metadata_recovered_from_atomic_npz=True,recovered_utc=stamp(),npz_sha256=filehash(p)))
   b=NativeBlock("sp","sp",obj["UTC"],"Pa",None,vals["latitude"],vals["longitude"],vals["latitude_indices"],vals["longitude_indices"],
      vals["sp"],vals["sp"],np.isfinite(vals["sp"]),vals["sp"],vals["area"],vals["distance"],vals["core"],lat,lon,radius,1,
      dict(cache=key,denominator="original geometric cells",source_identity=source.get("source_identity")))
   self.cache_hits+=1
  elif p.exists() or meta.exists():
   raise ValueError(f"incomplete cache geometry {key}; retained for diagnosis, no implicit fallback")
  else:
   b=self.reader.read_box("sp",utc,lat,lon,radius,halo=1,apply_sp=False);b.sp_pa=b.raw
   atomic_npz(p,binding_json=np.array(canon(binding)),sp=b.raw,latitude=b.latitude,longitude=b.longitude,latitude_indices=b.latitude_indices,longitude_indices=b.longitude_indices,area=b.cell_area_m2,distance=b.distance_km,core=b.core_mask)
   atomic_json(meta,dict(binding=binding,created_utc=stamp(),npz_sha256=filehash(p)))
   self.new_raw_reads+=1
  if len(self.refs)>=8:self.refs.pop(next(iter(self.refs)))
  self.refs[key]=b
  return b
 def get(self,key,utc,lat,lon,radius=800.):
  geo,_=self.geometry(utc,lat,lon,radius);self.verify_source_identity(key)
  with cache_item_lock(self.root/geo/(key+"_access.lock")):
   return self._get_unlocked(key,utc,lat,lon,radius)
 def _get_unlocked(self,key,utc,lat,lon,radius=800.):
  b=self.reference(utc,lat,lon,radius);geo,_=self.geometry(utc,lat,lon,radius)
  if key=="sp":return b
  p=self.root/geo/(key+".npz");j=self.root/geo/(key+".json")
  cfg=self.reader.map[key];binding=dict(key=key,source_identity=cfg.get("source_identity"),metadata_sha=cfg.get("metadata_sha256"),geometry=geo,
                                      mask="same native cell/time SP>=pressurePa" if cfg.get("pressure_level_hpa") is not None else "no SP layer mask")
  if p.exists():
   with np.load(p,allow_pickle=False) as z:
    if json.loads(str(z["binding_json"].item()))!=clean(binding):raise ValueError(f"field cache binding conflict {geo}/{key}")
    raw=z["raw"].copy();value=z["values"].copy();valid=z["support_mask"].copy()
   if not j.exists():
    atomic_json(j,dict(binding=binding,metadata_recovered_from_atomic_npz=True,recovered_utc=stamp(),npz_sha256=filehash(p)))
   self.cache_hits+=1
   return replace(b,key=key,variable=cfg["variable"],units=cfg["units"],pressure_level_pa=None if cfg.get("pressure_level_hpa") is None else cfg["pressure_level_hpa"]*100,
     raw=raw,values=value,support_mask=valid,sp_pa=b.sp_pa if cfg.get("pressure_level_hpa") is not None else None,metadata=dict(cache=str(p),**binding))
  if p.exists() or j.exists():raise ValueError(f"incomplete cached field {geo}/{key}; retained for explicit recovery")
  v=self.reader.read_aligned(key,utc,b)
  atomic_npz(p,binding_json=np.array(canon(binding)),raw=v.raw,values=v.values,support_mask=v.support_mask)
  atomic_json(j,dict(binding=binding,created_utc=stamp(),npz_sha256=filehash(p)))
  self.new_raw_reads+=1
  return v
class OceanMasks:
 def __init__(self,shape_path):
  self.shape_path=Path(shape_path)
  needed=[self.shape_path.with_suffix(x) for x in (".shp",".shx",".dbf",".prj")]
  if not all(p.is_file() for p in needed):raise FileNotFoundError("local NaturalEarth10m physical-land files missing; no automatic download/fallback")
  import cartopy.io.shapereader as shpreader
  from shapely.ops import unary_union
  self.land=unary_union(list(shpreader.Reader(str(self.shape_path)).geometries()))
  self.identities={str(p):filehash(p) for p in needed};self.cache={}
 def natural_earth_ocean(self,block):
  # Small LRU by native indices; same geometry at different UTC shares a mask.
  k=digest([block.latitude_indices.tolist(),block.longitude_indices.tolist()])
  if k not in self.cache:
   from shapely import contains_xy
   la,lo=np.meshgrid(block.latitude,block.longitude,indexing="ij")
   self.cache[k]=~np.asarray(contains_xy(self.land,(lo+180)%360-180,la),dtype=bool)
   if len(self.cache)>32:self.cache.pop(next(iter(self.cache)))
  return self.cache[k]
def subset_to(block,reference):
 # Both are native grids; integer index selection, never interpolation.
 li={int(x):i for i,x in enumerate(block.latitude_indices)};lj={int(x):i for i,x in enumerate(block.longitude_indices)}
 ii=np.array([li[int(x)] for x in reference.latitude_indices]);jj=np.array([lj[int(x)] for x in reference.longitude_indices])
 take=lambda a: a[np.ix_(ii,jj)] if a is not None else None
 return replace(reference,key=block.key,variable=block.variable,units=block.units,pressure_level_pa=block.pressure_level_pa,
                raw=take(block.raw),values=take(block.values),support_mask=take(block.support_mask),sp_pa=take(block.sp_pa),metadata=dict(block.metadata,native_subset=True))
def extract_one(row,cache,masks,out,semantics):
 spec=row_spec(row);reqid=str(row["request_id"]);binding=dict(request=row,normalised=spec,semantics_sha=digest(semantics),driver_sha=filehash(__file__))
 rid=digest(binding)[:20];folder=out/rid;folder.mkdir(parents=True,exist_ok=True);done=folder/"COMPLETE.json"
 if done.exists():
  prior=json.loads(done.read_text(encoding="utf-8"))
  if prior.get("binding")!=clean(binding):raise ValueError("request checkpoint binding conflict")
  for item in prior["outputs"]:
   p=folder/item["name"]
   if not p.exists() or filehash(p)!=item["sha256"]:raise ValueError(f"request checkpoint artifact mismatch {p}")
  return dict(request_id=reqid,request_hash=rid,status="REUSED_VERIFIED_CHECKPOINT",folder=str(folder),elapsed_seconds=0)
 started=stamp();tick=time.perf_counter();nr0=cache.new_raw_reads;ch0=cache.cache_hits
 atomic_json(folder/"STARTED.json",dict(binding=binding,started_utc=started,pid=os.getpid()))
 utc,lat,lon=spec["UTC"],spec["lat"],spec["lon"]
 base=cache.reference(utc,lat,lon,800.)
 fields={};mapfields={}
 wanted=FULL if spec["variable_set"]=="full_process" else CORE if spec["variable_set"]=="baseline_exposure" else ("sst",)
 if spec["need_map"]:
  for key in MAP:
   mapfields[key]=cache.get(key,utc,lat,lon,1500.)
   if key in wanted:fields[key]=subset_to(mapfields[key],base)
 for key in wanted:
  if key not in fields:fields[key]=cache.get(key,utc,lat,lon,800.)
 ring=(base.distance_km>=200)&(base.distance_km<800)
 ocean=masks.natural_earth_ocean(base)
 rows=[];wide={k:row.get(k,"") for k in ("request_id","module","sid","event_id","event_class","UTC","lat","lon","center_kind","lag_h")}
 wide.update(request_hash=rid,variable_set=spec["variable_set"],need_profile=spec["need_profile"],need_map=spec["need_map"])
 def add(name,a,units,domain=ring,domain_name="original_200_800km_annulus",note=""):
  st=area_weighted_support(a,base.cell_area_m2,domain,min_fraction=.9)
  rec=dict(request_id=reqid,request_hash=rid,variable=name,units=units,value=st["mean"],**{k:v for k,v in st.items() if k!="mean"},domain=domain_name,
           reason="PASS" if st["support_pass"] else "NO_ORIGINAL_DOMAIN" if st["original_area_m2"]<=0 else "BELOW_90PCT_ORIGINAL_AREA",note=note)
  rows.append(rec);wide[name]=st["mean"];wide[name+"__valid_fraction"]=st["valid_fraction"];wide[name+"__support_pass"]=st["support_pass"]
  return st
 # Fields are reported in raw physical units first. SST and flux use separately named masks.
 for key,f in fields.items():
  if key=="sst":
   add("sst_K",f.values,"K",ring&ocean,"NaturalEarth10m_original_ocean_annulus")
   add("sst_C",f.values-273.15,"degC",ring&ocean,"NaturalEarth10m_original_ocean_annulus")
  elif key not in ("slhf","sshf"):add(key,f.values,f.units)
 oceanfrac=float(base.cell_area_m2[ring&ocean].sum()/base.cell_area_m2[ring].sum()) if ring.any() else np.nan
 wide["sst_NaturalEarth10m_ocean_geometric_fraction"]=oceanfrac
 if all(k in fields for k in ("u200","v200","u850","v850")):
  for name,hi,lo in [("shear_200_850",200,850),("shear_500_850",500,850),("shear_200_500",200,500)]:
   ks=[f"u{hi}",f"v{hi}",f"u{lo}",f"v{lo}"]
   if not all(k in fields for k in ks):continue
   arr=[fields[k].values for k in ks];joint=np.isfinite(np.stack(arr)).all(axis=0)
   du,dv,mag=shear_components(*arr);du=np.where(joint,du,np.nan);dv=np.where(joint,dv,np.nan)
   sdu=add(name+"_u_ms",du,"m s-1");sdv=add(name+"_v_ms",dv,"m s-1")
   add(name+"_mean_point_magnitude_ms",np.where(joint,mag,np.nan),"m s-1",note="mean of gridpoint magnitudes; distinct from annular vector magnitude")
   wide[name+"_vector_magnitude_ms"]=float(np.hypot(sdu["mean"],sdv["mean"]))
   rows.append(dict(rows[-1],variable=name+"_vector_magnitude_ms",value=wide[name+"_vector_magnitude_ms"],raw_supported_mean=float(np.hypot(sdu["raw_supported_mean"],sdv["raw_supported_mean"])),note="magnitude of jointly supported annular mean upper-minus-lower vector"))
 if all(k in fields for k in ("r500","r700")):
  add("mid_rh_pct",.5*(fields["r500"].values+fields["r700"].values),"%",note="same-cell joint500/700hPa arithmetic mean then original-area support")
 if "d200" in fields:add("d200_1e5_s",fields["d200"].values*1e5,"1e-5 s-1")
 if "w500" in fields:add("minus_omega500_Pa_s",-fields["w500"].values,"Pa s-1")
 if "vo850" in fields:add("cyclonic_vo850_1e5_s",np.sign(lat)*fields["vo850"].values*1e5,"1e-5 s-1")
 if spec["variable_set"]=="full_process":
  lsm=cache.get("lsm",utc,lat,lon,800.);sea=(lsm.values<.5)&np.isfinite(lsm.values)
  wide["flux_ERA5lsm_ocean_geometric_fraction"]=float(base.cell_area_m2[ring&sea].sum()/base.cell_area_m2[ring].sum())
  for key in ("slhf","sshf"):
   add(key+"_J_m2",fields[key].values,"J m-2",ring&sea,"ERA5_lsm_lt0p5_original_ocean_annulus",note="one-hour accumulated value sampled every6h")
   add(key+"_upward_W_m2",-fields[key].values/3600.,"W m-2",ring&sea,"ERA5_lsm_lt0p5_original_ocean_annulus",note="minus ERA5 accumulatedJm-2 /3600s; not6h accumulation")
   add(key+"_upward_W_m2_all_annulus",-fields[key].values/3600.,"W m-2",note="descriptive full-annulus counterpart, distinct ocean mask")
  add("tp_sampled_1h_mm_h",fields["tp"].values*1000,"mm h-1",note="onehour accumulation endingUTC, sampled6hourly; not a continuous6h total")
  add("low_level_convergence_1e5_s",-fields["d850"].values*1e5,"1e-5 s-1")
  q=fields["q700"];u=fields["u700"].values;v=fields["v700"].values
  gx,gy=spherical_gradient(q.values,base.latitude,base.longitude)
  cu,cv=spec["motion_u_ms"],spec["motion_v_ms"]
  h=storm_relative_advection(q.values,u,v,cu,cv,base.latitude,base.longitude)
  vert=vertical_advection_700(fields["q600"].values,q.values,fields["q850"].values,fields["w700"].values)
  div=spherical_divergence(u,v,base.latitude,base.longitude)
  add("q700_dq_dx",gx,"kg kg-1 m-1");add("q700_dq_dy",gy,"kg kg-1 m-1")
  add("q700_horizontal_storm_relative_kgkg_s",h,"kg kg-1 s-1")
  add("q700_horizontal_storm_relative_gkg_day",h*1000*86400,"g kg-1 day-1")
  add("q700_vertical_advection_kgkg_s",vert,"kg kg-1 s-1",note="nonuniform600/700/850hPa derivative at700hPa; pressurePa")
  add("div700_s",div,"s-1")
  temporal_error=None
  try:
   before=(utc_datetime(utc)-timedelta(hours=6)).isoformat()+"Z";after=(utc_datetime(utc)+timedelta(hours=6)).isoformat()+"Z"
   qm=cache.get("q700",before,lat,lon,800.);qp=cache.get("q700",after,lat,lon,800.)
   # Same actual physical native cells; temporal SP masks are evaluated at their ownUTC.
   if not(np.array_equal(qm.latitude_indices,base.latitude_indices) and np.array_equal(qm.longitude_indices,base.longitude_indices) and np.array_equal(qp.latitude_indices,base.latitude_indices) and np.array_equal(qp.longitude_indices,base.longitude_indices)):raise ValueError("temporal native coordinates differ")
   dt=np.where(np.isfinite(q.values),(qp.values-qm.values)/(12*3600.),np.nan)
  except KeyError as e:
   dt=np.full_like(q.values,np.nan);temporal_error=str(e)
  moving=dt+cu*gx+cv*gy
  residual=moving-h-vert
  add("q700_fixed_location_tendency_kgkg_s",dt,"kg kg-1 s-1",note="centered12h difference of t±6 native values at identical physicalcoordinates")
  add("q700_moving_frame_tendency_kgkg_s",moving,"kg kg-1 s-1",note="fixedlocation tendency+c·native gradient; no future track controls")
  add("q700_unresolved_residual_kgkg_s",residual,"kg kg-1 s-1",note="unresolved diagnostic remainder, not observed microphysics/inner-core ventilation")
  joint=np.isfinite(np.stack([moving,h,vert,residual])).all(axis=0)
  for name,a in [("moving_tendency",moving),("horizontal_advection",h),("vertical_advection",vert),("unresolved_residual",residual)]:
   add("q_budget_common_"+name+"_kgkg_s",np.where(joint,a,np.nan),"kg kg-1 s-1",note="all4terms share exactly same native cells and original-annulus support")
  wide["q_temporal_failure_reason"]=temporal_error or ""
  wide["motion_valid"]=bool(np.isfinite(cu) and np.isfinite(cv))
  # Preserve derivative fields for selected P1 maps on800km native domain; main map fields remain1500km.
  if spec["need_map"]:atomic_npz(folder/"native_process_800km.npz",latitude=base.latitude,longitude=base.longitude,q700_horizontal=h,q700_vertical=vert,
    q700_fixed_tendency=dt,q700_moving_tendency=moving,q700_residual=residual,budget_joint_support=joint,cell_area_m2=base.cell_area_m2,annulus=ring)
 outputs=[]
 if spec["need_profile"]:
  pr=[];T=[];Q=[];tf=[];qf=[];jf=[];profile_fields={}
  for lev in LEVELS:
   blocks=[]
   for prefix in ("t","q"):
    key=f"{prefix}{lev}";block=fields.get(key)
    if block is None:block=cache.get(key,utc,lat,lon,800.)
    blocks.append(block)
    if int(lev)==600:profile_fields[prefix]=block
   tb,qb=blocks
   tj=area_weighted_support(tb.values,base.cell_area_m2,ring,min_fraction=.9)
   qj=area_weighted_support(qb.values,base.cell_area_m2,ring,min_fraction=.9)
   joint=np.isfinite(tb.values)&np.isfinite(qb.values)
   ts=area_weighted_support(np.where(joint,tb.values,np.nan),base.cell_area_m2,ring,min_fraction=.9)
   qs=area_weighted_support(np.where(joint,qb.values,np.nan),base.cell_area_m2,ring,min_fraction=.9)
   T.append(ts["mean"]);Q.append(qs["mean"]);tf.append(tj["valid_fraction"]);qf.append(qj["valid_fraction"]);jf.append(ts["valid_fraction"])
   pr.append(dict(request_id=reqid,pressure_hpa=int(lev),temperature_K=ts["mean"],specific_humidity_kgkg=qs["mean"],
      temperature_original_valid_fraction=tj["valid_fraction"],humidity_original_valid_fraction=qj["valid_fraction"],
      joint_Tq_valid_fraction=ts["valid_fraction"],joint_valid_area_m2=ts["valid_area_m2"],
      original_annulus_area_m2=ts["original_area_m2"],profile_estimand="each level T/q means on identical joint native support; PI of annular mean profile, not mean gridpointPI"))
  atomic_csv(folder/"profile.csv",pr)
  atomic_npz(folder/"profile.npz",pressure_hpa=LEVELS,temperature_K=np.array(T),specific_humidity_kgkg=np.array(Q),
    temperature_original_valid_fraction=np.array(tf),humidity_original_valid_fraction=np.array(qf),joint_Tq_valid_fraction=np.array(jf))
  from thermodynamics_v5 import entropy600_fields
  ent=entropy600_fields(profile_fields["t"].values,profile_fields["q"].values,base.sp_pa)
  core=(base.distance_km>=0)&(base.distance_km<100);environment=(base.distance_km>=100)&(base.distance_km<300)
  add("core600_saturated_entropy_J_kgK",ent["saturated_entropy_jkgk"],"J kg-1 K-1",core,"original_core_0_100km",note="600hPa saturation entropy computed at each native cell fromT/SP, then averaged; actualq notrequired")
  add("environment600_entropy_J_kgK",ent["entropy_jkgk"],"J kg-1 K-1",environment,"original_environment_100_300km",note="600hPa actualentropy from native T/q andsameSP before averaging")
  atomic_json(folder/"entropy600_metadata.json",dict(thermodynamics_sha256=filehash(Path(__file__).with_name("thermodynamics_v5.py")),
    definition="mean native-grid entropy; not entropy evaluated at mean T/q",pressure_pa=60000,core_radius_km=[0,100],environment_radius_km=[100,300],
    original_area_support_threshold=.90,core_q_required=False,environment_q_required=True,full_cell_count=int(base.raw.size)))
 if spec["need_map"]:
  ref=mapfields[MAP[0]]
  payload=dict(latitude=ref.latitude,longitude=ref.longitude,latitude_indices=ref.latitude_indices,longitude_indices=ref.longitude_indices,
    distance_km=ref.distance_km,core_mask=ref.core_mask,cell_area_m2=ref.cell_area_m2,sp_pa=ref.sp_pa)
  payload.update({k:b.values for k,b in mapfields.items()});payload.update({k+"__valid":b.support_mask for k,b in mapfields.items()})
  atomic_npz(folder/"native_map_1500km.npz",**payload)
 atomic_csv(folder/"scalars.csv",rows);atomic_json(folder/"request_values.json",wide)
 atomic_json(folder/"request.json",dict(binding=binding,started_utc=started,finished_utc=stamp(),raw_reads=cache.new_raw_reads-nr0,cache_hits=cache.cache_hits-ch0,elapsed_seconds=time.perf_counter()-tick,
    ocean_masks=masks.identities,scalar_count=len(rows),profile=spec["need_profile"],map=spec["need_map"]))
 for p in folder.iterdir():
  if p.name not in ("COMPLETE.json","STARTED.json") and p.is_file() and not p.name.endswith(".tmp"):
   outputs.append(dict(name=p.name,size=p.stat().st_size,sha256=filehash(p)))
 result=dict(binding=binding,status="COMPLETE",started_utc=started,finished_utc=stamp(),outputs=outputs,
             elapsed_seconds=time.perf_counter()-tick,raw_reads=cache.new_raw_reads-nr0,cache_hits=cache.cache_hits-ch0)
 atomic_json(done,result)
 return dict(request_id=reqid,request_hash=rid,status="COMPLETE",folder=str(folder),elapsed_seconds=result["elapsed_seconds"],raw_reads=result["raw_reads"],cache_hits=result["cache_hits"],output_bytes=sum(x["size"] for x in outputs))
def canonical_geometry_key(row):
 # Preserve upstream key as provenance, but standardise actual UTC/position before partitioning.
 # Different module encodings may otherwise assign the SAME raw cache to different workers.
 lat=number(row.get("lat"));lon=number(row.get("lon"))
 if not np.isfinite(lat+lon):return digest(["missing",str(row.get("request_id",""))])
 obj=[utc_datetime(row["UTC"]).isoformat()+"Z",(lat+0.0).hex(),(lon%360+0.0).hex()]
 return digest(obj)

def shard_assignment(row,count):
 return int(canonical_geometry_key(row),16)%int(count)

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--requests",type=Path,default=ROOT/"data"/"environment_requests_v5.csv")
 p.add_argument("--limit",type=int,default=1);p.add_argument("--run-all",action="store_true")
 p.add_argument("--request-id");p.add_argument("--barrier-dir",type=Path);p.add_argument("--shard-index",type=int,default=0);p.add_argument("--shard-count",type=int,default=1)
 p.add_argument("--max-open",type=int,default=64);p.add_argument("--chunk-cache-mib",type=int,default=16);p.add_argument("--rss-budget-gib",type=float,default=8)
 p.add_argument("--cache-root",type=Path,default=ROOT/"cache"/"native_fields")
 p.add_argument("--output-root",type=Path,default=ROOT/"data"/"environment_extract")
 a=p.parse_args()
 if a.limit<1 or a.shard_count<1 or not(0<=a.shard_index<a.shard_count):raise ValueError("invalid limit/shard")
 for output in (a.cache_root,a.output_root):
  if not output.resolve().is_relative_to(ROOT.resolve()):raise ValueError("outputs must remain in mechanism_revision_v5")
  output.mkdir(parents=True,exist_ok=True)
 sem=json.loads((ROOT/"config"/"extraction_semantics_v5.json").read_text(encoding="utf-8"))
 with a.requests.open(encoding="utf-8-sig",newline="") as f:requests=list(csv.DictReader(f))
 if a.request_id:requests=[r for r in requests if r["request_id"]==a.request_id]
 requests=[r for r in requests if shard_assignment(r,a.shard_count)==a.shard_index]
 requests.sort(key=lambda r:({"P1":0,"P3":1,"P4":2,"P5":3}.get(r.get("module",""),9),utc_datetime(r["UTC"]),canonical_geometry_key(r),not boolean(r.get("need_map",False)),str(r.get("request_id",""))))
 if not a.run_all:requests=requests[:a.limit]
 if not requests:raise ValueError("no selected real requests")
 ids=[r["request_id"] for r in requests]
 if len(ids)!=len(set(ids)):raise ValueError("request IDs must be unique")
 runid=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"_"+str(os.getpid())+f"_s{a.shard_index}of{a.shard_count}"
 aud=ROOT/"audit"/("extract_"+runid);aud.mkdir()
 lock=ROOT/"audit"/f"extract_environment_s{a.shard_index}of{a.shard_count}_active.lock"
 fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
 with os.fdopen(fd,"w",encoding="utf-8") as f:f.write(canon(dict(pid=os.getpid(),argv=sys.argv,runid=runid,started=stamp())))
 tick=time.perf_counter();records=[];failed=False;read_count=0
 try:
  import psutil
  process=psutil.Process(os.getpid())
  atomic_json(aud/"launch.json",dict(argv=sys.argv,python=sys.executable,requests=str(a.requests),requests_sha256=filehash(a.requests),selected_count=len(requests),semantics=sem,started_utc=stamp(),scope="bounded extraction only, no statistics",shard_index=a.shard_index,shard_count=a.shard_count,max_open=a.max_open,chunk_cache_mib=a.chunk_cache_mib,rss_budget_gib=a.rss_budget_gib,threads={k:os.environ[k] for k in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS","NUMEXPR_NUM_THREADS")}))
  masks=OceanMasks(sem["sst"]["shapefile"])
  if a.barrier_dir:
   atomic_json(a.barrier_dir/f"ready_{a.shard_index}.json",dict(pid=os.getpid(),shard=a.shard_index,ready_utc=stamp(),audit_directory=str(aud)))
   barrier_wait=time.perf_counter()
   while not (a.barrier_dir/"RELEASE.json").exists():
    if time.perf_counter()-barrier_wait>900:raise TimeoutError("4worker startup barrier not released; no extraction begun")
    time.sleep(.1)
   atomic_json(aud/"barrier_released.json",dict(released_utc=stamp(),barrier_dir=str(a.barrier_dir)))
  with NativeFieldReader(ROOT/"config"/"available_variable_map_v5.json",max_open=a.max_open,chunk_cache_bytes=a.chunk_cache_mib*1024**2) as reader:
   cache=ExtractionCache(reader,a.cache_root,sem)
   for i,row in enumerate(requests,1):
    try:
     if not np.isfinite(number(row.get("lat"))+number(row.get("lon"))):
      rec=dict(request_id=row.get("request_id"),status="EXCLUDED_MISSING_ACTUAL_POSITION",reason=row.get("source_support_reason","missing original position; no extrapolation"))
     else:
      if boolean(row.get("position_extrapolated",False)):raise ValueError("extrapolated source position prohibited")
      rec=extract_one(row,cache,masks,a.output_root,sem)
    except Exception:
     failed=True;rec=dict(request_id=row.get("request_id"),status="FAILED",traceback=traceback.format_exc())
     atomic_json(aud/(digest(row)[:16]+"_failure.json"),rec)
    records.append(rec)
    with (aud/"request_ledger.jsonl").open("a",encoding="utf-8") as f:f.write(canon(dict(index=i,utc=stamp(),**rec))+"\n")
    with (aud/"native_reads.jsonl").open("a",encoding="utf-8") as f:
     for read in reader.read_log:f.write(canon(read)+"\n")
    read_count+=len(reader.read_log);reader.read_log.clear()
    rss=process.memory_info().rss
    atomic_json(aud/"progress.json",dict(completed=i,total=len(requests),updated_utc=stamp(),last_record=rec,raw_read_operations=read_count,rss_bytes=rss,shard_index=a.shard_index,shard_count=a.shard_count))
    print(canon(dict(index=i,total=len(requests),rss_bytes=rss,**rec)),flush=True)
    if rss>a.rss_budget_gib*1024**3:
     failed=True;atomic_json(aud/"RESOURCE_MEMORY_STOP.json",dict(rss_bytes=rss,budget_gib=a.rss_budget_gib,checkpoint_completed=i,no_other_process_stopped=True));break
  summary=dict(status="COMPLETE_WITH_FAILURES" if failed else "COMPLETE",started_run_id=runid,finished_utc=stamp(),elapsed_seconds=time.perf_counter()-tick,
    attempted=len(records),selected=len(requests),succeeded=sum(r["status"] not in ("FAILED","EXCLUDED_MISSING_ACTUAL_POSITION") for r in records),excluded_missing_position=sum(r["status"]=="EXCLUDED_MISSING_ACTUAL_POSITION" for r in records),failed=sum(r["status"]=="FAILED" for r in records),records=records,exit_code=1 if failed else 0,raw_read_operations=read_count,shard_index=a.shard_index,shard_count=a.shard_count)
  atomic_json(aud/"execution.json",summary);print(canon(summary),flush=True)
 finally:
  if lock.exists():
   owner=json.loads(lock.read_text(encoding="utf-8"))
   if owner.get("pid")==os.getpid():os.replace(lock,aud/"released_lock.json")
 return 1 if failed else 0
if __name__=="__main__":sys.exit(main())
