"""Fixed WP/JASO P2 physical applicability screen, not a TN flux analysis."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,csv,json,hashlib,sys,os
import numpy as np
V=Path(__file__).resolve().parents[1];OUT=V/"p2_wave_applicability"
sys.path.insert(0,str(V/"src"))
R=6371008.8;OMEGA=7.292115e-5
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding="utf-8")
def rows(p):
 with Path(p).open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def csvsave(p,data):
 keys=list(dict.fromkeys(k for r in data for k in r))
 with Path(p).open("w",encoding="utf-8-sig",newline="") as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(data)
def boolean(x):
 if str(x).lower() in ("true","1"):return True
 if str(x).lower() in ("false","0"):return False
 raise ValueError("explicit Boolean required")
def geometry():
 return np.arange(119,282),np.arange(399,722)
def keys():return ("z200","u200","v200")
def selection(r,s):
 t=datetime.fromisoformat(r["t0"])
 conditions=dict(current117=boolean(r["current117"]),basin=r["basin_genesis"]==s["basin"],
   season=t.month in s["months"],latitude=s["latitude"][0]<=float(r["lat"])<=s["latitude"][1])
 return all(conditions.values()),";".join(k for k,v in conditions.items() if not v)
def prepare():
 OUT.mkdir(exist_ok=True);s=read(V/"analysis_spec_v5.json")["P2"]["wave_applicability"]
 spec=dict(frozen_utc=now(),scope=s,selection="event-level t0, all eligible events; not outcome or effect selected",
  domain_grid="native0.25 with one neighbour halo; diagnostic support cropped to exact fixed domain",
  quantities=["signed_zeta_over_f","absolute_zeta_over_f","actual_minus_geostrophic_wind",
   "ageostrophic_to_actual_speed_ratio","perturbation_to_basic_speed_ratio",
   "rms_perturbation_length_m","rms_basic_length_m","basic_variation_length_ratio","scale_Rossby_U_over_fL"],
  geostrophic="u_g=-dPhi/dy/f; v_g=dPhi/dx/f; Phi is original geopotential m2/s2",
  streamfunction="psi_prime=(Phi-Phi_month_sameUTC)/f; no arbitrary f floor",
  masks="same-cell same-UTC SP>=20000Pa; finite fields; fixed domain; lat>=20; basic speed>=5; distance>=800km",
  decision="Report area-weighted distributions and scales; no invented universal Ro/amplitude PASS cutoff. Final TN applicability requires QG/small-amplitude/slow-background and interpretable wave pattern review, not merely geographic mask.",
  no_TN_flux_calculation=True,no_P_q=True,no_data_download=True,
  reference=dict(authors="Takaya K; Nakamura H",year=2001,title="A Formulation of a Phase-Independent Wave-Activity Flux for Stationary and Migratory Quasigeostrophic Eddies on a Zonally Varying Basic Flow",doi="10.1175/1520-0469(2001)058<0608:AFOAPI>2.0.CO;2",url="https://journals.ametsoc.org/view/journals/atsc/58/6/1520-0469_2001_058_0608_afoapi_2.0.co_2.xml"),
  interpretation="Conditions are prerequisite diagnostics, not evidence that a wave caused departure; event diagnostic before aggregation",
  inputs={f:sha(V/f) for f in ("analysis_spec_v5.json","data/p1_event_identity_v5.csv","config/climatology_tiles_v5_1.json","config/available_variable_map_v5.json")})
 if (OUT/"spec.json").exists():
  old=read(OUT/"spec.json")
  if old["scope"]!=s or old["inputs"]!=spec["inputs"]:raise ValueError("Previously frozen inputs differ")
  return old
 write(OUT/"spec.json",spec)
 membership=[];requests=[]
 il,jl=geometry()
 for r in rows(V/"data/p1_event_identity_v5.csv"):
  selected,reason=selection(r,s);row=dict(r,selected=selected,exclusion_reason=reason);membership.append(row)
  if selected:
   t=datetime.fromisoformat(r["t0"])
   requests.append({k:r[k] for k in ("sid","event_id","event_class","t0","lat","lon","basin_genesis")} |
    dict(month=t.month,hour=t.hour,group=(t.month-1)*4+t.hour//6,fields="z200;u200;v200;sp",
    ilat_first=int(il[0]),ilat_last=int(il[-1]),ilon_first=int(jl[0]),ilon_last=int(jl[-1]),n_native_cells=int(len(il)*len(jl))))
 csvsave(OUT/"event_membership.csv",membership);csvsave(OUT/"input_requests.csv",requests)
 write(OUT/"membership_receipt.json",dict(created_utc=now(),eligible_events=len(requests),eligible_SID=len({r["sid"] for r in requests}),
  internal=sum(r["event_class"]=="internal" for r in requests),final=sum(r["event_class"]=="final" for r in requests),
  no_environment_effects_read=True,actual_fields_computed=False))
 return spec
def climate_gate(requests):
 il,jl=geometry();plan=read(V/"config/climatology_tiles_v5_1.json")
 lookup={(x["key"],x["ilat_tile"],x["ilon_tile"]):x for x in plan["jobs"]};pending=[];evidence=[]
 groups=sorted({int(r["group"]) for r in requests})
 for key in keys():
  for i in np.unique(il//90):
   for j in np.unique(jl//180):
    name=f"{key}_y{i}_x{j}";folder=V/"cache/climatology_1991_2020_v5_1"/name
    job=lookup.get((key,int(i),int(j)))
    for group in groups:
     reason="NOT_IN_FROZEN_PLAN" if not job or group not in job["groups"] else "CLIMATOLOGY_TILE_NOT_COMPLETE" if not (folder/"complete.json").exists() else ""
     if reason:pending.append(dict(kind="climatology",tile=name,group=group,reason=reason,path=str(folder)))
    if (folder/"complete.json").exists():evidence.append(dict(tile=name,complete=str(folder/"complete.json"),sha256=sha(folder/"complete.json")))
 return pending,evidence
def cached_event(event,spec,allow_source_read):
 t=event["t0"];token=hashlib.sha256(t.encode()).hexdigest()[:16];folder=OUT/"actual_fields"/token;cp=folder/"complete.json"
 if cp.exists():
  c=read(cp)
  if c["UTC"]!=t or c["spec_sha256"]!=sha(OUT/"spec.json"):raise ValueError("Actual block identity conflict")
  if sha(folder/"fields.npz")!=c["npz_sha256"]:raise ValueError("Actual fields changed")
  with np.load(folder/"fields.npz",allow_pickle=False) as z:return {k:z[k] for k in z.files},c
 if not allow_source_read:return None,dict(status="PENDING_ACTUAL_GEOGRAPHIC_BLOCK",UTC=t,folder=str(folder))
 from native_fields_v5 import NativeFieldReader
 vm=read(V/"config/available_variable_map_v5.json")["paths"];il,jl=geometry()
 with NativeFieldReader(vm,max_open=4,max_block_cells=100000) as reader:
  la,lo=reader._coordinates("sp");lat=la[il];lon=lo[jl]
  if not np.array_equal(lat,90-il*.25) or not np.array_equal(lon,jl*.25):raise ValueError("Audited grid identity conflict")
  arrays=dict(latitude=lat,longitude=lon,latitude_indices=il,longitude_indices=jl)
  sp=reader._read("sp",t,il,jl);arrays["sp"]=sp
  for key in keys():
   a,b=reader._coordinates(key)
   if not np.array_equal(a[il],lat) or not np.array_equal(b[jl],lon):raise ValueError("Field grid mismatch")
   value=reader._read(key,t,il,jl);arrays[key]=np.where(np.isfinite(sp)&(sp>=20000),value,np.nan)
  reads=list(reader.read_log)
 folder.mkdir(parents=True);np.savez_compressed(folder/"fields.npz",**arrays)
 c=dict(status="COMPLETE",UTC=t,spec_sha256=sha(OUT/"spec.json"),npz_sha256=sha(folder/"fields.npz"),read_log=reads,
  source_identities={k:vm[k]["source_identity"] for k in list(keys())+["sp"]},same_grid_SP_QC=True,finished_utc=now())
 write(cp,c);return arrays,c
def wsummary(value,area,mask):
 good=mask&np.isfinite(value);w=area[good];x=value[good]
 if not len(x):return dict(n=0,area_m2=0,mean=None,p10=None,p50=None,p90=None)
 order=np.argsort(x);q=np.interp([.1,.5,.9],np.cumsum(w[order])/w.sum(),x[order])
 return dict(n=len(x),area_m2=float(w.sum()),mean=float(np.average(x,weights=w)),p10=float(q[0]),p50=float(q[1]),p90=float(q[2]))
def diagnose(actual,basic,event):
 from native_fields_v5 import spherical_gradient,native_cell_area_m2,great_circle_distance_km
 lat=actual["latitude"];lon=actual["longitude"];phi=np.deg2rad(lat)[:,None];f=2*OMEGA*np.sin(phi)
 area=native_cell_area_m2(lat,lon);distance=great_circle_distance_km(lat[:,None],lon[None,:],float(event["lat"]),float(event["lon"]))
 z,u,v=[actual[k] for k in keys()];zb,ub,vb=[basic[k] for k in keys()]
 zx,zy=spherical_gradient(z,lat,lon);ug=-zy/f;vg=zx/f
 up=u-ub;vp=v-vb;speed=np.hypot(u,v);bs=np.hypot(ub,vb)
 vx,vy=spherical_gradient(v,lat,lon);ux,uy=spherical_gradient(u,lat,lon)
 zeta=vx-uy+u*np.tan(phi)/R
 mask=(lat[:,None]>=20)&(lat[:,None]<=60)&(lon[None,:]>=100)&(lon[None,:]<=180)&(distance>=800)&(bs>=5)
 mask&=np.isfinite(z)&np.isfinite(u)&np.isfinite(v)&np.isfinite(zb)&np.isfinite(ub)&np.isfinite(vb)
 perturb=np.hypot(up,vp);ag=np.hypot(u-ug,v-vg)
 with np.errstate(divide="ignore",invalid="ignore"):
  fields=dict(signed_zeta_over_f=zeta/f,absolute_zeta_over_f=abs(zeta/f),geostrophic_u=ug,geostrophic_v=vg,
    ageostrophic_speed_ms=ag,ageostrophic_to_actual_speed_ratio=np.where(speed>0,ag/speed,np.nan),
    perturbation_to_basic_speed_ratio=perturb/bs,geopotential_anomaly=z-zb,psi_prime=(z-zb)/f,basic_speed_ms=bs)
 summaries=[dict(metric=k,**wsummary(x,area,mask)) for k,x in fields.items() if k not in ("geostrophic_u","geostrophic_v","geopotential_anomaly","psi_prime")]
 def rms(x):return np.sqrt(wsummary(x*x,area,mask)["mean"])
 gup=spherical_gradient(up,lat,lon);gvp=spherical_gradient(vp,lat,lon)
 gub=spherical_gradient(ub,lat,lon);gvb=spherical_gradient(vb,lat,lon)
 gp=np.sqrt(sum(x*x for x in gup+gvp));gb=np.sqrt(sum(x*x for x in gub+gvb))
 Lp=rms(perturb)/rms(gp);Lb=rms(bs)/rms(gb)
 scales=dict(rms_perturbation_length_m=float(Lp),rms_basic_length_m=float(Lb),
  basic_variation_length_ratio=float(Lp/Lb),scale_Rossby_U_over_fL=float(rms(bs)/np.average(abs(f)*np.ones_like(bs),weights=area*mask)/Lp))
 return fields,mask,summaries,scales
def run(allow_source_read):
 spec=prepare();requests=rows(OUT/"input_requests.csv");pending,evidence=climate_gate(requests)
 if pending:
  csvsave(OUT/"pending_inputs.csv",pending)
  write(OUT/"status.json",dict(status="WAITING_FOR_FIXED_CLIMATOLOGY",execution="NOT_DIAGNOSED",verification="NOT_READY",
   scientific_conclusion="APPLICABILITY_UNDETERMINED_NOT_INAPPLICABLE",pending_rows=len(pending),events=len(requests),
   actual_raw_reads=0,updated_utc=now()));return
 from climatology_v5 import ClimatologyReader
 cr=ClimatologyReader();il,jl=geometry();allstats=[]
 for event in requests:
  actual,receipt=cached_event(event,spec,allow_source_read)
  if actual is None:pending.append(dict(kind="actual_fields",event_id=event["event_id"],reason=receipt["status"],path=receipt["folder"]));continue
  basic={k:cr.read(k,int(event["month"]),int(event["hour"]),il,jl)[0] for k in keys()}
  fields,mask,stats,scales=diagnose(actual,basic,event)
  folder=OUT/"diagnostics"/event["event_id"].replace("|","_");folder.mkdir(parents=True,exist_ok=True)
  np.savez_compressed(folder/"fields.npz",latitude=actual["latitude"],longitude=actual["longitude"],mask=mask,**fields)
  write(folder/"summary.json",dict(event=event,scales=scales,metrics=stats,actual_source=receipt,climatology_evidence=evidence,
   status="DIAGNOSED_PENDING_PHYSICAL_REVIEW_NOT_TN_APPROVAL",no_TN_flux=True,no_P_q=True))
  allstats += [dict(event_id=event["event_id"],sid=event["sid"],event_class=event["event_class"],**s) for s in stats]
 if pending:csvsave(OUT/"pending_inputs.csv",pending)
 if allstats:csvsave(OUT/"diagnostic_summary.csv",allstats)
 write(OUT/"status.json",dict(status="WAITING_FOR_ACTUAL_BLOCKS" if pending else "DIAGNOSED_PENDING_PHYSICAL_REVIEW",
  execution="PARTIAL" if pending else "COMPUTED",verification="PENDING_ACTUAL_MAP_AND_PHYSICAL_REVIEW",
  scientific_conclusion="APPLICABILITY_UNDETERMINED",pending=len(pending),events=len(requests),updated_utc=now(),
  no_TN_flux=True,no_P_q=True))
def main():
 p=argparse.ArgumentParser();p.add_argument("--prepare",action="store_true");p.add_argument("--run",action="store_true")
 p.add_argument("--allow-source-read",action="store_true");a=p.parse_args()
 if a.prepare==a.run:raise ValueError("Select exactly prepare or run")
 if a.prepare:prepare()
 else:run(a.allow_source_read)
 print(str(OUT));print((OUT/"status.json").read_text() if (OUT/"status.json").exists() else (OUT/"membership_receipt.json").read_text())
if __name__=="__main__":main()

