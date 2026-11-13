"""Build v5 requests only from frozen points; no ERA5 read or effect fit."""
import json,datetime,hashlib
from pathlib import Path
import pandas as pd,numpy as np
V=Path(__file__).resolve().parents[1];B=V.parent/"review_completion_v3_2"
track=pd.read_parquet(B/"sources/normalized_track_sp_qc_v3_2.parquet")
track["UTC"]=pd.to_datetime(track.UTC,utc=True)
assert not track.duplicated(["sid","UTC"]).any()
idx=track.set_index(["sid","UTC"])
h1=pd.read_csv(B/"e2_e3/H1/samples.csv",keep_default_na=False)
current=set(h1.loc[(h1.comparison=="new_available")&(h1.variable=="X1_sst_c"),"sid"])
assert len(current)==117
pairs=pd.read_parquet(B/"e2_e3/fixed_pairs.parquet")
# Resolve through the already catalogued frozen source, not guessed sibling structure.
inventory=json.loads((V.parent/"representation_value_v4/audit/duration_sources/source_inventory.json").read_text(encoding="utf-8"))
stormpath=next(x["path"] for x in inventory["sources"] if x["role"]=="frozen_main_sid")
storms=pd.read_csv(stormpath).set_index("sid")
def point(sid,t):
    key=(sid,pd.Timestamp(t))
    return idx.loc[key] if key in idx.index else None
def motion(sid,t):
    a=point(sid,t-pd.Timedelta(hours=6));b=point(sid,t)
    if a is None or b is None:return np.nan,np.nan
    la0,lo0,la1,lo1=map(float,[a.lat_track,a.lon_track,b.lat_track,b.lon_track])
    if not np.isfinite([la0,lo0,la1,lo1]).all():return np.nan,np.nan
    R=6371008.8
    dl=(lo1-lo0+180)%360-180
    return R*np.deg2rad(dl)*np.cos(np.deg2rad((la0+la1)/2))/21600,R*np.deg2rad(la1-la0)/21600
def fmt(t):return pd.Timestamp(t).isoformat()
rows=[];events=[]
def add(d):
    d.setdefault("variable_set","full_process");d.setdefault("need_profile",True);d.setdefault("need_map",False)
    d.setdefault("source_support_reason","observed_exactUTC" if np.isfinite([d["lat"],d["lon"]]).all() else "NO_OBSERVED_POSITION")
    d["position_extrapolated"]=False
    d["geometry_key"]=hashlib.sha256((d["UTC"]+"|"+str(d["lat"])+"|"+str(d["lon"])).encode()).hexdigest()[:20] if np.isfinite([d["lat"],d["lon"]]).all() else ""
    rows.append(d)
for p in pairs.to_dict("records"):
 sid=p["sid"];s=storms.loc[sid]
 for typ in ["internal","final"]:
  t0=pd.Timestamp(p["t0_"+typ]);e=p["event_id_"+typ]
  ev={"sid":sid,"event_id":e,"event_class":typ,"t0":fmt(t0),"current117":sid in current,"candidate123":True,
      "first_LMI_UTC":s.lmi_time,"last_LMI_UTC":s.lmi_last_time,"termination_UTC":s.termination_time,
      "t0_after_first_LMI":t0>=pd.Timestamp(s.lmi_time,tz="UTC") if pd.Timestamp(s.lmi_time).tzinfo is None else t0>=pd.Timestamp(s.lmi_time)}
  for f in ["lat","lon","wind_t0_kt","near_threshold_kt","age_h","delta_v24_backward","land_contact","basin_genesis"]:
   ev[f]=p[f+"_"+typ]
  for field in ["t0_USA_STATUS","t0_NATURE","state_class"]:ev[field]="not_yet_joined_from_accepted_crosswalk"
  events.append(ev)
  mu0,mv0=motion(sid,t0)
  for lag in range(-48,49,6):
   t=t0+pd.Timedelta(hours=lag);z=point(sid,t);mu,mv=motion(sid,t)
   base=dict(module="P1",sid=sid,event_id=e,event_class=typ,t0=fmt(t0),UTC=fmt(t),lag_h=lag,current117=sid in current,candidate123=True)
   add(dict(base,request_id=f"P1|{e}|along|{lag}",center_kind="actual_track",lat=float(z.lat_track) if z is not None else np.nan,lon=float(z.lon_track) if z is not None else np.nan,motion_u_ms=mu,motion_v_ms=mv,need_map=True,wind_kt=float(z.wind_kt_track) if z is not None else np.nan,stage=str(z.stage_v3) if z is not None else "",actual_track_point=z is not None))
   # Fixed-centre pre fields are separate; at/following t0 these requests serve SST displacement only.
   add(dict(base,request_id=f"P1|{e}|fixed|{lag}",center_kind="fixed_t0",lat=float(p["lat_"+typ]),lon=float(p["lon_"+typ]),motion_u_ms=mu0,motion_v_ms=mv0,need_map=lag<0,need_profile=lag<0,variable_set="full_process" if lag<0 else "SST_for_displacement",wind_kt=float(p["wind_t0_kt_"+typ]),stage="fixed_location_not_track_stage",actual_track_point=False))
pd.DataFrame(events).to_csv(V/"data/p1_event_identity_v5.csv",index=False,encoding="utf-8-sig")
p1=pd.DataFrame(rows);p1.to_csv(V/"data/p1_environment_requests_v5.csv",index=False,encoding="utf-8-sig")
p4=pd.read_csv(V/"p4_p5/P4_boundary_requests.csv",keep_default_na=False,na_values=[""])
for z in p4.to_dict("records"):
 t=pd.Timestamp(z["UTC"]);mu,mv=motion(z["sid"],t)
 add(dict(request_id="P4|"+z["request_id"],module="P4",sid=z["sid"],event_id=z["sid"]+"|"+z["boundary"],event_class=z["boundary"],t0=z["t0"],UTC=fmt(t),lag_h=z["lag_h"],center_kind="actual_track",lat=z["lat"],lon=z["lon"],motion_u_ms=mu,motion_v_ms=mv,wind_kt=z["wind_kt"],stage=z["frozen_stage"],source_support_reason=z["source_support_reason"]))
p5=pd.read_csv(V/"p4_p5/P5_matched_intensity_pairs.csv")
for z in p5.to_dict("records"):
 for side in ["pre","post"]:
  t=pd.Timestamp(z["UTC_"+side]);mu,mv=motion(z["sid"],t)
  add(dict(request_id=f"P5|{z['pair_id']}|{side}",module="P5",sid=z["sid"],event_id=z["pair_id"],event_class=side,t0=fmt(t),UTC=fmt(t),lag_h=0,center_kind="actual_track",lat=z["lat_"+side],lon=z["lon_"+side],motion_u_ms=mu,motion_v_ms=mv,wind_kt=z["wind_kt_"+side],stage=z["stage_"+side],pair_weight_within_sid=z["pair_weight_within_sid"]))
out=pd.DataFrame(rows)
assert not out.request_id.duplicated().any()
out.to_csv(V/"data/environment_requests_v5.csv",index=False,encoding="utf-8-sig")
audit=dict(created=datetime.datetime.now(datetime.timezone.utc).isoformat(),counts=out.groupby("module").size().to_dict(),valid_position=out.dropna(subset=["lat","lon"]).groupby("module").size().to_dict(),current117=117,candidate123=len(pairs),distinct_valid_geometries=out.loc[out.geometry_key.ne(""),"geometry_key"].nunique(),status="READY_FOR_PILOT_P3_APPEND_PENDING")
(V/"audit/environment_requests_build_v5.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
print(json.dumps(audit))
