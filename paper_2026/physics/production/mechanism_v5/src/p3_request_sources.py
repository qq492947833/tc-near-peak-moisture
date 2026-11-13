"""Reuse original P3 raw-reader functions without rebuilding cohort or raw-SHA scan."""
from pathlib import Path
import importlib.util,json,hashlib,sys
from datetime import datetime,timezone
import numpy as np,pandas as pd
V5=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('p3_original',V5/'src/p3_cohort.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
audit=[];suppressed=[]
original_register=m.register
def current_register(p,role,expected=None):
 p=Path(p)
 if p==m.RAW:
  old=json.loads((V5/'audit/p3_source_identity.json').read_text('utf-8'))
  previous=next(x for x in old['inputs'] if x['path']==str(p))
  assert p.stat().st_size==previous['bytes']
  audit.append({'path':str(p),'role':role,'current_bytes':p.stat().st_size,
    'current_mtime_ns':p.stat().st_mtime_ns,'previous_verified_sha256':previous['sha256'],
    'current_identity_scope':'size/mtime recorded; no repeated raw SHA; prior complete SHA retained'})
  return p
 return original_register(p,role,expected)
m.register=current_register
m.csv=lambda d,rel:suppressed.append({'original_audit_path':rel,'rows':len(d),'action':'not rewritten; original cohort artifacts retained'})
state=m.dictionary();raw=m.load_raw(state)
events=m.readcsv(V5/'data/p3_all_nonoverlap96h_events.csv')
events.t0=pd.to_datetime(events.t0,utc=True)
lookup=raw.set_index(['sid','UTC'])
previous=raw.sort_values(['sid','UTC']).copy()
previous['prev_UTC']=previous.groupby('sid').UTC.shift()
previous['prev_lat']=previous.groupby('sid').lat.shift()
previous['prev_lon']=previous.groupby('sid').lon.shift()
previous=previous.set_index(['sid','UTC'])
out=[]
for e in events.itertuples():
 for lag in [-24,-18,-12,-6,0]:
  utc=e.t0+pd.Timedelta(hours=lag);key=(e.sid,utc)
  z=previous.loc[key] if key in previous.index else None
  loc=z is not None and np.isfinite([z.lat,z.lon]).all()
  motion=loc and pd.notna(z.prev_UTC) and utc-z.prev_UTC==pd.Timedelta(hours=6) and np.isfinite([z.prev_lat,z.prev_lon]).all()
  if motion:
   dl=((z.lon-z.prev_lon+180)%360)-180
   cu=6371008.8*np.cos(np.deg2rad((z.lat+z.prev_lat)/2))*np.deg2rad(dl)/21600
   cv=6371008.8*np.deg2rad(z.lat-z.prev_lat)/21600
  else:cu=cv=np.nan
  out.append(dict(request_id=e.event_id+f'|lag{lag:+03d}',event_id=e.event_id,sid=e.sid,t0=e.t0,lag_h=lag,UTC=utc,
   lat=float(z.lat) if loc else np.nan,lon=float(z.lon) if loc else np.nan,
   observed_position=loc,position_support='observed_raw_MAIN_synoptic' if loc else 'missing_actual_position',
   source_line_number=int(z.source_line_number) if z is not None else np.nan,
   wind_kt=float(z.wind_kt) if z is not None else np.nan,state_class=z.state_class if z is not None else 'C',
   USA_STATUS=z.USA_STATUS if z is not None else '',NATURE=z.NATURE if z is not None else '',
   motion_u_past6h_ms=cu,motion_v_past6h_ms=cv,motion_speed_past6h_ms=float(np.hypot(cu,cv)),
   motion_valid=motion,motion_previous_UTC=z.prev_UTC if motion else pd.NaT,
   motion_max_source_UTC=utc if motion else pd.NaT,position_max_source_UTC=utc if loc else pd.NaT,
   Vref_kt=e.Vref_kt,main_first_per_sid=str(e.main_first_per_sid).lower()=='true',all_nonoverlap96h=True,
   center_policy='real_along_track_current_lag',interpolated=False))
d=pd.DataFrame(out)
assert len(d)==2617*5 and d.request_id.is_unique
assert d.UTC.le(d.t0).all()
assert d.loc[d.motion_valid,'motion_max_source_UTC'].le(d.loc[d.motion_valid,'t0']).all()
assert d.loc[~d.observed_position,['lat','lon']].isna().all().all()
d.to_csv(V5/'data/p3_environment_request_sources.csv',index=False,encoding='utf-8-sig')
rec={'status':'PASS','UTC':datetime.now(timezone.utc).isoformat(),'event_n':len(events),'request_n':len(d),
'position_valid_n':int(d.observed_position.sum()),'motion_valid_n':int(d.motion_valid.sum()),
'by_lag':d.groupby('lag_h')[['observed_position','motion_valid']].sum().reset_index().to_dict('records'),
'reuse':'p3_cohort.py dictionary/load_raw/readcsv functions, with original audit writes disabled and raw hashing replaced by recorded metadata check; no cohort selection rerun',
'raw_identity':audit,'preserved_original_audits':suppressed,'no_cohort_output_overwritten':True,'no_ERA5_read':True,
'raw_reader_checks':m.TESTS,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(V5/'audit/p3_environment_request_sources.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
print(json.dumps(rec,ensure_ascii=False,indent=2,default=str))

