# Public path adapter; the scientific functions below retain their original implementation.
import sys as _sys
from pathlib import Path as _Path
_PACKAGE=next(p for p in _Path(__file__).resolve().parents if (p/'release_paths.py').is_file())
_sys.path.insert(0,str(_PACKAGE))
from release_paths import ADDENDUM, UNIFIED, PACKAGE, bootstrap
from pathlib import Path
import runpy, json, hashlib, os
O=Path(__file__).resolve().parent
bootstrap(O/'temp')
import numpy as np
import pandas as pd
A=ADDENDUM
U=A/'unified_main_20261023_1642'
def read(p): return pd.read_csv(p,keep_default_na=False,na_values=['','nan','NaN'],low_memory=False)
def save(x,n): x.to_csv(O/n,index=False)
def js(n,x): (O/n).write_text(json.dumps(x,indent=2,default=str,allow_nan=False),encoding='utf8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
paths={'master':U/'source_data/main_sid.csv','events':U/'source_data/all_return_events.csv','track':A/'source_data/new_lifecycle_tracks.csv','env':A/'source_data/de_existing_sources/track_sp_qc_relabelled.csv','intervals':A/'source_data/new_intervals.csv'}
m=read(paths['master']); ev=read(paths['events']); tr=read(paths['track']); en=read(paths['env']); it=read(paths['intervals'])
assert len(m)==2340 and m.sid.is_unique and len(ev)==133 and ev.sid.nunique()==123
ids=set(m.sid);tr=tr[tr.sid.isin(ids)].copy();en=en[en.sid.isin(ids)].copy();it=it[it.sid.isin(ids)].copy()
for d,c in [(tr,'time'),(en,'UTC'),(it,'UTC'),(it,'interval_end'),(ev,'return_UTC')]:d[c]=pd.to_datetime(d[c],utc=True)
for c in ['genesis_time','maturity_time','resolution_time','termination_time','lmi_time']:m[c]=pd.to_datetime(m[c],utc=True)
assert not tr.duplicated(['sid','time']).any() and not en.duplicated(['sid','UTC']).any()
assert set(tr.sid)==ids
basins=['NA','EP','WP','NI','SI','SP','SA']
risks=[];dep=[];sidrows=[];recon=[];coverage=[]
mi=m.set_index('sid'); eventkeys={(r.sid,r.return_UTC):r.event_id for r in ev.itertuples()}
for sid,g in tr.groupby('sid',sort=True):
 g=g.sort_values('time').reset_index(drop=True);r=mi.loc[sid];t=g.time;w=g.wind.to_numpy(float);th=float(r.mature_threshold_wind)
 finite=np.isfinite(w);gap=t.shift(-1)-t;legal=finite&np.r_[finite[1:],False]&(gap==pd.Timedelta(hours=6))&(t.shift(-1)<=r.termination_time)
 hot=finite&(w>=th);active=False;episode=0;first=None;last_end=None;nr=0;failed=0;episodes=[]
 for i in range(len(g)-1):
  if t.iloc[i]<r.maturity_time:continue
  # An observed six-hour downcross opens a new opportunity episode. A missing
  # interval censors the old episode; a later observed downcross can re-enter.
  if hot[i] and legal.iloc[i] and not hot[i+1]:
   episode+=1;active=True;start=i+1
   info=dict(sid=sid,departure_order=episode,t0=t.iloc[start],wind_kt=w[start],threshold_kt=th,lat=g.lat.iloc[start],lon=g.lon.iloc[start],position_basin=g.basin.iloc[start],genesis_basin=r.basin_genesis,genesis_year=int(r.genesis_year),period=next(k for k,a,b in [('P1',1982,1992),('P2',1993,2003),('P3',2004,2014),('P4',2015,2023)] if a<=r.genesis_year<=b),first_departure=first is None)
   episodes.append(info)
   if first is None:first=info.copy()
  if not active or hot[i]:continue
  if not legal.iloc[i]:
   active=False;failed+=1;episodes[-1].update(end_UTC=t.iloc[i],end_reason='missing_wind_or_non6h_interval',return_event=False);continue
  # End-node basin gives the same discrete event location as original Figure 1b.
  # Start-node attribution is separately saved for a sensitivity analysis.
  returned=bool(hot[i+1] and (i+1<len(g)-1) and legal.iloc[i+1])
  terminal_hot=bool(hot[i+1] and not returned)
  eid=eventkeys.get((sid,t.iloc[i+1])) if returned else None
  if returned:
   assert eid is not None,('new unmatched event',sid,t.iloc[i+1]);nr+=1;recon.append(dict(sid=sid,event_id=eid,return_UTC=t.iloc[i+1],return_basin=g.basin.iloc[i+1],departure_order=episode))
  risks.append(dict(sid=sid,genesis_year=int(r.genesis_year),genesis_basin=r.basin_genesis,departure_order=episode,interval_start=t.iloc[i],interval_end=t.iloc[i+1],hours=6,basin_start=g.basin.iloc[i],basin_end=g.basin.iloc[i+1],wind_start=w[i],wind_end=w[i+1],threshold_kt=th,return_event=int(returned),terminal_hot_not_qualifying=int(terminal_hot),event_id=eid,stage_start=int(g.stage_code.iloc[i]),wind_filled_start=bool(g.wind_filled_flag.iloc[i]),wind_filled_end=bool(g.wind_filled_flag.iloc[i+1]),tropical_start=bool(g.is_tropical.iloc[i]),tropical_end=bool(g.is_tropical.iloc[i+1])))
  last_end=t.iloc[i+1]
  if hot[i+1]:
   active=False;episodes[-1].update(end_UTC=t.iloc[i+1],end_reason='qualifying_return' if returned else 'terminal_near_peak_not_eligible',return_event=returned)
 if active and episodes:episodes[-1].update(end_UTC=t.iloc[-1],end_reason='record_end',return_event=False)
 for e in episodes:
  if 'end_reason' not in e:e.update(end_UTC=t.iloc[-1],end_reason='record_end_without_next_interval',return_event=False)
  e['observable_hours']=(e['end_UTC']-e['t0']).total_seconds()/3600
 dep.extend(episodes)
 base=dict(sid=sid,basin_genesis=r.basin_genesis,genesis_year=int(r.genesis_year),period=next(k for k,a,b in [('P1',1982,1992),('P2',1993,2003),('P3',2004,2014),('P4',2015,2023)] if a<=r.genesis_year<=b),ever_return=int(sid in set(ev.sid)),n_return=nr,first_departure_UTC=None if first is None else first['t0'],departure_n=episode,censored_gap_episodes=failed,n_track_records=len(g),lifetime_h=float(r.lifetime_hours),record_end=r.termination_time,missing_wind_n=int((~finite).sum()),non6h_intervals=int(((gap!=pd.Timedelta(hours=6))&gap.notna()).sum()),filled_wind_n=int(g.wind_filled_flag.sum()))
 if first:
  base.update(reference_wind=first['wind_kt'],reference_lat=first['lat'],reference_lon=first['lon'],record_followup_h=(r.termination_time-first['t0']).total_seconds()/3600,first_return_h=(ev.loc[ev.sid.eq(sid),'return_UTC'].min()-first['t0']).total_seconds()/3600 if sid in set(ev.sid) else np.nan)
 sidrows.append(base)
risk=pd.DataFrame(risks);dep=pd.DataFrame(dep);sr=pd.DataFrame(sidrows);rr=pd.DataFrame(recon)
assert len(rr)==133 and set(rr.event_id)==set(ev.event_id) and rr.sid.nunique()==123
assert (rr.merge(ev[['event_id','return_basin']],on='event_id',suffixes=('_new','_old')).eval('return_basin_new == return_basin_old')).all()
assert int(sr.n_return.sum())==133
ref=sr.merge(en,left_on=['sid','first_departure_UTC'],right_on=['sid','UTC'],how='left',validate='1:1',suffixes=('','_env'))
for col in ['X1_sst_c','X2_shear_ms','X3_mid_rh_pct']:
 ref[col]=pd.to_numeric(ref[col+'_value'],errors='coerce').where(ref[col+'_qc'].eq(True))
ref['state_complete']=ref[['wind_kt','motion_past6h_ms','age_h','past24_land_contact','lat','month_sin','month_cos']].notna().all(axis=1)
ref['environment_complete']=ref[['X1_sst_c','X2_shear_ms','X3_mid_rh_pct']].notna().all(axis=1)
save(risk,'risk_intervals_6h.csv');save(dep,'departure_episodes.csv');save(sr,'sid_denominators.csv');save(ref,'first_departure_reference.csv');save(rr,'event_reconciliation.csv')
for b in basins:
 s=sr[sr.basin_genesis.eq(b)];q=risk[risk.basin_end.eq(b)];coverage.append(dict(basin=b,genesis_sid=len(s),returning_sid=int(s.ever_return.sum()),return_events=int(q.return_event.sum()),opportunity_intervals=len(q),cyclone_hours=int(q.hours.sum()),risk_unique_sid=q.sid.nunique(),mean_risk_hours_per_sid=float(q.hours.sum()/q.sid.nunique()) if q.sid.nunique() else None,rate_per1000h=float(q.return_event.sum()/q.hours.sum()*1000) if len(q) else None))
save(pd.DataFrame(coverage),'basin_counts.csv')
support=ref.groupby(['period','basin_genesis'],dropna=False).agg(n_sid=('sid','size'),events=('ever_return','sum'),reference_n=('first_departure_UTC','count'),env_n=('environment_complete','sum'),state_n=('state_complete','sum'),mean_followup_h=('record_followup_h','mean'),median_followup_h=('record_followup_h','median'),missing_wind_n=('missing_wind_n','sum'),filled_wind_n=('filled_wind_n','sum'),non6h_intervals=('non6h_intervals','sum')).reset_index();save(support,'support_by_period_basin.csv')
js('SOURCE_MANIFEST.json',{k:{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for k,p in paths.items()})
js('CHECKPOINT_prepare.json',dict(status='PREPARED_VALIDATED',main_sid=2340,return_sid=123,events=133,risk_intervals=len(risk),risk_hours=int(risk.hours.sum()),reference_sid=int(ref.first_departure_UTC.notna().sum()),environment_complete=int(ref.environment_complete.sum()),state_complete=int(ref.state_complete.sum()),departure_end_reasons=dep.end_reason.value_counts().to_dict(),old_sources_unchanged=True))
print((O/'CHECKPOINT_prepare.json').read_text());print(pd.DataFrame(coverage).to_string(index=False))
