"""Original interval functions used by current-master drivers; lifecycle entry is provided separately."""
from pathlib import Path
import ast, hashlib, json, sys
import numpy as np
import pandas as pd
import h5py
V=Path(__file__).resolve().parents[1]
R=V.parent
P=R.parent.parent
F19=P/"IBTrACS_interdecadal_v19"
SEED=20260920

def dependency(path):
 p=Path(path);return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}

def active_plan():
 pointer=json.loads((V/'configs/active_analysis_plan.json').read_text(encoding='utf-8'));p=V/pointer['relative_path'];assert dependency(p)['sha256']==pointer['sha256'];assert pointer['version']=='v4_v2_C2_G1_only';return pointer

def read(p,**kw):return pd.read_csv(p,keep_default_na=False,na_values=['','nan','NaN'],**kw)

def write(frame,name):
 p=V/'source_data'/name;p.parent.mkdir(parents=True,exist_ok=True)
 if p.suffix=='.parquet':frame.to_parquet(p,index=False)
 else:frame.to_csv(p,index=False)

def dump(name,x):
 p=V/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str),encoding='utf-8')

def utc(x):return pd.to_datetime(x,utc=True)

def runs(mask):
 idx=np.flatnonzero(mask)
 return np.split(idx,np.where(np.diff(idx)>1)[0]+1) if len(idx) else []

def later_category(lengths,supported=True):
 if not supported:return 'unknown','unknown'
 if any(not np.isfinite(h) or h<=0 or h%6!=0 for h in lengths):return 'unknown','unknown'
 if not lengths:return 'none','G0'
 if len(lengths)==1 and lengths[0]==6:return 'single_6h','G1'
 if len(lengths)>=2 and all(h==6 for h in lengths):return 'allpieces6h_multiple','G2'
 if any(h>=12 for h in lengths):return 'at_least_12h','G3'
 raise ValueError('Later episode duration must be positive multiples of six hours')

def contiguous(t,w,anchor,threshold):
 ai=np.flatnonzero(t==pd.Timestamp(anchor).to_datetime64());assert len(ai)==1
 a=b=int(ai[0]);qual=np.isfinite(w)&(w>=threshold)
 assert qual[a]
 while a>0 and t[a]-t[a-1]==np.timedelta64(6,'h') and qual[a-1]:a-=1
 while b+1<len(t) and t[b+1]-t[b]==np.timedelta64(6,'h') and qual[b+1]:b+=1
 mask=np.zeros(len(t),bool);mask[a:b+1]=True
 return mask,a,b

def intervals(g,row,config='main90',saved=True):
 g=g.sort_values('UTC').reset_index(drop=True).copy();t=utc(g.UTC).to_numpy(dtype='datetime64[ns]');w=g.wind_kt.to_numpy(float);finite=np.isfinite(w);n=len(g)
 nextt=np.r_[t[1:],np.datetime64('NaT')];gap=(nextt-t)/np.timedelta64(1,'h')
 term=pd.Timestamp(row['termination_time']).to_datetime64();mt=pd.Timestamp(row['maturity_time']).to_datetime64();rt=pd.Timestamp(row['resolution_time']).to_datetime64()
 valid=(gap==6)&finite&np.r_[finite[1:],False]&(nextt<=term)
 threshold=float(row['mature_threshold_wind']);R90,a,b=contiguous(t,w,row['lmi_time'],threshold)
 U90=finite&(w>=threshold)&(t>=mt)&(t<=term);F90=(t>=mt)&(t<rt)
 if saved:
  for name,value in [('R_selected',R90),('U_selected',U90),('F_selected',F90)]:assert np.array_equal(g[name].to_numpy(bool),value),(row['sid'],name)
  assert np.array_equal(g.valid_6h_interval.to_numpy(bool),valid)
 # A complete counting window must include actual right endpoint of final selected left node.
 def support(mask):
  ix=np.flatnonzero(mask)
  if not len(ix):return False,'empty_selection'
  lo,hi=int(ix[0]),min(int(ix[-1])+1,n-1)
  ok=bool(finite[lo:hi+1].all() and np.all(np.diff(t[lo:hi+1])==np.timedelta64(6,'h')))
  return ok,('complete' if ok else 'missing_wind_or_non6h_gap')
 complete,reason=support(F90|R90|U90)
 positive_near_interval=bool(np.any(U90&valid))
 if not positive_near_interval:complete=False;reason='NOT_ESTIMABLE_NO_LEGAL_NEARPEAK_INTERVAL'
 # Longest piece is selected by actual covered intervals, not number of nodes.
 legal_runs=runs(U90&valid);long=np.zeros(n,bool)
 if legal_runs:
  chosen=max(legal_runs,key=lambda ix:(len(ix),-int(ix[0])));long[chosen]=True
 R5,ca,cb=contiguous(t,w,row['lmi_time'],float(row['lmi'])-5)
 U5=finite&(w>=float(row['lmi'])-5)&(np.arange(n)>=ca)&(t<=term)
 qi=np.flatnonzero(U5);F5=(np.arange(n)>=ca)&(np.arange(n)<=int(qi[-1]))
 six,six_reason=support(F90|R90|U90|R5|U5|F5)
 if not positive_near_interval:six=False;six_reason='NOT_ESTIMABLE_NO_LEGAL_NEARPEAK_INTERVAL'
 # Unsupported physical windows are not interpreted as zero duration components.
 masks={'R':R90,'U':U90,'F':F90,'L':U90&~R90,'B':F90&~U90,'R_long':long,'R5':R5,'U5':U5,'F5':F5}
 result=dict(sid=row['sid'],name=row.get('name',''),config=config,genesis_year=int(pd.Timestamp(row['genesis_time']).year),basin_genesis=row['basin_genesis'],multi=bool(float(row['near_lmi_episode_count'])>1),near_lmi_episode_count=row['near_lmi_episode_count'],genesis_time=row['genesis_time'],maturity_time=row['maturity_time'],resolution_time=row['resolution_time'],termination_time=row['termination_time'],lmi_time=row['lmi_time'],lmi=float(row['lmi']),genesis_wind=float(row['genesis_wind']),threshold90_kt=threshold,threshold5_kt=float(row['lmi'])-5,physical_main_complete=complete,physical_main_reason=reason,positive_legal_nearpeak_interval=positive_near_interval,six_complete=six,six_reason=six_reason,R_start_equals_F=bool(t[a]==mt),frozen_complete_common_window=bool(row.get('complete_common_window',complete)))
 result['main_complete']=result['frozen_complete_common_window'] if saved else complete
 for k,mask in masks.items():
  result['T_'+k+'_h']=float(6*np.sum(mask&valid)) if (complete or k in ['R5','U5','F5']) else np.nan
  g['mask_'+k]=mask
  g[k+'_valid_h']=np.where(mask&valid,6.,0.)
 result['T_L_h']=result['T_U_h']-result['T_R_h'] if complete else np.nan
 result['T_B_h']=result['T_F_h']-result['T_U_h'] if complete else np.nan
 if complete:
  assert result['T_R_h']<=result['T_U_h']<=result['T_F_h'],row['sid']
  assert result['R_start_equals_F'],row['sid']
  assert result['T_F_h']==result['T_R_h']+result['T_L_h']+result['T_B_h']
  if saved:
   for k,old in [('R','T_R_h'),('U','T_U_h'),('F','T_F_h'),('L','T_later_h'),('B','T_bridge_h')]:assert result['T_'+k+'_h']==float(row[old]),(row['sid'],k)
 if saved:assert float(6*np.sum(R5&valid))==float(row['C5_covered_duration_h']),row['sid']
 later_runs=runs((U90&~R90)&valid);lengths=[6*len(ix) for ix in later_runs]
 result['later_interval_episode_n']=len(lengths);result['later_max_episode_h']=max(lengths,default=0)
 result['later_class'],result['later_group']=later_category(lengths,complete)
 g['later_episode_id']=0
 for no,ix in enumerate(later_runs,1):g.loc[ix,'later_episode_id']=no
 result['pre_maturity_near_h']=float(6*np.sum(valid&finite&(w>=threshold)&(t<mt)))
 result['U_fraction_F']=result['T_U_h']/result['T_F_h'] if complete and result['T_F_h']>0 else np.nan
 result['L_fraction_U']=result['T_L_h']/result['T_U_h'] if complete and result['T_U_h']>0 else np.nan
 result['period']=next((k for k,(lo,hi) in {'P1':(1982,1992),'P2':(1993,2003),'P3':(2004,2014),'P4':(2015,2023)}.items() if lo<=result['genesis_year']<=hi),'outside')
 g['UTC']=utc(g.UTC);g['interval_end']=pd.to_datetime(nextt,utc=True);g['interval_gap_h']=gap;g['valid_6h']=valid;g['valid_h']=np.where(valid,6.,0.);g['config']=config
 for k in ['sid','genesis_year','period','basin_genesis','multi','main_complete','six_complete']:g[k]=result[k]
 keep=['sid','UTC','interval_end','interval_gap_h','valid_6h','valid_h','wind_kt','wind_filled_flag','config','genesis_year','period','basin_genesis','multi','main_complete','six_complete','later_episode_id']+[c for c in g if c.startswith('mask_') or c.endswith('_valid_h')]
 return result,g[keep]

def ncframe(path):
 with h5py.File(path,'r') as h:
  def arr(k):
   d=h[k];a=d[:]
   if a.dtype.kind=='O':return np.array([x.decode() if isinstance(x,bytes) else str(x) for x in a])
   units=d.attrs.get('units',b'');units=units.decode() if isinstance(units,bytes) else units
   if ' since ' in units:
    unit,origin=units.split(' since ',1);return pd.Timestamp(origin,tz='UTC')+pd.to_timedelta(a,unit={'hours':'h','seconds':'s','days':'D'}[unit])
   return a
  fields=['sid','name','genesis_time','maturity_time','resolution_time','termination_time','lmi_time','basin_genesis','near_lmi_episode_count','genesis_wind','lmi','mature_threshold_wind']
  s=pd.DataFrame({k:arr(k) for k in fields});z=pd.DataFrame({'sid':s.sid.to_numpy()[arr('obs_storm_index')],'UTC':arr('obs_time'),'wind_kt':arr('obs_wind'),'wind_filled_flag':arr('obs_wind_filled_flag')})
 return s,z

def attach_state(d,inter):
 state=read(R/'review_completion_v3_2/audit/state_dependency/state_sid_utc_crosswalk_v3_2.csv');state.UTC=utc(state.UTC)
 fields=['sid','UTC','state_label','state_class','agency_conflict_nature_mx'];state=state[fields]
 g=inter.merge(state,on=['sid','UTC'],how='left',validate='1:1')
 end=state.rename(columns={c:c+'_end' for c in fields if c not in ['sid']}).rename(columns={'UTC_end':'interval_end'})
 g=g.merge(end,on=['sid','interval_end'],how='left',validate='m:1')
 conflict=g.agency_conflict_nature_mx.fillna(False)|g.agency_conflict_nature_mx_end.fillna(False)
 unknown=g.state_label.isna()|g.state_label_end.isna()|g.state_label.eq('unknown')|g.state_label_end.eq('unknown')
 tropical=g.state_label.eq('tropical')&g.state_label_end.eq('tropical')
 g['bridge_state']=np.select([conflict,unknown,tropical],['agency_conflict','unknown','both_endpoints_tropical'],default='other')
 rows=[]
 for sid,q in g.groupby('sid',sort=False):
  bridge=q[q.mask_B&q.valid_6h];rec={'sid':sid}
  for st in ['both_endpoints_tropical','other','unknown','agency_conflict']:rec['B_'+st+'_h']=float(bridge.loc[bridge.bridge_state.eq(st),'valid_h'].sum())
  rec['strict_tropical_bridge']=bool(len(bridge)==0 or bridge.bridge_state.eq('both_endpoints_tropical').all());rows.append(rec)
 d=d.merge(pd.DataFrame(rows),on='sid',validate='1:1');return d,g
