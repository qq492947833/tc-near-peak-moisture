"""P3 environment aggregation from complete v5 extraction/thermal caches only.
All requested rows must be terminal; no scientific fitting, imputation, or nc access here.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,sys,json,hashlib,traceback
import numpy as np
import pandas as pd
from p1_p2_analysis_v5 import verify_folder,boolean,number,utc
V5=Path(__file__).resolve().parents[1]
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def csv(p):return pd.read_csv(p,keep_default_na=False,na_values=[''],low_memory=False)
def main():
 p=argparse.ArgumentParser();p.add_argument('--cache-index',required=True);p.add_argument('--thermodynamics',required=True);p.add_argument('--out',required=True);a=p.parse_args()
 out=Path(a.out).resolve();assert V5.resolve() in out.parents
 hold_path=V5/'checkpoints/P3_THERMO_HOLD.json'
 if hold_path.exists() and read(hold_path).get('status')=='ACTIVE':
  blocked=dict(status='BLOCKED_PHYSICAL_INPUT_REVIEW',UTC=now(),hold_file=str(hold_path),hold_sha256=sha(hold_path),requested_out=str(out),real_effect_fits=0,exit_code=42)
  (out.parent/('P3_input_gate_hold_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.json')).write_text(json.dumps(blocked,indent=2),encoding='utf-8')
  print(json.dumps(blocked),flush=True);sys.exit(42)
 assert not out.exists();out.mkdir(parents=True);start=now();checks=[];prov=[];rows=[]
 def ck(name,value):
  checks.append({'test':name,'passed':bool(value)})
  if not value:raise AssertionError(name)
 inputs={k:{'path':str(v),'sha256':sha(v)} for k,v in {'cache_index':Path(a.cache_index),'thermodynamics':Path(a.thermodynamics),'source_requests':V5/'data/p3_environment_request_sources.csv','combined_requests':V5/'data/environment_requests_v5.csv','analysis_spec':V5/'analysis_spec_v5.json'}.items()}
 try:
  original=csv(inputs['source_requests']['path']);original['extraction_request_id']='P3|'+original.request_id
  requests=csv(inputs['combined_requests']['path']);requests=requests[requests.module.eq('P3')].copy()
  ck('exact_13085_requested_rows_2617_events',len(requests)==13085 and requests.event_id.nunique()==2617)
  ck('unique_bijective_source_and_combined_ID',not requests.request_id.duplicated().any() and set(requests.request_id)==set(original.extraction_request_id))
  ck('all_exact_past_and_t0_lags',requests.groupby('event_id').lag_h.apply(lambda x:set(x)=={-24,-18,-12,-6,0} and len(x)==5).all())
  oi=original.set_index('extraction_request_id');idx=csv(a.cache_index);idx=idx[idx.request_id.isin(requests.request_id)]
  ck('cache_index_unique',not idx.request_id.duplicated().any());missing=sorted(set(requests.request_id)-set(idx.request_id))
  invalid=idx[~idx.status.isin(['COMPLETE','REUSED_VERIFIED_CHECKPOINT'])]
  th=csv(a.thermodynamics);th=th[th.request_id.isin(requests.request_id)];ck('thermodynamic_index_unique',not th.request_id.duplicated().any())
  missingthermal=sorted(set(requests.loc[requests.lag_h.eq(0),'request_id'])-set(th.request_id))
  if missing or len(invalid) or missingthermal:
   pd.DataFrame({'request_id':missing,'reason':'EXTRACTION_NOT_COMPLETE'}).to_csv(out/'missing_extraction_requests.csv',index=False)
   pd.DataFrame({'request_id':missingthermal,'reason':'T0_THERMODYNAMIC_DERIVATION_NOT_COMPLETE'}).to_csv(out/'missing_thermal_requests.csv',index=False)
   invalid.to_csv(out/'nonterminal_requests.csv',index=False)
   (out/'readiness.json').write_text(json.dumps(dict(status='WAITING_FOR_REQUIRED_REQUESTS',missing_extractions=len(missing),nonterminal_extractions=len(invalid),missing_t0_thermal=len(missingthermal),started_utc=start,finished_utc=now(),inputs=inputs,real_fits=0,raw_ERA5_reads=0),indent=2),encoding='utf-8')
   print(json.dumps({'status':'WAITING_FOR_REQUIRED_REQUESTS','missing_extraction':len(missing),'missing_thermal':len(missingthermal)}));return
  idx=idx.set_index('request_id');th.to_csv(out/'thermodynamics_P3_snapshot.csv',index=False,encoding='utf-8-sig');ti=th.set_index('request_id')
  driver=read(V5/'audit/parallel_extraction_20261008T142439_1440384_all/launch.json')['source_identity']['driver_sha256']
  ck('driver_unchanged',sha(V5/'src/extract_environment_v5.py')==driver)
  for z in requests.to_dict('records'):
   rid=z['request_id'];s=oi.loc[rid];lag=int(z['lag_h']);t0=utc(z['t0']);u=utc(z['UTC']);f=Path(idx.loc[rid,'folder'])
   ck(rid+' source_identity',str(z['event_id'])==str(s.event_id) and str(z['sid'])==str(s.sid) and u==utc(s.UTC) and t0==utc(s.t0))
   ck(rid+' exact_past_information_time',u==t0+pd.Timedelta(hours=lag) and u<=t0 and utc(s.position_max_source_UTC)<=u)
   if boolean(s.motion_valid):ck(rid+' past_motion_time',utc(s.motion_max_source_UTC)<=u and utc(s.motion_previous_UTC)==u-pd.Timedelta(hours=6))
   ck(rid+' real_position_no_interpolation',boolean(s.observed_position) and not boolean(s.interpolated) and not boolean(z['position_extrapolated']) and z['center_kind']=='actual_track')
   for c in ['lat','lon','wind_kt']:ck(rid+' '+c,np.isclose(number(z[c]),number(s[c]),rtol=0,atol=1e-9,equal_nan=True))
   cp=verify_folder(f,z,driver,False)
   for o in cp['outputs']:ck(rid+' file '+o['name'],sha(f/o['name'])==o['sha256'] and (f/o['name']).stat().st_size==o['size'])
   sc=csv(f/'scalars.csv');ck(rid+' scalar_unique_identity',not sc.variable.duplicated().any() and set(sc.request_id)=={rid});sc=sc.set_index('variable')
   for variable,name,unit in [('shear200_850','shear_200_850_vector_magnitude_ms','m s-1'),('midRH','mid_rh_pct','%'),('SST','sst_C','degC')]:
    ck(rid+' required_'+name,name in sc.index);b=sc.loc[name];ck(rid+' unit_'+name,b.units==unit)
    support=number(b.valid_fraction);good=boolean(b.support_pass) and np.isfinite(number(b.value)) and support>=.9
    ck(rid+' support_fraction_'+name,not np.isfinite(support) or 0<=support<=1+1e-12)
    if variable=='midRH':
     layers=sc.loc[['r500','r700']];layer_good=all(boolean(x) for x in layers.support_pass) and (pd.to_numeric(layers.valid_fraction)>=.9).all()
     ck(rid+' RH_two_layers',not good or layer_good)
     if good:ck(rid+' RH_layer_mean',np.isclose(number(b.value),pd.to_numeric(layers.value).mean(),rtol=1e-12,atol=1e-12))
    rows.append(dict(request_id=rid,source_request_id=s.request_id,event_id=z['event_id'],sid=z['sid'],lag_h=lag,UTC=z['UTC'],variable=variable,value=number(b.value) if good else np.nan,support=support if good else 0.,valid_area_m2=b.valid_area_m2,original_area_m2=b.original_area_m2,units=unit,qc_reason=b.reason,source_folder=str(f),checkpoint_sha256=sha(f/'COMPLETE.json')))
   if lag==0:
    tr=ti.loc[rid];dp=Path(tr.derived_complete_path);ck(rid+' thermo_CP_sha',sha(dp)==tr.derived_complete_sha256);dc=read(dp)
    ck(rid+' thermo_source_binding',dc['status']=='COMPLETE' and dc['binding']['source']['checkpoint_sha256']==sha(f/'COMPLETE.json'))
    for o in dc['outputs']:ck(rid+' thermo '+o['name'],sha(dp.parent/o['name'])==o['sha256'])
    d=read(dp.parent/'derived.json');ck(rid+' thermo_values',d['request_id']==rid and np.isclose(number(tr.thermal_headroom_ms),number(d['thermal_headroom_ms']),rtol=1e-12,atol=1e-12,equal_nan=True))
    good=boolean(tr.pi_valid) and np.isfinite(number(tr.thermal_headroom_ms))
    ck(rid+' concurrent_t0_wind',utc(tr.wind_source_UTC)==u and np.isclose(number(tr.wind_for_headroom_kt),number(z['wind_kt']),atol=1e-9,equal_nan=True))
    if good:ck(rid+' headroom_definition',np.isclose(number(tr.thermal_headroom_ms),number(tr.pi_vmax_ms)-number(z['wind_kt'])*.5144444444444445,rtol=1e-12,atol=1e-9))
    rows.append(dict(request_id=rid,source_request_id=s.request_id,event_id=z['event_id'],sid=z['sid'],lag_h=lag,UTC=z['UTC'],variable='PIheadroom',value=number(tr.thermal_headroom_ms) if good else np.nan,support=float(good),valid_area_m2=np.nan,original_area_m2=np.nan,units='m s-1',qc_reason=tr.pi_quality_status,source_folder=str(dp.parent),checkpoint_sha256=sha(dp)))
   prov.append(dict(request_id=rid,folder=str(f),checkpoint_sha256=sha(f/'COMPLETE.json'),source_line_number=s.source_line_number))
   if len(prov)%1000==0:print(json.dumps({'at_utc':now(),'verified_requests':len(prov)}),flush=True)
  for k,v in inputs.items():ck(k+' unchanged',sha(v['path'])==v['sha256'])
  table=pd.DataFrame(rows);ck('complete_3_timeseries_plus_t0_headroom',len(table)==13085*3+2617 and not table.duplicated(['event_id','lag_h','variable']).any())
  table.to_csv(out/'P3_environment_exposures.csv',index=False,encoding='utf-8-sig');pd.DataFrame(prov).to_csv(out/'source_provenance.csv',index=False,encoding='utf-8-sig')
  status='PASS_P3_ALL_REQUESTED_EXTRACTIONS_AND_SOURCE_VALUES_VERIFIED';code=0
 except Exception:
  status='FAIL_P3_SOURCE_VERIFICATION';code=1;checks.append({'test':'exception','passed':False,'detail':traceback.format_exc()})
 pd.DataFrame(checks).to_csv(out/'checks.csv',index=False,encoding='utf-8-sig')
 report=dict(status=status,started_utc=start,finished_utc=now(),inputs=inputs,argv=sys.argv,code_sha256=sha(__file__),exit_code=code,check_count=len(checks),failed_checks=sum(not x['passed'] for x in checks),verified_requests=len(prov),raw_ERA5_reads=0,real_effect_fits=0)
 if code==0:report['exposures']={'path':str(out/'P3_environment_exposures.csv'),'sha256':sha(out/'P3_environment_exposures.csv')}
 (out/'source_verification_receipt.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report));sys.exit(code)
if __name__=='__main__':main()