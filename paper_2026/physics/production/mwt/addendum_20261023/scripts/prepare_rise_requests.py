from pathlib import Path
import sys,json,hashlib,datetime
import pandas as pd,numpy as np
W=Path(__file__).absolute().parents[1];sys.path.insert(0,str(W/'scripts'))
from build_rise_comparable_requests import build_rise_comparable_requests
S=W/'source_data'; names=['new_return_events.parquet','new_ascent_observations.parquet','new_resolution_observations.parquet','new_intervals.parquet','new_main_observations.parquet']
data=[pd.read_parquet(S/n) for n in names]
data[0]=data[0][data[0].ledger_eligible].copy()
out=build_rise_comparable_requests(*data)
checks=[]
def check(name,ok,detail=None):
 if not bool(ok):raise AssertionError(name)
 checks.append({'check':name,'passed':True,'detail':detail})
p=out['pairs'];r=out['requests'];c=out['coverage'];good=p[p.matched]
check('one_matching_row_per_event_bin',not p.duplicated(['event_id','wind_bin']).any())
check('all_source_events_retained',set(c.event_id)==set(data[0].event_id))
check('all_exact_requests_unique',not r.duplicated(['sid','UTC']).any())
check('no_position_extrapolation',not r.position_extrapolated.any())
check('future_Resolution_only',(good.UTC_Resolution>good.UTC_rise).all())
check('existing10kt_bin_mismatch_bound',good.wind_rise_minus_Resolution_kt.abs().lt(10).all())
# Independently reconstruct all representative selections by explicit min and time tie.
a=out['candidates']
for side in ['rise','Resolution']:
 k=['sid','event_id','wind_bin'] if side=='rise' else ['sid','wind_bin']
 for _,g in a[a.side.eq(side)].groupby(k):
  chosen=g[g.representative];best=g[g.distance_to_midpoint_kt.eq(g.distance_to_midpoint_kt.min())]
  expected=best.UTC.min() if side=='rise' else best.UTC.max()
  if len(chosen)!=1 or chosen.UTC.iloc[0]!=expected:raise AssertionError('Representative mismatch')
check('all_representative_ties_reconstructed',True)
# Actual data exact positional checks, including wrap-around-safe longitudes.
t=data[-1].copy();t['UTC']=pd.to_datetime(t.time,utc=True)
x=r[r.actual_track_point].merge(t[['sid','UTC','lat','lon','wind']],on=['sid','UTC'],suffixes=('_req','_source'),validate='one_to_one')
check('all_positions_source_exact',np.array_equal(x.lat_req,x.lat_source,equal_nan=True) and np.array_equal(x.lon_req,x.lon_source,equal_nan=True))
check('all_winds_source_exact',np.array_equal(x.wind_kt,x.wind,equal_nan=True))
for key,d in out.items():
 name={'pairs':'rise_comparable_state_pairs','coverage':'rise_matching_coverage','roles':'rise_point_roles','requests':'rise_unique_point_requests','candidates':'rise_matching_candidates'}[key]
 d.to_csv(S/(name+'.csv'),index=False,encoding='utf-8-sig');d.to_parquet(S/(name+'.parquet'),index=False)
receipt={'status':'WIND_ONLY_REQUESTS_COMPLETE_NO_ENVIRONMENT_OR_ENERGY_OUTCOMES_READ','created_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'n_source_events':len(data[0]),'n_source_SID':data[0].sid.nunique(),'n_matched_event':good.event_id.nunique(),'n_matched_SID':good.sid.nunique(),'n_matched_bins':len(good),'n_unique_point_requests':len(r),'n_positions_available':int(r.position_available.sum()),'n_missing_positions':int((~r.position_available).sum()),'n_request_SID':r.sid.nunique(),'max_abs_wind_mismatch_kt':float(good.wind_rise_minus_Resolution_kt.abs().max()),'unmatched_events':c.loc[c.matched_bin_count.eq(0),'event_id'].tolist(),'checks':checks,'input_sha256':{n:hashlib.sha256((S/n).read_bytes()).hexdigest() for n in names},'specification_sha256':hashlib.sha256((W/'config/rise_analysis_contract.json').read_bytes()).hexdigest(),'script_sha256':{n:hashlib.sha256((W/'scripts'/n).read_bytes()).hexdigest() for n in ['build_rise_comparable_requests.py','prepare_rise_requests.py']}}
(W/'reports/rise_request_build_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in receipt.items() if k not in ['checks','input_sha256','script_sha256','unmatched_events']},indent=2))
