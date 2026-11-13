"""Deterministic checks of released processed values; no fit or new resampling."""
from pathlib import Path
import argparse,json,ast,hashlib,importlib.util,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
def read(p):return pd.read_csv(ROOT/p,keep_default_na=False,na_values=['','nan','NaN'],float_precision='round_trip')
def main(output):
 output=Path(output);output.mkdir(parents=True,exist_ok=True);checks=[]
 def check(name,ok,**kw):
  checks.append(dict(name=name,passed=bool(ok),**kw))
  if not ok:raise AssertionError(name)
 ids=(ROOT/'inputs/main_sid.txt').read_text().splitlines()
 check('unique frozen main SID',len(ids)==len(set(ids))==2340)
 events=read('direct_mwt/inputs/departure_events.csv')
 check('117 paired SID / 234 departure events',events.sid.nunique()==117 and len(events)==234 and set(events.sid)<=set(ids))
 check('two events per SID',events.groupby('sid').size().eq(2).all())
 for prefix,source,ref,keys in [
 ('P4 transport','water/p4/P4_transport_SID_sources.csv','water/p4/P4_transport_estimates.csv',['boundary','field','estimand','units']),
 ('P4 background','water/p4/P4_background_SID_sources.csv','water/p4/P4_background_estimates.csv',['boundary','field','estimand','units']),
 ('centre moisture budget','water/paired/centre_budget_SID_pairs.csv','water/paired/centre_budget_estimates.csv',['field','policy','estimand','units'])]:
  x=read(source);y=read(ref)
  if not all(k in x for k in keys) or 'value' not in x:continue
  z=x.groupby(keys,dropna=False).value.agg(['mean','count']).reset_index().merge(y,on=keys,validate='one_to_one')
  err=float(np.max(np.abs(z['mean']-z.estimate)))
  check(prefix+' saved means',len(z)==len(y) and np.allclose(z['mean'],z.estimate,atol=1e-11,rtol=1e-12,equal_nan=True),rows=len(z),maximum_absolute_error=err)
  check(prefix+' support counts',np.array_equal(z['count'],z.n_sid))
  if {'internal','final','value'}<=set(x):check(prefix+' temporary minus decay direction',np.allclose(x.internal-x.final,x.value,atol=1e-12,rtol=1e-12,equal_nan=True))
 x=read('water/rh/RH_matched_decomposition_SID.csv');y=read('water/rh/RH_matched_decomposition_estimates.csv');keys=['boundary','window','level_hpa','region'];checks_n=0;errors=[]
 for row in y.to_dict('records'):
  mask=np.ones(len(x),bool)
  for k in keys:mask&=x[k].eq(row[k]).to_numpy()
  vals=x.loc[mask,row['component']];checks_n+=1;errors.append(abs(vals.mean()-row['estimate']))
  check('RH '+str(checks_n),np.isclose(vals.mean(),row['estimate'],atol=1e-11,rtol=1e-12) and vals.count()==row['n_sid'])
 check('RH additive identity',np.allclose(x.q_contribution+x.T_contribution+x.p_contribution,x.local_RH_change,atol=1e-11,rtol=1e-12,equal_nan=True))
 check('RH moving support identity',np.allclose(x.local_RH_change+x.sampling_support_remainder,x.native_RH_change,atol=1e-11,rtol=1e-12,equal_nan=True))
 e=read('recovery/cohort_recovered.csv');dr=read('recovery/bootstrap_draws.csv');ref=read('recovery/effects_with_CI.csv')
 check('recovery training SID subset',e.sid.nunique()==len(e)==2231 and set(e.sid)<=set(ids))
 check('five frozen SID folds',set(e.fold)==set(range(5)))
 check('recovery stored draw IDs',set(dr.draw)==set(range(1000)))
 for row in ref.to_dict('records'):
  q=dr[(dr.policy==row['policy'])&(dr.outcome==row['outcome'])&(dr.method==row['method'])].estimate.dropna();lo,hi=np.quantile(q,[.025,.975]);check('recovery interval '+row['policy']+'/'+row['outcome']+'/'+row['method'],len(q)==row['successful_draws'] and np.allclose([lo,hi],[row['ci_low'],row['ci_high']],atol=1e-12,rtol=1e-12),successful_saved_draws=len(q))
 # Existing hemisphere correction: same per-SID values and stored multiplicities.
 x=read('water/hemisphere/selected_sector_SID_values.csv');ref=read('water/hemisphere/corrected_16cells.csv');bank=np.load(ROOT/'water/hemisphere/regions_bootstrap.npz');sid=bank['sid'];weights=bank['weights'];n=0
 for (region,sector,field),g in x.groupby(['region','corrected_sector','field']):
  a=g.set_index('sid').value.reindex(sid).to_numpy();ok=np.isfinite(a);den=weights@ok.astype(float);draw=np.divide(weights@np.nan_to_num(a),den,out=np.full(len(weights),np.nan),where=den>0);lo,hi=np.nanquantile(draw,[.025,.975]);mu=float(np.nanmean(a));b=ref[ref.region.eq(region)&ref.sector.eq(sector)&ref.field.eq(field)&ref.estimand.eq('pre_mean')].iloc[0];n+=1
  check('hemisphere corrected '+str(n),np.allclose([mu,lo,hi],[b.estimate,b.lo,b.hi],atol=1e-13,rtol=1e-12) and ok.sum()==b.n_sid)
 check('hemisphere correction cells',n==16)
 result={'status':'PASS','scope':'Saved-value deterministic validation; no ERA5 extraction, model fitting or newly sampled bootstrap draws','checks':checks,'checks_passed':len(checks),'RH_summary_rows':checks_n,'RH_mean_maximum_absolute_error':max(errors),'main_sid':len(ids),'paired_departure_sid':events.sid.nunique(),'recovery_model_sid':len(e)}
 (output/'validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='checks'}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
