# Public path adapter; the scientific functions below retain their original implementation.
import sys as _sys
from pathlib import Path as _Path
_PACKAGE=next(p for p in _Path(__file__).resolve().parents if (p/'release_paths.py').is_file())
_sys.path.insert(0,str(_PACKAGE))
from release_paths import ADDENDUM, UNIFIED, PACKAGE, bootstrap
from pathlib import Path
import runpy,sys,json,hashlib,datetime
O=Path(__file__).resolve().parent
rt=bootstrap(O/'temp')
if sys.version_info[:2]==(3,11):sys.path=[p for p in sys.path if p!=rt['vendor']]
import numpy as np
import pandas as pd
U=UNIFIED
def read(p):return pd.read_csv(p,keep_default_na=False,na_values=['','nan','NaN'])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,float) and not np.isfinite(x):return None
 return x
def js(name,x):(O/name).write_text(json.dumps(clean(x),indent=2,allow_nan=False),encoding='utf-8')
C=json.loads((O/'FROZEN_CONFIG.json').read_text());e=read(O/'frozen_analysis_inputs.csv');training=read(O/'training_inputs.csv')
e['PI_minus_V_t0_ms']=e.PI_t0_ms-e.wind_t0_kt*(1852/3600)
assert np.allclose(e.PI_minus_V_t0_ms,e.t0_PIheadroom,rtol=0,atol=1e-12)
risks=[];contrasts=[];curves=[];drawrecords=[];counts=[];errors=[];envresults=[];environments=[]
orig=read(U/'recovery_refit/cohort_recovered.csv');bank=np.load(U/'recovery_refit/bootstrap_indices.npz')['indices']
for hem in ['NH','SH']:
 d=e[e.hemisphere.eq(hem)].reset_index(drop=True);points=json.loads((O/f'{hem}_point.json').read_text());draws=[json.loads(line) for line in (O/f'{hem}_bootstrap_draws.jsonl').read_text().splitlines()]
 assert len(draws)==1000 and sorted(x['draw'] for x in draws)==list(range(1000))
 ok=[v for v in draws if v['status']=='OK'];failed=[v for v in draws if v['status']!='OK'];errors+=failed
 counts.append(dict(hemisphere=hem,n_sid=len(d),n_training_sid=int(training.hemisphere.eq(hem).sum()),observed_return=int(d.recovery48.eq(1).sum()),complete_nonreturn=int(d.recovery48.eq(0).sum()),unknown=int(d.recovery48.isna().sum()),all_eight_observed=int(d.complete48.astype(str).str.lower().eq('true').sum()),early_return_incomplete=int((d.recovery48.eq(1)&~d.complete48.astype(str).str.lower().eq('true')).sum()),zero_information=int((~np.isfinite(d.U_h)&d.L_h.eq(0)).sum()),right_censored=int((~np.isfinite(d.U_h)).sum()),successful_draws=len(ok),failed_draws=len(failed)))
 for kind in ['spline','linear']:
  p={v['model']:v for v in points[kind]};a=np.array([[v['values'][kind][m]['difference'] for m in range(2)] for v in ok])*100
  for m,name in enumerate(C['models']):
   pp=p[name];dd=np.array([[v['values'][kind][m][key] for key in ['risk_low','risk_high','natural_risk']] for v in ok])*100
   for t,key in enumerate(['risk_low','risk_high','natural_risk']):
    ci=np.quantile(dd[:,t],[.025,.975]);risks.append(dict(hemisphere=hem,kind=kind,model=name,target=key,n_sid=len(d),n_training_sid=int(training.hemisphere.eq(hem).sum()),probability_pct=100*pp[key],ci_low_pct=ci[0],ci_high_pct=ci[1],successful_draws=len(ok)))
   ci=np.quantile(a[:,m],[.025,.975]);contrasts.append(dict(hemisphere=hem,kind=kind,model=name,low_abs_lat=C['hemispheres'][hem]['low_abs_lat'],high_abs_lat=C['hemispheres'][hem]['high_abs_lat'],n_sid=len(d),n_training_sid=int(training.hemisphere.eq(hem).sum()),difference_pp=pp['difference']*100,ci_low_pp=ci[0],ci_high_pp=ci[1],successful_draws=len(ok),interpretation='higher absolute latitude minus lower absolute latitude'))
   cc=np.array([v['values'][kind][m]['risk_curve'] for v in ok])*100
   for j,lat in enumerate(pp['grid']):
    ci=np.quantile(cc[:,j],[.025,.975]);curves.append(dict(hemisphere=hem,kind=kind,model=name,abs_latitude=lat,signed_latitude=lat if hem=='NH' else -lat,probability_pct=pp['risk_curve'][j]*100,ci_low_pct=ci[0],ci_high_pct=ci[1],n_sid=len(d),n_training_sid=int(training.hemisphere.eq(hem).sum()),curve_scope='between prespecified target latitudes; both target endpoints pass assigned-fold empirical support; finite bootstrap hull variation is retained',interval_scope='pointwise95%; not simultaneous confidence band'))
  change=a[:,0]-a[:,1];ci=np.quantile(change,[.025,.975]);contrasts.append(dict(hemisphere=hem,kind=kind,model='change_after_joint_environment',low_abs_lat=C['hemispheres'][hem]['low_abs_lat'],high_abs_lat=C['hemispheres'][hem]['high_abs_lat'],n_sid=len(d),n_training_sid=int(training.hemisphere.eq(hem).sum()),difference_pp=(p[C['models'][0]]['difference']-p[C['models'][1]]['difference'])*100,ci_low_pp=ci[0],ci_high_pp=ci[1],successful_draws=len(ok),interpretation='M0 latitude difference minus M1 latitude difference; paired within draw'))
  for i,draw in enumerate(ok):drawrecords.append(dict(draw=draw['draw'],hemisphere=hem,kind=kind,M0_difference_pp=a[i,0],M1_difference_pp=a[i,1],change_after_joint_environment_pp=change[i],n_training_draw=draw['n_draw'],n_target_draw=draw['n_target_draw'],n_unique_training_sid=draw['n_unique_sid'],n_unique_target_sid=draw['n_unique_target_sid']))
 # Descriptive environmental context with the same supported basin mixture.
 # Whole observed rows are resampled. These means are not substituted in the risk model.
 basins=sorted(d.basin_genesis.unique());env=C['environment_variables']+['wind_t0_kt','PI_minus_V_t0_ms'];targets=C['hemispheres'][hem]
 low=d.abs_lat.sub(targets['low_abs_lat']).abs().le(2.5).to_numpy();high=d.abs_lat.sub(targets['high_abs_lat']).abs().le(2.5).to_numpy()
 memberships=[d.basin_genesis.eq(b).to_numpy() for b in basins]
 for b,hit in zip(basins,memberships):environments.append(dict(hemisphere=hem,basin=b,reference_sid=int(hit.sum()),local_low_reference_sid=int((hit&low).sum()),local_high_reference_sid=int((hit&high).sum())))
 def emean(w):
  out=np.zeros((2,len(env)))
  for hit in memberships:
   mixture=w[hit].sum()/w.sum()
   for k,local in enumerate([low,high]):
    ww=w*(hit&local)
    if ww.sum()==0:return np.full((2,len(env)),np.nan)
    out[k]+=mixture*np.average(d[env].to_numpy(),axis=0,weights=ww)
  return out
 ep=emean(np.ones(len(d)));index=pd.Series(np.arange(len(d)),index=d.sid);mapping=orig.sid.map(index).fillna(-1).astype(int).to_numpy();ed=[]
 for ix in bank:
  selected=mapping[ix];selected=selected[selected>=0];ed.append(emean(np.bincount(selected,minlength=len(d))))
 ed=np.asarray(ed);valid=np.isfinite(ed).all(axis=(1,2))
 for j,var in enumerate(env):
  v=ed[valid,:,j];ci=np.quantile(v,[.025,.975],axis=0);diff=v[:,1]-v[:,0];dc=np.quantile(diff,[.025,.975]);envresults.append(dict(hemisphere=hem,variable=var,unit='RH percentage points' if var=='past_RH' else ('kt' if var=='wind_t0_kt' else 'm/s'),low_mean=ep[0,j],low_ci=ci[0,0],low_high_ci=ci[1,0],high_mean=ep[1,j],high_low_ci=ci[0,1],high_high_ci=ci[1,1],high_minus_low=ep[1,j]-ep[0,j],difference_ci_low=dc[0],difference_ci_high=dc[1],valid_draws=int(valid.sum()),description='basin-mixture-standardized descriptive pre-entry/prediction-time conditions within local latitude neighborhoods; PI minus V descriptive only and absent from model; not a component causal contribution'))
pd.DataFrame(risks).to_csv(O/'latitude_standardized_risks.csv',index=False)
pd.DataFrame(contrasts).to_csv(O/'latitude_probability_contrasts.csv',index=False)
pd.DataFrame(curves).to_csv(O/'latitude_standardized_curves.csv',index=False)
pd.DataFrame(drawrecords).to_csv(O/'paired_latitude_bootstrap_contrasts.csv',index=False)
pd.DataFrame(counts).to_csv(O/'analysis_outcome_support.csv',index=False)
pd.DataFrame(envresults).to_csv(O/'environment_descriptive_context.csv',index=False)
pd.DataFrame(environments).to_csv(O/'environment_context_support.csv',index=False)
ledger=read(O.parent/'historical_risk_entry_return_locations.csv');population=[]
for scope,z in [('main2340',ledger),('accepted2231',ledger[ledger.accepted_model]),('environment1256',ledger[ledger.environment_complete]),('training1245',ledger[ledger.sid.isin(training.sid)]),('common406',ledger[ledger.common_support]),('fold_supported322',ledger[ledger.sid.isin(e.sid)])]:
 for hem,g in list(z.groupby('entry_hemisphere'))+[('ALL',z)]:
  population.append(dict(scope=scope,hemisphere=hem,n_sid=len(g),observed_return48=int(g.return48.eq(1).sum()),known_complete_nonreturn48=int(g.return48.eq(0).sum()),unknown48=int(g.return48.isna().sum()),outcome='scheduled 48h return after first historical-threshold departure'))
pd.DataFrame(population).to_csv(O/'scope_population_counts.csv',index=False)
js('BOOTSTRAP_FAILURES.json',errors)
js('NUMERIC_RECEIPT.json',dict(status='V2_CORRECTED_TRAINING_COMPUTED_PENDING_INDEPENDENT_REVIEW',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),models=counts,primary=[r for r in contrasts if r['kind']=='spline'],sensitivity=[r for r in contrasts if r['kind']=='linear'],failure_n=len(errors),config_sha256=sha(O/'FROZEN_CONFIG.json'),source_bank_sha256=sha(U/'recovery_refit/bootstrap_indices.npz'),outputs={p.name:sha(p) for p in O.glob('*.csv')},hypothesis_tests='No new p-value/multiplicity family; paired effect estimates with95% percentile intervals, pointwise curve intervals. No significance search or explained fraction.'))
print(pd.DataFrame(contrasts).to_string(index=False));print(pd.DataFrame(counts).to_string(index=False));print(pd.DataFrame(envresults).to_string(index=False))
