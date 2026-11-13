"""Deterministic checks of bundled counts, inference families and saved draws.

This command does not fit a model, draw bootstrap samples, read ERA5 or plot.
"""
from pathlib import Path
import argparse,ast,hashlib,json,sys
import numpy as np
import pandas as pd
from scipy import stats
R=Path(__file__).resolve().parent
def read(p):return pd.read_csv(p,keep_default_na=False,na_values=['','nan','NaN'],float_precision='round_trip')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();checks=[]
 def ck(name,ok,detail=None):
  checks.append(dict(check=name,passed=bool(ok),detail=detail));assert bool(ok),name
 for p in R.rglob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'))
 ck('all Python sources parse',True)
 # Load only pure numerical functions from the delivered analysis entry point.
 tree=ast.parse((R/'history/agents/B/history_counts_trends.py').read_text())
 nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['bh','hac']]
 ns={'np':np,'stats':stats};exec(compile(ast.Module(body=nodes,type_ignores=[]),'<published B functions>','exec'),ns)
 bh,hac=ns['bh'],ns['hac']
 master=read(R/'inputs/addendum/unified_main_20261023_1642/source_data/main_sid.csv')
 events=read(R/'inputs/addendum/unified_main_20261023_1642/source_data/all_return_events.csv')
 ck('main cohort SID count',len(master)==master.sid.nunique()==2340)
 ck('research years',master.genesis_year.min()==1982 and master.genesis_year.max()==2023)
 ck('anchored returns',len(events)==133 and events.sid.nunique()==123)
 a=R/'history/agents/A';den=read(a/'sid_denominators.csv');risk=read(a/'risk_intervals_6h.csv');ref=read(a/'first_departure_reference.csv')
 ck('unique returning SID and repeated-event denominator',den.ever_return.sum()==123 and den.n_return.sum()==133 and len(den)==2340)
 ck('six-hour opportunity intervals',risk.hours.eq(6).all() and risk.return_event.sum()==133)
 counts=read(a/'basin_counts.csv')
 for row in counts.itertuples():
  rr=risk[risk.basin_end.eq(row.basin)]
  ck('opportunity denominator '+row.basin,int(rr.hours.sum())==row.cyclone_hours and int(rr.return_event.sum())==row.return_events)
 b=R/'history/agents/B';annual=read(b/'annual_basin_numerators_denominators.csv')
 glob=annual[annual.basin.eq('GLOBAL')].sort_values('year')
 ck('global annual independent-SID numerator',glob.n_return_sid.sum()==123 and glob.n_sid.sum()==2340 and glob.n_return_events.sum()==133)
 ck('global proportion uses aggregate SID counts',np.allclose(glob.rate_pp,100*glob.n_return_sid/glob.n_sid,rtol=0,atol=1e-12))
 trends=read(b/'annual_trends_and_basin_interactions.csv');ck('complete 11-test B BH family',len(trends)==11 and np.allclose(bh(trends.p_value.to_numpy()),trends.q_value,rtol=0,atol=2e-14))
 coef,cov=hac(glob.rate_pp.to_numpy(),glob.year.to_numpy());se=np.sqrt(cov[1,1]);ci=coef[1]+np.array([-1,1])*stats.t.ppf(.975,40)*se;g=trends[trends.kind.eq('global slope')].iloc[0]
 ck('latest global trend and interval replay',np.allclose([coef[1],*ci],[g.estimate,g.lo,g.hi],rtol=0,atol=1e-12),[float(coef[1]),*map(float,ci)])
 c=R/'history/agents/C';ct=read(c/'standardized_duration_trends.csv')
 ck('exact complete 16-test C BH family',len(ct)==16 and ct.family.nunique()==1 and np.allclose(bh(ct.p_boot.to_numpy()),ct.q_BH_new_family,rtol=0,atol=2e-14))
 f=read(c/'F_components_sid.csv');ck('F=R+L+B for every SID',len(f)==2340 and np.array_equal(f.T_R_h+f.T_L_h+f.T_B_h,f.T_F_h))
 for st in ['D','E','F','Res']:
  z=ct[ct.stage.eq(st)]
  bank=np.load(c/f'model_bootstrap_{st}.npz');draws=bank['trends'][:,:,0]
  for j,row in enumerate(z.itertuples()):
   d=draws[:,j];d=d[np.isfinite(d)];lo,hi=np.percentile(d,[2.5,97.5])
   ck('C interval from saved draws '+st+' '+row.target,len(d)==row.valid_draws and np.allclose([lo,hi],[row.ci_low,row.ci_high],rtol=0,atol=1e-11))
 lat=R/'spatial/agents/latitude_risk/v2_fold_supported';tr=read(lat/'training_inputs.csv');rr=read(lat/'frozen_analysis_inputs.csv');sr=json.loads((lat/'SUPPORT_RECEIPT.json').read_text())
 ck('latitude training and fixed target support',len(tr)==tr.sid.nunique()==1245 and len(rr)==rr.sid.nunique()==322 and tr.target_reference.sum()==322 and set(rr.sid)==set(tr.loc[tr.target_reference,'sid']))
 ck('inherited five SID folds',set(tr.fold)==set(range(5)))
 for name,key in [('FROZEN_CONFIG.json','configuration_sha256'),('training_inputs.csv','training_inputs_sha256'),('frozen_analysis_inputs.csv','support_inputs_sha256')]:ck('latitude frozen hash '+name,hashlib.sha256((lat/name).read_bytes()).hexdigest()==sr[key])
 ledger=read(R/'spatial/agents/latitude_risk/historical_risk_entry_return_locations.csv')
 ck('first-entry risk population and unknown outcomes',len(ledger)==ledger.sid.nunique()==2340 and ledger.return48.eq(1).sum()==314 and ledger.return48.eq(0).sum()==1351 and ledger.return48.isna().sum()==675)
 lc=read(lat/'latitude_probability_contrasts.csv')
 for hem in ['NH','SH']:
  rows=[json.loads(s) for s in (lat/f'{hem}_bootstrap_draws.jsonl').read_text().splitlines()]
  ck('1000 unchanged successful latitude draw IDs '+hem,len(rows)==1000 and sorted(v['draw'] for v in rows)==list(range(1000)) and all(v['status']=='OK'for v in rows))
  pts=json.loads((lat/f'{hem}_point.json').read_text())
  for kind in ['spline','linear']:
   aa=np.array([[r['values'][kind][j]['difference'] for j in range(2)]for r in rows])*100
   for j,model in enumerate(['M0_state','M1_state_joint_environment','change_after_joint_environment']):
    row=lc[lc.hemisphere.eq(hem)&lc.kind.eq(kind)&lc.model.eq(model)].iloc[0]
    vals=aa[:,j]if j<2 else aa[:,0]-aa[:,1];ci=np.quantile(vals,[.025,.975])
    ck('latitude paired interval '+hem+' '+kind+' '+model,np.allclose(ci,[row.ci_low_pp,row.ci_high_pp],rtol=0,atol=1e-12))
 result=dict(status='PASS',checks_passed=len(checks),checks=checks,scope='Syntax; frozen cohort and denominator identities; exact 11- and 16-test BH families; existing global HAC estimate; F decomposition; C and latitude saved-draw intervals. No model fitting, new resampling, ERA5/MWT processing or figure rendering.')
 args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps({'status':'PASS','checks':len(checks),'output':str(args.output)}))
if __name__=='__main__':main()
