# Public path adapter; the scientific functions below retain their original implementation.
import sys as _sys
from pathlib import Path as _Path
_PACKAGE=next(p for p in _Path(__file__).resolve().parents if (p/'release_paths.py').is_file())
_sys.path.insert(0,str(_PACKAGE))
from release_paths import ADDENDUM, UNIFIED, PACKAGE, bootstrap
from analyze_duration import *
import statsmodels.api as sm
def main():
 bank=np.load(R/'joint_year_sid_draws.npz');counts=bank['counts'];ny=bank['year_multiplicity'];sid_index={s:i for i,s in enumerate(bank['sid'])}
 spec=json.loads((R/'MODEL_SPEC_AND_AUDIT.json').read_text());raw=read(R/'stage_entry_source_frozen.csv');raw['lon_sin_entry']=np.sin(np.deg2rad(raw.entry_track_lon));raw['lon_cos_entry']=np.cos(np.deg2rad(raw.entry_track_lon))
 cv=read(R/'stage_duration_coverage_sid.csv');raw=raw.merge(cv[['sid','stage','covered_h','uncovered_h']],on=['sid','stage'],validate='1:1');support=read(R/'support_sid_stage.csv');checks=[];sens=[];diagnostic=[]
 for model in spec['models']:
  st=model['stage'];state=model['state'];g=read(R/f'common_support_{st}.csv');xd=read(R/f'design_{st}.csv');X=xd.drop(columns=['sid','weight']).to_numpy(float);Y=g.duration_h.to_numpy();w=xd.weight.to_numpy()
  independent=sm.WLS(Y,X,weights=w).fit();stored=read(R/'standardized_duration_trends.csv');target=stored[stored.stage.eq(st)&stored.target.eq(TARGETS[-1])].slope.iloc[0]
  assert abs(independent.params[1]-target)<1e-10
  checks.append(dict(test='independent_statsmodels_WLS_'+st,passed=True,error=abs(independent.params[1]-target),min_individual_prediction=float(independent.fittedvalues.min()),negative_individual_predictions=int((independent.fittedvalues<0).sum())))
  q=read(R/'standardized_annual_predictions.csv');q=q[q.stage.eq(st)]
  for tar,aa in q.groupby('target'):
   check=hac(aa.mean_h,aa.genesis_year)['slope'];reported=stored[stored.stage.eq(st)&stored.target.eq(tar)].slope.iloc[0];assert abs(check-reported)<1e-9
  checks.append(dict(test='annual_prediction_trends_'+st,passed=True))
  c=raw[(raw.stage==st)&raw.sid.isin(support[(support.stage==st)&support.complete].sid)].copy()
  cases=[('complete_case_sensitivity',c)]
  if st=='Res':cases.append(('exclude_missing_wind_tail',g[g.uncovered_h.eq(0)].copy()))
  for scope,h in cases:
   h=h.sort_values(['genesis_year','sid']).reset_index(drop=True);xx,names,groups,scale,mean=design(h,state);y=h[['duration_h','covered_h']].to_numpy();yc=h.genesis_year.to_numpy(int)-1982;ww=1/np.bincount(yc,minlength=42)[yc];_,tr,_,_=fit(xx,y,ww,groups)
   ix=np.array([sid_index[s]for s in h.sid]);draw=np.full((B,4),np.nan)
   for b in range(B):
    cc=counts[b,ix].astype(float);ss=np.bincount(yc,weights=cc,minlength=42);wb=cc*np.divide(ny[b,yc],ss[yc],out=np.zeros(len(h)),where=ss[yc]>0)
    try:draw[b]=fit(xx,y,wb,groups)[1][:,0]
    except (ValueError,np.linalg.LinAlgError):pass
   for j,tar in enumerate(TARGETS):sens.append(dict(stage=st,scope=scope,target=tar,n_sid=len(h),slope=tr[j,0],**interval(draw[:,j]),interval_scope='marginal percentile conditional on successful fixed attempts'))
   np.savez_compressed(R/f'sensitivity_draws_{st}_{scope}.npz',trends=draw)
  for bas,qq in g.groupby('basin_genesis'):
   for per,qqq in qq.groupby('period'):diagnostic.append(dict(stage=st,basin=bas,period=per,n_sid=len(qqq),land_at_entry=int(qqq.entry_land_current_v3.sum()),n_years=qqq.genesis_year.nunique()))
 # Original F family is reused unchanged; our replay must equal every reported
 # R, U, F, L, B slope and interval before the old q values are attached.
 old=read(U/'statistics/ruf/annual_slopes.csv');replay=read(R/'F_components_trends_HAC.csv');fr=[]
 for comp in ['R','L','B','F']:
  o=old[(old.scope=='main_all')&(old.metric==comp)].iloc[0];n=replay[replay.component.eq('T_'+comp+'_h')].iloc[0]
  for c,oldc in [('slope','estimate'),('ci_low','ci_low'),('ci_high','ci_high')]:assert abs(n[c]-o[oldc])<1e-10
  fr.append(dict(component=comp,n_sid=2340,estimate=o.estimate,ci_low=o.ci_low,ci_high=o.ci_high,p=o.p,q=o.q,family=o.family,family_size=5,origin='existing N1 inference independently replayed; unchanged',unit='h/decade'))
 save(pd.DataFrame(fr),'F_components_original_N1_replayed.csv');checks.append(dict(test='original_N1_F_components_and_intervals',passed=True))
 # Whole frozen-cohort Dec recording audit: annual means and original HAC3.
 dc=cv[cv.stage.eq('Res')].copy();aa=dc.groupby('genesis_year').agg(n_sid=('sid','size'),elapsed_h=('duration_h','mean'),covered_h=('covered_h','mean'),tail_missing_wind_h=('uncovered_h','mean'),tail_missing_fraction=('uncovered_h',lambda x:(x>0).mean())).reset_index();save(aa,'Dec_observation_annual.csv');ob=[]
 for c in ['elapsed_h','covered_h','tail_missing_wind_h','tail_missing_fraction']:ob.append(dict(metric=c,n_sid=2340,**hac(aa[c],aa.genesis_year),unit='fraction/decade'if c.endswith('fraction')else'h/decade'))
 no=dc[dc.uncovered_h.eq(0)].groupby('genesis_year').duration_h.mean();ob.append(dict(metric='elapsed_h_no_missing_wind_tail',n_sid=int(dc.uncovered_h.eq(0).sum()),**hac(no,no.index),unit='h/decade'))
 save(pd.DataFrame(ob),'Dec_observation_trends_HAC.csv');save(pd.DataFrame(sens),'sensitivity_intervals.csv');save(pd.DataFrame(diagnostic),'common_support_basin_period.csv')
 # Constant current-land indicator is explicitly the recorded reason for the
 # E bootstrap rank failures; no replacement or concealed pseudo-inverse fit.
 failures=read(R/'bootstrap_draw_status.csv');failures=failures[failures.status.ne('ok')];assert len(failures)==85;assert set(failures.zero_variation_columns)=={'entry_land_current_v3'}
 checks.append(dict(test='all_85_E_failed_draws_identified',passed=True,reason='no land-at-entry case in sampled draw; fixed design not estimable'))
 v=dict(status='INDEPENDENT_NUMERIC_VALIDATION_PASS',checks=checks,original_boundaries_unchanged=True,sensitivity_intervals_complete=True,source_read_only=True)
 (R/'VALIDATION.json').write_text(json.dumps(v,indent=2),encoding='utf8');print(json.dumps(v,indent=2));print(pd.DataFrame(ob).to_string(index=False));print(pd.DataFrame(sens).to_string(index=False))
if __name__=='__main__':main()
