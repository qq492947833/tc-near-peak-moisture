# Public path adapter; the scientific functions below retain their original implementation.
import sys as _sys
from pathlib import Path as _Path
_PACKAGE=next(p for p in _Path(__file__).resolve().parents if (p/'release_paths.py').is_file())
_sys.path.insert(0,str(_PACKAGE))
from release_paths import ADDENDUM, UNIFIED, PACKAGE, bootstrap
"""New C analyses: common-support hours-scale standardization; no old model refit.

All source files are read only. Each stage is one row per original SID. Stage
entry environmental exposures are the verified fixed-location -24:-6 h values.
Intervals resample three-year circular blocks jointly, then whole SIDs within
each selected genesis year. No future endpoint covariate enters a duration fit.
"""
from pathlib import Path
import runpy,json,sys,hashlib,time
R=Path(__file__).resolve().parent
bootstrap(R/'temp')
import numpy as np,pandas as pd
from scipy import stats
from audit_duration import read,save,hac,U
SEED=2026110903;B=2000
ENV=['X1_sst_c','X2_shear_ms','X3_mid_rh_pct']
YEARS=np.arange(1982,2024);STAGES=['D','E','F','Res'];TARGETS=['actual_composition','fixed_basin_season','fixed_basin_season_state','fixed_basin_season_state_environment']
def bh(p):
 p=np.asarray(p,float);o=np.argsort(p);q=np.empty(len(p));q[o]=np.minimum.accumulate((p[o]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1];return np.minimum(q,1)
def interval(v):
 v=np.asarray(v,float);v=v[np.isfinite(v)]
 return dict(ci_low=float(np.percentile(v,2.5)),ci_high=float(np.percentile(v,97.5)),p_boot=float(min(1,2*min((1+(v<=0).sum())/(len(v)+1),(1+(v>=0).sum())/(len(v)+1)))),valid_draws=len(v))
def build_bootstrap(master):
 rng=np.random.default_rng(SEED);sid_order=master.sid.tolist();yb=master.genesis_year.to_numpy(int)-1982
 counts=np.zeros((B,len(master)),np.uint16);ny=np.zeros((B,42),np.uint16)
 for b in range(B):
  starts=rng.integers(0,42,size=14);selected=((starts[:,None]+np.arange(3))%42).ravel();yy=np.bincount(selected,minlength=42);ny[b]=yy
  for y,k in enumerate(yy):
   if k:
    ids=np.flatnonzero(yb==y);counts[b,ids]=rng.multinomial(int(k)*len(ids),np.repeat(1/len(ids),len(ids)))
 np.savez_compressed(R/'joint_year_sid_draws.npz',counts=counts,year_multiplicity=ny,sid=np.asarray(sid_order),seed=SEED,block_years=3)
 return counts,ny
def design(g,state):
 data={'intercept':np.ones(len(g)),'decade':(g.genesis_year.to_numpy(float)-2002.5)/10};groups={k:[]for k in ['geo','state','env']}
 for bas in ['EP','WP','NI','SI','SP']:
  c='basin_'+bas;data[c]=g.basin_genesis.eq(bas).to_numpy(float);groups['geo'].append(c)
 # Two harmonics represent the seasonal cycle without arbitrary season bins.
 for h in [1,2]:
  for fun in ['sin','cos']:
   c=f'month_{fun}_{h}';data[c]=getattr(np,fun)(2*np.pi*h*(g.genesis_month.to_numpy(float)-1)/12);groups['geo'].append(c)
 for c in state: data[c]=g[c].to_numpy(float);groups['state'].append(c)
 for c in ENV:data[c]=g[c].to_numpy(float);groups['env'].append(c)
 names=[c for c,v in data.items() if c=='intercept' or np.std(v)>1e-12]
 X=np.column_stack([data[c]for c in names]);means=X[:,2:].mean(0);scales=X[:,2:].std(0);X[:,2:]=(X[:,2:]-means)/scales
 groups={k:[names.index(c)for c in cs if c in names]for k,cs in groups.items()}
 return X,names,groups,np.r_[1.,1.,scales],np.r_[0.,0.,means]
def fit(X,Y,w,groups):
 n,k=X.shape;A=X.T@(X*w[:,None]);rhs=X.T@(Y*w[:,None])
 if np.linalg.matrix_rank(A,tol=np.linalg.norm(A,ord=2)*1e-12)<k:raise ValueError('rank deficient')
 coef=np.linalg.solve(A,rhs);tw=w/w.sum();tt=X[:,1];tmean=tw@tt;den=tw@((tt-tmean)**2);feature_slope=((tw*(tt-tmean))@X)/den
 contrib={group:feature_slope[ix]@coef[ix,:]for group,ix in groups.items()}
 actual=feature_slope@coef
 trends=np.stack([actual,actual-contrib['geo'],actual-contrib['geo']-contrib['state'],coef[1]])
 return coef,trends,contrib,feature_slope
def main():
 t=time.time();master=read(U/'source_data/main_sid.csv');entry=read(R/'stage_entry_source_frozen.csv');coverage=read(R/'stage_duration_coverage_sid.csv')
 entry['lon_sin_entry']=np.sin(np.deg2rad(entry.entry_track_lon));entry['lon_cos_entry']=np.cos(np.deg2rad(entry.entry_track_lon))
 entry=entry.merge(coverage[['sid','stage','covered_h','uncovered_h']],on=['sid','stage'],validate='1:1')
 counts,ny=build_bootstrap(master);sid_index={s:i for i,s in enumerate(master.sid)}
 summaries=[];changes=[];coefs=[];envtrend=[];audit=[];periods=[];annual=[];statuses=[];supportrows=[];cutoffs=[];predrows=[];sens=[];statetrend=[];envcontribution=[]
 for stage in STAGES:
  z=entry[entry.stage.eq(stage)].copy()
  state=['entry_wind_kt','entry_track_lat','lon_sin_entry','lon_cos_entry','entry_land_current_v3']
  if stage!='D':state+=['entry_age_h','entry_motion_ms_v3']
  need=['duration_h','genesis_year','genesis_month',*state,*ENV]
  complete=z[need].notna().all(axis=1)&z.exposure_qc.eq(True)
  # Prespecified marginal common support across all four historical periods.
  # 1st--99th percentiles trim extreme extrapolation; finite support is retained
  # separately as a sensitivity. Categorical geography is not zero-imputed.
  common=complete.copy();trimcols=ENV+['entry_track_lat','entry_wind_kt']+(['entry_age_h','entry_motion_ms_v3']if stage!='D' else [])
  for c in trimcols:
   qq=z.loc[complete].groupby('period')[c].quantile([.01,.99]).unstack();lo=qq[.01].max();hi=qq[.99].min()
   if lo>hi:raise ValueError((stage,c,'no four-period common support'))
   common &= z[c].between(lo,hi);cutoffs.append(dict(stage=stage,variable=c,common_low=lo,common_high=hi))
  for idx,rr in z.iterrows():supportrows.append(dict(sid=rr.sid,stage=stage,period=rr.period,genesis_year=rr.genesis_year,complete=bool(complete.loc[idx]),common_support=bool(common.loc[idx]),missing_fields=';'.join(c for c in need if pd.isna(rr[c]))))
  for scope,select in [('common',common),('complete_case_sensitivity',complete)]:
   g=z[select].copy().sort_values(['genesis_year','sid']).reset_index(drop=True)
   assert g.sid.nunique()==len(g) and set(g.genesis_year)==set(YEARS)
   ycode=g.genesis_year.to_numpy(int)-1982;year_n=np.bincount(ycode,minlength=42);w=1/year_n[ycode]
   X,names,groups,scale,mean=design(g,state);Y=g[['duration_h','covered_h']].to_numpy(float)
   coef,trend,contrib,fs=fit(X,Y,w,groups)
   raw=g.groupby('genesis_year').duration_h.mean();assert abs(trend[0,0]-hac(raw,raw.index)['slope'])<1e-9
   if scope!='common':
    sens.extend(dict(stage=stage,scope=scope,target=tar,n_sid=len(g),slope=trend[j,0])for j,tar in enumerate(TARGETS));continue
   save(g,f'common_support_{stage}.csv');save(pd.DataFrame(X,columns=names).assign(sid=g.sid,weight=w),f'design_{stage}.csv')
   audit.append(dict(stage=stage,n_sid=len(g),complete_n=int(complete.sum()),original_n=len(z),rank=int(np.linalg.matrix_rank(X)),columns=len(names),year_min_n=int(year_n.min()),year_max_n=int(year_n.max()),state=state,excluded_constant_columns=[c for c in state if c not in names],entry_exposure='fixed entry-location -24,-18,-12,-6 h; at least 3 valid; source frozen',reference='pooled common-support distribution with equal total weight per genesis year',dynamic_D_unsupported='prior motion/land history not defined before operational genesis'if stage=='D'else''))
   # Conditional annual exposure changes and duration slopes use the same support.
   for yr,q in g.groupby('genesis_year'):
    row=dict(stage=stage,genesis_year=int(yr),n_sid=len(q),mean_duration_h=q.duration_h.mean(),mean_covered_h=q.covered_h.mean())
    row.update({c:q[c].mean()for c in ENV+state});annual.append(row)
   for period,q in g.groupby('period'):
    # Each represented year receives equal total weight in period summaries.
    yy=q.groupby('genesis_year')[['duration_h','covered_h',*ENV,*state]].mean().mean()
    periods.append(dict(stage=stage,period=period,n_sid=len(q),n_years=q.genesis_year.nunique(),**yy.to_dict()))
   Xm=np.sum(X*w[:,None],axis=0)/w.sum()
   for yr,q in g.groupby('genesis_year'):
    inds=q.index.to_numpy();actual=X[inds].mean(axis=0)
    for j,tar in enumerate(TARGETS):
     xx=actual.copy()
     for group in ['geo','state','env'][:j]:xx[groups[group]]=Xm[groups[group]]
     predrows.append(dict(stage=stage,genesis_year=int(yr),target=tar,mean_h=float(xx@coef[:,0]),n_sid=len(q)))
   ix=np.array([sid_index[s]for s in g.sid]);drawtr=np.full((B,4,2),np.nan);drawcoef=np.full((B,len(names),2),np.nan);drawenv=np.full((B,3),np.nan);drawfeature=np.full((B,len(names)),np.nan)
   fail=0
   for b in range(B):
    cc=counts[b,ix].astype(float);ss=np.bincount(ycode,weights=cc,minlength=42);ww=cc*np.divide(ny[b,ycode],ss[ycode],out=np.zeros(len(g)),where=ss[ycode]>0)
    try:
     cb,tb,_,fb=fit(X,Y,ww,groups);drawtr[b]=tb;drawcoef[b]=cb;drawfeature[b]=fb
     drawenv[b]=[fb[names.index(c)]*scale[names.index(c)]for c in ENV]
     statuses.append(dict(stage=stage,draw=b,status='ok'))
    except (ValueError,np.linalg.LinAlgError) as exc:
     fail+=1;statuses.append(dict(stage=stage,draw=b,status=str(exc),zero_variation_columns=';'.join(names[k]for k in range(1,len(names))if np.std(X[ww>0,k])<1e-12)))
   np.savez_compressed(R/f'model_bootstrap_{stage}.npz',trends=drawtr,coefficients=drawcoef,environment_trends=drawenv,feature_trends=drawfeature,names=np.asarray(names),targets=np.asarray(TARGETS))
   for j,tar in enumerate(TARGETS):summaries.append(dict(stage=stage,target=tar,n_sid=len(g),n_years=42,slope=trend[j,0],**interval(drawtr[:,j,0]),family='C_primary_standardized_trends_16',unit='h/decade'))
   for j,group in enumerate(['basin_season','entry_state','entry_environment']):changes.append(dict(stage=stage,removed_distribution=group,n_sid=len(g),slope_difference=trend[j,0]-trend[j+1,0],**interval(drawtr[:,j,0]-drawtr[:,j+1,0]),family='C_standardization_contrasts_12',unit='h/decade'))
   for c in ENV:
    k=names.index(c);coefs.append(dict(stage=stage,variable=c,n_sid=len(g),coefficient_h_per_native_unit=coef[k,0]/scale[k],**interval(drawcoef[:,k,0]/scale[k]),family='C_entry_environment_duration_associations_12'))
    ei=ENV.index(c);envtrend.append(dict(stage=stage,variable=c,n_sid=len(g),trend_per_decade=fs[k]*scale[k],**interval(drawenv[:,ei]),family='C_entry_environment_trends_12'))
    envcontribution.append(dict(stage=stage,variable=c,n_sid=len(g),hours_per_decade_contribution=fs[k]*coef[k,0],**interval(drawfeature[:,k]*drawcoef[:,k,0]),family='descriptive individual environment products; no new hypothesis family'))
   for c in state:
    if c not in names:continue
    k=names.index(c);zz=interval(drawfeature[:,k]*scale[k]);bb=interval(drawcoef[:,k,0]/scale[k]);statetrend.append(dict(stage=stage,variable=c,n_sid=len(g),trend_per_decade=fs[k]*scale[k],trend_ci_low=zz['ci_low'],trend_ci_high=zz['ci_high'],coefficient_h_per_unit=coef[k,0]/scale[k],coefficient_ci_low=bb['ci_low'],coefficient_ci_high=bb['ci_high'],valid_draws=zz['valid_draws'],family='descriptive entry-state summaries; no new hypothesis family'))
   # Time-recording sensitivity retains same support and same model design.
   for j,tar in enumerate(TARGETS):sens.append(dict(stage=stage,scope='effective_6h_wind_coverage_same_support',target=tar,n_sid=len(g),slope=trend[j,1],**interval(drawtr[:,j,1]),elapsed_minus_covered_slope=trend[j,0]-trend[j,1]))
   if stage=='Res':
    h=g[g.uncovered_h.eq(0)].copy();xx,nn,gg,sc,me=design(h,state);yc=h.genesis_year.to_numpy(int)-1982;ww=1/np.bincount(yc,minlength=42)[yc];_,ts,_,_=fit(xx,h[['duration_h','covered_h']].to_numpy(float),ww,gg)
    for j,tar in enumerate(TARGETS):sens.append(dict(stage=stage,scope='exclude_missing_wind_tail',target=tar,n_sid=len(h),slope=ts[j,0]))
   print(stage,'n',len(g),'complete',int(complete.sum()),'rank',len(names),'fail',fail,'slopes',trend[:,0],flush=True)
 for rows,name,col in [(summaries,'standardized_duration_trends.csv','p_boot'),(changes,'standardization_trend_contributions.csv','p_boot'),(coefs,'entry_environment_duration_associations.csv','p_boot'),(envtrend,'entry_environment_trends.csv','p_boot')]:
  tab=pd.DataFrame(rows);tab['q_BH_new_family']=bh(tab[col]);save(tab,name)
 for rows,name in [(periods,'common_support_period_means.csv'),(annual,'common_support_annual_values.csv'),(statuses,'bootstrap_draw_status.csv'),(supportrows,'support_sid_stage.csv'),(cutoffs,'common_support_cutoffs.csv'),(predrows,'standardized_annual_predictions.csv'),(sens,'duration_sensitivities.csv'),(statetrend,'entry_state_trends_associations.csv'),(envcontribution,'individual_environment_trend_products.csv')]:save(pd.DataFrame(rows),name)
 info=dict(status='C_STANDARDIZATION_ESTIMATED',elapsed_seconds=time.time()-t,models=audit,bootstrap=dict(B=B,seed=SEED,block_years=3,levels='joint circular genesis-year blocks then whole SIDs within selected year',all_stage_same_draw_bank=True,overlap='fixed four-period common marginal 1st--99th percentile intersection',failed_draws='reported; no replacement'),family_policy='Original seven primary and 30 H4 tests untouched; four separate new families of 16,12,12,12 tests',scientific_scope='Associational standardization in observed support; no causal mediation or explanation percentage',runtime=sys.version)
 (R/'MODEL_SPEC_AND_AUDIT.json').write_text(json.dumps(info,indent=2),encoding='utf8');(R/'CHECKPOINT.json').write_text(json.dumps(info,indent=2),encoding='utf8')
 print(pd.DataFrame(summaries).to_string(index=False));print(pd.DataFrame(envtrend).to_string(index=False));print(pd.DataFrame(coefs).to_string(index=False))
if __name__=='__main__':main()
