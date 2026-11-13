"""B2 fixed F-entry environment, identical X/SID for R/U/F and direct ratios."""
from pathlib import Path
import ast,json,sys,hashlib
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from prepare_durations_v4 import V,read,dump,active_plan,dependency
CORE=['X1_sst_c','X2_shear_ms','X3_mid_rh_pct','X4_d200_1e5s','X5_minus_omega500_pa_s','X6_signed_vo850_1e5s']
TARGETS=['log_R','log_U','log_F','GB','GL']; SEED=20260920; B=2000
BASE=V.parent/'review_completion_v3_2'; OUT=V/'results/B2'

def old_design_function():
 p=BASE/'frozen_code_dependencies/review_fix_v3/models/repair_h4_v3.py'
 tree=ast.parse(p.read_text(encoding='utf-8'))
 f=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='make_design')
 env={'np':np,'pd':pd,'CORE':CORE}
 exec(compile(ast.Module(body=[f],type_ignores=[]),str(p),'exec'),env)
 return env['make_design'],p

def solve(X,Y,scale,counts=None):
 n,p=X.shape; wt=np.ones(n) if counts is None else np.asarray(counts,float)
 A=(X/scale)*np.sqrt(wt[:,None]); Z=Y*np.sqrt(wt[:,None])
 u,s,vt=np.linalg.svd(A,full_matrices=False); cutoff=np.finfo(float).eps*max(n,p)*s[0]; rank=int((s>cutoff).sum())
 beta=(vt[:rank].T@((u[:,:rank].T@Z)/s[:rank,None]))/scale[:,None]
 null_norm=np.linalg.norm(vt[rank:],axis=0) if rank<p else np.zeros(p)
 estimable=null_norm<=1e-8
 return beta,rank,estimable,null_norm,(u,s,vt,cutoff)

def targets(duration):
 y=np.log(np.asarray(duration,float))
 if not np.isfinite(y).all():raise ValueError('All three physical covered durations must be positive and observed')
 return np.column_stack([y,y[:,2]-y[:,1],y[:,1]-y[:,0]])

def main():
 plan=active_plan(); OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'complete.json').exists():raise RuntimeError('Completed B2 exists; verify or replay independently, do not overwrite')
 ep=BASE/'sources/stage_entries_sp_qc_v3_2.csv'; hp=BASE/'sources/stage_entry_hour_sources_sp_qc_v3_2.csv'; dp=V/'source_data/durations.csv'
 e=read(ep); e=e[e.stage.eq('F')].reset_index(drop=True); assert e.sid.is_unique and len(e)==2368
 h=read(hp); h=h[h.stage.eq('F')].copy(); assert not h.duplicated(['sid','offset_h']).any()
 for sid,g in h.groupby('sid'):assert set(g.offset_h)=={-24,-18,-12,-6}
 assert (pd.to_datetime(h.UTC,utc=True)==pd.to_datetime(h.entry_UTC,utc=True)+pd.to_timedelta(h.offset_h,unit='h')).all()
 good=h.valid_all_six.astype(str).str.lower().isin(['true','1'])
 hc=h[good].groupby('sid').size().reindex(e.sid,fill_value=0).to_numpy(); assert np.array_equal(hc,e.n_valid.to_numpy(int))
 q=e.exposure_qc.astype(str).str.lower().isin(['true','1']).to_numpy(); assert np.array_equal(q,hc>=3)
 diffs={}
 for x in CORE:
  means=pd.to_numeric(h.loc[good,'r200_800__'+x]).groupby(h.loc[good,'sid']).mean().reindex(e.sid).to_numpy()
  observed=pd.to_numeric(e[x]).to_numpy(); ok=q&np.isfinite(observed)
  assert np.isfinite(observed[q]).all()
  diffs[x]=float(np.max(np.abs(means[ok]-observed[ok]))); assert diffs[x]<1e-9
 d=read(dp); f=e.merge(d,on='sid',how='left',suffixes=('','_duration'),validate='one_to_one')
 assert f.main_complete.notna().all(); assert (f.basin_genesis==f.basin_genesis_duration).all(); assert (f.genesis_year==f.genesis_year_duration).all()
 assert (pd.to_datetime(f.entry_UTC,utc=True)==pd.to_datetime(f.maturity_time,utc=True)).all()
 physical=f.main_complete.astype(bool).to_numpy(); assert f.loc[physical,'R_start_equals_F'].all()
 assert np.max(np.abs(f.loc[physical,'duration_h']-f.loc[physical,'T_F_h']))<1e-9
 make,oldpath=old_design_function(); derived,oldX,_,controls,cats,stages=make(f)
 assert controls==['month_sin','month_cos','year_center','lat_z','lon_sin','lon_cos','wind_z','age_z','land_entry']
 assert cats==['basin_NA','basin_NI','basin_SI','basin_SP','basin_WP']; assert stages==[]
 sdpath=V/'audit/environment_sources/B2_frozen_H4_reference_scale.csv'; sd=read(sdpath).set_index('variable').reference_sd
 design=oldX[controls+cats+CORE].astype(float); design[CORE]=design[CORE]/sd.reindex(CORE).to_numpy(); design.insert(0,'const',1.)
 positive=np.isfinite(f[['T_R_h','T_U_h','T_F_h']].to_numpy(float)).all(1)&f[['T_R_h','T_U_h','T_F_h']].gt(0).all(1).to_numpy()
 xc=np.isfinite(design.to_numpy(float)).all(1); mask=physical&positive&q&xc
 member=f[['sid','name','genesis_year','basin_genesis','period','multi','later_group','T_R_h','T_U_h','T_F_h','entry_UTC','n_valid','exposure_qc']].copy()
 member['physical_main_complete']=physical;member['positive_three_duration']=positive;member['complete_design']=xc;member['included_B2']=mask
 member['exclusion_reason']=[';'.join(a for a,b in [('incomplete_physical_duration',not physical[i]),('nonpositive_or_missing_duration',not positive[i]),('entry_exposure_QC',not q[i]),('missing_entry_control_or_exposure',not xc[i])] if b) or 'included' for i in range(len(f))]
 member.to_csv(OUT/'membership.csv',index=False)
 inputs=f.loc[mask,['sid','name','genesis_year','basin_genesis','period','multi','later_group','T_R_h','T_U_h','T_F_h','entry_UTC','entry_lat','entry_lon','entry_wind_kt','entry_age_h','entry_land_current_v3','n_valid']+CORE].reset_index(drop=True)
 X=design.loc[mask].to_numpy(float); names=list(design); Y=targets(inputs[['T_R_h','T_U_h','T_F_h']]); scale=np.linalg.norm(X,axis=0); assert (scale>0).all()
 ids=inputs.sid.to_numpy(str); n,p=X.shape; beta,rank,estim,null,svd=solve(X,Y,scale); assert estim.all(); df=n-rank
 # HC3 covariance from the same rank-revealing inverse as the point fit.
 u,s,vt,cutoff=svd; pinv=((vt[:rank].T/s[:rank])@u[:,:rank].T)/scale[:,None]; hat=np.sum(u[:,:rank]**2,axis=1); resid=Y-X@beta
 cov=np.stack([(pinv*(resid[:,k]/(1-hat))[None,:])@(pinv*(resid[:,k]/(1-hat))[None,:]).T for k in range(5)])
 se=np.stack([np.sqrt(np.maximum(np.diag(c),0)) for c in cov],axis=1); crit=student_t.ppf(.975,df)
 # Independent library check for each of five responses; same parameterization.
 max_lib=0.
 for k in range(5):
  m=sm.OLS(Y[:,k],X).fit(cov_type='HC3',use_t=True)
  max_lib=max(max_lib,float(np.max(np.abs(m.params-beta[:,k]))),float(np.max(np.abs(m.cov_params()-cov[k]))))
 assert max_lib<1e-8
 identity=max(np.max(np.abs(beta[:,3]-(beta[:,2]-beta[:,1]))),np.max(np.abs(beta[:,4]-(beta[:,1]-beta[:,0])))); assert identity<1e-10
 inputs.to_csv(OUT/'model_inputs.csv',index=False); xd=pd.DataFrame(X,columns=names);xd.insert(0,'sid',ids);xd.to_csv(OUT/'design_matrix.csv',index=False)
 yd=pd.DataFrame(Y,columns=TARGETS);yd.insert(0,'sid',ids);yd.to_csv(OUT/'responses.csv',index=False)
 np.savez_compressed(OUT/'fixed_design.npz',X=X,Y=Y,sids=ids,terms=np.array(names),targets=np.array(TARGETS),weights=np.ones(n),column_L2_scale=scale,reference_sd=sd.reindex(CORE).to_numpy(),reference_sd_variables=np.array(CORE),beta=beta,covariance_HC3=cov)
 rng=np.random.default_rng(SEED); boot=np.full((B,p,5),np.nan);counts_all=np.empty((B,n),dtype=np.uint16);status=[];enull=np.full((B,p),np.nan); target_ok=np.zeros((B,p),bool)
 for bi in range(B):
  ix=rng.integers(0,n,n);cnt=np.bincount(ix,minlength=n);counts_all[bi]=cnt
  try:
   b,r,ok,nn,ss=solve(X,Y,scale,cnt); enull[bi]=nn;target_ok[bi]=ok
   ident=max(np.max(np.abs(b[:,3]-b[:,2]+b[:,1])),np.max(np.abs(b[:,4]-b[:,1]+b[:,0])));assert ident<1e-9
   b[~ok]=np.nan;boot[bi]=b
   status.append(dict(draw=bi,rank=r,n_unique_sid=int((cnt>0).sum()),df=n-r,svd_cutoff=ss[3],condition=float(ss[1][0]/ss[1][r-1]),all_environment_estimable=bool(ok[-6:].all()),failed_environment_count=int((~ok[-6:]).sum()),max_direct_identity_error=float(ident),failure_reason='' if ok.all() else 'nonestimable_columns_reported_individually'))
  except np.linalg.LinAlgError as ex:status.append(dict(draw=bi,rank=np.nan,n_unique_sid=int((cnt>0).sum()),df=np.nan,failure_reason=type(ex).__name__))
  if (bi+1)%250==0:print('B2 draw',bi+1,'/',B,flush=True)
 np.savez_compressed(OUT/'bootstrap_draws.npz',coefficients=boot,sid_multiplicity=counts_all,sids=ids,terms=np.array(names),targets=np.array(TARGETS),estimable=target_ok,null_projection_norm=enull)
 pd.DataFrame(status).to_csv(OUT/'bootstrap_status.csv',index=False)
 rows=[]
 for k,targ in enumerate(TARGETS):
  for j,term in enumerate(names):
   vals=boot[:,j,k];valid=np.isfinite(vals); ci=np.percentile(vals[valid],[2.5,97.5]) if valid.any() else [np.nan,np.nan]
   rows.append(dict(target=targ,term=term,beta=beta[j,k],se=se[j,k],ci_low=beta[j,k]-crit*se[j,k],ci_high=beta[j,k]+crit*se[j,k],p=float(2*student_t.sf(abs(beta[j,k]/se[j,k]),df)) if se[j,k]>0 else np.nan,df=df,n_sid=n,rank=rank,bootstrap_low=ci[0],bootstrap_high=ci[1],n_boot_valid=int(valid.sum()),n_boot_failed=int((~valid).sum()),bootstrap_B=B,unit='log_duration_per_fixed_reference_SD' if term in CORE else 'log_duration_per_design_unit'))
 cf=pd.DataFrame(rows);cf.to_csv(OUT/'coefficients.csv',index=False)
 direct=cf[cf.target.isin(['GB','GL'])&cf.term.isin(CORE)].copy().rename(columns={'target':'contrast','term':'variable'});assert len(direct)==12
 direct['q']=multipletests(direct.p,method='fdr_bh')[1];direct['family']='N2_12_fixed_environment_coefficient_differences';direct['reference_sd']=direct.variable.map(sd)
 for c in ['beta','ci_low','ci_high','bootstrap_low','bootstrap_high']:direct[c+'_multiplicative_ratio_percent']=100*np.expm1(direct[c])
 direct.to_csv(OUT/'N2_contrasts.csv',index=False)
 pd.DataFrame({'sid':ids,'leverage':hat,**{'residual_'+t:Y[:,k]-X@beta[:,k] for k,t in enumerate(TARGETS)}}).to_csv(OUT/'residual_leverage_diagnostics.csv',index=False)
 # Persist the exact consumed small inputs so a reviewer need not read legacy output paths.
 e.to_csv(OUT/'source_F_entry_table.csv',index=False);h.to_csv(OUT/'source_F_entry_hour_ledger.csv',index=False)
 schema=dict(seed=SEED,B=B,sid_order='model_inputs.csv exact row order',multiplicity='same SID bootstrap multiset shared by all 5 responses; symmetric OLS sufficient statistic; no replacement success draws',coefficient_axes=['draw','term','target'],rank='SVD on fixed original column L2 scales, eps*max(n,p)*smax; no unscaled-rank threshold relaxation',estimability='relative projection of transformed unit target onto null space <=1e-8; each coefficient individually marked, not whole matrix rejection',confidence='HC3 t(df=n-rank) and percentile SID bootstrap both saved; N2 P/q from HC3 t',Z=controls+cats,no_future_controls=True)
 dump('results/B2/bootstrap_schema.json',schema)
 result=dict(status='COMPLETED_WITH_OWN_VERIFICATION',active_plan=plan,n_sid=n,rank=rank,df=df,zero_GB=int(np.isclose(Y[:,3],0).sum()),zero_GL=int(np.isclose(Y[:,4],0).sum()),source_F_entry_sid=len(e),source_exposure_QC=int(q.sum()),duration_common=int(physical.sum()),entry_hours_mean_max_abs_error=diffs,point_direct_identity_max_abs_error=float(identity),independent_statsmodels_max_abs_error=max_lib,n_boot_requested=B,n_boot_all_env_valid=int(np.sum(target_ok[:,-6:].all(1))),max_leverage=float(hat.max()),dependencies=[dependency(x) for x in [ep,hp,dp,sdpath,oldpath,Path(__file__)]],limitations=['Observed conditional association, not causality or prospective prediction','Large number of zero G retained; sparse nonzero recurrence information limits precision','No new selection, no H2 replay, no full E1-E6 rerun'])
 dump('results/B2/complete.json',result);print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()