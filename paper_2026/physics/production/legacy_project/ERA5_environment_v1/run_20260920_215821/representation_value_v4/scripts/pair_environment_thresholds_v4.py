"""Paired threshold environmental sensitivity using both-QC SID membership."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from analyze_environment_v4 import V,OUT,CORE,CONTRASTS,UNITS,B,SEED,read,save,dump,sha,plan,weighted_boot

def main():
 _,active=plan();assert (OUT/'run_audit.json').exists(),'Finish B1 primary before paired sensitivities'
 con=read(OUT/'per_SID_environment_contrasts.csv');ref=read(V/'audit/environment_sources/B1_frozen_reference_scale.csv').set_index('variable');rows=[];members=[];points=[];checks=[]
 for alt,scope in [('beta085','main_for_beta085'),('beta095','main_for_beta095'),('C5','six_common_main_all')]:
  original=np.load(OUT/'scopes'/scope/'sid_bootstrap.npz');sids=original['sids'];counts=original['sid_multiplicity'];columns=[(x,c) for x in CORE for c in CONTRASTS]
  a=con[con.config.eq('main')].pivot(index='sid',columns=['variable','contrast'],values='estimate').reindex(index=sids,columns=pd.MultiIndex.from_tuples(columns)).to_numpy(float)
  b=con[con.config.eq(alt)].pivot(index='sid',columns=['variable','contrast'],values='estimate').reindex(index=sids,columns=pd.MultiIndex.from_tuples(columns)).to_numpy(float)
  ok=np.isfinite(a)&np.isfinite(b);aa=np.where(ok,a,np.nan);bb=np.where(ok,b,np.nan);dif=bb-aa
  da,nn=weighted_boot(aa,counts);db,nb=weighted_boot(bb,counts);dd,nd=weighted_boot(dif,counts)
  np.testing.assert_array_equal(nn,nb);np.testing.assert_array_equal(nn,nd);np.testing.assert_allclose(dd,db-da,rtol=1e-9,atol=1e-12,equal_nan=True)
  target=OUT/'threshold_common';target.mkdir(exist_ok=True);np.savez_compressed(target/(alt+'_draws.npz'),sids=sids,main_input=aa,alternate_input=bb,main_draws=da,alternate_draws=db,difference_draws=dd,valid_n=nn,variables=np.array([x for x,c in columns]),contrasts=np.array([c for x,c in columns]),multiplicity_source=np.array(str(Path('scopes')/scope/'sid_bootstrap.npz')))
  for j,(x,c) in enumerate(columns):
   finite=np.isfinite(dd[:,j]);ci=np.quantile(dd[finite,j],[.025,.975]) if finite.any() else [np.nan,np.nan];sd=float(ref.loc[x,'reference_sd']);point=float(np.nanmean(dif[:,j]));r=dict(alternate=alt,comparison='alternate minus genesis-adjusted90',variable=x,contrast=c,n_sid=int(ok[:,j].sum()),n_physical_common_sid=len(sids),main_native_environment_eligible_n=int(np.isfinite(a[:,j]).sum()),alternate_native_environment_eligible_n=int(np.isfinite(b[:,j]).sum()),estimate=point,ci_low=ci[0],ci_high=ci[1],estimate_sd=point/sd,ci_low_sd=ci[0]/sd,ci_high_sd=ci[1]/sd,B_planned=B,B_valid=int(finite.sum()),B_failed=int((~finite).sum()),unit=UNITS[CORE.index(x)],P=np.nan,q=np.nan,test_family='none; prespecified descriptive threshold sensitivity')
   rows.append(r)
   for name,matrix,draw in [('main90',aa,da),(alt,bb,db)]:
    good=np.isfinite(draw[:,j]);ci0=np.quantile(draw[good,j],[.025,.975]) if good.any() else [np.nan,np.nan];points.append(dict(alternate=alt,representation_threshold=name,variable=x,contrast=c,n_sid=int(ok[:,j].sum()),estimate=float(np.nanmean(matrix[:,j])),ci_low=ci0[0],ci_high=ci0[1],estimate_sd=float(np.nanmean(matrix[:,j]))/sd,ci_low_sd=ci0[0]/sd,ci_high_sd=ci0[1]/sd,unit=UNITS[CORE.index(x)],paired_population='same eligible SID-factor-contrast across both thresholds'))
   for k,sid in enumerate(sids):members.append(dict(alternate=alt,sid=sid,variable=x,contrast=c,included=bool(ok[k,j]),main_environment_eligible=bool(np.isfinite(a[k,j])),alternate_environment_eligible=bool(np.isfinite(b[k,j])),exclusion_reason='' if ok[k,j] else 'one_or_both_thresholds_environment_QC_incomplete'))
  checks.append(dict(alternate=alt,scope_draw_source=scope,sampling_frame_sid=len(sids),B=B,direct_difference_equals_paired_draw_subtraction=True))
 save(pd.DataFrame(rows),OUT/'threshold_common_difference.csv');save(pd.DataFrame(points),OUT/'threshold_common_estimates.csv');save(pd.DataFrame(members),OUT/'threshold_common_membership.parquet');dump(OUT/'threshold_common_validation.json',dict(status='PASS',active_plan=active,checks=checks,no_new_families=True,no_historical_model_fit=True))
 # Bind and reuse existing pooled-hour E6 estimates as a different, explicitly named estimand.
 base=V.parent/'review_completion_v3_2';directory=base/'e6/new_available/f_composition';bind=[]
 for name in ['F_composition_summary_v3.csv','F_composition_bootstrap_v3.csv','F_composition_SID_sensitivity_v3.csv']:
  p=directory/name
  if p.exists():bind.append(dict(source_path=str(p),sha256=sha(p),role='Accepted v3.2 E6 pooled-hour sensitivity; distinct from new SID-equal B1',recomputed=False))
 save(pd.DataFrame(bind),OUT/'existing_E6_pooled_sensitivity_binding.csv');print(json.dumps({'status':'COMPLETE','threshold_pairs':len(checks),'estimates':len(rows),'E6_bindings':len(bind)}),flush=True)
if __name__=='__main__':main()
