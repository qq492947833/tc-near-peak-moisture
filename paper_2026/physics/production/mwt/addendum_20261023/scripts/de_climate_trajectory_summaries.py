from _release_paths import release_path as _release_path
"""The original 216 linear field-climatology targets on the same SID draw bank."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
A=Path(__file__).absolute().parents[1];O=A/'source_data/de_P4/climate_trajectory';V=Path(str(_release_path('mechanism_v5','')));U=V/'upper_profile_100_50_interpolation_v1';KINDS=['raw_common','climate_common','anomaly_common']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return pd.read_csv(p,keep_default_na=False,na_values=['','nan','NaN'],float_precision='round_trip',low_memory=False)
def calculate(table,pool,weights):
 rows=[];draws={};stage=[]
 for (boundary,lag,field),group in table.groupby(['boundary','lag_h','field'],sort=True):
  assert len(group)==2368 and group.sid.is_unique and set(group.sid)==set(pool)
  g=group.set_index('sid').reindex(pool);x=g[KINDS].to_numpy(float);masks=np.column_stack([g[k+'__valid'].eq(True).to_numpy()&np.isfinite(x[:,j]) for j,k in enumerate(KINDS)])
  assert np.array_equal(masks[:,0],masks[:,1]) and np.array_equal(masks[:,0],masks[:,2]);ok=masks[:,0];x[~ok]=np.nan
  assert np.allclose(x[ok,2],x[ok,0]-x[ok,1],rtol=0,atol=1e-11)
  n=weights@ok.astype(np.int32);distinct=(weights>0).astype(np.int32)@ok.astype(np.int32);z=np.divide(weights@np.where(ok[:,None],x,0),n[:,None],out=np.full((len(weights),3),np.nan),where=n[:,None]>0)
  means=x[ok].mean(0) if ok.any() else np.full(3,np.nan);ci=np.quantile(z[n>0],[.025,.975],axis=0,method='linear') if ok.sum()>=2 and (n>0).any() else np.full((2,3),np.nan)
  key=boundary+'__'+str(int(lag))+'__'+field
  for j,k in enumerate(KINDS):
   own=g.raw_own_support__valid.eq(True).to_numpy();r=dict(boundary=boundary,lag_h=int(lag),field=field,quantity=k,unit=('%' if field=='r700' and k!='anomaly_common' else {'q700':'g kg-1','r700':'pp','u700':'m s-1','v700':'m s-1'}[field]),estimate=means[j],ci_low=ci[0,j],ci_high=ci[1,j],n_sid=int(ok.sum()),n_original_sid_pool=len(pool),n_actual_position=int(g.actual_track_point.eq(True).sum()),n_raw_own_valid=int(own.sum()),raw_own_mean=float(g.loc[own,'raw_own_support'].mean()) if own.any() else np.nan,n_draws=2000,valid_draws=int((n>0).sum()),failed_draws=int((n==0).sum()),ci_method='pointwise percentile reused shared original2368SID trajectories; not simultaneous',P_q='NOT_TESTED_NO_NEW_FAMILY');rows.append(r);draws[key+'__'+k]=z[:,j]
  draws[key+'__valid_multiplicity']=n;draws[key+'__distinct_valid_SID']=distinct
  for st,sg in group.groupby('frozen_stage',dropna=False,sort=True):stage.append(dict(boundary=boundary,lag_h=int(lag),field=field,frozen_stage=st,n_requested=len(sg),n_common_valid=int(sg.raw_common__valid.eq(True).sum()),n_actual_position=int(sg.actual_track_point.eq(True).sum()),n_legal_forward6h=int(sg.legal_forward6h.eq(True).sum())))
 return pd.DataFrame(rows),draws,pd.DataFrame(stage)
def main():
 import psutil
 psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
 r=json.loads((A/'reports/de_P4_assembled_summaries_receipt.json').read_text());assert r['status']=='COMPLETE';new=A/'source_data/de_P4/P4_shared_climate_SID_source.csv';assert sha(new)==r['outputs'][new.name]
 receipt=json.loads((U/'p4field_sum/execution_receipt.json').read_text());bank=receipt['bank'];assert sha(bank['path'])==bank['sha256']
 with np.load(bank['path'],allow_pickle=False) as f:ix=f['sampled_sid_indices'];pool=f['ordered_sid'].astype(str)
 assert ix.shape==(2000,2368) and list(pool)==bank['binding']['ordered_sid'];w=np.zeros(ix.shape,np.int32);np.add.at(w,(np.arange(len(ix))[:,None],ix),1)
 old=read(U/'p4field_src/P4_shared_climate_SID_source.csv');accepted=read(U/'p4field_sum/P4_raw_climate_anomaly_CI_n.csv');check,_,_=calculate(old,pool,w);keys=['boundary','lag_h','field','quantity'];z=check.merge(accepted,on=keys,validate='one_to_one',suffixes=('_new','_old'));assert len(z)==216;err=0.
 for c in ['estimate','ci_low','ci_high','raw_own_mean']:
  aa=z[c+'_new'].to_numpy(float);bb=z[c+'_old'].to_numpy(float);assert np.allclose(aa,bb,rtol=0,atol=1e-10,equal_nan=True);err=max(err,float(np.nanmax(abs(aa-bb))))
 for c in ['n_sid','n_original_sid_pool','n_actual_position','n_raw_own_valid','valid_draws','failed_draws']:assert np.array_equal(z[c+'_new'],z[c+'_old'])
 result,draws,stage=calculate(read(new),pool,w);assert len(result)==216
 cols=['estimate','ci_low','ci_high','n_sid'];aa=result[result.boundary.eq('E_to_F')].set_index(keys)[cols].sort_index();bb=accepted[accepted.boundary.eq('E_to_F')].set_index(keys)[cols].sort_index();assert np.allclose(aa,bb,rtol=0,atol=1e-10,equal_nan=True)
 O.mkdir(exist_ok=True);result.to_csv(O/'P4_raw_climate_anomaly_CI_n.csv',index=False);stage.to_csv(O/'P4_actual_stage_composition.csv',index=False);np.savez_compressed(O/'pointwise_draws.npz',**draws)
 rr=dict(status='COMPLETE',targets=216,raw_reads=0,new_draws=0,model_refits=0,source_sha256=sha(new),original_bank_sha256=bank['sha256'],original_algorithm_sha256=sha(U/'p4_shared_field_anomaly_v1/p4_field_companion.py'),driver_sha256=sha(__file__),baseline_max_abs_difference=err,E_to_F_values_intervals_samples_unchanged=True,outputs={p.name:sha(p) for p in O.iterdir() if p.is_file()});(A/'reports/de_climate_trajectory_receipt.json').write_text(json.dumps(rr,indent=2));print(json.dumps(rr))
if __name__=='__main__':main()
