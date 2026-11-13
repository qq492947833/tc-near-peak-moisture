from _release_paths import release_path as _release_path
"""Retain complete original geographic-intersection comparisons and direct differences."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,hashlib,psutil
import numpy as np,pandas as pd
ADD=Path(__file__).absolute().parents[1];OUT=ADD/'source_data/de_RH'
PHY=Path(str(_release_path('project','03_Work/v6/physics')))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return pd.read_csv(p,float_precision='round_trip')
def infer(values,sids,pool,w):
    assert set(sids)<=set(pool);v=pd.Series(values,index=sids).reindex(pool).to_numpy(float);ok=np.isfinite(v);den=w@ok.astype(float);d=np.divide(w@np.nan_to_num(v,nan=0),den,out=np.full(len(w),np.nan),where=den>0)
    return dict(n_sid=int(ok.sum()),estimate=float(np.nanmean(v)),ci_low=float(np.nanquantile(d,.025)),ci_high=float(np.nanquantile(d,.975)),B_valid=int(np.isfinite(d).sum())),d
def main():
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);assert json.loads((ADD/'reports/de_RH_receipt.json').read_text())['status']=='COMPLETE'
    bank=np.load(PHY/'RH_matched_decomposition_bootstrap.npz',allow_pickle=False);pool=bank['sid'].astype(str);w=bank['weights'];x=read(OUT/'RH_native_SID_sources.csv')
    fields=['q_contribution','T_contribution','p_contribution','local_RH_change','sampling_support_remainder','native_RH_change','archived_RH_change','rh_archive_difference_change'];x=x[np.isfinite(x[fields]).all(1)].copy();assert set(x.sid)<=set(pool)
    x.to_csv(OUT/'RH_matched_decomposition_SID.csv',index=False);rows=[];draws=[]
    for keys,g in x.groupby(['boundary','window','level_hpa','region'],sort=True):
        for field in fields:
            st,dr=infer(g[field].to_numpy(),g.sid,pool,w);rows.append(dict(zip(['boundary','window','level_hpa','region'],keys),component=field,units='pp',**st));draws.append(dr)
    table=pd.DataFrame(rows).rename(columns={'ci_low':'lo','ci_high':'hi'});table.to_csv(OUT/'RH_matched_decomposition_estimates.csv',index=False);np.savez_compressed(OUT/'RH_matched_decomposition_bootstrap.npz',sid=pool,weights=w,draws=np.array(draws).T,targets=np.array([str(k) for k in table[['boundary','window','level_hpa','region','component']].to_numpy()]))
    x=read(OUT/'RH_fourway_with_original_same_SID.csv');rows=[];draws=[]
    for (bd,level),g in x.groupby(['boundary','level_hpa'],sort=True):
        for scope,flag in [('primary25','primary_common'),('coverage50','sensitivity50_common')]:
            z=g[g[flag]]
            for a,b in [('temporal_symmetric','local_RH_change'),('time_q_contribution','q_contribution'),('time_T_contribution','T_contribution'),('domain_symmetric','sampling_support_remainder')]:
                st,dr=infer((z[a]-z[b]).to_numpy(),z.sid,pool,w);rows.append(dict(boundary=bd,level_hpa=level,scope=scope,comparison=a+' minus '+b,units='percentage points',**st));draws.append(dr)
    pd.DataFrame(rows).to_csv(OUT/'RH_direct_method_differences.csv',index=False);np.savez_compressed(OUT/'RH_direct_method_draws.npz',draws=np.array(draws),sid=pool,weights=w)
    (ADD/'reports/de_RH_complete_summaries_receipt.json').write_text(json.dumps(dict(status='COMPLETE',script_sha256=sha(__file__),source_bank_sha256=sha(PHY/'RH_matched_decomposition_bootstrap.npz'),original_intersection_targets=len(table),direct_same_SID_targets=len(rows),raw_reads=0,model_fits=0,all_original_components_retained=True,outputs={p.name:sha(p) for p in OUT.glob('*.csv') if '_pilot' not in p.name}),indent=2)+'\n')
if __name__=='__main__':main()
