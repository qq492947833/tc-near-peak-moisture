from _release_paths import release_path as _release_path
"""Reaggregate existing D/E boundary physics on the single frozen main cohort."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
from datetime import datetime,timezone
import ast,json,hashlib,time
import numpy as np
import pandas as pd
import psutil

HERE=Path(__file__).absolute().parents[1];ADD=HERE.parent
PROJECT=Path(str(_release_path('project','')))
PHY=PROJECT/'03_Work/v6/physics'
RH=HERE/'statistics/rh';P4=HERE/'statistics/p4'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return pd.read_csv(p,keep_default_na=False,na_values=['','nan','NaN'],float_precision='round_trip',low_memory=False)
def save(x,p):x.to_csv(p,index=False)
def extract(p,names,env):
    nodes=[n for n in ast.parse(p.read_text('utf-8')).body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes}==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),env)

def main():
    tick=time.monotonic();proc=psutil.Process();proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    for dest in [RH,P4]:dest.mkdir(parents=True,exist_ok=True)
    ids=set((HERE/'source_data/main_sid.txt').read_text('utf-8').splitlines());assert len(ids)==2340
    env=dict(np=np,pd=pd,SEED=6101007)
    extract(PHY/'process_v6.py',['boot','summary_table'],env)
    extract(ADD/'scripts/de_rh_complete_summaries.py',['infer'],env)
    input_hashes={};checks=[]
    def filtered(rel,dest):
        p=ADD/'source_data'/rel;input_hashes[str(p)]=sha(p)
        x=read(p);z=x[x.sid.isin(ids)].copy();save(z,dest)
        checks.append(dict(source=str(p),original_sid_n=x.sid.nunique(),retained_sid_n=z.sid.nunique(),excluded_sid_n=x.loc[~x.sid.isin(ids),'sid'].nunique()))
        return z
    # Verify and regenerate the accepted RH resampling frame. Every diagnostic
    # retains its own finite common support within this fixed frame.
    bankpath=PHY/'RH_matched_decomposition_bootstrap.npz';input_hashes[str(bankpath)]=sha(bankpath)
    with np.load(bankpath) as z:old_pool=z['sid'].astype(str);old_weights=z['weights']
    reproduced=env['boot'](np.zeros((len(old_pool),1)))[5]
    assert np.array_equal(old_weights,reproduced)
    pool=np.array([s for s in old_pool if s in ids]);w=env['boot'](np.zeros((len(pool),1)))[5]
    np.savez_compressed(RH/'resampling_bank.npz',sid=pool,weights=w,seed=np.array(6101007))
    native=filtered('de_RH/RH_native_SID_sources.csv',RH/'RH_native_SID_sources.csv')
    fields=['q_contribution','T_contribution','p_contribution','local_RH_change','sampling_support_remainder','native_RH_change','archived_RH_change','rh_archive_difference_change']
    x=native[np.isfinite(native[fields]).all(1)].copy();assert set(x.sid)<=set(pool)
    save(x,RH/'RH_matched_decomposition_SID.csv');rows=[];draws=[]
    for keys,g in x.groupby(['boundary','window','level_hpa','region'],sort=True):
        for field in fields:
            st,dr=env['infer'](g[field].to_numpy(),g.sid,pool,w)
            rows.append(dict(zip(['boundary','window','level_hpa','region'],keys),component=field,units='pp',**st));draws.append(dr)
    table=pd.DataFrame(rows).rename(columns={'ci_low':'lo','ci_high':'hi'});assert len(table)==192
    save(table,RH/'RH_matched_decomposition_estimates.csv')
    np.savez_compressed(RH/'RH_matched_decomposition_bootstrap.npz',sid=pool,weights=w,draws=np.array(draws).T,
        targets=np.array([str(k) for k in table[['boundary','window','level_hpa','region','component']].to_numpy()]))
    four=filtered('de_RH/RH_fourway_SID.csv',RH/'RH_fourway_SID.csv')
    d=filtered('de_RH/RH_fourway_with_original_same_SID.csv',RH/'RH_fourway_with_original_same_SID.csv')
    cols=['original_support_endpoint_change','fixed_support_endpoint_change','temporal_symmetric','domain_symmetric','time_q_contribution','time_T_contribution','support_change','q_contribution','T_contribution','local_RH_change','sampling_support_remainder']
    rows=[];draws=[];difrows=[];difdraws=[]
    for (boundary,level),g in d.groupby(['boundary','level_hpa']):
        for scope,select in [('primary25','primary_common'),('coverage50','sensitivity50_common')]:
            z=g[g[select]];assert set(z.sid)<=set(pool)
            for var in cols:
                st,dr=env['infer'](z[var].to_numpy(),z.sid,pool,w)
                rows.append(dict(boundary=boundary,level_hpa=level,scope=scope,variable=var,units='percentage points',**st));draws.append(dr)
            for a,b in [('temporal_symmetric','local_RH_change'),('time_q_contribution','q_contribution'),('time_T_contribution','T_contribution'),('domain_symmetric','sampling_support_remainder')]:
                st,dr=env['infer']((z[a]-z[b]).to_numpy(),z.sid,pool,w)
                difrows.append(dict(boundary=boundary,level_hpa=level,scope=scope,comparison=a+' minus '+b,units='percentage points',**st));difdraws.append(dr)
    save(pd.DataFrame(rows),RH/'RH_summary.csv');save(pd.DataFrame(difrows),RH/'RH_direct_method_differences.csv')
    np.savez_compressed(RH/'RH_new_component_draws.npz',draws=np.array(draws),sid=pool,weights=w)
    np.savez_compressed(RH/'RH_direct_method_draws.npz',draws=np.array(difdraws),sid=pool,weights=w)
    save(d.groupby(['boundary','level_hpa']).agg(n_original_endpoint_pairs=('sid','size'),n_fourway_valid=('temporal_symmetric','count'),n_original_comparison25=('q_contribution','count'),n_primary_common=('primary_common','sum'),n_same_comparison50=('sensitivity50_common','sum')).reset_index(),RH/'RH_support_counts.csv')
    print('RH_COMPLETE',len(rows),len(difrows),'native_pool',len(pool),flush=True)
    for name,groups in [('P4_transport',['boundary','field','estimand','units']),
                        ('P4_background',['boundary','field','units','window','component']),
                        ('P4_transport_matched_windows',['boundary','field','units','target'])]:
        suffix='_SID.csv' if name.endswith('matched_windows') else '_SID_sources.csv'
        x=filtered('de_P4/'+name+suffix,P4/(name+suffix))
        tab,piv,dr,wgt=env['summary_table'](x,groups)
        assert set(piv.index)==ids
        save(tab,P4/(name+'_estimates.csv'))
        np.savez_compressed(P4/(name+'_bootstrap.npz'),sid=piv.index.to_numpy(str),targets=np.array([str(c) for c in piv.columns]),draws=dr,weights=wgt)
        print(name,'COMPLETE',len(tab),flush=True)
    pair=filtered('de_P4/PI_headroom_identity_pair_sources.csv',P4/'PI_headroom_identity_pair_sources.csv')
    rng=np.random.default_rng(6101032);rows=[];draws={}
    for bd in ['D_to_E','E_to_F']:
        p=pair[pair.boundary.eq(bd)].sort_values('sid');v=p[['delta_PI_ms','delta_concurrent_wind_ms','delta_headroom_ms']].to_numpy();v[:,1]*=-1
        ix=rng.integers(len(v),size=(2000,len(v)));dr=v[ix].mean(1);draws[bd]=dr;draws[bd+'_sid']=p.sid.to_numpy(str);draws[bd+'_sampled_indices']=ix
        assert np.max(np.abs(v[:,0]+v[:,1]-v[:,2]))<1e-10
        for i,c in enumerate(['PI','minus_V','headroom']):rows.append(dict(boundary=bd,component=c,units='m s-1',window='-24_0',estimate=v[:,i].mean(),lo=np.quantile(dr[:,i],.025),hi=np.quantile(dr[:,i],.975),n_sid=len(v)))
    save(pd.DataFrame(rows),P4/'PI_component_estimates.csv');np.savez_compressed(P4/'PI_component_bootstrap.npz',**draws)
    for p,h in input_hashes.items():assert sha(p)==h
    rec=dict(status='COMPLETE_RH_AND_BOUNDARY_PHYSICS_DESCRIPTIVE_ESTIMATORS_ON_FROZEN_MASTER',UTC=datetime.now(timezone.utc).isoformat(),
        main_sid_n=len(ids),main_sid_sha256=sha(HERE/'source_data/main_sid.txt'),RH_bootstrap_frame_n=len(pool),RH_seed=6101007,P4_seed=6101007,PI_seed=6101032,
        original_RH_bank_reproduced_exact=True,master_filter_checks=checks,source_hashes=input_hashes,
        statistical_methods_unchanged=True,raw_ERA5_reads=0,model_refits=0,
        output_hashes={str(p.relative_to(HERE)):sha(p) for folder in [RH,P4] for p in folder.iterdir() if p.is_file()},elapsed_s=time.monotonic()-tick)
    (HERE/'reports/boundary_summaries_receipt.json').write_text(json.dumps(rec,indent=2,ensure_ascii=False)+'\n','utf-8')
    print(json.dumps({k:rec[k] for k in ['status','main_sid_n','RH_bootstrap_frame_n','elapsed_s']}),flush=True)

if __name__=='__main__':main()
