from _release_paths import release_path as _release_path
"""Run unchanged R/U/F duration estimators within the frozen main SID set."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[k]='1'
from pathlib import Path
from datetime import datetime, timezone
import ast, json, hashlib, time
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t as student_t
from statsmodels.stats.multitest import multipletests
import psutil

HERE=Path(__file__).absolute().parents[1]
ADD=HERE.parent
V=Path(str(_release_path('legacy_project','ERA5_environment_v1/run_20260920_215821/representation_value_v4')))
OUT=HERE/'statistics/ruf'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    started=time.monotonic()
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    receipt=HERE/'reports/ruf_statistics_receipt.json'
    assert not receipt.exists(),'Completed output exists'
    OUT.mkdir(parents=True,exist_ok=True)
    freeze=json.loads((HERE/'reports/main_sid_freeze_receipt.json').read_text('utf-8'))
    assert sha(freeze['main_sid_path'])==freeze['main_sid_sha256']
    ids=set(Path(freeze['main_sid_path']).read_text('utf-8').splitlines())
    d=pd.read_parquet(HERE/'source_data/main_durations.parquet')
    assert set(d.sid)==ids and d.main_complete.all()
    # The revised interval ledger preserves R/U/F geometry but omits the five
    # accepted bridge-state metadata columns. Restore only those source fields
    # after checking every retained SID's duration/bridge identities.
    original_duration_path=V/'source_data/durations.csv'
    accepted=pd.read_csv(original_duration_path,keep_default_na=False,na_values=['','nan','NaN']).set_index('sid').loc[d.sid]
    for col in ['T_R_h','T_U_h','T_F_h','T_L_h','T_B_h','T_R_long_h','T_R5_h','T_U5_h','T_F5_h',
                'pre_maturity_near_h','later_interval_episode_n','later_max_episode_h']:
        assert np.array_equal(d[col].to_numpy(float),accepted[col].to_numpy(float),equal_nan=True),col
    for col in ['main_complete','six_complete','multi','later_class']:
        assert np.array_equal(d[col].to_numpy(),accepted[col].to_numpy()),col
    restored_columns=['B_agency_conflict_h','B_both_endpoints_tropical_h','B_other_h','B_unknown_h','strict_tropical_bridge']
    for col in restored_columns:
        assert col not in d
        d[col]=accepted[col].to_numpy()
    source=V/'scripts/analyze_durations_v4.py'
    names={'metric_matrix','sid_summary','annual','period','ratio_summary','influence'}
    nodes=[n for n in ast.parse(source.read_text('utf-8')).body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes}==names
    def put(frame,name):
        if Path(name).suffix=='.parquet':frame.to_parquet(OUT/name,index=False)
        else:frame.to_csv(OUT/name,index=False)
    env=dict(np=np,pd=pd,sm=sm,student_t=student_t,multipletests=multipletests,D=OUT,B=2000,SEED=20260920,
             PERIODS={'P1':(1982,1992),'P2':(1993,2003),'P3':(2004,2014),'P4':(2015,2023)},put=put)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),env)
    yearfile=V/'inputs/year_block_draws.npz'
    with np.load(yearfile) as z:draws={key:z[key] for key in z.files}
    # Regenerate the original block bank to verify source and seed exactly.
    rng=np.random.default_rng(20260920)
    for key,(lo,hi) in env['PERIODS'].items():
        years=np.arange(lo,hi+1);n=len(years);starts=rng.integers(0,n,size=(2000,(n+1)//2))
        expected=years[((starts[:,:,None]+[0,1])%n).reshape(2000,-1)[:,:n]]
        assert np.array_equal(draws[key],expected)
    scopes=[('main_all',d,False,True),('main_multi',d[d.multi],False,True),('main_nonmulti',d[~d.multi],False,True),
            ('six_common_all',d[d.six_complete],True,True),('six_common_multi',d[d.six_complete&d.multi],True,True),
            ('six_common_nonmulti',d[d.six_complete&~d.multi],True,True),
            ('exclude_single_6h',d[d.later_class.ne('single_6h')],False,False),
            ('strict_bridge_tropical',d[d.strict_tropical_bridge],False,False),
            ('without_Halola',d[d.sid.ne('2015188N08194')],False,False),
            ('without_Freddy2023',d[d.sid.ne('2023036S12117')],False,False)]
    for basin in ['WP','EP','SI','NA','SP','NI']:
        scopes.append(('without_basin_'+basin,d[d.basin_genesis.ne(basin)],False,False))
    threshold_support=[]
    for beta in ['085','095']:
        original=pd.read_csv(V/'source_data'/('durations_beta'+beta+'.csv'),keep_default_na=False,na_values=['','nan','NaN'])
        z=original[original.sid.isin(ids)&original.main_complete].copy()
        common=set(d.sid)&set(z.sid)
        # Available and paired scopes both remain subgroups of the one master.
        scopes.extend([('beta'+beta+'_available',z,False,False),('beta'+beta+'_common',z[z.sid.isin(common)],False,False),
                       ('main_for_beta'+beta,d[d.sid.isin(common)],False,False)])
        for sid in sorted(ids):
            threshold_support.append(dict(beta=beta,sid=sid,available_complete_support=sid in common))
    summaries=[];slopes=[];annuals=[];periods=[];points=[];ratios=[];members=[];scope_info=[]
    for label,z,six,full in scopes:
        print('RUF_SCOPE',label,len(z),flush=True)
        z=z.sort_values('sid').reset_index(drop=True)
        assert set(z.sid)<=ids
        if len(z)<2:continue
        members.extend(dict(scope=label,sid=sid) for sid in z.sid)
        summaries+=env['sid_summary'](z,label,six,full)
        sl,an=env['annual'](z,label,six,formal=label=='main_all');slopes+=sl;annuals+=an
        pe,pp=env['period'](z,label,draws,six);periods+=pe;points+=pp
        if full:ratios+=env['ratio_summary'](z,label)
        scope_info.append(dict(scope=label,n_sid=len(z),multi_n=int(z.multi.sum()),main_sid_n=len(ids),
                               mean_R_h=z.T_R_h.mean(),mean_U_h=z.T_U_h.mean(),mean_F_h=z.T_F_h.mean(),
                               mean_L_h=z.T_L_h.mean(),mean_B_h=z.T_B_h.mean(),
                               earlier_near_n=int(z.pre_maturity_near_h.gt(0).sum()),earlier_near_h=z.pre_maturity_near_h.sum()))
    for name,rows in [('duration_summaries',summaries),('annual_slopes',slopes),('annual_means',annuals),
                      ('period_changes',periods),('period_means',points),('duration_ratios',ratios),
                      ('membership',members),('scope_selection_summary',scope_info),('threshold_master_support',threshold_support)]:
        pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False)
    env['influence'](d)
    table=pd.DataFrame(summaries)
    selected=table[(table.scope=='main_all')&(table.metric=='F_minus_R')&(table.statistic=='mean')]
    assert len(selected)==1
    # Exact independent readback of the original reporting mean and all draws.
    with np.load(OUT/'sid_draws_main_all.npz') as bank:
        sorted_d=d.set_index('sid').loc[bank['sids']]
        delta=(sorted_d.T_F_h-sorted_d.T_R_h).to_numpy(float)
        values=bank['sid_multiplicity'].astype(float)@delta/len(ids)
        ix=list(bank['metrics']).index('F_minus_R');iy=list(bank['statistics']).index('mean')
        assert np.allclose(values,bank['values'][:,ix,iy],rtol=0,atol=1e-12)
        assert np.isclose(selected.estimate.iloc[0],delta.mean(),rtol=0,atol=1e-12)
    pd.DataFrame({'sid':sorted_d.index,'F_h':sorted_d.T_F_h,'R_h':sorted_d.T_R_h,'F_minus_R_h':delta}).to_csv(OUT/'F_minus_R_reporting_identity.csv',index=False)
    record=dict(status='COMPLETE_ORIGINAL_RUF_DURATION_ESTIMATORS_ON_FROZEN_MASTER',UTC=datetime.now(timezone.utc).isoformat(),
                main_sid_n=len(ids),main_sid_sha256=freeze['main_sid_sha256'],scopes=len(scopes),B=2000,seed=20260920,
                block_bank_reverified_exact=True,no_new_estimators=True,ERA5_reads=0,
                accepted_bridge_metadata_reused_after_exact_duration_identity=restored_columns,
                existing_FminusR_reporting_row=selected.iloc[0].to_dict(),
                threshold_support=[dict(beta=x,available_sid_n=int(pd.DataFrame(threshold_support).query('beta==@x').available_complete_support.sum())) for x in ['085','095']],
                excluded_old_expanded_scope='The old expanded cohort is outside the single current master population; no expanded base cohort is generated.',
                environmental_threshold_status='Separate original environmental sensitivity refits/summaries still require frozen master filtering',
                source_hashes=[dict(path=str(p),sha256=sha(p)) for p in [source,yearfile,HERE/'source_data/main_durations.parquet',original_duration_path,V/'source_data/durations_beta085.csv',V/'source_data/durations_beta095.csv']],
                outputs=[dict(path=str(p),sha256=sha(p)) for p in OUT.glob('*') if p.is_file()],elapsed_s=time.monotonic()-started)
    receipt.write_text(json.dumps(record,indent=2,ensure_ascii=False,default=str)+'\n',encoding='utf-8')
    print(json.dumps({k:record[k] for k in ['status','main_sid_n','scopes','existing_FminusR_reporting_row','elapsed_s']}))


if __name__=='__main__':main()
