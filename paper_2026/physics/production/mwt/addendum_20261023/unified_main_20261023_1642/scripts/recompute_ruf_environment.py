from _release_paths import release_path as _release_path
"""Repeat accepted B1 R/U/F environmental estimators on the frozen master.

All precomputed per-SID values are unchanged; cohort-dependent estimates and
bootstrap banks are regenerated. Original estimator functions execute unchanged
in isolated namespaces. No new environmental or threshold model is introduced.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[k]='1'
from pathlib import Path
from datetime import datetime, timezone
import ast, json, hashlib, time
import numpy as np
import pandas as pd
import psutil

HERE=Path(__file__).absolute().parents[1]
V=Path(str(_release_path('legacy_project','ERA5_environment_v1/run_20260920_215821/representation_value_v4')))
OUT=HERE/'statistics/ruf_environment'
CORE=['X1_sst_c','X2_shear_ms','X3_mid_rh_pct','X4_d200_1e5s','X5_minus_omega500_pa_s','X6_signed_vo850_1e5s']
UNITS=['degC','m s-1','percent','1e-5 s-1','Pa s-1','1e-5 s-1']
CONTRASTS=['F_minus_U','U_minus_R']
PERIODS={'P1':(1982,1992),'P2':(1993,2003),'P3':(2004,2014),'P4':(2015,2023)}


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    tick=time.monotonic()
    proc=psutil.Process();proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    OUT.mkdir(parents=True,exist_ok=True)
    freeze=json.loads((HERE/'reports/main_sid_freeze_receipt.json').read_text('utf-8'))
    assert sha(freeze['main_sid_path'])==freeze['main_sid_sha256']
    ids=set(Path(freeze['main_sid_path']).read_text('utf-8').splitlines())
    old=V/'results/B1'
    source=V/'scripts/analyze_environment_v4.py'
    source_text=source.read_text('utf-8')
    names={'read','dump','save','matrix_for','weighted_boot','summary_scope','loo','group_contributions'}
    nodes=[n for n in ast.parse(source_text).body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes}==names
    env=dict(Path=Path,np=np,pd=pd,json=json,hashlib=hashlib,time=time,sha=sha,
             V=V,OUT=OUT,CORE=CORE,UNITS=UNITS,CONTRASTS=CONTRASTS,B=2000,SEED=20260920,
             PERIODS=PERIODS,SHORT_LABEL='Excluding cyclones with exactly one 6-h later near-peak episode',__file__=__file__)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),env)
    read,save,dump=[env[x] for x in ['read','save','dump']]
    deps=[source,V/'scripts/pair_environment_thresholds_v4.py',V/'source_data/durations.csv',
          V/'source_data/durations_beta085.csv',V/'source_data/durations_beta095.csv',
          old/'per_SID_component_environment.parquet',old/'per_SID_environment_contrasts.csv',
          V/'audit/environment_sources/B1_frozen_reference_scale.csv',V/'inputs/year_block_draws.npz',
          HERE/'source_data/main_durations.parquet',HERE/'source_data/main_sid.txt']
    bindings={str(p):sha(p) for p in deps}
    active=dict(sha256=sha(HERE/'config/main_definition_frozen.json'),master_sid_sha256=freeze['main_sid_sha256'],
                original_method_source_sha256=sha(source),note='Same B1 estimators and prespecified scopes restricted to the single master')
    dump(HERE/'reports/ruf_environment_identity.json',dict(pid=proc.pid,create_time=proc.create_time(),
         command_line=proc.cmdline(),UTC=datetime.now(timezone.utc).isoformat(),source_hashes=bindings))
    d=read(V/'source_data/durations.csv');d=d[d.sid.isin(ids)].sort_values('sid').reset_index(drop=True)
    assert set(d.sid)==ids and d.main_complete.all()
    updated=read(HERE/'source_data/main_durations.parquet').set_index('sid').loc[d.sid]
    for col in ['T_R_h','T_U_h','T_F_h','T_L_h','T_B_h','T_R_long_h','T_R5_h','T_U5_h','T_F5_h',
                'later_interval_episode_n','later_max_episode_h','pre_maturity_near_h']:
        assert np.array_equal(d[col].to_numpy(float),updated[col].to_numpy(float),equal_nan=True),col
    for col in ['main_complete','six_complete','multi','basin_genesis','genesis_year']:
        assert np.array_equal(d[col].to_numpy(),updated[col].to_numpy()),col
    con=read(old/'per_SID_environment_contrasts.csv');con=con[con.sid.isin(ids)].copy()
    comp=read(old/'per_SID_component_environment.parquet');comp=comp[comp.sid.isin(ids)].copy()
    # Cache values are reused only because all defining window lengths and
    # relevant lifecycle boundary times agree for every retained SID.
    for col in ['maturity_time','resolution_time','termination_time','lmi_time']:
        assert np.array_equal(pd.to_datetime(d[col],utc=True),pd.to_datetime(updated[col],utc=True)),col
    assert set(con[con.config.eq('main')].sid)==ids
    save(con,OUT/'per_SID_environment_contrasts.csv')
    save(comp,OUT/'per_SID_component_environment.parquet')
    ref=read(V/'audit/environment_sources/B1_frozen_reference_scale.csv').set_index('variable')
    with np.load(V/'inputs/year_block_draws.npz') as z:year_draws={p:z[p] for p in PERIODS}
    rng=np.random.default_rng(20260920)
    for key,(lo,hi) in PERIODS.items():
        years=np.arange(lo,hi+1);n=len(years);starts=rng.integers(0,n,size=(2000,(n+1)//2))
        expected=years[((starts[:,:,None]+[0,1])%n).reshape(2000,-1)[:,:n]]
        assert np.array_equal(year_draws[key],expected)
    configs={'main':(d,None),'C5':(d[d.six_complete].copy(),None)}
    for beta in ['085','095']:
        q=read(V/'source_data'/('durations_beta'+beta+'.csv'))
        q=q[q.sid.isin(ids)&q.main_complete].copy()
        available=set(con[con.config.eq('beta'+beta)].sid)
        assert set(q.sid)<=available
        configs['beta'+beta]=(q,None)
    lookup={k:con[con.config.eq(k)].copy() for k in configs}
    env['loo'](d,lookup['main'],ref);env['group_contributions'](d,lookup['main'],ref)
    # Execute the original full prespecified scope-construction block unchanged.
    code=source_text[source_text.index(" scopes=[('main_all'"):source_text.index(' ss=[];pc=[];pm=[];mm=[];scopeaudit=[]')]
    env.update(d=d,configs=configs)
    exec(compile('\n'.join(line[1:] for line in code.splitlines()),str(source)+'::scope_block','exec'),env)
    scopes=env['scopes'];ss=[];pc=[];pm=[];mm=[];scopeaudit=[]
    for label,config,dd in scopes:
        assert set(dd.sid)<=ids
        if not len(dd):continue
        a,b,c,m=env['summary_scope'](dd,lookup[config],label,ref,year_draws,active['sha256'])
        ss.append(a);pc.append(b);pm.append(c);mm.append(m)
        scopeaudit.append(dict(scope=label,config=config,n_sampling_frame=len(dd),multi_n=int(dd.multi.sum()),
             G1_in_frame=int(dd.later_group.eq('G1').sum()),G2_in_frame=int(dd.later_group.eq('G2').sum()),
             n_environment_valid_min=int(a.n_sid.min()),n_environment_valid_max=int(a.n_sid.max())))
        dump(HERE/'reports/ruf_environment_status.json',dict(status='RUNNING',completed_scopes=len(ss),
             planned_scopes=len(scopes),latest_scope=label,pid=proc.pid,create_time=proc.create_time(),elapsed_s=time.monotonic()-tick))
    for name,z in [('paired_difference_summary.csv',pd.concat(ss,ignore_index=True)),
                   ('period_difference_changes.csv',pd.concat(pc,ignore_index=True)),
                   ('period_difference_means.csv',pd.concat(pm,ignore_index=True)),
                   ('membership_all_scopes.parquet',pd.concat(mm,ignore_index=True)),
                   ('scope_selection_summary.csv',pd.DataFrame(scopeaudit))]:save(z,OUT/name)
    dump(OUT/'run_audit.json',dict(status='COMPLETE_B1_RESTRICTED_MASTER',main_sid_n=len(ids),source_hashes=bindings))
    # The accepted paired-threshold main is used verbatim up to its optional
    # historical E6 binding. E6 depends on cohort and is not reused here.
    tsource=(V/'scripts/pair_environment_thresholds_v4.py').read_text('utf-8')
    tsource=tsource[tsource.index('def main():'):tsource.index(' # Bind and reuse existing pooled-hour E6')]
    tenv={**env,'plan':lambda:(None,active)}
    exec(compile(tsource,str(V/'scripts/pair_environment_thresholds_v4.py'),'exec'),tenv)
    tenv['main']()
    threshold=read(OUT/'threshold_common_difference.csv')
    assert len(threshold)==36 and set(threshold.alternate)=={'beta085','beta095','C5'}
    for p,h in bindings.items():assert sha(p)==h,'Source changed: '+p
    record=dict(status='COMPLETE_B1_AND_36_ORIGINAL_PAIRED_THRESHOLD_TARGETS_ON_FROZEN_MASTER',
         UTC=datetime.now(timezone.utc).isoformat(),main_sid_n=len(ids),main_sid_sha256=freeze['main_sid_sha256'],
         scopes=len(scopes),threshold_targets=len(threshold),bootstrap_attempts=2000,seed=20260920,
         unchanged_estimators=True,block_bank_verified=True,source_hashes=bindings,raw_ERA5_reads=0,
         source_per_sid_values_reused_after_exact_window_and_sid_identity=True,
         E6_pooled_hour_scope='Separate original pooled-hour analysis still requires frozen-master update; no old pooled-hour estimates promoted',
         output_hashes={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()},elapsed_s=time.monotonic()-tick)
    dump(HERE/'reports/ruf_environment_receipt.json',record)
    dump(HERE/'reports/ruf_environment_status.json',record)
    print(json.dumps({k:record[k] for k in ['status','main_sid_n','scopes','threshold_targets','elapsed_s']}),flush=True)


if __name__=='__main__':main()
