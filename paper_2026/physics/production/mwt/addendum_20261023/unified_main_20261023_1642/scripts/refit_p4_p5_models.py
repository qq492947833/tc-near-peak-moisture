from _release_paths import release_path as _release_path
"""Refit the original 25 P4/P5 jobs and 40 fixed-family targets on master SID."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
from datetime import datetime,timezone
import ast,sys,json,hashlib,math,csv,time
import numpy as np
import pandas as pd
import psutil
from scipy.optimize import linprog

HERE=Path(__file__).absolute().parents[1];ADD=HERE.parent
V=Path(str(_release_path('mechanism_v5','')))
U=V/'upper_profile_100_50_interpolation_v1';SRC=V/'src/p4_p5_analysis_v5.py'
INPUT=ADD/'source_data/de_P4';OUT=HERE/'statistics/p4_p5_models'
sys.dont_write_bytecode=True;sys.path.insert(0,str(V/'src'))
from v5_statistics import bootstrap_weights,weighted_fit,target_estimable,summarize_draws

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    tick=time.monotonic();proc=psutil.Process();proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    OUT.mkdir(parents=True,exist_ok=True)
    ids=set((HERE/'source_data/main_sid.txt').read_text('utf-8').splitlines());assert len(ids)==2340
    ctx=dict(np=np,pd=pd,json=json,hashlib=hashlib,datetime=datetime,timezone=timezone,Path=Path,math=math,csv=csv,os=os,
             linprog=linprog,bootstrap_weights=bootstrap_weights,weighted_fit=weighted_fit,target_estimable=target_estimable,
             summarize_draws=summarize_draws,__file__=str(SRC),BOOTSTRAP_REPEATS=2000,BOOTSTRAP_SEED=510082026)
    tree=ast.parse((V/'src/native_fields_v5.py').read_text('utf-8'))
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='utc_datetime']
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'accepted_native_UTC','exec'),ctx)
    tree=ast.parse(SRC.read_text('utf-8'))
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)]+[n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ['PROCESS_COLUMNS','BACKGROUND_NUMERIC_NAMES'] for t in n.targets)]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SRC),'exec'),ctx)
    def atomic(path,obj):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
        tmp.write_text(json.dumps(ctx['clean'](obj),ensure_ascii=False,indent=2)+'\n','utf-8');started=time.monotonic()
        while True:
            try:os.replace(tmp,path);return
            except OSError as exc:
                if getattr(exc,'winerror',None) not in (5,32,33) or time.monotonic()-started>60:raise
                time.sleep(.2)
    ctx['atomic_json']=atomic
    paths=[SRC,V/'src/v5_statistics.py',INPUT/'P4_environment_ready.csv',INPUT/'models/P4_fixed_pair_design.csv',
           U/'p45_execution_v2/readyC/attempt_0001/environment_ready.csv',V/'p4_p5/analysis_prepared/P5_fixed_pair_design.csv',
           V/'p4_p5/P5_per_SID_time_audit.csv',HERE/'source_data/main_sid.txt',Path(__file__)]
    bindings={str(p):sha(p) for p in paths}
    identity=dict(pid=proc.pid,create_time=proc.create_time(),command_line=proc.cmdline(),bindings=bindings,UTC=datetime.now(timezone.utc).isoformat())
    atomic(HERE/'reports/p4_p5_models_identity.json',identity)
    read=ctx['read_rows'];write=ctx['write_rows']
    old_env={r['request_id']:r for r in read(paths[4])}
    env={r['request_id']:r for r in read(INPUT/'P4_environment_ready.csv') if r['sid'] in ids}
    env.update({k:r for k,r in old_env.items() if r.get('module')=='P5' and r['sid'] in ids})
    timetable={r['sid']:r for r in read(V/'p4_p5/P5_per_SID_time_audit.csv') if r['sid'] in ids}
    levels=sorted(set(r['basin_genesis'] for r in timetable.values()));assert len(levels)==6
    pairs4=[r for r in read(INPUT/'models/P4_fixed_pair_design.csv') if r['sid'] in ids]
    pairs5=[r for r in read(V/'p4_p5/analysis_prepared/P5_fixed_pair_design.csv') if r['sid'] in ids]
    assert set(r['sid'] for r in pairs4)==ids
    write(OUT/'P4_fixed_pair_design.csv',pairs4);write(OUT/'P5_fixed_pair_design.csv',pairs5)
    evidence=json.loads((U/'p45_execution_v2/mergeC/attempt_0001/merge_receipt.json').read_text('utf-8'))
    provenance={}
    for item in evidence['job_evidence']:
        cp=Path(item['complete']);assert sha(cp)==item['sha256'];done=json.loads(cp.read_text('utf-8'))
        for f in done['outputs']:assert sha(cp.parent/f['name'])==f['sha256']
        provenance[item['job_id']]=cp.parent
    assert len(provenance)==25
    results=[];checks=[]
    for job,old_folder in provenance.items():
        parts=job.split('__');kind='P5' if parts[0]=='P5' else 'P4';process=parts[-1]
        selected=pairs5 if kind=='P5' else [r for r in pairs4 if r['boundary']==parts[0] and r['change_window']==parts[1]]
        a=ctx['build_pair_arrays'](selected,env,timetable,process,kind,levels)
        assert set(a['sid'])<=ids
        # Check every reused source value and design row against its latest
        # accepted E-revision output, or the unchanged original job where needed.
        candidate=INPUT/'models/jobs'/job/'paired_source_and_design.csv'
        latest_reanchored=kind=='P4' and candidate.exists()
        if kind=='P5' or not candidate.exists():candidate=old_folder/'paired_source_and_design.csv'
        prior=pd.read_csv(candidate,keep_default_na=False,na_values=['','nan','NaN'],float_precision='round_trip',low_memory=False)
        prior=prior[prior.sid.isin(ids)].reset_index(drop=True);current=pd.DataFrame(a['records'])
        assert current.sid.tolist()==prior.sid.tolist(),job
        cols=['value_before','value_after','difference_after_minus_before']+[c for c in prior if c.startswith('W_')]
        if kind=='P5':cols+=['A_signed']
        elif latest_reanchored:cols+=['allocation']
        maxerr=0.
        for c in cols:
            x=pd.to_numeric(current[c],errors='coerce').to_numpy(float);y=pd.to_numeric(prior[c],errors='coerce').to_numpy(float)
            assert np.array_equal(np.isfinite(x),np.isfinite(y)),(job,c)
            err=float(np.nanmax(abs(x-y))) if np.isfinite(x).any() else 0.;assert err<1e-10,(job,c,err);maxerr=max(maxerr,err)
        checks.append(dict(job=job,retained_sid_n=len(set(a['sid'])),retained_pair_n=len(a['sid']),source=str(candidate),max_absolute_source_error=maxerr))
        folder=OUT/'jobs'/job
        result=ctx['run_pair_job'](a,kind,process,job,folder,include_slope=kind=='P5' or parts[1]=='minus24_to_0',include_background=True)
        results.append(result)
        atomic(HERE/'reports/p4_p5_models_status.json',dict(status='RUNNING',completed_jobs=len(results),planned_jobs=25,
            latest=job,pid=proc.pid,create_time=proc.create_time(),elapsed_s=time.monotonic()-tick))
        print('P4_P5_COMPLETE',len(results),'/25',job,round(time.monotonic()-tick,1),flush=True)
    family=ctx['finalize_prespecified_families'](results,OUT);assert len(family)==40
    for p,h in bindings.items():assert sha(p)==h,'Read-only source changed: '+p
    record=dict(status='COMPLETE_ALL25_P4_P5_JOBS_AND40_ORIGINAL_FAMILY_TARGETS_ON_FROZEN_MASTER',main_sid_n=len(ids),
        P4_source_sid_n=len(set(r['sid'] for r in pairs4)),P5_source_sid_n=len(set(r['sid'] for r in pairs5)),
        unchanged_statistical_methods=True,draws=2000,seed=510082026,no_top_up=True,raw_ERA5_reads=0,source_checks=checks,
        bindings=bindings,output_hashes={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()},elapsed_s=time.monotonic()-tick)
    atomic(HERE/'reports/p4_p5_models_receipt.json',record);atomic(HERE/'reports/p4_p5_models_status.json',record)

if __name__=='__main__':main()
