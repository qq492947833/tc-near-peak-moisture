from _release_paths import release_path as _release_path
"""Recompute original stage climate and pointwise boundary companions on master."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import ast,sys,json,hashlib,datetime,csv,collections,math,importlib.util,time,shutil
import numpy as np
import pandas as pd
import psutil
HERE=Path(__file__).absolute().parents[1];ADD=HERE.parent;OUT=HERE/'statistics/p4'
PROJECT=Path(str(_release_path('project','')))
ST=PROJECT/'03_Work/v62/stages'
V=Path(str(_release_path('mechanism_v5','')))
NINE=V/'p4_p5/nine_lag_companion_v1'
sys.dont_write_bytecode=True;sys.path.insert(0,str(V/'src'))
from v5_statistics import bootstrap_weights

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return pd.read_csv(p,keep_default_na=False,na_values=['','nan','NaN'],float_precision='round_trip',low_memory=False)
def extract(text,path,names,env):
    nodes=[n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and (names is None or n.name in names)]
    if names:assert {n.name for n in nodes}==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
def dump(p,x):p.write_text(json.dumps(x,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else str(v))+'\n','utf-8')

def main():
    tick=time.monotonic();psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    ids=set((HERE/'source_data/main_sid.txt').read_text('utf-8').splitlines());assert len(ids)==2340
    bindings={};cache={}
    for name in ['P4_boundary_requests.csv','P4_environment_ready.csv','P4_shared_climate_SID_source.csv']:
        p=ADD/'source_data/de_P4'/name;bindings[str(p)]=sha(p);x=read(p);x=x[x.sid.isin(ids)].copy();assert set(x.sid)==ids
        x.to_csv(OUT/name,index=False);cache[name]=x
    source=NINE/'p4_nine_lag_companion.py'
    text=source.read_text('utf-8').replace("if len(pool)!=2368:raise ValueError('production draw pool must retain all frozen2368SID')", "if len(pool)!=2340:raise ValueError('master draw pool must contain2340SID')")
    ctx=dict(Path=Path,np=np,argparse=None,csv=csv,json=json,hashlib=hashlib,datetime=datetime,sys=sys,os=os,
             collections=collections,math=math,HERE=NINE,ROOT=V,NAMESPACE='P4_nine_lag_descriptive_v1',bootstrap_weights=bootstrap_weights,__file__=str(source))
    extract(text,source,None,ctx)
    cfg=json.loads((NINE/'frozen_config.json').read_text('utf-8'))
    points=ctx['prepare_points'](ctx['readrows'](OUT/'P4_boundary_requests.csv'),ctx['readrows'](OUT/'P4_environment_ready.csv'),cfg)
    assert len(points)==2340*2*9*5 and not any(p['input_missing'] for p in points)
    report=dict(process_gate=dict(allowed=list(cfg['processes']),pending={}),scope='Same pointwise estimators on frozen master2340; no complete-window gate',source=str(OUT/'P4_environment_ready.csv'))
    ctx['run_descriptive'](report,points,cfg,OUT/'nine_lag')
    print('NINE_LAG_COMPLETE',flush=True)
    # Same exact new whole-trajectory draw bank for the216 climate targets.
    bank=ctx['build_draw_plan'](sorted(ids),2000,510082026,OUT/'nine_lag/draw_indices')
    csource=ADD/'scripts/de_climate_trajectory_summaries.py';text=csource.read_text('utf-8')
    text=text.replace('assert len(group)==2368','assert len(group)==len(pool)').replace('reused shared original2368SID trajectories','shared frozen-master2340SID trajectories')
    cenv=dict(np=np,pd=pd,KINDS=['raw_common','climate_common','anomaly_common'])
    extract(text,csource,['calculate'],cenv)
    z,draws,stages=cenv['calculate'](cache['P4_shared_climate_SID_source.csv'],bank['sid'],bank['weights']);assert len(z)==216
    dest=OUT/'climate_trajectory';dest.mkdir(exist_ok=True)
    z.to_csv(dest/'P4_raw_climate_anomaly_CI_n.csv',index=False);stages.to_csv(dest/'P4_actual_stage_composition.csv',index=False)
    np.savez_compressed(dest/'pointwise_draws.npz',**draws)
    dump(dest/'receipt.json',dict(status='COMPLETE216_TARGETS',seed=510082026,B=2000,master_n=2340,draw_bank=bank['bank_path'],methods_unchanged=True))
    print('CLIMATE_TRAJECTORIES_COMPLETE',flush=True)
    # Stage companion: identical intervals and sampled-stage method; refresh
    # aggregate coverage using only the frozen main histories.
    dest=OUT/'stage_climate';dest.mkdir(exist_ok=True)
    for name in ['bootstrap_sid_order.csv','bootstrap_sid_multiplicities.npy']:
        src=HERE/'statistics/stages'/name;dst=dest/name
        if dst.exists():assert sha(dst)==sha(src)
        else:shutil.copy2(src,dst)
    trpath=ADD/'source_data/de_existing_sources/track_sp_qc_relabelled.parquet';tr=pd.read_parquet(trpath);tr=tr[tr.sid.isin(ids)].copy()
    tr.to_parquet(dest/'track_sp_qc_relabelled.parquet',index=False)
    src=ST/'climate_companion_b.py';text=src.read_text('utf-8').replace("pd.read_parquet(spec['inputs']['accepted_sp_track']['path'])","pd.read_parquet(REVISED_TRACK)")
    text=text.replace('registered_before_new_companion_estimates=True','registered_before_new_companion_estimates=False, frozen_master_update=True')
    def write_json(name,value):dump(dest/name,value)
    env=dict(np=np,pd=pd,json=json,SOURCE=OUT/'P4_shared_climate_SID_source.csv',EXPECTED=sha(OUT/'P4_shared_climate_SID_source.csv'),
             SPEC=ST.parent/'analysis_spec_v62.json',EXPECTED_B='55d78eed00f44d138bbe0c26382bec48eae527d0c18d534d62a2e05321e4a593',OUT=dest,
             sha=sha,write_json=write_json,REVISED_TRACK=dest/'track_sp_qc_relabelled.parquet',COMPONENTS=['raw_common','climate_common','anomaly_common'],
             STAGES=['D','E','F','Res'],PAIRS=[('D','E'),('E','F'),('F','Res')])
    extract(text,src,['run'],env);env['run']()
    for p,h in bindings.items():assert sha(p)==h
    dump(HERE/'reports/boundary_companions_receipt.json',dict(status='COMPLETE90_POINTWISE216_CLIMATE_AND42_STAGE_COMPANIONS_ON_MASTER',master_sid_n=2340,
         source_hashes=bindings,raw_ERA5_reads=0,new_methods=False,stage_climate_full_coverage_claim=False,elapsed_s=time.monotonic()-tick,
         output_hashes={str(p.relative_to(OUT)):sha(p) for d in [OUT/'nine_lag',OUT/'climate_trajectory',OUT/'stage_climate'] for p in d.rglob('*') if p.is_file()}))

if __name__=='__main__':main()
