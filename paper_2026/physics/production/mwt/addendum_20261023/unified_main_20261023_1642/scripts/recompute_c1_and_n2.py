from _release_paths import release_path as _release_path
"""Run original C1 and B2/N2 estimators with only the frozen master membership changed."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[key]='1'
from pathlib import Path
import ast,json,hashlib,datetime,time,sys,types
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t as student_t
from statsmodels.stats.multitest import multipletests
import psutil
U=Path(__file__).absolute().parents[1]
V=Path(str(_release_path('legacy_project','ERA5_environment_v1/run_20260920_215821/representation_value_v4')))
IDS=set((U/'source_data/main_sid.txt').read_text('utf-8').splitlines())
CORE=['X1_sst_c','X2_shear_ms','X3_mid_rh_pct','X4_d200_1e5s','X5_minus_omega500_pa_s','X6_signed_vo850_1e5s']
PERIODS={'P1':(1982,1992),'P2':(1993,2003),'P3':(2004,2014),'P4':(2015,2023)}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str),'utf-8')
def read(p):
 d=pd.read_parquet(p) if Path(p).suffix=='.parquet' else pd.read_csv(p,keep_default_na=False,na_values=[''],float_precision='round_trip')
 if 'sid' in d:d=d[d.sid.isin(IDS)].copy()
 for c in ['main_complete','six_complete','multi','included','exclude_correct_G1']:
  if c in d:
   z=d[c].astype(str).str.lower().map({'true':True,'false':False,'1':True,'0':False});assert z.notna().all(),c;d[c]=z.astype(bool)
 return d
def funcs(path,env,replace=()):
 txt=Path(path).read_text('utf-8')
 for a,b in replace:
  assert a in txt,a;txt=txt.replace(a,b)
 nodes=[n for n in ast.parse(txt).body if isinstance(n,ast.FunctionDef)]
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
 return env
def common():
 return dict(Path=Path,np=np,pd=pd,ast=ast,json=json,hashlib=hashlib,datetime=datetime,time=time,sys=sys,
 sm=sm,student_t=student_t,multipletests=multipletests,V=V,BASE=V.parent/'review_completion_v3_2',B=2000,SEED=20260920,
 CORE=CORE,PERIODS=PERIODS,UNITS=['degC','m s-1','percent','1e-5 s-1','Pa s-1','1e-5 s-1'],CONTRASTS=['F_minus_U','U_minus_R'],sha=sha)
def run_c1():
 out=U/'statistics/c1_threshold_multi';out.mkdir(parents=True,exist_ok=True)
 src=V/'scripts/C1_threshold_multi_completion_v4.py'
 if (U/'reports/c1_threshold_multi_receipt.json').exists():return
 a=funcs(V/'scripts/analyze_durations_v4.py',common());a['D']=out/'A'
 e=funcs(V/'scripts/analyze_environment_v4.py',common());e['OUT']=out/'B1'
 # Function namespaces retain the new destinations when the original main assigns A.D and E.OUT.
 A=types.SimpleNamespace(**a);E=types.SimpleNamespace(**e)
 ns=common();ns.update(A=A,E=E,O=out,__file__=str(src))
 funcs(src,ns,[('len(c2)==2368','len(c2)==2340'),('==2326','==2300'),('len(main)==2366','len(main)==2340'),('C2_selection_n=2326','C2_selection_n=2300'),('main_source_n=2366','main_source_n=2340')])
 ns['read']=read;ns['main']()
 flow=pd.read_csv(out/'sample_flow.csv');assert len(flow)==19 and flow.n_sid.max()<=2340
 assert set(read(out/'inputs/main_complete.csv').sid)==IDS
 binding=read(out/'A/C5_REUSED_membership.csv');assert binding.sid.nunique()==123
 dump(U/'reports/c1_threshold_multi_receipt.json',dict(status='COMPLETE_ORIGINAL_C1_SCOPES_ON_FROZEN_MASTER',master_n=len(IDS),A_scopes=8,B1_scopes=11,source=str(src),source_sha256=sha(src),driver_sha256=sha(__file__),master_sha256=sha(U/'source_data/main_sid.txt'),outputs={str(p.relative_to(out)):sha(p) for p in out.rglob('*.csv')}))
 print('C1_COMPLETE',flush=True)
def run_n2():
 out=U/'statistics/n2';out.mkdir(parents=True,exist_ok=True)
 if (U/'reports/n2_receipt.json').exists():return
 src=V/'scripts/analyze_B2_v4.py';ns=common();ns.update(OUT=out,TARGETS=['log_R','log_U','log_F','GB','GL'],__file__=str(src))
 funcs(src,ns,[('len(e)==2368','len(e)==2340')]);ns.update(read=read,active_plan=lambda:json.loads((U/'config/main_definition_frozen.json').read_text('utf-8')),dependency=lambda p:dict(path=str(p),sha256=sha(p)),dump=lambda p,x:dump(out/Path(p).name,x))
 ns['main']()
 z=pd.read_csv(out/'N2_contrasts.csv');assert len(z)==12 and set(z.contrast)=={'GB','GL'}
 inputs=pd.read_csv(out/'model_inputs.csv');assert set(inputs.sid)<=IDS
 dump(U/'reports/n2_receipt.json',dict(status='COMPLETE_ORIGINAL_N2_FIXED_F_ENTRY_MODELS_ON_FROZEN_MASTER',n_sid=len(inputs),master_n=len(IDS),source=str(src),source_sha256=sha(src),driver_sha256=sha(__file__),master_sha256=sha(U/'source_data/main_sid.txt'),outputs={str(p.relative_to(out)):sha(p) for p in out.iterdir() if p.is_file()}))
 print('N2_COMPLETE',flush=True)
if __name__=='__main__':
 proc=psutil.Process();proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
 dump(U/'reports/c1_n2_identity.json',dict(pid=proc.pid,create_time=proc.create_time(),cmdline=proc.cmdline()))
 run_c1();run_n2()
