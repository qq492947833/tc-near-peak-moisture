from _release_paths import release_path as _release_path
"""Run the accepted exact-wind sensitivity on the refitted master P5 jobs."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import ast,json,hashlib,sys,datetime,platform,shutil
import numpy as np,pandas as pd,psutil
U=Path(__file__).absolute().parents[1];B=U/'statistics/m6';B.mkdir(exist_ok=True)
OLD=Path(str(_release_path('legacy_project','v67_review_20261014/M6')))
V=Path(str(_release_path('mechanism_v5','')))
IDS=set((U/'source_data/main_sid.txt').read_text('utf8').splitlines());src=OLD/'run_m6.py'
assert (U/'reports/p4_p5_models_receipt.json').exists()
shutil.copyfile(OLD/'specification.json',B/'specification.json')
text=src.read_text('utf8');assert "B.parent/'M2/per_interval.csv'" in text;text=text.replace("B.parent/'M2/per_interval.csv'",str(repr(str(OLD.parent/'M2/per_interval.csv'))))
JOBS={p:U/'statistics/p4_p5_models/jobs'/('P5__'+p) for p in ['midRH','q700_storm_relative_horizontal_advection','q850_moisture_flux_convergence','shear200_500','PI_headroom']}
ns=dict(os=os,Path=Path,np=np,pd=pd,json=json,hashlib=hashlib,sys=sys,datetime=datetime,platform=platform,B=B,V=V,JOBS=JOBS)
exec(compile(ast.Module(body=[n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)],type_ignores=[]),str(src),'exec'),ns)
original_read=ns['read']
def read(p):
 d=original_read(p);return d[d.sid.isin(IDS)].copy() if 'sid' in d else d
ns['read']=read
proc=psutil.Process();proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
(U/'reports/m6_identity.json').write_text(json.dumps(dict(pid=proc.pid,create_time=proc.create_time(),cmdline=proc.cmdline())),'utf8')
ns['main']()
d=pd.read_csv(B/'sensitivity_results.csv');assert d.process.nunique()==5 and set(d.n_sid)<=set(range(1,2341))
receipt=dict(status='COMPLETE_ORIGINAL_EXACT_WIND_RESTRICTION_ON_FROZEN_MASTER',master_n=2340,source=str(src),source_sha256=ns['sha'](src),driver_sha256=ns['sha'](Path(__file__)),outputs={str(p.relative_to(B)):ns['sha'](p) for p in B.glob('*.csv')},review=ns['sha'](B/'numerical_and_estimand_check.json'))
(U/'reports/m6_receipt.json').write_text(json.dumps(receipt,indent=2),'utf8')
