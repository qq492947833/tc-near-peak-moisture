# Public path adapter; the scientific functions below retain their original implementation.
import sys as _sys
from pathlib import Path as _Path
_PACKAGE=next(p for p in _Path(__file__).resolve().parents if (p/'release_paths.py').is_file())
_sys.path.insert(0,str(_PACKAGE))
from release_paths import ADDENDUM, UNIFIED, PACKAGE, bootstrap
"""One frozen latitude comparison; reuse the accepted interval likelihood verbatim.
All inputs outside this directory are read-only. No ERA5/MWT reads or figure writes.
"""
from pathlib import Path
import runpy,sys,os,json,hashlib,ast,time,traceback,argparse,pickle
O=Path(__file__).resolve().parent
rt=bootstrap(O/'temp')
if sys.version_info[:2]==(3,11):sys.path=[p for p in sys.path if p!=rt['vendor']]
import numpy as np
import pandas as pd
import scipy
from scipy.special import expit
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits
threadpool_limits(1)
try:
 import psutil
 psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
except (ImportError, AttributeError, PermissionError):pass
U=UNIFIED
SOURCE=O/'interval_hazard.py'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return pd.read_csv(p,keep_default_na=False,na_values=['','nan','NaN'])
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,float) and not np.isfinite(x):return None
 return x
def js(p,x):
 temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(clean(x),indent=2,allow_nan=False),encoding='utf-8');os.replace(temp,p)
from interval_hazard import Basis, IntervalHazard
C=json.loads((O/'FROZEN_CONFIG.json').read_text());SR=json.loads((O/'SUPPORT_RECEIPT.json').read_text())
assert sha(O/'FROZEN_CONFIG.json')==SR['configuration_sha256']
assert sha(O/'frozen_analysis_inputs.csv')==SR['support_inputs_sha256']
assert sha(O/'training_inputs.csv')==SR['training_inputs_sha256']
FULL=read(O/'training_inputs.csv');FULL['fold']=FULL.fold.astype(int)
REFERENCE=read(O/'frozen_analysis_inputs.csv')
assert len(FULL)==1245 and FULL.sid.is_unique and len(REFERENCE)==322
assert int(FULL.target_reference.sum())==322 and set(REFERENCE.sid)==set(FULL.loc[FULL.target_reference,'sid'])
STATE=C['state_variables'];ENV=C['environment_variables']
class LatitudeDesign:
 def __init__(self,model,kind,basins):self.columns=STATE+(ENV if model=='M1_state_joint_environment' else []);self.kind=kind;self.basins=basins
 def fit(self,e):
  x=e[self.columns].to_numpy(float);assert np.isfinite(x).all()
  self.mean=x.mean(0);self.sd=x.std(0);self.sd=np.where(self.sd>0,self.sd,1.)
  self.lat_mean=float(e.abs_lat.mean());self.lat_sd=float(e.abs_lat.std(ddof=0))
  if self.kind=='spline':
   self.basis=Basis().fit(e.abs_lat.to_numpy())
   if len(np.unique(self.basis.knots))!=4:raise RuntimeError('Degenerate prespecified latitude knots')
  return self
 def transform(self,e,latitude=None):
  a=e.abs_lat.to_numpy() if latitude is None else np.full(len(e),latitude)
  lat=self.basis.transform(a) if self.kind=='spline' else ((a-self.lat_mean)/self.lat_sd)[:,None]
  z=(e[self.columns].to_numpy()-self.mean)/self.sd
  d=[e.basin_genesis.eq(b).to_numpy(float) for b in self.basins[1:]]
  return np.column_stack([np.ones(len(e)),lat,z]+d)
 def metadata(self):return dict(columns=self.columns,kind=self.kind,basins=self.basins,mean=self.mean,sd=self.sd,knots=self.basis.knots if self.kind=='spline' else None)
def estimate(e,hem,kind='spline',save_point=False):
 target=C['hemispheres'][hem];grid=np.linspace(target['low_abs_lat'],target['high_abs_lat'],21)
 basins=sorted(FULL[FULL.hemisphere.eq(hem)].basin_genesis.unique())
 values=[];details=[];predictions=[];coefrows=[]
 refbase=REFERENCE[REFERENCE.hemisphere.eq(hem)].reset_index(drop=True);refindex=pd.Series(np.arange(len(refbase)),index=refbase.sid)
 per_sid=np.full((2,len(refbase),3),np.nan)
 for model_index,model in enumerate(C['models']):
  sums=np.zeros(len(grid));natural_sum=0;checkn=0
  for fold in range(5):
   tr=e[e.fold.ne(fold)].copy();te=e[e.fold.eq(fold)&e.target_reference].copy()
   if len(tr)==0 or len(te)==0:raise RuntimeError('Empty inherited SID fold')
   if np.isfinite(tr.U_h.to_numpy()).sum()==0:raise RuntimeError('No training events')
   trans=LatitudeDesign(model,kind,basins).fit(tr);X=trans.transform(tr)
   fit=IntervalHazard().fit(X,tr.L_h.to_numpy()/6,tr.U_h.to_numpy()/6)
   natural=fit.predict(trans.transform(te));p=np.column_stack([fit.predict(trans.transform(te,a)) for a in grid])
   assert np.isfinite(p).all() and (p>=0).all() and (p<=1).all()
   sums+=p.sum(0);natural_sum+=natural.sum();checkn+=len(te)
   ri=refindex.loc[te.sid].to_numpy();per_sid[model_index,ri,0]=p[:,0];per_sid[model_index,ri,1]=p[:,-1];per_sid[model_index,ri,2]=natural
   if save_point:
    q=pd.DataFrame({'sid':te.sid.to_numpy(),'event_id':te.event_id.to_numpy(),'hemisphere':hem,'fold':fold,'model':model,'kind':kind,'risk_low':p[:,0],'risk_high':p[:,-1],'risk_observed_latitude':natural,'difference':p[:,-1]-p[:,0]})
    predictions.append(q)
    details.append(dict(hemisphere=hem,kind=kind,model=model,fold=fold,n_train=len(tr),n_test=len(te),n_training_events=int(np.isfinite(tr.U_h).sum()),transform=trans.metadata(),optimization=fit.optimization,n_interval=fit.n_interval,n_zero_information=fit.n_zero_information))
    z=(3 if kind=='spline' else 1)+1
    for k,col in enumerate(trans.columns):coefrows.append(dict(hemisphere=hem,model=model,kind=kind,fold=fold,variable=col,coefficient_per_unit=fit.coef[z+k]/trans.sd[k],coefficient_per_training_SD=fit.coef[z+k],n_train=len(tr)))
    with (O/'models'/f'{hem}_{kind}_{model}_fold{fold}.pkl').open('wb') as f:pickle.dump(dict(transform=trans,model=fit,train_sid=tr.sid.tolist(),test_sid=te.sid.tolist()),f,protocol=5)
  target_n=int(e.target_reference.sum());assert checkn==target_n
  values.append(dict(model=model,risk_low=sums[0]/target_n,risk_high=sums[-1]/target_n,difference=(sums[-1]-sums[0])/target_n,natural_risk=natural_sum/target_n,grid=grid,risk_curve=sums/target_n,n_training_sid=len(e),n_target_sid=target_n))
 return values,details,predictions,coefrows,per_sid
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--hemisphere',choices=['NH','SH'],required=True);ap.add_argument('--resume',action='store_true');ap.add_argument('--point-only',action='store_true');args=ap.parse_args()
 hem=args.hemisphere;e=FULL[FULL.hemisphere.eq(hem)].reset_index(drop=True)
 (O/'models').mkdir(exist_ok=True);start=time.monotonic()
 orig=read(U/'recovery_refit/cohort_recovered.csv');bank=np.load(U/'recovery_refit/bootstrap_indices.npz')['indices']
 assert bank.shape==(1000,2231) and np.array_equal(bank,np.random.default_rng(610102026).integers(0,2231,size=(1000,2231),dtype=np.int32))
 index=pd.Series(np.arange(len(e)),index=e.sid);mapping=orig.sid.map(index).fillna(-1).astype(int).to_numpy()
 pointfile=O/f'{hem}_point.json';pred=[];diags=[];coefs=[]
 if args.resume and pointfile.exists():points=json.loads(pointfile.read_text())
 else:
  points={}
  for kind in ['spline','linear']:
   v,d,p,c,_=estimate(e,hem,kind,True);points[kind]=v;diags+=d;pred+=p;coefs+=c
  js(pointfile,points);js(O/f'{hem}_point_fit_diagnostics.json',diags)
  pd.concat(pred,ignore_index=True).to_csv(O/f'{hem}_point_predictions.csv',index=False)
  pd.DataFrame(coefs).to_csv(O/f'{hem}_coefficient_diagnostics.csv',index=False)
  js(O/f'{hem}_run_identity.json',dict(config_sha256=sha(O/'FROZEN_CONFIG.json'),input_sha256=sha(O/'frozen_analysis_inputs.csv'),training_input_sha256=sha(O/'training_inputs.csv'),code_sha256=sha(__file__),accepted_source_sha256=sha(SOURCE),original_bank_sha256=sha(U/'recovery_refit/bootstrap_indices.npz'),bank='exact original1000x2231 SID draws filtered to training1245; target322subset retains identical SIDmultiplicity; paired all models',numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,python=sys.version,executable=sys.executable))
 print(hem,'POINT_COMPLETE',json.dumps(clean(points)),flush=True)
 if args.point_only:return
 out=O/f'{hem}_bootstrap_draws.jsonl';done=set()
 refbase=REFERENCE[REFERENCE.hemisphere.eq(hem)].reset_index(drop=True)
 refmap=orig.sid.map(pd.Series(np.arange(len(refbase)),index=refbase.sid)).fillna(-1).astype(int).to_numpy()
 weights=np.zeros((1000,len(refbase)),dtype=np.int16)
 for b,ix in enumerate(bank):
  z=refmap[ix];z=z[z>=0];weights[b]=np.bincount(z,minlength=len(refbase))
 np.savez_compressed(O/f'{hem}_bootstrap_reference_weights.npz',weights=weights,sid=refbase.sid.to_numpy(str),targets=np.array(['low','high','natural']))
 predpath=O/f'{hem}_bootstrap_per_sid_predictions.npy'
 prediction_bank=np.lib.format.open_memmap(predpath,mode='r+' if args.resume and predpath.exists() else 'w+',dtype=np.float64,shape=(1000,2,2,len(refbase),3))
 if out.exists():
  if not args.resume:raise RuntimeError('Draw log already exists; use explicit resume')
  for line in out.read_text().splitlines():done.add(json.loads(line)['draw'])
 for b,original_ix in enumerate(bank):
  if b in done:continue
  selected=mapping[original_ix];selected=selected[selected>=0];sample=e.iloc[selected].copy()
  result=dict(draw=b,hemisphere=hem,n_draw=len(sample),n_target_draw=int(sample.target_reference.sum()),n_unique_sid=sample.sid.nunique(),n_unique_target_sid=sample.loc[sample.target_reference,'sid'].nunique())
  try:
   result['values']={}
   for ki,kind in enumerate(['spline','linear']):
    vv,_,_,_,pp=estimate(sample,hem,kind);result['values'][kind]=vv;prediction_bank[b,ki]=pp
    for mi in range(2):
     reconstructed=np.sum(np.nan_to_num(pp[mi])*weights[b,:,None],axis=0)/weights[b].sum()
     expected=np.array([vv[mi]['risk_low'],vv[mi]['risk_high'],vv[mi]['natural_risk']])
     assert np.allclose(reconstructed,expected,rtol=0,atol=1e-12)
   result['status']='OK'
  except Exception:result.update(status='FAILED',error=traceback.format_exc())
  with out.open('a',encoding='utf-8') as f:f.write(json.dumps(clean(result),allow_nan=False)+'\n');f.flush()
  if (b+1)%25==0:
   prediction_bank.flush()
   js(O/f'{hem}_CHECKPOINT.json',dict(status='BOOTSTRAP_RUNNING',draws_completed=b+1,draws_planned=1000,elapsed_s=time.monotonic()-start));print(hem,'DRAW',b+1,'seconds',round(time.monotonic()-start,1),flush=True)
 prediction_bank.flush()
 js(O/f'{hem}_CHECKPOINT.json',dict(status='COMPUTED_PENDING_INDEPENDENT_REVIEW',draws_completed=1000,draws_planned=1000,elapsed_s=time.monotonic()-start));print(hem,'COMPLETE',flush=True)
if __name__=='__main__':main()
