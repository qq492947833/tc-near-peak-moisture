# Public path adapter; the scientific functions below retain their original implementation.
import sys as _sys
from pathlib import Path as _Path
_PACKAGE=next(p for p in _Path(__file__).resolve().parents if (p/'release_paths.py').is_file())
_sys.path.insert(0,str(_PACKAGE))
from release_paths import ADDENDUM, UNIFIED, PACKAGE, bootstrap
from analyze_duration import *
import statsmodels.api as sm
def gamma_fit(X,y,w,start=None):
 A=X.T@(X*w[:,None])
 if np.linalg.matrix_rank(A,tol=np.linalg.norm(A,2)*1e-12)<X.shape[1]:raise ValueError('rank deficient')
 inv=np.linalg.inv(A);beta=inv@X.T@(w*np.log(y))if start is None else start.copy()
 def objective(b):
  eta=np.clip(X@b,-25,25);return np.sum(w*(y*np.exp(-eta)+eta))
 old=objective(beta)
 for it in range(100):
  eta=np.clip(X@beta,-25,25);step=inv@X.T@(w*(y*np.exp(-eta)-1));alpha=1.
  while alpha>2**-20:
   new=beta+alpha*step;obj=objective(new)
   if obj<=old+1e-11:break
   alpha*=.5
  if np.max(np.abs(alpha*step))<1e-9:return new,it+1
  beta=new;old=obj
 raise ValueError('no convergence')
def fixed_slope(beta,X,w):
 eta=X@beta-X[:,1]*beta[1];ref=(w@np.exp(eta))/w.sum();t=(YEARS-2002.5)/10;pred=ref*np.exp(t*beta[1]);return (t@pred)/(t@t),pred
def main():
 bank=np.load(R/'joint_year_sid_draws.npz');counts=bank['counts'];ny=bank['year_multiplicity'];sid_index={s:i for i,s in enumerate(bank['sid'])};out=[];statuses=[];validation=[]
 for st in STAGES:
  g=read(R/f'common_support_{st}.csv');d=read(R/f'design_{st}.csv');X=d.drop(columns=['sid','weight']).to_numpy(float);w=d.weight.to_numpy();y=g.duration_h.to_numpy(float);beta,it=gamma_fit(X,y,w)
  direct=sm.GLM(y,X,family=sm.families.Gamma(sm.families.links.Log()),freq_weights=w).fit(maxiter=150,tol=1e-11)
  err=np.max(np.abs(beta-direct.params));assert err<1e-6,err
  slope,pred=fixed_slope(beta,X,w);draw=np.full(B,np.nan);idx=np.array([sid_index[s]for s in g.sid]);yc=g.genesis_year.to_numpy(int)-1982
  for b in range(B):
   cc=counts[b,idx].astype(float);ss=np.bincount(yc,weights=cc,minlength=42);wb=cc*np.divide(ny[b,yc],ss[yc],out=np.zeros(len(g)),where=ss[yc]>0)
   try:bb,_=gamma_fit(X,y,wb,beta);draw[b]=fixed_slope(bb,X,wb)[0];status='ok'
   except (ValueError,np.linalg.LinAlgError)as exc:status=str(exc)
   statuses.append(dict(stage=st,draw=b,status=status))
  np.save(R/f'gamma_duration_draws_{st}.npy',draw)
  out.append(dict(stage=st,n_sid=len(g),target='fully_fixed_reference_Gamma_loglink_predict_hours',slope=slope,**interval(draw),unit='h/decade',interval_scope='fixed 2000 attempts; conditional on successful fits',all_individual_predictions_positive=True))
  validation.append(dict(stage=st,statsmodels_coefficient_max_abs_error=float(err),iterations=it,all_standardized_year_means_positive=bool((pred>0).all())))
  print(st,slope,interval(draw),flush=True)
 save(pd.DataFrame(out),'positive_duration_model_sensitivity.csv');save(pd.DataFrame(statuses),'positive_duration_draw_status.csv');(R/'POSITIVE_MODEL_VALIDATION.json').write_text(json.dumps(validation,indent=2),encoding='utf8')
if __name__=='__main__':main()
