"""Frozen additive duration contrasts, estimability and interval functions."""
import numpy as np
from scipy import linalg
RANK_TOL=1e-10
EST_TOL=1e-9
def contrasts(X,w,ycode,groups,n_years,reference):
 """Actual annual means/reference replacement, then original weighted linear slope."""
 yw=np.bincount(ycode,weights=w,minlength=n_years);active=yw>0
 annual=np.column_stack([np.bincount(ycode,weights=w*X[:,k],minlength=n_years)for k in range(X.shape[1])])
 annual=annual[active]/yw[active,None];aw=yw[active]/yw.sum();tt=annual[:,1];tc=tt-aw@tt
 h=aw*tc/(aw@(tc*tc));cs=[];rows=[]
 for j in range(4):
  a=annual.copy()
  for group in ['geo','state','env'][:j]:a[:,groups[group]]=reference[groups[group]]
  cs.append(h@a);rows.append(a)
 c=np.stack(cs);tw=w/w.sum();t=X[:,1];fs=((tw*(t-tw@t))@X)/(tw@((t-tw@t)**2))
 alt=np.tile(fs,(4,1))
 for j in range(1,4):
  for group in ['geo','state','env'][:j]:alt[j,groups[group]]=0
 # The intercept is zero in exact arithmetic. The actual derived contrast is retained, not projected.
 err=np.max(abs(c-alt));eye=np.zeros(X.shape[1]);eye[1]=1
 c=np.vstack([c,c[0]-c[1],c[1]-c[2],c[2]-c[3],c[1]-c[3]])
 return c,fs,err,np.max(abs(c[3]-eye)),rows,yw

def solve_targets(X,Y,w,C):
 active=w>0;A=np.sqrt(w[active,None])*X[active];y=np.sqrt(w[active,None])*Y[active]
 scale=np.linalg.norm(A,axis=0);scale=np.where(scale>0,scale,1.0);Z=A/scale
 u,s,vt=linalg.svd(Z,full_matrices=False,check_finite=False,lapack_driver='gesdd')
 rank=int((s>s[0]*RANK_TOL).sum());cs=C/scale;norm=np.linalg.norm(cs,axis=1);den=np.where(norm>1e-14,norm,1.0)
 null=vt[rank:];res=np.linalg.norm(cs@null.T,axis=1)/den;ok=res<=EST_TOL
 # Compute only the requested estimable functionals; arbitrary coefficients are not used for extrapolation.
 val=np.full((len(C),Y.shape[1]),np.nan)
 val[ok]=(cs[ok]@vt[:rank].T)@((u[:,:rank].T@y)/s[:rank,None])
 gamma=vt[:rank].T@((u[:,:rank].T@y)/s[:rank,None])
 equiv=0.0
 if len(null)and ok.any():
  perturb=null.T@np.arange(1,len(null)+1,dtype=float)*max(1,np.linalg.norm(gamma))*10
  equiv=float(np.max(abs(cs[ok]@perturb)))
 stable=True;valuesensitivity=0.0
 for rt in [1e-12,1e-10,1e-8]:
  rr=int((s>s[0]*rt).sum());res2=np.linalg.norm(cs@vt[rr:].T,axis=1)/den
  for et in [1e-11,1e-9,1e-7]:stable &= bool(np.array_equal(res2<=et,ok))
  if rr!=rank and ok.any():
   vv=(cs[ok]@vt[:rr].T)@((u[:,:rr].T@y)/s[:rr,None]);valuesensitivity=max(valuesensitivity,float(np.max(abs(vv-val[ok]))))
 return dict(values=val,ok=ok,residual=res,rank=rank,columns=X.shape[1],condition_nonzero=float(s[0]/s[rank-1]),min_positive_relative=float(s[rank-1]/s[0]),max_null_relative=float(s[rank]/s[0])if rank<len(s)else 0.0,stable=stable,equivalent_solution_error=equiv,tolerance_value_error=valuesensitivity,beta=gamma/scale[:,None],scale=scale,null=null,cs=cs)

def bh(p):
 p=np.asarray(p,float);o=np.argsort(p);q=np.empty(len(p));q[o]=np.minimum.accumulate((p[o]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1];return np.minimum(q,1)

def interval(v):
 v=np.asarray(v,float);v=v[np.isfinite(v)]
 return dict(ci_low=float(np.percentile(v,2.5)),ci_high=float(np.percentile(v,97.5)),p_boot=float(min(1,2*min((1+(v<=0).sum())/(len(v)+1),(1+(v>=0).sum())/(len(v)+1)))),valid_draws=len(v))
