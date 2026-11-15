"""Exact pure-function excerpts; see adjusted_helper_sources.json."""
import numpy as np
import pandas as pd
from scipy.optimize import linprog
SEED=650117
REPS=2000

def bh_fixed_family(p,nominal_m=None):
 """Unavailable target occupies its prespecified family slot; never reduces m."""
 p=np.asarray(p,float);m=len(p) if nominal_m is None else int(nominal_m)
 if m<len(p):raise ValueError("family smaller than supplied prespecified targets")
 if np.any(np.isfinite(p)&((p<0)|(p>1))):raise ValueError("invalid P")
 vals=np.ones(m);vals[:len(p)]=np.where(np.isfinite(p),p,1.)
 order=np.argsort(vals,kind="stable")
 qsort=np.minimum(1.,np.minimum.accumulate((vals[order]*m/np.arange(1,m+1))[::-1])[::-1])
 q=np.empty(m);q[order]=qsort
 q=q[:len(p)];q[~np.isfinite(p)]=np.nan
 return q

def phase_support_fit(W,phase,row_weights,max_iter=100):
 """Unpenalized logit of phase on fixed W, outcome-blind; no regularized fallback."""
 from scipy.special import expit
 W=np.asarray(W,float);y=np.asarray(phase,float);w=np.asarray(row_weights,float)
 if not (np.isfinite(W).all() and np.isfinite(y).all() and np.isfinite(w).all()):
  raise ValueError("phase support requires finite background")
 if np.any((y!=0)&(y!=1)) or np.any(w<=0):raise ValueError("invalid phase/weights")
 scale=np.sqrt(np.average(W*W,axis=0,weights=w));scale=np.where(scale>0,scale,1.)
 Z=W/scale
 u,s,vt=np.linalg.svd(np.sqrt(w)[:,None]*Z,full_matrices=False)
 tol=np.finfo(float).eps*max(Z.shape)*s[0];rank=int((s>tol).sum())
 V=vt[:rank].T;D=Z@V;b=np.zeros(rank)
 def loss(beta):
  eta=D@beta
  return float(np.sum(w*(np.logaddexp(0,eta)-y*eta)))
 converged=False
 for it in range(max_iter):
  prob=expit(D@b);grad=D.T@(w*(prob-y))
  hess=D.T@((w*prob*(1-prob))[:,None]*D)
  step=np.linalg.lstsq(hess,grad,rcond=None)[0]
  current=loss(b);alpha=1.
  while alpha>2**-25 and loss(b-alpha*step)>current:
   alpha*=.5
  bn=b-alpha*step
  if np.max(np.abs(D@(bn-b)))<1e-9 and np.max(np.abs(grad))/w.sum()<1e-9:
   b=bn;converged=True;break
  b=bn
  if np.max(np.abs(D@b))>40:
   break
 prob=expit(D@b)
 return dict(probability=prob,beta=(V@b)/scale,rank=rank,iterations=it+1,
   converged=converged,status="CONVERGED" if converged else "NO_FINITE_CONVERGED_UNPENALIZED_LOGIT",
   log_likelihood=-loss(b),max_absolute_linear_predictor=float(np.max(np.abs(D@b))),
   column_scale=scale)

def joint_support_diagnostics(background,target,split,categories=None):
 """Outcome-blind convex-hull intersection witness, affine support and leverage.

This diagnostic does not trim/rematch SID or create a new significance family.
A feasible intersection does not establish causal positivity or full common support.
"""
 B=np.asarray(background,float);a=np.asarray(target,float)
 if B.ndim==1:B=B[:,None]
 good=np.isfinite(B).all(axis=1)&np.isfinite(a)
 B=B[good];a=a[good]
 cat=np.full(len(a),"all") if categories is None else np.asarray(categories,str)[good]
 results=[]
 for name in sorted(set(cat)):
  sel=cat==name;Z=B[sel];av=a[sel]
  left=av<split;right=av>=split
  item=dict(stratum=name,n_left=int(left.sum()),n_right=int(right.sum()),split=float(split))
  if not left.any() or not right.any():
   item.update(status="NO_BOTH_SIDES",intersection_witness=None);results.append(item);continue
  mu=Z.mean(axis=0);scale=Z.std(axis=0);scale=np.where(scale>0,scale,1.)
  Z=(Z-mu)/scale
  # Convex combinations of actual rows; no independently recombined marginal ranges.
  L=Z[left];R=Z[right]
  Aeq=np.vstack([np.column_stack([L.T,-R.T]),
   np.r_[np.ones(len(L)),np.zeros(len(R))],np.r_[np.zeros(len(L)),np.ones(len(R))]])
  beq=np.r_[np.zeros(Z.shape[1]),1.,1.]
  lp=linprog(np.zeros(len(L)+len(R)),A_eq=Aeq,b_eq=beq,bounds=(0,None),method="highs")
  item.update(status="CONVEX_HULLS_INTERSECT" if lp.success else "NO_JOINT_HULL_INTERSECTION",
    solver_status=int(lp.status),solver_message=str(lp.message),
    intersection_witness=(lp.x[:len(L)]@L*scale+mu).tolist() if lp.success else None,
    marginal_overlap_low=np.maximum(B[sel][left].min(axis=0),B[sel][right].min(axis=0)).tolist(),
    marginal_overlap_high=np.minimum(B[sel][left].max(axis=0),B[sel][right].max(axis=0)).tolist())
  D=np.column_stack([np.ones(len(Z)),av,Z]);u,s,vt=np.linalg.svd(D,full_matrices=False)
  rank=int(np.sum(s>np.finfo(float).eps*max(D.shape)*s[0]))
  item.update(design_rank=rank,design_columns=D.shape[1],
    max_observed_leverage=float(np.sum(u[:,:rank]**2,axis=1).max()),
    target_range=[float(av.min()),float(av.max())])
  results.append(item)
 return dict(n_complete_background=int(good.sum()),n_missing_background=int((~good).sum()),
  split_used_for_diagnostic_only=float(split),strata=results,
  qualification="intersection is only a joint-support witness, not proof of broad positivity; no population trimming")

def target_estimable(X,contrast,weights=None):
    """Scale columns before rank decisions; a redundant nuisance does not by itself invalidate target."""
    X=np.asarray(X,float);c=np.asarray(contrast,float)
    if weights is not None:X=np.sqrt(np.asarray(weights,float))[:,None]*X
    scale=np.linalg.norm(X,axis=0);scale=np.where(scale>0,scale,1.)
    Z=X/scale
    u,s,vt=np.linalg.svd(Z,full_matrices=False)
    tol=np.finfo(float).eps*max(Z.shape)*(s[0] if len(s) else 0.)
    rank=int(np.sum(s>tol))
    target=c/scale
    projected=vt[:rank].T@(vt[:rank]@target) if rank else np.zeros_like(target)
    err=np.linalg.norm(target-projected)
    return dict(estimable=bool(err<=64*np.finfo(float).eps*max(Z.shape)*max(1.,np.linalg.norm(target))),rank=rank,projection_error=float(err),singular_tolerance=float(tol),scaled_singular_values=s)

def weighted_fit(X,y,w=None,contrast=None):
    X=np.asarray(X,float);y=np.asarray(y,float);w=np.ones(len(y)) if w is None else np.asarray(w,float)
    good=np.isfinite(X).all(axis=1)&np.isfinite(y)&np.isfinite(w)&(w>0)
    X=X[good];y=y[good];w=w[good]
    if not len(y):raise ValueError("no finite weighted rows")
    scale=np.linalg.norm(np.sqrt(w)[:,None]*X,axis=0);scale=np.where(scale>0,scale,1.)
    Z=X/scale;beta=np.linalg.lstsq(np.sqrt(w)[:,None]*Z,np.sqrt(w)*y,rcond=None)[0]/scale
    test=target_estimable(X,np.zeros(X.shape[1]) if contrast is None else contrast,w)
    return dict(beta=beta,n=len(y),rank=test["rank"],df=len(y)-test["rank"],estimable=test["estimable"],fitted=X@beta)

def equal_weights(sid,event,mult=None):
    """Equal bins within event, events within SID; bootstrap SID multiplicity."""
    df=pd.DataFrame({'sid':sid,'event':event})
    nbin=df.groupby(['sid','event']).event.transform('size').to_numpy()
    nevent=df.groupby('sid').event.transform('nunique').to_numpy()
    weight=1./nbin/nevent
    if mult is not None:weight*=np.array([mult[s] for s in sid])
    return weight

def summarize(estimate,draws):
    a=np.asarray(draws,float);a=a[np.isfinite(a)]
    if not np.isfinite(estimate) or not len(a):return dict(estimate=estimate,lower=np.nan,upper=np.nan,p_centered=np.nan,valid_bootstrap=0,failed_bootstrap=REPS)
    lo,hi=np.quantile(a,[.025,.975])
    return dict(estimate=float(estimate),lower=float(lo),upper=float(hi),p_centered=float((1+np.count_nonzero(np.abs(a-estimate)>=abs(estimate)))/(len(a)+1)),valid_bootstrap=len(a),failed_bootstrap=REPS-len(a))

def raw_calc(sid,event,a,b):
    valid=np.isfinite(a)&np.isfinite(b)
    df=pd.DataFrame({'sid':sid[valid],'event':event[valid],'rise':b[valid],'decay':a[valid]})
    ev=df.groupby(['sid','event'])[['rise','decay']].mean();by=ev.groupby('sid').mean().sort_index()
    delta=(by.rise-by.decay).to_numpy();ix=np.random.default_rng(SEED).integers(0,len(by),size=(REPS,len(by)))
    draw=delta[ix].mean(axis=1)
    return dict(**summarize(delta.mean(),draw),n_sid=len(by),n_events=len(ev),n_bin_pairs=int(valid.sum()),rise_mean=float(by.rise.mean()),decay_mean=float(by.decay.mean()),sid_ESS=float(len(by)),row_ESS=float((equal_weights(sid[valid],event[valid]).sum())**2/np.sum(equal_weights(sid[valid],event[valid])**2))),draw

def adaptive(sid,ev,cats,A,C,y0,y1,mult,do_joint=True):
    positive=np.array([mult[s]>0 for s in sid]);idx=np.flatnonzero(positive)
    ss=sid[idx];ee=ev[idx];aa=A[idx];cc=C[idx];yy0=y0[idx];yy1=y1[idx];cat=cats[idx]
    base=equal_weights(ss,ee,mult)
    f=phase_support_fit(np.vstack([aa,cc]),np.r_[np.zeros(len(ss)),np.ones(len(ss))],np.r_[base/2,base/2])
    info=dict(logit_converged=f['converged'],logit_status=f['status'],iterations=f['iterations'])
    if not f['converged']:return dict(success=False,reason='SUPPORT_LOGIT_NO_FINITE_MLE',**info)
    p0,p1=np.split(f['probability'],2);keep=(p0>=.1)&(p0<=.9)&(p1>=.1)&(p1<=.9)
    unsupported=[];diagnostic=[]
    if do_joint and keep.any():
        d=joint_support_diagnostics(np.vstack([aa[keep,1:],cc[keep,1:]]),np.r_[np.zeros(keep.sum()),np.ones(keep.sum())],.5,np.r_[cat[keep],cat[keep]])
        diagnostic=d['strata'];unsupported=[x['stratum'] for x in diagnostic if x['status']!='CONVEX_HULLS_INTERSECT']
        keep &= ~np.isin(cat,unsupported)
    info.update(n_sid_support=len(set(ss[keep])),n_events_support=len(set(ee[keep])),n_bin_pairs_support=int(keep.sum()),supported_basins=sorted(set(cat[keep])),unsupported_basins=unsupported)
    if len(set(ss[keep]))<2:return dict(success=False,reason='NO_SUPPORTED_BASIN_PAIR_TARGET',**info)
    sw=equal_weights(ss[keep],ee[keep],mult);xx=np.column_stack([np.r_[np.zeros(keep.sum()),np.ones(keep.sum())],np.vstack([aa[keep],cc[keep]])])
    contrast=np.zeros(xx.shape[1]);contrast[0]=1.;weight=np.r_[sw/2,sw/2]
    raw=np.average(yy1[keep]-yy0[keep],axis=0,weights=sw)
    estimable=target_estimable(xx,contrast,weight)
    info.update(raw_same_support=raw,original_indices=idx[keep],pair_weights=sw,p0=p0,p1=p1,positive_indices=idx,joint_support=diagnostic,rank=estimable['rank'])
    if not estimable['estimable']:return dict(success=False,reason='PHASE_TARGET_NOT_ESTIMABLE',**info)
    # Same scaled weighted least squares as original weighted_fit, simultaneously
    # evaluating outcome columns; full fit is repeated each SID draw.
    scale=np.linalg.norm(np.sqrt(weight)[:,None]*xx,axis=0);scale=np.where(scale>0,scale,1.)
    coef=np.linalg.lstsq(np.sqrt(weight)[:,None]*(xx/scale),np.sqrt(weight)[:,None]*np.vstack([yy0[keep],yy1[keep]]),rcond=None)[0]/scale[:,None]
    estimate=coef[0]
    # Numerical agreement with original one-outcome helper on first point.
    if not np.allclose(weighted_fit(xx,np.r_[yy0[keep,0],yy1[keep,0]],weight,contrast)['beta'][0],estimate[0],rtol=1e-10,atol=1e-12):raise AssertionError('Multivariate evaluation changed original fit')
    return dict(success=True,reason='',estimate=estimate,**info)

def balance(A,C,w):
    ma=np.average(A,axis=0,weights=w);mc=np.average(C,axis=0,weights=w)
    va=np.average((A-ma)**2,axis=0,weights=w);vc=np.average((C-mc)**2,axis=0,weights=w)
    sd=np.sqrt((va+vc)/2);delta=mc-ma
    smd=np.divide(delta,sd,out=np.full(delta.shape,np.nan),where=sd>0);smd[(sd==0)&(delta==0)]=0
    return ma,mc,smd
