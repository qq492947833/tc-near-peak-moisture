"""Prespecified SID-resampling statistics for new v5.1 results only."""
import numpy as np
def bh(p, family_size=None):
    p=np.asarray(p,float);out=np.full(p.shape,np.nan);good=np.flatnonzero(np.isfinite(p))
    if not len(good):return out
    m=len(p) if family_size is None else int(family_size)
    if m<len(p):raise ValueError('nominal family size cannot be smaller than supplied family')
    order=good[np.argsort(p[good],kind="stable")]
    out[order]=np.minimum(1,np.minimum.accumulate((p[order]*m/np.arange(1,len(order)+1))[::-1])[::-1])
    return out
def bootstrap_weights(n,draws=2000,seed=510082026):
    if n<=0:raise ValueError("positive SID count required")
    rng=np.random.default_rng(seed);indices=rng.integers(0,n,size=(draws,n))
    weights=np.zeros((draws,n),dtype=np.int32)
    np.add.at(weights,(np.arange(draws)[:,None],indices),1)
    assert np.all(weights.sum(axis=1)==n)
    return indices,weights
def paired_mean_draws(values,weights,minimum_n=2):
    a=np.asarray(values,float)
    if a.ndim==1:a=a[:,None]
    good=np.isfinite(a);n=good.sum(axis=0)
    sums=np.where(good,a,0.)
    est=np.divide(sums.sum(axis=0),n,out=np.full(n.shape,np.nan),where=n>=minimum_n)
    den=np.asarray(weights,float)@good.astype(float)
    draws=np.divide(np.asarray(weights,float)@sums,den,out=np.full(den.shape,np.nan),where=den>=minimum_n)
    draws[:,n<minimum_n]=np.nan
    return est,draws,n
def summarize_draws(estimate,draws):
    d=np.asarray(draws,float);d=d[np.isfinite(d)];n=len(d)
    if not n:return dict(estimate=float(estimate),lo=np.nan,hi=np.nan,p=np.nan,valid_draws=0)
    lo,hi=np.quantile(d,[.025,.975],method="linear")
    p=min(1.,2*min((1+np.sum(d<=0))/(n+1),(1+np.sum(d>=0))/(n+1)))
    return dict(estimate=float(estimate),lo=float(lo),hi=float(hi),p=float(p),valid_draws=n)
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
def self_test():
    assert np.allclose(bh([.01,.04,.03]),[.03,.04,.04])
    assert np.allclose(bh([.01,np.nan,.04])[[0,2]],[.03,.06])
    idx,w=bootstrap_weights(3,7,5);a=np.array([[1,np.nan],[2,4],[3,6]])
    est,d,n=paired_mean_draws(a,w)
    for b in range(7):
        expected=np.nanmean(a[idx[b]],axis=0)
        expected[np.isfinite(a[idx[b]]).sum(axis=0)<2]=np.nan
        assert np.allclose(d[b],expected,equal_nan=True)
    assert np.allclose(est,[2,5]);assert np.array_equal(n,[3,2])
    x=np.arange(8.)
    X=np.column_stack([np.ones(8),x,x*1e8])
    assert target_estimable(X,[1,0,0])["estimable"]
    assert not target_estimable(X,[0,1,0])["estimable"]
    assert target_estimable(X,[0,1,1e8])["estimable"]
    fit=weighted_fit(X,2+3*x,contrast=[1,0,0])
    assert fit["estimable"] and abs(fit["beta"][0]-2)<1e-10 and fit["df"]==6
    return {"checks":"BH; draw-by-draw paired means; NaN support; scaled redundant nuisance and target; residualdf","passed":True}
if __name__=="__main__":print(self_test())
