"""Scientific functions copied without computational changes from the frozen implementation."""
import numpy as np
import pandas as pd


def interval(x):
    q=x[np.isfinite(x)]
    return [float(v) for v in np.percentile(q,[2.5,97.5])] if len(q) else [None,None]

def stats(sids,X,y):
    unique,ix=np.unique(sids,return_inverse=True);n=np.bincount(ix);w=1/n[ix]
    scales=np.sqrt(np.average(X*X,axis=0,weights=w));keep=scales>0
    # Only identicallyzero columns are omitted. Deterministic scaling is reversible.
    Z=X[:,keep]/scales[keep];p=Z.shape[1]
    G=np.zeros((len(unique),p,p));b=np.zeros((len(unique),p));yy=np.zeros(len(unique));sy=np.zeros(len(unique))
    np.add.at(G,ix,w[:,None,None]*Z[:,:,None]*Z[:,None,:]);np.add.at(b,ix,w[:,None]*Z*y[:,None]);np.add.at(yy,ix,w*y*y);np.add.at(sy,ix,w*y)
    a=int(np.flatnonzero(np.flatnonzero(keep)==1)[0]);return {'sid':unique,'G':G,'b':b,'yy':yy,'sy':sy,'scales':scales,'keep':keep,'a':a,'X':X,'y':y,'w':w}

def fit(g,b,yy,sy,n,a,ascale):
    eig,U=np.linalg.eigh(g);tol=max(float(eig.max()),1)*1e-12;valid=eig>tol
    c=np.zeros(len(b));c[a]=1;error=np.linalg.norm(c-U[:,valid]@(U[:,valid].T@c))
    if error>1e-8:return np.nan,np.nan,np.nan,int(valid.sum()),'target_not_estimable'
    beta=U[:,valid]@((U[:,valid].T@b)/eig[valid]);sse=yy-float(beta@b);sst=yy-sy*sy/n
    other=np.arange(len(b))!=a;rg=g[np.ix_(other,other)];rb=b[other];re,ru=np.linalg.eigh(rg);rv=re>max(float(re.max()),1)*1e-12
    betar=ru[:,rv]@((ru[:,rv].T@rb)/re[rv]);red=yy-float(betar@rb)
    if sst<=0 or not np.isfinite(beta).all() or sse< -1e-7*max(yy,1):return np.nan,np.nan,np.nan,int(valid.sum()),'numerical_or_zero_outcome_variance'
    inc=(red-sse)/sst;part=(red-sse)/red if red>0 else np.nan
    return float(beta[a]/ascale*.1),float(inc),float(part),int(valid.sum()),'estimable_rank_deficient' if valid.sum()<len(b) else 'estimable_full_rank'

def slopefit(st,draws=None):
    gs=st['G'].sum(0);bs=st['b'].sum(0);ascale=st['scales'][1];args=(float(st['yy'].sum()),float(st['sy'].sum()),len(st['sid']),st['a'],ascale)
    result=fit(gs,bs,*args)
    X=st['X'];y=st['y'];w=st['w'];scale=st['scales'];keep=st['keep'];Z=X[:,keep]/scale[keep];beta=np.linalg.lstsq(Z*np.sqrt(w[:,None]),y*np.sqrt(w),rcond=1e-12)[0]
    direct=float(beta[st['a']]/ascale*.1); assert np.isclose(result[0],direct,rtol=1e-8,atol=1e-9),(result,direct)
    if draws is None:return result,None,None
    # Per-SID means retain equal total SID weight under resampling; no row bootstrap.
    count=np.zeros((len(draws),len(st['sid'])),dtype=float)
    for k,ix in enumerate(draws):count[k]=np.bincount(ix,minlength=len(st['sid']))
    p=st['G'].shape[1];packed=np.column_stack([st['G'].reshape(len(st['sid']),-1),st['b'],st['yy'],st['sy']]);weighted=count@packed
    out=[];status=[]
    for k,r in enumerate(weighted):
        values=fit(r[:p*p].reshape(p,p),r[p*p:p*p+p],r[-2],r[-1],len(st['sid']),st['a'],ascale)
        out.append(values[:3]);status.append({'draw':k,'rank':values[3],'status':values[4]})
    return result,np.asarray(out),status
