"""Scientific functions copied without computational changes from the frozen implementation."""
import numpy as np
import pandas as pd
from scipy import stats


def aggregate_stage(d,var,controls,history=False):
    """Left-endpoint legal hours, each variable/model has its own exact mask."""
    valid=np.isfinite(d[var+'_value']) & d[var+'_qc'].astype(bool)
    valid &= (d[var+'_valid_fraction']>=.9) & (d.interval_hours_v3>0)
    for c in controls:valid &= np.isfinite(d[c])
    if history:valid &= d.past24_complete.astype(bool)
    a=d.loc[valid].copy(); a['weighted_y']=a[var+'_value']*a.interval_hours_v3
    for c in controls:a['weighted_'+c]=a[c]*a.interval_hours_v3
    keys=['sid','stage']; g=a.groupby(keys,sort=True,observed=True)
    r=g.agg(covered_h=('interval_hours_v3','sum'),n_valid_intervals=('interval_hours_v3','size'),
        ysum=('weighted_y','sum'),basin_genesis=('basin_genesis','first'),period=('period','first'))
    r['estimate']=r.pop('ysum')/r.covered_h
    for c in controls:r[c]=g['weighted_'+c].sum()/r.covered_h
    r['legal_h']=d.groupby(keys,observed=True).interval_hours_v3.sum().reindex(r.index)
    r['coverage_fraction']=r.covered_h/r.legal_h
    r['source_row_ids']=g.row_id.agg(lambda s:';'.join(s.astype(str)))
    return r.reset_index()

def make_pair(a,before,after,controls):
    keep=['sid','estimate','covered_h','legal_h','coverage_fraction']+list(controls)
    left=a.loc[a.stage==before,keep+['basin_genesis','period']]
    right=a.loc[a.stage==after,keep]
    p=left.merge(right,on='sid',suffixes=('_before','_after'),validate='one_to_one')
    p['delta']=p.estimate_after-p.estimate_before
    return p.sort_values('sid').reset_index(drop=True)

def bh_adjust(p):
    p=np.asarray(p,float);m=len(p);order=np.argsort(np.where(np.isfinite(p),p,np.inf))
    sorted_p=p[order];adj=np.minimum.accumulate((sorted_p*m/np.arange(1,m+1))[::-1])[::-1]
    r=np.empty(m);r[order]=np.minimum(adj,1);return r

def summarize_raw(y,weights):
    y=np.asarray(y,float); n=len(y); denom=weights.sum(axis=1)
    draw=np.divide(weights@y,denom,out=np.full(len(weights),np.nan),where=denom>0)
    sd=np.std(y,ddof=1);se=sd/np.sqrt(n);mean=y.mean()
    t=mean/se if se>0 else (0. if mean==0 else np.sign(mean)*np.inf)
    p=float(2*stats.t.sf(abs(t),n-1)) if n>1 else np.nan
    return dict(estimate=mean,ci_low=np.nanquantile(draw,.025),ci_high=np.nanquantile(draw,.975),
                n_pairs=n,sd_delta=sd,se_delta=se,t_statistic=t,df=n-1,p_value=p,
                bootstrap_attempted=len(draw),bootstrap_valid=int(np.isfinite(draw).sum()),
                bootstrap_failed=int((~np.isfinite(draw)).sum())),draw

def fit_pair(p,controls,weights=None,retain=(),quantile=.05,min_n=30,details=False):
    """Recompute endpoint support and refit for every SID-multiplicity draw.

    Endpoint quantiles are over complete paired SID records before restriction.
    This makes the denominator explicit and prevents a missing opposite stage
    from contributing a covariate value to the target's support threshold.
    """
    n=len(p);w=np.ones(n,dtype=int) if weights is None else np.asarray(weights,int)
    idx=np.repeat(np.arange(n),w); positive=w>0
    good=positive.copy(); ranges=[];exclusions=np.zeros((n,len(controls)),bool)
    result=dict(status='UNSET',estimate=np.nan,n_original_pairs=int(w.sum()),
                n_original_unique=int(positive.sum()),n_pairs=0,n_unique=0,ranges=ranges)
    if not len(idx):result['status']='NO_RESAMPLED_PAIRS';return result
    for j,c in enumerate(controls):
        a=p[c+'_before'].to_numpy(float);b=p[c+'_after'].to_numpy(float)
        qa=np.quantile(a[idx],[quantile,1-quantile]);qb=np.quantile(b[idx],[quantile,1-quantile])
        lo=max(qa[0],qb[0]);hi=min(qa[1],qb[1])
        fail=(a<lo)|(a>hi)|(b<lo)|(b>hi)
        exclusions[:,j]=fail;good &= ~fail
        ranges.append(dict(covariate=c,before_low=qa[0],before_high=qa[1],after_low=qb[0],
            after_high=qb[1],shared_low=lo,shared_high=hi,n_outside=int(w[fail].sum())))
    selected=np.repeat(np.flatnonzero(good),w[good]); k=len(selected)
    result.update(n_pairs=k,n_unique=int(good.sum()),retained_fraction=k/w.sum())
    if details:
        result['selected_mask']=good
        result['exclusion_reasons']=[';'.join(np.array(controls)[row]) for row in exclusions]
    if k==0:result['status']='NO_COMMON_ENDPOINT_SUPPORT';return result
    pp=p.iloc[selected]; y=pp.delta.to_numpy(float)
    result['raw_matched_estimate']=float(y.mean())
    result['before_matched_estimate']=float(pp.estimate_before.mean()) if 'estimate_before' in pp else np.nan
    result['after_matched_estimate']=float(pp.estimate_after.mean()) if 'estimate_after' in pp else np.nan
    if k<min_n or good.sum()<min_n:
        result['status']='FEWER_THAN_30_UNIQUE_SID';return result
    ds=np.column_stack([pp[c+'_after']-pp[c+'_before'] for c in controls])
    zero_supported=[]
    for j,c in enumerate(controls):
        mn=float(ds[:,j].min());mx=float(ds[:,j].max())
        ok=mn<=1e-12 and mx>=-1e-12
        zero_supported.append(ok or c in retain)
        ranges[j].update(delta_min=mn,delta_max=mx,zero_in_delta_range=ok,retained_observed=c in retain)
    if not all(zero_supported):result['status']='ZERO_OUTSIDE_DELTA_SUPPORT';return result
    mu=ds.mean(axis=0);scale=ds.std(axis=0)
    scale=np.where(scale>1e-12,scale,1.)
    z=(ds-mu)/scale
    target=ds.copy()
    for j,c in enumerate(controls):
        if c not in retain:target[:,j]=0.
    target=(target-mu)/scale
    names=['intercept']+['delta_'+c for c in controls]
    x=np.column_stack([np.ones(k),z]); xt=np.column_stack([np.ones(k),target])
    for category in ['basin_genesis','period']:
        for level in sorted(pp[category].unique())[1:]:
            cc=(pp[category].to_numpy()==level).astype(float)
            x=np.column_stack([x,cc]);xt=np.column_stack([xt,cc]);names.append(category+'='+str(level))
    # Target must be identifiable in the full declared model before redundant
    # nuisance columns are dropped in deterministic order.
    tm=xt.mean(axis=0);projection=tm@np.linalg.pinv(x)@x
    target_error=float(np.max(np.abs(tm-projection)))
    result['target_identifiability_error']=target_error
    if target_error>1e-7:result['status']='TARGET_NOT_IDENTIFIABLE';return result
    keep=[0];rank=1
    for j in range(1,x.shape[1]):
        newrank=np.linalg.matrix_rank(x[:,keep+[j]])
        if newrank>rank:keep.append(j);rank=newrank
    xx=x[:,keep];tt=xt[:,keep];beta=np.linalg.lstsq(xx,y,rcond=None)[0]
    pred=tt@beta;fitted=xx@beta
    result.update(status='ESTIMATED',estimate=float(pred.mean()),rank=rank,residual_df=k-rank,
        condition_number=float(np.linalg.cond(xx)),kept_columns=[names[j] for j in keep],
        dropped_columns=[names[j] for j in range(len(names)) if j not in keep],
        residual_sd=float(np.sqrt(np.sum((y-fitted)**2)/(k-rank))),
        max_leverage=float(np.max(np.sum(np.linalg.qr(xx,mode='reduced')[0]**2,axis=1))),
        basin_counts=pp.basin_genesis.value_counts().to_dict(),period_counts=pp.period.value_counts().to_dict())
    if details:
        result['coefficients']=dict(zip(result['kept_columns'],beta.tolist()))
        result['scaling']={c:dict(mean=mu[j],sd=scale[j]) for j,c in enumerate(controls)}
        result['prediction_by_sid']=dict(zip(pp.sid,pred))
    return result
