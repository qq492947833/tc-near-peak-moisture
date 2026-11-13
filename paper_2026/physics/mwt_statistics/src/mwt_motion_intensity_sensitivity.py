def mwt_motion_intensity_sensitivity(rows, centres, replicates=2000, seed=650117,
                                     pre_hours=(-24,-18,-12,-6), min_common=3,
                                     min_area=.8, minimum_SID=30):
    """Fixed two-covariate sensitivity of paired energy contrasts.

    Estimate the mean linear paired contrast at zero paired pre-window translation
    speed and wind-intensity differences. This is distinct from the raw paired mean.
    SID/offset support is shared by response and covariates. No bootstrap redraws.
    Returns summary, SID inputs and all attempted bootstrap outcomes.
    """
    import numpy as np
    import pandas as pd
    from scipy.spatial import Delaunay,QhullError
    if isinstance(replicates,bool) or int(replicates)!=replicates or replicates<1:raise ValueError('replicates must be positive integer')
    if len(set(pre_hours))!=len(pre_hours) or not 1<=min_common<=len(pre_hours):raise ValueError('Invalid common-offset rule')
    outcomes=['KE','APE','C_A_to_K','vertical_input_K_Wkg','vertical_input_A_Wkg','vertical_input_P_Wkg']
    r=rows[(rows.scheme=='main')&(rows.window==1)&(rows.region=='r0_300')&rows.lag_h.isin(pre_hours)].copy()
    r['level']=r.level.astype(str);r['sid']=r.sid.astype(str)
    if not set(r.event_class.unique())<={'internal','final'}:raise ValueError('Unexpected event class')
    if r.duplicated(['sid','event_class','lag_h','metric','level']).any():raise ValueError('Repeated SID/event observations')
    finite_area=np.isfinite(r.valid_area_fraction)
    if ((r.valid_area_fraction[finite_area]<0)|(r.valid_area_fraction[finite_area]>1+1e-12)).any():raise ValueError('Invalid area fraction outside [0,1]')
    r=r[((r.level=='700')&r.metric.isin(outcomes[:3]))|((r.level=='500-850')&r.metric.isin(outcomes[3:]))]
    r.loc[~np.isfinite(r.valid_area_fraction)|(r.valid_area_fraction<min_area)|~np.isfinite(r.value),'value']=np.nan
    c=centres[centres.lag_h.isin(pre_hours)].copy();c['sid']=c.sid.astype(str)
    if c.duplicated(['sid','event_class','lag_h']).any():raise ValueError('Repeated frozen covariate observations')
    if not set(r.sid)<=set(c.sid):raise ValueError('Unknown SID outside frozen covariate population')
    if 'event_id' in r and 'event_id' in c:
        identity=c[['sid','event_class','event_id']].drop_duplicates()
        if identity.duplicated(['sid','event_class']).any():raise ValueError('Conflicting frozen event IDs')
        expected=identity.set_index(['sid','event_class']).event_id
        keys=pd.MultiIndex.from_frame(r[['sid','event_class']])
        if not np.array_equal(r.event_id.to_numpy(),expected.reindex(keys).to_numpy()):raise ValueError('Frozen event identity mismatch')
    if 't0' in r and 't0' in c:
        identity=c[['sid','event_class','t0']].drop_duplicates()
        if identity.duplicated(['sid','event_class']).any():raise ValueError('Conflicting frozen departure UTC')
        expected=pd.to_datetime(identity.set_index(['sid','event_class']).t0,utc=True)
        keys=pd.MultiIndex.from_frame(r[['sid','event_class']])
        if not np.array_equal(pd.to_datetime(r.t0,utc=True).to_numpy(),expected.reindex(keys).to_numpy()):raise ValueError('Frozen departure UTC mismatch')
        if 'UTC' in r:
            if not pd.to_datetime(r.UTC,utc=True).equals(pd.to_datetime(r.t0,utc=True)+pd.to_timedelta(r.lag_h,unit='h')):raise ValueError('Observation UTC/relative-hour mismatch')
    if 'actual_track_point' in c:c=c[c.actual_track_point]
    if 'position_extrapolated' in c:c=c[~c.position_extrapolated]
    for metric,g in r.groupby('metric',sort=False):
        unit='J kg-1' if metric in ['KE','APE'] else 'W kg-1'
        if set(g.units.dropna())!={unit} or g.units.isna().any():raise ValueError('Physical units mismatch for '+metric)
    c['speed_ms']=np.hypot(c.motion_u_ms,c.motion_v_ms)
    for col in ['speed_ms','wind_kt']:c.loc[~np.isfinite(c[col]),col]=np.nan
    cov={}
    for col in ['speed_ms','wind_kt']:
        cv=c.pivot(index=['sid','lag_h'],columns='event_class',values=col)
        cov[col]=cv['internal']-cv['final']
    cv=pd.DataFrame(cov)
    def support(x):
        if len(x)<3 or np.linalg.matrix_rank(np.c_[np.ones(len(x)),x])<3:return False,'rank'
        try:
            inside=Delaunay(x).find_simplex(np.zeros((1,2)))[0]>=0
            return bool(inside),'ok' if inside else 'target_outside_hull'
        except QhullError:return False,'hull_degenerate'
    summaries=[];sources=[];draws=[]
    for metric in outcomes:
        g=r[r.metric==metric]
        if len(g)==0:raise ValueError('Declared primary outcome missing: '+metric)
        yy=g.pivot(index=['sid','lag_h'],columns='event_class',values='value')
        z=cv.join((yy.internal-yy.final).rename('energy'),how='inner').dropna()
        aa=[]
        for sid,t in z.groupby(level='sid',sort=True):
            if len(t)>=min_common:
                aa.append(dict(sid=sid,energy=t.energy.mean(),speed_ms=t.speed_ms.mean(),wind_kt=t.wind_kt.mean(),
                               n_offsets=len(t),offsets=','.join(map(str,t.index.get_level_values('lag_h')))))
        a=pd.DataFrame(aa,columns=['sid','energy','speed_ms','wind_kt','n_offsets','offsets']).sort_values('sid')
        x=a[['speed_ms','wind_kt']].to_numpy(float);y=a.energy.to_numpy(float);n=len(y)
        scale=x.std(axis=0,ddof=1) if n>=2 else np.ones(2);scale=np.where(scale>0,scale,1);xs=x/scale
        allowed,reason=support(xs)
        if n<minimum_SID:allowed=False;reason='below_minimum_SID'
        record=dict(metric=metric,n_sid=n,raw_mean=float(y.mean()) if n else np.nan,units=g.units.iloc[0],
                    status='estimated' if allowed else reason,attempts=replicates if allowed else 0)
        sources.append(a.assign(metric=metric))
        if not allowed:
            record.update(estimate=np.nan,lower=np.nan,upper=np.nan,successful=0,failed=0)
            summaries.append(record);continue
        beta=np.linalg.lstsq(np.c_[np.ones(n),xs],y,rcond=None)[0]
        rng=np.random.default_rng(seed);ix=rng.integers(n,size=(replicates,n))
        accepted=[];changes=[];fails={}
        for draw,idx in enumerate(ix):
            ok,why=support(xs[idx])
            if ok:
                bb=np.linalg.lstsq(np.c_[np.ones(n),xs[idx]],y[idx],rcond=None)[0]
                aa0=float(bb[0]);delta=aa0-float(y[idx].mean())
                accepted.append(aa0);changes.append(delta)
                draws.append(dict(metric=metric,draw=draw,status='ok',adjusted=aa0,raw=float(y[idx].mean()),adjustment=delta))
            else:
                fails[why]=fails.get(why,0)+1
                draws.append(dict(metric=metric,draw=draw,status=why,adjusted=np.nan,raw=float(y[idx].mean()),adjustment=np.nan))
        lo,hi=np.quantile(accepted,[.025,.975]) if accepted else (np.nan,np.nan)
        dlo,dhi=np.quantile(changes,[.025,.975]) if changes else (np.nan,np.nan)
        record.update(estimate=float(beta[0]),lower=float(lo),upper=float(hi),
                      speed_slope=float(beta[1]/scale[0]),wind_slope=float(beta[2]/scale[1]),
                      successful=len(accepted),failed=replicates-len(accepted),failure_reasons=str(fails),
                      interval_scope='all fixed attempts' if not fails else 'conditional on successful target-supported refits',
                      adjustment=float(beta[0]-y.mean()),adjustment_lower=float(dlo),adjustment_upper=float(dhi),
                      speed_difference_min=float(x[:,0].min()),speed_difference_max=float(x[:,0].max()),
                      wind_difference_min=float(x[:,1].min()),wind_difference_max=float(x[:,1].max()))
        summaries.append(record)
    return {'summary':pd.DataFrame(summaries),'SID_inputs':pd.concat(sources,ignore_index=True),
            'draws':pd.DataFrame(draws)}

