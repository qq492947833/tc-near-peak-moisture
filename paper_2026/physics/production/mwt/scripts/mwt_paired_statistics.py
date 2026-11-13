def mwt_paired_statistics(rows, events, centres, replicates=2000, seed=650117,
                          pre_hours=(-24,-18,-12,-6), min_common=3, min_area=0.8):
    """SID-paired inference from saved regional diagnostics; no field recalculation.

    rows: long DataFrame with sid,event_class,scheme,window,level,region,lag_h,
    metric,value,valid_area_fraction,units. Returns dataframes and actual bootstrap
    draws. The primary target is a mean temporary-minus-final SID difference.
    Pointwise intervals do not cover systematic diagnostic uncertainty.
    """
    import numpy as np
    import pandas as pd
    import hashlib
    r=rows.copy()
    required=['sid','event_class','scheme','window','level','region','lag_h','metric',
              'value','valid_area_fraction','units']
    if not set(required).issubset(r): raise ValueError('Missing fields: '+str(set(required)-set(r)))
    r['level']=r.level.astype(str);r['sid']=r.sid.astype(str)
    if isinstance(replicates,bool) or int(replicates)!=replicates or replicates<1:raise ValueError('replicates must be positive integer')
    if len(set(pre_hours))!=len(pre_hours) or not 1<=min_common<=len(pre_hours):raise ValueError('Invalid common-offset rule')
    e=events.copy();e['sid']=e.sid.astype(str)
    if e.duplicated(['sid','event_class']).any() or not e.groupby('sid').event_class.apply(lambda x:set(x)=={'internal','final'}).all():raise ValueError('Frozen events must contain exactly one event of each class per SID')
    if not set(r.sid)<=set(e.sid):raise ValueError('Unknown SID outside frozen event population')
    if 'event_id' in r and 'event_id' in e:
        expected=e.set_index(['sid','event_class']).event_id
        identity=pd.MultiIndex.from_frame(r[['sid','event_class']])
        if not np.array_equal(r.event_id.to_numpy(),expected.reindex(identity).to_numpy()):raise ValueError('Frozen event identity mismatch')
    if 't0' in r and 't0' in e:
        expected=pd.to_datetime(e.set_index(['sid','event_class']).t0,utc=True)
        identity=pd.MultiIndex.from_frame(r[['sid','event_class']])
        if not np.array_equal(pd.to_datetime(r.t0,utc=True).to_numpy(),expected.reindex(identity).to_numpy()):raise ValueError('Frozen departure UTC mismatch')
        if 'UTC' in r:
            actual=pd.to_datetime(r.UTC,utc=True)
            should=pd.to_datetime(r.t0,utc=True)+pd.to_timedelta(r.lag_h,unit='h')
            if not actual.equals(should):raise ValueError('Observation UTC/relative-hour mismatch')
    expected_units={k:'W kg-1' for k in ['dKE_dt','dAPE_dt','transport_K_h','transport_A_h','Gamma_K_h','Gamma_A_h','W_PG','C_A_to_K','vertical_input_K_Wkg','vertical_input_A_Wkg','vertical_input_P_Wkg']}
    expected_units.update({k:'J kg-1' for k in ['KE','APE']})
    expected_units.update({k:'W m-2' for k in ['vertical_input_K_Wm2','vertical_input_A_Wm2','vertical_input_P_Wm2']})
    expected_units.update({k:'Pa m2 s-3' for k in ['Q_K_p','Q_A_p','Q_P_p']})
    expected_units.update({k:'m3 s-3' for k in ['Q_K_lon','Q_K_lat','Q_A_lon','Q_A_lat','Q_P_lon','Q_P_lat']})
    for metric,g in r.groupby('metric',sort=False):
        if metric not in expected_units or set(g.units.dropna())!={expected_units[metric]} or g.units.isna().any():raise ValueError('Physical units mismatch for '+str(metric))
    finite_area=np.isfinite(r.valid_area_fraction)
    if ((r.valid_area_fraction[finite_area]<0)|(r.valid_area_fraction[finite_area]>1+1e-12)).any():raise ValueError('Invalid area fraction outside [0,1]')
    if not set(r.event_class.unique())<={'internal','final'}:raise ValueError('Unexpected event class')
    key=['scheme','window','level','region','metric']
    if r.duplicated(['sid','event_class','lag_h']+key).any():raise ValueError('Repeated SID/event observations')
    r.loc[~finite_area|(r.valid_area_fraction<min_area)|~np.isfinite(r.value),'value']=np.nan
    if r.groupby(key,dropna=False).units.nunique().gt(1).any():raise ValueError('Mixed units')
    banks={};draws={};serial=[]
    def estimate(df,label):
        df=df.sort_values('sid')
        v=df.difference.to_numpy(float);nt=len(v)
        identity='|'.join(df.sid.tolist())
        bankkey=hashlib.sha256(identity.encode()).hexdigest()
        if nt==0:return dict(n_sid=0,temporary_mean=np.nan,final_mean=np.nan,estimate=np.nan,lower=np.nan,upper=np.nan,p_centered=np.nan,draw_key='')
        if nt<2:
            return dict(n_sid=nt,temporary_mean=float(df.temporary.mean()),final_mean=float(df.final.mean()),estimate=float(v.mean()),lower=np.nan,upper=np.nan,p_centered=np.nan,draw_key='',interval_status='fewer than two independent SID')
        if bankkey not in banks:
            banks[bankkey]=np.random.default_rng(seed).integers(0,nt,size=(replicates,nt),dtype=np.int32)
        sample=v[banks[bankkey]].mean(axis=1)
        mean=float(v.mean());lo,hi=np.quantile(sample,[.025,.975])
        dkey=hashlib.sha256((label+'|'+identity).encode()).hexdigest()[:24]
        draws[dkey]=sample
        return dict(n_sid=nt,temporary_mean=float(df.temporary.mean()),final_mean=float(df.final.mean()),
                    estimate=mean,lower=float(lo),upper=float(hi),
                    p_centered=float((1+np.count_nonzero(np.abs(sample-mean)>=abs(mean)))/(replicates+1)),
                    draw_key=dkey)
    def paired_at_offsets(frame, minimum):
        wide=frame.pivot(index=['sid','lag_h'],columns='event_class',values='value')
        if not {'internal','final'}<=set(wide):return pd.DataFrame(columns=['sid','temporary','final','difference','n_offsets','offsets'])
        wide=wide.dropna(subset=['internal','final'])
        out=[]
        for sid,g in wide.groupby(level='sid',sort=True):
            if len(g)<minimum:continue
            a=float(g.internal.mean());b=float(g.final.mean())
            out.append(dict(sid=sid,temporary=a,final=b,difference=a-b,n_offsets=len(g),
                            offsets=','.join(map(str,g.index.get_level_values('lag_h').tolist()))))
        return pd.DataFrame(out,columns=['sid','temporary','final','difference','n_offsets','offsets'])
    records=[];pairs=[]
    for levels,g in r.groupby(key,sort=True,dropna=False):
        ident=dict(zip(key,levels));units=g.units.iloc[0]
        for lag,sub in g.groupby('lag_h',sort=True):
            p=paired_at_offsets(sub,1);label=str((levels,'lag',lag))
            st=estimate(p,label);records.append({**ident,'summary':'timecourse','lag_h':lag,'units':units,**st})
        sub=g[g.lag_h.isin(pre_hours)]
        p=paired_at_offsets(sub,min_common)
        if len(p):
            pp=p.assign(**ident,units=units);pairs.append(pp)
        st=estimate(p,str((levels,'pre')))
        records.append({**ident,'summary':'pre','lag_h':np.nan,'units':units,**st})
    effects=pd.DataFrame(records)
    sid_pairs=pd.concat(pairs,ignore_index=True) if pairs else pd.DataFrame(columns=['sid','temporary','final','difference','n_offsets','offsets']+key+['units'])
    primary_metrics=['KE','APE','C_A_to_K','vertical_input_K_Wkg','vertical_input_A_Wkg','vertical_input_P_Wkg']
    sel=(effects.summary=='pre')&(effects.scheme=='main')&(effects.window==1)&(effects.region=='r0_300')
    sel &= ((effects.level=='700')&effects.metric.isin(primary_metrics[:3]))|((effects.level=='500-850')&effects.metric.isin(primary_metrics[3:]))
    primary=effects.loc[sel].copy()
    if len(primary)!=6:raise ValueError('Primary family must contain all six declared outcomes, not a selected subset')
    valid=primary.p_centered.notna()
    ps=primary.loc[valid,'p_centered'];order=np.argsort(ps.to_numpy())
    sortedp=ps.to_numpy()[order];q=np.minimum.accumulate((sortedp*len(primary)/np.arange(1,len(sortedp)+1))[::-1])[::-1]
    primary['q_BH']=np.nan;primary.loc[ps.index[order],'q_BH']=np.minimum(q,1)
    # Matched-SID, matched-offset sensitivity. No subtraction of unrelated group means.
    sensitivity=[];sensitivity_pairs=[]
    base=r[(r.scheme=='main')&(r.region=='r0_300')&r.lag_h.isin(pre_hours)]
    for name,alt in [('input_shift',r[(r.scheme=='shift16d')&(r.region=='r0_300')&r.lag_h.isin(pre_hours)]),
                     ('radius_500',r[(r.scheme=='main')&(r.region=='r0_500')&r.lag_h.isin(pre_hours)])]:
        for levels,g in base.groupby(['window','level','metric'],sort=True):
            ag=alt[(alt.window==levels[0])&(alt.level==levels[1])&(alt.metric==levels[2])]
            a=g.pivot(index=['sid','lag_h'],columns='event_class',values='value').add_prefix('base_')
            b=ag.pivot(index=['sid','lag_h'],columns='event_class',values='value').add_prefix('alt_')
            both=a.join(b,how='inner').dropna()
            if not {'base_internal','base_final','alt_internal','alt_final'}<=set(both):continue
            pp=[]
            for sid,t in both.groupby(level='sid',sort=True):
                if len(t)<min_common:continue
                bv=float((t.base_internal-t.base_final).mean());av=float((t.alt_internal-t.alt_final).mean())
                pp.append(dict(sid=sid,temporary=av,final=bv,difference=av-bv,
                               n_offsets=len(t),offsets=','.join(map(str,t.index.get_level_values('lag_h')))))
            p=pd.DataFrame(pp,columns=['sid','temporary','final','difference','n_offsets','offsets'])
            ident=dict(zip(['window','level','metric'],levels))
            st=estimate(p,str((name,levels)))
            sensitivity.append(dict(sensitivity=name,**ident,units=g.units.iloc[0],**st))
            if len(p):sensitivity_pairs.append(p.assign(sensitivity=name,**ident))
    # Frozen past-24-hour land-contact condition at each event; no future survival rule.
    e=events.copy();e['sid']=e.sid.astype(str)
    sea=e.pivot(index='sid',columns='event_class',values='land_contact')
    seaids=set(sea.index[(sea.internal==0)&(sea.final==0)])
    for levels,g in sid_pairs[(sid_pairs.scheme=='main')&(sid_pairs.region=='r0_300')].groupby(['window','level','metric'],sort=True):
        sub=g[g.sid.isin(seaids)]
        sensitivity.append(dict(sensitivity='both_events_ocean',**dict(zip(['window','level','metric'],levels)),
                                units=g.units.iloc[0],**estimate(sub,str(('ocean',levels)))))
    # Actual baseline translation speeds, retained as a separate estimand.
    c=centres[centres.lag_h.isin(pre_hours)].copy();c['sid']=c.sid.astype(str)
    if c.duplicated(['sid','event_class','lag_h']).any():raise ValueError('Repeated frozen covariate observations')
    if not set(c.sid)<=set(e.sid):raise ValueError('Unknown covariate SID outside frozen events')
    if 'actual_track_point' in c:c=c[c.actual_track_point]
    if 'position_extrapolated' in c:c=c[~c.position_extrapolated]
    c['speed_ms']=np.hypot(c.motion_u_ms,c.motion_v_ms)
    baseline=[]
    for name,column in [('translation_speed_ms','speed_ms'),('wind_kt','wind_kt'),('latitude_deg','lat')]:
        t=c[['sid','event_class','lag_h',column]].rename(columns={column:'value'})
        t.loc[~np.isfinite(t.value),'value']=np.nan
        p=paired_at_offsets(t,min_common)
        if len(p):baseline.append(p.assign(metric=name))
    baseline_pairs=pd.concat(baseline,ignore_index=True) if baseline else pd.DataFrame(columns=['sid','temporary','final','difference','n_offsets','offsets','metric'])
    baseline_summary=[]
    for name,p in baseline_pairs.groupby('metric',sort=True):
        baseline_summary.append(dict(metric=name,**estimate(p,'baseline|'+name)))
    return {'effects':effects,'primary_family':primary,'sid_pre_pairs':sid_pairs,
            'sensitivities':pd.DataFrame(sensitivity),
            'sensitivity_pairs':pd.concat(sensitivity_pairs,ignore_index=True) if sensitivity_pairs else pd.DataFrame(),
            'baseline_pairs':baseline_pairs,'baseline_summary':pd.DataFrame(baseline_summary),
            'bootstrap_draws':draws,'bootstrap_index_banks':banks}

