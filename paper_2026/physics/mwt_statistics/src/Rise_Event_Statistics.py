def Rise_Event_Statistics(values, roles, pairs, events, intervals, bootstrap_B=2000,
                          seed=650117, pre_required_count=3, unit='',
                          integrate_rate=False):
    """Summarize one diagnostic on actual requests; return SID-based contrasts.

    values: request_id, value. Other inputs are the frozen request-role, matching,
    return-event and legal-interval tables. Missing observations remain missing.
    """
    import numpy as np
    import pandas as pd

    values=values[['request_id','value']].copy()
    if values.request_id.duplicated().any():raise ValueError('Duplicate diagnostic request')
    values['value']=pd.to_numeric(values.value,errors='coerce')
    values.loc[~np.isfinite(values.value),'value']=np.nan
    r=roles.copy();r['UTC']=pd.to_datetime(r.UTC,utc=True)
    r=r.merge(values,on='request_id',how='left',validate='many_to_one')
    event=events[events.ledger_eligible].copy()
    for key in ['valley_UTC','return_UTC','resolution_UTC']:event[key]=pd.to_datetime(event[key],utc=True)
    if event.event_id.duplicated().any():raise ValueError('Duplicate event ID')
    ordinal=event.set_index('event_id').return_order
    legal=intervals.copy()
    for key in ['UTC','interval_end']:legal[key]=pd.to_datetime(legal[key],utc=True)
    legal=legal[legal.valid_6h & legal.valid_h.eq(6)]
    rate_factor={'W kg-1':(1.,'J kg-1'),'W m-2':(1.,'J m-2'),
                 'g kg-1 day-1':(1./86400.,'g kg-1')}
    if integrate_rate and unit not in rate_factor:raise ValueError('Undefined rate integral units: '+unit)
    event_rows=[];coverage=[];composite_rows=[];bin_rows=[]

    def phase(nodes,interval_rows):
        nodes=nodes[['UTC','value']].drop_duplicates('UTC').set_index('UTC').value
        sec=interval_rows.valid_h.to_numpy(float)*3600.
        left=nodes.reindex(interval_rows.UTC).to_numpy(float)
        right=nodes.reindex(interval_rows.interval_end).to_numpy(float)
        good=np.isfinite(left)&np.isfinite(right)
        complete=bool(len(sec)>0 and good.all())
        integral=float(np.sum(.5*(left+right)*sec)) if complete else np.nan
        return {'mean':integral/float(sec.sum()) if complete else np.nan,
                'duration_h':float(sec.sum()/3600),'complete':complete,
                'n_intervals':len(sec),'n_valid_intervals':int(good.sum()),
                'rate_integral':integral*rate_factor[unit][0] if complete and integrate_rate else np.nan}

    resolution={}
    for sid,nodes in r[r.role.eq('whole_Resolution')].groupby('sid',sort=True):
        start=event.loc[event.sid.eq(sid),'resolution_UTC'].unique()
        if len(start)!=1:raise ValueError('More than one frozen Resolution anchor')
        selected=legal[legal.sid.eq(sid)&legal.UTC.ge(start[0])&legal.UTC.isin(nodes.UTC)]
        resolution[sid]=phase(nodes,selected)

    for e in event.itertuples(index=False):
        nodes=r[r.event_id.eq(e.event_id)]
        tr=nodes[nodes.role.eq('trough_composite')].copy()
        if tr.relative_hour.duplicated().any():raise ValueError('Duplicate relative time')
        pre=tr[tr.relative_hour.isin([-24,-18,-12,-6])]
        npre=int(pre.value.notna().sum());premean=float(pre.value.mean()) if npre>=pre_required_count else np.nan
        ts=tr.set_index('relative_hour').value
        prechange=float(ts.get(0,np.nan)-ts.get(-24,np.nan))
        for row in tr.itertuples(index=False):
            composite_rows.append({'sid':e.sid,'event_id':e.event_id,'relative_hour':row.relative_hour,
                                  'within_rise':bool(row.within_rise),'value':row.value})
        rise=nodes[nodes.role.eq('actual_rise')]
        selected=legal[legal.sid.eq(e.sid)&legal.UTC.ge(e.valley_UTC)&legal.interval_end.le(e.return_UTC)]
        rp=phase(rise,selected)
        expected=(e.return_UTC-e.valley_UTC).total_seconds()/3600
        if rp['duration_h']!=expected:raise ValueError('Rise has unsupported legal duration')
        state=rise.drop_duplicates('UTC').set_index('UTC').value
        change=float(state.get(e.return_UTC,np.nan)-state.get(e.valley_UTC,np.nan))
        res=resolution.get(e.sid,{'mean':np.nan,'duration_h':np.nan,'complete':False})
        rows=[('pre_mean',premean,unit),('pre_change',prechange,unit),('rise_mean',rp['mean'],unit),
              ('rise_endpoint_change',change,unit),('Resolution_mean',res['mean'],unit),
              ('raw_rise_minus_Resolution',rp['mean']-res['mean'],unit)]
        if integrate_rate:rows.append(('rise_rate_integral',rp['rate_integral'],rate_factor[unit][1]))
        for estimand,value,outunit in rows:
            event_rows.append({'sid':e.sid,'event_id':e.event_id,'return_order':e.return_order,
                               'estimand':estimand,'value':value,'units':outunit})
        coverage.append({'sid':e.sid,'event_id':e.event_id,'n_pre_valid':npre,
                         'rise_complete':rp['complete'],'rise_duration_h':rp['duration_h'],
                         'rise_n_intervals':rp['n_intervals'],'rise_n_valid_intervals':rp['n_valid_intervals'],
                         'Resolution_complete':res['complete'],'Resolution_duration_h':res['duration_h']})

    mr=r[r.role.eq('matched_rise')][['sid','event_id','pair_id','wind_bin','value']]
    rr=r[r.role.eq('matched_Resolution')][['sid','event_id','pair_id','wind_bin','value']]
    matched=mr.merge(rr,on=['sid','event_id','pair_id','wind_bin'],how='outer',suffixes=('_rise','_Resolution'),validate='one_to_one')
    assert set(matched.pair_id)==set(pairs.loc[pairs.matched,'pair_id'])
    matched['value']=matched.value_rise-matched.value_Resolution
    matched['joint_valid']=np.isfinite(matched.value)
    for (sid,eid),g in matched.groupby(['sid','event_id'],sort=True):
        good=g[g.joint_valid]
        for label,col in [('matched_rise','value_rise'),('matched_Resolution','value_Resolution'),('matched_rise_minus_Resolution','value')]:
            event_rows.append({'sid':sid,'event_id':eid,'return_order':int(ordinal.loc[eid]),
                               'estimand':label,'value':float(good[col].mean()) if len(good) else np.nan,
                               'units':unit,'n_joint_bins':len(good)})

    event_results=pd.DataFrame(event_rows)
    summaries=[];sid_rows=[];draws={};banks={}
    def bootstrap(series,key):
        series=series.dropna().sort_index();a=series.to_numpy(float);n=len(a)
        if not n:return {'estimate':np.nan,'lower':np.nan,'upper':np.nan,'n_sid':0}
        if n not in banks:banks[n]=np.random.default_rng(seed).integers(0,n,size=(bootstrap_B,n))
        d=a[banks[n]].mean(1);draws[key]=d
        lo,hi=np.quantile(d,[.025,.975])
        return {'estimate':float(a.mean()),'lower':float(lo),'upper':float(hi),'n_sid':n}

    for estimand,g in event_results.groupby('estimand',sort=True):
        # Equally average diagnostic-valid events within SID, then equal SID.
        valid=g[np.isfinite(g.value)]
        sid=valid.groupby('sid').value.mean()
        for s,val in sid.items():sid_rows.append({'sid':s,'estimand':estimand,'value':val,'units':g.units.iloc[0]})
        summaries.append({'estimand':estimand,'sample':'all_events','units':g.units.iloc[0],
                          'n_events':len(valid),**bootstrap(sid,estimand+'|all_events')})
        first=valid[valid.return_order.eq(1)].set_index('sid').value
        if first.index.duplicated().any():raise ValueError('Duplicate first legitimate event')
        summaries.append({'estimand':estimand,'sample':'first_event','units':g.units.iloc[0],
                          'n_events':len(first),**bootstrap(first,estimand+'|first_event')})
        common=sid.index.intersection(first.index)
        summaries.append({'estimand':estimand,'sample':'all_events_on_first_event_SID','units':g.units.iloc[0],
                          'n_events':int(valid.sid.isin(common).sum()),**bootstrap(sid.reindex(common),estimand+'|all_common')})
        summaries.append({'estimand':estimand,'sample':'all_minus_first_on_common_SID','units':g.units.iloc[0],
                          'n_events':int(valid.sid.isin(common).sum()),**bootstrap(sid.reindex(common)-first.reindex(common),estimand+'|difference')})

    composite=pd.DataFrame(composite_rows);time_summary=[];time_sid=[]
    for lag,g in composite.groupby('relative_hour',sort=True):
        good=g[np.isfinite(g.value)]
        sid=good.groupby('sid').value.mean()
        for s,val in sid.items():time_sid.append({'sid':s,'relative_hour':lag,'value':val})
        time_summary.append({'relative_hour':lag,'units':unit,'n_events':len(good),
                             'n_events_within_rise':int(good.within_rise.sum()),
                             'n_sid_within_rise':int(good.loc[good.within_rise,'sid'].nunique()),
                             **bootstrap(sid,'time|'+str(lag))})
    return {'summary':pd.DataFrame(summaries),'event_estimates':event_results,
            'sid_estimates':pd.DataFrame(sid_rows),'coverage':pd.DataFrame(coverage),
            'matched_bins':matched,'time_summary':pd.DataFrame(time_summary),
            'time_sid':pd.DataFrame(time_sid),'bootstrap_draws':draws}
