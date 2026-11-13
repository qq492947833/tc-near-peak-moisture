def build_rise_comparable_requests(events, ascent, resolution, intervals, tracks,
                                  bin_origin_kt=35., bin_width_kt=10.,
                                  composite_lags=tuple(range(-24,25,6)),
                                  earth_radius_m=6371008.8):
    """Build wind-only matched states and exact-point requests, without field outcomes.

    Required tables are the independently verified anchored return ledger, actual
    ascent/Resolution observations, legal intervals, and complete frozen tracks.
    Return matched states, coverage, role-to-point map, unique exact-point requests.
    """
    import numpy as np
    import pandas as pd
    import hashlib
    e=events.copy(); a=ascent.copy(); r=resolution.copy(); z=intervals.copy(); t=tracks.copy()
    e['sid']=e.sid.astype(str)
    if e.event_id.duplicated().any(): raise ValueError('Duplicate anchored event identity')
    if not e.ledger_eligible.all(): raise ValueError('Keep ineligible events in source ledger, not computation input')
    for c in ('valley_UTC','return_UTC','resolution_UTC'): e[c]=pd.to_datetime(e[c],utc=True)
    for d in (a,r,z): d['UTC']=pd.to_datetime(d.UTC,utc=True); d['sid']=d.sid.astype(str)
    t=t.rename(columns={'time':'UTC','wind':'wind_kt'}).copy()
    t['UTC']=pd.to_datetime(t.UTC,utc=True); t['sid']=t.sid.astype(str)
    if t.duplicated(['sid','UTC']).any() or z.duplicated(['sid','UTC']).any(): raise ValueError('Duplicate actual observation/interval')
    legal=z[['sid','UTC','valid_6h']]
    a=a.merge(legal,on=['sid','UTC'],how='left',validate='many_to_one')
    r=r[r.sid.isin(e.sid)].merge(legal,on=['sid','UTC'],how='left',validate='one_to_one')
    a['valid_6h']=a.valid_6h.fillna(False).astype(bool);r['valid_6h']=r.valid_6h.fillna(False).astype(bool)
    candidates=[]; reps={}
    for side,d in [('rise',a[a.is_ascent_interval_start]),('Resolution',r)]:
        d=d[d.valid_6h & np.isfinite(d.wind) & d.wind.ge(bin_origin_kt)].copy()
        d['wind_bin']=np.floor((d.wind-bin_origin_kt)/bin_width_kt).astype(int)
        d['bin_lower_kt']=bin_origin_kt+d.wind_bin*bin_width_kt
        d['bin_upper_kt']=d.bin_lower_kt+bin_width_kt
        d['bin_midpoint_kt']=d.bin_lower_kt+bin_width_kt/2
        d['distance_to_midpoint_kt']=(d.wind-d.bin_midpoint_kt).abs()
        keys=['sid','event_id','wind_bin'] if side=='rise' else ['sid','wind_bin']
        d=d.sort_values(keys+['distance_to_midpoint_kt','UTC'],ascending=[True]*len(keys)+[True,side=='rise'],kind='stable')
        d['representative']=~d.duplicated(keys)
        d['side']=side;candidates.append(d);reps[side]=d[d.representative].copy()
    x=reps['rise']; y=reps['Resolution']
    selected=['sid','wind_bin','UTC','wind','lat','lon','stage_code','obs_index','valid_6h']
    y=y[selected].rename(columns={c:c+'_Resolution' for c in selected if c not in ('sid','wind_bin')})
    pairs=x.merge(y,on=['sid','wind_bin'],how='left',validate='many_to_one')
    pairs['matched']=pairs.UTC_Resolution.notna()
    pairs=pairs.rename(columns={'UTC':'UTC_rise','wind':'wind_rise','lat':'lat_rise','lon':'lon_rise','stage_code':'stage_code_rise','obs_index':'obs_index_rise'})
    pairs['pair_id']=pairs.event_id+'|Vbin'+pairs.wind_bin.astype(str)
    pairs['wind_rise_minus_Resolution_kt']=pairs.wind_rise-pairs.wind_Resolution
    pairs['bin_count_event']=pairs.groupby('event_id').matched.transform('sum')
    pairs['event_count_SID']=pairs.groupby('sid').event_id.transform('nunique')
    good=pairs[pairs.matched]
    if not (good.UTC_rise < good.UTC_Resolution).all(): raise ValueError('Matched Resolution is not later')
    if not good.wind_rise_minus_Resolution_kt.abs().lt(bin_width_kt).all(): raise ValueError('Outside matching bin')
    roles=[]
    def add(sid,event_id,utc,role,**kw):
        roles.append(dict(sid=str(sid),event_id=str(event_id),UTC=pd.Timestamp(utc),role=role,**kw))
    for row in e.itertuples():
        for lag in composite_lags:
            utc=row.valley_UTC+pd.Timedelta(hours=lag)
            add(row.sid,row.event_id,utc,'trough_composite',relative_hour=lag,within_rise=(row.valley_UTC<=utc<=row.return_UTC))
        for row2 in a[a.event_id.eq(row.event_id)].itertuples():
            add(row.sid,row.event_id,row2.UTC,'actual_rise',relative_hour=(row2.UTC-row.valley_UTC).total_seconds()/3600,within_rise=True)
    # Each SID's full Resolution is requested once; multiple events share it only within SID.
    for row in r.itertuples(): add(row.sid,'',row.UTC,'whole_Resolution')
    for row in good.itertuples():
        add(row.sid,row.event_id,row.UTC_rise,'matched_rise',pair_id=row.pair_id,wind_bin=row.wind_bin)
        add(row.sid,row.event_id,row.UTC_Resolution,'matched_Resolution',pair_id=row.pair_id,wind_bin=row.wind_bin)
    role=pd.DataFrame(roles)
    keep=['sid','UTC','wind_kt','lat','lon','stage_code']
    keep += [c for c in ['basin','is_tropical','wind_filled_flag','obs_index','dist2land'] if c in t.columns]
    unique=role[['sid','UTC']].drop_duplicates().sort_values(['sid','UTC']).merge(t[keep],on=['sid','UTC'],how='left',validate='one_to_one',indicator=True)
    unique['actual_track_point']=unique['_merge'].eq('both');unique=unique.drop(columns='_merge')
    unique['position_available']=unique.actual_track_point & np.isfinite(unique.lat) & np.isfinite(unique.lon)
    unique['position_extrapolated']=False
    unique['request_id']=['rise|'+sid+'|'+dt.strftime('%Y%m%dT%H%MZ') for sid,dt in zip(unique.sid,unique.UTC)]
    previous=t[['sid','UTC','lat','lon']].copy();previous['UTC']+=pd.Timedelta(hours=6)
    previous=previous.rename(columns={'lat':'past6h_lat','lon':'past6h_lon'})
    unique=unique.merge(previous,on=['sid','UTC'],how='left',validate='one_to_one')
    valid=unique.position_available & np.isfinite(unique.past6h_lat) & np.isfinite(unique.past6h_lon)
    dl=(unique.lon-unique.past6h_lon+180)%360-180
    unique['motion_u_ms']=np.where(valid,earth_radius_m*np.deg2rad(dl)*np.cos(np.deg2rad((unique.lat+unique.past6h_lat)/2))/21600,np.nan)
    unique['motion_v_ms']=np.where(valid,earth_radius_m*np.deg2rad(unique.lat-unique.past6h_lat)/21600,np.nan)
    unique['motion_valid']=valid
    unique['geometry_identity']=[hashlib.sha256((dt.isoformat()+'|'+float(la).hex()+'|'+float(lo).hex()).encode()).hexdigest() if ok else '' for dt,la,lo,ok in zip(unique.UTC,unique.lat,unique.lon,unique.position_available)]
    unique['missing_reason']=np.where(~unique.actual_track_point,'no_actual_track_observation',np.where(~unique.position_available,'missing_actual_position',''))
    role=role.merge(unique[['sid','UTC','request_id','position_available','missing_reason']],on=['sid','UTC'],how='left',validate='many_to_one')
    coverage=e[['sid','event_id','return_order','valley_UTC','return_UTC','ascent_duration_h']].copy()
    counts=pairs.groupby('event_id').agg(rise_bin_count=('wind_bin','size'),matched_bin_count=('matched','sum'))
    coverage=coverage.join(counts,on='event_id');coverage[['rise_bin_count','matched_bin_count']]=coverage[['rise_bin_count','matched_bin_count']].fillna(0).astype(int)
    coverage['matching_reason']=np.where(coverage.matched_bin_count.gt(0),'shared_wind_bins','no_shared_legal_wind_bin')
    coverage['is_first_anchored_return']=coverage.return_order.eq(1)
    return {'pairs':pairs,'coverage':coverage,'roles':role,'requests':unique,'candidates':pd.concat(candidates,ignore_index=True)}
