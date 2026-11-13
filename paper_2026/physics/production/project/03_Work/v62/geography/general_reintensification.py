from _release_paths import release_path as _release_path
"""Geographic aggregation of the inherited >=10kt/24h post-first-LMI episodes.
No episode thresholds, windows or cluster labels are refitted/reselected.
"""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
import geography_tables as G
O=G.O;BASE=Path(str(_release_path('legacy_project','IBTrACS_interdecadal_v19/inputs/v16/data')))
details=G.read(BASE/'reintensification_episode_details.csv');counts=G.read(BASE/'cyclone_event_counts_unclipped.csv');tracks=G.read(BASE/'analysis_records.csv.gz');main=G.read(Path(str(_release_path('legacy_project','IBTrACS_interdecadal_v19/tables/analysis_per_SID_main_window.csv'))))
tracks.UTC=pd.to_datetime(tracks.UTC,utc=True);idx=tracks.set_index(['sid','UTC']);records=tracks.set_index('record_index');frozen=G.read(G.RUN/'representation_value_v4/source_data/durations.csv').set_index('sid')
detail_count=details.groupby('sid').size();joined=counts.set_index('sid').join(detail_count.rename('ledger_count')).fillna({'ledger_count':0});assert np.array_equal(joined.reintensification_event_count,joined.ledger_count)
rows=[]
for z in details.itertuples():
    t=pd.Timestamp(z.first_window_start,tz='UTC');end=t+pd.Timedelta(hours=24);ids=[int(x) for x in str(z.qualifying_record_indices).split(';')];q=records.loc[ids];assert q.sid.eq(z.sid).all();assert (q.delta_v24_forward>=10).all()
    g=idx.loc[(z.sid,t)];endpoint=idx.loc[(z.sid,end)] if (z.sid,end) in idx.index else None
    states=list(q.state_class.astype(str));ends=[]
    for r in q.itertuples():
        tt=r.UTC+pd.Timedelta(hours=24)
        if (z.sid,tt) in idx.index:ends.append(idx.loc[(z.sid,tt)])
    states+= [str(x.state_class) for x in ends]
    status='tropical_confirmed' if all(x=='A' for x in states) else 'ET_or_other_or_mixed' if any(x=='B' for x in states) else 'unknown'
    et=bool(q.USA_STATUS.fillna('').eq('EX').any() or q.NATURE.fillna('').eq('ET').any() or any(str(x.USA_STATUS)=='EX' or str(x.NATURE)=='ET' for x in ends))
    within=z.sid in set(main.sid);d=frozen.loc[z.sid] if z.sid in frozen.index else None
    rows.append(dict(**z._asdict(),first_window_start_UTC=t,first_window_end_UTC=end,start_lat=g.lat,start_lon=g.lon,start_basin=g.basin,start_state=g.state_class,start_USA_STATUS=g.USA_STATUS,start_NATURE=g.NATURE,start_wind_filled_flag=g.wind_filled_flag,end_lat=endpoint.lat if endpoint is not None else np.nan,end_lon=endpoint.lon if endpoint is not None else np.nan,end_basin=endpoint.basin if endpoint is not None else 'UNKNOWN',status_class=status,explicit_ET_record=et,within_main2368=within,within_common2366=bool(d is not None and d.main_complete),frozen_near_peak_recurrent=bool(d is not None and d.multi),first_window_after_resolution=bool(d is not None and t>=pd.Timestamp(d.resolution_time)),position_definition='First qualifying 24 h window start; distinct from near-peak return location'))
out=pd.DataFrame(rows);out['start_region'],out['start_ocean']=G.region(out.start_basin,out.start_lat,out.start_lon);out['end_region'],out['end_ocean']=G.region(out.end_basin,out.end_lat,out.end_lon);out.to_csv(O/'general_reintensification_all_frozen_episode_sources.csv',index=False)
used=out[out.within_main2368].copy();used.to_csv(O/'general_reintensification_main_episode_sources.csv',index=False)
tables=[]
for scope,dd in [('main2368',main),('common2366',main[main.sid.isin(frozen.index[frozen.main_complete])])]:
    cc=dd[['sid','basin_genesis','reintensification_event_count']].copy().rename(columns={'basin_genesis':'basin'});cc['ever']=cc.reintensification_event_count.gt(0);w=G.weights(len(cc),'general_reint_'+scope)
    for basin,mask in G.regions(cc):
        n=int(mask.sum());k=int(cc.loc[mask,'ever'].sum());p,ci,_=G.bootmean(cc.ever,mask,w);lo,hi=G.wilson(k,n);sparse=n<30 or k<5 or k==n
        tables.append(dict(scope=scope,region=basin,n_sid=n,n_ever_reintensified=k,events=int(cc.loc[mask,'reintensification_event_count'].sum()),fraction=p,ci_low=lo if sparse else ci[0],ci_high=hi if sparse else ci[1],CI_method='Wilson binomial' if sparse else '2000 SID bootstrap',geography='genesis basin',event_definition='inherited post-first-LMI >=10kt/24h contiguous qualifying-window episodes'))
pd.DataFrame(tables).to_csv(O/'general_reintensification_genesis_basin_rates.csv',index=False)
loc=[]
for reg,mask in G.regions(used.rename(columns={'start_basin':'basin','start_region':'region'})):
    q=used[mask];loc.append(dict(region=reg,events=len(q),unique_sid=q.sid.nunique(),tropical_confirmed=int(q.status_class.eq('tropical_confirmed').sum()),ET_other_mixed=int(q.status_class.eq('ET_or_other_or_mixed').sum()),unknown=int(q.status_class.eq('unknown').sum()),explicit_ET_events=int(q.explicit_ET_record.sum()),after_resolution_start=int(q.first_window_after_resolution.sum()),from_near_peak_recurrent_SID=int(q.frozen_near_peak_recurrent.sum()),quantity='event-start location counts; not regional probability'))
pd.DataFrame(loc).to_csv(O/'general_reintensification_location_status_counts.csv',index=False)
G.dump(O/'general_reintensification_receipt.json',dict(status='INHERITED_EPISODE_IDENTITIES_VERIFIED_AND_REGIONALLY_AGGREGATED',original_all_events=len(details),original_all_SID=details.sid.nunique(),main_events=len(used),main_SID=used.sid.nunique(),threshold_kt_per24h=10,source_count_equals_ledger_for_all_SIDs=True,near_peak_labels_unchanged=True,not_P3_risk_population=True,sources={str(p):G.sha(p) for p in [BASE/'reintensification_episode_details.csv',BASE/'cyclone_event_counts_unclipped.csv',BASE/'analysis_records.csv.gz']},source_definition='Original first-LMI index onward, finite forward24h gain>=10, contiguous true runs; first/last qualifying window timestamps and raw record IDs retained.'))
print(json.dumps({'main_events':len(used),'main_SID':used.sid.nunique(),'all_events':len(details)}))
