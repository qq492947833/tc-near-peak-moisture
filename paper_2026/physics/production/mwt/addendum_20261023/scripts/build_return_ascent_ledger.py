from _release_paths import release_path as _release_path
"""Retain all anchored near-peak returns and their observed valley-to-return records."""
from pathlib import Path
import ast,hashlib,json,datetime
import numpy as np
import pandas as pd

WORK=Path(__file__).absolute().parents[1];D=WORK/'source_data';R=WORK/'reports'
ORIGINAL_INTERVALS=Path(str(_release_path('legacy_project','ERA5_environment_v1/run_20260920_215821/representation_value_v4/scripts/prepare_durations_v4.py')))
FROZEN_DURATIONS=ORIGINAL_INTERVALS.parents[1]/'source_data/durations.csv'
FROZEN_EVENTS=WORK.parent/'source_data/events_frozen.csv'

def save(n,frame):
    frame.to_csv(D/(n+'.csv'),index=False,encoding='utf-8-sig');frame.to_parquet(D/(n+'.parquet'),index=False)

def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str),encoding='utf8')

def functions():
    names=['utc','runs','later_category','contiguous','intervals']
    tree=ast.parse(ORIGINAL_INTERVALS.read_text(encoding='utf8'))
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(nodes)==len(names)
    env={'np':np,'pd':pd};exec(compile(ast.Module(nodes,type_ignores=[]),'unchanged_interval_support','exec'),env)
    return env['intervals']

def contiguous_runs(mask,t):
    out=[]
    for i in np.flatnonzero(mask):
        if not out or i!=out[-1][-1]+1 or t[i]-t[out[-1][-1]]!=pd.Timedelta(hours=6):out.append([int(i)])
        else:out[-1].append(int(i))
    return out

def ledger(mode,intervals):
    sums=pd.read_parquet(D/(mode+'_main_SID.parquet'));tracks=pd.read_parquet(D/(mode+'_lifecycle_tracks.parquet'))
    tracks=tracks[tracks.sid.isin(sums.sid)].copy();tracks['obs_index']=tracks.groupby('sid',sort=False).cumcount()
    save(mode+'_main_observations',tracks)
    groups={k:g.sort_values('time').reset_index(drop=True) for k,g in tracks.groupby('sid',sort=False)}
    durations=[];episodes=[];events=[];departures=[];sequences=[];windows=[];res=[];interval_tables=[]
    for z in sums.to_dict('records'):
        sid=z['sid'];g=groups[sid];t=pd.DatetimeIndex(g.time);w=g.wind.to_numpy(float);finite=np.isfinite(w);threshold=float(z['mature_threshold_wind'])
        date=lambda k:pd.Timestamp(z[k])
        rr,ints=intervals(g.rename(columns={'time':'UTC','wind':'wind_kt'}),z,config='progress025_near090_'+mode,saved=False)
        # Preserve the published common F/C5/R90C comparison window, which is
        # broader than physical F support for two non-recurrent histories.
        common=np.flatnonzero(ints.mask_F|ints.mask_R5|ints.mask_R)
        lo,hi=int(common[0]),min(int(common[-1])+1,len(g)-1)
        rr['common_window_missing_n']=int(np.sum(~finite[lo:hi+1]))
        rr['common_window_non6h_gap_n']=int(np.sum(np.diff(t.asi8[lo:hi+1])!=21600e9))
        rr['common_window_first_UTC']=t[lo];rr['common_window_last_UTC']=t[hi]
        rr['main_complete']=rr['common_window_missing_n']==0 and rr['common_window_non6h_gap_n']==0
        ints['main_complete']=rr['main_complete']
        durations.append(rr);interval_tables.append(ints)
        legal=ints.valid_6h.to_numpy(bool);hot=finite&(w>=threshold);anchored=hot&(t>=date('maturity_time'));runs=contiguous_runs(anchored&legal,t)
        fullruns=contiguous_runs(hot&legal,t)
        rr['anchored_legal_episode_n']=len(runs);rr['full_lifecycle_legal_episode_n']=len(fullruns)
        rr['anchored_legal_recurrent']=len(runs)>1;rr['full_lifecycle_legal_recurrent']=len(fullruns)>1
        for no,run in enumerate(runs,1):
            episodes.append({'sid':sid,'episode_order':no,'start_UTC':t[run[0]],'last_selected_node_UTC':t[run[-1]],'interval_end_UTC':t[run[-1]]+pd.Timedelta(hours=6),'covered_h':len(run)*6,'threshold_kt':threshold,'main_complete':bool(rr['main_complete']),'basin_genesis':z['basin_genesis']})
        for i in range(len(g)-1):
            if not (t[i+1]-t[i]==pd.Timedelta(hours=6) and anchored[i] and g.stage_code.iloc[i]==3 and np.isfinite(w[i+1]) and w[i+1]<threshold):continue
            j=i+1;later=np.flatnonzero(anchored&(np.arange(len(g))>j))
            cls='internal' if len(later) else ('final' if t[j]==date('resolution_time') else 'unresolved')
            departures.append({'sid':sid,'event_id':f'{sid}|departure|{i:04d}','event_class':cls,'t0':t[j],'departure_obs_index':j,'previous_obs_index':i,'wind_kt':w[j],'threshold_kt':threshold,'lat':g.lat.iloc[j],'lon':g.lon.iloc[j],'basin':g.basin.iloc[j],'main_complete':bool(rr['main_complete'])})
        # Return ordinal is within the original first-LMI-anchored legal domain.
        for no,(left,right) in enumerate(zip(runs,runs[1:]),1):
            ret=right[0];start=left[-1]+1;span=np.arange(left[-1],ret+1)
            sub=span[finite[span]&~hot[span]]
            if not len(sub):
                # A gap that divides intervals is not an observed weakening/return.
                events.append({'sid':sid,'event_id':f'{sid}|return|{no:03d}','return_order':no,'ledger_eligible':False,'reason':'gap_split_without_observed_subthreshold_wind','return_UTC':t[ret],'main_complete':bool(rr['main_complete'])});continue
            dep=int(sub[0]);bridge=np.arange(dep,ret+1);complete=bool(finite[span].all() and np.all(np.diff(t.asi8[span])==21600e9))
            minwind=float(np.nanmin(w[bridge]));valley=int(bridge[np.flatnonzero(w[bridge]==minwind)[-1]])
            ascent=np.arange(valley,ret+1);sequence_complete=bool(finite[ascent].all() and np.all(np.diff(t.asi8[ascent])==21600e9))
            returned_legal=bool(legal[ret]);valid=bool(rr['main_complete'] and complete and sequence_complete and returned_legal)
            eid=f'{sid}|return|{no:03d}';legacy=f'{sid}|departure|{dep-1:04d}'
            row={'sid':sid,'event_id':eid,'return_order':no,'plot_order_class':'1st' if no==1 else ('2nd' if no==2 else '3rd_and_later'),'departure_event_id':legacy,'temporary_departure_UTC':t[dep],'valley_UTC':t[valley],'return_UTC':t[ret],'resolution_UTC':date('resolution_time'),'temporary_departure_obs_index':dep,'valley_obs_index':valley,'return_obs_index':ret,'near_peak_threshold_kt':threshold,'valley_wind_kt':w[valley],'return_wind_kt':w[ret],'net_wind_gain_kt':w[ret]-w[valley],'ascent_duration_h':(t[ret]-t[valley]).total_seconds()/3600,'bridge_duration_h':(t[ret]-t[dep]).total_seconds()/3600,'valley_tie_rule':'last observed exact minimum between departure and next return','minimum_tie_count':int(np.sum(w[bridge]==minwind)),'first_step_after_valley_h':(t[valley+1]-t[valley]).total_seconds()/3600 if valley<ret else np.nan,'first_step_wind_gain_kt':w[valley+1]-w[valley] if valley<ret and np.isfinite(w[valley+1]) else np.nan,'ascent_has_short_term_decrease':bool(np.any(np.diff(w[ascent])<0)),'ascent_observed_records':len(ascent),'ascent_expected_6h_records':int((t[ret]-t[valley]).total_seconds()/21600)+1,'ascent_missing_wind_n':int(np.sum(~finite[ascent])),'ascent_filled_wind_n':int(g.wind_filled_flag.iloc[ascent].sum()),'bridge_has_non6h_gap':not bool(np.all(np.diff(t.asi8[span])==21600e9)),'bridge_complete':complete,'main_complete':bool(rr['main_complete']),'ledger_eligible':valid,'reason':'eligible' if valid else ('original_common_window_incomplete' if not rr['main_complete'] else 'bridge_or_ascent_incomplete'),'basin_genesis':z['basin_genesis'],'original_domain':'first-LMI-anchored near-peak domain','stage_code':3,'stage_label':'F','mode':mode}
            for name,ix in [('temporary_departure',dep),('valley',valley),('return',ret)]:
                for col in ['lat','lon','basin','wind','is_tropical']:row[name+'_'+col]=g.iloc[ix][col]
            row['position_complete_anchors']=all(np.isfinite(row[x+'_'+c]) for x in ['temporary_departure','valley','return'] for c in ['lat','lon'])
            events.append(row)
            for ix in ascent:
                node=g.iloc[ix].to_dict();node.update(event_id=eid,return_order=no,UTC=t[ix],lag_from_valley_h=(t[ix]-t[valley]).total_seconds()/3600,is_ascent_interval_start=bool(ix<ret and legal[ix]),is_return_node=bool(ix==ret),ledger_eligible=valid,original_stage='F');sequences.append(node)
            lookup={value:ix for ix,value in enumerate(t)}
            for lag in range(-24,25,6):
                stamp=t[valley]+pd.Timedelta(hours=lag);ix=lookup.get(stamp)
                node={'sid':sid,'event_id':eid,'return_order':no,'valley_UTC':t[valley],'return_UTC':t[ret],'UTC':stamp,'lag_from_valley_h':lag,'ledger_eligible':valid,'record_present':ix is not None,'within_ascent_inclusive':bool(t[valley]<=stamp<=t[ret]),'within_ascent_interval':bool(t[valley]<=stamp<t[ret]),'role':'valley_aligned'}
                if ix is not None:node.update({col:g.iloc[ix][col] for col in ['obs_index','lat','lon','wind','wind_original','basin','stage_code','is_tropical','dist2land','wind_filled_flag']})
                windows.append(node)
        q=g[g.time.ge(date('resolution_time'))].copy();q['UTC']=q.time;q['resolution_UTC']=date('resolution_time');q['valid_wind']=np.isfinite(q.wind);q['position_valid']=np.isfinite(q.lat)&np.isfinite(q.lon);q['main_complete']=bool(rr['main_complete']);q['basin_genesis']=z['basin_genesis'];res.append(q)
    frames={'durations':pd.DataFrame(durations),'episodes':pd.DataFrame(episodes),'return_events':pd.DataFrame(events),'departures':pd.DataFrame(departures),'ascent_observations':pd.DataFrame(sequences),'valley_aligned_observations':pd.DataFrame(windows),'resolution_observations':pd.concat(res,ignore_index=True),'intervals':pd.concat(interval_tables,ignore_index=True)}
    for name,frame in frames.items():save(mode+'_'+name,frame)
    return frames

def main():
    revision=json.loads((R/'boundary_revision_summary.json').read_text(encoding='utf8'))
    assert revision['status']=='ONE_ALGORITHM_CHANGE_COMPUTED' and revision['original_files_unchanged']
    ints=functions();old=ledger('old',ints);new=ledger('new',ints)
    frozen=pd.read_csv(FROZEN_DURATIONS,keep_default_na=False,na_values=['','nan','NaN']).set_index('sid');old_d=old['durations'].set_index('sid').loc[frozen.index]
    for k in ['main_complete','T_R_h','T_U_h','T_F_h','T_L_h','T_B_h']:
        if k=='main_complete':assert np.array_equal(old_d[k].to_numpy(bool),frozen[k].to_numpy(bool)),k
        else:assert np.allclose(old_d[k],frozen[k],rtol=0,atol=0,equal_nan=True),k
    oe=old['return_events'];ne=new['return_events'];oe=oe[oe.ledger_eligible];ne=ne[ne.ledger_eligible]
    assert oe.sid.nunique()==123 and len(oe)==133,('Old returned-event reproduction differs',oe.sid.nunique(),len(oe))
    identity=['sid','temporary_departure_UTC','valley_UTC','return_UTC','net_wind_gain_kt','ascent_duration_h']
    comparison=oe[identity].merge(ne[identity],on=identity,how='outer',indicator=True,validate='1:1');save('old_new_return_identity',comparison)
    fe=pd.read_csv(FROZEN_EVENTS,keep_default_na=False);fe.t0=pd.to_datetime(fe.t0,utc=True).dt.tz_localize(None)
    dep=new['departures'];pairs=fe.merge(dep,on=['sid','event_id'],how='left',suffixes=('_frozen','_new'),validate='1:1')
    pairs['event_id_still_present']=pairs.t0_new.notna();pairs['anchor_UTC_unchanged']=pairs.t0_frozen.eq(pairs.t0_new);pairs['class_unchanged']=pairs.event_class_frozen.eq(pairs.event_class_new)
    pairs['threshold_unchanged']=pairs.near_threshold_kt.eq(pairs.threshold_kt)
    for c in ['lat','lon']:pairs[c+'_absolute_delta']=abs(pairs[c+'_frozen']-pairs[c+'_new'])
    save('frozen117_new_event_identity',pairs)
    basins=[]
    for mode,frames in [('old',old),('new',new)]:
        d=frames['durations'];e=frames['return_events'];eligible=d[d.main_complete];rec=set(e.loc[e.ledger_eligible,'sid'])
        for b,g in eligible.groupby('basin_genesis'):
            numerator=int(g.sid.isin(rec).sum());basins.append({'mode':mode,'basin_genesis':b,'recurrent_SID':numerator,'eligible_SID':len(g),'percentage':100*numerator/len(g),'denominator':'eligible complete anchored near-peak counting histories in genesis basin'})
    save('basin_recurrence_proportions',pd.DataFrame(basins))
    new_counts={int(key):int(ne.return_order.eq(key).sum()) for key in sorted(ne.return_order.unique())}
    summary={'status':'COMPLETE_NEW_BOUNDARY_AND_RETURN_LEDGER','old_main_SID':len(old['durations']),'old_common_SID':int(old['durations'].main_complete.sum()),'old_recurrent_SID':int(oe.sid.nunique()),'old_return_events':len(oe),'new_main_SID':len(new['durations']),'new_common_SID':int(new['durations'].main_complete.sum()),'new_recurrent_SID':int(ne.sid.nunique()),'new_return_events':len(ne),'new_event_order_counts':new_counts,'return_identity_comparison':comparison._merge.value_counts().to_dict(),'frozen117_events':len(fe),'frozen117_ID_anchor_class_threshold_all_unchanged':bool(pairs[['event_id_still_present','anchor_UTC_unchanged','class_unchanged','threshold_unchanged']].all(axis=None)),'maximum_frozen117_raw_coordinate_delta_deg':float(pairs[['lat_absolute_delta','lon_absolute_delta']].max().max()),'new_ascent_short_term_decrease_events':int(ne.ascent_has_short_term_decrease.sum()),'new_last_minimum_tie_events':int(ne.minimum_tie_count.gt(1).sum()),'first_step_nonpositive_or_non6h':ne.loc[(ne.first_step_after_valley_h!=6)|(ne.first_step_wind_gain_kt<=0),['sid','event_id','first_step_after_valley_h','first_step_wind_gain_kt']].to_dict('records'),'general_reintensification_events_used':False,'resolution_matching':'Not selected by this ledger; all observed Resolution candidates saved for fixed rule selection before environmental comparisons','no_environment_or_energy_values_examined':True}
    dump(R/'return_ledger_summary.json',summary)
    dump(R/'return_ledger_sources.json',{'original_intervals_source':str(ORIGINAL_INTERVALS),'original_intervals_sha256':hashlib.sha256(ORIGINAL_INTERVALS.read_bytes()).hexdigest(),'frozen_original_durations':str(FROZEN_DURATIONS),'frozen_events':str(FROZEN_EVENTS),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print(json.dumps(summary,indent=2,ensure_ascii=False,default=str))

if __name__=='__main__':main()

