def compare(base,obs,b,other):
    bs=set(base.sid);xs=set(b.sid);common=bs&xs;m=base.merge(b,on='sid',suffixes=('_base','_test'),validate='one_to_one');r=dict(n_valid=len(xs),n_common_with_baseline=len(common),n_removed_from_baseline=len(bs-xs),n_added_vs_baseline=len(xs-bs),baseline_retention=len(common)/len(bs),experiment_overlap=len(common)/len(xs),sample_jaccard=len(common)/len(bs|xs))
    for name in ['escalation','maturity','resolution']:
        v=((pd.to_datetime(m[name+'_time_test'])-pd.to_datetime(m[name+'_time_base'])).dt.total_seconds()/3600).abs()
        r['median_abs_'+name+'_shift_h']=float(v.median());r['p90_abs_'+name+'_shift_h']=float(v.quantile(.9))
    for name in ['development','escalation','maturity','resolution']:
        a=m[name+'_duration_hours_base'];b_=m[name+'_duration_hours_test'];d=b_-a
        r[name+'_duration_MAD_h']=float(d.abs().median());r[name+'_duration_spearman']=float(spearmanr(a,b_,nan_policy='omit').statistic) if len(m)>=3 and a.nunique()>1 and b_.nunique()>1 else np.nan;r['median_'+name+'_change_h']=float(d.median())
    a=obs[obs.sid.isin(common)].merge(other[other.sid.isin(common)],on=['sid','time'],suffixes=('_base','_test'),validate='one_to_one');r['stage_assignment_agreement']=float(a.stage_code_base.eq(a.stage_code_test).mean())
    mask=m.landfall_event_count_base.gt(0)&m.landfall_event_count_test.gt(0);r['n_common_landfall']=int(mask.sum());r['landfall_stage_agreement']=float(m.loc[mask,'first_landfall_stage_code_base'].eq(m.loc[mask,'first_landfall_stage_code_test']).mean())
    x=m.near_lmi_episode_count_base.gt(1)|m.reintensification_event_count_base.gt(0);y=m.near_lmi_episode_count_test.gt(1)|m.reintensification_event_count_test.gt(0);r['complex_lifecycle_class_agreement']=float(x.eq(y).mean());return r
