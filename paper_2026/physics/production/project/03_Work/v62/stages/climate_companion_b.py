from _release_paths import release_path as _release_path
"""Existing P4 native common-cell q700/r700, sampled-stage companion only."""
from analysis_b import *

SOURCE=Path(str(_release_path('mechanism_v5','upper_profile_100_50_interpolation_v1/p4field_src/P4_shared_climate_SID_source.csv')))
EXPECTED='29a912ed01df4aa50557dd6d0bbadba316357bb1b38251ed2ba8df9b14e800ac'
COMPONENTS=['raw_common','climate_common','anomaly_common']

def run():
    assert sha(SOURCE)==EXPECTED
    spec=json.loads(SPEC.read_text(encoding='utf-8-sig'))['module_B']
    binding=dict(module_B_hash=EXPECTED_B,source=str(SOURCE),source_sha256=EXPECTED,
        target='q700/r700 stage averages restricted to the existing boundary-sampled actual-track legal6h intervals',
        components=COMPONENTS,spatial_support='native raw/climate/anomaly identical cells; all original fraction>=.9',
        deduplication='SID UTC field; duplicate numeric values must match <=1e-10 before first-row collapse',
        time_support='merge to accepted frozen actual SID UTC; exact legal forward6h only; no post-terminal6h',
        weight='within SID stage use accepted6h; equal SID. Report partial coverage, not full-stage climatology',
        inference='descriptive percentile2000 shared originalSID bank; no P family or significance selection',
        registered_before_new_companion_estimates=True)
    write_json('climate_companion_binding.json',binding)
    cols=['request_id','sid','UTC','field','frozen_stage','actual_track_point','legal_forward6h']+COMPONENTS
    cols += [c+s for c in COMPONENTS for s in ['__valid','__fraction']]
    x=pd.read_csv(SOURCE,usecols=cols,keep_default_na=False)
    x=x[x.field.isin(['q700','r700'])].copy(); n_input=len(x)
    for c in COMPONENTS:
        x[c]=pd.to_numeric(x[c],errors='coerce')
        x[c+'__fraction']=pd.to_numeric(x[c+'__fraction'],errors='coerce')
    x['UTC']=pd.to_datetime(x.UTC,utc=True,errors='coerce')
    d=pd.read_parquet(spec['inputs']['accepted_sp_track']['path'])
    key=['sid','UTC','field'];dupx=x[x.duplicated(key,keep=False)]
    errors={}
    for c in COMPONENTS:
        g=dupx.groupby(key,dropna=False)[c]
        errors[c]=float((g.max()-g.min()).max())
        assert errors[c]<1e-10
        assert (g.apply(lambda a:a.notna().nunique())<=1).all()
    x=x.drop_duplicates(key).merge(d[['row_id','sid','UTC','stage','interval_hours_v3','legal_interval_v3']],on=['sid','UTC'],how='left',validate='many_to_one',indicator=True)
    supported=x['_merge'].eq('both') & x.legal_interval_v3.fillna(False) & x.actual_track_point.eq(True) & x.legal_forward6h.eq(True)
    for c in COMPONENTS:
        supported &= np.isfinite(x[c]) & x[c+'__valid'].eq(True) & (x[c+'__fraction']>=.9)
    x['selected']=supported;x.to_csv(OUT/'climate_companion_UTC_audit.csv',index=False)
    v=x.loc[supported].copy()
    assert v.stage.eq(v.frozen_stage).all()
    closure=np.abs(v.raw_common-v.climate_common-v.anomaly_common)
    assert closure.max()<1e-10
    legal=d.groupby(['sid','stage']).interval_hours_v3.sum()
    source=[];cov=[]
    for (sid,stage,field),g in v.groupby(['sid','stage','field'],sort=True):
        hours=g.interval_hours_v3.to_numpy(float);total=hours.sum()
        rr=dict(sid=sid,stage=stage,field=field,covered_h=total,legal_h=legal.loc[(sid,stage)],
            n_valid_intervals=len(g),coverage_fraction=total/legal.loc[(sid,stage)],
            source_row_ids=';'.join(g.row_id))
        for c in COMPONENTS:rr[c]=float(np.average(g[c],weights=hours))
        source.append(rr)
    a=pd.DataFrame(source);a.to_csv(OUT/'climate_companion_stage_sid.csv',index=False)
    si=pd.read_csv(OUT/'bootstrap_sid_order.csv',keep_default_na=False).set_index('sid')['index']
    weights=np.load(OUT/'bootstrap_sid_multiplicities.npy')
    summaries=[];draws=[];pairs=[]
    def append_est(z,values,identity):
        w=weights[:,si.loc[z.sid].to_numpy(int)];dd=(w@values)/w.sum(axis=1)
        r={**identity,'estimate':float(np.mean(values)),'ci_low':float(np.quantile(dd,.025)),
            'ci_high':float(np.quantile(dd,.975)),'n_sid':len(z),'bootstrap_attempted':2000,
            'bootstrap_valid':int(np.isfinite(dd).sum())}
        summaries.append(r)
        draws.append(pd.DataFrame({**identity,'draw':np.arange(2000),'estimate':dd}))
    for field in ['q700','r700']:
        for stage in STAGES:
            z=a[(a.field==field)&(a.stage==stage)]
            for c in COMPONENTS:
                append_est(z,z[c].to_numpy(float),dict(field=field,stage=stage,contrast='',component=c,
                    units='g kg-1' if field=='q700' else ('pp' if c=='anomaly_common' else '%')))
            full=d[d.stage==stage].groupby('sid').interval_hours_v3.sum()
            cov.append(dict(field=field,stage=stage,n_accepted_stage_sid=int((full>0).sum()),
                n_companion_sid=len(z),covered_h=float(z.covered_h.sum()),all_stage_legal_h=float(full.sum()),
                pooled_legal_fraction=float(z.covered_h.sum()/full.sum()),
                median_within_included_sid_fraction=float(z.coverage_fraction.median()),
                n_full_coverage_sid=int((z.coverage_fraction>=1-1e-12).sum())))
        for frm,to in PAIRS:
            left=a[(a.field==field)&(a.stage==frm)].drop(columns='stage')
            right=a[(a.field==field)&(a.stage==to)].drop(columns=['stage','field'])
            p=left.merge(right,on='sid',suffixes=('_before','_after'),validate='one_to_one')
            p['contrast']=f'{to}-{frm}';pairs.append(p)
            for c in COMPONENTS:
                append_est(p,(p[c+'_after']-p[c+'_before']).to_numpy(float),dict(field=field,stage='',contrast=f'{to}-{frm}',component=c,
                    units='g kg-1' if field=='q700' else 'pp'))
    pd.DataFrame(summaries).to_csv(OUT/'climate_companion_summary.csv',index=False)
    pd.DataFrame(cov).to_csv(OUT/'climate_companion_coverage.csv',index=False)
    pd.concat(draws).to_parquet(OUT/'climate_companion_bootstrap.parquet',index=False)
    pd.concat(pairs).to_csv(OUT/'climate_companion_adjacent_sid.csv',index=False)
    write_json('climate_companion_receipt.json',dict(status='COMPLETE',input_rows=n_input,deduplicated_rows=len(x),
        valid_source_rows=len(v),selected_unique_SID=v.sid.nunique(),duplicate_max_difference=errors,
        max_native_common_identity_error=float(closure.max()),summary_rows=len(summaries),
        full_stage_climatology_claim=False,source_sha256=EXPECTED,raw_nc_reads=0,
        stage_scope='existing P4 boundary-neighbourhood intervals only; partial stage coverage explicitly quantified',
        absent_core_climate=['SST','r500 for 700/500 RH','d200','w500','vo850','climate mean shear magnitude']))
    print(pd.DataFrame(cov).to_string(index=False),flush=True)

if __name__=='__main__':run()
