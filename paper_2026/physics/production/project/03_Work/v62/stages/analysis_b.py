"""Registered v6.2 B: accepted stage-internal data, paired contrasts and support.

One original SID is the sampling unit. Historical inputs are read-only.
"""
import os
for _k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[_k]='1'
from pathlib import Path
import sys,json,hashlib,time,platform
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from scipy import stats

OUT=Path(__file__).resolve().parent
SPEC=OUT.parent/'analysis_spec_v62.json'
EXPECTED_B='55d78eed00f44d138bbe0c26382bec48eae527d0c18d534d62a2e05321e4a593'
GEO=['lat','lon_sin','lon_cos','month_sin','month_cos']
STATE=GEO+['wind_kt','age_h','dv24_kt','center_land','past24_land_contact']
STAGES=['D','E','F','Res']
PAIRS=list(zip(STAGES[:-1],STAGES[1:]))

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,(np.integer,np.bool_)):return x.item()
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    return x

def write_json(name,x):
    (OUT/name).write_text(json.dumps(clean(x),indent=2,ensure_ascii=False),encoding='utf-8')

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

def execute():
    started=time.time(); spec=json.loads(SPEC.read_text(encoding='utf-8-sig'))
    conf=spec['module_B']; bhash=hashlib.sha256(json.dumps(conf,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    if bhash!=EXPECTED_B:raise RuntimeError(f'Frozen B hash mismatch {bhash}')
    if conf['status']!='FROZEN_FOR_EXECUTION':raise RuntimeError('B not authorized')
    source_checks={}
    for key,v in conf['inputs'].items():
        actual=sha(v['path']); source_checks[key]=dict(path=v['path'],sha256=actual,match=actual==v['sha256'])
        if actual!=v['sha256']:raise RuntimeError('Input changed: '+key)
    write_json('input_hash_verification.json',source_checks)
    d=pd.read_parquet(conf['inputs']['accepted_sp_track']['path']).sort_values(['sid','UTC']).reset_index(drop=True)
    next_utc=d.groupby('sid').UTC.shift(-1);gap=(next_utc-d.UTC).dt.total_seconds()/3600
    assert len(d)==64210 and d.sid.nunique()==2368 and not d[['sid','UTC']].duplicated().any()
    assert set(d.basin_genesis.unique())==set(['NA','EP','WP','NI','SI','SP'])
    assert ((d.interval_hours_v3==6)==d.legal_interval_v3).all()
    assert (gap[d.interval_hours_v3>0]==6).all()
    assert (d.loc[d.terminal_point_v3,'interval_hours_v3']==0).all()
    assert set(d.interval_hours_v3.unique())=={0.,6.}
    sids=sorted(d.sid.unique());sid_index={s:i for i,s in enumerate(sids)}
    pd.DataFrame(dict(sid=sids,index=np.arange(len(sids)))).to_csv(OUT/'bootstrap_sid_order.csv',index=False)
    rng=np.random.Generator(np.random.PCG64(2026101102))
    weights=rng.multinomial(len(sids),np.repeat(1/len(sids),len(sids)),size=2000).astype(np.uint16)
    np.save(OUT/'bootstrap_sid_multiplicities.npy',weights)
    old=pd.read_csv(conf['inputs']['stage_sid']['path'],keep_default_na=False)
    raw_summary=pd.read_csv(conf['inputs']['stage_raw_existing']['path'],keep_default_na=False)
    reference=pd.read_csv(conf['inputs']['reference_scale']['path'],keep_default_na=False).set_index('variable')
    reconstructed=[];raw_pairs=[];raw_rows=[];raw_draws=[];composition=[];models={};stage_model=[]
    verify=[];stages=[]
    for vv in conf['variables']:
        var=vv['variable'];unit=vv['unit'];ref=reference.loc[var]
        a=aggregate_stage(d,var,[]);a['variable']=var;a['unit']=unit;a['scheme']='raw';stages.append(a)
        o=old.loc[old.variable==var]
        cmp=a.merge(o,on=['sid','stage'],suffixes=('_new','_old'),validate='one_to_one',how='outer',indicator=True)
        assert (cmp['_merge']=='both').all()
        err=float(np.max(np.abs(cmp.estimate_new-cmp.estimate_old)))
        assert err<1e-10 and np.allclose(cmp.covered_h_new,cmp.covered_h_old)
        assert np.allclose(cmp.coverage_fraction_new,cmp.coverage_fraction_old)
        verify.append(dict(variable=var,n_sid_stage=len(a),max_abs_error=err))
        for stage in STAGES:
            aa=a[a.stage==stage];rr=raw_summary[(raw_summary.variable==var)&(raw_summary.stage==stage)].iloc[0]
            assert abs(aa.estimate.mean()-rr.estimate)<1e-10 and len(aa)==rr.n_sid
            row=rr.to_dict();row.update(unit=unit,legal_h=float(aa.legal_h.sum()),
                mean_temporal_coverage=float(aa.coverage_fraction.mean()),min_temporal_coverage=float(aa.coverage_fraction.min()),
                ci_provenance='accepted E1 SP-v3.2 seed20260920; exact stage means/n reconstructed',
                original_bootstrap_seed=20260920)
            reconstructed.append(row)
        for frm,to in PAIRS:
            p=make_pair(a,frm,to,[]);ii=[sid_index[s] for s in p.sid]
            r,draw=summarize_raw(p.delta.to_numpy(float),weights[:,ii])
            r.update(variable=var,unit=unit,contrast=f'{to}-{frm}',stage_from=frm,stage_to=to,
                before_matched_estimate=float(p.estimate_before.mean()),after_matched_estimate=float(p.estimate_after.mean()),
                covered_h_before=float(p.covered_h_before.sum()),covered_h_after=float(p.covered_h_after.sum()),
                reference_sd=ref.reference_sd,estimate_standardized=r['estimate']/ref.reference_sd,
                ci_low_standardized=r['ci_low']/ref.reference_sd,ci_high_standardized=r['ci_high']/ref.reference_sd,
                status='ESTIMATED',ci_method='2000 SID percentile bootstrap',p_method='two-sided paired t; mean SID difference=0')
            raw_rows.append(r);p['variable']=var;p['contrast']=f'{to}-{frm}';p['unit']=unit;raw_pairs.append(p)
            raw_draws.append(pd.DataFrame(dict(draw=np.arange(2000),variable=var,contrast=f'{to}-{frm}',estimate=draw)))
            for group in ['basin_genesis','period']:
                for lev,pp in p.groupby(group,sort=True):
                    composition.append(dict(variable=var,contrast=f'{to}-{frm}',group=group,level=lev,n_pairs=len(pp),
                        estimate=pp.delta.mean(),direction=int(np.sign(pp.delta.mean())),unit=unit,
                        covered_h_before=pp.covered_h_before.sum(),covered_h_after=pp.covered_h_after.sum(),
                        mean_coverage_before=pp.coverage_fraction_before.mean(),mean_coverage_after=pp.coverage_fraction_after.mean()))
        for scheme,ctrl,history in [('geography',GEO,False),('state',STATE,True)]:
            a=aggregate_stage(d,var,ctrl,history);a['scheme']=scheme;a['variable']=var;a['unit']=unit;stage_model.append(a)
            for frm,to in PAIRS:
                models[(var,scheme,frm,to)]=make_pair(a,frm,to,ctrl)
    rs=pd.DataFrame(raw_rows);rs['p_BH18']=bh_adjust(rs.p_value);rs['BH_family_size']=18
    rs.to_csv(OUT/'raw_adjacent_contrasts.csv',index=False)
    pd.DataFrame(reconstructed).to_csv(OUT/'raw_stage_summary.csv',index=False)
    pd.concat(stages).to_parquet(OUT/'raw_stage_sid.parquet',index=False)
    pd.concat(raw_pairs).to_csv(OUT/'raw_adjacent_sid.csv',index=False)
    pd.concat(raw_draws).to_csv(OUT/'raw_adjacent_bootstrap.csv',index=False)
    pd.DataFrame(composition).to_csv(OUT/'basin_period_direction_check.csv',index=False)
    pd.concat(stage_model).to_parquet(OUT/'model_stage_sid.parquet',index=False)
    write_json('raw_reconstruction_receipt.json',dict(status='PASSED',variables=verify,rows=len(d),sid=len(sids),
        terminal_zero=int(d.terminal_point_v3.sum()),gap_zero=int(((gap!=6)&~d.terminal_point_v3).sum()),
        positive_legal_intervals=int((d.interval_hours_v3>0).sum()),raw_family_size=len(rs),spec_B_hash=bhash))
    print('RAW COMPLETE:24 accepted states reconstructed,18 matched contrasts; entering 36 support/refit targets',flush=True)
    all_adjust=[];all_members=[];all_ranges=[];all_details=[];all_draws=[]
    for number,((var,scheme,frm,to),p) in enumerate(models.items(),1):
        t0=time.time();ctrl=GEO if scheme=='geography' else STATE;retain=[] if scheme=='geography' else ['age_h']
        point=fit_pair(p,ctrl,retain=retain,details=True)
        identity=dict(variable=var,scheme=scheme,contrast=f'{to}-{frm}',stage_from=frm,stage_to=to,
                      unit=next(v['unit'] for v in conf['variables'] if v['variable']==var))
        member=p.copy();member['selected']=point.get('selected_mask',np.zeros(len(p),bool))
        member['exclusion_reasons']=point.get('exclusion_reasons',['NO_RESAMPLED_PAIRS']*len(p))
        for k,v in identity.items():member[k]=v
        all_members.append(member)
        for r in point['ranges']:all_ranges.append({**identity,**r})
        detail={**identity,**{k:v for k,v in point.items() if k not in ['selected_mask','exclusion_reasons']}}
        all_details.append(detail)
        ii=np.array([sid_index[s] for s in p.sid]);drawrows=[]
        for b in range(2000):
            fit=fit_pair(p,ctrl,weights=weights[b,ii],retain=retain)
            drawrows.append({**identity,'draw':b,**{k:fit.get(k,np.nan) for k in ['estimate','raw_matched_estimate',
                'n_pairs','n_unique','status','condition_number','rank','residual_df','target_identifiability_error']}})
        draws=pd.DataFrame(drawrows);good=np.isfinite(draws.estimate)
        out={**identity,**{k:point.get(k,np.nan) for k in ['estimate','raw_matched_estimate','before_matched_estimate',
            'after_matched_estimate','n_original_pairs','n_pairs','n_unique','retained_fraction','status','rank','residual_df',
            'condition_number','residual_sd','max_leverage','target_identifiability_error']},
            'ci_low':draws.loc[good,'estimate'].quantile(.025) if good.any() else np.nan,
            'ci_high':draws.loc[good,'estimate'].quantile(.975) if good.any() else np.nan,
            'bootstrap_attempted':2000,'bootstrap_valid':int(good.sum()),'bootstrap_failed':int((~good).sum()),
            'failure_reasons':json.dumps(draws.loc[~good,'status'].value_counts().to_dict()),
            'raw_on_support_ci_low':draws.raw_matched_estimate.quantile(.025),
            'raw_on_support_ci_high':draws.raw_matched_estimate.quantile(.975),
            'reference_sd':reference.loc[var,'reference_sd'],'target_age':'observed delta_age retained' if scheme=='state' else 'not adjusted'}
        # An unsupported point target remains NA even if some bootstrap samples
        # meet the gate. Draws are retained, not repurposed as a point interval.
        if not np.isfinite(out['estimate']):out['ci_low']=out['ci_high']=np.nan
        all_adjust.append(out);all_draws.append(draws)
        pd.DataFrame(all_adjust).to_csv(OUT/'adjusted_adjacent_summary.csv',index=False)
        draws.to_csv(OUT/f'bootstrap_{scheme}_{var}_{to}-{frm}.csv',index=False)
        write_json('analysis_progress.json',dict(status='RUNNING',completed_targets=number,total_targets=36,
            latest=identity,point_status=point['status'],valid_B=int(good.sum()),elapsed_seconds=time.time()-started))
        print(f'{number}/36 {scheme} {var} {to}-{frm} n={point["n_pairs"]} {point["status"]} Bvalid={good.sum()} seconds={time.time()-t0:.1f}',flush=True)
    pd.concat(all_members).to_csv(OUT/'adjustment_pair_membership.csv',index=False)
    pd.DataFrame(all_ranges).to_csv(OUT/'adjustment_support_ranges.csv',index=False)
    write_json('adjustment_model_details.json',all_details)
    pd.concat(all_draws).to_parquet(OUT/'adjustment_bootstrap_all.parquet',index=False)
    receipt=dict(status='COMPLETED',science_status='estimated and unsupported targets both retained',
        spec_B_hash=bhash,analysis_source_hash=sha(__file__),input_checks='input_hash_verification.json',
        raw_states=24,raw_contrasts=18,adjusted_targets=36,bootstrap_B=2000,seed=2026101102,
        bootstrap_bank_sha256=sha(OUT/'bootstrap_sid_multiplicities.npy'),
        utc_finished=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.time()-started,
        python=sys.executable,python_version=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
        raw_nc_reads=0,workers=1,native_threads=1)
    write_json('B_analysis_receipt.json',receipt);write_json('analysis_progress.json',receipt)
    print(json.dumps(receipt),flush=True)

if __name__=='__main__':execute()
