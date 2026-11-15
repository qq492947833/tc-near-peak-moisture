"""Frozen-design background adjustment for six new same-wind layer diagnostics.

All writes are confined to the independent run. Original scientific functions
are copied verbatim in _adjusted_legacy_helpers.py with hashed provenance.
"""
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'
from pathlib import Path
import argparse, json, hashlib, time
import numpy as np
import pandas as pd
from _adjusted_legacy_helpers import (
    equal_weights, summarize, raw_calc, adaptive, balance,
    target_estimable, bh_fixed_family,
)

from configuration import OUTPUT_ROOT as ROOT, BACKGROUND_ROOT as OLD, initialize_output

SEED=650117
REPS=2000
JOBS=[{'window':w,'metric':m,'units':'W kg-1','level':'500-850','region':'r0_300'}
      for w in (0,1) for m in ('Gamma_K_total','Gamma_A_total','C_A_to_K')]
# These eight columns only audit additivity. They do not add statistical tests.
QA_JOBS=[{'window':w,'metric':f'Gamma_{energy}_{part}','units':'W kg-1','level':'500-850','region':'r0_300'}
         for w in (0,1) for energy in ('K','A') for part in ('h','p')]

def _clean(v):
    if isinstance(v,dict):return {str(k):_clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple,np.ndarray)):return [_clean(x) for x in v]
    if isinstance(v,np.generic):return _clean(v.item())
    if isinstance(v,Path):return str(v)
    if isinstance(v,float) and not np.isfinite(v):return None
    return v

def _write_json(path, obj):
    path=Path(path).resolve()
    if not path.is_relative_to(ROOT):raise ValueError('Output escapes independent run')
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(_clean(obj),ensure_ascii=False,indent=2),encoding='utf-8')
    os.replace(temp,path)

def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()

def load_frozen_background():
    """Read unchanged matches and already audited design columns; no refitting."""
    p=pd.read_parquet(OLD/'results/matched_covariates.parquet').sort_values(['sid','event_id','pair_id']).reset_index(drop=True)
    names=[c.removeprefix('decay__') for c in p if c.startswith('decay__')]
    A=p[['decay__'+c for c in names]].to_numpy(float)
    C=p[['rise__'+c for c in names]].to_numpy(float)
    assert len(p)==165 and p.sid.nunique()==108 and p.event_id.nunique()==115
    return p,A,C,names

def _diagnostic_extra(out,A,C,sid,event,mult):
    """Record the exact test used by the original helper, without changing it."""
    if 'original_indices' not in out:return {}
    k=np.asarray(out['original_indices'],int)
    sw=equal_weights(sid[k],event[k],mult)
    x=np.column_stack([np.r_[np.zeros(len(k)),np.ones(len(k))],np.vstack([A[k],C[k]])])
    w=np.r_[sw/2,sw/2];c=np.zeros(x.shape[1]);c[0]=1
    t=target_estimable(x,c,w)
    scale=np.linalg.norm(np.sqrt(w)[:,None]*x,axis=0);scale=np.where(scale>0,scale,1)
    residual_tol=64*np.finfo(float).eps*max(x.shape)*max(1.,np.linalg.norm(c/scale))
    return {'design_columns':x.shape[1],'rank':t['rank'],'phase_target_estimable':t['estimable'],
            'estimability_residual':t['projection_error'],'estimability_tolerance':residual_tol,
            'svd_tolerance':t['singular_tolerance'],'full_rank_coefficients':t['rank']==x.shape[1]}

def validate_original_background(draw_ids=(0,1,100,999,1999)):
    """Reproduce old 16-d point and selected stored draws using exact helpers."""
    p,A,C,names=load_frozen_background()
    data=pd.read_parquet(OLD/'results/request_values.parquet')
    y0=[];y1=[]
    old_jobs=[{'window':w,'metric':m} for w in (0,1) for m in ('APE','C_A_to_K')]
    for job in old_jobs:
        v=data.loc[data.slow_boundary_days.eq(16)&data.window.eq(job['window'])&data.metric.eq(job['metric'])].set_index('request_id').value
        y0.append(v.reindex(p.decay_request_id).to_numpy());y1.append(v.reindex(p.rise_request_id).to_numpy())
    y0=np.array(y0).T;y1=np.array(y1).T
    sid=p.sid.to_numpy(str);ev=p.event_id.to_numpy(str);cat=p.basin_genesis.to_numpy(str)
    point=adaptive(sid,ev,cat,A,C,y0,y1,dict.fromkeys(np.unique(sid),1))
    original=json.loads((OLD/'results/BACKGROUND_POINT.json').read_text())
    errors=[]
    for j,job in enumerate(old_jobs):
        for field in ('estimate','raw_same_support'):
            actual=point[field][j];expected=original[field][j+4]
            errors.append(dict(kind='point',field=field,**job,expected=expected,actual=actual,absolute_error=abs(actual-expected),passed=bool(np.isclose(actual,expected,rtol=1e-10,atol=1e-12))))
    support={k:point[k] for k in ('n_sid_support','n_events_support','n_bin_pairs_support')}
    assert support=={'n_sid_support':105,'n_events_support':112,'n_bin_pairs_support':162}
    old_boot=np.load(OLD/'cache/bootstrap_background.npz')
    unique=np.unique(sid)
    generated=np.random.default_rng(SEED).integers(0,len(unique),size=(REPS,len(unique)))
    assert np.array_equal(generated,old_boot['sid_indices'])
    for draw in draw_ids:
        mult=dict(zip(unique,np.bincount(generated[draw],minlength=len(unique))))
        out=adaptive(sid,ev,cat,A,C,y0,y1,mult)
        assert out['success']
        for j,job in enumerate(old_jobs):
            for field,key in [('estimate','adjusted'),('raw_same_support','raw')]:
                actual=out[field][j];expected=old_boot[key][draw,j+4]
                errors.append(dict(kind='draw',draw_id=draw,field=field,**job,expected=expected,actual=actual,absolute_error=abs(actual-expected),passed=bool(np.isclose(actual,expected,rtol=1e-10,atol=1e-12))))
    result={'passed':all(x['passed'] for x in errors),'support':support,'checks':len(errors),'draw_ids':draw_ids,'maximum_absolute_error':max(x['absolute_error'] for x in errors),'errors':errors,'source':str(OLD),'covariates':names,'sid_draw_indices_equal':True}
    _write_json(ROOT/'diagnostics/adjusted_baseline_validation.json',result)
    assert result['passed'],'Frozen adjusted baseline mismatch'
    return result

def load_new_outcomes(request_values,pairs,jobs=None):
    """Bind actual SID/UTC fields to frozen request identities, never rematch."""
    jobs=JOBS if jobs is None else jobs
    d=pd.read_parquet(request_values)
    if 'variant' in d:d=d[d.variant.eq('primary')]
    d=d[d.level.astype(str).eq('500-850')&d.region.eq('r0_300')&d.window.isin([0,1])&d.metric.isin([j['metric'] for j in jobs])].copy()
    if 'scheme' in d:d=d[d.scheme.eq('main')]
    d['UTC']=pd.to_datetime(d.UTC,utc=True);d['sid']=d.sid.astype(str)
    req=pd.read_parquet(OLD/'inputs_requests.parquet')[['request_id','sid','UTC']].copy()
    req['UTC']=pd.to_datetime(req.UTC,utc=True);req['sid']=req.sid.astype(str)
    y0=[];y1=[]
    for job in jobs:
        v=d[d.window.eq(job['window'])&d.metric.eq(job['metric'])][['sid','UTC','value']]
        if v.duplicated(['sid','UTC']).any():raise ValueError('New request outcomes have duplicate SID/UTC/diagnostic rows')
        z=req.merge(v,on=['sid','UTC'],how='left',validate='many_to_one').set_index('request_id').value
        y0.append(z.reindex(pairs.decay_request_id).to_numpy(float));y1.append(z.reindex(pairs.rise_request_id).to_numpy(float))
    return np.array(y0).T,np.array(y1).T

def transfer_algebra(values):
    """Check four total=h+p identities on one shared-support outcome matrix."""
    v=np.asarray(values,float);all_jobs=JOBS+QA_JOBS
    errors=[];labels=[];tolerances=[]
    for w in (0,1):
        for energy in ('K','A'):
            indices=[next(i for i,j in enumerate(all_jobs) if j['window']==w and j['metric']==f'Gamma_{energy}_{part}') for part in ('total','h','p')]
            total,horiz,vert=[v[...,i] for i in indices]
            errors.append(total-horiz-vert)
            tolerances.append(1e-12+1e-10*np.maximum(abs(total),abs(horiz)+abs(vert)))
            labels.append(f'window{w}_Gamma_{energy}_total_minus_h_minus_p')
    return np.stack(errors,axis=-1),np.stack(tolerances,axis=-1),labels

def _algebra_report(values,scope):
    error,tolerance,labels=transfer_algebra(values)
    finite=np.isfinite(error)&np.isfinite(tolerance)
    return {'scope':scope,'units':'W kg-1','labels':labels,
            'n_checked':int(finite.sum()),'n_failed':int((finite&(abs(error)>tolerance)).sum()),
            'status':'NOT_EVALUABLE' if not finite.any() else ('PASS' if not (finite&(abs(error)>tolerance)).any() else 'FAIL'),
            'maximum_absolute_error':float(np.max(abs(error[finite]))) if finite.any() else None,
            'absolute_tolerance':1e-12,'relative_tolerance':1e-10}

def _raw_statistics(sid,event,y0,y1):
    if np.count_nonzero(np.isfinite(y0)&np.isfinite(y1))==0:
        return dict(**summarize(np.nan,[]),n_sid=0,n_events=0,n_bin_pairs=0,rise_mean=np.nan,decay_mean=np.nan,sid_ESS=0,row_ESS=0),np.full(REPS,np.nan)
    return raw_calc(sid,event,y0,y1)

def _old_support_comparison(pairs,common,conditional):
    data=pd.read_parquet(OLD/'results/request_values.parquet')
    sid=pairs.sid.to_numpy(str);event=pairs.event_id.to_numpy(str)
    rows=[]
    for w in (0,1):
        for metric in ('APE','C_A_to_K'):
            v=data[data.slow_boundary_days.eq(16)&data.window.eq(w)&data.metric.eq(metric)].set_index('request_id').value
            a=v.reindex(pairs.decay_request_id).to_numpy();b=v.reindex(pairs.rise_request_id).to_numpy()
            for label,mask in [('old_original_support',np.ones(len(pairs),bool)),('old_new_common_support',common),('old_new_conditional_support',conditional)]:
                stats,_=_raw_statistics(sid[mask],event[mask],a[mask],b[mask])
                rows.append(dict(window=w,metric=metric,level='700',region='r0_300',target=label,units='J kg-1' if metric=='APE' else 'W kg-1',**stats,source_file=str(OLD/'results/request_values.parquet')))
    return pd.DataFrame(rows)

def run_adjusted(request_values,output_root=None):
    output=Path(output_root) if output_root else ROOT/'statistics/adjusted'
    output=output.resolve()
    if not output.is_relative_to(ROOT):raise ValueError('Outputs must stay inside this run')
    output.mkdir(parents=True,exist_ok=True)
    p,A,C,cov_names=load_frozen_background()
    sid=p.sid.to_numpy(str);ev=p.event_id.to_numpy(str);cats=p.basin_genesis.to_numpy(str)
    y0,y1=load_new_outcomes(request_values,p)
    common=np.isfinite(y0).all(axis=1)&np.isfinite(y1).all(axis=1)
    background=common&np.isfinite(A).all(axis=1)&np.isfinite(C).all(axis=1)
    if len(np.unique(sid[background]))<2:raise ValueError('Fewer than two background-complete SIDs')
    q0,q1=load_new_outcomes(request_values,p,QA_JOBS)
    if not (np.isfinite(q0[common]).all() and np.isfinite(q1[common]).all()):
        raise ValueError('Horizontal/vertical QA components missing on six-target support; do not silently restrict support')
    all0=np.column_stack([y0,q0]);all1=np.column_stack([y1,q1])
    input_qa=[_algebra_report(all0[common],'matched_decay_inputs'),_algebra_report(all1[common],'matched_rise_inputs')]
    _write_json(output/'INPUT_TRANSFER_ALGEBRA.json',input_qa)
    if any(r['n_failed'] for r in input_qa):raise ValueError('Input total != horizontal+vertical on common support')
    rows=[];draws={}
    for j,job in enumerate(JOBS):
        for label,mask in [('raw_all_original_matches',np.ones(len(p),bool)),('raw_common_six_support',common),('raw_background_complete',background)]:
            rec,draw=_raw_statistics(sid[mask],ev[mask],y0[mask,j],y1[mask,j])
            rows.append(dict(job_id=j,target=label,**job,**rec));draws[f'{label}_{j}']=draw
    s=sid[background];e=ev[background];cat=cats[background];aa=A[background];cc=C[background];a=all0[background];b=all1[background]
    unique=np.unique(s)
    point=adaptive(s,e,cat,aa,cc,a,b,dict.fromkeys(unique,1))
    point.update(_diagnostic_extra(point,aa,cc,s,e,dict.fromkeys(unique,1)))
    point_saved=dict(point)
    for key in ('estimate','raw_same_support'):
        if key in point_saved:
            point_saved['qa_'+key]=np.asarray(point_saved[key])[len(JOBS):]
            point_saved[key]=np.asarray(point_saved[key])[:len(JOBS)]
    point_saved['qa_jobs']=QA_JOBS
    _write_json(output/'BACKGROUND_POINT.json',point_saved)
    index=np.random.default_rng(SEED).integers(0,len(unique),size=(REPS,len(unique)))
    binding={'request_values_path':str(Path(request_values).resolve()),'sha256':_sha(request_values),'jobs':JOBS,'qa_jobs':QA_JOBS,'sid_population':unique.tolist(),'covariates':cov_names,'seed':SEED,'reps':REPS,'old_covariates_sha256':_sha(OLD/'results/matched_covariates.parquet')}
    bindpath=output/'INPUT_BINDING.json'
    cache=output/'bootstrap_background.npz';recordpath=output/'bootstrap_records.json'
    adjusted=np.full((REPS,len(JOBS)+len(QA_JOBS)),np.nan);raw=np.full_like(adjusted,np.nan);records=[]
    if cache.exists():
        if not bindpath.exists() or json.loads(bindpath.read_text())!=binding:raise ValueError('Cannot resume: source binding changed')
        z=np.load(cache);adjusted=np.column_stack([z['adjusted'],z['qa_adjusted']]);raw=np.column_stack([z['raw'],z['qa_raw']]);records=json.loads(recordpath.read_text());assert np.array_equal(index,z['sid_indices'])
    else:_write_json(bindpath,binding)
    for draw in range(len(records),REPS):
        mult=dict(zip(unique,np.bincount(index[draw],minlength=len(unique))))
        try:
            out=adaptive(s,e,cat,aa,cc,a,b,mult)
            if 'raw_same_support' in out:raw[draw]=out['raw_same_support']
            if out['success']:adjusted[draw]=out['estimate']
            rec={k:out.get(k) for k in ['success','reason','n_sid_support','n_events_support','n_bin_pairs_support','iterations','rank','supported_basins','unsupported_basins']}
            rec.update(_diagnostic_extra(out,aa,cc,s,e,mult));rec['draw_id']=draw
            records.append(rec)
        except Exception as ex:records.append({'draw_id':draw,'success':False,'reason':type(ex).__name__+': '+str(ex)})
        if (draw+1)%100==0 or draw==REPS-1:
            with (output/'bootstrap_background.tmp').open('wb') as f:np.savez_compressed(f,adjusted=adjusted[:,:len(JOBS)],raw=raw[:,:len(JOBS)],qa_adjusted=adjusted[:,len(JOBS):],qa_raw=raw[:,len(JOBS):],sid_indices=index,sid_population=unique)
            os.replace(output/'bootstrap_background.tmp',cache)
            _write_json(recordpath,records)
            _write_json(ROOT/'checkpoints/ADJUSTED_STATISTICS.json',{'state':'RUNNING','completed_draws':draw+1,'attempted':REPS,'failures':sum(not x['success'] for x in records),'output':str(output)})
            print(f'ADJUSTED {draw+1}/{REPS}, failures={sum(not x["success"] for x in records)}',flush=True)
    for j,job in enumerate(JOBS):
        rp=np.asarray(point.get('raw_same_support',np.full(len(JOBS),np.nan)))[j]
        ap=np.asarray(point.get('estimate',np.full(len(JOBS),np.nan)))[j]
        for target,est,d in [('raw_same_supported_background',rp,raw[:,j]),('background_adjusted',ap,adjusted[:,j]),('adjusted_minus_raw_same_support',ap-rp,adjusted[:,j]-raw[:,j])]:
            w=np.asarray(point.get('pair_weights',[]),float)
            rows.append(dict(job_id=j,target=target,**job,**summarize(est,d),n_sid=point.get('n_sid_support',0),n_events=point.get('n_events_support',0),n_bin_pairs=point.get('n_bin_pairs_support',0),sid_ESS=point.get('n_sid_support',0),row_ESS=w.sum()**2/(w*w).sum() if len(w) else np.nan))
            draws[f'{target}_{j}']=d
    frame=pd.DataFrame(rows);frame['q_BH']=np.nan;frame['family_id']='descriptive_support_comparison'
    for target,family in [('raw_all_original_matches','MWT_vertical_wind_raw_6'),('background_adjusted','MWT_vertical_wind_adjusted_6'),('adjusted_minus_raw_same_support','MWT_vertical_wind_adjustment_change_6')]:
        select=frame.target.eq(target);assert select.sum()==6
        frame.loc[select,'q_BH']=bh_fixed_family(frame.loc[select,'p_centered'],6);frame.loc[select,'family_id']=family
    frame['n_attempted']=REPS;frame['status']=np.where(np.isfinite(frame.estimate),'ESTIMATED','NOT_ESTIMABLE')
    frame['comparison']='reintensification minus decay at frozen same-SID/shared-wind-bin representatives'
    frame['source_file']=str(Path(request_values).resolve());frame['interval_method']='SID percentile 95%; support and outcomes refitted every draw'
    frame.to_csv(output/'ADJUSTED_WIND_MATCHED.csv',index=False,encoding='utf-8-sig')
    frame.to_parquet(output/'ADJUSTED_WIND_MATCHED.parquet',index=False)
    np.savez_compressed(output/'TARGET_DRAWS.npz',**draws)
    algebra=[]
    for scope,values in [('adjusted_draws',adjusted),('raw_same_support_draws',raw),('adjustment_change_draws',adjusted-raw)]:
        errors,tolerances,labels=transfer_algebra(values)
        algebra.append(_algebra_report(values,scope))
        pd.DataFrame({'draw_id':np.arange(REPS),**{labels[k]:errors[:,k] for k in range(4)},'maximum_absolute_error':np.max(np.abs(errors),axis=1),'all_four_finite':np.isfinite(errors).all(axis=1),'all_within_tolerance':(np.isfinite(errors)&(np.abs(errors)<=tolerances)).all(axis=1),'supported_sid_count':[r.get('n_sid_support') for r in records],'supported_event_count':[r.get('n_events_support') for r in records],'supported_matched_units':[r.get('n_bin_pairs_support') for r in records]}).to_csv(output/f'ALGEBRA_{scope}.csv',index=False,encoding='utf-8-sig')
    for field in ('estimate','raw_same_support'):
        if field in point:algebra.append(_algebra_report(point[field],'point_'+field))
    _write_json(output/'TRANSFER_ALGEBRA_QA.json',{'passed':all(x['status']=='PASS' for x in input_qa+algebra),'checks':input_qa+algebra,'support_defined_by':'six primary targets only; eight auxiliary outcomes use exactly the same retained rows, weights, fitted design and SID draw','auxiliary_jobs':QA_JOBS,'additional_hypothesis_tests':0})
    np.savez_compressed(output/'AUXILIARY_TRANSFER_DRAWS.npz',adjusted=adjusted[:,len(JOBS):],raw_same_support=raw[:,len(JOBS):],adjustment_change=adjusted[:,len(JOBS):]-raw[:,len(JOBS):],metric=np.array([j['metric'] for j in QA_JOBS]),window=np.array([j['window'] for j in QA_JOBS]),sid_indices=index,sid_population=unique)
    if any(x['n_failed'] for x in algebra):raise AssertionError('Adjusted or raw transfer additivity failed')
    pd.DataFrame(records).to_csv(output/'bootstrap_fit_status.csv',index=False,encoding='utf-8-sig')
    support=p.copy();support['new_six_common_finite']=common;support['background_complete']=background;support['new_conditional_support']=False
    if 'original_indices' in point:support.loc[np.flatnonzero(background)[point['original_indices']],'new_conditional_support']=True
    support['exclusion_reason']=np.select([~common,common&~background,background&~support.new_conditional_support],['NEW_DIAGNOSTIC_NONFINITE','BACKGROUND_MISSING','OUTSIDE_ADAPTIVE_SUPPORT'],default='RETAINED')
    support.to_csv(output/'matched_support.csv',index=False,encoding='utf-8-sig')
    _old_support_comparison(p,common,support.new_conditional_support.to_numpy()).to_csv(output/'OLD_METRICS_ON_NEW_SUPPORT.csv',index=False,encoding='utf-8-sig')
    balance_rows=[]
    for label,ok in [('complete_background',background),('conditional_support',support.new_conditional_support.to_numpy())]:
        if not ok.any():continue
        w=equal_weights(sid[ok],ev[ok]);ma,mc,smd=balance(A[ok],C[ok],w)
        for i,name in enumerate(cov_names):balance_rows.append(dict(population=label,covariate=name,decay_mean=ma[i],rise_mean=mc[i],SMD_rise_minus_decay=smd[i]))
    pd.DataFrame(balance_rows).to_csv(output/'covariate_balance.csv',index=False,encoding='utf-8-sig')
    status={'state':'COMPUTED_PENDING_QA','n_original_sid':108,'n_original_events':115,'n_original_units':165,'n_common_units':int(common.sum()),'n_background_units':int(background.sum()),'point_support':{k:point.get(k) for k in ['n_sid_support','n_events_support','n_bin_pairs_support']},'attempted_draws':REPS,'successful_adjusted_draws':int(np.isfinite(adjusted[:,:len(JOBS)]).all(axis=1).sum()),'failed_adjusted_draws':int((~np.isfinite(adjusted[:,:len(JOBS)]).all(axis=1)).sum()),'full_rank_draws':sum(bool(r.get('full_rank_coefficients')) for r in records),'transfer_algebra_passed':all(x['status']=='PASS' for x in input_qa+algebra),'auxiliary_transfer_hypothesis_tests':0,'output':str(output)}
    _write_json(output/'STATUS.json',status);_write_json(ROOT/'checkpoints/ADJUSTED_STATISTICS.json',status)
    return frame,status

if __name__=='__main__':
    initialize_output()
    parser=argparse.ArgumentParser();parser.add_argument('--validate-baseline',action='store_true');parser.add_argument('--request-values');parser.add_argument('--output')
    args=parser.parse_args()
    for name in ('TEMP','TMP','TMPDIR','MPLCONFIGDIR'):os.environ[name]=str(ROOT/'tmp')
    if args.validate_baseline:print(json.dumps(_clean(validate_original_background()),ensure_ascii=False))
    if args.request_values:
        _,status=run_adjusted(args.request_values,args.output);print(json.dumps(_clean(status),ensure_ascii=False))
    if not args.validate_baseline and not args.request_values:parser.error('Choose --validate-baseline or --request-values')
