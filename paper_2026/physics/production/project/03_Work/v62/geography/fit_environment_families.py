"""Frozen C pooled pre-event models, on one measured-environment common set.

Regional and model-comparison intervals resample fixed OOF SID scores. They do
not repeat nuisance estimation and are not the global P3 full-refit intervals.
"""
from pathlib import Path
import os,sys,json,hashlib,pickle,datetime,traceback
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import numpy as np,pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits
threadpool_limits(1)
O=Path(__file__).resolve().parent;W6=O.parents[1]/'v6'
sys.path.insert(0,str(O.parent/'causal'))
from fit_policies_v6 import IntervalHazard
from geography_tables import read,bools,regions,sha

FAMILIES={'dynamics':['shear200_850','shear500_850','shear200_500','cyclonic_vo850'],'moisture':['midRH','q700_inner','q700_outer'],'thermodynamics':['SST','PI_headroom']}
ENV=sum(FAMILIES.values(),[])
BASE=['lat','lon_sin','lon_cos','month_sin','month_cos','wind_t0_kt','threshold_gap_kt','log_age','delta_v24_backward_kt','motion_speed_prev_ms','current_land_contact','past_land_contact']
BASINS=['EP','NA','WP','NI','SI','SP','SA','UNKNOWN'];BOXES=['SCS_box','Japan_Sea_box']
MODELS={'M0':[],**{'M0+'+k:v for k,v in FAMILIES.items()},'M0+all':ENV}
B=2000;SEED=62112026

def jdump(path,x):path.write_text(json.dumps(x,indent=2,ensure_ascii=False,default=lambda x:x.item() if hasattr(x,'item') else str(x)),encoding='utf-8')

def design_fit(e,columns):
    x=e[columns].to_numpy(float);med=np.nanmedian(x,axis=0);med=np.nan_to_num(med);missing=~np.isfinite(x)
    x=np.where(missing,med,x);mean=x.mean(0);sd=x.std(0);sd=np.where(sd>0,sd,1.)
    # Missing flags are allowed only for inherited M0 history fields. Every
    # primary environmental predictor was measured at >=3 actual pre-times.
    misscols=[k for k in BASE if e[k].isna().any()]
    trans=dict(columns=columns,medians=med.tolist(),means=mean.tolist(),scales=sd.tolist(),missing_indicator_columns=misscols,basin_reference=BASINS[0],basin_indicators=BASINS[1:],box_indicators=BOXES)
    names=['Intercept']+columns+['missing_'+k for k in misscols]+['basin_'+k for k in BASINS[1:]]+['box_'+k for k in BOXES]
    trans['design_columns']=names;return trans

def design_apply(e,tr):
    x=e[tr['columns']].to_numpy(float);x=(np.where(np.isfinite(x),x,np.array(tr['medians']))-np.array(tr['means']))/np.array(tr['scales'])
    a=[np.ones((len(e),1)),x]
    if tr['missing_indicator_columns']:a.append(e[tr['missing_indicator_columns']].isna().to_numpy(float))
    a+=[e.basin.eq(k).to_numpy(float)[:,None] for k in tr['basin_indicators']]
    a+=[e.region.eq(k).to_numpy(float)[:,None] for k in tr['box_indicators']]
    return np.column_stack(a)

def likelihood(model,X,L,U):
    l=np.asarray(L,int);u=np.where(np.isfinite(U),U,8).astype(int);ev=np.isfinite(U);n=len(X);ii=np.arange(n)
    eta=(X@model.coef[:model.p])[:,None]+np.r_[0,model.coef[model.p:]][None,:]
    ls=np.column_stack([np.zeros(n),-np.cumsum(np.logaddexp(0,eta),axis=1)]);ll=ls[ii,l].copy()
    ix=np.flatnonzero(ev);logratio=np.minimum(ls[ix,u[ix]]-ls[ix,l[ix]],-1e-15);ll[ix]+=np.log(-np.expm1(logratio))
    return ll

def fit_fold(tr,te,columns,model,fold):
    try:
        assert not set(tr.sid)&set(te.sid)
        assert int(np.isfinite(tr.U_h.to_numpy(float)).sum())>=20
        trans=design_fit(tr,columns);Xt=design_apply(tr,trans);Xv=design_apply(te,trans)
        hazard=IntervalHazard().fit(Xt,tr.L_h.to_numpy(float)/6,tr.U_h.to_numpy(float)/6)
        m=hazard.predict(Xv);ll=likelihood(hazard,Xv,te.L_h.to_numpy(float)/6,te.U_h.to_numpy(float)/6)
        assert np.isfinite(m).all() and np.isfinite(ll).all()
        return trans,hazard,m,ll
    except Exception:
        jdump(O/('environment_model_failure_'+model.replace('+','_')+'_fold'+str(fold)+'.json'),dict(status='FAILED_NO_REPLACEMENT',model=model,fold=int(fold),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),train_sid=tr.sid.tolist(),test_sid=te.sid.tolist(),train_indices=tr.index.tolist(),test_indices=te.index.tolist(),train_frame_sha256=hashlib.sha256(tr.to_csv(index=False).encode()).hexdigest(),test_frame_sha256=hashlib.sha256(te.to_csv(index=False).encode()).hexdigest(),code_sha256=sha(Path(__file__)),error=traceback.format_exc()))
        raise

def boot(v,mask,w,weight=None):
    v=np.asarray(v,float);mask=np.asarray(mask,bool)&np.isfinite(v)
    if weight is None:weight=np.ones(len(v))
    weight=np.asarray(weight,float);mask &=np.isfinite(weight)
    den=w[:,mask]@weight[mask];draw=np.full(len(w),np.nan);np.divide(w[:,mask]@(weight[mask]*v[mask]),den,out=draw,where=den>0)
    n=int(mask.sum());est=float(np.sum(v[mask]*weight[mask])/np.sum(weight[mask])) if n and np.sum(weight[mask])>0 else np.nan
    ci=np.nanquantile(draw,[.025,.975]) if n>=3 and np.isfinite(draw).any() else [np.nan,np.nan]
    return est,ci,draw

def merge_pre():
    receipts=[json.loads((O/f'pre_environment_receipt_s{i}of2.json').read_text()) for i in range(2)]
    assert all(x['status']=='COMPLETE' for x in receipts)
    d=pd.concat([read(O/f'pre_environment_sources_s{i}of2.csv') for i in range(2)],ignore_index=True)
    assert not d.duplicated(['request_id','variable']).any()
    cohort=read(O/'G3_departure_risk_SID_source.csv');expected=set(cohort.event_id)
    assert d.event_id.nunique()==len(expected) and set(d.event_id)==expected and d.request_id.nunique()==4*len(expected)
    assert set(d.lag_h)=={-24,-18,-12,-6}
    assert (pd.to_datetime(d.UTC,utc=True)<pd.to_datetime(d.t0,utc=True)).all()
    assert (pd.to_datetime(d.UTC,utc=True)==pd.to_datetime(d.t0,utc=True)+pd.to_timedelta(d.lag_h,unit='h')).all()
    keys=d.groupby(['sid','variable']).agg(n=('UTC','size'),unique_UTC=('UTC','nunique'),unique_lag=('lag_h','nunique'))
    assert keys.eq(4).all().all()
    meta=d[['sid','event_id','t0']].drop_duplicates();assert meta.sid.is_unique
    meta=meta.merge(cohort[['sid','event_id','t0']],on='sid',suffixes=('_request','_cohort'),validate='1:1')
    assert meta.event_id_request.eq(meta.event_id_cohort).all()
    assert pd.to_datetime(meta.t0_request,utc=True).eq(pd.to_datetime(meta.t0_cohort,utc=True)).all()
    # Check all published per-request components; workers reuse the genuine pilot.
    component_rows=[]
    for folder in d.source_folder.unique():
        folder=Path(folder);r=json.loads((folder/'COMPLETE.json').read_text())
        entry=dict(request_id=r['request']['request_id'],source_folder=str(folder),driver_sha256=r['driver_sha256'],COMPLETE_sha256=sha(folder/'COMPLETE.json'))
        for name,key in [('values.json','values_sha256'),('PI_trace.json','PI_trace_sha256'),('profiles.npz','profiles_sha256')]:
            assert sha(folder/name)==r[key],str(folder/name);entry[key]=r[key]
        component_rows.append(entry)
    pd.DataFrame(component_rows).to_csv(O/'pre_component_manifest.csv',index=False)
    d=d.sort_values(['sid','lag_h','variable']);d.to_csv(O/'pre_environment_sources.csv',index=False)
    a=d.groupby(['sid','variable']).value.agg(pre_mean='mean',valid_times='count').reset_index();a.loc[a.valid_times.lt(3),'pre_mean']=np.nan
    a.to_csv(O/'pre_environment_SID_means_long.csv',index=False)
    wide=a.pivot(index='sid',columns='variable',values='pre_mean').reset_index()
    cnt=a.pivot(index='sid',columns='variable',values='valid_times').add_suffix('_valid_times').reset_index()
    combined=wide.merge(cnt,on='sid',validate='1:1');combined.to_csv(O/'pre_environment_SID_means.csv',index=False)
    # Rename exposure columns from the old t0 cohort so they cannot substitute.
    cohort=cohort.rename(columns={k:k+'_t0_inherited' for k in ENV if k in cohort})
    allset=cohort.merge(combined,on='sid',validate='1:1');allset['environment_complete']=np.isfinite(allset[ENV].to_numpy(float)).all(1)
    allset.to_csv(O/'environment_model_all_risk_SID_source.csv',index=False)
    jdump(O/'pre_environment_merge_receipt.json',dict(status='MERGED_UNIQUE_REQUEST_AND_COMPONENT_SHA_VERIFIED',requests=d.request_id.nunique(),rows=len(d),sid=d.sid.nunique(),common_primary_environment_sid=int(allset.environment_complete.sum()),source_sha256={str(O/f'pre_environment_sources_s{i}of2.csv'):sha(O/f'pre_environment_sources_s{i}of2.csv') for i in range(2)},receipt_sources=receipts))
    return allset

def main():
    spec=json.loads((O.parent/'analysis_spec_v62.json').read_text());assert spec['module_C']['status']=='FROZEN_FOR_EXECUTION'
    allset=merge_pre();e=allset[allset.environment_complete].sort_values('sid').reset_index(drop=True)
    assert len(e)>=100 and e.sid.is_unique
    for k in ['current_land_contact','past_land_contact']:e[k]=pd.to_numeric(e[k],errors='coerce')
    e.to_csv(O/'environment_model_common_SID_source.csv',index=False)
    pred=read(W6/'causal/point_predictions.csv');pred=pred[pred.policy.eq('shear200_850')].set_index('sid').loc[e.sid]
    g=pred.g48.to_numpy(float);R=bools(pred.R48).to_numpy();Y=pred.Y48.to_numpy(float)
    assert np.array_equal(pred.index.to_numpy(),e.sid.to_numpy())
    assert np.array_equal(pred.event_id.to_numpy(),e.event_id.to_numpy())
    assert np.array_equal(pred.fold.to_numpy(int),e.fold.to_numpy(int))
    assert np.array_equal(R,bools(e.complete48).to_numpy())
    assert np.allclose(Y,e.recovery48.to_numpy(float),rtol=0,atol=0,equal_nan=True)
    assert np.isfinite(g).all() and ((g>0)&(g<=1)).all()
    weights=R*np.minimum(1/g,20);Y0=np.where(np.isfinite(Y),Y,0.)
    indices=np.random.default_rng(SEED).integers(0,len(e),size=(B,len(e)),dtype=np.int32)
    np.savez_compressed(O/'environment_model_SID_bootstrap_indices.npz',indices=indices,sid=e.sid.to_numpy(str))
    w=np.array([np.bincount(ix,minlength=len(e)) for ix in indices],dtype=np.int16)
    out=O/'environment_models';out.mkdir(exist_ok=True);scores=[];coeff=[];foldchecks=[]
    for model,extra in MODELS.items():
        columns=BASE+extra
        for fold in range(5):
            tr=e[e.fold.ne(fold)];te=e[e.fold.eq(fold)]
            trans,hazard,m,ll=fit_fold(tr,te,columns,model,fold)
            name=model.replace('+','_')+f'_fold{fold}'
            fit=dict(model=model,fold=int(fold),class_name='fit_policies_v6.IntervalHazard',penalty='0.5 * sum(theta_except_intercept**2); ridge strength 1',time_bins=8,hazard=hazard.coef.tolist(),p=hazard.p,transform=trans,optimization=hazard.optimization,train_sid=tr.sid.tolist(),test_sid=te.sid.tolist(),train_indices=tr.index.tolist(),test_indices=te.index.tolist(),formula='logit(h_j)=X beta + alpha_j; alpha_1=0; risk=1-product(1-h_j)')
            jdump(out/(name+'.json'),fit)
            with (out/(name+'.pkl')).open('wb') as f:pickle.dump(fit,f,protocol=5)
            for k,value in zip(trans['design_columns']+['hazard_bin'+str(j) for j in range(2,9)],hazard.coef):
                scale='intercept' if k=='Intercept' else 'logit hazard offset relative to first 6h bin' if k.startswith('hazard_bin') else 'training-fold SD (including binary land-contact history)' if k in columns else 'unstandardized binary basin/box/missingness indicator'
                coeff.append(dict(model=model,fold=fold,covariate=k,coefficient=value,scale=scale))
            for z,prediction,lik in zip(te.itertuples(),m,ll):scores.append(dict(sid=z.sid,fold=fold,model=model,m48=prediction,observed_interval_log_likelihood=lik,L_h=z.L_h,U_h=z.U_h,zero_likelihood_information=bool(not np.isfinite(z.U_h) and z.L_h==0)))
            foldchecks.append(dict(model=model,fold=fold,training_sid=len(tr),test_sid=len(te),training_interval_events=int(np.isfinite(tr.U_h.to_numpy(float)).sum()),SID_overlap=0,optimization=hazard.optimization))
            print(model,fold,'complete',len(tr),len(te),flush=True)
    s=pd.DataFrame(scores);s=s.merge(e[['sid','basin','region','recovery48','complete48']],on='sid',validate='m:1')
    observation=pd.DataFrame(dict(sid=e.sid,g48=g,R48=R,Y48=Y,observation_weight=weights))
    s=s.merge(observation,on='sid',validate='m:1');s['squared_error']=(s.m48-s.Y48.fillna(0))**2;s['weighted_Brier_numerator']=s.squared_error*s.observation_weight
    s.to_csv(O/'environment_model_OOF_scores.csv',index=False)
    pd.DataFrame(coeff).to_csv(O/'environment_model_coefficients.csv',index=False);jdump(O/'environment_model_fold_checks.json',foldchecks)
    measures=[];comparisons=[];regional=[];cal=[];drawcols={};base=s[s.model.eq('M0')].set_index('sid').loc[e.sid];baseLL=base.observed_interval_log_likelihood.to_numpy();baseB=(base.m48.to_numpy()-Y0)**2
    for model in MODELS:
        a=s[s.model.eq(model)].set_index('sid').loc[e.sid];m=a.m48.to_numpy();ll=a.observed_interval_log_likelihood.to_numpy();br=(m-Y0)**2
        for metric,value,wt in [('interval_log_likelihood',ll,None),('complete_observation_IPCW_Brier',br,weights)]:
            estimate,ci,draw=boot(value,np.ones(len(e),bool),w,wt);measures.append(dict(model=model,metric=metric,n_sid=len(e),n_observed_positive_weight=int((weights>0).sum()) if wt is not None else np.nan,weight_sum=float(weights.sum()) if wt is not None else np.nan,weight_ESS=float(weights.sum()**2/np.square(weights).sum()) if wt is not None else np.nan,estimate=estimate,ci_low=ci[0],ci_high=ci[1],B_valid=int(np.isfinite(draw).sum()),inference='2000 fixed-OOF SID bootstrap, conditional on fits'));drawcols[model+'|'+metric]=draw
        for metric,value,wt,better in [('interval_log_likelihood',ll-baseLL,None,'positive'),('complete_observation_IPCW_Brier',br-baseB,weights,'negative')]:
            estimate,ci,draw=boot(value,np.ones(len(e),bool),w,wt);comparisons.append(dict(model=model,reference='M0',metric=metric,n_sid=len(e),n_observed_positive_weight=int((weights>0).sum()) if wt is not None else np.nan,weight_sum=float(weights.sum()) if wt is not None else np.nan,weight_ESS=float(weights.sum()**2/np.square(weights).sum()) if wt is not None else np.nan,paired_difference=estimate,ci_low=ci[0],ci_high=ci[1],B_valid=int(np.isfinite(draw).sum()),improvement_direction=better,inference='paired 2000 fixed-OOF SID bootstrap'));drawcols[model+'|minus_M0|'+metric]=draw
        for region,mask in regions(e):
            estimate,ci,draw=boot(m*100,mask,w);regional.append(dict(model=model,region=region,n_sid=int(mask.sum()),risk_percent=estimate,ci_low=ci[0],ci_high=ci[1],B_valid=int(np.isfinite(draw).sum()),inference='2000 fixed-OOF SID bootstrap, conditional on fits'));drawcols[model+'|risk|'+region]=draw
        decile=pd.qcut(m,10,labels=False,duplicates='drop')
        for dec in sorted(set(decile)):
            mask=(decile==dec)&R;ww=weights[mask];cal.append(dict(model=model,decile=int(dec)+1,n_all=int((decile==dec).sum()),n_complete=int(mask.sum()),weight_sum=ww.sum(),predicted=float(np.average(m[mask],weights=ww)) if ww.sum()>0 else np.nan,observed=float(np.average(Y0[mask],weights=ww)) if ww.sum()>0 else np.nan))
    for name,rows in [('environment_model_metrics',measures),('environment_model_paired_comparisons',comparisons),('environment_model_region_risk',regional),('environment_model_calibration',cal)]:pd.DataFrame(rows).to_csv(O/(name+'.csv'),index=False)
    pd.DataFrame(drawcols).to_csv(O/'environment_model_fixed_OOF_bootstrap_draws.csv',index_label='draw')
    composition=[];season=[];history=[]
    for pop,z in [('full_risk',allset),('common_environment',e)]:
        for region,mask in regions(z):
            q=z[mask];composition.append(dict(population=pop,region=region,n_sid=len(q),known_return=int(q.recovery48.eq(1).sum()),known_nonreturn=int(q.recovery48.eq(0).sum()),unknown=int(q.recovery48.isna().sum()),median_age_h=q.age_h.median(),median_wind_t0=q.wind_t0_kt.median(),median_lat=q.lat.median()))
            months=pd.to_datetime(q.t0,utc=True).dt.month
            for month in range(1,13):season.append(dict(population=pop,region=region,month=month,season={12:'DJF',1:'DJF',2:'DJF',3:'MAM',4:'MAM',5:'MAM',6:'JJA',7:'JJA',8:'JJA',9:'SON',10:'SON',11:'SON'}[month],n_sid=int(months.eq(month).sum()),fraction=float(months.eq(month).mean()) if len(q) else np.nan))
            for k in BASE:
                a=pd.to_numeric(q[k],errors='coerce');history.append(dict(population=pop,region=region,covariate=k,n_sid=len(q),n_measured=int(a.notna().sum()),n_missing=int(a.isna().sum()),mean=a.mean(),median=a.median(),q25=a.quantile(.25),q75=a.quantile(.75)))
    pd.DataFrame(composition).to_csv(O/'environment_common_population_composition.csv',index=False)
    pd.DataFrame(season).to_csv(O/'environment_common_population_month_season.csv',index=False)
    pd.DataFrame(history).to_csv(O/'environment_common_population_history.csv',index=False)
    jdump(O/'environment_family_fit_receipt.json',dict(status='COMPLETE_PENDING_REVIEW',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),common_sid=len(e),full_risk_sid=len(allset),interval_events=int(np.isfinite(e.U_h.to_numpy(float)).sum()),early_returns_incomplete=int((e.recovery48.eq(1)&~bools(e.complete48)).sum()),zero_likelihood_information=int((~np.isfinite(e.U_h.to_numpy(float))&e.L_h.eq(0)).sum()),complete48=int(R.sum()),models=len(MODELS)*5,bootstrap_draws=B,model_family=list(MODELS),environment_common_columns=ENV,global_P3_fits_unchanged=True,observation_model='Inherited shear-branch OOF probability, complete48 indicator independent of outcome; weight cap20',Brier='Hajek weighted mean among complete outcomes; early incomplete events remain in hazard likelihood',inference='All intervals conditional on fixed cross-fitted models; no mediation interpretation',source_sha256={str(p):sha(p) for p in [Path(__file__),O/'pre_environment_sources.csv',O/'environment_model_common_SID_source.csv',O.parent/'analysis_spec_v62.json',W6/'causal/point_predictions.csv']}))
    print('C environment family model comparison complete',len(e),flush=True)

if __name__=='__main__':main()
