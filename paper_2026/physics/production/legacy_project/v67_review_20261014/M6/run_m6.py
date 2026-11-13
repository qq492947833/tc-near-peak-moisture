from _release_paths import release_path as _release_path
"""One frozen equal-wind sensitivity; same-SID inference; no raw ERA5 access."""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import numpy as np,pandas as pd,json,hashlib,sys,datetime,platform
B=Path(__file__).resolve().parent
V=Path(str(_release_path('mechanism_v5','')))
JOBS={p:V/('p45_nonpi_r1/jobs/j'+str(j)) for p,j in [('midRH',21),('q700_storm_relative_horizontal_advection',22),('q850_moisture_flux_convergence',23),('shear200_500',24)]}
JOBS['PI_headroom']=V/'upper_profile_100_50_interpolation_v1/p45_execution_v2/piC/attempt_0001/jobs/j25'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(path,x):path.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False,default=str)+'\n',encoding='utf8')
def read(p):return pd.read_csv(p,keep_default_na=False,na_values=['','nan','NaN'])
def save(path,d):d.to_csv(path,index=False,encoding='utf-8-sig')
def interval(x):
    q=x[np.isfinite(x)]
    return [float(v) for v in np.percentile(q,[2.5,97.5])] if len(q) else [None,None]
def sample(sids):
    rng=np.random.default_rng(670620026);ind=rng.integers(0,len(sids),size=(2000,len(sids)),dtype=np.int32)
    return ind
def stats(sids,X,y):
    unique,ix=np.unique(sids,return_inverse=True);n=np.bincount(ix);w=1/n[ix]
    scales=np.sqrt(np.average(X*X,axis=0,weights=w));keep=scales>0
    # Only identicallyzero columns are omitted. Deterministic scaling is reversible.
    Z=X[:,keep]/scales[keep];p=Z.shape[1]
    G=np.zeros((len(unique),p,p));b=np.zeros((len(unique),p));yy=np.zeros(len(unique));sy=np.zeros(len(unique))
    np.add.at(G,ix,w[:,None,None]*Z[:,:,None]*Z[:,None,:]);np.add.at(b,ix,w[:,None]*Z*y[:,None]);np.add.at(yy,ix,w*y*y);np.add.at(sy,ix,w*y)
    a=int(np.flatnonzero(np.flatnonzero(keep)==1)[0]);return {'sid':unique,'G':G,'b':b,'yy':yy,'sy':sy,'scales':scales,'keep':keep,'a':a,'X':X,'y':y,'w':w}
def fit(g,b,yy,sy,n,a,ascale):
    eig,U=np.linalg.eigh(g);tol=max(float(eig.max()),1)*1e-12;valid=eig>tol
    c=np.zeros(len(b));c[a]=1;error=np.linalg.norm(c-U[:,valid]@(U[:,valid].T@c))
    if error>1e-8:return np.nan,np.nan,np.nan,int(valid.sum()),'target_not_estimable'
    beta=U[:,valid]@((U[:,valid].T@b)/eig[valid]);sse=yy-float(beta@b);sst=yy-sy*sy/n
    other=np.arange(len(b))!=a;rg=g[np.ix_(other,other)];rb=b[other];re,ru=np.linalg.eigh(rg);rv=re>max(float(re.max()),1)*1e-12
    betar=ru[:,rv]@((ru[:,rv].T@rb)/re[rv]);red=yy-float(betar@rb)
    if sst<=0 or not np.isfinite(beta).all() or sse< -1e-7*max(yy,1):return np.nan,np.nan,np.nan,int(valid.sum()),'numerical_or_zero_outcome_variance'
    inc=(red-sse)/sst;part=(red-sse)/red if red>0 else np.nan
    return float(beta[a]/ascale*.1),float(inc),float(part),int(valid.sum()),'estimable_rank_deficient' if valid.sum()<len(b) else 'estimable_full_rank'
def slopefit(st,draws=None):
    gs=st['G'].sum(0);bs=st['b'].sum(0);ascale=st['scales'][1];args=(float(st['yy'].sum()),float(st['sy'].sum()),len(st['sid']),st['a'],ascale)
    result=fit(gs,bs,*args)
    X=st['X'];y=st['y'];w=st['w'];scale=st['scales'];keep=st['keep'];Z=X[:,keep]/scale[keep];beta=np.linalg.lstsq(Z*np.sqrt(w[:,None]),y*np.sqrt(w),rcond=1e-12)[0]
    direct=float(beta[st['a']]/ascale*.1); assert np.isclose(result[0],direct,rtol=1e-8,atol=1e-9),(result,direct)
    if draws is None:return result,None,None
    # Per-SID means retain equal total SID weight under resampling; no row bootstrap.
    count=np.zeros((len(draws),len(st['sid'])),dtype=float)
    for k,ix in enumerate(draws):count[k]=np.bincount(ix,minlength=len(st['sid']))
    p=st['G'].shape[1];packed=np.column_stack([st['G'].reshape(len(st['sid']),-1),st['b'],st['yy'],st['sy']]);weighted=count@packed
    out=[];status=[]
    for k,r in enumerate(weighted):
        values=fit(r[:p*p].reshape(p,p),r[p*p:p*p+p],r[-2],r[-1],len(st['sid']),st['a'],ascale)
        out.append(values[:3]);status.append({'draw':k,'rank':values[3],'status':values[4]})
    return result,np.asarray(out),status
def newrow(process,analysis,representation,estimate,values,n_sid,n_pairs,unit,metric,**kw):
    lo,hi=interval(values);return dict(process=process,analysis=analysis,representation=representation,metric=metric,estimate=float(estimate),ci_low=lo,ci_high=hi,n_sid=n_sid,n_pairs=n_pairs,units=unit,attempted_draws=len(values),failed_draws=int((~np.isfinite(values)).sum()),CI_scope='whole-SID percentile on fixed observed support; no nominal population coverage claim' if np.isfinite(values).all() else 'conditional successful draws only; not a population95% interval',**kw)
def main():
    sys.stdout.reconfigure(encoding='utf8');spec=json.loads((B/'specification.json').read_text('utf8'));assert spec['bootstrap']['seed']==670620026
    geo=read(V/'p4_p5/P5_matched_intensity_pairs.csv'); assert geo.pair_id.is_unique
    inputs=[B/'specification.json',V/'p4_p5/P5_matched_intensity_pairs.csv',V/'src/p4_p5_analysis_v5.py']
    inputs += [d/f for d in JOBS.values() for f in ['paired_source_and_design.csv','raw_draws.npz','slope_design_and_draws.npz','summary.json']]
    dump(B/'source_manifest.json',{'sources':[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in inputs],'created_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    geom=geo[['sid','pair_id','wind_kt_pre','wind_kt_post','wind_filled_flag_pre','wind_filled_flag_post','UTC_pre','UTC_post']].copy()
    geom['strict_equal_wind']=geom.wind_kt_pre.eq(geom.wind_kt_post)
    # Existing representative forward six-hour signs are a descriptive label audit.
    wind=read(B.parent/'M2/per_interval.csv');wind.UTC=pd.to_datetime(wind.UTC,utc=True);wind['forward6h_change_kt']=wind.next_wind_kt-wind.wind_kt
    for side in ['pre','post']:
        geom['UTC_'+side]=pd.to_datetime(geom['UTC_'+side],utc=True)
        vals=wind[['sid','UTC','forward6h_change_kt','legal']].rename(columns={'UTC':'UTC_'+side,'forward6h_change_kt':'forward6h_change_kt_'+side,'legal':'source_legal_'+side})
        geom=geom.merge(vals,on=['sid','UTC_'+side],how='left',validate='many_to_one');assert geom['source_legal_'+side].all()
    geom['both_actual_enhance_then_weaken']=geom.forward6h_change_kt_pre.gt(0)&geom.forward6h_change_kt_post.lt(0)
    save(B/'fixed_pair_wind_and_tendency_ledger.csv',geom)
    branch=[]
    for name,g in [('all_original',geom),('strict_equal_wind',geom[geom.strict_equal_wind])]:
        for side in ['pre','post']:
            for label,mask in [('increasing',g['forward6h_change_kt_'+side]>0),('constant',g['forward6h_change_kt_'+side]==0),('decreasing',g['forward6h_change_kt_'+side]<0)]:branch.append({'population':name,'side':side,'tendency_class':label,'pair_n':int(mask.sum()),'total_pairs':len(g),'sid_n':g.sid.nunique(),'unit':'windkt difference over actualnext6h'})
        branch.append({'population':name,'side':'both','tendency_class':'pre_increase_and_post_decrease','pair_n':int(g.both_actual_enhance_then_weaken.sum()),'total_pairs':len(g),'sid_n':g.sid.nunique(),'unit':'windkt difference over actualnext6h'})
    save(B/'branch_sign_summary.csv',pd.DataFrame(branch));results=[];retention=[];qa=[]
    for process,path in JOBS.items():
        out=B/process;out.mkdir(exist_ok=True);f=read(path/'paired_source_and_design.csv');old=json.loads((path/'summary.json').read_text('utf8'));npz=np.load(path/'slope_design_and_draws.npz');raw=np.load(path/'raw_draws.npz')
        f=f.merge(geom[['pair_id','strict_equal_wind','wind_kt_pre','wind_kt_post','both_actual_enhance_then_weaken']],on='pair_id',validate='one_to_one');assert np.array_equal(f.sid.astype(str),npz['sid'])
        assert np.allclose(f.W_before__wind_kt,f.wind_kt_pre,equal_nan=True) and np.allclose(f.W_after__wind_kt,f.wind_kt_post,equal_nan=True)
        unit=old['units'];good=f.process_pair_valid & np.isfinite(f.difference_after_minus_before);strict=good&f.strict_equal_wind
        base=f[good].groupby('sid').difference_after_minus_before.mean().sort_index();restricted=f[strict].groupby('sid').difference_after_minus_before.mean().sort_index();common=base.reindex(restricted.index)
        assert np.isclose(base.mean(),old['raw']['estimate'],atol=1e-10)
        draws=sample(restricted.index);a=common.to_numpy()[draws].mean(1);b=restricted.to_numpy()[draws].mean(1)
        np.savez_compressed(out/'raw_common_draws.npz',sid=restricted.index.to_numpy(str),sampled_sid_indices=draws,original_common=a,strict=b,direct_strict_minus_common=b-a)
        r=old['raw'];results.append(dict(process=process,analysis='mean',representation='original_all_reused',metric='post_minus_pre',estimate=r['estimate'],ci_low=r['lo'],ci_high=r['hi'],n_sid=r['n_sid'],n_pairs=r['n_pairs'],units=unit,attempted_draws=r['attempted_draws'],failed_draws=r['failed_draws'],CI_scope='reused frozen result'))
        results += [newrow(process,'mean','original_commonSID',common.mean(),a,len(common),int((good&f.sid.isin(common.index)).sum()),unit,'post_minus_pre'),newrow(process,'mean','strict_equal_wind',restricted.mean(),b,len(restricted),int(strict.sum()),unit,'post_minus_pre'),newrow(process,'mean','direct_strict_minus_common',np.mean(restricted-common),b-a,len(common),int(strict.sum()),unit,'change_in_post_minus_pre')]
        save(out/'perSID_mean_comparison.csv',pd.DataFrame({'sid':restricted.index,'original_common':common,'strict':restricted,'strict_minus_original_common':restricted-common}).reset_index(drop=True))
        X=npz['X'];y=npz['y'];sid=npz['sid'].astype(str);valid=np.isfinite(X).all(1)&np.isfinite(y);assert np.array_equal(valid,npz['input_valid'])
        exact=valid&f.strict_equal_wind.to_numpy(bool);commonsid=np.unique(sid[exact]);same=valid&np.isin(sid,commonsid)
        originalstats=stats(sid[valid],X[valid],y[valid]);po,_,_=slopefit(originalstats);assert np.isclose(po[0],old['continuous_slope']['estimate'],rtol=1e-8,atol=1e-9)
        st0=stats(sid[same],X[same],y[same]);st1=stats(sid[exact],X[exact],y[exact]);assert np.array_equal(st0['sid'],st1['sid'])
        ds=sample(st1['sid']);p0,v0,s0=slopefit(st0,ds);p1,v1,s1=slopefit(st1,ds)
        np.savez_compressed(out/'slope_common_draws.npz',sid=st1['sid'],sampled_sid_indices=ds,original_common=v0,strict=v1,direct=v1-v0,metrics=np.array(['slope_per0.1A','incremental_R2','partial_R2']))
        status=pd.DataFrame(s0).merge(pd.DataFrame(s1),on='draw',suffixes=('_original_common','_strict'));save(out/'slope_draw_status.csv',status)
        o=old['continuous_slope'];results.append(dict(process=process,analysis='slope',representation='original_all_reused',metric='A_slope',estimate=o['estimate'],ci_low=o['lo'],ci_high=o['hi'],n_sid=o['n_sid'],n_pairs=o['n_pairs'],units=unit+' per0.1A',attempted_draws=o['attempted_draws'],failed_draws=o['failed_draws'],CI_scope='reused frozen result'))
        for j,(metric,units) in enumerate([('A_slope',unit+' per0.1A'),('incremental_R2','fraction of environmental-difference variance'),('partial_R2','fraction of nuisance-residual environmental-difference variance')]):
            results += [newrow(process,'slope','original_commonSID',p0[j],v0[:,j],len(commonsid),int(same.sum()),units,metric),newrow(process,'slope','strict_equal_wind',p1[j],v1[:,j],len(commonsid),int(exact.sum()),units,metric),newrow(process,'slope','direct_strict_minus_common',p1[j]-p0[j],v1[:,j]-v0[:,j],len(commonsid),int(exact.sum()),units,metric)]
            if j>0:results.append(dict(process=process,analysis='slope',representation='original_all_point',metric=metric,estimate=po[j],ci_low=None,ci_high=None,n_sid=len(originalstats['sid']),n_pairs=int(valid.sum()),units=units,attempted_draws=0,failed_draws=0,CI_scope='descriptive in-sample point only; existing unchanged design'))
        retention.append(dict(process=process,raw_original_sid=len(base),raw_strict_sid=len(restricted),raw_original_pairs=int(good.sum()),raw_strict_pairs=int(strict.sum()),raw_pairs_retained_fraction=float(strict.sum()/good.sum()),slope_original_sid=int(len(originalstats['sid'])),slope_strict_sid=len(commonsid),slope_original_pairs=int(valid.sum()),slope_strict_pairs=int(exact.sum()),raw_strict_pairs_actual_increase_and_decrease=int((strict&f.both_actual_enhance_then_weaken).sum())))
        f['original_process_valid']=good;f['strict_process_valid']=strict;f['original_slope_valid']=valid;f['original_slope_commonSID']=same;f['strict_slope_valid']=exact;save(out/'per_pair_support_and_values.csv',f)
        save(out/'perSID_retention.csv',f.groupby('sid').agg(original_pair_n=('pair_id','size'),original_valid_pairs=('original_process_valid','sum'),strict_valid_pairs=('strict_process_valid','sum'),original_slope_valid_pairs=('original_slope_valid','sum'),strict_slope_valid_pairs=('strict_slope_valid','sum')).reset_index())
        qa.append({'process':process,'original_mean_and_slope_reconstructed':True,'point_GLS_vs_directSVD_agree':True,'strict_zero_columns':[str(x) for x in npz['column_names'][~st1['keep']]],'same_SID_n':len(commonsid),'nuisance_design_columns':npz['column_names'].tolist(),'original_common_point_rank':p0[3],'strict_point_rank':p1[3],'strict_bootstrap_failures':int((~np.isfinite(v1[:,0])).sum()),'attempted_draws':2000})
        dump(B/'progress.json',{'completed_processes':len(qa),'last':process,'total':5});print(process,'rawstrict',len(restricted),int(strict.sum()),float(restricted.mean()),'slope',p1,'failures',qa[-1]['strict_bootstrap_failures'],flush=True)
    save(B/'sensitivity_results.csv',pd.DataFrame(results));save(B/'retention_summary.csv',pd.DataFrame(retention));dump(B/'numerical_and_estimand_check.json',qa)
    manifest=json.loads((B/'source_manifest.json').read_text('utf8'));unchanged=all(sha(Path(x['path']))==x['sha256'] for x in manifest['sources'])
    dump(B/'completion.json',{'status':'COMPLETED_ONE_FIXED_RESTRICTION','completed_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_unchanged':unchanged,'new_extraction':False,'new_matching':False,'M7_standardization_rerun':False,'runtime':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__},'bootstrap_B':2000,'bootstrap_seed':670620026,'response':'post-minus-pre environmental difference','review_pending':True})
if __name__=='__main__':main()
