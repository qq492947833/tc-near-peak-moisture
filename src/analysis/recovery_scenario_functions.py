"""Frozen scenario model functions; explicit full_refit entry only."""
from pathlib import Path
import json,hashlib,traceback,warnings,pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.exceptions import ConvergenceWarning
from . import recovery_frozen as f
HERE=None
OLD=None
SCENARIOS=['USA_original_nonsynoptic','archive_end_no_unseen_return','archive_end_unseen_return']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def inputs():
    e=f.read(OLD/'cohort_recovered.csv');e['_rowid']=np.arange(len(e));e.complete48=e.complete48.astype(str).str.lower().eq('true');e.fold=e.fold.astype(int)
    a=f.read(HERE/'per_SID_record_audit.csv').set_index('sid').loc[e.sid];terminal=a.archive_ends_before48.astype(str).str.lower().eq('true').to_numpy()
    datasets={};x=f.read(HERE/'nonsynoptic_original_USA_reports.csv');changes=[]
    q=e.copy();q['R_scenario']=e.complete48
    for sid,z in x.groupby('sid'):
        ix=q.index[q.sid.eq(sid)][0];lag=z.lag_h.min();upper=6*np.ceil(lag/6)
        if not np.isfinite(q.loc[ix,'U_h']) or upper<q.loc[ix,'U_h']:
            changes.append({'sid':sid,'original_Y':q.loc[ix,'recovery48'],'old_L':q.loc[ix,'L_h'],'old_U':q.loc[ix,'U_h'],'new_U':upper,'observed_hour':lag})
            q.loc[ix,'L_h']=min(q.loc[ix,'L_h'],6*np.floor(lag/6));q.loc[ix,'U_h']=upper;q.loc[ix,'recovery48']=1
    datasets[SCENARIOS[0]]=q
    pd.DataFrame(changes).to_csv(HERE/'scenario_nonsynoptic_changes.csv',index=False)
    for name,hit in zip(SCENARIOS[1:],[False,True]):
        q=e.copy();unknown=q.recovery48.isna().to_numpy();target=terminal&unknown
        q['R_scenario']=q.complete48|terminal
        q.loc[target,'recovery48']=int(hit)
        if hit:q.loc[target,'U_h']=48
        else:q.loc[target,'L_h']=48;q.loc[target,'U_h']=np.inf
        assert np.all(q.loc[target,'L_h']<48) if hit else np.all(q.loc[target,'L_h']==48)
        assert int(target.sum())==417
        datasets[name]=q
    for name,q in datasets.items():
        assert q.sid.tolist()==e.sid.tolist();assert q[f.BASE+['shear200_850','midRH']].equals(e[f.BASE+['shear200_850','midRH']])
        assert (~q.R_scenario|q.recovery48.notna()).all()
        q.to_csv(HERE/(name+'_cohort.csv'),index=False)
    return e,datasets
def common(train,test,policy):
    tr=f.Transform(policy).fit(train);wt=tr.transform(train);wv=tr.transform(test)
    at=train[policy].to_numpy(float);a=test[policy].to_numpy(float);delta=f.POLICIES[policy]
    exposure=Ridge(alpha=1).fit(wt,at);res=at-exposure.predict(wt);sd=float(res.std());lo,hi=np.quantile(res,[.05,.95]);mu=exposure.predict(wv)
    bn=train.groupby('basin_genesis').sid.nunique();eligible=test.basin_genesis.map(bn).fillna(0).ge(20).to_numpy()
    shifted,covered=f.shift(a,mu,lo,hi,delta,eligible,policy);ratio=f.ratio(a,mu,sd,lo,hi,delta,eligible,policy)
    basis=f.Basis().fit(at);Xt=f.design(wt,basis,at);X0=f.design(wv,basis,a);Xd=f.design(wv,basis,shifted)
    return Xt,X0,Xd,covered,ratio,dict(transform=tr,exposure=exposure,basis=basis,exposure_residual_sd=sd,support_bounds=[lo,hi],basin_n=bn)
def estimate_set(e,sets,indices=None,point=False):
    ee=e if indices is None else e.iloc[indices].copy()
    ss=sets if indices is None else {s:q.iloc[indices].copy() for s,q in sets.items()}
    allparts={(s,p):[] for s in sets for p in f.POLICIES};fail={};ops=[]
    for policy in f.POLICIES:
        for fold in range(5):
            mask=ee.fold.eq(fold).to_numpy()
            if not mask.any():
                for s in sets:fail[(s,policy)]='empty test fold'
                continue
            try:Xt,X0,Xd,covered,r,meta=common(ee.loc[~mask],ee.loc[mask],policy)
            except Exception:
                for s in sets:fail[(s,policy)]=traceback.format_exc()
                continue
            for s,q in ss.items():
                if (s,policy) in fail:continue
                try:
                    tr=q.loc[~mask];te=q.loc[mask]
                    with warnings.catch_warnings(record=True) as ww:
                        warnings.simplefilter('always')
                        hazard=f.IntervalHazard().fit(Xt,tr.L_h.to_numpy(float)/6,tr.U_h.to_numpy(float)/6)
                        gm=f.logistic(Xt[:,1:],tr.R_scenario.to_numpy(bool));g=f.proba(gm,X0[:,1:])
                    if any(issubclass(w.category,ConvergenceWarning) for w in ww):raise RuntimeError('logistic ConvergenceWarning: '+';'.join(str(w.message) for w in ww))
                    m=hazard.predict(X0);md=hazard.predict(Xd);R=te.R_scenario.to_numpy(bool);y=te.recovery48.fillna(0).to_numpy(float)
                    unc=R/g;w=R*np.minimum(1/g,20);resid=w*(y-m);residu=unc*(y-m)
                    p=pd.DataFrame(dict(sid=te.sid.to_numpy(),event_id=te.event_id.to_numpy(),fold=fold,policy=policy,scenario=s,R=R,Y=te.recovery48.to_numpy(),g=g,m=m,md=md,ratio=r,covered=covered,
                        natural_gcomp=m,policy_gcomp=md,natural_onestep=m+resid,policy_onestep=md+r*resid,natural_ipw=w*y,policy_ipw=r*w*y,natural_onestep_uncapped=m+residu,policy_onestep_uncapped=md+r*residu))
                    assert np.isfinite(p.select_dtypes('number').drop(columns=['Y']).to_numpy()).all()
                    allparts[(s,policy)].append(p)
                    ops.append(dict(scenario=s,policy=policy,fold=fold,**hazard.optimization,observation_iterations=int(np.max(gm.n_iter_)) if hasattr(gm,'n_iter_') else 0,observation_min=float(g.min()),support_n=int(covered.sum()),warning_messages=[str(w.message) for w in ww]))
                    if point:
                        dest=HERE/'point_models'/s;dest.mkdir(parents=True,exist_ok=True)
                        with (dest/(policy+'_fold'+str(fold)+'.pkl')).open('wb') as ff:pickle.dump(dict(**meta,hazard=hazard,observation=gm,train_sid=tr.sid.tolist(),test_sid=te.sid.tolist()),ff)
                except Exception:fail[(s,policy)]=traceback.format_exc()
    results=[]
    for (s,p),parts in allparts.items():
        if (s,p) in fail:results.append(dict(scenario=s,policy=p,status='FAILED',error=fail[(s,p)]));continue
        pred=pd.concat(parts,ignore_index=True);assert len(pred)==len(ee)
        if point:pred.to_csv(HERE/(s+'_'+p+'_predictions.csv'),index=False)
        for method in ['gcomp','onestep','ipw','onestep_uncapped']:
            natural=pred['natural_'+method].mean()*100;policymean=pred['policy_'+method].mean()*100
            results.append(dict(scenario=s,policy=p,method=method,status='OK',natural_risk_pct=natural,policy_risk_pct=policymean,contrast_pp=policymean-natural,n=len(pred),observed_n=int(pred.R.sum()),support_fraction=pred.covered.mean()))
    return results,ops
