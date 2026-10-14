from pathlib import Path
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
import sys,json,time,argparse,pickle,hashlib,traceback
import numpy as np
import pandas as pd
from scipy.optimize import minimize,check_grad
from scipy.special import expit,logit
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression,Ridge
from threadpoolctl import threadpool_limits
threadpool_limits(1)
HERE=Path(os.environ.get('FLTCCF_RECOVERY_RUN_DIR', Path(__file__).resolve().parent))
SEED=610102026
POLICIES={'shear200_850':-2.,'midRH':5.}
BASE=['lat','lon_sin','lon_cos','month_sin','month_cos','wind_t0_kt','threshold_gap_kt','log_age','delta_v24_backward_kt','motion_speed_prev_ms','current_land_contact','past_land_contact','past_shear','past_RH','past_SST','t0_SST','t0_PIheadroom']
BASINS=['EP','NA','NI','SI','SP','WP','SA','UNKNOWN']
def read(p):return pd.read_csv(p,keep_default_na=False,na_values=['','nan','NaN'])
def ess(w):return float(np.sum(w)**2/np.sum(np.square(w))) if np.sum(np.square(w)) else 0.
class Transform:
    def __init__(self,policy):self.policy=policy;self.names=BASE+(['midRH'] if policy=='shear200_850' else ['shear200_850'])
    def fit(self,e):
        x=e[self.names].to_numpy(float);self.med=np.nanmedian(x,axis=0);self.med=np.nan_to_num(self.med)
        x=np.where(np.isfinite(x),x,self.med);self.mean=x.mean(0);self.sd=x.std(0);self.sd=np.where(self.sd>0,self.sd,1)
        return self
    def transform(self,e):
        x=e[self.names].to_numpy(float);missing=~np.isfinite(x);x=(np.where(missing,self.med,x)-self.mean)/self.sd
        return np.column_stack([x,missing.astype(float)]+[e.basin_genesis.eq(b).to_numpy(float) for b in BASINS[1:]])
class Basis:
    def fit(self,a):
        self.knots=np.quantile(a,[.05,.35,.65,.95]);b=self.raw(a);self.mean=b.mean(0);self.sd=b.std(0);self.sd=np.where(self.sd>0,self.sd,1);return self
    def raw(self,a):
        a=np.asarray(a);k=self.knots
        def t(j):return (np.maximum(a-k[j],0)**3-np.maximum(a-k[-1],0)**3)/(k[-1]-k[j])
        return np.column_stack([a,(t(0)-t(2))/(k[-1]-k[0])**2,(t(1)-t(2))/(k[-1]-k[0])**2])
    def transform(self,a):return (self.raw(a)-self.mean)/self.sd
def design(w,basis,a):return np.column_stack([np.ones(len(w)),basis.transform(a),w])

class IntervalHazard:
    """Discrete first-event likelihood: exact bins, interval bins, right censoring."""
    def fit(self,X,L,U):
        self.p=X.shape[1];n=len(X);idx=np.arange(n);l=np.asarray(L,int);u=np.where(np.isfinite(U),np.asarray(U),8).astype(int);event=np.isfinite(U)
        self.n_interval=int((event&(u-l>1)).sum());self.n_zero_information=int((~event&(l==0)).sum())
        def fun(theta):
            eta=X@theta[:self.p];eta=eta[:,None]+np.r_[0,theta[self.p:]][None,:]
            h=expit(eta);logS=np.column_stack([np.zeros(n),-np.cumsum(np.logaddexp(0,eta),axis=1)])
            ll=logS[idx,l].copy();wstep=np.zeros((n,8))
            wstep[:]=np.arange(8)[None,:]<l[:,None]
            if event.any():
                rr=np.flatnonzero(event);ratio=np.exp(np.minimum(logS[rr,u[rr]]-logS[rr,l[rr]],-1e-15));den=1-ratio
                ll[rr]+=np.log1p(-ratio)
                wstep[rr]=( (np.arange(8)[None,:]<l[rr,None])-ratio[:,None]*(np.arange(8)[None,:]<u[rr,None]) )/den[:,None]
            score=wstep*h
            grad=np.r_[X.T@score.sum(1),score[:,1:].sum(0)]
            penalty=theta.copy();penalty[0]=0
            return -ll.sum()+.5*np.sum(penalty**2),grad+penalty
        initial=np.zeros(self.p+7);initial[0]=-3.6
        fit=minimize(fun,initial,jac=True,method='L-BFGS-B',options={'maxiter':350,'ftol':1e-10,'gtol':1e-5})
        if not fit.success:raise RuntimeError('interval model: '+fit.message)
        self.coef=fit.x;self.optimization={'success':bool(fit.success),'iterations':fit.nit,'objective':float(fit.fun),'max_gradient':float(np.max(np.abs(fit.jac)))}
        return self
    def predict(self,X):
        eta=(X@self.coef[:self.p])[:,None]+np.r_[0,self.coef[self.p:]][None,:]
        return -np.expm1(-np.logaddexp(0,eta).sum(1))

def logistic(X,y):
    y=np.asarray(y,int)
    if len(np.unique(y))==1:return float(y[0])
    return LogisticRegression(C=1,solver='lbfgs',max_iter=1000,tol=1e-7).fit(X,y)
def proba(model,X):return np.full(len(X),model) if isinstance(model,float) else model.predict_proba(X)[:,1]
def shift(a,mu,lo,hi,delta,eligible,policy):
    valid=(a>=mu+lo)&(a<=mu+hi)&(a+delta>=mu+lo)&(a+delta<=mu+hi)&eligible
    if policy=='shear200_850':valid&=a+delta>=0
    else:valid&=(a+delta>=0)&(a+delta<=100)
    return np.where(valid,a+delta,a),valid
def ratio(a,mu,sd,lo,hi,delta,eligible,policy):
    _,s=shift(a,mu,lo,hi,delta,eligible,policy)
    _,pre=shift(a-delta,mu,lo,hi,delta,eligible,policy)
    dr=np.exp(np.clip(-.5*((a-delta-mu)/sd)**2+.5*((a-mu)/sd)**2,-700,700))
    return (~s).astype(float)+pre.astype(float)*dr

def fit_one(train,test,policy,save=None):
    tr=Transform(policy).fit(train);wt=tr.transform(train);wv=tr.transform(test)
    at=train[policy].to_numpy(float);a=test[policy].to_numpy(float);delta=POLICIES[policy]
    exposure=Ridge(alpha=1).fit(wt,at);res=at-exposure.predict(wt);sd=float(np.std(res));lo,hi=np.quantile(res,[.05,.95]);mu=exposure.predict(wv)
    basin_n=train.groupby('basin_genesis').sid.nunique();eligible=test.basin_genesis.map(basin_n).fillna(0).ge(20).to_numpy()
    shifted,covered=shift(a,mu,lo,hi,delta,eligible,policy);r=ratio(a,mu,sd,lo,hi,delta,eligible,policy)
    basis=Basis().fit(at);Xt=design(wt,basis,at);X0=design(wv,basis,a);Xd=design(wv,basis,shifted)
    hazard=IntervalHazard().fit(Xt,train.L_h.to_numpy(float)/6,train.U_h.to_numpy(float)/6)
    m0=hazard.predict(X0);md=hazard.predict(Xd)
    # Complete-record selection does not use recovery status.
    R=train.complete48.astype(bool).to_numpy();gm=logistic(Xt[:,1:],R);g0=proba(gm,X0[:,1:]);gd=proba(gm,Xd[:,1:])
    R24=train.delta24.notna().to_numpy();gm24=logistic(Xt[:,1:],R24);g24train=proba(gm24,Xt[:,1:]);g240=proba(gm24,X0[:,1:]);g24d=proba(gm24,Xd[:,1:])
    wind=Ridge(alpha=1,fit_intercept=False).fit(Xt[R24],train.loc[R24,'delta24'],sample_weight=np.minimum(1/g24train[R24],20))
    v0=wind.predict(X0);vd=wind.predict(Xd)
    out=pd.DataFrame({'event_id':test.event_id.to_numpy(),'sid':test.sid.to_numpy(),'fold':test.fold.to_numpy(),'policy':policy,'A':a,'A_shift':shifted,'covered':covered,'density_ratio':r,'density_residual':a-mu,'density_sd':sd,'support_low':mu+lo,'support_high':mu+hi,'R48':test.complete48.to_numpy(),'Y48':test.recovery48.to_numpy(),'g48':g0,'g48_shift':gd,'m48':m0,'m48_shift':md,'R24':test.delta24.notna().to_numpy(),'Y24':test.delta24.to_numpy(),'g24':g240,'g24_shift':g24d,'m24':v0,'m24_shift':vd,'basin':test.basin_genesis.to_numpy()})
    for outcome in ['48','24']:
        y=out['Y'+outcome].fillna(0).to_numpy();R=out['R'+outcome].astype(bool).to_numpy();g=out['g'+outcome].to_numpy();m=out['m'+outcome].to_numpy();diff=out['m'+outcome+'_shift'].to_numpy()-m
        out['ipcw'+outcome+'_uncapped']=R/g;out['ipcw'+outcome]=R*np.minimum(1/g,20)
        out['gcomp'+outcome]=diff
        out['onestep'+outcome]=diff+(r-1)*out['ipcw'+outcome]*(y-m)
        out['ipw'+outcome]=(r-1)*out['ipcw'+outcome]*y
        out['onestep'+outcome+'_uncapped']=diff+(r-1)*out['ipcw'+outcome+'_uncapped']*(y-m)
    # Benchmark measured baseline associations; no causal interpretation.
    modified=test.copy();modified['delta_v24_backward_kt']=modified.delta_v24_backward_kt+5
    Xbm=design(tr.transform(modified),basis,a)
    out['benchmark_pastwind5_risk']=hazard.predict(Xbm)-m0;out['benchmark_pastwind5_kt']=wind.predict(Xbm)-v0
    if save is not None:
        with open(save,'wb') as f:pickle.dump(dict(transform=tr,basis=basis,exposure=exposure,residual_sd=sd,residual_bounds=[lo,hi],basin_n=basin_n,hazard=hazard,observation48=gm,observation24=gm24,wind=wind,train_rows=train['_rowid'].tolist(),test_rows=test['_rowid'].tolist(),train_sid=train.sid.tolist(),test_sid=test.sid.tolist()),f,protocol=5)
    return out
def crossfit(e,save_dir=None):
    parts=[]
    for policy in POLICIES:
        for fold in range(5):
            tr=e[e.fold.ne(fold)];te=e[e.fold.eq(fold)]
            p=save_dir/f'{policy}_fold{fold}.pkl' if save_dir else None
            parts.append(fit_one(tr,te,policy,p))
    return pd.concat(parts,ignore_index=True)
def summarize(pred):
    rows=[]
    for p,g in pred.groupby('policy',sort=False):
        for outcome in ['48','24']:
            scale=100 if outcome=='48' else 1
            for method in ['gcomp','onestep','ipw','onestep_uncapped']:
                col=method+outcome if method!='onestep_uncapped' else 'onestep'+outcome+'_uncapped'
                v=g[col].to_numpy()*scale
                rows.append(dict(policy=p,outcome='recovery48_pp' if outcome=='48' else 'delta24_kt',method=method,estimate=float(np.mean(v)),coverage=float(g.covered.mean()),n=len(g),empirical_if_se=float(np.std(v,ddof=1)/np.sqrt(len(v)))))
    return pd.DataFrame(rows)
def validate():
    tests=[]
    for policy,delta in POLICIES.items():
        mu=15 if policy.startswith('shear') else 60;sd=6
        grid=np.linspace(mu-9*sd,mu+9*sd,200001);p=norm.pdf(grid,mu,sd)
        rr=ratio(grid,mu,sd,-1.645*sd,1.645*sd,delta,np.ones(len(grid),bool),policy)
        integral=np.trapz(p*rr,grid);assert abs(integral-1)<1e-4
        y,c=shift(grid,mu,-1.645*sd,1.645*sd,delta,np.ones(len(grid),bool),policy)
        lhs=np.trapz(y*p,grid);rhs=np.trapz(grid*p*rr,grid);assert abs(lhs-rhs)<.003
        tests.append(dict(test=policy+'_all_preimages_integral',integral=float(integral),shift_mean_error=float(lhs-rhs),max_ratio=float(rr.max())))
    # Constant hazard survival identity and interval objective on no-covariate data.
    h=.1;assert np.isclose(-np.expm1(8*np.log1p(-h)),1-(1-h)**8)
    a=np.arange(4.);same,c=shift(a,0,-100,100,0,np.ones(4,bool),'shear200_850');assert np.array_equal(a,same)
    (HERE/'implementation_tests.json').write_text(json.dumps(tests,indent=2));return tests
def sensitivities(pred):
    rows=[];diag=[];balance=[]
    for policy,g in pred.groupby('policy'):
        r=g.density_ratio.to_numpy();cov=g.covered.mean()
        for outcome in ['48','24']:
            factor=100 if outcome=='48' else 1
            m=g['m'+outcome].to_numpy();md=g['m'+outcome+'_shift'].to_numpy();q=g['g'+outcome].to_numpy();qd=g['g'+outcome+'_shift'].to_numpy()
            shifts=[-2,-1,0,1,2] if outcome=='48' else [-10,-5,0,5,10]
            for b in shifts:
                if outcome=='48':
                    natural=q*m+(1-q)*expit(logit(np.clip(m,1e-8,1-1e-8))+b);changed=qd*md+(1-qd)*expit(logit(np.clip(md,1e-8,1-1e-8))+b)
                else:natural=m+(1-q)*b;changed=md+(1-qd)*b
                rows.append(dict(policy=policy,outcome=outcome,MNAR_offset=b,offset_units='log_odds' if outcome=='48' else 'kt',estimate=float(np.mean(changed-natural)*factor)))
            w=g['ipcw'+outcome].to_numpy();unc=g['ipcw'+outcome+'_uncapped'].to_numpy()
            estimate=g['gcomp'+outcome].mean()*factor
            diag.append(dict(policy=policy,outcome=outcome,coverage=float(cov),ratio_mean=float(r.mean()),ratio_max=float(r.max()),ratio_ESS=ess(r),complete_n=int(g['R'+outcome].sum()),observed_IPCW_ESS=ess(w),policy_observed_ESS=ess(r*w),uncapped_weight_max=float(unc.max()),capped_n=int((unc>20).sum()),g_min=float(q.min()),g_median=float(np.median(q)),tipping_offset_per_shifted_person=estimate/cov,benchmark_observed_pastwind5=float(g['benchmark_pastwind5_risk' if outcome=='48' else 'benchmark_pastwind5_kt'].mean()*factor)))
        # Check inverse density target against actual pushed-forward exposure means.
        balance.append(dict(policy=policy,n=len(g),natural_A=float(g.A.mean()),empirical_shifted_A=float(g.A_shift.mean()),ratio_weighted_A=float(np.sum(r*g.A)/np.sum(r)),mean_ratio=float(r.mean()),mean_residual=float(g.density_residual.mean()),residual_sd=float(g.density_residual.std()),normal_model_sd=float(g.density_sd.mean())))
    pd.DataFrame(rows).to_csv(HERE/'MNAR_sensitivity.csv',index=False);pd.DataFrame(diag).to_csv(HERE/'weight_and_bias_diagnostics.csv',index=False);pd.DataFrame(balance).to_csv(HERE/'density_diagnostics.csv',index=False)
    pred.groupby(['policy','basin']).agg(n=('sid','size'),coverage=('covered','mean'),ratio_mean=('density_ratio','mean')).to_csv(HERE/'support_by_basin.csv')
