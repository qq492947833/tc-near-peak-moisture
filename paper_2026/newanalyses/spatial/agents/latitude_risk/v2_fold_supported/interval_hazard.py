"""Original restricted-cubic basis and interval-censored hazard likelihood.
Extracted without changing class implementations from fit_policies_v6.py.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

class Basis:
    def fit(self,a):
        self.knots=np.quantile(a,[.05,.35,.65,.95]);b=self.raw(a);self.mean=b.mean(0);self.sd=b.std(0);self.sd=np.where(self.sd>0,self.sd,1);return self
    def raw(self,a):
        a=np.asarray(a);k=self.knots
        def t(j):return (np.maximum(a-k[j],0)**3-np.maximum(a-k[-1],0)**3)/(k[-1]-k[j])
        return np.column_stack([a,(t(0)-t(2))/(k[-1]-k[0])**2,(t(1)-t(2))/(k[-1]-k[0])**2])
    def transform(self,a):return (self.raw(a)-self.mean)/self.sd

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
