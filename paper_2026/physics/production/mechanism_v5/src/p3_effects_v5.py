"""P3 cross-fitted modified-initial-environment-policy g-computation.
No double-robustness claim. --self-test never uses real environmental estimates.
--run requires explicit exposure CSV and creates a new output directory.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,traceback,warnings,sys,os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.exceptions import ConvergenceWarning

V5=Path(__file__).resolve().parents[1]
SEED=510082026
POLICIES={'shear200_850':-2.,'midRH':5.}
BASINS=['EP','NA','NI','SI','SP','WP','SA','UNKNOWN']
BASE=['lat','lon_sin','lon_cos','month_sin','month_cos','wind_t0_kt','threshold_gap_kt','log_age',
      'delta_v24_backward_kt','motion_speed_prev_ms','current_land_contact','past_land_contact',
      'past_shear','past_RH','past_SST','t0_SST','t0_PIheadroom']
TESTS=[]
class FitFailure(RuntimeError):pass
class EstimabilityFailure(FitFailure):pass
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(name,ok,detail=None):
 TESTS.append({'test':name,'passed':bool(ok),'detail':detail});assert ok,(name,detail)
def booleans(s):
 if pd.api.types.is_bool_dtype(s):return s.fillna(False)
 z=s.astype(str).str.strip().str.lower()
 if not z.isin(['true','false','1','0','1.0','0.0','','nan','none']).all():raise ValueError('Invalid boolean encoding')
 return z.isin(['true','1','1.0'])
def readcsv(p):return pd.read_csv(p,keep_default_na=False,na_values=['','nan','NaN'])
def fold(sid):return int(hashlib.sha256((str(SEED)+'|'+str(sid)).encode()).hexdigest()[:12],16)%5
def effective_n(w):
 w=np.asarray(w,float);return float(w.sum()**2/np.square(w).sum()) if len(w) and np.square(w).sum()>0 else 0.
def fitweights(w,mass):
 w=np.asarray(w,float)
 if not np.isfinite(w).all() or (w<0).any() or w.sum()<=0 or mass<=0:raise FitFailure('invalid fit weights or reference mass')
 return w/w.sum()*mass
def logistic(X,y,w,mass):
 y=np.asarray(y,int)
 if len(np.unique(y))<2:raise FitFailure('outcome single class: no finite logistic target')
 m=LogisticRegression(C=1.,solver='lbfgs',max_iter=1000,tol=1e-8,fit_intercept=True)
 with warnings.catch_warnings():
  warnings.simplefilter('error',ConvergenceWarning);m.fit(X,y,sample_weight=fitweights(w,mass))
 return m
def observation_model(X,y,w,mass):
 y=np.asarray(y,int)
 if len(np.unique(y))==1:
  if y[0]==1:return 1.
  raise FitFailure('No observed outcome support in training fold')
 return logistic(X,y,w,mass)
def probability(model,X):return np.full(len(X),model) if isinstance(model,float) else model.predict_proba(X)[:,1]
def ridge(X,y,w,mass):return Ridge(alpha=1.,fit_intercept=True).fit(X,y,sample_weight=fitweights(w,mass))

class BaselineTransform:
 def __init__(self,policy):self.policy=policy
 def fit(self,e):
  self.names=BASE+['other_A'];z=self.raw(e)
  with warnings.catch_warnings():
   warnings.simplefilter('ignore',RuntimeWarning);self.median=np.nanmedian(z,axis=0)
  self.all_missing=~np.isfinite(self.median);self.median=np.where(self.all_missing,0.,self.median)
  x=np.where(np.isfinite(z),z,self.median)
  self.mean=x.mean(axis=0);self.sd=x.std(axis=0);self.sd=np.where(self.sd>0,self.sd,1.)
  return self
 def raw(self,e):
  z=e[BASE].apply(pd.to_numeric,errors='coerce').to_numpy(float)
  other='midRH' if self.policy=='shear200_850' else 'shear200_850'
  return np.column_stack([z,e[other].to_numpy(float)])
 def transform(self,e):
  z=self.raw(e);missing=~np.isfinite(z);x=(np.where(missing,self.median,z)-self.mean)/self.sd
  basin=e.basin_genesis.fillna('UNKNOWN').astype(str);basin=basin.where(basin.isin(BASINS),'UNKNOWN')
  return np.column_stack([x,missing.astype(float)]+[(basin==b).to_numpy(float) for b in BASINS[1:]])
 def metadata(self):return {'names':self.names,'median':self.median.tolist(),'mean':self.mean.tolist(),'sd':self.sd.tolist(),'all_missing':self.all_missing.tolist()}

class ExposureBasis:
 def __init__(self,kind):self.kind=kind
 def fit(self,a):
  a=np.asarray(a,float);self.knots=np.quantile(a,[.05,.35,.65,.95])
  if self.kind=='spline' and (len(np.unique(self.knots))<4 or np.diff(self.knots).min()<=np.finfo(float).eps*max(1.,np.abs(self.knots).max())):
   raise EstimabilityFailure('Fewer than four numerically distinct prespecified spline knots')
  b=self.raw(a);self.mean=b.mean(axis=0);self.sd=b.std(axis=0)
  if (self.sd<=np.finfo(float).eps).any():raise EstimabilityFailure('Degenerate exposure basis')
  return self
 def raw(self,a):
  a=np.asarray(a,float)
  if self.kind=='linear':return a[:,None]
  k=self.knots;scale=(k[-1]-k[0])**2
  def d(j):return (np.maximum(a-k[j],0)**3-np.maximum(a-k[-1],0)**3)/(k[-1]-k[j])
  return np.column_stack([a,(d(0)-d(2))/scale,(d(1)-d(2))/scale])
 def transform(self,a):return (self.raw(a)-self.mean)/self.sd

def target_estimable(X,delta):
 if not len(delta):return {'estimable':True,'rank':0,'max_relative_projection_error':0.}
 M=np.column_stack([np.ones(len(X)),X])
 D=np.column_stack([np.zeros(len(delta)),delta])
 norm=np.linalg.norm(M,axis=0);norm=np.where(norm>0,norm,1.)
 M=M/norm;D=D/norm
 _,s,vt=np.linalg.svd(M,full_matrices=False)
 tol=np.finfo(float).eps*max(M.shape)*s[0]
 rank=int((s>tol).sum());rows=vt[:rank]
 residual=D-(D@rows.T)@rows
 denom=np.linalg.norm(D,axis=1)
 relative=np.linalg.norm(residual,axis=1)/np.maximum(denom,np.finfo(float).tiny)
 # Rounding tolerance derives from dimension and machine precision; never tuned to effect.
 threshold=100*np.finfo(float).eps*max(M.shape)
 ok=bool(np.all(relative<=threshold))
 return {'estimable':ok,'rank':rank,'columns':M.shape[1],'svd_tolerance':float(tol),
         'max_relative_projection_error':float(relative.max(initial=0.)),'projection_threshold':float(threshold)}

def long_history(events,long):
 long=long.copy();long.UTC=pd.to_datetime(long.UTC,utc=True)
 for c in ['finite_wind','observed_threshold_return','at_risk_contiguous','event_observed_contiguous','censored_contiguous']:
  long[c]=booleans(long[c])
 events=events.copy();events.t0=pd.to_datetime(events.t0,utc=True)
 z=long.merge(events[['event_id','wind_t0_kt','delta_v6_backward_kt']],on='event_id',validate='many_to_one')
 z=z.sort_values(['event_id','lag_h'])
 z['previous_wind']=z.groupby('event_id').wind_kt.shift()
 z.loc[z.lag_h.eq(6),'previous_wind']=z.loc[z.lag_h.eq(6),'wind_t0_kt']
 z['previous_previous_wind']=z.groupby('event_id').previous_wind.shift()
 z['history_dv6']=z.previous_wind-z.previous_previous_wind
 z.loc[z.lag_h.eq(6),'history_dv6']=z.loc[z.lag_h.eq(6),'delta_v6_backward_kt']
 z['history_gap']=z.previous_wind-z.Vref_kt
 z['previous_state']=z.groupby('event_id').state_class.shift().fillna('A')
 return z.drop(columns=['wind_t0_kt','delta_v6_backward_kt'])

def assemble_inputs(exposure_path,cohort='main'):
 all_events=readcsv(V5/'data/p3_all_nonoverlap96h_events.csv')
 all_events.t0=pd.to_datetime(all_events.t0,utc=True);all_events['main_first_per_sid']=booleans(all_events.main_first_per_sid)
 e=all_events[all_events.main_first_per_sid].copy() if cohort=='main' else all_events.copy()
 for c in ['lat','lon_sin','lon_cos','month_sin','month_cos','wind_t0_kt','threshold_gap_kt','age_h','delta_v24_backward_kt',
           'motion_speed_prev_ms','current_land_contact','past_land_contact','delta_v24_forward_kt']:
  e[c]=pd.to_numeric(e[c],errors='coerce')
 e['log_age']=np.log1p(e.age_h)
 x=readcsv(exposure_path);required={'request_id','event_id','sid','lag_h','variable','value','support'}
 if not required.issubset(x):raise ValueError('Exposure table missing '+str(required-set(x.columns)))
 x=x[x.event_id.isin(e.event_id)].copy()
 # Accepted driver variable aliases identify the ring-mean vector shear, never mean gridpoint magnitude.
 aliases={'shear_200_850_vector_magnitude_ms':'shear200_850','sst_C':'SST',
          'mid_rh_pct':'midRH','PI_headroom_ms':'PIheadroom'}
 x['variable']=x.variable.replace(aliases)
 if x.duplicated(['event_id','lag_h','variable']).any():raise ValueError('Duplicate exposure event lag variable')
 if not x.variable.isin(['shear200_850','midRH','SST','PIheadroom']).all():
  x=x[x.variable.isin(['shear200_850','midRH','SST','PIheadroom'])].copy()
 if not x.lag_h.isin([-24,-18,-12,-6,0]).all():raise ValueError('Future or unregistered exposure lag')
 sidmap=e.set_index('event_id').sid
 if not (x.sid==x.event_id.map(sidmap)).all():raise ValueError('Exposure crosses SID')
 x.value=pd.to_numeric(x.value,errors='coerce')
 # Explicit mixed boolean/fraction parsing; unexpected nonempty encodings are errors.
 raw_support=x.support.astype(str).str.strip().str.lower()
 normalized=raw_support.replace({'true':'1','false':'0','none':'','nan':''})
 support=pd.to_numeric(normalized,errors='coerce')
 if (support.isna()&normalized.ne('')).any():raise ValueError('Unrecognized support encoding')
 if ((support.dropna()<0)|(support.dropna()>1)).any():raise ValueError('support must be fraction0–1 or boolean, not percent')
 valid=support.ge(.9)
 x['valid']=valid&np.isfinite(x.value);x.loc[~x.valid,'value']=np.nan
 timing=[]
 for var,ename in [('shear200_850','shear200_850'),('midRH','midRH'),('SST','t0_SST'),('PIheadroom','t0_PIheadroom')]:
  a=x[x.variable.eq(var)&x.lag_h.eq(0)].set_index('event_id')
  e[ename]=e.event_id.map(a.value)
 for var,name in [('shear200_850','past_shear'),('midRH','past_RH'),('SST','past_SST')]:
  a=x[x.variable.eq(var)&x.lag_h.lt(0)].groupby('event_id').value.agg(['mean','count'])
  a.loc[a['count'].lt(3),'mean']=np.nan
  e[name]=e.event_id.map(a['mean']);e[name+'_valid_times']=e.event_id.map(a['count']).fillna(0).astype(int)
 reasons=[]
 for z in e.itertuples():
  why=[]
  if not np.isfinite(z.shear200_850):why.append('t0_shear_missing_or_failed_QC')
  if not np.isfinite(z.midRH):why.append('t0_RH_missing_or_failed_QC')
  reasons.append(';'.join(why))
 e['exposure_exclusion']=reasons;exclusions=e[['event_id','sid','exposure_exclusion']].copy()
 e=e[e.exposure_exclusion.eq('')].copy().reset_index(drop=True)
 e['fold']=e.sid.map(fold);e['_source_i']=np.arange(len(e))
 l=readcsv(V5/'data/p3_longitudinal_followup.csv');l=l[l.event_id.isin(e.event_id)]
 l=long_history(e,l)
 metadata={'cohort':cohort,'before_exposure_events':len(exclusions),'after_exposure_events':len(e),'after_exposure_sid':e.sid.nunique(),
  'input_exposure_sha256':sha(exposure_path),'effects_not_yet_fit':True,
  'counts':exclusions.exposure_exclusion.value_counts().to_dict(),'baseline_columns':BASE,'forbidden_future_controls_present':False}
 return e,l,exclusions,metadata

def expanded(events,long,indices=None):
 e=events.copy() if indices is None else events.iloc[indices].copy()
 e=e.reset_index(drop=True);e['_instance']=np.arange(len(e))
 l=e[['event_id','_instance']].merge(long,on='event_id',how='left',validate='many_to_many')
 return e,l

def response_design(e,basis,w,a,time_lag=None,horizon=48):
 z=np.column_stack([basis.transform(a),w])
 if time_lag is not None:
  bins=(np.asarray(time_lag,int)//6)-1
  z=np.column_stack([z,np.eye(horizon//6)[bins,1:]])
 return z
def event_map(e,l):
 ix=e.set_index('_instance')
 return ix.loc[l._instance].reset_index(drop=True)

def reference_masses(e,l,horizon):
 out={}
 for f in range(5):
  tr=e[e.fold.ne(f)]
  r=l[l._instance.isin(tr._instance)&l.lag_h.le(horizon)&l.at_risk_contiguous]
  out[f]={'events':len(tr),'censor':len(r),'hazard':int(r.finite_wind.sum()),
          'endpoint_observed':int(tr.delta_v24_forward_kt.notna().sum())}
 return out

def condition_policy(train,test,wtrain,wtest,policy,mass):
 a=train[policy].to_numpy(float);m=ridge(wtrain,a,np.ones(len(a)),mass)
 residual=a-m.predict(wtrain);low,high=np.quantile(residual,[.05,.95])
 mean=m.predict(wtest);a0=test[policy].to_numpy(float);candidate=a0+POLICIES[policy]
 counts=train.groupby('basin_genesis').sid.nunique()
 coverage=(a0>=mean+low)&(a0<=mean+high)&(candidate>=mean+low)&(candidate<=mean+high)&test.basin_genesis.map(counts).fillna(0).ge(20).to_numpy()
 if policy=='shear200_850':coverage&=candidate>=0
 new=np.where(coverage,candidate,a0)
 return new,coverage,{'residual_low':float(low),'residual_high':float(high),'basin_unique_sid':counts.to_dict()}

def censor_features(e,l,transform,train_scale=None,horizon=48,policy='shear200_850'):
 mapped=event_map(e,l);w=transform.transform(mapped);a=mapped[policy].to_numpy(float)
 # Raw exposure standardized using the baseline training transform's treatment moments supplied separately.
 a=(a-transform.A_mean)/transform.A_sd
 history=l[['history_gap','history_dv6']].to_numpy(float);missing=~np.isfinite(history)
 if train_scale is None:
  with warnings.catch_warnings():
   warnings.simplefilter('ignore',RuntimeWarning);med=np.nanmedian(history,axis=0)
  med=np.where(np.isfinite(med),med,0.);filled=np.where(missing,med,history)
  mu=filled.mean(axis=0);sd=filled.std(axis=0);sd=np.where(sd>0,sd,1.)
  train_scale=(med,mu,sd)
 med,mu,sd=train_scale;history=(np.where(missing,med,history)-mu)/sd
 bins=l.lag_h.to_numpy(int)//6-1
 state=np.column_stack([(l.previous_state=='B').to_numpy(float),(l.previous_state=='C').to_numpy(float)])
 return np.column_stack([a,w,np.eye(horizon//6)[bins,1:],history,missing.astype(float),state]),train_scale

def survival_weights(l,g):
 l=l.copy();l['g']=g
 with np.errstate(divide='ignore',over='ignore'):
  l['logg']=np.log(g);l['logG']=l.groupby('_instance').logg.cumsum()
  l['ipcw_uncapped']=np.exp(-l.logG)
 l['ipcw']=np.minimum(l.ipcw_uncapped,20.)
 return l

def fold_fit(train,test,ltrain,ltest,policy,kind,outcome,mass,identity=False):
 horizon=96 if outcome=='return96' else 48
 transform=BaselineTransform(policy).fit(train)
 transform.A_mean=float(train[policy].mean());transform.A_sd=float(train[policy].std(ddof=0))
 if not transform.A_sd>0:raise EstimabilityFailure('No training exposure variation')
 wt=transform.transform(train);wv=transform.transform(test)
 basis=ExposureBasis(kind).fit(train[policy])
 av=test[policy].to_numpy(float)
 shifted,cov,support=condition_policy(train,test,wt,wv,policy,mass['events'])
 if identity:shifted=av.copy();cov=np.zeros(len(av),bool)
 diag={'fold':int(test.fold.iloc[0]),'policy':policy,'kind':kind,'outcome':outcome,'n_train':len(train),'n_test':len(test),
       'support':support,'missing_W':transform.metadata(),'knots':basis.knots.tolist(),'penalty_reference_mass':mass}
 if outcome=='delta24':
  Ct=np.column_stack([(train[policy].to_numpy()-transform.A_mean)/transform.A_sd,wt])
  Cv=np.column_stack([(av-transform.A_mean)/transform.A_sd,wv])
  ot=train.delta_v24_forward_kt.notna().to_numpy()
  gm=observation_model(Ct,ot,np.ones(len(ot)),mass['events'])
  gt=probability(gm,Ct);gv=probability(gm,Cv)
  with np.errstate(divide='ignore'):uncap=1/gt
  ipcw=np.minimum(uncap,20.)
  xt=response_design(train,basis,wt,train[policy]);xv=response_design(test,basis,wv,av);xs=response_design(test,basis,wv,shifted)
  est=target_estimable(xt[ot],(xs-xv)[cov])
  if not est['estimable']:raise EstimabilityFailure('Endpoint intervention contrast outside training row space '+str(est))
  model=ridge(xt[ot],train.loc[ot,'delta_v24_forward_kt'],ipcw[ot],mass['endpoint_observed'])
  natural=model.predict(xv);intervened=model.predict(xs)
  diag.update(estimability=est,censor_type='exact24h endpoint observation',cap_fraction=float((uncap[ot]>20).mean()),
      ipcw_ESS=effective_n(ipcw[ot]),uncapped_weight_max=float(np.max(uncap[ot])))
  observed=test.delta_v24_forward_kt.notna().to_numpy()
  evalw=np.where(observed,np.minimum(np.divide(1.,gv,out=np.full_like(gv,np.inf),where=gv>0),20.),0.)
  metric=pd.DataFrame({'_instance':test._instance,'y':test.delta_v24_forward_kt,'prediction':natural,'weight':evalw,'observed':observed})
 else:
  rt=ltrain[ltrain.lag_h.le(horizon)&ltrain.at_risk_contiguous].copy()
  rv=ltest[ltest.lag_h.le(horizon)&ltest.at_risk_contiguous].copy()
  ct,hscale=censor_features(train,rt,transform,horizon=horizon,policy=policy)
  cv,_=censor_features(test,rv,transform,hscale,horizon=horizon,policy=policy)
  gm=observation_model(ct,rt.finite_wind,np.ones(len(rt)),mass['censor'])
  rt=survival_weights(rt,probability(gm,ct));rv=survival_weights(rv,probability(gm,cv))
  observed=rt.finite_wind.to_numpy()
  mt=event_map(train,rt)
  xt=response_design(mt,basis,transform.transform(mt),mt[policy],rt.lag_h,horizon)
  # One intervention design contrast per individual suffices; time columns have zero contrast.
  base=response_design(test,basis,wv,av,np.full(len(test),6),horizon)
  shiftedbase=response_design(test,basis,wv,shifted,np.full(len(test),6),horizon)
  est=target_estimable(xt[observed],(shiftedbase-base)[cov])
  if not est['estimable']:raise EstimabilityFailure('Hazard intervention contrast outside training row space '+str(est))
  model=logistic(xt[observed],rt.loc[observed,'event_observed_contiguous'],rt.loc[observed,'ipcw'],mass['hazard'])
  risks=[]
  for a in [av,shifted]:
   h=[]
   for lag in range(6,horizon+1,6):
    xp=response_design(test,basis,wv,a,np.full(len(test),lag),horizon);h.append(model.predict_proba(xp)[:,1])
   h=np.column_stack(h);risks.append(1-np.prod(1-h,axis=1))
  natural,intervened=risks
  w=rt.loc[observed,'ipcw'];wu=rt.loc[observed,'ipcw_uncapped']
  diag.update(estimability=est,censor_type='sequential6h history conditional',cap_fraction=float(wu.gt(20).mean()),
   ipcw_ESS=effective_n(w),uncapped_weight_max=float(wu.max()),min_observation_survival=float(np.exp(rt.loc[observed,'logG']).min()))
  evaluated=rv[rv.finite_wind&(rv.event_observed_contiguous|rv.lag_h.eq(horizon))].copy()
  pred=pd.Series(natural,index=test._instance)
  metric=pd.DataFrame({'_instance':evaluated._instance,'y':evaluated.event_observed_contiguous.astype(float),
       'prediction':evaluated._instance.map(pred),'weight':evaluated.ipcw,'observed':True})
 diag['shift_coverage']=float(cov.mean())
 return pd.DataFrame({'_instance':test._instance,'event_id':test.event_id,'sid':test.sid,'basin':test.basin_genesis,
     'natural':natural,'shifted':intervened,'difference':intervened-natural,'policy_applied':cov,
     'A_observed':av,'A_policy':shifted}),diag,metric

def estimate(events,long,policy,kind='spline',outcome='return48',indices=None,reference=None,identity=False):
 e,l=expanded(events,long,indices)
 if reference is None:
  er,lr=expanded(events,long);reference=reference_masses(er,lr,96 if outcome=='return96' else 48)
 predictions=[];diagnostics=[];metrics=[]
 for f in range(5):
  train=e[e.fold.ne(f)].copy();test=e[e.fold.eq(f)].copy()
  if not len(test) or not len(train):raise FitFailure('Empty SID fold')
  lt=l[l._instance.isin(train._instance)].copy();lv=l[l._instance.isin(test._instance)].copy()
  pred,diag,metric=fold_fit(train,test,lt,lv,policy,kind,outcome,reference[f],identity)
  predictions.append(pred);diagnostics.append(diag);metrics.append(metric)
 p=pd.concat(predictions,ignore_index=True);m=pd.concat(metrics,ignore_index=True)
 scale=1. if outcome=='delta24' else 100.
 status='ESTIMATED' if p.policy_applied.any() else 'IDENTITY_ONLY_NO_SUPPORTED_SHIFT'
 result={'policy':policy,'kind':kind,'outcome':outcome,'estimate':float(p.difference.mean()*scale),
   'natural_mean':float(p.natural.mean()*scale),'policy_mean':float(p.shifted.mean()*scale),'unit':'kt' if outcome=='delta24' else 'percentage points',
   'shift_coverage':float(p.policy_applied.mean()),'events':len(e),'sid':e.sid.nunique(),'status':status,
   'calibration_weight_ESS':effective_n(m.weight),'calibration_weight_sum':float(m.weight.sum()),
   'weighted_MSE_or_Brier':float(np.average((m.y-m.prediction)**2,weights=m.weight)) if m.weight.sum()>0 else np.nan}
 if outcome!='delta24':
  result['bias_tipping_gamma_pp']=result['estimate']/result['shift_coverage'] if result['shift_coverage']>0 else np.nan
 return result,p,diagnostics,m

def bh(p):
 p=np.asarray(p,float);q=np.full(len(p),np.nan);valid=np.isfinite(p);work=np.where(valid,p,1.)
 order=np.argsort(work);ranked=np.minimum.accumulate((work[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]
 q[order]=np.minimum(1,ranked);q[~valid]=np.nan
 return q
def summarize_draws(point,draw):
 rows=[]
 for z in point.to_dict('records'):
  g=draw[(draw.policy==z['policy'])&(draw.kind==z['kind'])&(draw.outcome==z['outcome'])]
  good=g[g.status.eq('ESTIMATED')&np.isfinite(g.estimate)]
  z.update(attempted_draws=len(g),successful_draws=len(good),failed_draws=len(g)-len(good))
  if len(good)>1 and z.get('status')=='ESTIMATED':
   z['ci_low'],z['ci_high']=np.quantile(good.estimate,[.025,.975])
   z['p']=min(1.,2*min((1+good.estimate.le(0).sum())/(len(good)+1),(1+good.estimate.ge(0).sum())/(len(good)+1)))
  else:z.update(ci_low=np.nan,ci_high=np.nan,p=np.nan)
  z['inference_scope']='restricted valid-draw distribution; failures disclosed' if len(good)<len(g) else 'all scheduled SID-refit draws'
  rows.append(z)
 out=pd.DataFrame(rows);out['q']=np.nan;out['family']='descriptive sensitivity; no additional family'
 for outcome,name in [('return48','main48_two_shifts'),('delta24','aux24_two_shifts')]:
  ix=out.index[(out.kind=='spline')&(out.outcome==outcome)]
  assert len(ix)==2,'Prespecified family must contain both policies'
  out.loc[ix,'q']=bh(out.loc[ix,'p']);out.loc[ix,'family']=name
 return out

def natural_bounds(events):
 rows=[]
 for horizon in [48,96]:
  y=pd.to_numeric(events[f'return{horizon}'],errors='coerce');n=len(y)
  rows.append({'horizon_h':horizon,'n':n,'confirmed':int(y.eq(1).sum()),'known_no_return':int(y.eq(0).sum()),'unknown':int(y.isna().sum()),
       'natural_risk_lower':float(y.eq(1).sum()/n),'natural_risk_upper':float((y.eq(1).sum()+y.isna().sum())/n),
       'interpretation':'bounds for observed natural outcome only, NOT causal policy bounds'})
 return pd.DataFrame(rows)

def run_bundle(events,long,B=2000,out=None,models=('spline','linear'),resume=False):
 if out is None:raise ValueError('Explicit new output directory required')
 out=Path(out)
 if V5.resolve() not in out.resolve().parents:raise ValueError('Output must remain in mechanism_revision_v5')
 def frame_identity(d):return hashlib.sha256(d.to_csv(index=False).encode('utf-8')).hexdigest()
 identity={'code_sha256':sha(__file__),'spec_sha256':sha(V5/'analysis_spec_v5.json'),
  'addendum_sha256':sha(V5/'technical_addendum_v5.json'),'baseline_sha256':frame_identity(events),
  'followup_sha256':frame_identity(long),'B':B,'seed':SEED,'models':list(models),
  'legacy_model_used':False,'claim':'cross-fitted IPCW g-computation; not double robust'}
 if resume:
  previous=json.loads((out/'run_identity.json').read_text('utf-8'))
  for k,value in identity.items():
   if previous.get(k)!=value:raise ValueError('Resume identity mismatch '+k)
 else:
  out.mkdir(parents=True,exist_ok=False)
  (out/'run_identity.json').write_text(json.dumps(dict(identity,start_utc=now()),indent=2),encoding='utf-8')
  events.to_csv(out/'model_baseline_inputs.csv',index=False);long.to_csv(out/'longitudinal_inputs.csv',index=False)
 tasks=[(p,k,y) for p in POLICIES for k in models for y in ['return48','delta24','return96']]
 er,lr=expanded(events,long);refs={h:reference_masses(er,lr,h) for h in [48,96]}
 if not resume:(out/'penalty_reference_mass.json').write_text(json.dumps(refs,indent=2),encoding='utf-8')
 point=[];pointpred=[];alldiag=[]
 if resume and (out/'point_estimates.csv').exists():
  point=readcsv(out/'point_estimates.csv').to_dict('records')
 else:
  design_arrays={};design_meta=[]
  for p in POLICIES:
   for f in range(5):
    tr=events[events.fold.ne(f)];va=events[events.fold.eq(f)]
    trans=BaselineTransform(p).fit(tr);key=f'{p}_fold{f}'
    design_arrays[key+'_W_train']=trans.transform(tr);design_arrays[key+'_W_test']=trans.transform(va)
    design_arrays[key+'_A_train']=tr[p].to_numpy();design_arrays[key+'_A_test']=va[p].to_numpy()
    design_arrays[key+'_train_event_ids']=tr.event_id.to_numpy(str);design_arrays[key+'_test_event_ids']=va.event_id.to_numpy(str)
    design_meta.append({'key':key,'transformation':trans.metadata(),'W_columns':BASE+['other_A']+
       [x+'_missing' for x in BASE+['other_A']]+['basin_'+b for b in BASINS[1:]],
       'time_and_censor_design_reconstruction':'src/p3_effects_v5.py response_design/censor_features; observed long source rows retained'})
  np.savez_compressed(out/'point_baseline_designs.npz',**design_arrays)
  (out/'point_design_manifest.json').write_text(json.dumps(design_meta,indent=2),encoding='utf-8')
  for p,k,y in tasks:
   try:
    result,pred,diag,metric=estimate(events,long,p,k,y,reference=refs[96 if y=='return96' else 48])
    point.append(result);pred['policy']=p;pred['kind']=k;pred['outcome']=y;pointpred.append(pred);alldiag.extend(diag)
    metric.to_csv(out/f'calibration_{p}_{k}_{y}.csv',index=False)
   except Exception as exc:
    point.append({'policy':p,'kind':k,'outcome':y,'estimate':np.nan,'status':'POINT_FAILED','error':str(exc)})
    (out/f'point_failure_{p}_{k}_{y}.txt').write_text(traceback.format_exc(),encoding='utf-8')
  pd.DataFrame(point).to_csv(out/'point_estimates.csv',index=False)
  if pointpred:pd.concat(pointpred).to_csv(out/'crossfit_predictions.csv',index=False)
  (out/'point_diagnostics.json').write_text(json.dumps(alldiag,indent=2,default=str),encoding='utf-8')
 rng=np.random.default_rng(SEED);sids=np.sort(events.sid.unique());sidrows={s:np.flatnonzero(events.sid.to_numpy()==s) for s in sids}
 draws=[];sid_draws=[]
 log=out/'bootstrap_draws.jsonl'
 if resume and log.exists():
  for text in log.read_text('utf-8').splitlines():
   if text.strip():draws.append(json.loads(text))
 completed={(d['draw'],d['policy'],d['kind'],d['outcome']) for d in draws}
 if len(completed)!=len(draws):raise ValueError('Duplicate completed draw/task records')
 with log.open('a' if resume else 'w',encoding='utf-8') as f:
  for b in range(B):
   selected=rng.integers(0,len(sids),len(sids));sid_draws.append(selected)
   indices=np.concatenate([sidrows[sids[i]] for i in selected])
   for p,k,y in tasks:
    key=(b,p,k,y)
    if key in completed:continue
    row={'draw':b,'policy':p,'kind':k,'outcome':y}
    try:
     result,_,diag,_=estimate(events,long,p,k,y,indices,refs[96 if y=='return96' else 48])
     row.update(result);row['max_cap_fraction']=max(d['cap_fraction'] for d in diag)
     row['min_censor_ESS']=min(d['ipcw_ESS'] for d in diag)
    except Exception as exc:
     row.update(status='CONTRAST_NOT_ESTIMABLE' if isinstance(exc,EstimabilityFailure) else 'FIT_FAILED',estimate=np.nan,error=str(exc))
    draws.append(row);completed.add(key);f.write(json.dumps(row,default=str)+'\n');f.flush()
   if b%10==0:np.savez_compressed(out/'sid_bootstrap_draw_indices.npz',sids=sids,draws=np.asarray(sid_draws))
 np.savez_compressed(out/'sid_bootstrap_draw_indices.npz',sids=sids,draws=np.asarray(sid_draws))
 d=pd.DataFrame(draws).sort_values(['draw','policy','kind','outcome']);d.to_csv(out/'bootstrap_draws.csv',index=False)
 summary=summarize_draws(pd.DataFrame(point),d);summary.to_csv(out/'effects_CI_P_q.csv',index=False)
 natural_bounds(events).to_csv(out/'natural_followup_bounds.csv',index=False)
 if (out/'crossfit_predictions.csv').exists():
  pp=readcsv(out/'crossfit_predictions.csv');pp.policy_applied=booleans(pp.policy_applied)
  support=pp.groupby(['policy','kind','outcome','basin']).agg(events=('sid','size'),SID=('sid','nunique'),shifted=('policy_applied','sum')).reset_index()
  support['coverage']=support.shifted/support.events;support.to_csv(out/'support_by_basin.csv',index=False)
 (out/'completion.json').write_text(json.dumps({'end_utc':now(),'scheduled_draws':B,'task_count':len(tasks),
   'failed_task_draws':int((d.status!='ESTIMATED').sum()),'scientific_acceptance':'pending independent review; file generation alone is not acceptance'},indent=2),encoding='utf-8')
 return summary

def completed_endpoint_scenario(events,long,scenario):
 """Explicit artificial completion sensitivity, never replaces the raw follow-up."""
 if scenario not in ['unknown_no_return','unknown_return_at_first_missing']:raise ValueError(scenario)
 z=long.copy();z['finite_wind_observed']=z.finite_wind
 z['scenario']=scenario
 for e in events.itertuples():
  ix=z.index[z.event_id.eq(e.event_id)&z.lag_h.le(48)]
  rows=z.loc[ix].sort_values('lag_h')
  if pd.notna(e.return48) and float(e.return48)==1.:
   hit=float(e.return48_first_observed_lag_h)
  elif pd.notna(e.return48) or scenario=='unknown_no_return':hit=np.inf
  else:
   missing=rows.loc[~rows.finite_wind,'lag_h']
   if not len(missing):raise ValueError('Unknown endpoint without missing slot')
   hit=float(missing.iloc[0])
  z.loc[ix,'finite_wind']=True
  z.loc[ix,'at_risk_contiguous']=z.loc[ix,'lag_h'].le(hit)
  z.loc[ix,'event_observed_contiguous']=z.loc[ix,'lag_h'].eq(hit)
  z.loc[ix,'censored_contiguous']=False
 return z

def write_post_estimation_sensitivities(events,long,out):
 out=Path(out)
 completion=[]
 for scenario in ['unknown_no_return','unknown_return_at_first_missing']:
  sl=completed_endpoint_scenario(events,long,scenario)
  for policy in POLICIES:
   try:
    r,p,d,m=estimate(events,sl,policy,'spline','return48')
    r['scenario']=scenario;r['assumption']='Artificially completed unknown outcomes; this is not a no-assumption causal bound'
    completion.append(r)
   except Exception as exc:completion.append({'scenario':scenario,'policy':policy,'status':'FAILED','error':str(exc)})
 pd.DataFrame(completion).to_csv(out/'unknown_followup_model_scenarios.csv',index=False)
 if (out/'crossfit_predictions.csv').exists():
  pred=readcsv(out/'crossfit_predictions.csv')
  rows=[]
  for (policy,kind,outcome),g in pred.groupby(['policy','kind','outcome']):
   if outcome=='delta24':continue
   shifted=booleans(g.policy_applied);coverage=float(shifted.mean());rd=float(g.difference.mean())
   if shifted.any():
    min_gamma=float(g.loc[shifted,'shifted'].max()-1);max_gamma=float(g.loc[shifted,'shifted'].min())
   else:min_gamma=max_gamma=np.nan
   for gamma_pp in [-20,-10,-5,0,5,10,20]:
    gamma=gamma_pp/100
    rows.append({'policy':policy,'kind':kind,'outcome':outcome,'gamma_pp':gamma_pp,'shift_coverage':coverage,
     'RD_corrected_pp':100*(rd-coverage*gamma),'point_tipping_gamma_pp':100*rd/coverage if coverage>0 else np.nan,
     'constant_bias_probability_feasible':bool(shifted.any() and min_gamma<=gamma<=max_gamma),
     'constant_bias_gamma_min_pp':100*min_gamma,'constant_bias_gamma_max_pp':100*max_gamma,
     'interpretation':'Assumed average bias within shifted group; not a measured confounder effect; infeasible gamma is linear approximation only'})
  pd.DataFrame(rows).to_csv(out/'unmeasured_bias_tipping_scenarios.csv',index=False)

def synthetic_data(n=250,seed=177):
 rng=np.random.default_rng(seed);rows=[];long=[]
 for i in range(n):
  sid=f'SYNTH{i:05d}';a=float(rng.uniform(4,18));rh=float(rng.uniform(35,80));t0=pd.Timestamp('2000-01-01',tz='UTC')+pd.Timedelta(days=i)
  r={name:float(rng.normal()) for name in BASE}
  r.update(sid=sid,event_id=sid+'|event',t0=t0,basin_genesis='NA',fold=fold(sid),shear200_850=a,midRH=rh,
   wind_t0_kt=65.,threshold_gap_kt=-5.,Vref_kt=70.,delta_v6_backward_kt=-5.,delta_v24_forward_kt=np.nan,
   lat=15+rng.normal(),current_land_contact=float(rng.random()<.1),past_land_contact=float(rng.random()<.2))
  if i%9==0:r['past_RH']=np.nan
  h=float(expit(-2.5-.05*(a-10)+.015*(rh-60)))
  eventstep=next((j for j in range(1,17) if rng.random()<h),99)
  censorstep=int(rng.integers(2,13)) if rng.random()<.18 else 99
  for j in range(1,17):
   finite=j<censorstep
   w=71. if j>=eventstep else 65-.3*j
   if not finite:w=np.nan
   at_risk=j<=min(eventstep,censorstep)
   long.append({'event_id':r['event_id'],'sid':sid,'t0':t0,'lag_h':j*6,'UTC':t0+pd.Timedelta(hours=j*6),
     'wind_kt':w,'Vref_kt':70.,'state_class':'A','finite_wind':finite,'observed_threshold_return':finite and j>=eventstep,
     'at_risk_contiguous':at_risk,'event_observed_contiguous':at_risk and finite and j==eventstep,
     'censored_contiguous':at_risk and not finite})
   if j==4 and finite:r['delta_v24_forward_kt']=w-65
  for h0 in [48,96]:
   k=h0//6
   returned=eventstep<=k and eventstep<censorstep
   y=1. if returned else 0. if censorstep>k else np.nan
   r['return'+str(h0)]=y;r[f'return{h0}_first_observed_lag_h']=eventstep*6 if returned else np.nan
  rows.append(r)
 e=pd.DataFrame(rows);l=long_history(e,pd.DataFrame(long))
 return e,l

def self_test():
 start=now()
 check('boolean_false_explicit',booleans(pd.Series(['False','True','0','1'])).tolist()==[False,True,False,True])
 check('constant_hazard_risk',abs((1-np.prod(1-np.full(8,.1)))-(1-.9**8))<1e-14)
 e,l=synthetic_data()
 er,lr=expanded(e,l,[0,0,1,2])
 check('bootstrap_duplicates_keep_SID_fold',er.groupby('sid').fold.nunique().max()==1 and len(er)==4)
 check('bootstrap_duplicate_longitudinal_complete',lr.groupby('_instance').size().eq(16).all())
 check('no_cross_SID_long_join',lr.groupby('_instance').sid.nunique().eq(1).all())
 tr=BaselineTransform('shear200_850').fit(e.iloc[:150])
 before=tr.median.copy();modified=e.iloc[150:].copy();modified['past_RH']=1e9;tr.transform(modified)
 check('missing_W_median_training_only',np.array_equal(before,tr.median))
 future=e.copy();future['future_LMI']=999;future['return48']=1
 check('future_outcome_columns_never_enter_W',np.allclose(tr.raw(e),tr.raw(future),equal_nan=True))
 s=ExposureBasis('spline').fit(e.shear200_850);check('natural_cubic_three_columns',s.transform(e.shear200_850).shape[1]==3)
 tail=s.raw(np.array([s.knots[-1]+1,s.knots[-1]+2,s.knots[-1]+3]))
 check('natural_cubic_linear_tail',np.max(np.abs(np.diff(tail,n=2,axis=0)))<1e-10)
 try:ExposureBasis('spline').fit(np.ones(50));failed=False
 except EstimabilityFailure:failed=True
 check('degenerate_knots_failure_retained',failed)
 X=np.column_stack([np.arange(1,21.),2*np.arange(1,21.)])
 check('redundant_model_estimable_target',target_estimable(X,np.array([[1.,2.]]))['estimable'])
 check('redundant_model_unestimable_target',not target_estimable(X,np.array([[1.,0.]]))['estimable'])
 rng=np.random.default_rng(3);xx=rng.normal(size=(80,3));yy=(rng.random(80)<expit(xx[:,0])).astype(int)
 m1=logistic(xx,yy,np.ones(80),80);m2=logistic(np.repeat(xx,2,axis=0),np.repeat(yy,2),np.ones(160),80)
 check('fixed_reference_penalty_duplicate_invariance',np.max(abs(m1.coef_-m2.coef_))<1e-6)
 refs=reference_masses(*expanded(e,l),48)
 identity,pi,di,mi=estimate(e,l,'shear200_850',identity=True,reference=refs)
 check('identity_policy_exact_zero',identity['estimate']==0 and pi.difference.eq(0).all())
 result,pred,diag,metric=estimate(e,l,'shear200_850',reference=refs)
 check('crossfit_synthetic_completed',np.isfinite(result['estimate']) and len(pred)==len(e))
 check('risk_probabilities_bounded',pred[['natural','shifted']].ge(0).all().all() and pred[['natural','shifted']].le(1).all().all())
 check('outside_support_identity',pred.loc[~pred.policy_applied,'A_observed'].eq(pred.loc[~pred.policy_applied,'A_policy']).all())
 check('inside_shear_exact_shift',np.allclose(pred.loc[pred.policy_applied,'A_policy']-pred.loc[pred.policy_applied,'A_observed'],-2))
 rr,pp,dd,mm=estimate(e,l,'midRH',kind='linear',outcome='delta24',reference=refs)
 check('RH_shift_five_pp',np.allclose(pp.loc[pp.policy_applied,'A_policy']-pp.loc[pp.policy_applied,'A_observed'],5))
 check('endpoint_observation_model_separate',all(d['censor_type']=='exact24h endpoint observation' for d in dd))
 r96,p96,d96,m96=estimate(e,l,'shear200_850',kind='linear',outcome='return96')
 check('96h_horizon_runs',np.isfinite(r96['estimate']))
 b=natural_bounds(e);check('unknown_followup_bounds_not_zero',b.loc[b.horizon_h.eq(48),'natural_risk_upper'].iloc[0]>b.loc[b.horizon_h.eq(48),'natural_risk_lower'].iloc[0])
 fake=pd.DataFrame([{'policy':p,'kind':'spline','outcome':y,'estimate':1.,'status':'ESTIMATED'} for p in POLICIES for y in ['return48','delta24']])
 dr=[]
 for z in fake.to_dict('records'):
  for k in range(5):dr.append(dict(z,draw=k,estimate=float(k-2)))
 sm=summarize_draws(fake,pd.DataFrame(dr))
 check('BH_two_separate_complete_families',sm.groupby('family').size().to_dict()=={'aux24_two_shifts':2,'main48_two_shifts':2})
 check('CI_from_direct_effect_draws',np.allclose(sm.ci_low,-1.9) and np.allclose(sm.ci_high,1.9))
 for sc in ['unknown_no_return','unknown_return_at_first_missing']:
  ss=completed_endpoint_scenario(e,l,sc)
  check('explicit_completion_'+sc,ss[ss.lag_h.le(48)].finite_wind.all())
 report={'status':'PASS','start_utc':start,'end_utc':now(),'tests_n':len(TESTS),'tests':TESTS,
    'synthetic_only':True,'real_P3_effects_run':False,'code_sha256':sha(__file__),
    'scope':'implementation smoke and meaningful mathematical invariants; not actual-data scientific acceptance',
    'bootstrap_full2000_executed':False,'synthetic_point_summary_for_debug_only':result}
 out=V5/'audit/p3_effects_synthetic_validation.json'
 out.write_text(json.dumps(report,indent=2,default=str),encoding='utf-8')
 print(json.dumps({k:report[k] for k in ['status','tests_n','start_utc','end_utc','real_P3_effects_run']},indent=2))
 return report

def cli():
 parser=argparse.ArgumentParser()
 parser.add_argument('--self-test',action='store_true')
 parser.add_argument('--prepare-input',type=Path)
 parser.add_argument('--run',action='store_true')
 parser.add_argument('--resume',action='store_true')
 parser.add_argument('--exposures',type=Path)
 parser.add_argument('--cohort',choices=['main','nonoverlap96'],default='main')
 parser.add_argument('--B',type=int,default=2000)
 parser.add_argument('--out',type=Path)
 args=parser.parse_args()
 if args.self_test:return self_test()
 if args.prepare_input:
  e,l,ex,meta=assemble_inputs(args.prepare_input,args.cohort)
  if args.out is None:raise ValueError('Explicit output directory required')
  if V5.resolve() not in args.out.resolve().parents:raise ValueError('Output must remain in mechanism_revision_v5')
  args.out.mkdir(parents=True,exist_ok=False)
  e.to_csv(args.out/'baseline_inputs.csv',index=False);l.to_csv(args.out/'followup_inputs.csv',index=False)
  ex.to_csv(args.out/'exposure_exclusions.csv',index=False)
  (args.out/'input_readiness.json').write_text(json.dumps(meta,indent=2,default=str),encoding='utf-8')
  print(json.dumps(meta,indent=2));return
 if args.run:
  if args.exposures is None or args.out is None:raise ValueError('--run requires --exposures and --out')
  if args.B<1:raise ValueError('B must be positive')
  e,l,ex,meta=assemble_inputs(args.exposures,args.cohort)
  result=run_bundle(e,l,args.B,args.out,resume=args.resume)
  ex.to_csv(args.out/'exposure_exclusions.csv',index=False)
  (args.out/'input_readiness.json').write_text(json.dumps(meta,indent=2,default=str),encoding='utf-8')
  write_post_estimation_sensitivities(e,l,args.out)
  print(result.to_string(index=False));return
 parser.print_help()

if __name__=='__main__':
 try:cli()
 except BaseException:
  p=V5/'audit'/('p3_effects_failure_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'.json')
  p.write_text(json.dumps({'UTC':now(),'traceback':traceback.format_exc(),'synthetic_tests':TESTS},indent=2),encoding='utf-8')
  raise

