# Public path adapter; the scientific functions below retain their original implementation.
import sys as _sys
from pathlib import Path as _Path
_PACKAGE=next(p for p in _Path(__file__).resolve().parents if (p/'release_paths.py').is_file())
_sys.path.insert(0,str(_PACKAGE))
from release_paths import ADDENDUM, UNIFIED, PACKAGE, bootstrap
from pathlib import Path
import runpy,json,time,sys
ROOT=Path(__file__).resolve().parent
SUPPORT_MODE='complete' if '--complete-case-sensitivity' in sys.argv else 'overlap'
O=ROOT if SUPPORT_MODE=='overlap' else ROOT/'sensitivity_complete'
O.mkdir(exist_ok=True)
bootstrap(O/'temp')
import numpy as np
import pandas as pd
def expit(x):return 1/(1+np.exp(-np.clip(x,-700,700)))

def read(n):return pd.read_csv(ROOT/n,keep_default_na=False,na_values=['','nan','NaN'])
def save(d,n):d.to_csv(O/n,index=False)
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,(np.integer,np.bool_)):return x.item()
 if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
 return x
def js(n,x):(O/n).write_text(json.dumps(clean(x),indent=2,allow_nan=False),encoding='utf8')
def wilson(k,n):
 if n==0:return np.nan,np.nan
 z=1.959963984540054;p=k/n;c=(p+z*z/(2*n))/(1+z*z/n);h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n);return c-h,c+h

m=read('sid_denominators.csv').sort_values('sid').reset_index(drop=True)
ref=read('first_departure_reference.csv').set_index('sid').loc[m.sid].reset_index()
risk=read('risk_intervals_6h.csv');N=len(m);years=np.arange(1982,2024);iy={y:np.flatnonzero(m.genesis_year.to_numpy()==y) for y in years};ix={s:i for i,s in enumerate(m.sid)}
basins=['NA','EP','WP','NI','SI','SP','SA'];major=['NA','EP','WP','SI'];bi={b:j for j,b in enumerate(basins)}
ex=np.zeros((N,7));events=np.zeros_like(ex);exstart=ex.copy();evstart=ex.copy()
for (s,b),g in risk.groupby(['sid','basin_end']):ex[ix[s],bi[b]]=len(g)*6;events[ix[s],bi[b]]=g.return_event.sum()
for (s,b),g in risk.groupby(['sid','basin_start']):exstart[ix[s],bi[b]]=len(g)*6;evstart[ix[s],bi[b]]=g.return_event.sum()
gen=np.array([m.basin_genesis.eq(b) for b in basins]).T.astype(float);ret=gen*m.ever_return.to_numpy()[:,None]
def weights(rng,L):
 yy=np.concatenate([(rng.integers(0,42)+np.arange(L))%42 for j in range(int(np.ceil(42/L)))])[:42]
 w=np.zeros(N)
 for yi in yy:
  ids=iy[years[yi]]
  if len(ids):np.add.at(w,rng.choice(ids,len(ids),replace=True),1)
 return w

# Same support throughout all three probability comparisons. Four basins have
# >=20 total return SIDs. Sparse NI/SP are retained descriptively only.
ref['abs_lat']=ref.lat.abs();ref['year_decade']=(ref.genesis_year-2002.5)/10
ref['season_sin']=ref.month_sin*np.where(ref.lat<0,-1,1)
ref['season_cos']=ref.month_cos*np.where(ref.lat<0,-1,1)
state=['season_sin','season_cos','year_decade','abs_lat','wind_kt','age_h','motion_past6h_ms','past24_land_contact']
environment=['X3_mid_rh_pct','X2_shear_ms','X1_sst_c']
cols=state+environment
eligible=ref.basin_genesis.isin(major)&ref[cols].notna().all(axis=1)
trimvars=['abs_lat','wind_kt','age_h','motion_past6h_ms']+environment
ranges=[];supported=eligible.copy()
for c in trimvars:
 qs=ref[eligible].groupby('basin_genesis')[c].quantile([.01,.99]).unstack()
 lo,hi=qs[.01].max(),qs[.99].min();supported &= ref[c].between(lo,hi)
 ranges.append(dict(variable=c,low=lo,high=hi,basin_quantiles=qs.to_dict()))
if SUPPORT_MODE=='complete':supported=eligible.copy()
ref['complete_major']=eligible;ref['common_support']=supported
save(ref,'analysis_support_all_sid.csv');js('support_ranges.json',ranges)
d=ref[supported].copy();idx=np.flatnonzero(supported);y=d.ever_return.to_numpy(float);b=d.basin_genesis.to_numpy();n=len(d)
assert n>500 and y.sum()>30
features=d[cols].to_numpy(float);mu=features.mean(axis=0);sd=features.std(axis=0);Z=(features-mu)/sd
D=np.column_stack([np.ones(n)]+[(b==x).astype(float) for x in major[1:]])
X1=np.column_stack([D,Z[:,:len(state)]]);X2=np.column_stack([D,Z]); designs=[X1,X2]
penalties=[np.r_[0.,np.repeat(.16,3),np.ones(len(state))],np.r_[0.,np.repeat(.16,3),np.ones(len(cols))]]
def fit(X,y,w,pen,beta=None):
 beta=np.zeros(X.shape[1]) if beta is None else beta.copy()
 if beta is None:beta[0]=-3
 for it in range(80):
  p=expit(X@beta);v=np.maximum(p*(1-p),1e-8)*w
  H=X.T@(v[:,None]*X)+np.diag(pen)
  grad=X.T@(w*(p-y))+pen*beta
  step=np.linalg.solve(H+np.eye(H.shape[0])*1e-9,grad)
  # A bounded Newton step keeps rare resamples stable without changing penalty.
  def loss(v):
   eta=X@v
   return np.sum(w*(np.logaddexp(0,eta)-y*eta))+.5*np.sum(pen*v*v)
  old=loss(beta);alpha=1.
  while alpha>1e-9 and loss(beta-alpha*step)>old+1e-10:alpha*=.5
  beta-=alpha*step
  if np.max(np.abs(alpha*step))<1e-7:return beta,it+1
 raise RuntimeError('logit did not converge')
def prob(X,coef,w):
 ans=[]
 for basin in major:
  xx=X.copy();xx[:,1:4]=np.array([basin==q for q in major[1:]])
  ans.append(np.average(expit(xx@coef),weights=w))
 return np.array(ans)
def calc(w,init=None):
 ww=w[idx];raw=np.array([np.average(y[b==q],weights=ww[b==q]) for q in major]);co=[];pr=[raw];its=[]
 for j,(X,pen) in enumerate(zip(designs,penalties)):
  c,i=fit(X,y,ww,pen,None if init is None else init[j]);co.append(c);its.append(i);pr.append(prob(X,c,ww))
 return np.array(pr),co,its
point,coef,iterations=calc(np.ones(N));B=2000 if SUPPORT_MODE=='overlap' else 1000;seed=2026110901;rng=np.random.default_rng(seed)
draw=np.zeros((B,3,4));countdraw=np.zeros((B,7,3));cfs=np.zeros((B,X2.shape[1]));envmean=np.zeros((B,4,3));elapsed=time.monotonic()
for j in range(B):
 w=weights(rng,3);draw[j],cc,it=calc(w,coef);cfs[j]=cc[1]
 den=w@gen;k=w@ret;h=w@ex;ee=w@events
 countdraw[j,:,0]=np.divide(k,den,out=np.full(7,np.nan),where=den>0)
 countdraw[j,:,1]=np.divide(ee,h,out=np.full(7,np.nan),where=h>0)*1000
 countdraw[j,:,2]=ee
 for q,basin in enumerate(major):
  hit=(b==basin);envmean[j,q]=np.average(d.loc[hit,environment],axis=0,weights=w[idx][hit])
 if (j+1)%100==0:
  js('CHECKPOINT_model.json',dict(status='RUNNING',draws_done=j+1,draws_planned=B,n_support=n,events_support=int(y.sum()),elapsed_s=time.monotonic()-elapsed));print('bootstrap',j+1,flush=True)
np.savez_compressed(O/'bootstrap_draws.npz',probability=draw,basin_counts=countdraw,environment_coefficients=cfs,environment_means=envmean)
rows=[]
for q,basin in enumerate(basins):
 n0=int(gen[:,q].sum());k=int(ret[:,q].sum());lo,hi=wilson(k,n0);h=ex[:,q].sum();e=events[:,q].sum();ci=np.nanquantile(countdraw[:,q], [.025,.975],axis=0) if n0 else np.full((2,3),np.nan)
 rows.append(dict(basin=basin,eligible_sid=n0,returning_sid=k,proportion_pct=k/n0*100 if n0 else np.nan,proportion_pct_low=ci[0,0]*100,proportion_pct_high=ci[1,0]*100,wilson_low_pct=lo*100,wilson_high_pct=hi*100,event_count=int(e),risk_hours=int(h),rate_per1000h=e/h*1000 if h else np.nan,rate_low=ci[0,1],rate_high=ci[1,1],risk_sid=int((ex[:,q]>0).sum()),start_basin_events=int(evstart[:,q].sum()),start_basin_hours=int(exstart[:,q].sum()),start_basin_rate_per1000h=evstart[:,q].sum()/exstart[:,q].sum()*1000 if exstart[:,q].sum() else np.nan))
save(pd.DataFrame(rows),'basin_estimates_ci.csv')
rows=[]
for z,model in enumerate(['unadjusted','season_state','season_state_environment']):
 for q,basin in enumerate(major):
  ci=np.quantile(draw[:,z,q],[.025,.975]);rows.append(dict(model=model,basin=basin,n_sid=int((b==basin).sum()),events=int(y[b==basin].sum()),probability_pct=100*point[z,q],ci_low_pct=100*ci[0],ci_high_pct=100*ci[1]))
save(pd.DataFrame(rows),'standardized_probabilities.csv')
rows=[]
for a in range(4):
 for bb in range(a+1,4):
  for z,model in enumerate(['unadjusted','season_state','season_state_environment']):
   dd=100*(draw[:,z,a]-draw[:,z,bb]);ci=np.quantile(dd,[.025,.975]);rows.append(dict(contrast=major[a]+'-'+major[bb],model=model,difference_pp=100*(point[z,a]-point[z,bb]),ci_low=ci[0],ci_high=ci[1],bootstrap_sign_p=2*min(np.mean(dd<=0),np.mean(dd>=0))))
  atten=100*((draw[:,1,a]-draw[:,1,bb])-(draw[:,2,a]-draw[:,2,bb]));ci=np.quantile(atten,[.025,.975]);rows.append(dict(contrast=major[a]+'-'+major[bb],model='change_after_environment',difference_pp=100*((point[1,a]-point[1,bb])-(point[2,a]-point[2,bb])),ci_low=ci[0],ci_high=ci[1],bootstrap_sign_p=2*min(np.mean(atten<=0),np.mean(atten>=0))))
save(pd.DataFrame(rows),'basin_probability_differences.csv')
er=[]
for j,c in enumerate(environment):
 ci=np.quantile(cfs[:,4+len(state)+j]/sd[len(state)+j],[.025,.975]);beta=coef[1][4+len(state)+j]/sd[len(state)+j]
 er.append(dict(variable=c,log_odds_per_unit=beta,ci_low=ci[0],ci_high=ci[1],unit='percentage point RH' if j==0 else ('m/s' if j==1 else 'degC'),penalty_per_standardized_coefficient=1))
save(pd.DataFrame(er),'environment_associations.csv')
er=[]
for a,basin in enumerate(major):
 for j,c in enumerate(environment):
  ci=np.quantile(envmean[:,a,j],[.025,.975]);er.append(dict(basin=basin,variable=c,mean=float(d.loc[b==basin,c].mean()),ci_low=ci[0],ci_high=ci[1],n_sid=int((b==basin).sum())))
save(pd.DataFrame(er),'environment_by_basin.csv')
sens=[]
for L in [1,5]:
 dd=[];rgen=np.random.default_rng(seed+L)
 for j in range(500):dd.append(calc(weights(rgen,L),coef)[0])
 dd=np.array(dd);np.save(O/f'sensitivity_block{L}.npy',dd)
 for a in range(4):
  for bb in range(a+1,4):
   for z,label in enumerate(['unadjusted','season_state','season_state_environment']):
    ci=np.quantile(100*(dd[:,z,a]-dd[:,z,bb]),[.025,.975]);sens.append(dict(block_years=L,contrast=major[a]+'-'+major[bb],model=label,ci_low=ci[0],ci_high=ci[1]))
save(pd.DataFrame(sens),'block_length_sensitivity.csv')
support_summary=d.groupby('basin_genesis').agg(n=('sid','size'),events=('ever_return','sum'));save(support_summary.reset_index(),'common_support_counts.csv')
# Model diagnostics are descriptive; no post-hoc feature selection.
p=expit(X2@coef[1]);diag=dict(n=n,events=int(y.sum()),n_parameters=X2.shape[1],condition_number=float(np.linalg.cond(X2.T@(p[:,None]*(1-p[:,None])*X2)+np.diag(penalties[1]))),max_abs_standardized_coefficient=float(np.max(np.abs(coef[1]))),brier=float(np.mean((y-p)**2)),iterations=iterations,bootstrap_failed=0,raw_yearly_outcome_lag1_autocorrelation=float(m.groupby('genesis_year').ever_return.mean().autocorr()))
js('MODEL_CONFIG.json',dict(seed=seed,bootstrap_replicates=B,block_years=3,bootstrap='Circular moving blocks of global genesis years, same selected years for all basins; resample entire SIDs within each selected year; retain all recurrent observations of each SID together',sensitivity_block_years=[1,5],sensitivity_replicates=500,models=['raw empirical proportion','ridge logistic basin + season + year + initial state','same model plus RH, shear, SST'],state_variables=state,environment_variables=environment,continuous_penalty=1,intercept_penalty=0,basin_penalty=.16,basin_regularization_reason='Unpenalized basin pilot failed at a separated resample; weak Normal(0,2.5^2) contrast penalty added before completing any inference; no resamples dropped',standardization_target='pooled observed common-support SID covariate distribution, bootstrap reweighted',support='complete covariates in NA EP WP SI, trimmed to intersection of within-basin 1st-99th percentile intervals for prespecified continuous covariates',reference='first observed downcross after first-LMI anchored near-peak window; all exposures at that time, outcomes strictly later',outcome='at least one qualifying subsequent return within the recorded lifecycle, not a fixed-48h outcome',followup='observed-lifecycle probability does not identify the probability beyond record termination; risk intervals stop at first missing interval in each departure episode; no extrapolation',diagnostics=diag))
config=json.loads((O/'MODEL_CONFIG.json').read_text());config['support_mode']=SUPPORT_MODE
if SUPPORT_MODE=='complete':config['support']='Sensitivity only: all complete cases in the four adequately sampled basins; no marginal-range trimming, hence more extrapolation than primary common-support analysis'
js('MODEL_CONFIG.json',config)
js('CHECKPOINT_model.json',dict(status='COMPLETE',draws_done=B,draws_planned=B,n_support=n,events_support=int(y.sum()),elapsed_s=time.monotonic()-elapsed,bootstrap_failed=0,support_mode=SUPPORT_MODE))
print('COMPLETE',support_summary.to_string(),json.dumps(diag),flush=True)
