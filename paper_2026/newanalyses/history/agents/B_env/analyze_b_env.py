# Public path adapter; the scientific functions below retain their original implementation.
import sys as _sys
from pathlib import Path as _Path
_PACKAGE=next(p for p in _Path(__file__).resolve().parents if (p/'release_paths.py').is_file())
_sys.path.insert(0,str(_PACKAGE))
from release_paths import ADDENDUM, UNIFIED, PACKAGE, bootstrap
from pathlib import Path
import runpy,json,sys,hashlib,time,ast
ROOT=Path(__file__).resolve().parent
MODE='temporal_overlap' if '--temporal-overlap' in sys.argv else 'A_overlap'
O=ROOT if MODE=='A_overlap' else ROOT/'sensitivity_temporal_overlap'
O.mkdir(exist_ok=True)
bootstrap(O/'temp')
import numpy as np
import pandas as pd
AP=ROOT.parent/'A';SOURCE=AP/'analysis_support_all_sid.csv'
allref=pd.read_csv(SOURCE,keep_default_na=False,na_values=['','nan','NaN']).sort_values('sid').reset_index(drop=True)
assert len(allref)==2340 and allref.sid.is_unique and allref.ever_return.sum()==123 and allref.n_return.sum()==133
basins=['NA','EP','WP','SI'];years=np.arange(1982,2024);tt=(years-2002.5)/10;unit=['RH percentage points/decade','m/s/decade','degC/decade','m/s/decade','land-contact percentage points/decade']
envcols=['X3_mid_rh_pct','X2_shear_ms','X1_sst_c'];state=['season_sin','season_cos','abs_lat','wind_kt','age_h','motion_past6h_ms','past24_land_contact'];features=state+envcols
trendcols=envcols+['motion_past6h_ms','past24_land_contact']
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,(np.integer,np.bool_)):return x.item()
 if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
 return x
def js(n,x):(O/n).write_text(json.dumps(clean(x),indent=2,allow_nan=False),encoding='utf8')
def save(d,n):d.to_csv(O/n,index=False)
def slope(x,y,w):
 x=np.asarray(x,float);y=np.asarray(y,float);w=np.asarray(w,float);keep=np.isfinite(x)&np.isfinite(y)&(w>0)
 x,y,w=x[keep],y[keep],w[keep];xm=np.average(x,weights=w);ym=np.average(y,weights=w)
 return np.sum(w*(x-xm)*(y-ym))/np.sum(w*(x-xm)**2)
def bh(p):
 p=np.asarray(p);ii=np.argsort(p);q=np.empty(len(p));q[ii]=np.minimum(1,np.minimum.accumulate((p[ii]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]);return q

support=allref.common_support.astype(bool).copy();ranges=[]
if MODE=='temporal_overlap':
 support[:]=False
 for basin in basins:
  valid=allref.basin_genesis.eq(basin)&allref[features].notna().all(axis=1)
  take=valid.copy()
  for c in ['abs_lat','wind_kt','age_h','motion_past6h_ms']+envcols:
   q=allref.loc[valid].groupby('period')[c].quantile([.01,.99]).unstack();lo,hi=q[.01].max(),q[.99].min();take&=allref[c].between(lo,hi)
   ranges.append(dict(basin=basin,variable=c,lo=lo,hi=hi,period_quantiles=q.to_dict()))
  support|=take
d=allref[support].copy().reset_index(drop=True);Nall=len(allref);N=len(d);idx=np.flatnonzero(support)
assert d[features].notna().all().all();assert set(d.basin_genesis)==set(basins)
allref['B_env_support']=support;save(allref,'support_all_sid.csv');js('support_ranges.json',ranges)
supporttab=d.groupby(['basin_genesis','period']).agg(n_sid=('sid','size'),return_sid=('ever_return','sum'),n_years=('genesis_year','nunique')).reset_index();save(supporttab,'support_by_period.csv')
supportyear=d.groupby(['basin_genesis','genesis_year']).agg(n_sid=('sid','size'),return_sid=('ever_return','sum')).reindex(pd.MultiIndex.from_product([basins,years],names=['basin_genesis','genesis_year'])).reset_index();supportyear['has_denominator']=supportyear.n_sid.notna();save(supportyear,'support_by_year.csv')
b=d.basin_genesis.to_numpy();t=(d.genesis_year.to_numpy()-2002.5)/10;y=d.ever_return.to_numpy(float)
basinix=[np.flatnonzero(b==bb) for bb in basins];yearix=[[np.flatnonzero((b==bb)&(d.genesis_year.to_numpy()==yr)) for yr in years] for bb in basins]
globalyearix=[np.flatnonzero(allref.genesis_year.to_numpy()==yr) for yr in years]
mu=d[features].mean().to_numpy();sd=d[features].std(ddof=0).to_numpy();z=(d[features].to_numpy()-mu)/sd
base=np.column_stack([np.ones(N)]+[(b==bb).astype(float) for bb in basins[1:]]+[t]+[t*(b==bb) for bb in basins[1:]])
X=[base,np.column_stack([base,z[:,:len(state)]]),np.column_stack([base,z])]
penbase=np.array([0,.16,.16,.16,0,.16,.16,.16]);P=[penbase,np.r_[penbase,np.ones(len(state))],np.r_[penbase,np.ones(len(features))]]
# Load only the previously verified pure numerical functions, without executing A.
tree=ast.parse((AP/'model_a.py').read_text());ns={'np':np};nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['fit','expit']];assert len(nodes)==2
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(AP/'model_a.py')+':pure_functions','exec'),ns);fit=ns['fit'];expit=ns['expit']

def balance(w,yearmult):
 raw=w[idx];ans=np.zeros(N)
 for a,ii in enumerate(basinix):
  for j,jj in enumerate(yearix[a]):
   total=raw[jj].sum()
   if total>0:ans[jj]=raw[jj]/total*yearmult[j]
  if ans[ii].sum()>0:ans[ii]*=len(ii)/ans[ii].sum()
 return ans
def bootstrap(rng,L):
 selected=np.concatenate([(rng.integers(0,42)+np.arange(L))%42 for j in range(int(np.ceil(42/L)))])[:42];ym=np.bincount(selected,minlength=42);w=np.zeros(Nall)
 for yi in selected:
  ii=globalyearix[yi]
  if len(ii):np.add.at(w,rng.choice(ii,len(ii),replace=True),1)
 return w,ym
def calculate(w,ym,init=None):
 fw=balance(w,ym);coeff=[];curves=np.empty((3,4,42));slopes=np.empty((3,4));iterations=[]
 for mod in range(3):
  c,it=fit(X[mod],y,fw,P[mod],None if init is None else init[mod]);coeff.append(c);iterations.append(it)
  for a,ii in enumerate(basinix):
   temporal=c[4]+(c[4+a] if a>0 else 0);baseeta=X[mod][ii]@c-temporal*t[ii]
   pr=expit(baseeta[None,:]+tt[:,None]*temporal);curves[mod,a]=np.average(pr,axis=1,weights=fw[ii]);slopes[mod,a]=slope(tt,curves[mod,a]*100,np.ones(42))
 ev=np.empty((4,5))
 for a,ii in enumerate(basinix):
  for k,col in enumerate(trendcols):ev[a,k]=slope(t[ii],d[col].to_numpy()[ii],fw[ii])*(100 if col=='past24_land_contact' else 1)
 actualcurve=np.full((4,42),np.nan);same=np.empty((4,3));p=expit(X[2]@coeff[2]);raw=w[idx]
 for a,ii in enumerate(basinix):
  for j,jj in enumerate(yearix[a]):
   if raw[jj].sum()>0:actualcurve[a,j]=np.average(p[jj],weights=raw[jj])
  active=np.isfinite(actualcurve[a]);same[a,0]=slope(tt,actualcurve[a]*100,ym);same[a,1]=slope(tt[active],curves[2,a,active]*100,ym[active]);same[a,2]=same[a,0]-same[a,1]
 return slopes,curves,coeff,ev,same,actualcurve,iterations

point,curves,coef,envpoint,samepoint,actualcurve,iters=calculate(np.ones(Nall),np.ones(42))
B=2000 if MODE=='A_overlap' else 1000;seed=2026110903;rng=np.random.default_rng(seed);start=time.monotonic()
dr=np.empty((B,3,4));edr=np.empty((B,4,5));sdr=np.empty((B,4,3));cdr=np.empty((B,3,4,42));coeffdr=np.empty((B,X[2].shape[1]));maxiter=0
for k in range(B):
 w,ym=bootstrap(rng,3);sl,cu,co,ee,ss,ac,it=calculate(w,ym,coef);dr[k]=sl;edr[k]=ee;sdr[k]=ss;cdr[k]=cu;coeffdr[k]=co[2];maxiter=max(maxiter,max(it))
 if (k+1)%100==0:
  js('CHECKPOINT.json',dict(status='RUNNING',support=MODE,n_sid=N,returns=int(y.sum()),draws_completed=k+1,planned=B,elapsed_s=time.monotonic()-start));print(MODE,'draw',k+1,'elapsed',round(time.monotonic()-start,1),flush=True)
np.savez_compressed(O/'bootstrap_draws.npz',trend=dr,environment_trend=edr,same_model=sdr,curves=cdr,full_coefficients=coeffdr)
labels=['natural_composition','fixed_state','fixed_state_environment'];rows=[];cd=[]
for m,label in enumerate(labels):
 for a,bb in enumerate(basins):
  ci=np.quantile(dr[:,m,a],[.025,.975]);rows.append(dict(model=label,basin=bb,estimate_pp_per_decade=point[m,a],ci_low=ci[0],ci_high=ci[1],n_sid=len(basinix[a]),return_sid=int(y[basinix[a]].sum()),observed_years=d.iloc[basinix[a]].genesis_year.nunique(),curve_years=42))
  for j,yr in enumerate(years):
   ci=np.quantile(cdr[:,m,a,j],[.025,.975])*100;cd.append(dict(model=label,basin=bb,year=int(yr),probability_pct=curves[m,a,j]*100,ci_low=ci[0],ci_high=ci[1],observed_n=len(yearix[a][j]),empirical_denominator_available=bool(len(yearix[a][j]))))
save(pd.DataFrame(rows),'standardized_trend_estimates.csv');save(pd.DataFrame(cd),'annual_standardized_probability_curves.csv')
rows=[]
for l,r,label in [(0,1,'natural_minus_fixed_state'),(1,2,'fixed_state_minus_fixed_environment')]:
 for a,bb in enumerate(basins):
   dd=dr[:,l,a]-dr[:,r,a];ci=np.quantile(dd,[.025,.975]);rows.append(dict(contrast=label,basin=bb,estimate_pp_per_decade=point[l,a]-point[r,a],ci_low=ci[0],ci_high=ci[1],bootstrap_sign_p=min(1.,2*min(np.mean(dd<=0),np.mean(dd>=0)))))
z=pd.DataFrame(rows);z['q_value']=np.nan;mask=z.contrast.eq('fixed_state_minus_fixed_environment');z.loc[mask,'q_value']=bh(z.loc[mask,'bootstrap_sign_p']);z['family']=np.where(mask,'B_environment_standardization_four_basins','state_diagnostic_descriptive');save(z,'standardization_trend_differences.csv')
rows=[]
for a,bb in enumerate(basins):
 for k,col in enumerate(trendcols):
  dd=edr[:,a,k];ci=np.quantile(dd,[.025,.975]);rows.append(dict(basin=bb,variable=col,unit=unit[k],estimate=envpoint[a,k],ci_low=ci[0],ci_high=ci[1],bootstrap_sign_p=min(1.,2*min(np.mean(dd<=0),np.mean(dd>=0))),variation_status='constant_in_support_descriptive_only' if d.iloc[basinix[a]][col].nunique()==1 else 'variable'))
z=pd.DataFrame(rows);z['q_value']=bh(z.bootstrap_sign_p);z['family']='B_within_basin_environment_history_20';save(z,'environment_history_trends.csv')
rows=[]
for a,bb in enumerate(basins):
 for k,label in enumerate(['actual_annual_distribution','fixed_joint_distribution_same_years','actual_minus_fixed_joint']):
  ci=np.quantile(sdr[:,a,k],[.025,.975]);rows.append(dict(basin=bb,estimand=label,estimate_pp_per_decade=samepoint[a,k],ci_low=ci[0],ci_high=ci[1],model='M2 identical coefficient vector; annual covariate distribution versus fixed joint within-basin reference; both on observed calendar years'))
save(pd.DataFrame(rows),'same_model_actual_vs_fixed_joint.csv')
annual=d.groupby(['basin_genesis','genesis_year'])[trendcols].agg(['mean','count']);annual.columns=['_'.join(c) for c in annual.columns];save(annual.reset_index(),'annual_environment_means.csv')
sens=[]
if MODE=='A_overlap':
 for L in [1,5]:
  draws=[];rgen=np.random.default_rng(seed+L)
  for k in range(500):
   w,ym=bootstrap(rgen,L);draws.append(calculate(w,ym,coef)[0])
  dd=np.asarray(draws);np.save(O/f'block{L}_sensitivity_draws.npy',dd)
  for m,label in enumerate(labels):
   for a,bb in enumerate(basins):
    ci=np.quantile(dd[:,m,a],[.025,.975]);sens.append(dict(block_years=L,model=label,basin=bb,ci_low=ci[0],ci_high=ci[1]))
 save(pd.DataFrame(sens),'block_length_sensitivity.csv')
diag=[];fw=balance(np.ones(Nall),np.ones(42))
for j in range(3):
 p=expit(X[j]@coef[j]);grad=X[j].T@(fw*(p-y))+P[j]*coef[j];h=X[j].T@((fw*p*(1-p))[:,None]*X[j])+np.diag(P[j]);diag.append(dict(model=labels[j],n_parameters=X[j].shape[1],score_max_abs=float(abs(grad).max()),hessian_condition=float(np.linalg.cond(h)),iterations=iters[j],unweighted_brier=float(np.mean((p-y)**2))))
js('MODEL_CONFIG.json',dict(support=MODE,n_sid=N,returns=int(y.sum()),seed=seed,B=B,block_years=3,state_variables=state,environment_variables=envcols,basins=basins,years=years,regularization=P,coefficients=coef,feature_mean=mu,feature_sd=sd,diagnostics=diag,bootstrap_max_iterations=maxiter,source_path=str(SOURCE),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),design_sha256=hashlib.sha256((ROOT/'DESIGN_BEFORE_ANALYSIS.json').read_bytes()).hexdigest(),numerical_function_source=str(AP/'model_a.py'),numerical_function_source_sha256=hashlib.sha256((AP/'model_a.py').read_bytes()).hexdigest(),weights='equal observed basin-year contributions with stable within-basin total SID weight; same bootstrap-selected years for all basins',scope='Sequential regression-standardized 42-year probability trends; not causal mediation. Independent same-M2 natural/fixed joint-distribution contrast supplied.',missing_years='Predicted curves include all42 calendar years; empirical zero-denominator basin-years are flagged, never assigned zero proportion'))
js('CHECKPOINT.json',dict(status='COMPLETE_COMPUTATION',support=MODE,n_sid=N,returns=int(y.sum()),bootstrap_completed=B,bootstrap_failed=0,elapsed_s=time.monotonic()-start))
print(pd.DataFrame(rows).to_string(index=False),flush=True)
