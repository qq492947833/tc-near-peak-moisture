# Public path adapter; the scientific functions below retain their original implementation.
import sys as _sys
from pathlib import Path as _Path
_PACKAGE=next(p for p in _Path(__file__).resolve().parents if (p/'release_paths.py').is_file())
_sys.path.insert(0,str(_PACKAGE))
from release_paths import ADDENDUM, UNIFIED, PACKAGE, bootstrap
"""Frozen SID outcomes; annual basin interactions and equal-year composition decomposition."""
from pathlib import Path
import os,sys,json,platform,hashlib,itertools
O=Path(__file__).resolve().parent;R=O.parents[1]
for k in ['TEMP','TMP','TMPDIR','MPLCONFIGDIR','XDG_CACHE_HOME']:
 p=O/'temp'/k;p.mkdir(parents=True,exist_ok=True);os.environ[k]=str(p)
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
# Native platform is detected by the installed scientific Python environment.
import numpy as np,pandas as pd
from scipy import stats
def dump(n,x):(O/n).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
def save(n,d):d.to_csv(O/n,index=False,float_format='%.17g')
def bh(p):
 p=np.array(p);order=np.argsort(p);v=np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1];q=np.empty(len(p));q[order]=np.minimum(1,v);return q
def hac(y,years,maxlags=3):
 x=np.c_[np.ones(len(years)),(years-2002.5)/10];inv=np.linalg.inv(x.T@x);b=inv@x.T@y;u=y-x@b
 score=x[:,:,None]*u[:,None,:]if y.ndim>1 else x*u[:,None]
 score=score.reshape(len(y),-1);S=score.T@score
 for lag in range(1,maxlags+1):
  g=score[lag:].T@score[:-lag];S+=(1-lag/(maxlags+1))*(g+g.T)
 J=np.kron(inv,np.eye(y.shape[1]))if y.ndim>1 else inv
 cov=J@S@J.T*len(y)/(len(y)-2)
 return b,cov
def wilson(k,n):
 if n==0:return np.nan,np.nan
 z=stats.norm.ppf(.975);p=k/n;den=1+z*z/n;c=(p+z*z/(2*n))/den;h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den;return 100*(c-h),100*(c+h)
def hac_unbalanced(Y,years):
 x=np.c_[np.ones(len(years)),(years-2002.5)/10];m=Y.shape[1];b=np.zeros((2,m));J=np.zeros((2*m,2*m));scores=np.zeros((len(years),2*m));ns=[]
 for j in range(m):
  ok=np.isfinite(Y[:,j]);n=ok.sum();ns.append(int(n));xx=x[ok];yy=Y[ok,j];iv=np.linalg.inv(xx.T@xx);b[:,j]=iv@xx.T@yy;u=yy-xx@b[:,j]
  cols=[j,m+j];J[np.ix_(cols,cols)]=iv;scores[np.ix_(ok,cols)]=xx*u[:,None]*np.sqrt(n/(n-2))
 S=scores.T@scores
 for lag in range(1,4):
  g=scores[lag:].T@scores[:-lag];S+=(1-lag/4)*(g+g.T)
 return b,J@S@J.T,ns
P=[('P1',1982,1992),('P2',1993,2003),('P3',2004,2014),('P4',2015,2023)]
basins=['NA','EP','WP','NI','SI','SP'];rich=['NA','EP','WP','SI'];years=np.arange(1982,2024)
d=pd.read_csv(R/'agents/A/first_departure_reference.csv',float_precision='round_trip',keep_default_na=False,na_values=['']);assert len(d)==d.sid.nunique()==2340
assert d.basin_genesis.eq('NA').sum()==343 and d.basin_genesis.eq('WP').sum()==748
assert d.ever_return.sum()==123 and d.n_return.sum()==133
assert set(d.genesis_year)==set(years)
dump('DESIGN_BEFORE_COUNTS.json',dict(cohort=2340,outcome='At least one observed qualified return after first-LMI anchored near-peak segment',genesis_year='Full genesis UTC year, cross-year histories retained',periods=P,annual_trends='42 annual numerator/denominator proportions; equal-year OLS, basin-by-decade interaction; lag3 Bartlett score covariance jointly across basins and years, finite correction/t40',new_test_family='Four adequately supported basin slopes, six pairwise interaction contrasts and new observed-return global slope: 11 tests BH; NI/SP descriptive counts and Wilson intervals only',decomposition='Exact symmetric P1/P4 decomposition of original global equal-year target. Basin weights are mean annual basin shares; basin conditional rates weight each SID by inverse global annual count. Zero denominator basin-years contribute zero weight, not zero rate.',decomposition_CI='2000 within-period circular3year blocks, synchronized all basins; SID resampling within selected year',seed=2026110902,source=str(R/'agents/A/first_departure_reference.csv'),sha256=hashlib.sha256((R/'agents/A/first_departure_reference.csv').read_bytes()).hexdigest()))
annual=[];period=[]
for basin in basins+['SA','GLOBAL']:
 z=d if basin=='GLOBAL'else d[d.basin_genesis==basin]
 for year in years:
  a=z[z.genesis_year==year];n=len(a);k=int(a.ever_return.sum());lo,hi=wilson(k,n)
  annual.append(dict(basin=basin,year=int(year),n_sid=n,n_return_sid=k,n_return_events=int(a.n_return.sum()),rate_pp=100*k/n if n else np.nan,lo_wilson=lo,hi_wilson=hi,status='estimated'if n else'no denominator'))
 for name,start,end in P:
  a=z[z.genesis_year.between(start,end)];n=len(a);k=int(a.ever_return.sum());lo,hi=wilson(k,n)
  period.append(dict(basin=basin,period=name,start=start,end=end,n_years=end-start+1,n_sid=n,n_return_sid=k,n_return_events=int(a.n_return.sum()),rate_pp=100*k/n if n else np.nan,lo_wilson=lo,hi_wilson=hi,status='estimated'if n else'no denominator'))
annual=pd.DataFrame(annual);period=pd.DataFrame(period);save('annual_basin_numerators_denominators.csv',annual);save('period_basin_numerators_denominators.csv',period)
Y=annual[annual.basin.isin(rich)].pivot(index='year',columns='basin',values='rate_pp')[rich].values
b,V,ns=hac_unbalanced(Y,years);s=b[1];C=V[4:,4:];tc=stats.t.ppf(.975,40);trend=[]
for i,basin in enumerate(rich):
 se=np.sqrt(C[i,i]);df=ns[i]-2;crit=stats.t.ppf(.975,df);p=2*stats.t.sf(abs(s[i]/se),df)
 trend.append(dict(estimand=basin,kind='basin slope',estimate=s[i],se=se,lo=s[i]-crit*se,hi=s[i]+crit*se,p_value=p,n_years=ns[i],unit='percentage points per decade'))
for i,j in itertools.combinations(range(4),2):
 se=np.sqrt(C[i,i]+C[j,j]-2*C[i,j]);v=s[i]-s[j];df=min(ns[i],ns[j])-2;crit=stats.t.ppf(.975,df);p=2*stats.t.sf(abs(v/se),df)
 trend.append(dict(estimand=rich[i]+' minus '+rich[j],kind='basin-by-year interaction contrast',estimate=v,se=se,lo=v-crit*se,hi=v+crit*se,p_value=p,n_years=min(ns[i],ns[j]),unit='percentage points per decade'))
globalY=annual[annual.basin=='GLOBAL'].rate_pp.values;gb,gV=hac(globalY,years);gse=np.sqrt(gV[1,1]);trend.append(dict(estimand='GLOBAL anchored-return SID fraction',kind='global slope',estimate=gb[1],se=gse,lo=gb[1]-tc*gse,hi=gb[1]+tc*gse,p_value=2*stats.t.sf(abs(gb[1]/gse),40),n_years=42,unit='percentage points per decade'))
trend=pd.DataFrame(trend);trend['q_value']=bh(trend.p_value);trend['family']='B_return_trends_11';save('annual_trends_and_basin_interactions.csv',trend)
save('annual_joint_slope_covariance.csv',pd.DataFrame(C,index=rich,columns=rich).reset_index(names='basin'))
counts=np.zeros((42,6,2),int)
for i,y in enumerate(years):
 for j,ba in enumerate(basins):
  z=d[(d.genesis_year==y)&(d.basin_genesis==ba)];k=int(z.ever_return.sum());counts[i,j]=[len(z)-k,k]
def target(ct):
 n=ct.sum(axis=2);events=ct[:,:,1];tot=n.sum(axis=1)
 w=(n/tot[:,None]).mean(axis=0);r=(events/tot[:,None]).mean(axis=0)/w
 return w,r,(events.sum(axis=1)/tot).mean()
def decomp(c1,c4):
 w1,r1,g1=target(c1);w4,r4,g4=target(c4);within=(w1+w4)/2*(r4-r1)*100;composition=(r1+r4)/2*(w4-w1)*100
 assert abs(within.sum()+composition.sum()-(g4-g1)*100)<1e-10
 return np.r_[within,composition,within.sum(),composition.sum(),(g4-g1)*100]
v=decomp(counts[:11],counts[-9:]);rng=np.random.default_rng(2026110902);draws=[]
def sample(c):
 n=len(c);ix=np.concatenate([(s+np.arange(3))%n for s in rng.integers(n,size=int(np.ceil(n/3)))])[:n]
 out=[]
 for i in ix:
  ct=c[i];out.append(rng.multinomial(ct.sum(),ct.ravel()/ct.sum()).reshape(6,2))
 return np.array(out)
for k in range(2000):
 a=sample(counts[:11]);z=sample(counts[-9:])
 if(a.sum(axis=(0,2))==0).any()or(z.sum(axis=(0,2))==0).any():continue
 draws.append(decomp(a,z))
draws=np.array(draws);np.save(O/'composition_bootstrap_draws.npy',draws)
lo,hi=np.quantile(draws,[.025,.975],axis=0)
names=['within_'+ba for ba in basins]+['composition_'+ba for ba in basins]+['total_within','total_composition','global_change']
save('global_equal_year_P4_minus_P1_decomposition.csv',pd.DataFrame(dict(component=names,estimate_pp=v,ci_low=lo,ci_high=hi,accepted_draws=len(draws),attempted_draws=2000)))
quality=[]
for name,start,end in P:
 z=d[d.genesis_year.between(start,end)]
 quality.append(dict(period=name,n_sid=len(z),n_return=int(z.ever_return.sum()),mean_followup_h=z.record_followup_h.mean(),median_followup_h=z.record_followup_h.median(),missing_tail_sid=int((z.missing_wind_n>0).sum()),mean_missing_wind_n=z.missing_wind_n.mean(),environment_complete=int(z.environment_complete.sum()),state_complete=int(z.state_complete.sum()),max_non6h=int(z.non6h_intervals.max())))
save('period_observation_coverage_audit.csv',pd.DataFrame(quality))
dump('CHECKPOINT_COUNTS_TRENDS.json',dict(status='COUNTS_TRENDS_AND_COMPOSITION_COMPLETE_PENDING_ENVIRONMENT_STANDARDIZATION',main_sid_n=2340,return_sid_n=123,events=133,global_return_trend_pp_decade=float(gb[1]),decomposition=dict(zip(names,map(float,v))),successful_decomposition_draws=len(draws),remaining=['fixed reference environment within-basin trends','observation coverage sensitivity','independent validation','manuscript integration']))
print(trend[['estimand','estimate','lo','hi','q_value']].to_string(index=False));print('decomposition totals',dict(zip(names[-3:],v[-3:])))
