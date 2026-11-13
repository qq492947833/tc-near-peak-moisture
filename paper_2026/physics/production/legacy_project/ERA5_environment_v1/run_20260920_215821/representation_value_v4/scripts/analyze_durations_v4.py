"""v4 A and duration-only C; fixed representation inputs, no ERA5 or H2 fitting."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t as student_t
from statsmodels.stats.multitest import multipletests
from prepare_durations_v4 import read,dump,V,SEED,F19,active_plan,dependency
B=2000
PERIODS={'P1':(1982,1992),'P2':(1993,2003),'P3':(2004,2014),'P4':(2015,2023)}
D=V/'results/A';D.mkdir(parents=True,exist_ok=True)

def put(x,name):
 p=D/name
 if p.suffix=='.parquet':x.to_parquet(p,index=False)
 else:x.to_csv(p,index=False)
def block_draws():
 dest=V/'inputs/year_block_draws.npz'
 rng=np.random.default_rng(SEED);out={}
 for p,(lo,hi) in PERIODS.items():
  years=np.arange(lo,hi+1);n=len(years);starts=rng.integers(0,n,size=(B,(n+1)//2));out[p]=years[((starts[:,:,None]+[0,1])%n).reshape(B,-1)[:,:n]]
 if dest.exists():
  existing=np.load(dest)
  for p in out:assert np.array_equal(existing[p],out[p])
 else:np.savez_compressed(dest,**out)
 rows=[{'draw':i,'period':p,'position':j,'year':int(y)} for p,a in out.items() for i,row in enumerate(a) for j,y in enumerate(row)]
 pd.DataFrame(rows).to_csv(V/'inputs/year_block_draws.csv.gz',index=False,compression='gzip')
 dump('inputs/year_block_draws_schema.json',dict(seed=SEED,B=B,periods=PERIODS,block_years=2,circular=True,order=list(PERIODS),array_axes=['replicate','draw_position'],rule='Each duplicated real year duplicates all SID records and their multiplicity; never use membership isin for draws'))
 return out

def metric_matrix(d,six=False):
 cols=['T_R_h','T_U_h','T_F_h','T_L_h','T_B_h','T_R_long_h'];m=d[cols].to_numpy(float);names=['R','U','F','L','B','R_long']
 add=[m[:,2]-m[:,0],m[:,1]-m[:,5],m[:,2]-m[:,5]];an=['F_minus_R','U_minus_R_long','F_minus_R_long']
 if six:
  c=d[['T_R5_h','T_U5_h','T_F5_h']].to_numpy(float);m=np.column_stack([m,c]);names+=['R5','U5','F5']
  add += [c[:,1]-c[:,0],c[:,2]-c[:,1],c[:,2]-c[:,0],c[:,0]-m[:,0],c[:,1]-m[:,1],c[:,2]-m[:,2],(c[:,2]-c[:,1])-m[:,4],(c[:,1]-c[:,0])-m[:,3]]
  an+=['U5_minus_R5','F5_minus_U5','F5_minus_R5','R5_minus_R90','U5_minus_U90','F5_minus_F90','threshold_interaction_B','threshold_interaction_L']
 return np.column_stack([m,*add]),names+an

def sid_summary(d,label,six=False,fullquant=False):
 m,names=metric_matrix(d,six);n=len(d);rng=np.random.default_rng(SEED);idx=rng.integers(0,n,size=(B,n),dtype=np.int32)
 statnames=['mean','median','p90','p95'] if fullquant else ['mean'];v=np.empty((B,len(names),len(statnames)))
 cached=D/('sid_draws_'+label+'.npz')
 if cached.exists():
  old=np.load(cached);assert np.array_equal(old['sids'],d.sid.to_numpy(str)) and list(old['metrics'])==names and list(old['statistics'])==statnames
  assert np.array_equal(old['sid_multiplicity'],np.array([np.bincount(ix,minlength=n) for ix in idx],dtype=np.uint16));v=old['values'];print('Reused existing SID draws',label,flush=True)
 else:
  for start in range(0,B,25):
   z=m[idx[start:start+25]];v[start:start+25,:,0]=z.mean(1)
   if fullquant:
    for k,q in enumerate([.5,.9,.95],1):v[start:start+25,:,k]=np.quantile(z,q,axis=1)
 point=np.column_stack([m.mean(0),*[np.quantile(m,q,axis=0) for q in [.5,.9,.95]]]) if fullquant else m.mean(0)[:,None]
 rows=[]
 for j,name in enumerate(names):
  for k,st in enumerate(statnames):
   a=v[:,j,k];lo,hi=np.quantile(a,[.025,.975]);rows.append(dict(scope=label,metric=name,statistic=st,estimate=point[j,k],ci_low=lo,ci_high=hi,n_sid=n,unit='h',B_planned=B,B_valid=B,seed=SEED,estimand='distribution of within-SID difference' if ('minus' in name or 'interaction' in name) else 'distribution of SID duration'))
 # Quantile differences are distinct from quantiles of individual differences.
 if fullquant:
  for a,b,nm in [('F','U','quantile_F_minus_U'),('U','R','quantile_U_minus_R'),('U','R_long','quantile_U_minus_R_long')]:
   ia,ib=names.index(a),names.index(b)
   for k,st in enumerate(statnames):
    dif=v[:,ia,k]-v[:,ib,k];lo,hi=np.quantile(dif,[.025,.975]);rows.append(dict(scope=label,metric=nm,statistic=st,estimate=point[ia,k]-point[ib,k],ci_low=lo,ci_high=hi,n_sid=n,unit='h',B_planned=B,B_valid=B,seed=SEED,estimand='difference of marginal duration distribution statistics; paired SID draw'))
 multiplicity=np.array([np.bincount(ix,minlength=n) for ix in idx],dtype=np.uint16)
 if not cached.exists():np.savez_compressed(cached,sid_multiplicity=multiplicity,sids=d.sid.to_numpy(str),values=v,metrics=np.array(names),statistics=np.array(statnames))
 return rows

def annual(d,label,six=False,formal=False):
 m,names=metric_matrix(d,six);tmp=pd.DataFrame(m,columns=names);tmp['year']=d.genesis_year.to_numpy();ag=tmp.groupby('year')[names].mean();ncounts=tmp.groupby('year').size();x=np.column_stack([np.ones(len(ag)),(ag.index.to_numpy()-2002.5)/10]);rank=np.linalg.matrix_rank(x);df=len(ag)-rank
 coefs=[];fits={};records=[]
 for name in names:
  y=ag[name].to_numpy();fit=sm.OLS(y,x).fit(cov_type='HAC',cov_kwds={'maxlags':3,'use_correction':True},use_t=True)
  if not np.all(np.diff(ag.index.to_numpy())==1):
   score=x*fit.resid[:,None];meat=score.T@score
   for ii in range(len(ag)):
    for jj in range(ii+1,len(ag)):
     gap=int(ag.index[jj]-ag.index[ii])
     if gap<=3:meat+=(1-gap/4)*(np.outer(score[ii],score[jj])+np.outer(score[jj],score[ii]))
   bread=np.linalg.inv(x.T@x);fit._results.cov_params_default=(bread@meat@bread)*len(ag)/df
  fits[name]=fit
  lo,hi=fit.conf_int(alpha=.05)[1];coefs.append(dict(scope=label,metric=name,estimate=fit.params[1],se=fit.bse[1],ci_low=lo,ci_high=hi,p=fit.pvalues[1] if formal and name in ['R','U','F','L','B'] else np.nan,q=np.nan,n_sid=len(d),n_year=len(ag),rank=rank,df=df,HAC_lag=3,finite_sample_correction=True,reference='Student t',unit='h/decade',family='N1' if formal and name in ['R','U','F','L','B'] else 'descriptive sensitivity; no new test family'))
  for year,value in ag[name].items():records.append(dict(scope=label,year=year,metric=name,value=value,n_sid=int(ncounts.loc[year]),unit='h',year_weight=1.))
 if formal:
  p=np.array([r['p'] for r in coefs[:5]]);q=multipletests(p,method='fdr_bh')[1]
  for k in range(5):coefs[k]['q']=q[k]
 # Joint HAC slope covariance for the five constituents; exact contrasts use same X.
 y=ag[['R','U','F','L','B']].to_numpy();coef=np.linalg.lstsq(x,y,rcond=None)[0];res=y-x@coef;bread=np.linalg.inv(x.T@x);score=res*x[:,1,None] if False else None
 influence=(x@bread[:,1])[:,None]*res;cov=influence.T@influence
 for ii in range(len(ag)):
  for jj in range(ii+1,len(ag)):
   gap=int(ag.index[jj]-ag.index[ii])
   if gap<=3:cov+=(1-gap/4)*(np.outer(influence[ii],influence[jj])+np.outer(influence[jj],influence[ii]))
 cov*=len(ag)/df
 assert np.allclose(np.diag(cov),[fits[k].cov_params()[1,1] for k in ['R','U','F','L','B']],rtol=1e-8,atol=1e-10)
 assert np.allclose(fits['F'].params-fits['U'].params,fits['B'].params)
 assert np.allclose(fits['U'].params-fits['R'].params,fits['L'].params)
 assert np.isclose(cov[2,2]+cov[1,1]-2*cov[2,1],cov[4,4],atol=1e-10)
 assert np.isclose(cov[1,1]+cov[0,0]-2*cov[1,0],cov[3,3],atol=1e-10)
 pd.DataFrame(cov,index=['R','U','F','L','B'],columns=['R','U','F','L','B']).to_csv(D/('annual_slope_covariance_'+label+'.csv'))
 return coefs,records

def period(d,label,draws,six=False):
 m,names=metric_matrix(d,six);tmp=pd.DataFrame(m,columns=names);tmp['year']=d.genesis_year.to_numpy();means={};boot={};nums={};pointrows=[]
 for p,(lo,hi) in PERIODS.items():
  z=tmp[tmp.year.between(lo,hi)];means[p]=z[names].mean().to_numpy();nums[p]=len(z)
  for j,name in enumerate(names):pointrows.append(dict(scope=label,period=p,metric=name,estimate=means[p][j],n_sid=len(z),n_year=z.year.nunique(),unit='h',estimand='SID equal period mean'))
  years=np.arange(lo,hi+1);summ=z.groupby('year')[names].sum().reindex(years,fill_value=0).to_numpy();counts=z.groupby('year').size().reindex(years,fill_value=0).to_numpy();sel=draws[p]-lo;den=counts[sel].sum(1);num=summ[sel].sum(1);boot[p]=np.divide(num,den[:,None],out=np.full_like(num,np.nan),where=den[:,None]>0)
 dif=boot['P4']-boot['P1'];point=means['P4']-means['P1'];rows=[]
 for j,name in enumerate(names):
  values=dif[:,j];ok=np.isfinite(values);ci=np.quantile(values[ok],[.025,.975]) if ok.any() else [np.nan,np.nan];rows.append(dict(scope=label,metric=name,estimate=point[j],ci_low=ci[0],ci_high=ci[1],P1_n_sid=nums['P1'],P4_n_sid=nums['P4'],B_planned=B,B_valid=int(ok.sum()),B_failed=int((~ok).sum()),unit='h',contrast='P4 minus P1',estimand='SID equal period mean; repeated sampled years duplicate their entire SID mass',seed=SEED))
 np.savez_compressed(D/('period_draws_'+label+'.npz'),**boot,difference=dif,metrics=np.array(names))
 return rows,pointrows

def ratio_summary(d,label):
 values=np.column_stack([d.T_U_h/d.T_F_h,d.T_L_h/d.T_U_h]);n=len(d);idx=np.random.default_rng(SEED).integers(0,n,size=(B,n),dtype=np.int32);draw=np.array([values[ix].mean(0) for ix in idx]);names=['SID_mean_U_over_F','SID_mean_L_over_U'];rows=[]
 for j,k in enumerate(names):
  ci=np.quantile(draw[:,j],[.025,.975]);rows.append(dict(scope=label,metric=k,estimate=values[:,j].mean(),ci_low=ci[0],ci_high=ci[1],n_sid=n,unit='fraction',estimand='mean of individual ratios'))
 rows+= [dict(scope=label,metric='ratio_of_means_U_over_F',estimate=d.T_U_h.mean()/d.T_F_h.mean(),ci_low=np.nan,ci_high=np.nan,n_sid=n,unit='fraction',estimand='ratio of group means; distinct from mean individual ratio'),dict(scope=label,metric='ratio_of_means_L_over_U',estimate=d.T_L_h.mean()/d.T_U_h.mean(),ci_low=np.nan,ci_high=np.nan,n_sid=n,unit='fraction',estimand='ratio of group means; distinct from mean individual ratio')]
 np.savez_compressed(D/('ratio_draws_'+label+'.npz'),draws=draw,metrics=np.array(names))
 return rows

def influence(d):
 # Exact leave-one-SID means and yearly slope differences; no outlier removal from main.
 cols=['T_L_h','T_B_h'];years=np.arange(1982,2024);x=np.column_stack([np.ones(42),(years-2002.5)/10]);slope_weights=np.linalg.pinv(x)[1];q=d.groupby('genesis_year')[cols].agg(['sum','count']);out=[]
 allmean=d[cols].mean().to_numpy();allannual=d.groupby('genesis_year')[cols].mean().reindex(years).to_numpy();allsl=slope_weights@allannual
 for row in d.itertuples():
  yy=int(row.genesis_year);j=yy-1982;vals=np.array([row.T_L_h,row.T_B_h]);cnt=float(q.loc[yy,(cols[0],'count')]);old=allannual[j];new=(old*cnt-vals)/(cnt-1);slope=allsl+slope_weights[j]*(new-old);mean=(allmean*len(d)-vals)/(len(d)-1)
  for k,name in enumerate(['L','B']):out.append(dict(deleted_sid=row.sid,name=row.name,basin=row.basin_genesis,year=yy,metric=name,n_remaining=len(d)-1,mean=mean[k],mean_change=mean[k]-allmean[k],slope=slope[k],slope_change=slope[k]-allsl[k],mean_unit='h',slope_unit='h/decade'))
 put(pd.DataFrame(out),'leave_one_sid_influence.csv')

def main():
 plan=active_plan()
 raw=read(V/'source_data/durations.csv');d=raw[raw.main_complete].copy();assert len(d)==2366
 draws=block_draws();summ=[];slopes=[];annuals=[];periods=[];periodpoints=[];ratios=[];member=[]
 dump('results/A/draw_schema.json',dict(SID_draw='sid_multiplicity[replicate,SID] with matching sids; each row sums original n. Exact resampled multiset retained; order irrelevant to symmetric means/quantiles. Same multiset all representations.',values_axes=['replicate','metric','statistic'],B=B,seed=SEED,period_draw='arrays retain actual real calendar years and repeated year multiplicity'))
 scopes=[('main_all',d,False,True),('main_multi',d[d.multi],False,True),('main_nonmulti',d[~d.multi],False,True),('six_common_all',d[d.six_complete],True,True),('six_common_multi',d[d.six_complete&d.multi],True,True),('six_common_nonmulti',d[d.six_complete&~d.multi],True,True)]
 # Rule-based and fixed-case sensitivities; all retain their honest cohort denominator.
 scopes.extend([('exclude_single_6h',d[d.later_class.ne('single_6h')],False,False),('strict_bridge_tropical',d[d.strict_tropical_bridge],False,False),('without_Halola',d[d.sid.ne('2015188N08194')],False,False),('without_Freddy2023',d[d.sid.ne('2023036S12117')],False,False)])
 for basin in ['WP','EP','SI','NA','SP','NI']:scopes.append(('without_basin_'+basin,d[d.basin_genesis.ne(basin)],False,False))
 for beta in ['085','095']:
  z=read(V/'source_data'/('durations_beta'+beta+'.csv'));z=z[z.main_complete]
  scopes.append(('beta'+beta+'_available',z,False,False));common=set(d.sid)&set(z.sid)
  scopes.append(('beta'+beta+'_common',z[z.sid.isin(common)],False,False));scopes.append(('main_for_beta'+beta,d[d.sid.isin(common)],False,False))
 ex=read(V/'source_data/durations_expanded.csv');ex=ex[ex.main_complete]
 scopes.extend([('expanded_estimable',ex,False,False),('expanded_added_estimable',ex[ex.is_added],False,False)])
 for label,z,six,full in scopes:
  print('Scope',label,len(z),flush=True);z=z.sort_values('sid').reset_index(drop=True)
  if len(z)<2:continue
  member.extend([dict(scope=label,sid=sid) for sid in z.sid]);summ+=sid_summary(z,label,six,full);sl,an=annual(z,label,six,formal=label=='main_all');slopes+=sl;annuals+=an;pe,pp=period(z,label,draws,six);periods+=pe;periodpoints+=pp
  if full:ratios+=ratio_summary(z,label)
 put(pd.DataFrame(summ),'duration_summaries.csv');put(pd.DataFrame(slopes),'annual_slopes.csv');put(pd.DataFrame(annuals),'annual_means.csv');put(pd.DataFrame(periods),'period_changes.csv');put(pd.DataFrame(periodpoints),'period_means.csv');put(pd.DataFrame(ratios),'duration_ratios.csv');put(pd.DataFrame(member),'membership.csv')
 influence(d)
 # Scope distribution/selection descriptions, including the pre-maturity counting range audit.
 rr=[]
 for label,z,_,_ in scopes:
  rr.append(dict(scope=label,n_sid=len(z),multi_n=int(z.multi.sum()),mean_lmi_kt=z.lmi.mean(),P1_n=int(z.period.eq('P1').sum()),P4_n=int(z.period.eq('P4').sum()),mean_R_h=z.T_R_h.mean(),mean_F_h=z.T_F_h.mean(),later_none_n=int(z.later_class.eq('none').sum()),later_only6_n=int(z.later_class.eq('single_6h').sum()),later_multiple6_n=int(z.later_class.eq('allpieces6h_multiple').sum()),later_unknown_n=int(z.later_class.eq('unknown').sum()),later_ge12_n=int(z.later_class.eq('at_least_12h').sum()),earlier_near_n=int(z.pre_maturity_near_h.gt(0).sum()),earlier_near_h=z.pre_maturity_near_h.sum()))
 put(pd.DataFrame(rr),'scope_selection_summary.csv')
 old=read(F19/'source_data/MATURITY_OCCUPANCY_BY_PERIOD/maturity_occupancy_by_period.csv');pp=pd.DataFrame(periodpoints);checks=[]
 for og,scope in [('all_complete','main_all'),('multiple_near_LMI','main_multi')]:
  for rec in old[old.group.eq(og)].to_dict('records'):
   for field,metric in [('F_mean_h','F'),('U_mean_h','U'),('bridge_mean_h','B'),('later_mean_h','L'),('added_mean_h','F_minus_R')]:
    now=pp[(pp.scope==scope)&(pp.period==rec['period'])&(pp.metric==metric)].iloc[0];checks.append(dict(scope=scope,period=rec['period'],metric=metric,old_field=field,old_value=rec[field],current_value=now['estimate'],difference=now['estimate']-rec[field],same_n=int(now.n_sid)==int(rec['n']),source=str(F19/'source_data/MATURITY_OCCUPANCY_BY_PERIOD/maturity_occupancy_by_period.csv'),disposition='existing point estimate reused/verified; new paired draws, direct contrasts and intervals added'))
 reuse=pd.DataFrame(checks);assert reuse.same_n.all() and np.allclose(reuse.difference,0,atol=1e-10);put(reuse,'existing_S25_reuse_check.csv')
 put(d[['sid','later_class','later_interval_episode_n','later_max_episode_h','pre_maturity_near_h','T_L_h','T_B_h','strict_tropical_bridge','B_both_endpoints_tropical_h','B_other_h','B_unknown_h','B_agency_conflict_h']],'short_state_scope_audit.csv')
 dump('checkpoints/A_complete.json',dict(scopes=len(scopes),main_n=len(d),planned_draws=B,seed=SEED,N1_family_n=5,original_H2_untouched=True,active_plan=plan,dependencies=[dependency(__file__),dependency(V/'source_data/durations.csv'),dependency(V/'inputs/year_block_draws.npz')]))
 print('A and duration C complete',flush=True)
if __name__=='__main__':main()
