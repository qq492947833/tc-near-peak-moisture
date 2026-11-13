# Public path adapter; the scientific functions below retain their original implementation.
import sys as _sys
from pathlib import Path as _Path
_PACKAGE=next(p for p in _Path(__file__).resolve().parents if (p/'release_paths.py').is_file())
_sys.path.insert(0,str(_PACKAGE))
from release_paths import ADDENDUM, UNIFIED, PACKAGE, bootstrap
from pathlib import Path
import runpy,json,time
R=Path(__file__).resolve().parent
bootstrap(R/'temp')
import numpy as np
import pandas as pd
def read(p):return pd.read_csv(p,keep_default_na=False,na_values=['','nan','NaN'])
def sl(x,y,w):
 keep=np.isfinite(y)&(w>0);x,y,w=x[keep],y[keep],w[keep];x=x-np.average(x,weights=w);return np.sum(w*x*(y-np.average(y,weights=w)))/np.sum(w*x*x)
def logistic(x):return 1/(1+np.exp(-np.clip(x,-700,700)))
def bh(p):
 order=np.argsort(p);q=np.empty(len(p));q[order]=np.minimum(1,np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]);return q
receipts=[]
for O in [R,R/'sensitivity_temporal_overlap']:
 cfg=json.loads((O/'MODEL_CONFIG.json').read_text());allref=read(O/'support_all_sid.csv');mask=allref.B_env_support.to_numpy(bool);d=allref[mask].copy();Nall=len(allref);idx=np.flatnonzero(mask);N=len(d);basins=cfg['basins'];features=cfg['state_variables']+cfg['environment_variables'];nst=len(cfg['state_variables']);years=np.arange(1982,2024);tt=(years-2002.5)/10;t=(d.genesis_year.to_numpy()-2002.5)/10;b=d.basin_genesis.to_numpy()
 z=(d[features].to_numpy()-np.array(cfg['feature_mean']))/np.array(cfg['feature_sd']);base=np.column_stack([np.ones(N)]+[(b==bb).astype(float) for bb in basins[1:]]+[t]+[t*(b==bb) for bb in basins[1:]]);X=np.column_stack([base,z]);E=z[:,-3:]
 bix=[np.flatnonzero(b==bb) for bb in basins];yix=[[np.flatnonzero((b==bb)&(d.genesis_year.to_numpy()==yr)) for yr in years] for bb in basins];gix=[np.flatnonzero(allref.genesis_year.to_numpy()==yr) for yr in years]
 def boot(rng):
  selected=np.concatenate([(rng.integers(0,42)+np.arange(3))%42 for j in range(14)]);ym=np.bincount(selected,minlength=42);w=np.zeros(Nall)
  for yi in selected:
   ii=gix[yi]
   if len(ii):np.add.at(w,rng.choice(ii,len(ii),replace=True),1)
  return w,ym
 def refweights(raw,ym):
  fw=np.zeros(N)
  for a,ii in enumerate(bix):
   for j,jj in enumerate(yix[a]):
    if raw[jj].sum()>0:fw[jj]=raw[jj]/raw[jj].sum()*ym[j]
  return fw
 def calc(co,w,ym):
  raw=w[idx];fw=refweights(raw,ym);p=logistic(X@co);state_eta=X@co-E@co[-3:];environment_eta=E@co[-3:];pfix=np.empty(N);curves=np.full((4,2,42),np.nan);out=np.empty((4,3))
  for a,ii in enumerate(bix):
   pfix[ii]=np.average(logistic(state_eta[ii,None]+environment_eta[None,ii]),axis=1,weights=fw[ii])
   for j,jj in enumerate(yix[a]):
    if raw[jj].sum()>0:
     curves[a,0,j]=np.average(p[jj],weights=raw[jj]);curves[a,1,j]=np.average(pfix[jj],weights=raw[jj])
   out[a,0]=sl(tt,100*curves[a,0],ym);out[a,1]=sl(tt,100*curves[a,1],ym);out[a,2]=out[a,0]-out[a,1]
  return out,curves
 point,pointcurves=calc(np.array(cfg['coefficients'][2]),np.ones(Nall),np.ones(42));saved=np.load(O/'bootstrap_draws.npz');B=cfg['B'];draw=np.empty((B,4,3));rng=np.random.default_rng(cfg['seed']);start=time.monotonic()
 for k in range(B):
  w,ym=boot(rng);draw[k]=calc(saved['full_coefficients'][k],w,ym)[0]
  if (k+1)%200==0:print(cfg['support'],'environment_only',k+1,'elapsed',round(time.monotonic()-start,1),flush=True)
 # This exact equality verifies that seed reconstruction matches the original
 # stored fit/weight pairing, rather than combining unrelated bootstrap draws.
 assert np.allclose(point[:,0],read(O/'same_model_actual_vs_fixed_joint.csv').query("estimand == 'actual_annual_distribution'").set_index('basin').loc[basins].estimate_pp_per_decade,atol=1e-10)
 assert np.allclose(draw[:,:,0],saved['same_model'][:,:,0],atol=1e-10)
 np.save(O/'environment_only_bootstrap_draws.npy',draw)
 rows=[]
 for a,bb in enumerate(basins):
  for k,label in enumerate(['actual_environment','fixed_reference_environment_state_unchanged','actual_minus_fixed_environment']):
   ci=np.quantile(draw[:,a,k],[.025,.975]);rows.append(dict(basin=bb,estimand=label,estimate_pp_per_decade=point[a,k],ci_low=ci[0],ci_high=ci[1],p_bootstrap=min(1.,2*min(np.mean(draw[:,a,k]<=0),np.mean(draw[:,a,k]>=0))),state_treatment='each SID actual year/state unchanged',environment_reference='same basin all-period joint environmental vectors, equal-year reference'))
 table=pd.DataFrame(rows);table['q_environment_only_family']=np.nan;take=table.estimand.eq('actual_minus_fixed_environment');table.loc[take,'q_environment_only_family']=bh(table.loc[take,'p_bootstrap'].to_numpy());table.to_csv(O/'same_model_environment_only_trends.csv',index=False)
 cc=[]
 for a,bb in enumerate(basins):
  for j,yr in enumerate(years):cc.append(dict(basin=bb,year=int(yr),n_sid=len(yix[a][j]),actual_probability_pct=pointcurves[a,0,j]*100,fixed_environment_probability_pct=pointcurves[a,1,j]*100))
 pd.DataFrame(cc).to_csv(O/'same_model_environment_only_annual_predictions.csv',index=False)
 # Joint support audit: preserve each donor's complete environment vector,
 # then diagnose synthetic state/environment combinations using observed covariance.
 support=[];fw=refweights(np.ones(N),np.ones(42))
 for a,bb in enumerate(basins):
  ii=bix[a];zz=z[ii];ww=fw[ii]/fw[ii].sum();mean=np.average(zz,axis=0,weights=ww);cen=zz-mean;cov=(cen*ww[:,None]).T@cen;inv=np.linalg.pinv(cov,rcond=1e-10);observed=np.einsum('ij,jk,ik->i',cen,inv,cen);s=cen[:,:nst];e=cen[:,nst:];dist=np.einsum('ij,jk,ik->i',s,inv[:nst,:nst],s)[:,None]+np.einsum('ij,jk,ik->i',e,inv[nst:,nst:],e)[None,:]+2*s@inv[:nst,nst:]@e.T;pairw=ww[:,None]*ww[None,:];q99=np.quantile(observed,.99)
  support.append(dict(basin=bb,n_sid=len(ii),joint_distance_reference='pseudoinverse covariance of observed joint standardized state/environment within basin',observed_joint_mahalanobis99=q99,reference_pair_mass_beyond_observed99=float(pairw[dist>q99].sum()),reference_pair_mass_beyond_observed_max=float(pairw[dist>observed.max()+1e-9].sum()),marginal_support_unchanged=True,caveat='Marginal bounds and Mahalanobis diagnostics do not prove full positivity; donor replacement can create unobserved conditional combinations'))
 pd.DataFrame(support).to_csv(O/'environment_only_joint_support_audit.csv',index=False)
 receipts.append(dict(support=cfg['support'],n_sid=N,events=int(d.ever_return.sum()),draws=B,exact_same_M2_actual_draws_reproduced=True,maximum_actual_draw_difference=float(abs(draw[:,:,0]-saved['same_model'][:,:,0]).max()),elapsed_s=time.monotonic()-start))
(R/'ENVIRONMENT_ONLY_RECEIPT.json').write_text(json.dumps(dict(status='COMPLETE_SAME_MODEL_ENVIRONMENT_ONLY_STANDARDIZATION',supports=receipts,definition='Each actual SID year and state retained; replace only joint environmental vector by fixed same-basin reference; synchronized paired bootstrap'),indent=2),encoding='utf8');print(json.dumps(receipts,indent=2))
