from _release_paths import release_path as _release_path
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import argparse,json,hashlib,io,time,sys,ast
import numpy as np,pandas as pd
V=Path(str(_release_path('mechanism_v5','')))
O=Path(__file__).resolve().parent
SEED=6101007
def js(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
def bh(p,m):
 p=np.asarray(p);q=np.full_like(p,np.nan);ix=np.flatnonzero(np.isfinite(p));ix=ix[np.argsort(p[ix])];q[ix]=np.minimum(1,np.minimum.accumulate((p[ix]*m/np.arange(1,len(ix)+1))[::-1])[::-1]);return q
def boot(a,B=2000,seed=SEED):
 a=np.asarray(a,float);a=a[:,None] if a.ndim==1 else a;ok=np.isfinite(a);n=ok.sum(0);mean=np.divide(np.nansum(a,0),n,out=np.full(a.shape[1],np.nan),where=n>0);rng=np.random.default_rng(seed);draw=[];weights=[]
 for j in range(0,B,100):
  ix=rng.integers(len(a),size=(min(100,B-j),len(a)));w=np.zeros((len(ix),len(a)),np.int16);np.add.at(w,(np.arange(len(ix))[:,None],ix),1);den=w@ok.astype(float);dd=np.divide(w@np.nan_to_num(a,nan=0),den,out=np.full(den.shape,np.nan),where=den>0);draw.append(dd);weights.append(w)
 dd=np.concatenate(draw);lo,hi=np.nanquantile(dd,[.025,.975],axis=0);return mean,lo,hi,n,dd,np.concatenate(weights)
def summary_table(rows,group,value='value',sid='sid'):
 x=pd.DataFrame(rows);pivot=x.pivot(index=sid,columns=group,values=value);m,l,h,n,d,w=boot(pivot.to_numpy());out=pd.DataFrame({'estimate':m,'lo':l,'hi':h,'n_sid':n},index=pivot.columns).reset_index();return out,pivot,d,w
def centre():
 x=pd.read_csv(V/'upper_profile_100_50_interpolation_v1/p13_stats_v1/p1_C/process_sources.csv',low_memory=False)
 x=x[x.current117 & x.lag_h.between(-24,24)];rows=[];support=[]
 for (sid,var),g in x.groupby(['sid','variable'],sort=True):
  z=g.pivot(index='lag_h',columns=['event_class','center_kind'],values='value').reindex(range(-24,25,6));cols=pd.MultiIndex.from_product([['internal','final'],['actual_track','fixed_t0']]);z=z.reindex(columns=cols);a=z.to_numpy(copy=True).reshape(9,2,2);common=np.isfinite(a).all((1,2));a[~common]=np.nan
  for policy in ['all_along','all_fixed','historical_mixed']:
   y=a[:,:,0] if policy=='all_along' else a[:,:,1] if policy=='all_fixed' else np.concatenate([a[:4,:,1],a[4:,:,0]])
   values={};values['pre_mean']=np.nanmean(y[:4],0) if common[:4].sum()>=3 else np.full(2,np.nan)
   values['change_0_24']=y[-1]-y[4] if common[[4,8]].all() else np.full(2,np.nan)
   values['integral_0_24']=np.trapezoid(y[4:],dx=6,axis=0) if common[4:].all() else np.full(2,np.nan)
   if common[:4].sum()>=3 and common[4:].all():
    direction=-1 if var in ('midRH','PI_headroom','SST') else 1
    bad=direction*(y[4:]-values['pre_mean'])>0
    values['adverse_hours_0_24']=np.sum(bad[:-1]&bad[1:],axis=0)*6.
   for estimand,v in values.items():
    units=g.units.iloc[0];units=units+' h' if estimand.startswith('integral') else 'h' if estimand.startswith('adverse') else units
    rows.append(dict(sid=sid,variable=var,policy=policy,estimand=estimand,units=units,internal=v[0],final=v[1],difference=v[0]-v[1]))
  support.append(dict(sid=sid,variable=var,n_joint_lags=int(common.sum()),n_pre_joint=int(common[:4].sum()),full_0_24=bool(common[4:].all())))
 r=pd.DataFrame(rows);r.to_csv(O/'centre_SID_sources.csv',index=False);pd.DataFrame(support).to_csv(O/'centre_joint_support.csv',index=False)
 tab,piv,d,w=summary_table(rows,['variable','policy','estimand','units'],value='difference');tab.to_csv(O/'centre_comparison.csv',index=False);np.savez_compressed(O/'centre_bootstrap.npz',sid=piv.index.to_numpy(str),targets=np.array([str(c) for c in piv.columns]),draws=d,weights=w)
 req=pd.read_csv(V/'results/P1_P2_20261009_002220/enriched_requests.csv');r=req[req.current117 & req.lag_h.between(-24,24)].pivot(index=['sid','event_id','lag_h'],columns='center_kind',values=['lat','lon']);pa=np.deg2rad(r[('lat','actual_track')]);pf=np.deg2rad(r[('lat','fixed_t0')]);dl=np.deg2rad(r[('lon','actual_track')]-r[('lon','fixed_t0')]);r['centre_separation_km']=12742.0176*np.arcsin(np.sqrt(np.clip(np.sin((pa-pf)/2)**2+np.cos(pa)*np.cos(pf)*np.sin(dl/2)**2,0,1)));r.to_csv(O/'centre_displacement.csv')
 js(O/'centre_receipt.json',{'status':'computed','n_sid_start':117,'new_raw_field_reads':0,'policy_support':'same SID event lag finite intersection; own equal-radius physical domain per policy','upper_profile':'C accepted derived70 real50; original exclusion retained','rows':len(tab)})
 print('centre complete',len(tab),flush=True)
def spatial():
 b=V/'results/P1_spatial_supervisor_20261009_012142';old=pd.read_csv(b/'map_statistics.csv');old=old[old.primary_map_family.astype(str).str.lower()=='true'];report=[]
 for field,g in old.groupby('field'):
  pp=g.p.dropna();report.append({'field':field,'rows':len(g),'m':5672,'valid_p':len(pp),'min_p':float(pp.min()),'at_floor':int((pp<=2/2001+1e-12).sum()),'old_q05':int((g.q<.05).sum()),'minimum_rank_for_floor':int(np.ceil(5672*(2/2001)/.05))})
 z=np.load(b/'current117_fixed_pre_shear_perSID.npz');a=z['anomaly_common'];mask=z['display_mask'];md=pd.read_csv(b/'current117_SID.csv');B=199999;out=[]
 # independent SID units, paired differences, observed mean centred null, masks retained.
 for fi,field in [(0,'q700_gkg'),(3,'z200_m')]:
  arrays=[];meta=[]
  for hemi in ['NH','SH']:
   sel=md.hemisphere.eq(hemi).to_numpy();d=(a[:,0,:,fi]-a[:,1,:,fi])[:,:,mask];d[~sel]=np.nan;arrays.append(d.reshape(len(d),-1))
   for s in range(4):
    for iy,ix in np.argwhere(mask):meta.append((hemi,str(z['summaries'][s]),float(z['x_m'][iy,ix]/1000),float(z['y_m'][iy,ix]/1000)))
  d=np.concatenate(arrays,axis=1);good=np.isfinite(d);n=good.sum(0);est=np.divide(np.nansum(d,0),n,out=np.full(len(n),np.nan),where=n>0);null=np.where(good,d-est,0);rng=np.random.default_rng(SEED+13);tail=np.zeros(len(n),int);valid=np.zeros(len(n),int)
  for j in range(0,B,100):
   ix=rng.integers(len(d),size=(min(100,B-j),len(d)));w=np.zeros((len(ix),len(d)),np.int16);np.add.at(w,(np.arange(len(ix))[:,None],ix),1);den=w@good.astype(float);draw=np.divide(w@null,den,out=np.full(den.shape,np.nan),where=den>=2);tail+=np.sum(np.abs(draw)>=np.abs(est),0);valid+=np.isfinite(draw).sum(0)
  p=(1+tail)/(valid+1);p[n<10]=np.nan;q=bh(p,5672)
  for j,info in enumerate(meta):out.append(dict(field=field,hemisphere=info[0],summary=info[1],x_km=info[2],y_km=info[3],estimate=est[j],n_sid=n[j],B_attempted=B,B_valid=valid[j],tail_count=tail[j],p=p[j],q=q[j],m=5672))
  print('spatial',field,'q05',int(np.sum(q<.05)),flush=True)
 pd.DataFrame(out).to_csv(O/'spatial_centered_null_full_family_B199999.csv',index=False);js(O/'spatial_resolution_audit_B199999.json',{'old':report,'new_B':B,'new_p_min':1/(B+1),'null':'mean-centred SID bootstrap; paired population difference, not event-label randomization','field_correlation':'one SID multiplicity carries entire field','scope':'post hoc supplementary sensitivity; old full family preserved','seed':SEED+13})
def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['centre','spatial']);a=p.parse_args();globals()[a.mode]()
if __name__=='__main__':main()


