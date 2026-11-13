# Public path adapter; the scientific functions below retain their original implementation.
import sys as _sys
from pathlib import Path as _Path
_PACKAGE=next(p for p in _Path(__file__).resolve().parents if (p/'release_paths.py').is_file())
_sys.path.insert(0,str(_PACKAGE))
from release_paths import ADDENDUM, UNIFIED, PACKAGE, bootstrap
from pathlib import Path
import runpy,json,hashlib,sys
R=Path(__file__).resolve().parent
bootstrap(R/'temp')
import numpy as np,pandas as pd
from scipy import stats
U=UNIFIED
def read(p):return pd.read_csv(p,keep_default_na=False,na_values=['','nan','NaN'],float_precision='round_trip',low_memory=False)
def save(d,n):d.to_csv(R/n,index=False,float_format='%.17g')
def hac(y,x=None,lag=3):
 y=np.asarray(y,float);n=len(y);x=np.arange(n,dtype=float) if x is None else np.asarray(x,float);x=x-x.mean()
 X=np.column_stack([np.ones(n),x]);bread=np.linalg.inv(X.T@X);b=bread@X.T@y;res=y-X@b;score=X*res[:,None];S=score.T@score
 for k in range(1,lag+1):
  sk=score[k:].T@score[:-k];S+=(1-k/(lag+1))*(sk+sk.T)
 cov=bread@S@bread*n/(n-2);se=np.sqrt(cov[1,1]);tc=stats.t.ppf(.975,n-2)
 return dict(slope=b[1]*10,ci_low=(b[1]-tc*se)*10,ci_high=(b[1]+tc*se)*10,p=2*stats.t.sf(abs(b[1]/se),n-2),n_years=n)
def main():
 paths={'master':U/'source_data/main_sid.csv','duration':U/'source_data/main_durations.csv','entries':U/'statistics/h4_models/stage_entries_revised.csv','rows':U.parent/'source_data/de_existing_sources/analysis_rows.csv','track':U.parent/'source_data/de_existing_sources/track_sp_qc_relabelled.csv','old_primary':U/'statistics/primary/primary_trends.csv'}
 d=read(paths['master']);r=read(paths['rows']);r=r[r.sid.isin(d.sid)].copy();f=read(paths['duration']);e=read(paths['entries']);tr=read(paths['track']);tr=tr[tr.sid.isin(d.sid)].copy()
 assert len(d)==d.sid.nunique()==2340 and set(r.sid)==set(d.sid)==set(f.sid)==set(e.sid)
 for tab,col in [(r,'UTC'),(r,'interval_end'),(tr,'UTC'),(e,'entry_UTC')]:tab[col]=pd.to_datetime(tab[col],utc=True)
 stages={'D':('genesis_time','escalation_time','development_duration_hours'),'E':('escalation_time','maturity_time','escalation_duration_hours'),'F':('maturity_time','resolution_time','maturity_duration_hours'),'Res':('resolution_time','termination_time','resolution_duration_hours')}
 for c in ['genesis_time','escalation_time','maturity_time','resolution_time','termination_time','lmi_time','lmi_last_time']:
  d[c]=pd.to_datetime(d[c],utc=True);r[c]=pd.to_datetime(r[c],utc=True)
  assert (r[c]==r.sid.map(d.set_index('sid')[c])).all(),c
 coverage=[]
 for s,(a,b,col) in stages.items():
  z=e[e.stage.eq(s)].set_index('sid').loc[d.sid]
  assert np.array_equal(z.entry_UTC.to_numpy(),d[a].to_numpy()),s
  assert np.array_equal(z.duration_h.to_numpy(),d[col].to_numpy()),s
  assert np.array_equal((d[b]-d[a]).dt.total_seconds().to_numpy()/3600,d[col].to_numpy()),s
  mask=r.UTC.ge(r[a])&r.UTC.lt(r[b]);valid=mask&r.legal&r.valid_6h
  cv=r.loc[valid].groupby('sid').valid_h.sum().reindex(d.sid,fill_value=0)
  for i,rr in d.iterrows():coverage.append(dict(sid=rr.sid,stage=s,genesis_year=rr.genesis_year,period='P'+str(np.searchsorted([1992,2003,2014],rr.genesis_year,side='left')+1),basin_genesis=rr.basin_genesis,duration_h=rr[col],covered_h=cv.loc[rr.sid],uncovered_h=rr[col]-cv.loc[rr.sid],max_record_gap_h=rr.max_record_gap_hours,max_wind_gap_h=rr.max_wind_gap_hours,n_wind_filled=rr.n_wind_filled))
 cv=pd.DataFrame(coverage);assert cv.uncovered_h.ge(0).all();save(cv,'stage_duration_coverage_sid.csv')
 # Reconstruct each F component from actual left-endpoint masks and legal 6 h support.
 fs=[]
 for comp in ['R','L','B','F']:
  v=r[ r['mask_'+comp]&r.valid_6h&r.legal].groupby('sid').valid_h.sum().reindex(d.sid,fill_value=0)
  old=f.set_index('sid').loc[d.sid,'T_'+comp+'_h']
  assert np.array_equal(v.to_numpy(),old.to_numpy()),comp
  fs.append(v.rename('T_'+comp+'_h'))
 fs=pd.concat(fs,axis=1).reset_index().merge(d[['sid','genesis_year','basin_genesis','maturity_duration_hours']],on='sid',validate='1:1')
 fs['F_uncovered_h']=fs.maturity_duration_hours-fs.T_F_h
 fs['period']=pd.cut(fs.genesis_year,[1981,1992,2003,2014,2023],labels=['P1','P2','P3','P4'])
 assert np.array_equal(fs.T_F_h,(fs.T_R_h+fs.T_L_h+fs.T_B_h))
 assert fs.T_L_h.gt(0).sum()==123 and fs.loc[fs.T_L_h.eq(0),'T_B_h'].eq(0).all()
 assert (r.loc[r.mask_B&r.valid_6h,'wind_kt']<r.loc[r.mask_B&r.valid_6h,'mature_threshold_wind']).all()
 save(fs,'F_components_sid.csv')
 primary=[];annual=[];old=read(paths['old_primary']).set_index('metric')
 for stage,metric in [('D','D_h'),('E','E_h'),('F','F_h'),('Res','Resolution_h')]:
  a=cv[cv.stage.eq(stage)].groupby('genesis_year').agg(n=('sid','size'),mean_h=('duration_h','mean'),covered_h=('covered_h','mean'),uncovered_h=('uncovered_h','mean')).reset_index();a['stage']=stage;annual.append(a)
  z=hac(a.mean_h,a.genesis_year);z.update(stage=stage,metric=metric,old_slope=old.loc[metric,'slope'],old_ci_low=old.loc[metric,'ci_low'],old_ci_high=old.loc[metric,'ci_high'],n_sid=2340)
  for k in ['slope','ci_low','ci_high']:assert abs(z[k]-old.loc[metric,k])<1e-10,(stage,k,z[k],old.loc[metric,k])
  primary.append(z)
 save(pd.DataFrame(primary),'S1c_exact_reproduction.csv');save(pd.concat(annual),'stage_annual_full_cohort.csv')
 fsum=[]
 fa=fs.groupby('genesis_year').agg(n_sid=('sid','size'),**{c:(c,'mean')for c in ['T_R_h','T_L_h','T_B_h','T_F_h','F_uncovered_h']}).reset_index();save(fa,'F_components_annual.csv')
 for c in ['T_R_h','T_L_h','T_B_h','T_F_h','F_uncovered_h']:
  z={'component':c,'mean_h':fs[c].mean(),**hac(fa[c],fa.genesis_year)} if fa[c].std()>0 else dict(component=c,mean_h=fs[c].mean(),slope=0,ci_low=0,ci_high=0,p=np.nan,n_years=42)
  fsum.append(z)
 save(pd.DataFrame(fsum),'F_components_trends_HAC.csv')
 counts=e.groupby(['stage','period']).agg(n_sid=('sid','nunique'),env3_valid=('X3_mid_rh_pct',lambda x:x.notna().sum())).reset_index()
 for c in ['X1_sst_c','X2_shear_ms','X3_mid_rh_pct','entry_wind_kt','entry_motion_ms_v3','entry_past_land_contact_v3','entry_land_current_v3','entry_age_h','entry_track_lat','entry_track_lon']:
  q=e.groupby(['stage','period'])[c].apply(lambda x:x.notna().sum()).reset_index(name=c+'_n');counts=counts.merge(q,on=['stage','period'])
 save(counts,'entry_coverage_period.csv');save(e,'stage_entry_source_frozen.csv')
 per=cv.groupby(['stage','period']).agg(n_sid=('sid','nunique'),mean_duration_h=('duration_h','mean'),mean_covered_h=('covered_h','mean'),mean_uncovered_h=('uncovered_h','mean'),n_with_gap=('uncovered_h',lambda x:(x>0).sum())).reset_index();save(per,'track_coverage_period.csv')
 audit=dict(status='PRIMARY_REPRODUCED_AND_F_COMPONENTS_VALIDATED',n_sid=2340,periods={'P1':[1982,1992,11],'P2':[1993,2003,11],'P3':[2004,2014,11],'P4':[2015,2023,9]},boundary_entry_duration_matches=True,F_component_identity=True,F_zero_contributions=2217,F_recurrent=123,F_gap_sid_n=int(fs.F_uncovered_h.gt(0).sum()),stage_gap_sid_n=cv.groupby('stage').uncovered_h.apply(lambda x:int((x>0).sum())).to_dict(),main_sid_with_record_gaps=int(d.max_record_gap_hours.gt(6).sum()),sources=[dict(role=k,path=str(p),resolved=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())for k,p in paths.items()])
 (R/'AUDIT_DURATION.json').write_text(json.dumps(audit,indent=2),encoding='utf8');(R/'CHECKPOINT.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
 print(json.dumps(audit,indent=2));print(pd.DataFrame(primary).to_string(index=False));print(pd.DataFrame(fsum).to_string(index=False));print(per.to_string(index=False));print(counts.to_string(index=False))
if __name__=='__main__':main()
